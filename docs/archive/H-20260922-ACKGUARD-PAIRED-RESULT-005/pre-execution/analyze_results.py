"""Pure, outcome-blind JSONL analysis for the H-005 guard matrix.

The analyzer consumes already persisted branch and step records.  It does not
import an environment, a model, a checkpoint, or a budget implementation and
does not open SQLite.  A malformed, incomplete, or identity-mismatched input
is retained in the output as ``evaluation incomplete``; no primary estimate or
confidence interval is produced for such an input.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


GAMMA = 0.99
PREFERENCE = (0.8, 0.2)
REWARD_SCALES = {"task": 0.5, "energy": 1.0}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260922
TIE_TOLERANCE = 1e-6
NUMERIC_TOLERANCE = 1e-6
FEEDBACK_NON_REJECTED = ("accepted", "awaiting_ack", "noop", "reuse_existing")
_MISSING = object()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects without silently skipping malformed records."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _as_int(value: Any) -> int | None:
    if not _finite(value):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        return None
    return int(numeric)


def _same_number(left: Any, right: Any, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    return _finite(left) and _finite(right) and abs(float(left) - float(right)) <= tolerance


def control_key(row: Mapping[str, Any]) -> str:
    """Return the stable pair key used by the manifest.

    Historical R rows did not persist ``control_branch_key``.  Their key is
    reconstructed from the persisted prefix/repeat/mode identity, never from a
    vector-return or a positional row number.
    """

    value = row.get("control_branch_key")
    if isinstance(value, str) and value:
        return value
    prefix = row.get("prefix_id")
    repeat = _as_int(row.get("repeat"))
    if not isinstance(prefix, str) or repeat is None:
        raise ValueError("row has no control_branch_key or complete prefix/repeat identity")
    mode = row.get("mode", "R")
    if mode is None:
        mode = "R"
    return f"{prefix}|repeat-{repeat}|mode-{mode}"


def _vector(row: Mapping[str, Any]) -> tuple[float, float]:
    values = row.get("vector_reward")
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"missing vector_reward for {control_key(row)} step={row.get('step')}")
    if not all(_finite(value) for value in values):
        raise ValueError(f"non-finite vector_reward for {control_key(row)} step={row.get('step')}")
    return float(values[0]), float(values[1])


def discounted_vectors(rows: Iterable[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """Compute strict discounted vector returns, rejecting duplicate steps."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = control_key(row)
        grouped[key].append(row)
    output: dict[str, tuple[float, float]] = {}
    for key, branch_rows in grouped.items():
        by_index: dict[int, Mapping[str, Any]] = {}
        for row in branch_rows:
            index = _as_int(row.get("gamma_index"))
            step = _as_int(row.get("step"))
            if index is None or step is None or step != index + 1:
                raise ValueError(f"invalid step identity for {key}")
            if index in by_index:
                raise ValueError(f"duplicate step identity for {key}: gamma_index={index}")
            by_index[index] = row
        indices = sorted(by_index)
        if indices != list(range(len(indices))):
            raise ValueError(f"partial step sequence for {key}: {indices}")
        task_total = 0.0
        energy_total = 0.0
        for index in indices:
            task, energy = _vector(by_index[index])
            task_total += GAMMA**index * task
            energy_total += GAMMA**index * energy
        output[key] = (task_total, energy_total)
    return output


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def utility_from_vectors(task_return: float, energy_return: float) -> float:
    """Apply the frozen H-005 utility preference to discounted returns."""

    return PREFERENCE[0] * REWARD_SCALES["task"] * float(task_return) + PREFERENCE[1] * REWARD_SCALES["energy"] * float(energy_return)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON manifest {path}: {exc}") from exc


def _manifest_expectations(payload: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, int], str], list[str]]:
    issues: list[str] = []
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        return [], {}, {}, ["manifest:rows_missing"]

    expected: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    by_prefix_repeat: dict[tuple[str, int], str] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            issues.append(f"manifest:row_{index}_not_object")
            continue
        pair_key = raw.get("control_branch_key")
        branch_id = raw.get("branch_id")
        prefix_id = raw.get("prefix_id")
        parent_id = raw.get("parent_id")
        condition = raw.get("condition")
        model_seed = _as_int(raw.get("model_seed"))
        repeat = _as_int(raw.get("repeat"))
        exogenous_key = raw.get("historical_exogenous_key", raw.get("exogenous_key"))
        if not isinstance(pair_key, str) or not pair_key:
            issues.append(f"manifest:row_{index}:control_branch_key")
            continue
        if pair_key in by_key:
            issues.append(f"manifest:duplicate_pair:{pair_key}")
        if not isinstance(branch_id, str) or not branch_id:
            issues.append(f"manifest:row_{index}:branch_id")
        if not isinstance(prefix_id, str) or not prefix_id:
            issues.append(f"manifest:row_{index}:prefix_id")
        if not isinstance(parent_id, str) or not parent_id:
            issues.append(f"manifest:row_{index}:parent_id")
        if not isinstance(condition, str) or model_seed is None or repeat is None:
            issues.append(f"manifest:row_{index}:matrix_identity")
        if not isinstance(exogenous_key, str) or not exogenous_key:
            issues.append(f"manifest:row_{index}:exogenous_key")
        if not isinstance(pair_key, str) or not isinstance(prefix_id, str) or repeat is None:
            continue
        expected_row = {
            "pair_key": pair_key,
            "branch_id": branch_id,
            "prefix_id": prefix_id,
            "parent_id": parent_id,
            "condition": condition,
            "model_seed": model_seed,
            "repeat": repeat,
            "exogenous_key": exogenous_key,
            "split": raw.get("split", "train"),
            "prefix_index": _as_int(raw.get("prefix_index")),
        }
        expected.append(expected_row)
        by_key.setdefault(pair_key, expected_row)
        prefix_key = (prefix_id, repeat)
        if prefix_key in by_prefix_repeat and by_prefix_repeat[prefix_key] != pair_key:
            issues.append(f"manifest:duplicate_prefix_repeat:{prefix_id}|repeat-{repeat}")
        by_prefix_repeat[prefix_key] = pair_key

    declared_counts = payload.get("counts") if isinstance(payload, Mapping) else None
    if len(rows) != 24:
        issues.append(f"manifest:expected_24_rows:{len(rows)}")
    if len(expected) != 24:
        issues.append(f"manifest:usable_expected_rows:{len(expected)}")
    if isinstance(declared_counts, Mapping):
        for name, wanted in (("branches", 24), ("parents", 8), ("repeats", 3)):
            actual = _as_int(declared_counts.get(name))
            if actual != wanted:
                issues.append(f"manifest:counts:{name}={actual}")
    parents = {item.get("parent_id") for item in expected}
    repeats = {item.get("repeat") for item in expected}
    if len(parents) != 8:
        issues.append(f"manifest:expected_8_parents:{len(parents)}")
    if repeats != {0, 1, 2}:
        issues.append(f"manifest:expected_repeats:{sorted(repeats)}")
    per_parent: dict[str, list[int]] = defaultdict(list)
    for item in expected:
        per_parent[str(item.get("parent_id"))].append(int(item["repeat"]))
        if item.get("condition") != "W1":
            issues.append(f"manifest:condition:{item.get('pair_key')}")
        if item.get("model_seed") != 1101:
            issues.append(f"manifest:model_seed:{item.get('pair_key')}")
        if item.get("prefix_index") not in (None, 0):
            issues.append(f"manifest:prefix_index:{item.get('pair_key')}")
        if not str(item.get("pair_key", "")).endswith("|mode-R"):
            issues.append(f"manifest:pair_mode:{item.get('pair_key')}")
    if any(sorted(values) != [0, 1, 2] for values in per_parent.values()) or len(per_parent) != 8:
        issues.append("manifest:each_parent_must_have_three_repeats")
    return expected, by_key, by_prefix_repeat, sorted(set(issues))


def _candidate_pair(row: Mapping[str, Any], by_key: Mapping[str, Mapping[str, Any]], by_prefix_repeat: Mapping[tuple[str, int], str]) -> str | None:
    explicit = row.get("control_branch_key")
    if isinstance(explicit, str) and explicit in by_key:
        return explicit
    branch_id = row.get("branch_id")
    for item in by_key.values():
        if isinstance(branch_id, str) and branch_id == item.get("branch_id"):
            return str(item["pair_key"])
    prefix = row.get("prefix_id")
    repeat = _as_int(row.get("repeat"))
    if isinstance(prefix, str) and repeat is not None:
        return by_prefix_repeat.get((prefix, repeat))
    return None


def _select_rows(rows: Sequence[Mapping[str, Any]], by_key: Mapping[str, Mapping[str, Any]], by_prefix_repeat: Mapping[tuple[str, int], str], treatment: str) -> tuple[dict[str, list[Mapping[str, Any]]], int]:
    selected: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ignored = 0
    for row in rows:
        if treatment == "R" and row.get("mode") != "R":
            ignored += 1
            continue
        if treatment == "guard" and row.get("mode") == "R":
            ignored += 1
            continue
        key = _candidate_pair(row, by_key, by_prefix_repeat)
        if key is None:
            ignored += 1
            continue
        selected[key].append(row)
    return dict(selected), ignored


def _identity_issues(row: Mapping[str, Any], expected: Mapping[str, Any], treatment: str, level: str) -> list[str]:
    issues: list[str] = []
    prefix = expected.get("pair_key")
    fields = ("prefix_id", "parent_id", "condition", "model_seed", "repeat", "exogenous_key")
    for field in fields:
        actual = row.get(field)
        if field == "model_seed":
            actual = _as_int(actual)
        elif field == "repeat":
            actual = _as_int(actual)
        if actual is None:
            issues.append(f"{level}:missing_{field}:{prefix}")
        elif actual != expected.get(field):
            issues.append(f"{level}:identity_{field}:{prefix}")
    split = row.get("split")
    if split is not None and split != expected.get("split", "train"):
        issues.append(f"{level}:identity_split:{prefix}")
    if treatment == "R" and split is None:
        issues.append(f"{level}:missing_split:{prefix}")
    control = row.get("control_branch_key")
    if treatment == "guard":
        if control != expected.get("pair_key"):
            issues.append(f"{level}:identity_control_branch_key:{prefix}")
        if row.get("branch_id") != expected.get("branch_id"):
            issues.append(f"{level}:identity_branch_id:{prefix}")
        if row.get("mode") not in (None, "guard", "guard-on"):
            issues.append(f"{level}:identity_mode:{prefix}")
    else:
        if row.get("mode") != "R":
            issues.append(f"{level}:identity_mode:{prefix}")
        if control is not None and control != expected.get("pair_key"):
            issues.append(f"{level}:identity_control_branch_key:{prefix}")
    return issues


def _step_identity_issues(row: Mapping[str, Any], expected: Mapping[str, Any], treatment: str) -> list[str]:
    issues: list[str] = []
    prefix = expected.get("pair_key")
    for field in ("prefix_id", "parent_id", "condition"):
        if row.get(field) != expected.get(field):
            issues.append(f"step:identity_{field}:{prefix}")
    if _as_int(row.get("model_seed")) != expected.get("model_seed"):
        issues.append(f"step:identity_model_seed:{prefix}")
    if _as_int(row.get("repeat")) != expected.get("repeat"):
        issues.append(f"step:identity_repeat:{prefix}")
    actual_exogenous = row.get("exogenous_key")
    if actual_exogenous is not None and actual_exogenous != expected.get("exogenous_key"):
        issues.append(f"step:identity_exogenous_key:{prefix}")
    if treatment == "guard":
        if row.get("control_branch_key") != expected.get("pair_key"):
            issues.append(f"step:identity_control_branch_key:{prefix}")
        if row.get("branch_id") != expected.get("branch_id"):
            issues.append(f"step:identity_branch_id:{prefix}")
    else:
        if row.get("mode") != "R":
            issues.append(f"step:identity_mode:{prefix}")
        if row.get("control_branch_key") is not None and row.get("control_branch_key") != expected.get("pair_key"):
            issues.append(f"step:identity_control_branch_key:{prefix}")
    return issues


def _row_error_issues(row: Mapping[str, Any], level: str, key: str) -> list[str]:
    issues: list[str] = []
    for field in ("error", "failure", "exception"):
        value = row.get(field)
        if value not in (None, "", False, 0, [], {}):
            issues.append(f"{level}:{field}:{key}")
    for field in ("partial", "incomplete"):
        if row.get(field) is True:
            issues.append(f"{level}:{field}:{key}")
    status = row.get("status")
    if status in {"error", "failed", "failure", "partial", "incomplete", "technical_stop"}:
        issues.append(f"{level}:status_{status}:{key}")
    if row.get("end_reason") in {"error", "technical_stop", "exception", "failure"}:
        issues.append(f"{level}:end_reason_{row.get('end_reason')}:{key}")
    if row.get("budget_status") in {"unknown", "pending", "reserved_pending_finalization", "pending_before_reserve"}:
        issues.append(f"{level}:budget_{row.get('budget_status')}:{key}")
    return issues


def _record_map(value: Any) -> dict[str, Mapping[str, Any]] | None:
    if not isinstance(value, Mapping):
        return None
    if not value:
        return {}
    if not all(isinstance(task_id, str) and isinstance(record, Mapping) for task_id, record in value.items()):
        return None
    return {str(task_id): record for task_id, record in value.items()}


def _step_feedback(row: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any] | None, list[str]]:
    issues: list[str] = []
    actual_feedback = row.get("actual_feedback", _MISSING)
    info: Mapping[str, Any] | None = None
    if actual_feedback is not _MISSING:
        if not isinstance(actual_feedback, Mapping):
            issues.append("step:actual_feedback_not_object")
        else:
            info = actual_feedback
            if "info" in actual_feedback and "feedback" not in actual_feedback:
                issues.append("step:nested_actual_feedback_info")
    feedback = info.get("feedback", _MISSING) if info is not None else row.get("feedback", _MISSING)
    if feedback is _MISSING and "feedback" in row:
        feedback = row.get("feedback")
    if feedback is _MISSING:
        issues.append("step:feedback_missing")
    elif not isinstance(feedback, str):
        issues.append("step:feedback_not_string")
    return feedback, info, issues


def _step_completion_records(row: Mapping[str, Any], info: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]] | None:
    if info is not None and "completion_records" in info:
        return _record_map(info.get("completion_records"))
    if "completion_records" in row:
        return _record_map(row.get("completion_records"))
    return None


def _validate_probability_ledger(row: Mapping[str, Any], legal_mask: Sequence[bool], original: int, action: int) -> list[str]:
    issues: list[str] = []
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, list):
        return ["step:guard_probabilities_missing"]
    if len(probabilities) != len(legal_mask):
        return ["step:guard_probability_length"]
    if all(_finite(value) for value in probabilities):
        return issues
    seen: set[int] = set()
    selected: list[int] = []
    originals: list[int] = []
    for entry in probabilities:
        if not isinstance(entry, Mapping):
            issues.append("step:guard_probability_entry")
            continue
        index = _as_int(entry.get("action"))
        probability = entry.get("probability")
        legal = entry.get("legal")
        if index is None or index < 0 or index >= len(legal_mask) or not _finite(probability):
            issues.append("step:guard_probability_value")
            continue
        if index in seen:
            issues.append("step:guard_probability_duplicate_action")
        seen.add(index)
        if legal is not None and bool(legal) != bool(legal_mask[index]):
            issues.append("step:guard_probability_legal_mismatch")
        if entry.get("selected") is True:
            selected.append(index)
        if entry.get("original_argmax") is True:
            originals.append(index)
    if seen != set(range(len(legal_mask))):
        issues.append("step:guard_probability_actions_partial")
    if selected and selected != [action]:
        issues.append("step:guard_probability_selected_mismatch")
    if originals and originals != [original]:
        issues.append("step:guard_probability_original_mismatch")
    return issues


def _required_count(mapping: Any, name: str) -> int | None:
    if not isinstance(mapping, Mapping):
        return None
    return _as_int(mapping.get(name))


def _cohort_records(row: Mapping[str, Any], treatment: str, step_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]] | None, list[str]]:
    issues: list[str] = []
    if treatment == "guard":
        last_info = row.get("last_info")
        if last_info is not None:
            if not isinstance(last_info, Mapping):
                issues.append("branch:last_info_not_object")
            elif "info" in last_info and "completion_records" not in last_info:
                issues.append("branch:nested_last_info")
            elif "completion_records" in last_info:
                records = _record_map(last_info.get("completion_records"))
                if records is not None:
                    return records, issues
                issues.append("branch:completion_records_not_object")
        for step in reversed(step_records):
            actual = step.get("actual_feedback")
            info = actual if isinstance(actual, Mapping) else None
            records = _step_completion_records(step, info)
            if records:
                return records, issues
        records = _record_map(row.get("all_task_outcome"))
        if records is not None:
            return records, issues
    else:
        records = _record_map(row.get("all_task_outcome"))
        if records is not None:
            return records, issues
        records = _record_map(row.get("task_outcome"))
        if records is not None:
            return records, issues
        for step in reversed(step_records):
            actual = step.get("actual_feedback")
            info = actual if isinstance(actual, Mapping) else None
            records = _step_completion_records(step, info)
            if records:
                return records, issues
    issues.append("branch:completion_records_missing")
    return None, issues


def _task_ids(row: Mapping[str, Any], treatment: str) -> tuple[list[str] | None, list[str]]:
    if treatment == "guard":
        return None, []
    values = row.get("task_ids")
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        return None, ["branch:R_task_ids_missing"]
    if len(set(values)) != len(values):
        return None, ["branch:R_task_ids_duplicate"]
    return list(values), []


def _branch_summary(row: Mapping[str, Any], expected: Mapping[str, Any], treatment: str) -> tuple[dict[str, Any], list[str]]:
    key = str(expected["pair_key"])
    issues = _identity_issues(row, expected, treatment, "branch")
    issues.extend(_row_error_issues(row, "branch", key))
    env_steps = _as_int(row.get("env_steps"))
    if env_steps is None or env_steps < 1:
        issues.append(f"branch:env_steps:{key}")
    terminated = row.get("terminated")
    truncated = row.get("truncated")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        issues.append(f"branch:terminal_flags_missing:{key}")
    elif not terminated or truncated:
        issues.append(f"branch:nonterminal_or_truncated:{key}")
    if row.get("end_reason") != "terminated":
        issues.append(f"branch:end_reason:{key}")
    for field in ("rows", "successful_env_steps", "verified_steps"):
        if row.get(field) is not None and _as_int(row.get(field)) != env_steps:
            issues.append(f"branch:{field}_mismatch:{key}")
    initial_counts = row.get("initial_counts")
    final_counts = row.get("final_counts")
    initial_completed = _required_count(initial_counts, "completed")
    final_completed = _required_count(final_counts, "completed")
    if initial_completed is None or final_completed is None:
        issues.append(f"branch:completed_counts_missing:{key}")
    completed_delta = (final_completed - initial_completed) if initial_completed is not None and final_completed is not None else None
    energy_used = float(row["energy_used"]) if _finite(row.get("energy_used")) else None
    if energy_used is None:
        issues.append(f"branch:energy_used_missing:{key}")
    initial_energy = float(row["initial_energy"]) if _finite(row.get("initial_energy")) else None
    final_energy = float(row["final_energy"]) if _finite(row.get("final_energy")) else None
    if initial_energy is not None and final_energy is not None and energy_used is not None and not _same_number(initial_energy - final_energy, energy_used, NUMERIC_TOLERANCE * 2):
        issues.append(f"branch:energy_total_mismatch:{key}")
    task_ids, task_id_issues = _task_ids(row, treatment)
    issues.extend(f"{item}:{key}" for item in task_id_issues)

    actor_field = "actor_forward_calls" if treatment == "guard" else "actor_calls"
    world_field = "world_forward_calls" if treatment == "guard" else "world_calls"
    actor_count = _as_int(row.get(actor_field)) if row.get(actor_field) is not None else None
    world_count = _as_int(row.get(world_field)) if row.get(world_field) is not None else None
    if actor_count is None:
        issues.append(f"branch:{actor_field}_missing:{key}")
    if world_count is None:
        issues.append(f"branch:{world_field}_missing:{key}")
    reported: dict[str, int | None] = {}
    if treatment == "guard":
        for field in ("command_count", "task_unavailable_count", "guard_trigger_count", "action_change_count", "noop_count"):
            value = _as_int(row.get(field))
            reported[field] = value
            if value is None:
                issues.append(f"branch:{field}_missing:{key}")

    summary = {
        "treatment": treatment,
        "control_branch_key": key,
        "branch_id": row.get("branch_id"),
        "parent_id": str(expected["parent_id"]),
        "repeat": int(expected["repeat"]),
        "prefix_id": str(expected["prefix_id"]),
        "condition": expected["condition"],
        "model_seed": int(expected["model_seed"]),
        "exogenous_key": expected["exogenous_key"],
        "env_steps": env_steps,
        "task_ids": task_ids,
        "task_vector_return": None,
        "energy_vector_return": None,
        "utility": None,
        "energy_used": energy_used,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "initial_completed": initial_completed,
        "final_completed": final_completed,
        "completed_delta": completed_delta,
        "all_task_completed": _as_int(row.get("all_completed")) if row.get("all_completed") is not None else final_completed,
        "all_host_confirmed": _as_int(row.get("all_host_confirmed")) if row.get("all_host_confirmed") is not None else None,
        "on_time_physical": None,
        "on_time_host": None,
        "actor_forward_calls": actor_count,
        "world_forward_calls": world_count,
        "model_forward_calls": actor_count,
        "accepted_count": None,
        "rejected_count": None,
        "task_unavailable_count": None,
        "submit_count": None,
        "feedback_histogram": {},
        "noop_count": reported.get("noop_count"),
        "noop_count_from_actions": None,
        "guard_trigger_count": reported.get("guard_trigger_count"),
        "action_change_count": reported.get("action_change_count"),
        "reported_command_count": reported.get("command_count"),
        "reported_task_unavailable_count": reported.get("task_unavailable_count"),
        "reported_noop_count": reported.get("noop_count"),
        "_decision_rows": row.get("decisions", row.get("decision_rows")),
        "technical_status": "complete" if not issues else "incomplete",
    }
    return summary, sorted(set(issues))


def _validate_steps(rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any], treatment: str, branch: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    key = str(expected["pair_key"])
    issues: list[str] = []
    expected_count = branch.get("env_steps")
    if expected_count is None:
        expected_count = len(rows)
    if len(rows) != int(expected_count):
        issues.append(f"steps:count:{key}:{len(rows)}!={expected_count}")
    by_step: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        step = _as_int(row.get("step"))
        if step is None:
            issues.append(f"steps:missing_step:{key}")
            continue
        if step in by_step:
            issues.append(f"steps:duplicate_step:{key}:{step}")
        by_step[step] = row
    ordered_steps = [by_step[index] for index in sorted(by_step)]
    if sorted(by_step) != list(range(1, int(expected_count) + 1)):
        issues.append(f"steps:partial_sequence:{key}")

    task_total = 0.0
    vector_energy_total = 0.0
    step_energy_used_total = 0.0
    feedback_histogram: dict[str, int] = defaultdict(int)
    accepted = rejected = task_unavailable = submits = 0
    action_changes = noop_actions = guard_triggers = 0
    completion_records: list[Mapping[str, Any]] = []
    for row in ordered_steps:
        issues.extend(_row_error_issues(row, "step", key))
        issues.extend(_step_identity_issues(row, expected, treatment))
        step = _as_int(row.get("step"))
        gamma_index = _as_int(row.get("gamma_index"))
        if step is None or gamma_index is None or gamma_index != step - 1:
            issues.append(f"step:gamma_index:{key}:{step}")
        vector = row.get("vector_reward")
        if not isinstance(vector, (list, tuple)) or len(vector) != 2 or not all(_finite(value) for value in vector):
            issues.append(f"step:vector_reward:{key}:{step}")
        else:
            task_total += GAMMA ** int(gamma_index or 0) * float(vector[0])
            vector_energy_total += GAMMA ** int(gamma_index or 0) * float(vector[1])
        gamma = row.get("gamma")
        if not _same_number(gamma, GAMMA, 1e-12):
            issues.append(f"step:gamma:{key}:{step}")
        scales = row.get("reward_scales")
        if not isinstance(scales, Mapping) or not _same_number(scales.get("task"), REWARD_SCALES["task"], 1e-12) or not _same_number(scales.get("energy"), REWARD_SCALES["energy"], 1e-12):
            issues.append(f"step:reward_scales:{key}:{step}")
        preference = row.get("behavior_preference")
        if not isinstance(preference, (list, tuple)) or len(preference) != 2 or not _same_number(preference[0], PREFERENCE[0], 1e-12) or not _same_number(preference[1], PREFERENCE[1], 1e-12):
            issues.append(f"step:behavior_preference:{key}:{step}")
        feedback, info, feedback_issues = _step_feedback(row)
        issues.extend(f"{item}:{key}:{step}" for item in feedback_issues)
        if isinstance(feedback, str):
            feedback_histogram[feedback] += 1
            if feedback == "accepted":
                accepted += 1
            if feedback not in FEEDBACK_NON_REJECTED:
                rejected += 1
            if feedback == "task_unavailable":
                task_unavailable += 1
        submit = row.get("submit_command")
        if not isinstance(submit, bool):
            issues.append(f"step:submit_command:{key}:{step}")
        elif submit:
            submits += 1
        energy_delta = row.get("energy_used_delta")
        if not _finite(energy_delta):
            issues.append(f"step:energy_used_delta:{key}:{step}")
        else:
            step_energy_used_total += float(energy_delta)
        before = row.get("energy_before")
        after = row.get("energy_after")
        if not _finite(before) or not _finite(after):
            issues.append(f"step:energy_before_after:{key}:{step}")
        elif _finite(energy_delta) and not _same_number(float(energy_delta), max(0.0, float(before) - float(after)), NUMERIC_TOLERANCE):
            issues.append(f"step:energy_delta_mismatch:{key}:{step}")
        terminated = row.get("terminated")
        truncated = row.get("truncated")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            issues.append(f"step:terminal_flags:{key}:{step}")
        elif (step == expected_count and (terminated is not True or truncated is not False)) or (step != expected_count and (terminated or truncated)):
            issues.append(f"step:terminal_sequence:{key}:{step}")
        records = _step_completion_records(row, info)
        if records:
            completion_records.append(records)

        action = _as_int(row.get("action"))
        original = _as_int(row.get("original_action"))
        legal_mask = row.get("legal_mask")
        if not isinstance(legal_mask, list) or not legal_mask or not all(isinstance(value, (bool, int)) for value in legal_mask):
            issues.append(f"step:legal_mask:{key}:{step}")
        else:
            mask = [bool(value) for value in legal_mask]
            if action is None or action < 0 or action >= len(mask) or not mask[action]:
                issues.append(f"step:action_not_legal:{key}:{step}")
            if treatment == "guard":
                if original is None or original < 0 or original >= len(mask) or not mask[original]:
                    issues.append(f"step:original_action_not_legal:{key}:{step}")
                if action is not None and original is not None:
                    action_changes += int(action != original)
                if action == 24:
                    noop_actions += 1
                issues.extend(f"{item}:{key}:{step}" for item in _validate_probability_ledger(row, mask, original if original is not None else -1, action if action is not None else -1))
            elif action == 24:
                noop_actions += 1
        triggered = row.get("triggered")
        if isinstance(triggered, bool):
            guard_triggers += int(triggered)

    if ordered_steps and ordered_steps[-1].get("terminated") is not True:
        issues.append(f"steps:last_not_terminated:{key}")
    branch_energy = branch.get("energy_used")
    if branch_energy is None or not _same_number(step_energy_used_total, float(branch_energy), NUMERIC_TOLERANCE * 2):
        issues.append(f"steps:energy_total_mismatch:{key}")
    if treatment == "guard":
        if branch.get("reported_command_count") is not None and branch["reported_command_count"] != submits:
            issues.append(f"steps:command_count_mismatch:{key}")
        if branch.get("reported_task_unavailable_count") is not None and branch["reported_task_unavailable_count"] != task_unavailable:
            issues.append(f"steps:task_unavailable_count_mismatch:{key}")
        if branch.get("action_change_count") is not None and branch["action_change_count"] != action_changes:
            issues.append(f"steps:action_change_count_mismatch:{key}")
        if branch.get("reported_noop_count") is not None and branch["reported_noop_count"] != noop_actions:
            issues.append(f"steps:noop_count_mismatch:{key}")
        if guard_triggers and branch.get("guard_trigger_count") is not None and branch["guard_trigger_count"] != guard_triggers:
            issues.append(f"steps:guard_trigger_count_mismatch:{key}")
        decision_rows = branch.get("_decision_rows")
        if decision_rows is not None:
            if not isinstance(decision_rows, list) or len(decision_rows) != int(expected_count):
                issues.append(f"decisions:count:{key}")
            else:
                decision_by_step: dict[int, Mapping[str, Any]] = {}
                for decision in decision_rows:
                    if not isinstance(decision, Mapping):
                        issues.append(f"decisions:not_object:{key}")
                        continue
                    decision_step = _as_int(decision.get("step"))
                    if decision_step is None or decision_step in decision_by_step:
                        issues.append(f"decisions:duplicate_or_missing_step:{key}")
                        continue
                    decision_by_step[decision_step] = decision
                if sorted(decision_by_step) != list(range(1, int(expected_count) + 1)):
                    issues.append(f"decisions:partial_sequence:{key}")
                decision_triggers = 0
                for step_number, step_row in ((index, by_step.get(index)) for index in sorted(by_step)):
                    decision = decision_by_step.get(step_number)
                    if decision is None or step_row is None:
                        continue
                    decision_original = _as_int(decision.get("original_action"))
                    decision_final = _as_int(decision.get("final_action", decision.get("action")))
                    if decision_original != _as_int(step_row.get("original_action")) or decision_final != _as_int(step_row.get("action")):
                        issues.append(f"decisions:step_action_mismatch:{key}:{step_number}")
                    if decision.get("triggered") is True:
                        decision_triggers += 1
                if branch.get("guard_trigger_count") is not None and branch["guard_trigger_count"] != decision_triggers:
                    issues.append(f"decisions:guard_trigger_count_mismatch:{key}")

    summary = {
        "task_vector_return": task_total,
        "energy_vector_return": vector_energy_total,
        "utility": utility_from_vectors(task_total, vector_energy_total),
        "accepted_count": accepted,
        "rejected_count": rejected,
        "task_unavailable_count": task_unavailable,
        "submit_count": submits,
        "feedback_histogram": dict(sorted(feedback_histogram.items())),
        "noop_count_from_actions": noop_actions,
        "action_change_count_from_steps": action_changes,
        "guard_trigger_count_from_steps": guard_triggers if guard_triggers else None,
        "completion_records": completion_records[-1] if completion_records else None,
        "raw_last_step": ordered_steps[-1] if ordered_steps else None,
    }
    return summary, sorted(set(issues))


def _base_output(inputs: Mapping[str, Any], expected_count: int = 0) -> dict[str, Any]:
    return {
        "schema": "ackguard-corrected-result-analysis/2.0.0",
        "status": "evaluation incomplete",
        "evaluation_status": "evaluation incomplete",
        "inputs": dict(inputs),
        "counts": {"expected_pairs": expected_count, "guard_branches": 0, "R_controls": 0, "paired": 0, "parents": 0},
        "primary": {
            "preference": list(PREFERENCE),
            "gamma": GAMMA,
            "reward_scales": dict(REWARD_SCALES),
            "tie_tolerance": TIE_TOLERANCE,
            "parent_aggregates": [],
            "parent_macro_guard_minus_R_utility": None,
            "parent_cluster_bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "ci": [None, None]},
        },
        "paired_results": [],
        "technical_stops": {"reasons": []},
    }


def branch_record(row: Mapping[str, Any], vectors: Mapping[str, tuple[float, float]], treatment: str) -> dict[str, Any]:
    """Compatibility helper for callers that only need a vector branch row."""

    key = control_key(row)
    if key not in vectors:
        raise ValueError(f"no step vector record for {key}")
    task, energy = vectors[key]
    return {
        "treatment": treatment,
        "control_branch_key": key,
        "parent_id": str(row.get("parent_id")),
        "repeat": _as_int(row.get("repeat")),
        "prefix_id": str(row.get("prefix_id")),
        "utility": utility_from_vectors(task, energy),
        "task_vector_return": task,
        "energy_vector_return": energy,
        "energy_used": float(row["energy_used"]) if _finite(row.get("energy_used")) else None,
        "technical_status": "complete",
    }


def analyze(
    guard_results: Path,
    guard_steps: Path,
    r_results: Path,
    r_steps: Path,
    manifest: Path | None = None,
) -> dict[str, Any]:
    inputs = {
        "manifest": str(manifest) if manifest is not None else None,
        "guard_results": str(guard_results),
        "guard_step_rewards": str(guard_steps),
        "R_results": str(r_results),
        "R_step_rewards": str(r_steps),
    }
    if manifest is None:
        output = _base_output(inputs)
        output["technical_stops"]["reasons"] = ["manifest_required"]
        return output

    try:
        manifest_payload = _load_json(manifest)
        expected_rows, by_key, by_prefix_repeat, manifest_issues = _manifest_expectations(manifest_payload)
    except ValueError as exc:
        output = _base_output(inputs)
        output["technical_stops"]["reasons"] = [f"manifest_error:{exc}"]
        return output
    output = _base_output(inputs, len(expected_rows))
    reasons: list[str] = list(manifest_issues)
    try:
        guard_branch_rows = read_jsonl(guard_results)
        guard_step_rows = read_jsonl(guard_steps)
        r_branch_rows = read_jsonl(r_results)
        r_step_rows = read_jsonl(r_steps)
    except (OSError, ValueError) as exc:
        output["technical_stops"]["reasons"] = reasons + [f"input_error:{exc}"]
        return output

    selected_guard, ignored_guard = _select_rows(guard_branch_rows, by_key, by_prefix_repeat, "guard")
    selected_r, ignored_r = _select_rows(r_branch_rows, by_key, by_prefix_repeat, "R")
    selected_guard_steps, ignored_guard_steps = _select_rows(guard_step_rows, by_key, by_prefix_repeat, "guard")
    selected_r_steps, ignored_r_steps = _select_rows(r_step_rows, by_key, by_prefix_repeat, "R")
    output["counts"].update({
        "guard_branches": len(selected_guard),
        "R_controls": len(selected_r),
        "ignored_guard_rows": ignored_guard,
        "ignored_R_rows": ignored_r,
        "ignored_guard_step_rows": ignored_guard_steps,
        "ignored_R_step_rows": ignored_r_steps,
    })
    duplicate_guard = sorted(key for key, values in selected_guard.items() if len(values) != 1)
    duplicate_r = sorted(key for key, values in selected_r.items() if len(values) != 1)
    if duplicate_guard:
        reasons.append("duplicate_guard:" + ",".join(duplicate_guard))
    if duplicate_r:
        reasons.append("duplicate_R:" + ",".join(duplicate_r))

    guard_summaries: dict[str, dict[str, Any]] = {}
    r_summaries: dict[str, dict[str, Any]] = {}
    guard_issues: dict[str, list[str]] = {}
    r_issues: dict[str, list[str]] = {}
    for expected in expected_rows:
        key = str(expected["pair_key"])
        if len(selected_guard.get(key, [])) == 1:
            summary, branch_errors = _branch_summary(selected_guard[key][0], expected, "guard")
            step_summary, step_errors = _validate_steps(selected_guard_steps.get(key, []), expected, "guard", summary)
            summary.update({field: value for field, value in step_summary.items() if field != "completion_records"})
            guard_summaries[key] = summary
            guard_issues[key] = sorted(set(branch_errors + step_errors))
        else:
            guard_issues[key] = [f"missing_guard:{key}"] if not selected_guard.get(key) else [f"duplicate_guard:{key}"]
        if len(selected_r.get(key, [])) == 1:
            summary, branch_errors = _branch_summary(selected_r[key][0], expected, "R")
            step_summary, step_errors = _validate_steps(selected_r_steps.get(key, []), expected, "R", summary)
            summary.update({field: value for field, value in step_summary.items() if field != "completion_records"})
            r_summaries[key] = summary
            r_issues[key] = sorted(set(branch_errors + step_errors))
        else:
            r_issues[key] = [f"missing_R:{key}"] if not selected_r.get(key) else [f"duplicate_R:{key}"]

    paired: list[dict[str, Any]] = []
    all_pair_valid = True
    for expected in expected_rows:
        key = str(expected["pair_key"])
        left = guard_summaries.get(key)
        right = r_summaries.get(key)
        pair_errors = list(guard_issues.get(key, [])) + list(r_issues.get(key, []))
        if left is not None and right is not None:
            for branch_summary, raw_row, treatment, step_rows in (
                (left, selected_guard[key][0], "guard", selected_guard_steps.get(key, [])),
                (right, selected_r[key][0], "R", selected_r_steps.get(key, [])),
            ):
                cohort_records, cohort_errors = _cohort_records(raw_row, treatment, step_rows)
                pair_errors.extend(cohort_errors)
                cohort = right.get("task_ids") or []
                if cohort_records is None:
                    continue
                missing = [task_id for task_id in cohort if task_id not in cohort_records]
                if missing:
                    pair_errors.append(f"cohort:task_records_missing:{treatment}:{key}:{','.join(missing)}")
                branch_summary["on_time_physical"] = sum(cohort_records.get(task_id, {}).get("physical_arrival_before_deadline") is True for task_id in cohort if task_id in cohort_records)
                branch_summary["on_time_host"] = sum(cohort_records.get(task_id, {}).get("host_confirmation_before_deadline") is True for task_id in cohort if task_id in cohort_records)
        if left is not None and right is not None and not pair_errors:
            difference = float(left["utility"]) - float(right["utility"])
            pair_row = {
                "control_branch_key": key,
                "parent_id": expected["parent_id"],
                "repeat": int(expected["repeat"]),
                "guard": left,
                "R": right,
                "guard_minus_R_utility": difference,
                "descriptive_tie": abs(difference) <= TIE_TOLERANCE,
                "evaluation_valid": True,
            }
        else:
            pair_row = {
                "control_branch_key": key,
                "parent_id": expected.get("parent_id"),
                "repeat": expected.get("repeat"),
                "guard": left,
                "R": right,
                "guard_minus_R_utility": None,
                "descriptive_tie": None,
                "evaluation_valid": False,
                "errors": sorted(set(pair_errors)),
            }
            all_pair_valid = False
            reasons.extend(pair_errors)
        paired.append(pair_row)

    output["counts"]["paired"] = len(paired)
    output["counts"]["parents"] = len({row.get("parent_id") for row in paired if row.get("parent_id") is not None})
    output["paired_results"] = paired
    output["technical_stops"].update({
        "missing_guard": sorted(key for key in by_key if key not in selected_guard),
        "missing_R": sorted(key for key in by_key if key not in selected_r),
        "duplicate_guard": duplicate_guard,
        "duplicate_R": duplicate_r,
        "ignored_R_rows_are_excluded": True,
        "feedback_non_rejected": list(FEEDBACK_NON_REJECTED),
    })

    parent_aggregates: list[dict[str, Any]] = []
    if all_pair_valid and len(paired) == 24 and not manifest_issues:
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in paired:
            by_parent[str(row["parent_id"])].append(row)
        for parent_id in sorted(by_parent):
            rows = sorted(by_parent[parent_id], key=lambda row: int(row["repeat"]))
            repeats = [int(row["repeat"]) for row in rows]
            if repeats != [0, 1, 2]:
                reasons.append(f"parent_repeat_sequence:{parent_id}")
                continue
            parent_aggregates.append({
                "parent_id": parent_id,
                "repeat_count": len(rows),
                "repeats": repeats,
                "guard_minus_R_utility_mean": _mean([float(row["guard_minus_R_utility"]) for row in rows]),
            })
    if len(parent_aggregates) != 8:
        all_pair_valid = False
        reasons.append(f"parent_aggregate_count:{len(parent_aggregates)}")

    if all_pair_valid and len(parent_aggregates) == 8:
        parent_values = [float(row["guard_minus_R_utility_mean"]) for row in parent_aggregates]
        macro = _mean(parent_values)
        rng = random.Random(BOOTSTRAP_SEED)
        bootstrap = [_mean([rng.choice(parent_values) for _ in parent_values]) for _ in range(BOOTSTRAP_REPLICATES)]
        output["status"] = "complete"
        output["evaluation_status"] = "complete"
        output["primary"].update({
            "parent_aggregates": parent_aggregates,
            "parent_macro_guard_minus_R_utility": macro,
            "parent_cluster_bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence": 0.95,
                "ci": [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)],
            },
        })
    else:
        output["primary"]["parent_aggregates"] = parent_aggregates
    output["technical_stops"]["reasons"] = sorted(set(reasons))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--guard-results", type=Path, required=True)
    parser.add_argument("--guard-step-rewards", type=Path, required=True)
    parser.add_argument("--r-results", type=Path, required=True)
    parser.add_argument("--r-step-rewards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.guard_results, args.guard_step_rewards, args.r_results, args.r_step_rewards, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "paired": result["counts"]["paired"], "parents": result["counts"]["parents"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
