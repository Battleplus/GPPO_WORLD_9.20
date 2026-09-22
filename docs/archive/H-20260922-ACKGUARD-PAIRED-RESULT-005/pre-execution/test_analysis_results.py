from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("analyze_results.py")
SPEC = importlib.util.spec_from_file_location("ackguard_h005_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _identity(parent: int, repeat: int) -> dict[str, object]:
    parent_id = f"parent-{parent:02d}"
    prefix = f"{parent_id}|W1|seed-1101|prefix-0"
    return {
        "parent_id": parent_id,
        "prefix_id": prefix,
        "condition": "W1",
        "model_seed": 1101,
        "repeat": repeat,
        "exogenous_key": f"replan-feature-v1|{parent_id}|W1|0|repeat-{repeat}",
        "branch_id": f"{prefix}|repeat-{repeat}|guard-on",
        "control_branch_key": f"{prefix}|repeat-{repeat}|mode-R",
    }


def _manifest() -> dict[str, object]:
    rows = []
    for parent in range(8):
        for repeat in range(3):
            ident = _identity(parent, repeat)
            rows.append(
                {
                    "branch_id": ident["branch_id"],
                    "control_branch_key": ident["control_branch_key"],
                    "condition": ident["condition"],
                    "historical_exogenous_key": ident["exogenous_key"],
                    "model_seed": ident["model_seed"],
                    "parent_id": ident["parent_id"],
                    "prefix_id": ident["prefix_id"],
                    "prefix_index": 0,
                    "repeat": repeat,
                    "split": "train",
                }
            )
    return {"counts": {"branches": 24, "parents": 8, "repeats": 3}, "rows": rows}


def _records() -> dict[str, dict[str, object]]:
    return {
        "task-0": {"physical_arrival_before_deadline": True, "host_confirmation_before_deadline": True},
        "task-1": {"physical_arrival_before_deadline": False, "host_confirmation_before_deadline": True},
        "task-2": {"physical_arrival_before_deadline": True, "host_confirmation_before_deadline": False},
    }


def _probabilities(original: int, selected: int) -> list[dict[str, object]]:
    return [
        {
            "action": action,
            "legal": True,
            "probability": 1.0 / 25.0,
            "original_argmax": action == original,
            "selected": action == selected,
        }
        for action in range(25)
    ]


def _step(ident: dict[str, object], treatment: str, step: int, task_value: float, energy_value: float) -> dict[str, object]:
    selected = 24 if treatment == "guard" and step == 2 else 1
    original = 2 if treatment == "guard" and step == 2 else selected
    feedback = "noop" if selected == 24 else ("task_unavailable" if treatment == "R" and step == 1 else "accepted")
    records = _records()
    row: dict[str, object] = {
        "step": step,
        "gamma_index": step - 1,
        "vector_reward": [task_value, energy_value],
        "gamma": 0.99,
        "reward_scales": {"task": 0.5, "energy": 1.0},
        "behavior_preference": [0.8, 0.2],
        "action": selected,
        "legal_mask": [1] * 25,
        "submit_command": True,
        "energy_before": 10.0 - 0.25 * (step - 1),
        "energy_after": 10.0 - 0.25 * step,
        "energy_used_delta": 0.25,
        "terminated": step == 2,
        "truncated": False,
        "feedback": feedback,
        "parent_id": ident["parent_id"],
        "prefix_id": ident["prefix_id"],
        "condition": ident["condition"],
        "model_seed": ident["model_seed"],
        "repeat": ident["repeat"],
        "mode": "R" if treatment == "R" else None,
    }
    if treatment == "guard":
        row.update(
            {
                "branch_id": ident["branch_id"],
                "control_branch_key": ident["control_branch_key"],
                "exogenous_key": ident["exogenous_key"],
                "original_action": original,
                "probabilities": _probabilities(original, selected),
                "actual_feedback": {"feedback": feedback, "completion_records": records},
            }
        )
    else:
        # Historical train labels omit control/exogenous/original-action fields.
        row["mode"] = "R"
    return row


def _branch(ident: dict[str, object], treatment: str, parent: int, repeat: int) -> dict[str, object]:
    records = _records()
    row: dict[str, object] = {
        "parent_id": ident["parent_id"],
        "prefix_id": ident["prefix_id"],
        "condition": ident["condition"],
        "model_seed": ident["model_seed"],
        "repeat": repeat,
        "exogenous_key": ident["exogenous_key"],
        "split": "train",
        "env_steps": 2,
        "initial_counts": {"completed": 0, "expired": 0},
        "final_counts": {"completed": 2, "expired": 0},
        "initial_energy": 10.0,
        "final_energy": 9.5,
        "energy_used": 0.5,
        "terminated": True,
        "truncated": False,
        "end_reason": "terminated",
    }
    if treatment == "guard":
        row.update(
            {
                "branch_id": ident["branch_id"],
                "control_branch_key": ident["control_branch_key"],
                "actor_forward_calls": 0,
                "world_forward_calls": 0,
                "command_count": 2,
                "task_unavailable_count": 0,
                "guard_trigger_count": 0,
                "action_change_count": 1,
                "noop_count": 1,
                "all_completed": 2,
                "last_info": {"feedback": "noop", "completion_records": records},
            }
        )
    else:
        row.update(
            {
                "mode": "R",
                "task_ids": ["task-0", "task-1"],
                "task_outcome": {key: records[key] for key in ("task-0", "task-1")},
                "actor_calls": 0,
                "world_calls": 0,
            }
        )
    return row


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    manifest = tmp_path / "branch-manifest.json"
    guard_results = tmp_path / "guard-branch-results.jsonl"
    guard_steps = tmp_path / "guard-step-vector-rewards.jsonl"
    r_results = tmp_path / "r-branch-results.jsonl"
    r_steps = tmp_path / "r-step-vector-rewards.jsonl"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    guard_rows = []
    guard_step_rows = []
    r_rows = []
    r_step_rows = []
    for parent in range(8):
        for repeat in range(3):
            ident = _identity(parent, repeat)
            guard_rows.append(_branch(ident, "guard", parent, repeat))
            r_rows.append(_branch(ident, "R", parent, repeat))
            # Unequal parent effects and repeat variation exercise cluster means.
            guard_step_rows.extend(
                [
                    _step(ident, "guard", 1, float(parent + repeat + 1), 0.0),
                    _step(ident, "guard", 2, 0.0, 0.0),
                ]
            )
            r_step_rows.extend([_step(ident, "R", 1, 0.0, 0.0), _step(ident, "R", 2, 0.0, 0.0)])
    for path, rows in ((guard_results, guard_rows), (guard_steps, guard_step_rows), (r_results, r_rows), (r_steps, r_step_rows)):
        path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
    return manifest, guard_results, guard_steps, r_results, r_steps


def _run(paths: tuple[Path, Path, Path, Path, Path]) -> dict[str, object]:
    manifest, guard_results, guard_steps, r_results, r_steps = paths
    return MODULE.analyze(guard_results, guard_steps, r_results, r_steps, manifest)


def test_known_utility_and_parent_cluster_aggregation(tmp_path: Path):
    result = _run(_fixture(tmp_path))
    assert result["status"] == "complete"
    assert result["counts"]["paired"] == 24
    assert result["counts"]["ignored_R_rows"] == 0
    assert len(result["primary"]["parent_aggregates"]) == 8
    expected_parent_means = [(parent + 1 + 1) * 0.8 * 0.5 for parent in range(8)]
    observed = [row["guard_minus_R_utility_mean"] for row in result["primary"]["parent_aggregates"]]
    assert observed == pytest.approx(expected_parent_means)
    assert result["primary"]["parent_macro_guard_minus_R_utility"] == pytest.approx(sum(expected_parent_means) / 8)
    assert result["primary"]["parent_cluster_bootstrap"]["replicates"] == 10000
    assert result["primary"]["parent_cluster_bootstrap"]["seed"] == 20260922


def test_real_field_shape_direct_actual_feedback_and_secondary_counts(tmp_path: Path):
    result = _run(_fixture(tmp_path))
    assert result["status"] == "complete"
    pair = result["paired_results"][0]
    guard = pair["guard"]
    r_control = pair["R"]
    assert guard["actor_forward_calls"] == 0
    assert guard["world_forward_calls"] == 0
    assert r_control["actor_forward_calls"] == 0
    assert r_control["world_forward_calls"] == 0
    assert guard["accepted_count"] == 1
    assert guard["rejected_count"] == 0
    assert guard["submit_count"] == 2
    assert guard["noop_count"] == 1
    assert guard["feedback_histogram"] == {"accepted": 1, "noop": 1}
    assert r_control["accepted_count"] == 1
    assert r_control["rejected_count"] == 1
    assert r_control["task_unavailable_count"] == 1
    assert r_control["submit_count"] == 2
    assert r_control["feedback_histogram"]["task_unavailable"] == 1
    assert guard["on_time_physical"] == 1
    assert guard["on_time_host"] == 2
    assert r_control["on_time_physical"] == 1
    assert r_control["on_time_host"] == 2
    assert guard["all_task_completed"] == 2
    assert r_control["all_task_completed"] == 2
    assert guard["completed_delta"] == 2
    assert r_control["completed_delta"] == 2
    assert set(result["technical_stops"]["feedback_non_rejected"]) == {"accepted", "awaiting_ack", "noop", "reuse_existing"}


def test_r_selection_ignores_other_train_rows(tmp_path: Path):
    paths = _fixture(tmp_path)
    extra = _branch(_identity(0, 0), "R", 0, 0)
    extra.update({"prefix_id": "other-parent|W1|seed-9999|prefix-9", "parent_id": "other-parent", "model_seed": 9999, "condition": "W2"})
    paths[3].open("a", encoding="utf-8").write(json.dumps(extra) + "\n")
    result = _run(paths)
    assert result["status"] == "complete"
    assert result["counts"]["R_controls"] == 24
    assert result["counts"]["ignored_R_rows"] == 1


@pytest.mark.parametrize(
    "mutation,needle",
    [
        ("duplicate", "duplicate_guard"),
        ("missing", "missing_R"),
        ("partial", "steps:count"),
        ("nonfinite", "step:vector_reward"),
        ("identity", "branch:identity_parent_id"),
    ],
)
def test_incomplete_inputs_reject_primary_and_ci(tmp_path: Path, mutation: str, needle: str):
    paths = _fixture(tmp_path)
    manifest, guard_results, guard_steps, r_results, r_steps = paths
    if mutation == "duplicate":
        first = json.loads(guard_results.read_text(encoding="utf-8").splitlines()[0])
        guard_results.open("a", encoding="utf-8").write(json.dumps(first) + "\n")
    elif mutation == "missing":
        rows = r_results.read_text(encoding="utf-8").splitlines()
        r_results.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    elif mutation == "partial":
        rows = guard_steps.read_text(encoding="utf-8").splitlines()
        guard_steps.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8")
    elif mutation == "nonfinite":
        rows = [json.loads(line) for line in guard_steps.read_text(encoding="utf-8").splitlines()]
        rows[0]["vector_reward"][0] = float("nan")
        guard_steps.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
    elif mutation == "identity":
        rows = [json.loads(line) for line in guard_results.read_text(encoding="utf-8").splitlines()]
        rows[0]["parent_id"] = "wrong-parent"
        guard_results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = _run(paths)
    assert result["status"] == "evaluation incomplete"
    assert result["primary"]["parent_macro_guard_minus_R_utility"] is None
    assert result["primary"]["parent_cluster_bootstrap"]["ci"] == [None, None]
    assert any(needle in reason for reason in result["technical_stops"]["reasons"])


def test_nested_actual_feedback_is_rejected(tmp_path: Path):
    paths = _fixture(tmp_path)
    guard_steps = paths[2]
    rows = [json.loads(line) for line in guard_steps.read_text(encoding="utf-8").splitlines()]
    rows[0]["actual_feedback"] = {"info": rows[0]["actual_feedback"]}
    guard_steps.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = _run(paths)
    assert result["status"] == "evaluation incomplete"
    assert any("nested_actual_feedback_info" in reason for reason in result["technical_stops"]["reasons"])


def test_no_vector_return_fallback_and_no_silent_duplicate_step(tmp_path: Path):
    paths = _fixture(tmp_path)
    guard_steps = paths[2]
    rows = [json.loads(line) for line in guard_steps.read_text(encoding="utf-8").splitlines()]
    rows[0].pop("vector_reward")
    rows[1]["step"] = 1
    guard_steps.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = _run(paths)
    assert result["status"] == "evaluation incomplete"
    assert result["primary"]["parent_cluster_bootstrap"]["ci"] == [None, None]
    assert any("step:vector_reward" in reason or "steps:duplicate_step" in reason for reason in result["technical_stops"]["reasons"])
