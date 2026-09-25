"""Integrity audit for a completed frozen joint-consequence validation run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = {"current_rule", "public_joint", "zero_delta", "joint_model"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    selections = run / "selections.jsonl"
    branches = run / "selected-branches.jsonl"
    identity = load_json(run / "run-identity.json")
    selection_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    selection_duplicates = []
    selection_count = 0
    for line in selections.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        selection_count += 1
        key = (row["condition"], row["parent_episode_id"], row["prefix_id"])
        if key in selection_rows:
            selection_duplicates.append(key)
        selection_rows[key] = row
    branch_count = 0
    duplicate_keys = []
    branch_keys: set[tuple[str, str, str, int, str]] = set()
    groups: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    branch_selection_mismatches = []
    task_set_mismatches = []
    safety_violations = 0
    censored = Counter()
    method_counts = Counter()
    for line in branches.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        branch_count += 1
        condition, parent, prefix = row["condition"], row["parent_episode_id"], row["prefix_id"]
        repeat, method = int(row["repeat"]), row["method"]
        key = (condition, parent, prefix, repeat, method)
        if key in branch_keys:
            duplicate_keys.append(key)
        branch_keys.add(key)
        method_counts[(condition, method)] += 1
        groups[(condition, parent, prefix, repeat)][method] = row
        selection = selection_rows.get((condition, parent, prefix))
        if selection and int(row["branch"]["selected_action"]) != int(selection["selection"]["selected_actions"][method]):
            branch_selection_mismatches.append(key)
        expected_task_set = set(selection.get("task_set", [])) if selection else set()
        actual_task_set = {item["task_id"] for item in row["branch"].get("task_outcomes", [])}
        if selection and expected_task_set != actual_task_set:
            task_set_mismatches.append(key)
        if row["branch"]["metrics"]["on_time_count"] is None:
            censored[(condition, method)] += 1
        safety_violations += int(row["branch"]["metrics"].get("safety_violation_count", 0))
    incomplete_groups = [key for key, value in groups.items() if set(value) != METHODS]
    paired_key_mismatch = []
    for key, value in groups.items():
        exogenous = {row["branch"]["label_provenance"]["shared_exogenous_key"] for row in value.values()}
        if len(exogenous) != 1:
            paired_key_mismatch.append(key)
    input_model_hash = sha256_file(args.model)
    checks = {
        "run_identity_status_completed": identity.get("status") == "completed",
        "model_hash_matches_run_identity": identity.get("frozen_model", {}).get("sha256") == input_model_hash,
        "selection_rows_128": selection_count == 128,
        "selection_duplicates_zero": not selection_duplicates,
        "branch_identity_duplicates_zero": not duplicate_keys,
        "every_group_has_four_methods": not incomplete_groups,
        "selection_action_matches_branch": not branch_selection_mismatches,
        "fixed_task_set_matches_selection": not task_set_mismatches,
        "paired_exogenous_key_matches": not paired_key_mismatch,
        "safety_violations_zero": safety_violations == 0,
    }
    result = {
        "schema": "world-gppo-9.11-joint-consequence-independent-validation-integrity/0.1.0",
        "run_id": identity.get("run_id"),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "counts": {"selection_rows": selection_count, "branch_rows": branch_count, "complete_prefix_repeat_groups": sum(set(value) == METHODS for value in groups.values()), "incomplete_groups": len(incomplete_groups), "safety_violations": safety_violations},
        "method_counts": {f"{condition}/{method}": count for (condition, method), count in sorted(method_counts.items())},
        "censored_counts": {f"{condition}/{method}": count for (condition, method), count in sorted(censored.items())},
        "sha256": {"model": input_model_hash, "selections": sha256_file(selections), "selected_branches": sha256_file(branches)},
        "anomalies": {"selection_duplicates": selection_duplicates[:10], "branch_identity_duplicates": duplicate_keys[:10], "incomplete_groups": [list(key) for key in incomplete_groups[:10]], "selection_action_mismatches": branch_selection_mismatches[:10], "task_set_mismatches": task_set_mismatches[:10], "paired_exogenous_key_mismatches": [list(key) for key in paired_key_mismatch[:10]]},
    }
    (run / "integrity-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
