"""Audit T-01 raw JSONL, hashes, split isolation and causal invariants."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.dataset import SPLITS, audit_manifest, sha256_file  # noqa: E402
from gppo_world.registry import SCHEMA_VERSION  # noqa: E402


FORBIDDEN = {
    "future_action",
    "future_action_mask",
    "future_confirmation",
    "future_graph",
    "future_observation",
    "ground_truth_event",
    "internal_drop_decision",
    "optimal_action",
    "oracle_action",
    "truth_event",
    "truth_event_occurred_at",
    "truth_occurred_at",
}


def _forbidden_paths(value, path="root"):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN:
                found.append(child_path)
            found.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    manifest_audit = audit_manifest(manifest)
    errors.extend(manifest_audit["errors"])
    observed_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    profile_policy_split = defaultdict(set)
    episode_steps: dict[str, int] = {}
    truth_only_count = 0
    for split in SPLITS:
        record = manifest["files"][split]
        path = Path(record["path"])
        if not path.is_file():
            errors.append(f"missing {split} file: {path}")
            continue
        if sha256_file(path) != record["sha256"]:
            errors.append(f"{split} sha256 mismatch")
        line_count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                line_count += 1
                if row.get("schema_version") != SCHEMA_VERSION:
                    errors.append(f"{split}:{line_number} schema mismatch")
                if not str(row["episode_id"]).startswith(f"{split}/"):
                    errors.append(f"{split}:{line_number} episode split mismatch")
                expected_step = episode_steps.get(row["episode_id"], 0)
                if int(row["step"]) != expected_step:
                    errors.append(f"{split}:{line_number} non-contiguous step")
                episode_steps[row["episode_id"]] = expected_step + 1
                decision_time = float(row["decision_time"])
                for evidence in row["evidence_t"]:
                    if float(evidence["received_at"]) > decision_time:
                        errors.append(f"{split}:{line_number} future evidence")
                    paths = _forbidden_paths(evidence.get("payload", {}), "evidence.payload")
                    truth_only_count += len(paths)
                    errors.extend(f"{split}:{line_number} forbidden {path}" for path in paths)
                execution = row["execution"]
                if int(execution["graph_version"]) != int(row["graph_t"]["graph_version"]):
                    errors.append(f"{split}:{line_number} graph version mismatch")
                action = execution["executed_action"]
                if execution["accepted"]:
                    if action is None or not bool(row["graph_t"]["action_mask"][int(action)]):
                        errors.append(f"{split}:{line_number} illegal executed action")
                    else:
                        action_counts[str(action)] += 1
                observed_counts[split] += 1
                profile_policy_split[(split, row["scenario_id"])].add(row["behavior_policy"])
        if line_count != int(record["transitions"]):
            errors.append(f"{split} transition count mismatch")
    required_policies = {"random_legal", "greedy", "gppo"}
    for split in SPLITS:
        for profile in ("normal", "single", "sequential", "overlap", "burst", "long_gap", "weak_comm"):
            if profile_policy_split[(split, profile)] != required_policies:
                errors.append(f"{split}/{profile} missing behavior policy")
    if set(action_counts) != {str(index) for index in range(17)}:
        errors.append("action coverage does not include all 0..16 actions")
    if truth_only_count != 0:
        errors.append("truth-only online fields are nonzero")
    checkpoint = Path(manifest["gppo_behavior_checkpoint"]["path"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != manifest["gppo_behavior_checkpoint"]["sha256"]:
        errors.append("GPPO behavior checkpoint missing or hash mismatch")
    report = {
        "passed": not errors,
        "errors": errors,
        "manifest_audit": manifest_audit,
        "observed_transitions": dict(observed_counts),
        "action_coverage": dict(sorted(action_counts.items(), key=lambda item: int(item[0]))),
        "truth_only_online_field_count": truth_only_count,
        "checkpoint_sha256": manifest["gppo_behavior_checkpoint"]["sha256"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
