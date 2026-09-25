"""Derive a corrected metric view from an existing smoke ledger; never reruns env."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_gppo import build_observed_event_labels


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    raw = root / "training-ledger.jsonl"
    corrected = root / "training-ledger-v3.jsonl"
    erratum = root / "ledger-metric-erratum-v3.json"
    manifest = root / "artifact-manifest-v3.json"
    if not raw.is_file():
        raise SystemExit(f"raw ledger missing: {raw}")
    if any(path.exists() for path in (corrected, erratum, manifest)):
        raise SystemExit("derived correction already exists; refusing overwrite")
    raw_hash = digest(raw)
    corrected_rows = 0
    previous_by_episode = {}
    with raw.open(encoding="utf-8") as source, corrected.open("x", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            records = row.get("completion_records", {})
            key = (row.get("tape_id"), row.get("episode"))
            prior = previous_by_episode.get(key)
            is_episode_start = prior is None or float(row["time"]) <= float(prior["time"])
            prior_counts = {"completed": 0, "expired": 0} if is_episode_start else prior["counts"]
            counts = row.get("physical_or_deadline_counts", {})
            new_physical = max(0, int(counts.get("completed", 0)) - int(prior_counts.get("completed", 0)))
            new_deadline = max(0, int(counts.get("expired", 0)) - int(prior_counts.get("expired", 0)))
            expected_task_reward = (new_physical - new_deadline) / 6.0
            if abs(float(row["vector_reward"][0]) - expected_task_reward) > 1e-6:
                raise ValueError(f"task reward/count delta mismatch at step {row['step']}")
            # In v1 the value named host_confirmed_total was actually the
            # environment's physical-arrival `counts.completed` field.
            row.pop("host_confirmed_total", None)
            row["physical_on_time_total"] = int(counts.get("completed", 0))
            row["physical_deadline_failure_total"] = int(counts.get("expired", 0))
            row["host_confirmed_total"] = sum(item.get("host_confirmation_time") is not None for item in records.values())
            row["host_on_time_total"] = sum(item.get("host_confirmation_before_deadline") is True for item in records.values())
            row["task_consequence_target"] = [new_physical / 6.0, new_deadline / 6.0]
            row["task_consequence_valid"] = True
            row["state_target_valid"] = all(np.isfinite(value) for value in row["next_observation_flat"])
            row["vector_reward_valid"] = all(np.isfinite(value) for value in row["vector_reward"])
            row["censor_reason"] = None
            current_public = {
                "time": row["time"], "uavs": row["current_public_uavs"], "tasks": row["current_public_tasks"],
                "public_entity_ids": row["public_entity_ids"],
            }
            next_public = {
                "time": row["time"] + 1.0, "uavs": row["next_public_uavs"], "tasks": row["next_public_tasks"],
                "public_entity_ids": row["next_public_entity_ids"],
            }
            recomputed_event = build_observed_event_labels(current_public, next_public, initial_energy=9.0)
            if not np.array_equal(np.asarray(row["event_labels"], dtype=np.float32), recomputed_event["labels"]):
                raise ValueError(f"event label replay mismatch at step {row['step']}")
            if not np.array_equal(np.asarray(row["event_mask"], dtype=np.bool_), recomputed_event["mask"]):
                raise ValueError(f"event mask replay mismatch at step {row['step']}")
            row["event_mask_reasons"] = recomputed_event["reasons"]
            row["ledger_derivation"] = "raw observation/records; labels and masks statically replayed; no simulator step"
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            corrected_rows += 1
            previous_by_episode[key] = {
                "time": float(row["time"]),
                "counts": {"completed": int(counts.get("completed", 0)), "expired": int(counts.get("expired", 0))},
            }
    erratum_data = {
        "status": "derived_metric_and_target_view_created_raw_preserved",
        "raw_ledger": raw.name, "raw_sha256": raw_hash,
        "derived_ledger": corrected.name, "derived_rows": corrected_rows,
        "correction": "The v1 field host_confirmed_total held physical_or_deadline_counts.completed. The v3 view recomputes physical/deadline/host totals from source counts and per-task completion_records, reconstructs one-step task-consequence labels, and statically replays event labels/masks from adjacent saved public observations.",
        "source_values_preserved": True,
        "no_simulation_or_training_rerun": True,
    }
    erratum.write_text(json.dumps(erratum_data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != manifest.name:
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest.write_text(json.dumps({"files": files, "derived_after_run": True}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**erratum_data, "derived_sha256": digest(corrected), "erratum_sha256": digest(erratum), "manifest_sha256": digest(manifest)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
