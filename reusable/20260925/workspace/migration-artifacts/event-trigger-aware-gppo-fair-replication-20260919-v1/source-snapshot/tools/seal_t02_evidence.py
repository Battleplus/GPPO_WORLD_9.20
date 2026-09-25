"""Seal portable, hash-linked evidence for the accepted T-02 checkpoint release."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.dataset import sha256_file  # noqa: E402


REPOSITORY = "https://github.com/Battleplus/GPPO-WORLD-9.2"
TAG = "t02-base-wm-v0.1.0"
RELEASE = f"{REPOSITORY}/releases/tag/{TAG}"
ASSET = f"{REPOSITORY}/releases/download/{TAG}"
BASELINE_COMMIT = "2a9bb9f87b9d543df144f4d108ba970c924151f9"
T01_TAG = "t01-data-v0.1.0"
T01_ASSET = f"{REPOSITORY}/releases/download/{T01_TAG}"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_output", type=Path)
    parser.add_argument("failed_metrics", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    output = args.local_output.resolve()
    evidence = args.evidence_dir.resolve()
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("result") != "PASS" or not all(metrics.get("gates", {}).values()):
        raise RuntimeError("T-02 metrics are not a complete PASS")
    if metrics["input_audit"].get("truth_only_online_field_count") != 0:
        raise RuntimeError("truth-only online inputs are nonzero")
    if metrics["action_shuffle"].get("illegal_alternative_count") != 0:
        raise RuntimeError("illegal action counterfactuals are nonzero")

    portable = copy.deepcopy(metrics)
    portable["source_commit"] = args.source_commit
    portable["source_baseline_commit"] = BASELINE_COMMIT
    portable["release"] = {"tag": TAG, "url": RELEASE, "target_source_commit": args.source_commit}
    for split in ("train", "validation", "test"):
        portable["dataset"][split]["path"] = f"{T01_ASSET}/{split}.jsonl"
    portable["input_audit"]["manifest_path"] = (
        f"{REPOSITORY}/blob/main/nodes/T-01/evidence/dataset-manifest.json"
    )
    portable["checkpoints"]["graph_wm"]["path"] = f"{ASSET}/graph_wm_seed20260902.pt"
    portable["checkpoints"]["flat_gru"]["path"] = f"{ASSET}/flat_gru_seed20260902.pt"
    _write(evidence / "metrics.json", portable)

    input_audit = copy.deepcopy(portable["input_audit"])
    input_audit["source_commit"] = args.source_commit
    input_audit["source_baseline_commit"] = BASELINE_COMMIT
    _write(evidence / "input-audit.json", input_audit)

    training_config = {
        "source_commit": args.source_commit,
        "source_baseline_commit": BASELINE_COMMIT,
        "seed": metrics["training_config"]["seed"],
        "command": (
            "D:/anaconda/python.exe tools/train_t02_world_model.py artifacts/T01/dataset "
            "artifacts/T02 --epochs 80 --patience 15 --seed 20260902"
        ),
        "training_config": metrics["training_config"],
        "model_config": metrics["model_config"],
        "dataset_sha256": {
            split: metrics["dataset"][split]["sha256"] for split in ("train", "validation", "test")
        },
        "runtime": metrics["runtime"],
        "training_seconds": metrics["training_seconds"],
        "selection": metrics["selection"],
    }
    _write(evidence / "training-config.json", training_config)

    graph_history = output / "graph-training-history.json"
    flat_history = output / "flat-training-history.json"
    failed_path = args.failed_metrics.resolve()
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    failed_record = {
        "result": failed.get("result"),
        "created_at": failed.get("created_at"),
        "reason": "Pre-calibration sparse threshold run retained as a negative result.",
        "failed_gates": sorted(name for name, passed in failed.get("gates", {}).items() if not passed),
        "graph_state_mae": failed.get("test", {}).get("graph_wm", {}).get("state_mae"),
        "last_value_state_mae": failed.get("test", {}).get("last_value", {}).get("state_mae"),
        "local_sha256": sha256_file(failed_path),
        "asset_uri": f"{ASSET}/failed-pre-calibration-metrics.json",
    }
    _write(evidence / "failed-run.json", failed_record)

    log_manifest = {
        "source_commit": args.source_commit,
        "seed": metrics["training_config"]["seed"],
        "final_exit_code": 0,
        "final_result": metrics["result"],
        "assets": {
            "metrics": {"uri": f"{ASSET}/metrics.json", "sha256": sha256_file(metrics_path)},
            "input_audit": {
                "uri": f"{ASSET}/input-audit.json",
                "sha256": sha256_file(output / "input-audit.json"),
            },
            "graph_history": {"uri": f"{ASSET}/graph-training-history.json", "sha256": sha256_file(graph_history)},
            "flat_history": {"uri": f"{ASSET}/flat-training-history.json", "sha256": sha256_file(flat_history)},
            "failed_run": {"uri": failed_record["asset_uri"], "sha256": failed_record["local_sha256"]},
        },
    }
    _write(evidence / "training-log-manifest.json", log_manifest)

    graph_checkpoint = metrics["checkpoints"]["graph_wm"]
    flat_checkpoint = metrics["checkpoints"]["flat_gru"]
    checkpoint_manifest = {
        "artifact_id": "gppo-world/T-02/base-wm/v0.1.0/20260902",
        "node_id": "T-02",
        "status": "accepted",
        "source_commit": args.source_commit,
        "source_baseline_commit": BASELINE_COMMIT,
        "parent_artifact_id": "gppo-world/T-01/coverage-gppo/v0.1.0/1101",
        "schema_version": metrics["schema_version"],
        "registry_sha256": metrics["registry_sha256"],
        "seed": metrics["training_config"]["seed"],
        "release": RELEASE,
        "accepted_checkpoint_uri": f"{ASSET}/graph_wm_seed20260902.pt",
        "accepted_checkpoint_sha256": graph_checkpoint["sha256"],
        "flat_baseline_checkpoint_uri": f"{ASSET}/flat_gru_seed20260902.pt",
        "flat_baseline_checkpoint_sha256": flat_checkpoint["sha256"],
        "checkpoint_roundtrip_max_abs": metrics["checkpoint_roundtrip_max_abs"],
        "flat_checkpoint_roundtrip_max_abs": metrics["flat_checkpoint_roundtrip_max_abs"],
        "config_uri": "training-config.json",
        "config_sha256": sha256_file(evidence / "training-config.json"),
        "metrics_uri": "metrics.json",
        "metrics_sha256": sha256_file(evidence / "metrics.json"),
        "input_audit_uri": "input-audit.json",
        "input_audit_sha256": sha256_file(evidence / "input-audit.json"),
        "training_log_manifest_uri": "training-log-manifest.json",
        "training_log_manifest_sha256": sha256_file(evidence / "training-log-manifest.json"),
        "failed_run_uri": "failed-run.json",
        "failed_run_sha256": sha256_file(evidence / "failed-run.json"),
        "created_at": metrics["created_at"],
    }
    _write(evidence / "checkpoint-manifest.json", checkpoint_manifest)
    print(json.dumps({"result": "PASS", "checkpoint_manifest": checkpoint_manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
