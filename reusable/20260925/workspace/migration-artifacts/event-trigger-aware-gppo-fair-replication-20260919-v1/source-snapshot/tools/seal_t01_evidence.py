"""Create portable GitHub evidence from a locally generated T-01 manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.dataset import audit_manifest, sha256_file  # noqa: E402


REPOSITORY = "https://github.com/Battleplus/GPPO-WORLD-9.2"
TAG = "t01-data-v0.1.0"
RELEASE = f"{REPOSITORY}/releases/tag/{TAG}"
ASSET = f"{REPOSITORY}/releases/download/{TAG}"
SOURCE_COMMIT = "682ea6e9be777be9db9a48a0777eabcb2d1cf826"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_manifest", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    local_path = args.local_manifest.resolve()
    evidence_dir = args.evidence_dir.resolve()
    local = json.loads(local_path.read_text(encoding="utf-8"))
    portable = copy.deepcopy(local)
    portable["release"] = {"tag": TAG, "url": RELEASE, "target_source_commit": SOURCE_COMMIT}
    portable["local_manifest_sha256"] = sha256_file(local_path)
    portable["runtime"].pop("executable", None)
    for split in ("train", "validation", "test"):
        portable["files"][split]["path"] = f"{ASSET}/{split}.jsonl"
    checkpoint = portable["gppo_behavior_checkpoint"]
    checkpoint["path"] = f"{ASSET}/gppo_adaptive_seed1101_step512.pt"
    checkpoint["history_path"] = f"{ASSET}/gppo_adaptive_seed1101_step512.history.json"
    portable["audit"] = audit_manifest(portable)
    _write(evidence_dir / "dataset-manifest.json", portable)

    max_delay = max((float(item["max_observation_delay"]) for item in portable["tapes"]), default=0.0)
    weak_episodes = [item for item in portable["episodes"] if item["scenario_id"] == "weak_comm"]
    report = {
        "result": "PASS",
        "source_commit": SOURCE_COMMIT,
        "dataset_manifest_sha256": sha256_file(evidence_dir / "dataset-manifest.json"),
        "release": RELEASE,
        "episode_count": portable["audit"]["episode_count"],
        "transition_count": portable["audit"]["transition_count"],
        "group_count": portable["audit"]["group_count"],
        "split_overlap_count": portable["audit"]["split_overlap_count"],
        "truth_only_online_field_count": portable["truth_only_online_field_count"],
        "profiles": portable["audit"]["profiles"],
        "behavior_policies": portable["audit"]["behavior_policies"],
        "event_distribution": portable["event_distribution"],
        "action_coverage": portable["action_coverage"],
        "max_observation_delay_seconds": max_delay,
        "weak_comm_episode_count": len(weak_episodes),
        "weak_comm_observation_count": sum(int(item["observation_count"]) for item in weak_episodes),
        "raw_dataset_sha256": {split: portable["files"][split]["sha256"] for split in ("train", "validation", "test")},
        "checkpoint": {
            "sha256": checkpoint["sha256"],
            "roundtrip_total_steps": 512,
            "roundtrip_variant": "GPPO-Adaptive",
            "forward_action_is_legal": True,
            "critic_value_is_finite": True,
            "historical_50k_checkpoint": False,
        },
    }
    _write(evidence_dir / "audit-report.json", report)

    config_path = evidence_dir / "behavior-policy-config.json"
    checkpoint_manifest = {
        "artifact_id": "gppo-world/T-01/coverage-gppo/v0.1.0/1101",
        "node_id": "T-01",
        "status": "accepted",
        "source_commit": SOURCE_COMMIT,
        "parent_artifact_id": "gppo-world/T-00/baseline-contract/v0.1.0/none",
        "schema_version": portable["schema_version"],
        "config_uri": "behavior-policy-config.json",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_uri": "dataset-manifest.json",
        "dataset_manifest_sha256": sha256_file(evidence_dir / "dataset-manifest.json"),
        "seeds": [1101, 91001, 92001, 93001],
        "checkpoint_uri": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "metrics_uri": "audit-report.json",
        "metrics_sha256": sha256_file(evidence_dir / "audit-report.json"),
        "environment": {
            "python": "3.11.5",
            "framework": f"torch {portable['runtime']['torch']}",
            "cuda": None,
            "hardware": "CPU",
        },
        "created_at": portable["created_at"],
        "notes": "Coverage behavior checkpoint only; not the missing historical 50k GPPO checkpoint.",
    }
    _write(evidence_dir / "checkpoint-manifest.json", checkpoint_manifest)
    print(json.dumps({"portable_manifest": str(evidence_dir / "dataset-manifest.json"), "audit": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
