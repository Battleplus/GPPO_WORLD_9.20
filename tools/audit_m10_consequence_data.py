"""Audit generated M-10 consequence data without training or model selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import audit_consequence_manifest, load_consequence_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = audit_consequence_manifest(manifest, data)
    report = {"manifest_audit": audit, "splits": {}}
    for split in ("train", "validation", "test", "ood"):
        path = data / manifest["files"][split]["path"]
        examples = load_consequence_jsonl(path, expected_horizon_steps=int(manifest["prediction_horizon_steps"]), strict_identity=True)
        prefixes = {(example.parent_episode_id, example.prefix_id) for example in examples}
        candidate_counts = Counter((example.parent_episode_id, example.prefix_id) for example in examples)
        values = {
            name: [float(getattr(example.target, name)) for example in examples]
            for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")
        }
        report["splits"][split] = {
            "records": len(examples),
            "parent_episode_count": len({example.parent_episode_id for example in examples}),
            "prefix_count": len(prefixes),
            "candidate_branch_count": len(examples),
            "candidate_counts_per_prefix": sorted(Counter(candidate_counts.values()).items()),
            "label_masks": {name: int(sum(bool(example.masks[name]) for example in examples)) for name in values},
            "label_ranges": {name: [min(items), max(items)] for name, items in values.items()},
            "deadline_risk_positive_negative": {"positive": sum(value > 0.5 for value in values["deadline_risk"]), "negative": sum(value <= 0.5 for value in values["deadline_risk"])},
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
