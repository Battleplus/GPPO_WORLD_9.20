"""No-update audit for the narrowed system-energy residual hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import load_consequence_jsonl  # noqa: E402
from gppo_world.consequence_model import ConsequenceModelConfig, Graph5ActionConsequenceWorldModel  # noqa: E402
from tools.evaluate_m10_consequence_quality import grouped, model_predictions, public_physics, ranking_metrics  # noqa: E402


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted))) if actual.size else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent fallback")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    horizon = int(manifest["prediction_horizon_steps"])
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Graph5ActionConsequenceWorldModel(ConsequenceModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    splits: dict[str, Any] = {}
    train_examples = load_consequence_jsonl(
        args.data / manifest["files"]["train"]["path"], expected_horizon_steps=horizon, strict_identity=True
    )
    train_energy = np.asarray([item.target.energy_delta for item in train_examples if item.masks["energy_delta"]], dtype=np.float64)
    train_physics = np.asarray([public_physics(item, float(horizon))["energy_delta"] for item in train_examples if item.masks["energy_delta"]], dtype=np.float64)
    residual_offset = float(np.mean(train_energy - train_physics))
    for split in ("train", "validation"):
        examples = load_consequence_jsonl(
            args.data / manifest["files"][split]["path"], expected_horizon_steps=horizon, strict_identity=True
        )
        predictions = model_predictions(model, examples, device)
        valid = [item for item in examples if item.masks["energy_delta"]]
        actual = np.asarray([item.target.energy_delta for item in valid], dtype=np.float64)
        physics = np.asarray([public_physics(item, float(horizon))["energy_delta"] for item in valid], dtype=np.float64)
        learned = np.asarray([predictions[(item.parent_episode_id, item.prefix_id, item.target.action)]["energy_delta"] for item in valid], dtype=np.float64)
        physics_residual = physics + residual_offset
        splits[split] = {
            "records": len(examples),
            "parents": len({item.parent_episode_id for item in examples}),
            "prefixes": len({(item.parent_episode_id, item.prefix_id) for item in examples}),
            "valid_energy_labels": len(valid),
            "target_scope": "system energy across all UAV resources; candidate travel/service remain local",
            "train_residual_offset": residual_offset,
            "mae": {
                "public_physics_zero_residual": mae(actual, physics),
                "public_physics_plus_train_mean_residual": mae(actual, physics_residual),
                "previous_absolute_model": mae(actual, learned),
            },
            "candidate_ranking": {
                "previous_absolute_model": ranking_metrics(examples, predictions, "energy_delta"),
            },
        }
        physics_values = {}
        for item in examples:
            key = (item.parent_episode_id, item.prefix_id, item.target.action)
            physics_values[key] = {"energy_delta": public_physics(item, float(horizon))["energy_delta"]}
        splits[split]["candidate_ranking"]["public_physics"] = ranking_metrics(examples, physics_values, "energy_delta")
        splits[split]["candidate_ranking"]["zero_residual_equals_physics"] = True
    report = {
        "status": "no_update_audit_complete",
        "protocol": "world-gppo-9.11-energy-residual/0.1.0",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "selection_note": "train/validation only; no test/OOD read by this tool",
        "splits": splits,
        "decision": "do_not_start_fusion_from_existing_absolute_model; residual training remains a separately budgeted future experiment",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
