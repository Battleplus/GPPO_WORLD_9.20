"""Verify deterministic continuous-versus-resumed consequence-model pilots."""

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
from gppo_world.consequence_model import (  # noqa: E402
    ConsequenceModelConfig,
    Graph5ActionConsequenceWorldModel,
    graph_to_device,
)
from gppo_world.dataset import sha256_file  # noqa: E402


TRACE_KEYS = (
    "optimizer_step",
    "epoch",
    "data_index",
    "parent_episode_id",
    "prefix_id",
    "action",
    "total_loss",
    "gradient_norm",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compare_tree(left: Any, right: Any, path: str = "root") -> dict[str, Any]:
    result = {"equal": True, "max_abs_diff": 0.0, "first_difference": None}

    def visit(a: Any, b: Any, here: str) -> None:
        if not result["equal"] and result["first_difference"] is not None:
            return
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                result.update(equal=False, first_difference=f"{here}: tensor metadata")
                return
            if a.numel() and (a.is_floating_point() or a.is_complex()):
                difference = float((a.detach().cpu() - b.detach().cpu()).abs().max())
                result["max_abs_diff"] = max(result["max_abs_diff"], difference)
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                result.update(equal=False, first_difference=f"{here}: tensor values")
            return
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype:
                result.update(equal=False, first_difference=f"{here}: array metadata")
                return
            if a.size and np.issubdtype(a.dtype, np.number):
                difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                result["max_abs_diff"] = max(result["max_abs_diff"], difference)
            if not np.array_equal(a, b):
                result.update(equal=False, first_difference=f"{here}: array values")
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                result.update(equal=False, first_difference=f"{here}: mapping keys")
                return
            for key in sorted(a, key=str):
                visit(a[key], b[key], f"{here}.{key}")
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                result.update(equal=False, first_difference=f"{here}: sequence length")
                return
            for index, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, f"{here}[{index}]")
            return
        if a != b:
            result.update(equal=False, first_difference=f"{here}: {a!r} != {b!r}")

    visit(left, right, path)
    return result


def load_prediction(checkpoint: Path, example: Any) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Graph5ActionConsequenceWorldModel(ConsequenceModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model.predict_candidates(
            graph_to_device(example.graph, "cpu"),
            [example.target.action],
            history=example.history,
        )
    return {key: value.detach().cpu() for key, value in prediction.as_dict().items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = load_consequence_jsonl(
        args.data / manifest["files"]["validation"]["path"],
        expected_horizon_steps=int(manifest["prediction_horizon_steps"]),
        strict_identity=True,
    )
    traces = [read_jsonl(directory / "updates.jsonl") for directory in (args.continuous, args.resumed)]
    trace_projection = [[{key: row[key] for key in TRACE_KEYS} for row in trace] for trace in traces]
    continuous_recovery = torch.load(args.continuous / "checkpoints" / "last-recovery.pt", map_location="cpu", weights_only=False)
    resumed_recovery = torch.load(args.resumed / "checkpoints" / "last-recovery.pt", map_location="cpu", weights_only=False)
    continuous_best = args.continuous / "checkpoints" / "best-inference.pt"
    resumed_best = args.resumed / "checkpoints" / "best-inference.pt"
    continuous_prediction_1 = load_prediction(continuous_best, examples[0])
    continuous_prediction_2 = load_prediction(continuous_best, examples[0])
    resumed_prediction = load_prediction(resumed_best, examples[0])

    report = {
        "status": "passed",
        "trace": {
            "continuous_updates": len(traces[0]),
            "resumed_updates": len(traces[1]),
            "sample_order_and_numerics": compare_tree(trace_projection[0], trace_projection[1], "trace"),
        },
        "last_recovery": {
            "model": compare_tree(continuous_recovery["model_state_dict"], resumed_recovery["model_state_dict"], "model"),
            "optimizer": compare_tree(continuous_recovery["optimizer_state_dict"], resumed_recovery["optimizer_state_dict"], "optimizer"),
            "rng": compare_tree(continuous_recovery["recovery_state"]["rng_state"], resumed_recovery["recovery_state"]["rng_state"], "rng"),
            "data_order": compare_tree(
                continuous_recovery["recovery_state"]["data_order"],
                resumed_recovery["recovery_state"]["data_order"],
                "data_order",
            ),
            "optimizer_steps": [
                continuous_recovery["recovery_state"]["optimizer_steps"],
                resumed_recovery["recovery_state"]["optimizer_steps"],
            ],
        },
        "best_checkpoint": {
            "continuous_sha256": sha256_file(continuous_best),
            "resumed_sha256": sha256_file(resumed_best),
            "reload_prediction": compare_tree(continuous_prediction_1, continuous_prediction_2, "reload_prediction"),
            "cross_run_prediction": compare_tree(continuous_prediction_1, resumed_prediction, "cross_run_prediction"),
        },
        "notes": [
            "elapsed_seconds and run_id are intentionally excluded from deterministic equality",
            "test uses previously inspected development validation data and is not a blind evaluation",
        ],
    }
    checks = [
        report["trace"]["sample_order_and_numerics"],
        report["last_recovery"]["model"],
        report["last_recovery"]["optimizer"],
        report["last_recovery"]["rng"],
        report["last_recovery"]["data_order"],
        report["best_checkpoint"]["reload_prediction"],
        report["best_checkpoint"]["cross_run_prediction"],
    ]
    if len(traces[0]) != len(traces[1]) or len(set(report["last_recovery"]["optimizer_steps"])) != 1 or not all(item["equal"] for item in checks):
        report["status"] = "failed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
