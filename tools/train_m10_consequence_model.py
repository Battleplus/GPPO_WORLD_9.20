"""Bounded server-side training entry for the 9.11 consequence model.

This command is intentionally not invoked by local verification.  It refuses
to reuse an existing run directory, records all state needed for recovery, and
keeps test/OOD out of model selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import audit_consequence_manifest, load_consequence_jsonl
from gppo_world.consequence_model import ActionConsequenceWorldModel, ConsequenceModelConfig, consequence_loss
from gppo_world.dataset import sha256_file
from gppo_world.model import GraphWorldModel


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: ActionConsequenceWorldModel, examples, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    with torch.no_grad():
        for example in examples:
            prediction = model.predict_candidates(example.graph, [example.target.action], history=example.history)
            targets = {
                "travel_time": torch.tensor([example.target.travel_time], device=device),
                "service_progress": torch.tensor([example.target.service_progress], device=device),
                "energy_delta": torch.tensor([example.target.energy_delta], device=device),
                "deadline_risk": torch.tensor([example.target.deadline_risk], device=device),
            }
            losses = consequence_loss(prediction, targets)
            for name, value in losses.items():
                totals.setdefault(name, []).append(float(value))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-world-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-dim", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.threads < 1 or args.history_dim < 0:
        parser.error("epochs, patience, and threads must be positive; history-dim cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; do not silently switch hardware")
    if args.out.exists():
        parser.error(f"refusing to reuse existing output directory: {args.out}")
    output = args.out.resolve()
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    started = datetime.now(timezone.utc).isoformat()
    write_json(output / "run-status.json", {"run_id": args.run_id, "status": "running", "started_at": started})
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        audit = audit_consequence_manifest(manifest, args.data)
        write_json(output / "input-audit.json", audit)
        if not audit["passed"]:
            raise ValueError("consequence manifest audit failed")
        expected_horizon = int(protocol["prediction_horizon_steps"])
        train = load_consequence_jsonl(args.data / manifest["files"]["train"]["path"], expected_horizon_steps=expected_horizon)
        validation = load_consequence_jsonl(args.data / manifest["files"]["validation"]["path"], expected_horizon_steps=expected_horizon)
        base, base_extra = GraphWorldModel.load(args.base_world_model, map_location=device)
        base.to(device)
        model_config = ConsequenceModelConfig(horizon_steps=expected_horizon, history_dim=args.history_dim)
        model = ActionConsequenceWorldModel(base, model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        seed_everything(args.seed)
        history: list[dict[str, float | int]] = []
        best_state = None
        best_score = float("inf")
        stale = 0
        rng = random.Random(args.seed)
        for epoch in range(args.epochs):
            model.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            train_losses = []
            for index in order:
                example = train[index]
                optimizer.zero_grad(set_to_none=True)
                prediction = model.predict_candidates(example.graph, [example.target.action], history=example.history)
                targets = {
                    "travel_time": torch.tensor([example.target.travel_time], device=device),
                    "service_progress": torch.tensor([example.target.service_progress], device=device),
                    "energy_delta": torch.tensor([example.target.energy_delta], device=device),
                    "deadline_risk": torch.tensor([example.target.deadline_risk], device=device),
                }
                losses = consequence_loss(prediction, targets)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(losses["total"].detach().cpu()))
            validation_metrics = evaluate(model, validation, device)
            score = validation_metrics["total"]
            history.append({"epoch": epoch + 1, "train_total": float(np.mean(train_losses)), "validation_total": score})
            if score < best_score - 1e-8:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError("no finite validation checkpoint was produced")
        model.load_state_dict(best_state)
        checkpoint = {
            "format": ActionConsequenceWorldModel.__name__ + "/v1",
            "protocol": protocol["protocol"],
            "run_id": args.run_id,
            "model_config": asdict(model_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recovery_state": {"epoch": len(history), "best_validation_total": best_score, "next_epoch": len(history) + 1},
            "base_world_model_sha256": sha256_file(args.base_world_model),
            "base_world_model_extra": base_extra,
        }
        checkpoint_path = output / "checkpoints" / "action-consequence.pt"
        checkpoint_path.parent.mkdir(parents=True)
        torch.save(checkpoint, checkpoint_path)
        write_json(output / "training-history.json", history)
        result = {
            "status": "complete",
            "run_id": args.run_id,
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "device": str(device), "threads": args.threads},
            "protocol": protocol,
            "input_audit": audit,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "epochs_completed": len(history),
            "best_validation_total": best_score,
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        }
        write_json(output / "metrics.json", result)
        write_json(output / "run-status.json", {"run_id": args.run_id, "status": "complete", "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json(output / "run-complete.json", result)
        return 0
    except Exception as exc:
        write_json(output / "run-status.json", {"run_id": args.run_id, "status": "failed", "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
