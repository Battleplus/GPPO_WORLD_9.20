"""Bounded server-side training entry for the 9.11 consequence model.

The entry is fail-closed: it will not silently change devices, reuse a run,
or apply the legacy 3-type/17-action checkpoint to the M-10 Graph-5 contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.consequence_data import audit_consequence_manifest, load_consequence_jsonl  # noqa: E402
from gppo_world.consequence_model import (  # noqa: E402
    ActionConsequenceWorldModel,
    ConsequenceModelConfig,
    Graph5ActionConsequenceWorldModel,
    consequence_loss,
    graph_to_device,
)
from gppo_world.dataset import sha256_file  # noqa: E402
from gppo_world.model import GraphWorldModel  # noqa: E402


class TrainingStop(RuntimeError):
    """A bounded stop which is recorded separately from a failed run."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    """Seed every available RNG before constructing the trainable model."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("recovery contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def assert_finite(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"non-finite {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite(child, f"{label}[{index}]")


def assert_finite_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for name, parameter in model.named_parameters():
        assert_finite(parameter, f"parameter:{name}")
        if parameter.grad is not None:
            assert_finite(parameter.grad, f"gradient:{name}")
    assert_finite(optimizer.state, "optimizer_state")


def targets_for(example: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([getattr(example.target, name)], dtype=torch.float32, device=device)
        for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")
    }


def masks_for(example: Any, device: torch.device) -> dict[str, torch.Tensor]:
    values = example.masks or {name: True for name in ("travel_time", "service_progress", "energy_delta", "deadline_risk")}
    return {name: torch.tensor([bool(values[name])], dtype=torch.bool, device=device) for name in values}


def evaluate(model: ActionConsequenceWorldModel, examples: list[Any], device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    with torch.no_grad():
        for example in examples:
            graph = graph_to_device(example.graph, device)
            history = example.history.to(device) if example.history is not None else None
            prediction = model.predict_candidates(graph, [example.target.action], history=history)
            losses = consequence_loss(prediction, targets_for(example, device), masks=masks_for(example, device))
            assert_finite(losses, "validation_loss")
            for name, value in losses.items():
                totals.setdefault(name, []).append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def recovery_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, identity: dict[str, Any], *, epoch: int, next_index: int, order: list[int], steps: int, best_score: float, stale: int, history: list[dict[str, Any]], stop_reason: str | None) -> dict[str, Any]:
    payload = {
        "format": "gppo-action-consequence-recovery/v2",
        "run_identity": identity,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recovery_state": {
            "epoch": epoch,
            "next_index": next_index,
            "data_order": list(order),
            "optimizer_steps": steps,
            "best_validation_total": best_score,
            "stale": stale,
            "history": history,
            "stop_reason": stop_reason,
            "rng_state": capture_rng_state(),
        },
    }
    assert_finite(payload["model_state_dict"], "checkpoint_parameters")
    assert_finite(payload["optimizer_state_dict"], "checkpoint_optimizer_state")
    return payload


def save_recovery(path: Path, *args: Any, **kwargs: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(recovery_payload(*args, **kwargs), temporary)
    temporary.replace(path)


def save_best(path: Path, model: torch.nn.Module, identity: dict[str, Any], epoch: int, score: float) -> None:
    payload = {
        "format": "gppo-action-consequence-inference/v2",
        "run_identity": identity,
        "epoch": epoch,
        "validation_total": score,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    assert_finite(payload["model_state_dict"], "best_inference_parameters")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def runtime_record(device: torch.device, threads: int, started_at: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "host": socket.gethostname(),
        "device_requested": str(device),
        "threads": threads,
        "started_at": started_at,
    }
    if device.type == "cuda":
        result.update({"cuda_version": torch.version.cuda, "cuda_device_count": torch.cuda.device_count(), "cuda_device_name": torch.cuda.get_device_name(device)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-world-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-dim", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1 or args.history_dim < 0:
        parser.error("epochs, patience, threads must be positive; history-dim cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; do not silently switch hardware")
    output = args.out.resolve()
    if output.exists() and not args.resume:
        parser.error(f"refusing to reuse existing output directory without --resume: {output}")
    if not output.exists():
        if args.resume:
            parser.error("--resume requires an existing run directory")
        output.mkdir(parents=True)
    for path in (args.protocol, args.manifest):
        if not path.is_file():
            parser.error(f"missing input file: {path}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    observation_contract = str(protocol.get("observation_contract", ""))
    if observation_contract not in {"gppo-graph-3type-17action", "m10-graph5-5type-25action"}:
        raise RuntimeError(
            "target observation contract is not implemented by this entry: "
            f"{observation_contract!r}"
        )
    if manifest.get("protocol") != protocol.get("protocol"):
        raise RuntimeError("dataset and protocol versions differ; regenerate the dataset with the exact protocol")
    if observation_contract == "gppo-graph-3type-17action" and not args.base_world_model:
        parser.error("--base-world-model is required for the legacy Graph-3/17-action contract")
    if args.base_world_model and not args.base_world_model.is_file():
        parser.error(f"missing base world model: {args.base_world_model}")
    max_updates = args.max_updates if args.max_updates is not None else int(protocol.get("world_model_max_optimizer_updates", 10000))
    max_wall = args.max_wall_seconds if args.max_wall_seconds is not None else float(protocol.get("world_model_max_wall_seconds", 7200.0))
    if max_updates < 1 or max_wall <= 0:
        parser.error("max-updates and max-wall-seconds must be positive")
    identity = {
        "run_id": args.run_id,
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest),
        "base_world_model_sha256": sha256_file(args.base_world_model) if args.base_world_model else None,
        "observation_contract": observation_contract,
        "seed": args.seed,
        "history_dim": args.history_dim,
    }
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    recovery_path = checkpoints / "last-recovery.pt"
    best_path = checkpoints / "best-inference.pt"
    status_path = output / "run-status.json"
    started = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    seed_everything(args.seed)
    identity_path = output / "run-identity.json"
    if args.resume:
        if not recovery_path.is_file() or not identity_path.is_file():
            raise RuntimeError("--resume requires run-identity.json and last-recovery.pt")
        stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored_identity != identity:
            raise RuntimeError("resume identity mismatch; inputs, seed, and run-id must be identical")
        old_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if old_status.get("status") == "complete":
            raise RuntimeError("run is already complete; do not overwrite it with --resume")
        if old_status.get("status") == "running" and old_status.get("host") == socket.gethostname():
            pid = int(old_status.get("pid", 0))
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise RuntimeError("run status is uncertain; verify the process before --resume") from exc
                else:
                    raise RuntimeError(f"run appears alive with pid {pid}; do not duplicate it")
    else:
        write_json(identity_path, identity)
    write_json(output / "runtime.json", runtime_record(device, args.threads, started))
    write_json(status_path, {"run_id": args.run_id, "status": "running", "pid": os.getpid(), "host": socket.gethostname(), "started_at": started})
    started_clock = time.monotonic()
    try:
        audit = audit_consequence_manifest(manifest, args.data)
        write_json(output / "input-audit.json", audit)
        if not audit["passed"]:
            raise ValueError("consequence manifest audit failed")
        horizon = int(protocol["prediction_horizon_steps"])
        files = manifest["files"]
        train = load_consequence_jsonl(args.data / files["train"]["path"], expected_horizon_steps=horizon, strict_identity=True)
        validation = load_consequence_jsonl(args.data / files["validation"]["path"], expected_horizon_steps=horizon, strict_identity=True)
        model_config = ConsequenceModelConfig(horizon_steps=horizon, history_dim=args.history_dim)
        base_extra: dict[str, Any] = {}
        if observation_contract == "gppo-graph-3type-17action":
            base, base_extra = GraphWorldModel.load(args.base_world_model, map_location=device)
            base.to(device)
            if base.num_actions != 17:
                raise RuntimeError(f"legacy base model action count is {base.num_actions}; expected 17 in compatibility mode")
            model = ActionConsequenceWorldModel(base, model_config).to(device)
        else:
            model = Graph5ActionConsequenceWorldModel(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        history: list[dict[str, Any]] = []
        best_score, stale, epoch, next_index, steps = float("inf"), 0, 0, 0, 0
        order: list[int] = []
        if args.resume:
            payload = torch.load(recovery_path, map_location=device, weights_only=False)
            if payload.get("run_identity") != identity:
                raise RuntimeError("recovery checkpoint identity mismatch")
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            recovery = payload["recovery_state"]
            epoch = int(recovery["epoch"])
            next_index = int(recovery["next_index"])
            order = [int(value) for value in recovery["data_order"]]
            steps = int(recovery["optimizer_steps"])
            best_score = float(recovery["best_validation_total"])
            stale = int(recovery["stale"])
            history = list(recovery["history"])
            restore_rng_state(recovery["rng_state"])
            if order and (sorted(order) != list(range(len(train))) or not 0 <= next_index <= len(order)):
                raise RuntimeError("recovery data order is invalid for the audited train split")
        while epoch < args.epochs:
            if not order:
                order = list(range(len(train)))
                random.shuffle(order)
                next_index = 0
            model.train()
            train_losses: list[float] = []
            while next_index < len(order):
                if steps >= max_updates:
                    raise TrainingStop("max_optimizer_updates")
                if time.monotonic() - started_clock >= max_wall:
                    raise TrainingStop("max_wall_seconds")
                example = train[order[next_index]]
                optimizer.zero_grad(set_to_none=True)
                graph = graph_to_device(example.graph, device)
                visible_history = example.history.to(device) if example.history is not None else None
                prediction = model.predict_candidates(graph, [example.target.action], history=visible_history)
                losses = consequence_loss(prediction, targets_for(example, device), masks=masks_for(example, device))
                assert_finite(losses, "loss")
                losses["total"].backward()
                assert_finite_state(model, optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                assert_finite(norm, "gradient_norm")
                optimizer.step()
                assert_finite_state(model, optimizer)
                train_losses.append(float(losses["total"].detach().cpu()))
                steps += 1
                next_index += 1
                save_recovery(recovery_path, model, optimizer, identity, epoch=epoch, next_index=next_index, order=order, steps=steps, best_score=best_score, stale=stale, history=history, stop_reason=None)
            validation_metrics = evaluate(model, validation, device)
            score = validation_metrics["total"]
            history.append({"epoch": epoch + 1, "train_total": float(np.mean(train_losses)), "validation": validation_metrics, "optimizer_steps": steps, "elapsed_seconds": time.monotonic() - started_clock})
            if score < best_score - 1e-8:
                best_score, stale = score, 0
                save_best(best_path, model, identity, epoch + 1, score)
            else:
                stale += 1
            epoch += 1
            order, next_index = [], 0
            stop_reason = "early_stopping" if stale >= args.patience else None
            save_recovery(recovery_path, model, optimizer, identity, epoch=epoch, next_index=0, order=[], steps=steps, best_score=best_score, stale=stale, history=history, stop_reason=stop_reason)
            write_json(output / "training-history.json", history)
            if stop_reason:
                break
        if not best_path.is_file():
            raise RuntimeError("no finite validation checkpoint was produced")
        stop_reason = "early_stopping" if stale >= args.patience else "epoch_budget"
        result = {"status": "complete", "run_id": args.run_id, "runtime": runtime_record(device, args.threads, started), "protocol": protocol, "input_audit": audit, "train_examples": len(train), "validation_examples": len(validation), "epochs_completed": len(history), "actual_optimizer_steps": steps, "best_validation_total": best_score, "stop_reason": stop_reason, "checkpoints": {"best_inference": {"path": str(best_path), "sha256": sha256_file(best_path)}, "last_recovery": {"path": str(recovery_path), "sha256": sha256_file(recovery_path)}}, "base_world_model_extra": base_extra}
        write_json(output / "metrics.json", result)
        write_json(status_path, {"run_id": args.run_id, "status": "complete", "pid": os.getpid(), "host": socket.gethostname(), "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "stop_reason": stop_reason})
        write_json(output / "run-complete.json", result)
        return 0
    except TrainingStop as exc:
        write_json(status_path, {"run_id": args.run_id, "status": "stopped", "pid": os.getpid(), "host": socket.gethostname(), "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "stop_reason": exc.reason})
        raise
    except Exception as exc:
        write_json(status_path, {"run_id": args.run_id, "status": "failed", "pid": os.getpid(), "host": socket.gethostname(), "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
