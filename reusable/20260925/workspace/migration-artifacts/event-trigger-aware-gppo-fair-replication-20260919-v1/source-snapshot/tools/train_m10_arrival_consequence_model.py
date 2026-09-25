"""Bounded local training for the arrival-to-region consequence model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.arrival_consequence_data import ArrivalExample, audit_arrival_manifest, file_sha256, load_arrival_jsonl  # noqa: E402
from gppo_world.arrival_consequence_model import ArrivalModelConfig, Graph5ArrivalConsequenceModel, arrival_loss  # noqa: E402


class BoundedStop(RuntimeError):
    pass


def json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def finite_module(model: torch.nn.Module, label: str) -> None:
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {label} parameter {name}")


def batch_loss(model: Graph5ArrivalConsequenceModel, examples: list[ArrivalExample], device: torch.device) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = {"arrival_time_nll": [], "deadline_bce": [], "failure_bce": []}
    for item in examples:
        graph = item.graph.to(device)
        prediction = model.predict_candidates(graph, [item.action])
        targets = {"arrival_time": torch.tensor([float(item.target["arrival_time"] or 0.0)], device=device), "deadline": torch.tensor([float(bool(item.target["arrival_before_deadline_physical"]))], device=device), "failure": torch.tensor([float(bool(item.target["execution_or_energy_failure"]))], device=device)}
        masks = {"arrival_time": torch.tensor([item.masks["arrival_time"]], device=device), "deadline": torch.tensor([item.masks["deadline"]], device=device), "failure": torch.tensor([item.masks["failure"]], device=device)}
        try:
            losses = arrival_loss(prediction, targets, masks)
        except ValueError as exc:
            if str(exc) != "all arrival consequence heads are masked":
                raise
            # NOOP and fully censored candidates remain in the dataset and
            # audit, but correctly contribute no gradient for this head.
            continue
        for name, value in losses.items():
            if name != "total":
                values[name].append(value)
    present = {name: torch.stack(items).mean() for name, items in values.items() if items}
    if not present:
        raise ValueError("batch has no valid labels")
    present["total"] = torch.stack(tuple(present.values())).mean()
    return present


@torch.no_grad()
def evaluate(model: Graph5ArrivalConsequenceModel, examples: list[ArrivalExample], device: torch.device) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = {"arrival_time_nll": [], "deadline_bce": [], "failure_bce": []}
    for item in examples:
        try:
            losses = batch_loss(model, [item], device)
        except ValueError as exc:
            if str(exc) != "batch has no valid labels":
                raise
            continue
        for name, value in losses.items():
            if name != "total":
                values[name].append(float(value.detach().cpu()))
    out = {name: (float(np.mean(items)) if items else float("nan")) for name, items in values.items()}
    valid = [v for v in out.values() if np.isfinite(v)]
    out["total"] = float(np.mean(valid)) if valid else float("nan")
    if not np.isfinite(out["total"]):
        raise FloatingPointError("validation loss is non-finite or has no valid head")
    return out


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, identity: dict[str, Any], epoch: int, next_index: int, order: list[int], steps: int, best: float, stale: int, history: list[dict[str, Any]], elapsed: float, stop_reason: str | None) -> dict[str, Any]:
    finite_module(model, "checkpoint")
    return {"format": "gppo-arrival-consequence-training-v1", "run_identity": identity, "model_config": {"hidden_dim": 64, "horizon_steps": identity["horizon_steps"]}, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "recovery_state": {"epoch": epoch, "next_index": next_index, "data_order": order, "optimizer_steps": steps, "best_validation_total": best, "stale": stale, "history": history, "elapsed_seconds": elapsed, "stop_reason": stop_reason, "rng_state": rng_state()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-updates", type=int, default=4096)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent fallback")
    if min(args.epochs, args.patience, args.batch_size, args.threads, args.max_updates) < 1 or args.max_wall_seconds <= 0:
        raise SystemExit("training bounds must be positive")
    output = args.out.resolve()
    if output.exists() and not args.resume:
        raise SystemExit(f"refusing existing output without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    horizon = int(protocol["prediction_horizon_steps"] if "prediction_horizon_steps" in protocol else manifest["prediction_horizon_steps"])
    if protocol.get("protocol") != manifest.get("protocol") or protocol.get("observation_contract") != manifest.get("observation_contract"):
        raise SystemExit("protocol and dataset contract mismatch")
    if protocol.get("task_semantics") != "arrival_to_region":
        raise SystemExit("arrival protocol required")
    manifest_audit = audit_arrival_manifest(manifest, args.data)
    json_write(output / "input-audit.json", manifest_audit)
    if not manifest_audit["passed"]:
        raise SystemExit("dataset manifest audit failed")
    train = load_arrival_jsonl(args.data / manifest["files"]["train"]["path"], horizon)
    validation = load_arrival_jsonl(args.data / manifest["files"]["validation"]["path"], horizon)
    identity = {"run_id": args.run_id, "protocol_sha256": file_sha256(args.protocol), "manifest_sha256": file_sha256(args.manifest), "data_sha256": {split: file_sha256(args.data / manifest["files"][split]["path"]) for split in ("train", "validation")}, "source_sha256": {"entry": file_sha256(Path(__file__)), "model": file_sha256(ROOT / "gppo_world" / "arrival_consequence_model.py"), "data_loader": file_sha256(ROOT / "gppo_world" / "arrival_consequence_data.py")}, "protocol": protocol["protocol"], "observation_contract": protocol["observation_contract"], "horizon_steps": horizon, "seed": args.seed, "device": args.device, "threads": args.threads, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "max_updates": args.max_updates, "max_wall_seconds": args.max_wall_seconds, "deterministic_algorithms": False, "training_scope": "formal arrival data; train/validation only"}
    identity_path = output / "run-identity.json"
    recovery_path = checkpoints / "last-recovery.pt"
    best_path = checkpoints / "best-inference.pt"
    status_path = output / "run-status.json"
    if args.resume:
        if not identity_path.is_file() or not recovery_path.is_file():
            raise SystemExit("--resume requires existing run identity and recovery checkpoint")
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemExit("resume identity mismatch")
        old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if old.get("status") == "complete":
            raise SystemExit("run is complete; do not resume")
        if old.get("status") == "running" and old.get("host") == socket.gethostname() and int(old.get("pid", 0)) != os.getpid():
            raise SystemExit("run status is still active; do not duplicate")
    else:
        json_write(identity_path, identity)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    seed_everything(args.seed)
    model = Graph5ArrivalConsequenceModel(ArrivalModelConfig(hidden_dim=64, horizon_steps=horizon)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epoch, next_index, order, steps, best, stale, history, elapsed_before = 0, 0, [], 0, float("inf"), 0, [], 0.0
    if args.resume:
        payload = torch.load(recovery_path, map_location=device, weights_only=False)
        if payload.get("run_identity") != identity:
            raise SystemExit("recovery identity mismatch")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        rec = payload["recovery_state"]
        epoch, next_index, order, steps, best, stale, history, elapsed_before = int(rec["epoch"]), int(rec["next_index"]), list(rec["data_order"]), int(rec["optimizer_steps"]), float(rec["best_validation_total"]), int(rec["stale"]), list(rec["history"]), float(rec.get("elapsed_seconds", 0.0))
        restore_rng(rec["rng_state"])
    json_write(output / "runtime.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "cpu": platform.processor(), "host": socket.gethostname(), "device": str(device), "threads": args.threads, "started_at": datetime.now(timezone.utc).isoformat(), "cuda": {"available": torch.cuda.is_available(), "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "version": torch.version.cuda if device.type == "cuda" else None}})
    json_write(status_path, {"run_id": args.run_id, "status": "running", "pid": os.getpid(), "host": socket.gethostname()})
    start = time.monotonic()
    stop_reason: str | None = None
    try:
        while epoch < args.epochs:
            if not order:
                order = list(range(len(train)))
                random.shuffle(order)
                next_index = 0
            model.train()
            while next_index < len(order):
                elapsed = elapsed_before + time.monotonic() - start
                if steps >= args.max_updates:
                    raise BoundedStop("max_optimizer_updates")
                if elapsed >= args.max_wall_seconds:
                    raise BoundedStop("max_wall_seconds")
                if args.stop_after_updates is not None and steps >= args.stop_after_updates:
                    raise BoundedStop("planned_interruption")
                batch_indices = order[next_index:next_index + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                try:
                    losses = batch_loss(model, [train[index] for index in batch_indices], device)
                except ValueError as exc:
                    if str(exc) != "batch has no valid labels":
                        raise
                    history.append({"kind": "skipped_batch", "epoch": epoch, "batch_size": len(batch_indices), "reason": "all_labels_masked", "elapsed_seconds": elapsed})
                    next_index += len(batch_indices)
                    continue
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("non-finite loss")
                losses["total"].backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
                if not np.isfinite(grad_norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                if any(not torch.isfinite(value).all() for value in model.parameters()):
                    raise FloatingPointError("non-finite parameter after optimizer step")
                if any(not torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)):
                    raise FloatingPointError("non-finite optimizer state")
                steps += 1
                next_index += len(batch_indices)
                history.append({"kind": "update", "epoch": epoch, "optimizer_step": steps, "batch_size": len(batch_indices), "loss": float(losses["total"].detach().cpu()), "grad_norm": grad_norm, "elapsed_seconds": elapsed})
                if steps % 25 == 0:
                    torch.save(checkpoint_payload(model, optimizer, identity, epoch, next_index, order, steps, best, stale, history, elapsed, None), recovery_path)
                    with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(history[-1], sort_keys=True) + "\n")
            val = evaluate(model, validation, device)
            history.append({"kind": "validation", "epoch": epoch, "optimizer_step": steps, **val})
            if val["total"] < best:
                best, stale = val["total"], 0
                torch.save({"format": "gppo-arrival-consequence-inference-v1", "run_identity": identity, "model_config": {"hidden_dim": 64, "horizon_steps": horizon}, "epoch": epoch, "validation": val, "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, best_path)
            else:
                stale += 1
            epoch += 1
            order, next_index = [], 0
            elapsed = elapsed_before + time.monotonic() - start
            torch.save(checkpoint_payload(model, optimizer, identity, epoch, next_index, order, steps, best, stale, history, elapsed, None), recovery_path)
            if stale >= args.patience:
                raise BoundedStop("validation_patience")
    except BoundedStop as exc:
        stop_reason = str(exc)
        elapsed = elapsed_before + time.monotonic() - start
        torch.save(checkpoint_payload(model, optimizer, identity, epoch, next_index, order, steps, best, stale, history, elapsed, stop_reason), recovery_path)
    except Exception as exc:
        stop_reason = f"failed:{type(exc).__name__}:{exc}"
        elapsed = elapsed_before + time.monotonic() - start
        torch.save(checkpoint_payload(model, optimizer, identity, epoch, next_index, order, steps, best, stale, history, elapsed, stop_reason), recovery_path)
        json_write(status_path, {"run_id": args.run_id, "status": "failed", "stop_reason": stop_reason, "optimizer_steps": steps})
        raise
    elapsed = elapsed_before + time.monotonic() - start
    status = "complete" if epoch >= args.epochs else "stopped"
    json_write(status_path, {"run_id": args.run_id, "status": status, "stop_reason": stop_reason or "completed_epochs", "optimizer_steps": steps, "epoch": epoch, "elapsed_seconds": elapsed, "best_validation_total": best})
    json_write(output / "training-summary.json", {"run_id": args.run_id, "status": status, "stop_reason": stop_reason or "completed_epochs", "optimizer_steps": steps, "epochs_completed": epoch, "elapsed_seconds": elapsed, "best_validation_total": best, "train_records": len(train), "validation_records": len(validation), "checkpoints": {"best": str(best_path), "last_recovery": str(recovery_path)}})
    print(json.dumps({"run_id": args.run_id, "status": status, "optimizer_steps": steps, "elapsed_seconds": elapsed, "stop_reason": stop_reason or "completed_epochs"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
