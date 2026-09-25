"""Bounded trainer for the M-10 joint task-set consequence pilot."""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.joint_consequence_data import JointConsequenceExample, audit_joint_manifest, file_sha256, load_joint_jsonl  # noqa: E402
from gppo_world.joint_consequence_model import Graph5JointConsequenceModel, JointConsequenceModelConfig, joint_loss  # noqa: E402


class BoundedStop(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def finite_module(model: torch.nn.Module, label: str) -> None:
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all(): raise FloatingPointError(f"non-finite {label} parameter {name}")


def example_prediction(model: Graph5JointConsequenceModel, item: JointConsequenceExample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, prediction = model.predict_candidates(item.graph.to(device), item.task_set_size, [item.action])
    target = torch.tensor([[float(item.new_on_time_count or 0.0), float(item.deadline_failure_count or 0.0)]], device=device)
    mask = torch.tensor([item.mask], device=device)
    return prediction, target, mask


def evaluate(model: Graph5JointConsequenceModel, examples: list[JointConsequenceExample], device: torch.device) -> dict[str, float]:
    model.eval(); values=[]; ontime=[]; failure=[]
    with torch.no_grad():
        for item in examples:
            if not item.mask: continue
            prediction, target, _ = example_prediction(model, item, device)
            values.append(float((prediction-target).square().mean().cpu()))
            ontime.append(float((prediction[0,0]-target[0,0]).square().cpu()))
            failure.append(float((prediction[0,1]-target[0,1]).square().cpu()))
    if not values: raise FloatingPointError("validation has no valid joint labels")
    return {"total_mse": float(np.mean(values)), "new_on_time_mse": float(np.mean(ontime)), "deadline_failure_mse": float(np.mean(failure)), "valid_records": len(values)}


def checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, identity: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    finite_module(model, "checkpoint")
    return {"format": "gppo-m10-joint-consequence-training-v1", "run_identity": identity, "model_config": {"hidden_dim": 64}, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "recovery_state": recovery}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu"); parser.add_argument("--seed", type=int, default=1101); parser.add_argument("--threads", type=int, default=4); parser.add_argument("--epochs", type=int, default=4); parser.add_argument("--patience", type=int, default=2); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--max-updates", type=int, default=512); parser.add_argument("--max-wall-seconds", type=float, default=1200.0); parser.add_argument("--stop-after-updates", type=int); parser.add_argument("--resume", action="store_true")
    args=parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable; no silent fallback")
    if min(args.epochs,args.patience,args.threads,args.batch_size,args.max_updates)<1 or args.max_wall_seconds<=0: raise SystemExit("training bounds must be positive")
    output=args.out.resolve()
    if output.exists() and not args.resume: raise SystemExit(f"refusing existing output without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True); checkpoints=output/"checkpoints"; checkpoints.mkdir(exist_ok=True)
    protocol=json.loads(args.protocol.read_text(encoding="utf-8")); manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    if protocol.get("protocol") != manifest.get("protocol"): raise SystemExit("protocol and manifest mismatch")
    audit=audit_joint_manifest(manifest,args.data); write_json(output/"input-audit.json",audit)
    if not audit["passed"]: raise SystemExit("joint manifest audit failed")
    train=load_joint_jsonl(args.data/manifest["files"]["train"]["path"],expected_horizon_steps=6); validation=load_joint_jsonl(args.data/manifest["files"]["validation"]["path"],expected_horizon_steps=6)
    identity={"run_id":args.run_id,"protocol_sha256":file_sha256(args.protocol),"manifest_sha256":file_sha256(args.manifest),"data_sha256":{s:file_sha256(args.data/manifest["files"][s]["path"]) for s in ("train","validation")},"source_sha256":{"entry":file_sha256(Path(__file__)),"model":file_sha256(ROOT/"gppo_world"/"joint_consequence_model.py"),"data":file_sha256(ROOT/"gppo_world"/"joint_consequence_data.py")},"protocol":protocol["protocol"],"seed":args.seed,"device":args.device,"threads":args.threads,"epochs":args.epochs,"patience":args.patience,"batch_size":args.batch_size,"max_updates":args.max_updates,"max_wall_seconds":args.max_wall_seconds,"training_scope":"development train/validation only"}
    identity_path=output/"run-identity.json"; status_path=output/"run-status.json"; last_path=checkpoints/"last-recovery.pt"; best_path=checkpoints/"best-inference.pt"
    if args.resume:
        if not identity_path.is_file() or not last_path.is_file(): raise SystemExit("resume requires existing identity and last checkpoint")
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity: raise SystemExit("resume identity mismatch")
    else: write_json(identity_path,identity)
    torch.set_num_threads(args.threads); device=torch.device(args.device); seed_everything(args.seed)
    model=Graph5JointConsequenceModel(JointConsequenceModelConfig(hidden_dim=64)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    epoch=0; next_index=0; order=[]; steps=0; best=float("inf"); stale=0; history=[]; elapsed_before=0.0
    if args.resume:
        payload=torch.load(last_path,map_location=device,weights_only=False); model.load_state_dict(payload["model_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"]); rec=payload["recovery_state"]; epoch=int(rec["epoch"]); next_index=int(rec["next_index"]); order=list(rec["data_order"]); steps=int(rec["optimizer_steps"]); best=float(rec["best_validation_total"]); stale=int(rec["stale"]); history=list(rec["history"]); elapsed_before=float(rec.get("elapsed_seconds",0.0)); restore_rng(rec["rng_state"])
    write_json(output/"runtime.json",{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"cpu":platform.processor(),"host":socket.gethostname(),"device":str(device),"threads":args.threads,"started_at":datetime.now(timezone.utc).isoformat(),"cuda":{"available":torch.cuda.is_available(),"name":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"version":torch.version.cuda if device.type=="cuda" else None}})
    write_json(status_path,{"run_id":args.run_id,"status":"running","pid":os.getpid(),"host":socket.gethostname()}); start=time.monotonic(); stop_reason=None
    try:
        while epoch<args.epochs:
            if not order: order=list(range(len(train))); random.shuffle(order); next_index=0
            model.train()
            while next_index<len(order):
                elapsed=elapsed_before+time.monotonic()-start
                if steps>=args.max_updates: raise BoundedStop("max_optimizer_updates")
                if elapsed>=args.max_wall_seconds: raise BoundedStop("max_wall_seconds")
                if args.stop_after_updates is not None and steps>=args.stop_after_updates: raise BoundedStop("planned_interruption")
                indices=order[next_index:next_index+args.batch_size]; predictions=[]; targets=[]; masks=[]; optimizer.zero_grad(set_to_none=True)
                for index in indices:
                    prediction,target,mask=example_prediction(model,train[index],device); predictions.append(prediction); targets.append(target); masks.append(mask)
                loss=joint_loss(torch.cat(predictions),torch.cat(targets),torch.cat(masks)); loss.backward(); grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0))
                if not np.isfinite(grad_norm): raise FloatingPointError("non-finite gradient norm")
                optimizer.step(); finite_module(model,"after optimizer step")
                if any(not torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)): raise FloatingPointError("non-finite optimizer state")
                steps+=1; next_index+=len(indices); item={"kind":"update","epoch":epoch,"optimizer_step":steps,"batch_size":len(indices),"loss":float(loss.detach().cpu()),"grad_norm":grad_norm,"elapsed_seconds":elapsed}; history.append(item)
                if steps%25==0:
                    torch.save(checkpoint(model,optimizer,identity,{"epoch":epoch,"next_index":next_index,"data_order":order,"optimizer_steps":steps,"best_validation_total":best,"stale":stale,"history":history,"elapsed_seconds":elapsed,"stop_reason":None,"rng_state":rng_state()}),last_path)
                    with (output/"updates.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(item,sort_keys=True)+"\n")
            val=evaluate(model,validation,device); history.append({"kind":"validation","epoch":epoch,"optimizer_step":steps,**val})
            if val["total_mse"]<best: best=val["total_mse"]; stale=0; torch.save({"format":"gppo-m10-joint-consequence-inference-v1","run_identity":identity,"model_config":{"hidden_dim":64},"epoch":epoch,"validation":val,"model_state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},best_path)
            else: stale+=1
            epoch+=1; order=[]; next_index=0; elapsed=elapsed_before+time.monotonic()-start; torch.save(checkpoint(model,optimizer,identity,{"epoch":epoch,"next_index":next_index,"data_order":order,"optimizer_steps":steps,"best_validation_total":best,"stale":stale,"history":history,"elapsed_seconds":elapsed,"stop_reason":None,"rng_state":rng_state()}),last_path)
            if stale>=args.patience: raise BoundedStop("validation_patience")
    except BoundedStop as exc:
        stop_reason=str(exc); elapsed=elapsed_before+time.monotonic()-start; torch.save(checkpoint(model,optimizer,identity,{"epoch":epoch,"next_index":next_index,"data_order":order,"optimizer_steps":steps,"best_validation_total":best,"stale":stale,"history":history,"elapsed_seconds":elapsed,"stop_reason":stop_reason,"rng_state":rng_state()}),last_path)
    except Exception as exc:
        stop_reason=f"failed:{type(exc).__name__}:{exc}"; elapsed=elapsed_before+time.monotonic()-start; torch.save(checkpoint(model,optimizer,identity,{"epoch":epoch,"next_index":next_index,"data_order":order,"optimizer_steps":steps,"best_validation_total":best,"stale":stale,"history":history,"elapsed_seconds":elapsed,"stop_reason":stop_reason,"rng_state":rng_state()}),last_path); write_json(status_path,{"run_id":args.run_id,"status":"failed","stop_reason":stop_reason,"optimizer_steps":steps}); raise
    elapsed=elapsed_before+time.monotonic()-start; final_status="complete" if epoch>=args.epochs else "stopped"; write_json(status_path,{"run_id":args.run_id,"status":final_status,"stop_reason":stop_reason or "completed_epochs","optimizer_steps":steps,"epoch":epoch,"elapsed_seconds":elapsed,"best_validation_total":best}); write_json(output/"training-summary.json",{"run_id":args.run_id,"status":final_status,"stop_reason":stop_reason or "completed_epochs","optimizer_steps":steps,"epochs_completed":epoch,"elapsed_seconds":elapsed,"best_validation_total":best,"train_records":len(train),"validation_records":len(validation),"checkpoints":{"best":str(best_path),"last_recovery":str(last_path)}}); print(json.dumps({"run_id":args.run_id,"status":final_status,"optimizer_steps":steps,"elapsed_seconds":elapsed,"stop_reason":stop_reason or "completed_epochs"},sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
