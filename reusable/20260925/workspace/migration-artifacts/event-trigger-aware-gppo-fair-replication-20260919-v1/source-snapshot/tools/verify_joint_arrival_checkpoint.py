"""Verify/load an existing joint-run recovery checkpoint without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy
from gppo_world.joint_training import GROUPS, _make_optimizers, finite_optimizer, finite_parameters, load_joint_checkpoint
from gppo_world.m10_environment import M10Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", choices=("B", "C", "D"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    verification_path = run_dir / "checkpoint-restore-verification.json"
    inference_path = run_dir / "smoke-inference.pt"
    if verification_path.exists() or inference_path.exists():
        raise SystemExit("verification artifacts already exist; refusing to overwrite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    resolved = json.loads((run_dir / "resolved-config.json").read_text(encoding="utf-8"))
    if resolved["seed"] < 0:
        raise SystemExit("invalid run seed")
    cfg = M10Config(**resolved["environment"])
    device = torch.device(args.device)
    spec = GROUPS[args.group]
    torch.manual_seed(int(resolved["seed"]))
    policy = JointGraphPreferencePolicy(cfg, history=True).to(device)
    world = ActionConditionedTemporalWorldModel().to(device) if spec.world_model else None
    train_config = __import__("gppo_world.joint_gppo", fromlist=["JointTrainConfig"]).JointTrainConfig(**resolved["training"])
    policy_optimizer, world_optimizer = _make_optimizers(policy, world, spec, train_config)
    preference_rng = np.random.default_rng(int(resolved["seed"]) + 0xA17E)
    checkpoint = run_dir / "last-recovery.pt"
    payload, runtime = load_joint_checkpoint(
        checkpoint, expected_run_id=args.run_id, expected_group=args.group,
        policy=policy, world=world, policy_optimizer=policy_optimizer,
        world_optimizer=world_optimizer, device=device, preference_rng=preference_rng,
    )
    if payload["counters"]["environment_steps"] != len((run_dir / "training-ledger.jsonl").read_text(encoding="utf-8").splitlines()):
        raise RuntimeError("recovery counter does not match persisted decision ledger")
    if not finite_parameters(policy) or (world is not None and not finite_parameters(world)):
        raise RuntimeError("recovered parameters are not finite")
    if not finite_optimizer(policy_optimizer) or (world_optimizer is not None and not finite_optimizer(world_optimizer)):
        raise RuntimeError("recovered optimizer state is not finite")
    environment = runtime["environment"]
    if environment.public_snapshot_digest() != runtime["public_digest"]:
        raise RuntimeError("restored environment public digest mismatch")
    saved_obs = runtime["observation"]
    digest_from_saved_obs = (int(saved_obs["version"]), tuple(float(x) for x in saved_obs["flat"]), tuple(bool(x) for x in saved_obs["mask"]))
    if digest_from_saved_obs != runtime["public_digest"]:
        raise RuntimeError("saved observation does not match the restored simulator snapshot")
    with (run_dir / "training-ledger.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    illegal = [row["step"] for row in rows if not row["action_legal"] or not row["action_mask"][row["action"]]]
    if illegal:
        raise RuntimeError(f"illegal action in persisted training ledger: {illegal[:5]}")
    if args.group == "D":
        summary = json.loads((run_dir / "smoke-summary.json").read_text(encoding="utf-8"))
        if not summary["event_gradient_updates"] or not any(row.get("event_shared_encoder_grad_norm", 0.0) > 0 for row in summary["world_updates"]):
            raise RuntimeError("D checkpoint lacks evidence of event-loss gradient into shared encoder")
    torch.save({
        "format": "joint-arrival-inference/1.0.0", "protocol": payload["protocol"],
        "run_id": args.run_id, "group": args.group,
        "policy_state_dict": policy.state_dict(),
        "world_state_dict": None if world is None else world.state_dict(),
        "identity": payload["identity"],
    }, inference_path)
    verification = {
        "status": "passed", "run_id": args.run_id, "group": args.group,
        "recovery_sha256": sha256_file(checkpoint),
        "inference_sha256": sha256_file(inference_path),
        "strict_policy_load": True, "strict_world_model_load": world is None or payload["world_state_dict"] is not None,
        "policy_optimizer_restored": True, "world_optimizer_restored": world_optimizer is None or payload["world_optimizer_state_dict"] is not None,
        "rng_states_restored": True, "simulator_and_episode_history_restored": True,
        "public_observation_digest_match": True,
        "environment_steps": payload["counters"]["environment_steps"],
        "policy_optimizer_steps": payload["counters"]["policy_optimizer_steps"],
        "world_optimizer_steps": payload["counters"]["world_optimizer_steps"],
        "ledger_rows": len(rows), "illegal_ledger_actions": len(illegal),
    }
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    files = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact-manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (run_dir / "artifact-manifest.json").write_text(json.dumps({"run_id": args.run_id, "files": files}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
