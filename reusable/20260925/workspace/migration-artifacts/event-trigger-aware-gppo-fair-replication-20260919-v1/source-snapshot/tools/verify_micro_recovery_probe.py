"""No-step, no-update same-boundary reload probe for the micro checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gppo_world.joint_gppo import ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy
from gppo_world.m10_environment import M10Config
from run_event_trigger_aware_gppo import probe


def state_delta(a, b) -> float:
    if isinstance(a, dict):
        keys = set(a) | set(b)
        return max((state_delta(a[k], b[k]) for k in keys), default=0.0)
    if torch.is_tensor(a) and torch.is_tensor(b):
        return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return float(np.max(np.abs(a - b))) if a.size else 0.0
    return 0.0 if a == b else float("inf")


def deep_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(deep_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if torch.is_tensor(a) and torch.is_tensor(b):
        return bool(torch.equal(a, b))
    return a == b


def load_one(path: Path, config: M10Config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = JointGraphPreferencePolicy(config, history=True).eval()
    world = ActionConditionedTemporalWorldModel().eval()
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    world.load_state_dict(payload["world_state_dict"], strict=True)
    runtime = payload["runtime"]
    preference = runtime["episode_preference"]
    result = probe(policy, world, runtime["observation"], runtime["policy_hidden"], runtime["world_hidden"], preference, torch.device("cpu"))
    action = int(torch.argmax(result["evaluation"]["distribution"].logits, dim=-1).item())
    return payload, action, result["evaluation"]["distribution"].logits.detach().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    config_payload = json.loads((Path(r"E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final\run\training\seed-1101\WD\resolved-config.json")).read_text(encoding="utf-8"))
    names = {field.name for field in M10Config.__dataclass_fields__.values()}
    config = M10Config(**{key: value for key, value in config_payload["environment"].items() if key in names})
    results = {}
    for group in ("P_train", "T_train"):
        path = args.root / group / "last-recovery.pt"
        a, action_a, logits_a = load_one(path, config)
        b, action_b, logits_b = load_one(path, config)
        results[group] = {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "same_boundary_state_max_abs_delta": state_delta(a["policy_state_dict"], b["policy_state_dict"]),
            "optimizer_state_max_abs_delta": state_delta(a["policy_optimizer_state_dict"], b["policy_optimizer_state_dict"]),
            "rng_equal": deep_equal(a["rng"], b["rng"]),
            "runtime_counters_equal": deep_equal(a["counters"], b["counters"]),
            "next_deterministic_action_equal": action_a == action_b,
            "next_logits_max_abs_delta": float((logits_a - logits_b).abs().max().item()),
            "action_a": action_a, "action_b": action_b,
            "environment_steps": 0, "optimizer_updates": 0,
        }
    payload = {"status": "passed", "scope": "same-boundary reload probe; no uninterrupted U path persisted", "results": results}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
