"""Verify pilot checkpoint reload and first-decision reproducibility only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.arrival_gppo_fusion import CandidateAwarePolicy, FrozenArrivalConsequenceScorer, choose_candidate_action  # noqa: E402
from gppo_world.m10_environment import M10Config, M10Environment, weak_communication_tape  # noqa: E402
from gppo_world.m10_training import M10ActorCritic, _act  # noqa: E402


def load_policy(path: Path, config: M10Config, device: torch.device, fusion: bool):
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(payload["metadata"])
    base = M10ActorCritic(
        uav_count=config.uav_count, task_capacity=config.task_capacity,
        action_count=config.action_count, encoder=str(metadata.get("encoder", "graph")),
        type_count=int(metadata.get("type_count", 5)), history=(True if fusion else bool(metadata.get("history", False))),
        context_dim=0, region_count=config.region_count, target_count=config.target_count,
        event_capacity=config.event_capacity, relation_width=config.relation_width,
    )
    policy = CandidateAwarePolicy(base) if fusion else base
    policy = policy.to(device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    if payload.get("optimizer_state_dict") is None or "recovery_state" not in payload:
        raise AssertionError(f"checkpoint recovery payload missing: {path}")
    return policy, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arrival-model", type=Path, required=True)
    parser.add_argument("--arrival-model-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    result = json.loads((args.run / "pilot-results.json").read_text(encoding="utf-8"))
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    tapes = list(weak_communication_tape("validation", count=16, base_seed=93001, level="composite"))
    scorer = FrozenArrivalConsequenceScorer.from_checkpoint(args.arrival_model, device=args.device, expected_sha256=args.arrival_model_sha256)
    checks = []
    for name in ("GPPO", "GPPO-History", "GPPO-History-CandidateArrival"):
        path = Path(result["groups"][name]["checkpoint"])
        policy, payload = load_policy(path, config, device, fusion=name.endswith("CandidateArrival"))
        env = M10Environment(config, tapes[0])
        obs = env.reset()
        expected_action = int(result["groups"][name]["evaluation"]["episodes"][0]["actions"][0]["action"])
        with torch.no_grad():
            if name.endswith("CandidateArrival"):
                action, _, features, _ = choose_candidate_action(policy, scorer, obs, device=device, deterministic=True)
                feature_repeat = scorer.score_observation(obs).dense()
                prediction_consistent = bool(torch.equal(features.dense().cpu(), feature_repeat.cpu()))
            else:
                action, _, _, _ = _act(policy, obs["flat"], obs["mask"], None, device, deterministic=True)
                prediction_consistent = True
        checks.append({
            "group": name,
            "checkpoint": str(path),
            "checkpoint_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "first_action_saved": expected_action,
            "first_action_reloaded": int(action),
            "first_action_match": int(action) == expected_action,
            "state_dict_reload": True,
            "optimizer_state_present": bool(payload.get("optimizer_state_dict")),
            "recovery_state_present": "recovery_state" in payload,
            "prediction_consistent": prediction_consistent,
        })
    passed = all(item["first_action_match"] and item["state_dict_reload"] and item["optimizer_state_present"] and item["recovery_state_present"] and item["prediction_consistent"] for item in checks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"status": "passed" if passed else "failed", "checks": checks, "scope": "checkpoint load and one public first-decision replay; no training and no full evaluation"}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed" if passed else "failed", "checks": checks}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
