"""Prove post-action Shadow is read-only against the real GPPO baseline env."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from gppo_world.calibration import ShadowCalibration
from gppo_world.contracts import snapshot_from_gppo
from gppo_world.model import EventAwareGraphWorldModel
from gppo_world.shadow import ShadowRequest, ShadowRuntime


def _canonical_hash(value) -> str:
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        if isinstance(item, set):
            return sorted(item)
        if hasattr(item, "to_dict"):
            return item.to_dict()
        return repr(item)

    payload = json.dumps(value, default=default, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    baseline_dir = args.baseline_dir.resolve()
    ppo_dir = baseline_dir / "ppo_allocation"
    for path in (str(baseline_dir), str(ppo_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from random_event.environment import ActionSubmission, RandomEventAllocationEnv
    from random_event.graph import build_graph_state

    model, _ = EventAwareGraphWorldModel.load(args.checkpoint.resolve())
    calibration = ShadowCalibration.from_dict(
        json.loads(args.calibration.resolve().read_text(encoding="utf-8"))
    )
    runtime = ShadowRuntime(model, calibration, model_version="t03-eawm-v0.1.0/eawm_hard_seed20260903")
    env = RandomEventAllocationEnv(
        initial_seed=993702343,
        event_seed=20260904,
        mode="sequential",
        events_per_episode=5,
        max_decisions=max(20, args.steps + 2),
    )
    env.reset(seed=993702343)
    records = []
    accepted = 0
    episode_index = 0
    step = 0
    while accepted < args.steps:
        ctx = env.begin_decision()
        legal = torch.nonzero(ctx.graph.action_mask, as_tuple=False).flatten().tolist()
        action = int(legal[accepted % len(legal)])
        pre_graph = snapshot_from_gppo(ctx.graph)
        decision_time = float(env.current_time)
        post_graph, _, terminated, truncated, info = env.submit_action(
            ActionSubmission.from_decision(action, ctx)
        )
        if info.get("stale_decision", False):
            continue
        executed_action = int(info.get("repaired_action", action))
        request = ShadowRequest(
            episode_id=f"baseline-{episode_index}",
            step=step,
            graph=pre_graph,
            executed_action=executed_action,
            evidence=(),
            action_version=int(ctx.action_version),
            decision_time=decision_time,
            execution_accepted=True,
            expected_post_graph_version=int(env.graph_version),
            expected_post_action_version=int(env.decision_version),
        )
        env_hash_before = _canonical_hash(env.snapshot())
        belief_hash_before = env.runtime_bridge.adapter.get_snapshot_hash()
        mask_before = build_graph_state(env).action_mask.detach().clone()
        versions_before = (int(env.graph_version), int(env.decision_version))
        spy_calls = {"step": 0, "submit_action": 0, "command": 0, "belief_mutation": 0}

        original_step = env.step
        original_submit = env.submit_action
        original_command = env.runtime_bridge.issue_assignment_command
        original_apply = env.runtime_bridge.apply_confirmed_to_env

        def denied(name):
            def fail(*_args, **_kwargs):
                spy_calls[name] += 1
                raise AssertionError(f"Shadow attempted forbidden baseline call: {name}")
            return fail

        env.step = denied("step")
        env.submit_action = denied("submit_action")
        env.runtime_bridge.issue_assignment_command = denied("command")
        env.runtime_bridge.apply_confirmed_to_env = denied("belief_mutation")
        try:
            result = runtime.observe(
                request,
                version_reader=lambda: (int(env.graph_version), int(env.decision_version)),
            )
        finally:
            env.step = original_step
            env.submit_action = original_submit
            env.runtime_bridge.issue_assignment_command = original_command
            env.runtime_bridge.apply_confirmed_to_env = original_apply

        env_hash_after = _canonical_hash(env.snapshot())
        belief_hash_after = env.runtime_bridge.adapter.get_snapshot_hash()
        mask_after = build_graph_state(env).action_mask.detach().clone()
        versions_after = (int(env.graph_version), int(env.decision_version))
        record = {
            "episode": episode_index,
            "step": step,
            "input_versions": [ctx.graph_version, ctx.action_version],
            "post_versions": list(versions_after),
            "shadow_valid": result.valid,
            "fallback_reason": result.fallback_reason,
            "env_snapshot_unchanged": env_hash_before == env_hash_after,
            "belief_unchanged": belief_hash_before == belief_hash_after,
            "action_mask_unchanged": torch.equal(mask_before, mask_after),
            "versions_unchanged": versions_before == versions_after,
            "forbidden_spy_calls": spy_calls,
        }
        records.append(record)
        accepted += 1
        step += 1
        if terminated or truncated:
            episode_index += 1
            step = 0
            runtime.reset()
            env.reset(seed=993702343 + episode_index)

    counters = runtime.counters
    gates = {
        "real_baseline_environment_used": True,
        "at_least_one_valid_shadow_result": any(item["shadow_valid"] for item in records),
        "environment_snapshot_unchanged": all(item["env_snapshot_unchanged"] for item in records),
        "runtime_belief_unchanged": all(item["belief_unchanged"] for item in records),
        "action_mask_unchanged": all(item["action_mask_unchanged"] for item in records),
        "graph_and_action_versions_unchanged": all(item["versions_unchanged"] for item in records),
        "no_forbidden_baseline_calls": all(
            not any(item["forbidden_spy_calls"].values()) for item in records
        ),
        "shadow_write_and_submission_counters_zero": all(
            counters[name] == 0
            for name in (
                "belief_write_count",
                "action_mask_write_count",
                "graph_version_write_count",
                "action_version_write_count",
                "action_submission_count",
            )
        ),
    }
    output = {
        "result": "PASS" if all(gates.values()) else "FAIL",
        "baseline_dir": str(baseline_dir),
        "baseline_commit_expected": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
        "steps": args.steps,
        "shadow_counters": counters,
        "records": records,
        "gates": gates,
    }
    _write(args.output.resolve(), output)
    print(json.dumps({"result": output["result"], "gates": gates}, indent=2))
    return 0 if output["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
