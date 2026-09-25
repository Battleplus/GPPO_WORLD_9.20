"""Detailed, non-training audit for the weak-communication A/B/C runs.

This evaluator consumes frozen final-test tapes and frozen checkpoints.  It is
kept separate from training so the final tape is never used for model or
threshold selection.  Byte counts are explicitly labelled as canonical audit
serialization estimates; this simulator has no physical wire codec.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
from gppo_world.m10_training import _act, _policy_input_bundle, _trigger_decision
from tools.run_m10_r3 import load_policy, load_world


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def canonical_bytes(record: dict[str, Any]) -> int:
    return len(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def latency_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"samples": int(len(array)),
            "mean_ms": float(array.mean()) if len(array) else None,
            "p95_ms": float(np.percentile(array, 95)) if len(array) else None,
            "p99_ms": float(np.percentile(array, 99)) if len(array) else None,
            "raw_ms": array.tolist()}


def run_one(policy_path: Path, world, *, mode: str, scenarios: list[Any], config: M10Config,
            device: str, threshold: float, max_wait: int) -> dict[str, Any]:
    policy, policy_meta = load_policy(policy_path, config, device)
    device_obj = torch.device(device)
    records = []
    for scenario in scenarios:
        env = M10Environment(config, scenario)
        obs = env.reset()
        hidden = None
        last_action = None
        since = max_wait
        done = False
        steps = actor_calls = continuation = world_calls = 0
        total_reward = 0.0
        reasons: dict[str, int] = {}
        conditions_count: dict[str, int] = {}
        chain_latency: list[float] = []
        decision_trace: list[dict[str, Any]] = []
        info: dict[str, Any] = {"counts": {"completed": 0, "expired": 0, "rejected": 0}, "energy": {}}
        while not done and steps < int(config.horizon / config.decision_interval) + 2:
            start = time.perf_counter()
            if mode == "periodic":
                # A is the no-world baseline.  It must not pay world-model
                # cost or receive a context-sized vector merely because the
                # detailed evaluator also runs B/C.
                is_base = str(policy_meta.get("fusion", "base")) == "base"
                vector, _, _, full_vector = _policy_input_bundle(
                    env, obs, None if is_base else world,
                    fusion="base" if is_base else "world", device=device_obj,
                    trigger_threshold=threshold)
                if not is_base:
                    world_calls += 1
                should, reason, conditions = _trigger_decision(
                    obs, fusion="base", risk_active=False, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=max_wait)
                decision_vector = vector if is_base else full_vector
            elif mode == "rules":
                public = np.concatenate((np.asarray(obs["flat"], dtype=np.float32),
                                         np.zeros(world.context_dim, dtype=np.float32)))
                should, reason, conditions = _trigger_decision(
                    obs, fusion="triggered", risk_active=False, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=max_wait)
                decision_vector = public
                if should:
                    _, _, _, decision_vector = _policy_input_bundle(
                        env, obs, world, fusion="world", device=device_obj, trigger_threshold=threshold)
                    world_calls += 1
            else:
                vector, active, _, full_vector = _policy_input_bundle(
                    env, obs, world, fusion="triggered", device=device_obj, trigger_threshold=threshold)
                world_calls += 1
                should, reason, conditions = _trigger_decision(
                    obs, fusion="triggered", risk_active=active, last_action=last_action,
                    steps_since_replan=since, max_replan_interval=max_wait)
                decision_vector = full_vector if should else vector
            for key, active in conditions.items():
                if active:
                    conditions_count[key] = conditions_count.get(key, 0) + 1
            decision_trace.append({"time": float(obs["time"]), "replan": bool(should), "reason": reason})
            if should:
                action, _, _, hidden = _act(policy, decision_vector, obs["mask"], hidden, device_obj, deterministic=True)
                actor_calls += 1
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                context_dim = int(getattr(world, "context_dim", 0))
                continuation_vector = np.concatenate((np.asarray(obs["flat"], dtype=np.float32),
                                                       np.zeros(context_dim, dtype=np.float32)))
                _, hidden = policy.value_only(torch.tensor(continuation_vector, dtype=torch.float32, device=device_obj)[None, :], hidden)
                if last_action is None:
                    raise RuntimeError("continuation before first action")
                action = int(last_action)
                continuation += 1
            last_action = action
            since = 0 if should else since + 1
            obs, reward, done, info = env.step(action, submit_command=should)
            total_reward += float(reward)
            steps += 1
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            chain_latency.append((time.perf_counter() - start) * 1000.0)

        comm = list(info.get("communication_log", []))
        by_link: dict[str, dict[str, int]] = {}
        bytes_by_link: dict[str, int] = {}
        for item in comm:
            link = str(item.get("link", "unknown"))
            status = str(item.get("status", "unknown"))
            by_link.setdefault(link, {})[status] = by_link.setdefault(link, {}).get(status, 0) + 1
            bytes_by_link[link] = bytes_by_link.get(link, 0) + canonical_bytes(item)
        execution_log = list(getattr(env.execution, "log", []))
        accepted_ids = [str(x.get("command_id")) for x in execution_log if x.get("result") == "accepted"]
        duplicate_accepts = len(accepted_ids) - len(set(accepted_ids))
        unauthorized = sum(1 for x in execution_log if x.get("result") in ("stale", "fenced", "expired", "unknown_command"))
        events = [x for x in getattr(env.clock, "log", []) if x.get("kind") in ("damage", "disconnect", "reconnect")]
        receipt_records = [x for x in comm if x.get("link") == "telemetry" and x.get("status") == "received"]
        recovery = []
        for event in events:
            if event.get("kind") in ("disconnect", "damage"):
                received = [x for x in receipt_records
                            if x.get("entity") == event.get("resource")
                            and x.get("field") in ("connected", "alive")
                            and x.get("measured_at") is not None
                            and float(x["measured_at"]) >= float(event["time"])]
                receipt = min(received, key=lambda x: float(x["time"])) if received else None
                receipt_time = float(receipt["time"]) if receipt else None
                decision = next((x for x in decision_trace if x["replan"] and receipt_time is not None
                                 and x["time"] >= receipt_time), None)
                service = next((x for x in getattr(env.clock, "log", [])
                                if x.get("kind") == "service" and x.get("resource") == event.get("resource")
                                and float(x.get("start", -1)) >= float(event["time"])), None)
                recovery.append({"event": event, "first_legal_telemetry_time": receipt_time,
                                 "communication_wait": (receipt_time - event["time"]) if receipt_time is not None else None,
                                 "replan_time": decision["time"] if decision else None,
                                 "decision_recovery": (decision["time"] - event["time"]) if decision else None,
                                 "service_recovery_time": service.get("start") if service else None,
                                 "service_recovery_delay": (service.get("start") - event["time"]) if service else None,
                                 "note": "Times are simulator timestamps; null means no observed legal recovery before episode end."})
        records.append({
            "tape_id": scenario.tape_id, "scenario_seed": scenario.seed, "return": total_reward,
            "steps": steps, "completed": info["counts"]["completed"], "expired": info["counts"]["expired"],
            "rejected": info["counts"]["rejected"], "energy_remaining": float(sum(info["energy"].values())),
            "actor_calls": actor_calls, "continuation_steps": continuation, "world_model_calls": world_calls,
            "replan_reasons": reasons, "trigger_condition_counts": conditions_count,
            "communication": {"by_link_status": by_link, "canonical_audit_bytes": bytes_by_link,
                               "retransmissions": 0, "expired_messages": sum(v.get("expired", 0) for v in by_link.values()),
                               "duplicate_or_stale_messages": sum(v.get("stale_or_duplicate", 0) for v in by_link.values())},
            "execution_audit": {"accepted_command_count": len(accepted_ids), "duplicate_accepts": duplicate_accepts,
                                 "unauthorized_or_fenced": unauthorized},
            "recovery_events": recovery, "latency": latency_summary(chain_latency),
        })
    keys = ("return", "completed", "expired", "rejected", "energy_remaining", "actor_calls", "continuation_steps", "world_model_calls")
    summary = {key: {"mean": float(np.mean([r[key] for r in records])), "std": float(np.std([r[key] for r in records]))} for key in keys}
    summary["latency"] = latency_summary([x for r in records for x in r["latency"]["raw_ms"]])
    summary["communication"] = {"by_link_status": {}, "canonical_audit_bytes": {}, "retransmissions": 0,
                                 "expired_messages": 0, "duplicate_or_stale_messages": 0}
    for r in records:
        for link, statuses in r["communication"]["by_link_status"].items():
            for status, count in statuses.items():
                summary["communication"]["by_link_status"].setdefault(link, {})[status] = summary["communication"]["by_link_status"].setdefault(link, {}).get(status, 0) + count
        for link, count in r["communication"]["canonical_audit_bytes"].items():
            summary["communication"]["canonical_audit_bytes"][link] = summary["communication"]["canonical_audit_bytes"].get(link, 0) + count
        summary["communication"]["expired_messages"] += r["communication"]["expired_messages"]
        summary["communication"]["duplicate_or_stale_messages"] += r["communication"]["duplicate_or_stale_messages"]
    return {"mode": mode, "policy": str(policy_path), "policy_metadata": policy_meta, "episodes": records, "summary": summary,
            "protocol_notes": {"retransmission_definition": "automatic retransmission is not implemented; count is explicitly zero",
                                "byte_definition": "canonical JSON UTF-8 length of audit records, not a physical network measurement",
                                "recovery_definition": "raw simulator event time to first legally received telemetry retained; unpaired service recovery is null"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--policy-a", type=Path, required=True)
    parser.add_argument("--policy-b", type=Path, required=True)
    parser.add_argument("--policy-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-wait", type=int, default=3)
    args = parser.parse_args()
    payload = json.loads(args.tapes.read_text(encoding="utf-8"))
    scenarios = [scenario_from_dict(x) for x in payload["final_test"]]
    config = M10Config()
    obs_dim = len(scenarios[0].tasks) * 0
    from gppo_world.m10_training import collect_world_dataset
    obs_dim = len(collect_world_dataset(episodes=1, seed=1, config=config, scenarios=scenarios[:1], split="test")[0]["obs"])
    world = load_world(args.world, obs_dim, config.action_count, args.device)
    result = {
        "protocol": "weak communication detailed final-test audit; no training or selection",
        "tape_split": "final_test", "tape_ids": [s.tape_id for s in scenarios],
        "groups": {
            "A-graph5-base-periodic": run_one(args.policy_a, world, mode="periodic", scenarios=scenarios, config=config, device=args.device, threshold=args.threshold, max_wait=args.max_wait),
            "B-graph5-world-periodic": run_one(args.policy_b, world, mode="periodic", scenarios=scenarios, config=config, device=args.device, threshold=args.threshold, max_wait=args.max_wait),
            "C-graph5-triggered": run_one(args.policy_c, world, mode="model-risk", scenarios=scenarios, config=config, device=args.device, threshold=args.threshold, max_wait=args.max_wait),
            "B-graph5-world-rules": run_one(args.policy_b, world, mode="rules", scenarios=scenarios, config=config, device=args.device, threshold=args.threshold, max_wait=args.max_wait),
        },
    }
    dump(args.output, result)


if __name__ == "__main__":
    main()
