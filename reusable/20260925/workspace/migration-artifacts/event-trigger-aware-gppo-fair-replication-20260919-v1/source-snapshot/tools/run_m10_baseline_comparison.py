"""Run the M-10 legal traditional, GPPO, and GPPO-History comparison.

This runner deliberately excludes the consequence model.  It is a bounded
baseline completion for the same Graph-5/25-action and weak-communication
contract.  Pilot and formal outputs are separate and never reused.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, scenario_to_dict, weak_communication_tape  # noqa: E402
from gppo_world.m10_training import PPOConfig, evaluate_policy, save_policy, train_policy  # noqa: E402
from tools.evaluate_m10_weak_comm_details import run_one  # noqa: E402


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def tape_payload(base_seed: int = 93001) -> dict[str, list[M10Scenario]]:
    return {
        "train": list(weak_communication_tape("train", count=32, base_seed=base_seed, level="composite")),
        "validation": list(weak_communication_tape("validation", count=16, base_seed=base_seed, level="composite")),
        "final_test": list(weak_communication_tape("test", count=16, base_seed=base_seed, level="composite")),
    }


def freeze_tapes(tapes: dict[str, list[M10Scenario]], output: Path, base_seed: int) -> Path:
    ids = {key: [item.tape_id for item in values] for key, values in tapes.items()}
    all_ids = [item for values in ids.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("tape ids overlap")
    path = output / "tapes.json"
    dump(path, {key: [scenario_to_dict(item) for item in values] for key, values in tapes.items()})
    dump(output / "tape-audit.json", {
        "base_seed": base_seed,
        "split_counts": {key: len(values) for key, values in tapes.items()},
        "all_tape_ids_unique": True,
        "communication": tapes["train"][0].communication.to_dict(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    return path


def _task_public_values(obs: dict[str, Any], action: int, now: float) -> tuple[float, float, float, float]:
    task = action % 6
    row = np.asarray(obs["tasks"], dtype=np.float32)[task]
    # TaskPolicyView encodes four values per public field: value, known,
    # fresh, age. These columns are public telemetry, never simulator truth.
    deadline = float(row[8])
    remaining = float(row[12])
    priority = float(row[16])
    distance = float(np.asarray(obs["graph"]["relations"], dtype=np.float32).reshape(24, 4)[action, 0])
    return distance, deadline, remaining, priority


def traditional_action(obs: dict[str, Any]) -> int:
    legal = [int(index) for index, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
    candidates = [action for action in legal if action < 24]
    if not candidates:
        return 24
    now = float(obs["time"])
    scored = []
    for action in candidates:
        distance, deadline, remaining, priority = _task_public_values(obs, action, now)
        score = 2.0 * distance + remaining / max(deadline - now, 0.1) - 0.25 * priority
        scored.append((score, action))
    return min(scored, key=lambda item: (item[0], item[1]))[1]


def latency_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean_ms": float(array.mean()) if array.size else None,
        "p95_ms": float(np.percentile(array, 95)) if array.size else None,
        "p99_ms": float(np.percentile(array, 99)) if array.size else None,
        "raw_ms": array.tolist(),
    }


def canonical_bytes(item: dict[str, Any]) -> int:
    return len(json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def detailed_rules(scenarios: list[M10Scenario], config: M10Config, device: str) -> dict[str, Any]:
    records = []
    device_obj = torch.device(device)
    for scenario in scenarios:
        env = M10Environment(config, scenario)
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        latencies: list[float] = []
        info: dict[str, Any] = {"counts": {"completed": 0, "expired": 0, "rejected": 0}, "energy": {}}
        action_trace = []
        while not done and steps < int(config.horizon / config.decision_interval) + 2:
            start = time.perf_counter()
            action = traditional_action(obs)
            action_trace.append({"time": float(obs["time"]), "action": action, "submit_command": True})
            obs, reward, done, info = env.step(action, submit_command=True)
            total_reward += float(reward)
            steps += 1
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            latencies.append((time.perf_counter() - start) * 1000.0)
        communication = list(info.get("communication_log", []))
        by_link: dict[str, dict[str, int]] = {}
        bytes_by_link: dict[str, int] = {}
        for item in communication:
            link = str(item.get("link", "unknown"))
            status = str(item.get("status", "unknown"))
            by_link.setdefault(link, {})[status] = by_link.setdefault(link, {}).get(status, 0) + 1
            bytes_by_link[link] = bytes_by_link.get(link, 0) + canonical_bytes(item)
        accepted_ids = [str(item.get("command_id")) for item in getattr(env.execution, "log", []) if item.get("result") == "accepted"]
        execution_log = list(getattr(env.execution, "log", []))
        events = [item for item in getattr(env.clock, "log", []) if item.get("kind") in ("damage", "disconnect", "reconnect")]
        telemetry = [item for item in communication if item.get("link") == "telemetry" and item.get("status") == "received"]
        recovery = []
        for event in events:
            if event.get("kind") not in ("damage", "disconnect"):
                continue
            received = [item for item in telemetry if item.get("entity") == event.get("resource") and item.get("field") in ("connected", "alive") and item.get("measured_at") is not None and float(item["measured_at"]) >= float(event["time"])]
            receipt = min(received, key=lambda item: float(item["time"])) if received else None
            recovery.append({
                "event": event,
                "first_legal_telemetry_time": float(receipt["time"]) if receipt else None,
                "communication_wait": float(receipt["time"]) - float(event["time"]) if receipt else None,
                "note": "null means no legally observed recovery before episode end",
            })
        records.append({
            "tape_id": scenario.tape_id,
            "scenario_seed": scenario.seed,
            "return": total_reward,
            "steps": steps,
            "completed": info["counts"]["completed"],
            "expired": info["counts"]["expired"],
            "rejected": info["counts"]["rejected"],
            "energy_remaining": float(sum(info["energy"].values())),
            "actor_calls": steps,
            "continuation_steps": 0,
            "world_model_calls": 0,
            "communication": {"by_link_status": by_link, "canonical_audit_bytes": bytes_by_link, "retransmissions": 0, "expired_messages": sum(v.get("expired", 0) for v in by_link.values()), "duplicate_or_stale_messages": sum(v.get("stale_or_duplicate", 0) for v in by_link.values())},
            "execution_audit": {"accepted_command_count": len(accepted_ids), "duplicate_accepts": len(accepted_ids) - len(set(accepted_ids)), "unauthorized_or_fenced": sum(1 for item in execution_log if item.get("result") in ("stale", "fenced", "expired", "unknown_command"))},
            "recovery_events": recovery,
            "action_trace": action_trace,
            "latency": latency_summary(latencies),
        })
    return summarize_records(records, variant="legal-public-urgency-distance", policy=None)


def summarize_records(records: list[dict[str, Any]], *, variant: str, policy: str | None) -> dict[str, Any]:
    keys = ("return", "completed", "expired", "rejected", "energy_remaining", "actor_calls", "continuation_steps", "world_model_calls")
    summary = {key: {"mean": float(np.mean([record[key] for record in records])), "std": float(np.std([record[key] for record in records]))} for key in keys}
    summary["latency"] = latency_summary([value for record in records for value in record["latency"]["raw_ms"]])
    summary["communication"] = {"by_link_status": {}, "canonical_audit_bytes": {}, "retransmissions": 0, "expired_messages": 0, "duplicate_or_stale_messages": 0}
    for record in records:
        for link, statuses in record["communication"]["by_link_status"].items():
            for status, count in statuses.items():
                summary["communication"]["by_link_status"].setdefault(link, {})[status] = summary["communication"]["by_link_status"].setdefault(link, {}).get(status, 0) + count
        for link, count in record["communication"]["canonical_audit_bytes"].items():
            summary["communication"]["canonical_audit_bytes"][link] = summary["communication"]["canonical_audit_bytes"].get(link, 0) + count
        summary["communication"]["expired_messages"] += record["communication"]["expired_messages"]
        summary["communication"]["duplicate_or_stale_messages"] += record["communication"]["duplicate_or_stale_messages"]
    return {"variant": variant, "policy": policy, "episodes": records, "summary": summary, "world_model_calls": 0, "protocol_notes": {"byte_definition": "canonical JSON UTF-8 audit serialization proxy, not physical traffic", "recovery_definition": "simulator event to first legally received telemetry; null is not zero"}}


def save_training(output: Path, variant: str, seed: int, tapes: dict[str, list[M10Scenario]], config: M10Config, device: str, steps: int) -> dict[str, Any]:
    history = variant.endswith("History-Graph5")
    policy, metadata = train_policy(
        variant=variant,
        encoder="graph",
        type_count=5,
        history=history,
        fusion="base",
        model=None,
        seed=seed,
        steps=steps,
        env_config=config,
        ppo_config=PPOConfig(rollout_steps=256, update_epochs=4),
        device=device,
        scenarios=tapes["train"],
    )
    path = output / "training" / variant / f"seed-{seed}" / "policy.pt"
    save_policy(path, policy, metadata)
    dump(path.with_name("metadata.json"), metadata)
    return {"variant": variant, "seed": seed, "steps": steps, "checkpoint": str(path), "metadata": metadata}


def evaluate_learned(checkpoint: Path, scenarios: list[M10Scenario], config: M10Config, device: str) -> dict[str, Any]:
    # run_one reloads the saved policy and emits per-event, safety, communication,
    # and complete-chain latency records on the same public periodic contract.
    return run_one(checkpoint, None, mode="periodic", scenarios=scenarios, config=config, device=device, threshold=0.1, max_wait=3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=93001)
    parser.add_argument("--pilot-steps", type=int, default=512)
    parser.add_argument("--formal-steps", type=int, default=8192)
    parser.add_argument("--seeds", default="1101,2203,3307")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no silent fallback")
    if args.threads < 1 or args.threads > 4:
        parser.error("threads must be between 1 and 4")
    if args.mode == "formal" and args.formal_steps > 8192:
        parser.error("formal-steps exceeds registered 8192 per seed cap")
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if args.mode == "formal" and seeds != [1101, 2203, 3307]:
        parser.error("formal seeds are frozen to 1101,2203,3307")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    config = M10Config()
    tapes = tape_payload(args.base_seed)
    freeze_tapes(tapes, args.output, args.base_seed)
    dump(args.output / "runtime.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": args.device, "threads": args.threads, "cuda_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "cuda_version": torch.version.cuda if args.device == "cuda" else None, "started_at": time.time()})
    results: dict[str, Any] = {"mode": args.mode, "protocol": "world-gppo-9.11-baseline-comparison/0.1.0", "device": args.device, "threads": args.threads, "training_performed": args.mode == "formal" or args.mode == "pilot", "groups": {}}
    results["groups"]["traditional"] = detailed_rules(tapes["validation" if args.mode == "pilot" else "final_test"], config, args.device)
    variants = ("GPPO-Graph5", "GPPO-History-Graph5")
    run_seeds = [1101] if args.mode == "pilot" else seeds
    for variant in variants:
        entries = []
        for seed in run_seeds:
            entry = save_training(args.output, variant, seed, tapes, config, args.device, args.pilot_steps if args.mode == "pilot" else args.formal_steps)
            evaluation = evaluate_learned(Path(entry["checkpoint"]), tapes["validation" if args.mode == "pilot" else "final_test"], config, args.device)
            entry["evaluation"] = evaluation
            entries.append(entry)
        results["groups"][variant] = entries
    dump(args.output / ("pilot-results.json" if args.mode == "pilot" else "formal-results.json"), results)
    dump(args.output / "run-complete.json", {"status": "complete", "mode": args.mode, "completed_at": time.time(), "formal_training_steps": 2 * 3 * args.formal_steps if args.mode == "formal" else 0})
    print(json.dumps({"status": "complete", "mode": args.mode, "output": str(args.output), "formal_steps": 2 * 3 * args.formal_steps if args.mode == "formal" else 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
