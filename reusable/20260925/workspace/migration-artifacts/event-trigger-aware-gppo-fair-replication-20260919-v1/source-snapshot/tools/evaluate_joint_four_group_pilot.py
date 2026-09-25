"""Evaluate immutable A-D pilot checkpoints on the frozen validation tape."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import (
    EVENT_NAMES, ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy,
    JointTrainConfig, build_observed_event_labels, masked_normalized_preference,
)
from gppo_world.joint_training import (
    GROUPS, _mask_safe, _obs_tensor, _relation_for_action, _vector_reward,
)
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
from gppo_world.m10_training import M10ActorCritic, _act, _policy_input_bundle
from tools.run_m10_baseline_comparison import traditional_action


RUN_ROOT = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"
PREFERENCES = ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _transaction(run_dir: Path, step: int) -> Path:
    matches = sorted((run_dir / "transactions").glob(f"txn-step-{step:08d}-policy-*.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one immutable checkpoint at step={step}, found {len(matches)}")
    return matches[0]


def _load_policy(group: str, checkpoint: Path, env_config: M10Config, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("group") != group or payload.get("protocol") != "world-gppo-9.11-arrival-joint-pref-wm-event/0.2.0":
        raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
    if group == "A":
        policy = M10ActorCritic(
            uav_count=env_config.uav_count, task_capacity=env_config.task_capacity,
            action_count=env_config.action_count, encoder="graph", type_count=5,
            history=True, context_dim=0, region_count=env_config.region_count,
            target_count=env_config.target_count, event_capacity=env_config.event_capacity,
            relation_width=env_config.relation_width,
        ).to(device)
        policy.load_state_dict(payload["model_state_dict"], strict=True)
        return policy.eval(), None
    policy = JointGraphPreferencePolicy(env_config, history=True).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    world = ActionConditionedTemporalWorldModel().to(device) if GROUPS[group].world_model else None
    if world is not None:
        world.load_state_dict(payload["world_state_dict"], strict=True)
        world.eval()
    return policy.eval(), world


def _decision(group: str, policy, world, obs, hidden, world_hidden, preference, device):
    world_ms = 0.0
    actor_ms = 0.0
    by_action = {}
    if group == "A":
        vector, _, _, _ = _policy_input_bundle(
            None, obs, None, fusion="base", device=device, trigger_threshold=0.5,
        )
        synchronize(device)
        t0 = time.perf_counter()
        action, log_prob, value, next_hidden = _act(
            policy, vector, _mask_safe(obs["mask"]), hidden, device, deterministic=True,
        )
        synchronize(device)
        actor_ms = (time.perf_counter() - t0) * 1000.0
        return action, log_prob, value, next_hidden, world_hidden, by_action, world_ms, actor_ms

    obs_t = _obs_tensor(obs, device)
    mask_np = _mask_safe(obs["mask"])
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)[None, :]
    with torch.inference_mode():
        synchronize(device)
        t0 = time.perf_counter()
        features, pair, next_hidden = policy.encode(obs_t, hidden)
        synchronize(device)
        actor_ms += (time.perf_counter() - t0) * 1000.0
        if world is not None:
            t0 = time.perf_counter()
            candidate, by_action = world.predict_all_candidates(
                features, world_hidden, obs, use_events=GROUPS[group].event_auxiliary,
            )
            synchronize(device)
            world_ms = (time.perf_counter() - t0) * 1000.0
        else:
            candidate = torch.zeros((1, 25, 17), dtype=torch.float32, device=device)
        t0 = time.perf_counter()
        evaluated = policy.evaluate_encoded(
            features, pair, preference, candidate, mask_t,
        )
        action_t = evaluated["distribution"].probs.argmax(dim=-1)
        log_prob = float(evaluated["distribution"].log_prob(action_t).item())
        value = evaluated["critic_values"][0].detach().cpu().tolist()
        action = int(action_t.item())
        synchronize(device)
        actor_ms += (time.perf_counter() - t0) * 1000.0
    if not bool(mask_np[action]):
        raise RuntimeError("deterministic validation policy emitted an illegal action")
    if world is not None:
        world_hidden = by_action[action]["hidden"].detach()
    return action, log_prob, value, next_hidden, world_hidden, by_action, world_ms, actor_ms


def _episode(group: str, policy, world, scenario, env_config, train_config, device, preference):
    env = M10Environment(env_config, scenario)
    obs = env.reset()
    hidden = None
    world_hidden = None
    pref = None if group == "A" else masked_normalized_preference(preference, device=device)
    initial_energy = env_config.uav_count * env_config.initial_energy
    previous_counts = {"completed": 0, "expired": 0}
    previous_energy = float(initial_energy)
    totals = {"task_return": 0.0, "energy_return": 0.0, "scalar_return": 0.0}
    steps = actor_calls = world_calls = predictions_computed = predictions_consumed = 0
    illegal_actions = 0
    feedback = Counter()
    proxy_bytes = 0
    comm_proxy_messages = 0
    complete_latency: list[float] = []
    actor_latency: list[float] = []
    world_latency: list[float] = []
    event_data = {name: {"prob": [], "label": [], "valid": []} for name in EVENT_NAMES}
    wm_sq = Counter()
    wm_abs = Counter()
    wm_n = Counter()
    done = False
    info = {"counts": {"completed": 0, "expired": 0, "rejected": 0}, "energy": {}, "completion_records": {}}
    max_steps = int(env_config.horizon / env_config.decision_interval) + 2
    while not done and steps < max_steps:
        synchronize(device)
        decision_start = time.perf_counter()
        old_obs = obs
        action, _, _, hidden_next, world_hidden_next, by_action, world_ms, actor_ms = _decision(
            group, policy, world, obs, hidden, world_hidden, pref, device,
        )
        actor_latency.append(actor_ms)
        if world is not None:
            world_latency.append(world_ms)
            world_calls += 1
            predictions_computed += 25
            predictions_consumed += int(np.asarray(obs["mask"], dtype=np.bool_).sum())
        if not bool(np.asarray(obs["mask"], dtype=np.bool_)[action]):
            illegal_actions += 1
        obs, reward, done, info = env.step(action)
        synchronize(device)
        complete_latency.append((time.perf_counter() - decision_start) * 1000.0)
        actor_calls += 1
        steps += 1
        totals["scalar_return"] += float(reward)
        feedback[str(info.get("feedback", "unknown"))] += 1
        for item in info.get("communication_delta", []):
            comm_proxy_messages += 1
            proxy_bytes += len(json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        vector_reward, consequence, previous_counts, previous_energy = _vector_reward(
            info, previous_counts, previous_energy, env_config,
        )
        totals["task_return"] += float(vector_reward[0])
        totals["energy_return"] += float(vector_reward[1])

        if world is not None:
            output = by_action[action]
            event = build_observed_event_labels(
                old_obs, obs, initial_energy=env_config.initial_energy,
                urgent_slack=train_config.urgent_slack_steps,
                low_energy_fraction=train_config.low_energy_fraction,
                same_episode=True,
            )
            logits = output["event_logits"][0].detach()
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for idx, name in enumerate(EVENT_NAMES):
                event_data[name]["prob"].append(float(probabilities[idx]))
                event_data[name]["label"].append(float(event["labels"][idx]))
                event_data[name]["valid"].append(bool(event["mask"][idx]))
            if not bool(info.get("terminated", False)):
                with torch.inference_mode():
                    next_features, _, _ = policy.encode(_obs_tensor(obs, device), hidden_next)
                for key, target in (("next_state", next_features[0]),
                                    ("vector_reward", torch.as_tensor(vector_reward, device=device)),
                                    ("task_consequence", torch.as_tensor(consequence, device=device))):
                    pred = output[key][0].detach()
                    diff = pred - target.to(pred.device, dtype=pred.dtype)
                    wm_sq[key] += float(diff.square().mean().cpu())
                    wm_abs[key] += float(diff.abs().mean().cpu())
                    wm_n[key] += 1
        hidden = hidden_next
        world_hidden = world_hidden_next

    counts = info.get("counts", {})
    records = info.get("completion_records", {})
    host_confirmed = sum(record.get("host_confirmation_time") is not None for record in records.values())
    host_on_time = sum(record.get("host_confirmation_before_deadline") is True for record in records.values())
    completed = int(counts.get("completed", 0))
    expired = int(counts.get("expired", 0))
    unresolved = max(0, len(scenario.tasks) - completed - expired)
    final_energy = float(sum(float(v) for v in info.get("energy", {}).values()))
    execution_log = list(getattr(env.execution, "log", []))
    explicit_violations = [
        row for row in execution_log
        if row.get("safety_violation") is True or row.get("unauthorized_execution") is True
    ]
    decision_metrics = {
        "steps": steps, "actor_calls": actor_calls, "world_forward_calls": world_calls,
        "candidate_predictions_computed": predictions_computed,
        "candidate_predictions_consumed": predictions_consumed,
        "world_latency_ms": world_latency, "actor_critic_latency_ms": actor_latency,
        "complete_decision_latency_ms": complete_latency,
        "communication_proxy_messages": comm_proxy_messages,
        "communication_proxy_bytes": proxy_bytes,
    }
    return {
        "tape_id": scenario.tape_id, "tasks_total": len(scenario.tasks),
        "physical_on_time": completed, "deadline_failed": expired, "unresolved": unresolved,
        "host_confirmed_final": host_confirmed, "host_on_time": host_on_time,
        "scalar_environment_return": totals["scalar_return"],
        "task_objective_return": totals["task_return"], "energy_objective_return": totals["energy_return"],
        "protocol_utility": None if pref is None else float(
            pref[0].item() * train_config.task_reward_scale * totals["task_return"]
            + pref[1].item() * train_config.energy_reward_scale * totals["energy_return"]
        ),
        "energy_used": initial_energy - final_energy,
        "correct_rejections_by_feedback": dict(feedback),
        "illegal_action_count": illegal_actions,
        "explicit_safety_violations_recorded": len(explicit_violations),
        "safety_audit_scope": "illegal mask actions and explicit execution-log safety flags; absence is not production assurance",
        "decision_metrics": decision_metrics,
        "event_predictions": event_data,
        "world_model_errors": {
            key: {"valid_decisions": int(wm_n[key]), "mse_mean": wm_sq[key] / wm_n[key] if wm_n[key] else None,
                  "mae_mean": wm_abs[key] / wm_n[key] if wm_n[key] else None}
            for key in ("next_state", "vector_reward", "task_consequence")
        },
    }


def _summarize(rows, train_event_rates=None):
    task_n = sum(row["tasks_total"] for row in rows)
    out = {
        "episodes": len(rows), "tasks_total": task_n,
        "physical_on_time": sum(row["physical_on_time"] for row in rows),
        "physical_on_time_rate": sum(row["physical_on_time"] for row in rows) / task_n if task_n else None,
        "deadline_failed": sum(row["deadline_failed"] for row in rows),
        "unresolved": sum(row["unresolved"] for row in rows),
        "host_confirmed_final": sum(row["host_confirmed_final"] for row in rows),
        "host_on_time": sum(row["host_on_time"] for row in rows),
        "scalar_environment_return_mean": float(np.mean([row["scalar_environment_return"] for row in rows])) if rows else None,
        "task_objective_return_mean": float(np.mean([row["task_objective_return"] for row in rows])) if rows else None,
        "energy_objective_return_mean": float(np.mean([row["energy_objective_return"] for row in rows])) if rows else None,
        "protocol_utility_mean": float(np.mean([row["protocol_utility"] for row in rows if row["protocol_utility"] is not None]))
        if any(row["protocol_utility"] is not None for row in rows) else None,
        "energy_used_mean": float(np.mean([row["energy_used"] for row in rows])) if rows else None,
        "illegal_actions": sum(row["illegal_action_count"] for row in rows),
        "explicit_safety_violations_recorded": sum(row["explicit_safety_violations_recorded"] for row in rows),
    }
    rejection_counts = Counter()
    for row in rows:
        rejection_counts.update(row.get("correct_rejections_by_feedback", {}))
    out["feedback_counts"] = dict(rejection_counts)
    for metric, key in (("complete_decision_latency_ms", "complete_decision_latency_ms"),
                        ("actor_critic_latency_ms", "actor_critic_latency_ms"),
                        ("world_latency_ms", "world_latency_ms")):
        values = [x for row in rows for x in row["decision_metrics"][key]]
        arr = np.asarray(values, dtype=np.float64)
        out[metric] = {"samples": int(arr.size), "mean": float(arr.mean()) if arr.size else None,
                       "p95": float(np.percentile(arr, 95)) if arr.size else None,
                       "p99": float(np.percentile(arr, 99)) if arr.size else None,
                       "scope": "server CPU inference plus environment step; excludes artifact writes"}
    out["communication_proxy_messages"] = sum(row["decision_metrics"]["communication_proxy_messages"] for row in rows)
    out["communication_proxy_bytes"] = sum(row["decision_metrics"]["communication_proxy_bytes"] for row in rows)
    if rows and all(name in rows[0].get("event_predictions", {}) for name in EVENT_NAMES):
        event_summary = {}
        for name in EVENT_NAMES:
            probabilities, labels, valid = [], [], []
            for row in rows:
                event = row["event_predictions"][name]
                probabilities.extend(event["prob"])
                labels.extend(event["label"])
                valid.extend(event["valid"])
            p = np.asarray(probabilities, dtype=np.float64)
            y = np.asarray(labels, dtype=np.float64)
            m = np.asarray(valid, dtype=bool)
            p, y = p[m], y[m]
            positive = int(y.sum())
            negative = int(len(y) - positive)
            if len(y):
                clipped = np.clip(p, 1e-7, 1 - 1e-7)
                bce = float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())
                brier = float(np.mean((p - y) ** 2))
                pred = p >= 0.5
                tp, fp, fn = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0))), int(np.sum((~pred) & (y == 1)))
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn) if positive else None
                f1 = (2 * precision * recall / max(1e-12, precision + recall)) if positive and precision + recall else 0.0 if positive else None
                baseline_rate = None if train_event_rates is None else train_event_rates.get(name)
                baseline_bce = None
                if baseline_rate is not None:
                    br = float(np.clip(baseline_rate, 1e-7, 1 - 1e-7))
                    baseline_bce = float(-(y * np.log(br) + (1 - y) * np.log(1 - br)).mean())
            else:
                bce = brier = precision = recall = f1 = baseline_bce = None
            event_summary[name] = {
                "valid": int(len(y)), "positive": positive, "negative": negative,
                "masked": int(len(valid) - int(m.sum())), "bce": bce, "brier": brier,
                "precision_at_0_5": precision, "recall_at_0_5": recall, "f1_at_0_5": f1,
                "training_occurrence_rate": None if train_event_rates is None else train_event_rates.get(name),
                "training_rate_baseline_bce": baseline_bce,
                "status": "insufficient_positive_support" if positive == 0 else "measured",
            }
        out["event_heads"] = event_summary
        for key in ("next_state", "vector_reward", "task_consequence"):
            vals = [row["world_model_errors"][key] for row in rows]
            count = sum(x["valid_decisions"] for x in vals)
            out.setdefault("world_model_errors", {})[key] = {
                "valid_decisions": count,
                "mse_mean": sum((x["mse_mean"] or 0) * x["valid_decisions"] for x in vals) / count if count else None,
                "mae_mean": sum((x["mae_mean"] or 0) * x["valid_decisions"] for x in vals) / count if count else None,
            }
    return out


def _training_event_rates(run_dir: Path) -> dict[str, float]:
    totals = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    positives = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    path = run_dir / "training-ledger.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            event = row.get("event_labels", {})
            labels, mask = event.get("labels"), event.get("mask")
            if labels is None or mask is None:
                continue
            labels = np.asarray(labels, dtype=np.int64).reshape(-1)
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if labels.size == len(EVENT_NAMES) and mask.size == len(EVENT_NAMES):
                totals += mask.astype(np.int64)
                positives += labels * mask.astype(np.int64)
    return {name: float(positives[i] / totals[i]) if totals[i] else 0.0 for i, name in enumerate(EVENT_NAMES)}


def main() -> int:
    if not (RUN_ROOT / "pilot-training-summary.json").is_file():
        raise SystemExit("pilot training summary is missing; refusing evaluation")
    tapes = json.loads((RUN_ROOT / "frozen-tapes.json").read_text(encoding="utf-8"))
    validation = [scenario_from_dict(item) for item in tapes["validation"]]
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    train_config = JointTrainConfig(seed=1101, rollout_steps=128)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    results = {"validation_tape_sha256": sha256(RUN_ROOT / "frozen-tapes.json"), "groups": {}, "rule": None}
    total_start = time.perf_counter()
    for group in ("A", "B", "C", "D"):
        run_dir = RUN_ROOT / group
        summary = json.loads((run_dir / "pilot-group-summary.json").read_text(encoding="utf-8"))
        if summary.get("environment_steps") != 4096:
            raise RuntimeError(f"group {group} is not a full fixed-budget checkpoint")
        checkpoints = [4096] if group == "A" else [1024, 2048, 3072, 4096]
        group_rows = {}
        training_event_rates = _training_event_rates(RUN_ROOT / "D") if group in ("C", "D") else None
        for step in checkpoints:
            checkpoint = _transaction(run_dir, step)
            policy, world = _load_policy(group, checkpoint, env_config, device)
            preference_rows = {}
            prefs = (None,) if group == "A" else PREFERENCES
            warm_scenario = validation[0]
            warm_env = M10Environment(env_config, warm_scenario)
            warm_obs = warm_env.reset()
            warm_pref = (0.5, 0.5) if group == "A" else PREFERENCES[0]
            synchronize(device)
            _decision(group, policy, world, warm_obs, None, None,
                      None if group == "A" else masked_normalized_preference(warm_pref, device=device), device)
            synchronize(device)
            for pref in prefs:
                key = "not_applicable" if pref is None else "(" + ",".join(f"{x:.1f}" for x in pref) + ")"
                episode_rows = [
                    _episode(group, policy, world, scenario, env_config, train_config, device,
                             (0.5, 0.5) if pref is None else pref)
                    for scenario in validation
                ]
                preference_rows[key] = {
                    "preference": pref, "episodes": episode_rows,
                    "summary": _summarize(episode_rows, training_event_rates),
                }
            group_rows[str(step)] = {
                "checkpoint": str(checkpoint.relative_to(RUN_ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_counters": torch.load(checkpoint, map_location="cpu", weights_only=False).get("counters"),
                "preferences": preference_rows,
            }
        results["groups"][group] = group_rows
    rule_rows = []
    for scenario in validation:
        env = M10Environment(env_config, scenario)
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        previous_counts = {"completed": 0, "expired": 0}
        previous_energy = float(env_config.uav_count * env_config.initial_energy)
        task_return = energy_return = 0.0
        feedback = Counter()
        full_latency = []
        actor_latency = []
        proxy_bytes = 0
        proxy_messages = 0
        while not done and steps < int(env_config.horizon / env_config.decision_interval) + 2:
            decision_start = time.perf_counter()
            actor_start = time.perf_counter()
            action = int(traditional_action(obs))
            actor_latency.append((time.perf_counter() - actor_start) * 1000.0)
            if not bool(np.asarray(obs["mask"], dtype=bool)[action]):
                raise RuntimeError("frozen legal rule emitted an illegal action")
            obs, reward, done, info = env.step(action)
            full_latency.append((time.perf_counter() - decision_start) * 1000.0)
            total_reward += float(reward)
            vector, _, previous_counts, previous_energy = _vector_reward(info, previous_counts, previous_energy, env_config)
            task_return += float(vector[0]); energy_return += float(vector[1])
            feedback[str(info.get("feedback", "unknown"))] += 1
            for item in info.get("communication_delta", []):
                proxy_messages += 1
                proxy_bytes += len(json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
            steps += 1
        records = info.get("completion_records", {})
        rule_rows.append({
            "tape_id": scenario.tape_id, "tasks_total": len(scenario.tasks),
            "physical_on_time": int(info.get("counts", {}).get("completed", 0)),
            "deadline_failed": int(info.get("counts", {}).get("expired", 0)),
            "unresolved": max(0, len(scenario.tasks) - int(info.get("counts", {}).get("completed", 0)) - int(info.get("counts", {}).get("expired", 0))),
            "host_confirmed_final": sum(x.get("host_confirmation_time") is not None for x in records.values()),
            "host_on_time": sum(x.get("host_confirmation_before_deadline") is True for x in records.values()),
            "scalar_environment_return": total_reward, "task_objective_return": task_return,
            "energy_objective_return": energy_return, "energy_used": env_config.uav_count * env_config.initial_energy - sum(info.get("energy", {}).values()),
            "protocol_utility": None,
            "correct_rejections_by_feedback": dict(feedback),
            "illegal_action_count": 0,
            "explicit_safety_violations_recorded": 0,
            "decision_metrics": {
                "steps": steps, "actor_calls": steps, "world_forward_calls": 0,
                "candidate_predictions_computed": 0, "candidate_predictions_consumed": 0,
                "world_latency_ms": [], "actor_critic_latency_ms": actor_latency,
                "complete_decision_latency_ms": full_latency,
                "communication_proxy_messages": proxy_messages,
                "communication_proxy_bytes": proxy_bytes,
            },
        })
    results["rule"] = {"episodes": rule_rows, "summary": _summarize(rule_rows)}
    results["evaluation_wall_seconds"] = time.perf_counter() - total_start
    results["latency_protocol"] = "one warmup inference per loaded checkpoint; no warmup episode step; CPU timing; complete latency includes environment step; file writes excluded"
    results["scope"] = "fixed 16-parent validation only; development evidence, not formal test or generalization result"
    output = RUN_ROOT / "validation-results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "evaluation_wall_seconds": results["evaluation_wall_seconds"],
        "rule": results["rule"]["summary"],
        "groups": {g: {step: {pref: block["summary"]["physical_on_time"]
                             for pref, block in payload["preferences"].items()}
                        for step, payload in rows.items()}
                   for g, rows in results["groups"].items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
