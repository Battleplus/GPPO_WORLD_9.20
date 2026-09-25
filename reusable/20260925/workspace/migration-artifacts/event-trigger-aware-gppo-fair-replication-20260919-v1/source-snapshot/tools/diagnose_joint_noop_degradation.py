"""Frozen C/D R-Z-L intervention diagnostic on the existing validation tapes.

This is a development diagnostic only. It never updates parameters and never
uses labels or hidden simulator truth for action selection.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gppo_world.joint_gppo import JointTrainConfig, masked_normalized_preference
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict
from gppo_world.joint_gppo import ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy
from gppo_world.joint_training import _mask_safe, _obs_tensor
from gppo_world.joint_gppo import _base_logits

RUN_ROOT = ROOT / "runs" / "joint-four-group-single-seed-pilot-cpu-20260916-v1"
OUT_ROOT = ROOT / "runs" / "joint-noop-degradation-diagnostic-cpu-20260916-v1"
MODES = ("R", "Z", "L")
GROUPS = ("C", "D")
PREFERENCE = (0.5, 0.5)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return repr(value)


def public_digest(obs: dict[str, Any]) -> str:
    payload = json.dumps(_canonical(obs), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(group: str, checkpoint: Path, env_config: M10Config, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    expected = "world-gppo-9.11-arrival-joint-pref-wm-event/0.2.0"
    if payload.get("group") != group or payload.get("protocol") != expected:
        raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
    policy = JointGraphPreferencePolicy(env_config, history=True).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    world = ActionConditionedTemporalWorldModel().to(device)
    world.load_state_dict(payload["world_state_dict"], strict=True)
    policy.eval()
    world.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return policy, world


def _decision(group: str, mode: str, policy, world, obs, hidden, world_hidden, preference, device):
    obs_t = _obs_tensor(obs, device)
    mask_np = _mask_safe(obs["mask"])
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)[None, :]
    with torch.inference_mode():
        features, pair, next_hidden = policy.encode(obs_t, hidden)
        candidate, by_action = world.predict_all_candidates(
            features, world_hidden, obs, use_events=(group == "D"),
        )
        pref = masked_normalized_preference(preference, device=device)
        pref_batch = pref[None, :] if pref.ndim == 1 else pref
        base = _base_logits(policy.base, features, pair)
        preference_logits = policy.preference_actor(torch.cat((features, pref_batch), dim=-1))
        raw_candidate_logits = policy.candidate_actor(candidate).squeeze(-1)
        zero_candidate = torch.zeros_like(candidate)
        zero_candidate_logits = policy.candidate_actor(zero_candidate).squeeze(-1)
        if mode == "R":
            candidate_input = candidate
            candidate_logits = raw_candidate_logits
        elif mode == "Z":
            candidate_input = zero_candidate
            candidate_logits = zero_candidate_logits
        elif mode == "L":
            candidate_input = candidate
            candidate_logits = torch.zeros_like(raw_candidate_logits)
        else:
            raise ValueError(mode)
        del candidate_input  # kept conceptually explicit; logits are the same branch below
        final_logits = base + preference_logits + candidate_logits
        masked_logits = final_logits.masked_fill(~mask_t, float("-inf"))
        probabilities = torch.softmax(masked_logits, dim=-1)
        action = int(probabilities.argmax(dim=-1).item())
        if not bool(mask_np[action]):
            raise RuntimeError(f"{group}/{mode} selected illegal action {action}")
        legal_non_noop = int(mask_np[:24].sum())
        raw_features = candidate[0].detach().cpu().numpy()
        row = {
            "legal_action_count": int(mask_np.sum()),
            "legal_non_noop_count": legal_non_noop,
            "base_logits_min": float(base.min().item()),
            "base_logits_max": float(base.max().item()),
            "preference_logits_min": float(preference_logits.min().item()),
            "preference_logits_max": float(preference_logits.max().item()),
            "raw_candidate_logits_min": float(raw_candidate_logits.min().item()),
            "raw_candidate_logits_max": float(raw_candidate_logits.max().item()),
            "zero_input_candidate_logits_min": float(zero_candidate_logits.min().item()),
            "zero_input_candidate_logits_max": float(zero_candidate_logits.max().item()),
            "candidate_feature_min": float(raw_features.min()),
            "candidate_feature_max": float(raw_features.max()),
            "final_logits_min": float(final_logits.min().item()),
            "final_logits_max": float(final_logits.max().item()),
            "noop_probability": float(probabilities[0, 24].item()),
            "action": action,
            "candidate_branch_mode": mode,
            "world_hidden_next_norm": float(by_action[action]["hidden"].norm().item()),
        }
    return action, next_hidden, by_action[action]["hidden"].detach(), row


def _episode_rows(group: str, mode: str, policy, world, raw_tape: dict[str, Any], env_config, device):
    scenario = scenario_from_dict(raw_tape)
    env = M10Environment(env_config, scenario)
    obs = env.reset()
    hidden = None
    world_hidden = None
    total_steps = 0
    noop = noop_with_nonnoop = legal_non_noop_steps = 0
    feedback = Counter()
    decision_rows = []
    done = False
    max_steps = int(env_config.horizon / env_config.decision_interval) + 2
    while not done and total_steps < max_steps:
        digest = public_digest(obs)
        action, hidden_next, world_hidden_next, logits = _decision(
            group, mode, policy, world, obs, hidden, world_hidden, PREFERENCE, device,
        )
        if action == 24:
            noop += 1
            if logits["legal_non_noop_count"] > 0:
                noop_with_nonnoop += 1
        if logits["legal_non_noop_count"] > 0:
            legal_non_noop_steps += 1
        old_obs = obs
        obs, reward, done, info = env.step(action)
        feedback[str(info.get("feedback", "unknown"))] += 1
        logits.update({"step": total_steps, "public_state_digest": digest, "reward": float(reward)})
        decision_rows.append(logits)
        hidden, world_hidden = hidden_next, world_hidden_next
        total_steps += 1
        del old_obs
    counts = info.get("counts", {})
    records = info.get("completion_records", {})
    execution_log = list(getattr(env.execution, "log", []))
    completed = int(counts.get("completed", 0))
    expired = int(counts.get("expired", 0))
    tasks_total = len(scenario.tasks)
    commands_submitted = sum(row.get("result") == "awaiting_ack" for row in execution_log)
    accepted = sum(row.get("result") == "accepted" for row in execution_log)
    return {
        "group": group, "mode": mode, "tape_id": scenario.tape_id,
        "tasks_total": tasks_total, "steps": total_steps,
        "physical_on_time": completed, "deadline_failed": expired,
        "unresolved": max(0, tasks_total - completed - expired),
        "host_confirmed_final": sum(row.get("host_confirmation_time") is not None for row in records.values()),
        "host_on_time": sum(row.get("host_confirmation_before_deadline") is True for row in records.values()),
        "noop_count": noop, "noop_with_nonnoop_legal": noop_with_nonnoop,
        "legal_non_noop_decisions": legal_non_noop_steps,
        "feedback_counts": dict(feedback),
        "commands_submitted": int(commands_submitted), "accepted_commands": int(accepted),
        "execution_log_entries": len(execution_log),
        "decision_rows": decision_rows,
    }


def _summary(rows):
    def values(key): return [x[key] for x in rows]
    return {
        "episodes": len(rows), "tasks_total": sum(values("tasks_total")),
        "physical_on_time": sum(values("physical_on_time")),
        "deadline_failed": sum(values("deadline_failed")),
        "unresolved": sum(values("unresolved")),
        "host_confirmed_final": sum(values("host_confirmed_final")),
        "host_on_time": sum(values("host_on_time")),
        "noop_count": sum(values("noop_count")),
        "noop_with_nonnoop_legal": sum(values("noop_with_nonnoop_legal")),
        "legal_non_noop_decisions": sum(values("legal_non_noop_decisions")),
        "commands_submitted": sum(values("commands_submitted")),
        "accepted_commands": sum(values("accepted_commands")),
        "execution_log_entries": sum(values("execution_log_entries")),
        "feedback_counts": dict(Counter({k: sum(row.get("feedback_counts", {}).get(k, 0) for row in rows)
                                          for k in {key for row in rows for key in row.get("feedback_counts", {})}})),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    tapes = json.loads((RUN_ROOT / "frozen-tapes.json").read_text(encoding="utf-8"))["validation"]
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {
        "kind": "joint-noop-degradation-diagnostic",
        "scope": "frozen CPU validation diagnostic; no optimizer updates; no new tapes",
        "protocol": "world-gppo-9.11-arrival-joint-pref-wm-event/0.2.0",
        "preference": PREFERENCE,
        "run_root": str(RUN_ROOT),
        "validation_tape_sha256": sha256(RUN_ROOT / "frozen-tapes.json"),
        "checkpoint_sha256": {}, "groups": {}, "common_public_state_comparison": {},
        "started_at": time.time(),
    }
    for group in GROUPS:
        checkpoint = RUN_ROOT / group / "transactions" / "txn-step-00004096-policy-000032-world-000256.pt"
        results["checkpoint_sha256"][group] = sha256(checkpoint)
        policy, world = _load(group, checkpoint, env_config, device)
        all_rows = []
        common_rows = []
        # Lockstep copies expose the same public state until a mode diverges;
        # full rows remain separate and are never substituted by this prefix.
        for tape in tapes:
            envs = {mode: M10Environment(env_config, scenario_from_dict(tape)) for mode in MODES}
            obs = {mode: envs[mode].reset() for mode in MODES}
            hidden = {mode: None for mode in MODES}
            world_hidden = {mode: None for mode in MODES}
            done = {mode: False for mode in MODES}
            last_info = {mode: {"counts": {}, "completion_records": {}} for mode in MODES}
            mode_stats = {mode: {"noop_count": 0, "noop_with_nonnoop_legal": 0,
                                 "legal_non_noop_decisions": 0, "feedback_counts": Counter(),
                                 "decision_rows": []} for mode in MODES}
            step = 0
            max_steps = int(env_config.horizon / env_config.decision_interval) + 2
            while not all(done.values()) and step < max_steps:
                digests = {mode: public_digest(obs[mode]) for mode in MODES if not done[mode]}
                same_state = len(set(digests.values())) == 1 and len(digests) == 3
                lock = {"tape_id": tape["tape_id"], "step": step, "same_public_state": same_state, "digests": digests, "modes": {}}
                decisions = {}
                for mode in MODES:
                    if done[mode]:
                        continue
                    action, hidden_next, world_next, row = _decision(
                        group, mode, policy, world, obs[mode], hidden[mode], world_hidden[mode], PREFERENCE, device,
                    )
                    decisions[mode] = (action, hidden_next, world_next, row)
                    lock["modes"][mode] = row
                if same_state:
                    common_rows.append(lock)
                for mode, (action, hidden_next, world_next, row) in decisions.items():
                    obs[mode], _, done[mode], info = envs[mode].step(action)
                    last_info[mode] = info
                    hidden[mode], world_hidden[mode] = hidden_next, world_next
                    row["step"] = step
                    row["public_state_digest"] = digests[mode]
                    mode_stats[mode]["decision_rows"].append(row)
                    if action == 24:
                        mode_stats[mode]["noop_count"] += 1
                        if row["legal_non_noop_count"] > 0:
                            mode_stats[mode]["noop_with_nonnoop_legal"] += 1
                    if row["legal_non_noop_count"] > 0:
                        mode_stats[mode]["legal_non_noop_decisions"] += 1
                    mode_stats[mode]["feedback_counts"][str(info.get("feedback", "unknown"))] += 1
                step += 1
            for mode in MODES:
                info = last_info[mode]
                env = envs[mode]
                scenario = env.scenario
                counts = info.get("counts", {})
                records = info.get("completion_records", {})
                execution_log = list(getattr(env.execution, "log", []))
                completed = int(counts.get("completed", 0))
                expired = int(counts.get("expired", 0))
                tasks_total = len(scenario.tasks)
                all_rows.append({
                    "group": group, "mode": mode, "tape_id": scenario.tape_id,
                    "tasks_total": tasks_total, "steps": len(mode_stats[mode]["decision_rows"]),
                    "physical_on_time": completed, "deadline_failed": expired,
                    "unresolved": max(0, tasks_total - completed - expired),
                    "host_confirmed_final": sum(row.get("host_confirmation_time") is not None for row in records.values()),
                    "host_on_time": sum(row.get("host_confirmation_before_deadline") is True for row in records.values()),
                    "noop_count": mode_stats[mode]["noop_count"],
                    "noop_with_nonnoop_legal": mode_stats[mode]["noop_with_nonnoop_legal"],
                    "legal_non_noop_decisions": mode_stats[mode]["legal_non_noop_decisions"],
                    "feedback_counts": dict(mode_stats[mode]["feedback_counts"]),
                    "commands_submitted": int(sum(row.get("result") == "awaiting_ack" for row in execution_log)),
                    "accepted_commands": int(sum(row.get("result") == "accepted" for row in execution_log)),
                    "execution_log_entries": len(execution_log),
                    "decision_rows": mode_stats[mode]["decision_rows"],
                })
        results["groups"][group] = {mode: {"summary": _summary([row for row in all_rows if row["mode"] == mode]),
                                           "episodes": [row for row in all_rows if row["mode"] == mode]}
                                     for mode in MODES}
        results["common_public_state_comparison"][group] = common_rows
    results["finished_at"] = time.time()
    # Drop duplicated per-decision rows from the report only after the values
    # have been retained in the separate full ledger below.
    report = json.loads(json.dumps(results))
    for group in GROUPS:
        for mode in MODES:
            report["groups"][group][mode]["episodes"] = [
                {key: value for key, value in row.items() if key != "decision_rows"}
                for row in report["groups"][group][mode]["episodes"]
            ]
    (OUT_ROOT / "diagnostic-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_ROOT / "diagnostic-full-ledger.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT_ROOT), "groups": {g: {m: results["groups"][g][m]["summary"] for m in MODES} for g in GROUPS},
                      "checkpoint_sha256": results["checkpoint_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
