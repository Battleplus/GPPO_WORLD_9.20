"""Bounded event-trigger-aware GPPO comparison on the frozen WD start points.

The historical runners store one PPO sample per environment step.  This
runner intentionally uses a separate decision-level contract: continuation
steps are logged but never receive an actor log-probability.  Completed
decision windows use semi-Markov discounted returns and GAE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.event_trigger_semimarkov import (  # noqa: E402
    DecisionWindow, decision_level_gae, discounted_window, validate_window_contract,
)
from gppo_world.budget_executor import BudgetExhausted, PersistentBudget  # noqa: E402
from gppo_world.joint_gppo import (  # noqa: E402
    JOINT_PROTOCOL, ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy,
    JointTrainConfig, masked_normalized_preference,
)
from gppo_world.joint_training import (  # noqa: E402
    _TrainingLedger, _atomic_bytes, _atomic_json, _atomic_torch_save, _batch_encode,
    _mask_safe, _obs_tensor, _relation_for_action, _vector_reward,
    build_observed_event_labels, finite_optimizer, finite_parameters,
    scalarized_preference_advantage,
)
from gppo_world.m10_environment import M10Config, M10Environment, scenario_from_dict  # noqa: E402


SEEDS = (1101, 2203, 3307)
CONDITIONS = ("I", "W1", "W2")
PREFERENCE = (0.8, 0.2)
SOURCE_FINAL = Path(r"E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final")
CHECKPOINT_HASHES = {
    1101: "5b010a2f7eed0af0e3c64333b5c7846f18ffd0e72aff5705719003e1791b5930",
    2203: "f1e12262bb548f5a8329fbcf993fcf7238abd50c8fe892adc1b7ef29c4763af2",
    3307: "548b848a776dc31a4ffa96be7e50daf1f6f27929e2f314f272bf141a69063941",
}
TAPE_HASH = "5d1cfaed3683e3e1c512f4ff67d1f93566dae2b418942b67df3f1c81cb5eaea7"
MAX_REPLAN_INTERVAL = 1
WINDOW_PROTOCOL = "event-trigger-semi-markov-window-boundary-v3-history-replay"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    _atomic_json(path, value if isinstance(value, Mapping) else {"value": value})


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, DecisionWindow):
        return asdict(value)
    raise TypeError(type(value).__name__)


def load_config() -> M10Config:
    payload = json.loads((SOURCE_FINAL / "run/training/seed-1101/WD/resolved-config.json").read_text(encoding="utf-8"))
    names = {field.name for field in M10Config.__dataclass_fields__.values()}
    return M10Config(**{key: value for key, value in payload["environment"].items() if key in names})


def load_tapes() -> dict[str, dict[str, tuple[Any, ...]]]:
    tape_path = SOURCE_FINAL / "historical-formal-tapes-851001-851003.json"
    actual = sha256(tape_path)
    if actual != TAPE_HASH:
        raise RuntimeError(f"frozen tape hash mismatch: {actual}")
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, tuple[Any, ...]]] = {}
    for split in ("train", "validation"):
        result[split] = {
            condition: tuple(scenario_from_dict(item) for item in payload[split][condition])
            for condition in CONDITIONS
        }
    return result


def interleave(parts: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    result: list[Any] = []
    for index in range(max(len(part) for part in parts)):
        for part in parts:
            if index < len(part):
                result.append(part[index])
    return tuple(result)


def public_dispatch_opportunity(obs: Mapping[str, Any]) -> bool:
    mask = np.asarray(obs["mask"], dtype=np.bool_)
    continuation = {int(value) for value in obs.get("continuation_actions", ())}
    active_slots = {action % 6 for action in continuation if 0 <= action < 24}
    uavs = np.asarray(obs.get("uavs", []), dtype=np.float64)
    tasks = np.asarray(obs.get("tasks", []), dtype=np.float64)

    def valid(row: np.ndarray, field: int) -> bool:
        offset = field * 4
        return len(row) > offset + 2 and row[offset + 1] > 0.5 and row[offset + 2] > 0.5

    for action in np.flatnonzero(mask[:-1]):
        action = int(action)
        if action in continuation:
            continue
        uav_index, task_slot = divmod(action, 6)
        if task_slot in active_slots or uav_index >= len(uavs) or task_slot >= len(tasks):
            continue
        if valid(uavs[uav_index], 5) and uavs[uav_index][20] > 0.5 and valid(tasks[task_slot], 5) and tasks[task_slot][20] > 0.5:
            return True
    return False


def trigger_decision(obs: Mapping[str, Any], previous_obs: Mapping[str, Any] | None,
                     last_action: int | None, steps_since_replan: int) -> tuple[bool, list[str], dict[str, Any]]:
    flags = {key: bool(value) for key, value in obs.get("trigger_flags", {}).items()}
    conditions: dict[str, Any] = {
        "initial": last_action is None,
        "safety": False,
        "confirmed_fault": flags.get("confirmed_fault", False),
        "link_recovery": flags.get("link_recovery", False),
        "task_arrival": flags.get("task_arrival", False),
        "completion_or_invalidation": flags.get("completion_or_invalidation", False),
        "low_energy": False,
        "max_wait": steps_since_replan >= MAX_REPLAN_INTERVAL,
        "continuation_valid": False,
        "dispatch_opportunity": public_dispatch_opportunity(obs),
    }
    if last_action is not None:
        continuation = {int(value) for value in obs.get("continuation_actions", ())}
        conditions["continuation_valid"] = last_action in continuation
        conditions["safety"] = not conditions["continuation_valid"]
    if previous_obs is not None:
        old_uavs = np.asarray(previous_obs.get("uavs", []), dtype=np.float64)
        new_uavs = np.asarray(obs.get("uavs", []), dtype=np.float64)
        threshold = 0.10 * 9.0
        if old_uavs.ndim == 2 and new_uavs.shape == old_uavs.shape and old_uavs.shape[1] > 2:
            conditions["low_energy"] = bool(np.any((old_uavs[:, 2] > threshold) & (new_uavs[:, 2] <= threshold)))
    priority = ("initial", "safety", "confirmed_fault", "link_recovery", "task_arrival",
                "completion_or_invalidation", "dispatch_opportunity", "low_energy", "max_wait")
    reasons = [name for name in priority if conditions[name]]
    return bool(reasons), reasons, conditions


def checkpoint_path(seed: int) -> Path:
    path = SOURCE_FINAL / f"run/training/seed-{seed}/WD/last-recovery.pt"
    actual = sha256(path)
    if actual != CHECKPOINT_HASHES[seed]:
        raise RuntimeError(f"checkpoint hash mismatch for seed {seed}: {actual}")
    return path


def load_wd(seed: int, env_config: M10Config, device: torch.device):
    path = checkpoint_path(seed)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("group") != "WD" or payload.get("protocol") != JOINT_PROTOCOL:
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    policy = JointGraphPreferencePolicy(env_config, history=True).to(device)
    world = ActionConditionedTemporalWorldModel().to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    world.load_state_dict(payload["world_state_dict"], strict=True)
    policy.eval()
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return policy, world, payload


def probe(policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel,
          obs: Mapping[str, Any], policy_hidden: torch.Tensor | None,
          world_hidden: torch.Tensor | None, preference: torch.Tensor, device: torch.device):
    obs_tensor = _obs_tensor(obs, device)
    mask = torch.as_tensor(_mask_safe(obs["mask"]), dtype=torch.bool, device=device)[None, :]
    p_hidden = torch.zeros((1, 1, 128), dtype=torch.float32, device=device) if policy_hidden is None else policy_hidden
    w_hidden = torch.zeros((1, 128), dtype=torch.float32, device=device) if world_hidden is None else world_hidden
    with torch.no_grad():
        features, pair, next_policy_hidden = policy.encode(obs_tensor, p_hidden)
        candidates, by_action = world.predict_all_candidates(features, w_hidden, obs, use_events=True)
        evaluation = policy.evaluate_encoded(features, pair, preference, candidates, mask)
    return {
        "features": features, "pair": pair, "candidates": candidates,
        "by_action": by_action, "evaluation": evaluation,
        "next_policy_hidden": next_policy_hidden,
    }


def decision_value(policy: JointGraphPreferencePolicy, world: ActionConditionedTemporalWorldModel,
                   obs: Mapping[str, Any], policy_hidden: torch.Tensor | None,
                   world_hidden: torch.Tensor | None, preference: torch.Tensor, device: torch.device) -> np.ndarray:
    return probe(policy, world, obs, policy_hidden, world_hidden, preference, device)["evaluation"]["critic_values"][0].detach().cpu().numpy().astype(np.float32)


def _select_replay_hidden(replay: Mapping[str, Any], obs: Mapping[str, Any],
                          action: int, submit_command: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the same hidden-state transition used by online execution.

    A submitted action is checked against the current command mask.  A
    continuation is checked against the public continuation contract instead;
    it must not be rejected merely because the command mask no longer permits
    a fresh submission.
    """
    action = int(action)
    if submit_command:
        mask = _mask_safe(obs["mask"])
        if not (0 <= action < len(mask) and bool(mask[action])):
            raise RuntimeError("historical new action is not legal during hidden-state replay")
    else:
        continuation_actions = {int(value) for value in obs.get("continuation_actions", ())}
        if action not in continuation_actions:
            raise RuntimeError("historical continuation action is not legal during hidden-state replay")
    by_action = replay["by_action"]
    if action not in by_action:
        raise RuntimeError("historical action has no world-model hidden transition")
    return replay["next_policy_hidden"], by_action[action]["hidden"]


def load_saved_decision_types(ledger_path: Path, expected_steps: int, group: str, seed: int,
                              episode_index: int | None = None,
                              scenario_id: str | None = None) -> list[bool]:
    """Load the explicit actor/submit contract from a saved step ledger."""
    submits: list[bool] = []
    with ledger_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "decision_or_continuation":
                continue
            if row.get("group") != group or int(row.get("seed")) != int(seed):
                raise RuntimeError(f"ledger identity mismatch at line {line_number + 1}")
            if episode_index is not None and int(row.get("episode_index")) != int(episode_index):
                continue
            if scenario_id is not None and str(row.get("scenario_id")) != str(scenario_id):
                continue
            actor = bool(row.get("actor_decision"))
            submit = bool(row.get("command_submitted"))
            if actor != submit:
                raise RuntimeError(f"actor/submit decision contract mismatch at step {row.get('step')}")
            submits.append(submit)
    if len(submits) != int(expected_steps):
        raise RuntimeError(f"saved decision ledger has {len(submits)} rows, expected {expected_steps}")
    return submits


def rebuild_hidden_from_history(policy: JointGraphPreferencePolicy,
                                world: ActionConditionedTemporalWorldModel,
                                history_observations: Sequence[Mapping[str, Any]],
                                history_actions: Sequence[int],
                                history_submits: Sequence[bool],
                                preference: torch.Tensor,
                                device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Rebuild recurrent/cache state after a policy update without stepping the environment."""
    if not (len(history_observations) == len(history_actions) == len(history_submits)):
        raise RuntimeError("history observation/action/decision prefix length mismatch")
    policy_hidden: torch.Tensor | None = None
    world_hidden: torch.Tensor | None = None
    calls = 0
    for historical_obs, historical_action, historical_submit in zip(history_observations, history_actions, history_submits):
        replay = probe(policy, world, historical_obs, policy_hidden, world_hidden, preference, device)
        calls += 1
        policy_hidden, world_hidden = _select_replay_hidden(
            replay, historical_obs, historical_action, historical_submit
        )
    return policy_hidden, world_hidden, calls


def make_sample(probe_result: Mapping[str, Any], obs: Mapping[str, Any], action: int,
                old_log_prob: float, preference: torch.Tensor, policy_hidden: torch.Tensor,
                world_hidden: torch.Tensor, episode_index: int, scenario_id: str,
                window_id: int, behavior_policy_version: int) -> dict[str, Any]:
    evaluation = probe_result["evaluation"]
    return {
        "obs": np.asarray(obs["flat"], dtype=np.float32).copy(),
        "mask": _mask_safe(obs["mask"]).copy(),
        "action": int(action), "old_log_prob": float(old_log_prob),
        "old_values": evaluation["critic_values"][0].detach().cpu().numpy().astype(np.float32),
        "policy_hidden_before": policy_hidden[0, 0].detach().cpu().numpy().astype(np.float32),
        "candidate_features": probe_result["candidates"][0].detach().cpu().numpy().astype(np.float32),
        "preference": preference.detach().cpu().numpy().astype(np.float32),
        "episode_index": int(episode_index), "scenario_id": scenario_id,
        "window_id": int(window_id), "behavior_policy_version": int(behavior_policy_version),
    }


def weighted_update(policy: JointGraphPreferencePolicy, samples: Sequence[dict[str, Any]],
                    windows: Sequence[DecisionWindow], optimizer: torch.optim.Optimizer,
                    config: JointTrainConfig, device: torch.device) -> dict[str, float]:
    advantages_np, returns_np = decision_level_gae(windows, config.gamma, config.gae_lambda)
    _, _, _, preference, _, features, pair, _ = _batch_encode(policy, samples, device)
    candidates = torch.as_tensor(np.stack([item["candidate_features"] for item in samples]), dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.stack([item["mask"] for item in samples]), dtype=torch.bool, device=device)
    evaluation = policy.evaluate_encoded(features, pair, preference, candidates, masks)
    actions = torch.as_tensor([item["action"] for item in samples], dtype=torch.long, device=device)
    old_log_prob = torch.as_tensor([item["old_log_prob"] for item in samples], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
    log_prob = evaluation["distribution"].log_prob(actions)
    ratio = torch.exp(log_prob - old_log_prob)
    scalar_adv = scalarized_preference_advantage(preference, advantages, config)
    clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    actor_loss = -torch.minimum(ratio * scalar_adv, clipped_ratio * scalar_adv).mean()
    value_loss = F.mse_loss(evaluation["critic_values"], returns)
    entropy = evaluation["distribution"].entropy().mean()
    total = actor_loss + config.value_weight * value_loss - config.entropy_weight * entropy
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite semi-Markov PPO loss")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("non-finite semi-Markov PPO gradient")
    optimizer.step()
    if not finite_parameters(policy) or not finite_optimizer(optimizer):
        raise FloatingPointError("non-finite policy or optimizer after semi-Markov update")
    return {
        "policy_loss": float(actor_loss.detach()), "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()), "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        "approx_kl": float((old_log_prob - log_prob).mean().detach()),
        "clip_fraction": float(((ratio - 1.0).abs() > config.clip_epsilon).float().mean().detach()),
        "decision_samples": len(samples), "k_mean": float(np.mean([item.k for item in windows])),
        "k_max": int(max(item.k for item in windows)),
        "weighted_advantage_mean": float(scalar_adv.detach().mean()),
    }


def save_runtime_checkpoint(path: Path, *, run_id: str, group: str, policy: JointGraphPreferencePolicy,
                            world: ActionConditionedTemporalWorldModel, optimizer: torch.optim.Optimizer,
                            env: M10Environment, obs: Mapping[str, Any], policy_hidden: torch.Tensor | None,
                            world_hidden: torch.Tensor | None, pending: Mapping[str, Any] | None,
                            preference_rng: np.random.Generator, episode_preference: torch.Tensor,
                            counters: Mapping[str, Any], identity: Mapping[str, Any],
                            runtime_state: Mapping[str, Any] | None = None) -> str:
    payload = {
        "format": "event-trigger-aware-gppo/1.0.0", "run_id": run_id, "group": group,
        "protocol": JOINT_PROTOCOL, "identity": dict(identity), "counters": dict(counters),
        "persistence": {"transaction_id": str(identity.get("transaction_id", "")),
                        "last_committed_step": int(counters.get("last_committed_step", -1)),
                        "ledger": dict(identity.get("ledger", {}))},
        "policy_state_dict": copy.deepcopy(policy.state_dict()),
        "world_state_dict": copy.deepcopy(world.state_dict()),
        "policy_optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "preference": copy.deepcopy(preference_rng.bit_generator.state)},
        "runtime": {"environment": env, "observation": dict(obs),
                    "policy_hidden": None if policy_hidden is None else policy_hidden.detach().cpu(),
                    "world_hidden": None if world_hidden is None else world_hidden.detach().cpu(),
                    "pending": copy.deepcopy(pending), "episode_preference": episode_preference.detach().cpu(),
                    "public_digest": env.public_snapshot_digest(),
                    **({} if runtime_state is None else copy.deepcopy(dict(runtime_state)))},
    }
    _atomic_torch_save(path, payload)
    return sha256(path)


def _ledger_row(run_id: str, group: str, seed: int, step: int, episode: int,
                scenario_id: str, obs: Mapping[str, Any], next_obs: Mapping[str, Any],
                action: int, actor: bool, submit: bool, info: Mapping[str, Any],
                preference: np.ndarray, policy_version: int, world_version: int,
                k_so_far: int, reward: np.ndarray, event_record: Mapping[str, Any],
                window_id: int | None = None) -> dict[str, Any]:
    return {
        "record_type": "decision_or_continuation", "schema": "event-trigger-semi-markov-ledger/1.0.0",
        "run_id": run_id, "group": group, "seed": seed, "step": step, "episode_index": episode,
        "scenario_id": scenario_id, "time": float(obs["time"]), "actor_decision": actor,
        "action": int(action), "command_submitted": bool(submit), "legal_mask": np.asarray(obs["mask"], dtype=np.bool_).astype(int).tolist(),
        "continuation_action": obs.get("continuation_action"), "continuation_actions": list(obs.get("continuation_actions", ())),
        "preference": preference.tolist(), "reward_vector": reward.tolist(), "k_so_far": int(k_so_far),
        "window_id": None if window_id is None else int(window_id),
        "policy_version": int(policy_version), "world_model_version": int(world_version),
        "next_public_time": float(next_obs["time"]), "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)), "feedback": info.get("feedback"),
        "communication_delta": info.get("communication_delta", []), "execution_feedback": {
            key: info.get(key) for key in ("command_submitted", "command_id", "lease_renewal", "feedback",
                                           "completion_records", "counts", "terminated", "truncated")
        },
        "event_labels": {"names": list(event_record.get("version", "")) if False else None,
                         "labels": np.asarray(event_record["labels"]).tolist(),
                         "mask": np.asarray(event_record["mask"]).tolist(),
                         "reasons": event_record["reasons"]},
    }


class _WindowLedger:
    """Durable decision-window closure records, separate from step ledger."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to append to existing window ledger: {path}")
        self.stream = path.open("xb")

    def append(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=json_default).encode("utf-8") + b"\n"
        self.stream.write(payload)
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def run_training(*, group: str, seed: int, output_dir: Path, env_config: M10Config,
                 train_config: JointTrainConfig, scenarios: Sequence[Any], max_steps: int,
                 max_policy_updates: int, wall_seconds: float, smoke: bool = False,
                 budget: PersistentBudget | None = None,
                 budget_attempt_id: str | None = None,
                 tape_sha256: str = TAPE_HASH,
                 fixed_preference: tuple[float, float] | None = None,
                 resume_from: Path | None = None,
                 resume_ledger: Path | None = None,
                 capture_boundary_checkpoint: bool = False,
                 stop_after_update_boundary: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    policy, world, payload = load_wd(seed, env_config, device)
    policy.train()
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=train_config.policy_lr)
    resume_payload = None if resume_from is None else torch.load(resume_from, map_location=device, weights_only=False)
    if resume_payload is not None:
        if resume_payload.get("group") != group or resume_payload.get("protocol") != JOINT_PROTOCOL:
            raise RuntimeError(f"resume identity mismatch: {resume_from}")
        policy.load_state_dict(resume_payload["policy_state_dict"], strict=True)
        world.load_state_dict(resume_payload["world_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["policy_optimizer_state_dict"])
    suffix = "smoke" if smoke else "formal"
    if budget_attempt_id:
        suffix = str(budget_attempt_id)
    run_id = f"event-trigger-aware-{group}-seed-{seed}-{suffix}-window-v3"
    budget_run_id = f"{group}/seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger = _TrainingLedger(output_dir / "training-ledger.jsonl", run_id=run_id)
    window_ledger = _WindowLedger(output_dir / "decision-window-ledger.jsonl")
    if resume_payload is None:
        preference_rng = np.random.default_rng(seed + 0xA17E)
        preference = masked_normalized_preference(
            np.asarray(fixed_preference, dtype=np.float32) if fixed_preference is not None
            else preference_rng.dirichlet(np.ones(2)), device=device
        )
        scenario_index = 0
        episode_index = 0
        env = M10Environment(env_config, scenarios[scenario_index])
        obs = env.reset()
        history_observations: list[Mapping[str, Any]] = []
        history_actions: list[int] = []
        history_submits: list[bool] = []
        previous_obs: Mapping[str, Any] | None = None
        last_action: int | None = None
        steps_since_replan = MAX_REPLAN_INTERVAL
        policy_hidden = None
        world_hidden = None
        pending: dict[str, Any] | None = None
        active_window_id: int | None = None
        next_window_id = 0
        prev_counts = {"completed": 0, "expired": 0}
        prev_energy = float(env_config.uav_count * env_config.initial_energy)
    else:
        runtime = resume_payload["runtime"]
        resume_rng = resume_payload["rng"]
        random.setstate(resume_rng["python"])
        np.random.set_state(resume_rng["numpy"])
        torch.set_rng_state(resume_rng["torch"])
        preference_rng = np.random.default_rng()
        preference_rng.bit_generator.state = copy.deepcopy(resume_rng["preference"])
        preference = runtime["episode_preference"].to(device)
        env = runtime["environment"]
        obs = runtime["observation"]
        history_observations = copy.deepcopy(runtime.get("history_observations", []))
        history_actions = [int(item) for item in runtime.get("history_actions", [])]
        history_submits = [bool(item) for item in runtime.get("history_submits", [])]
        previous_obs = copy.deepcopy(runtime.get("previous_obs"))
        last_action = runtime.get("last_action")
        steps_since_replan = int(runtime.get("steps_since_replan", MAX_REPLAN_INTERVAL))
        policy_hidden = runtime.get("policy_hidden")
        world_hidden = runtime.get("world_hidden")
        pending = copy.deepcopy(runtime.get("pending"))
        active_window_id = runtime.get("active_window_id")
        next_window_id = int(runtime.get("next_window_id", 0))
        scenario_index = int(runtime.get("scenario_index", 0))
        episode_index = int(runtime.get("episode_index", 0))
        if history_actions and not history_submits:
            if resume_ledger is None:
                raise RuntimeError("resume checkpoint lacks decision types and no saved ledger was supplied")
            history_submits = load_saved_decision_types(
                resume_ledger, len(history_actions), group, seed,
                episode_index=episode_index, scenario_id=env.scenario.tape_id,
            )
        prev_counts = copy.deepcopy(runtime.get("prev_counts", {"completed": 0, "expired": 0}))
        prev_energy = float(runtime.get("prev_energy", env_config.uav_count * env_config.initial_energy))
    completed_samples: list[dict[str, Any]] = []
    completed_windows: list[DecisionWindow] = []
    resume_counters = {} if resume_payload is None else dict(resume_payload.get("counters", {}))
    transitions_logged = int(resume_counters.get("environment_steps", 0))
    steps = int(resume_counters.get("environment_steps", 0))
    policy_updates = int(resume_counters.get("policy_optimizer_steps", 0))
    actor_calls = int(resume_counters.get("actor_calls", 0))
    continuation_steps = int(resume_counters.get("continuation_steps", 0))
    world_calls = int(resume_counters.get("world_forward_calls", 0))
    replay_forward_calls = int(resume_counters.get("replay_forward_calls", 0))
    replay_seconds = float(resume_counters.get("replay_seconds", 0.0))
    trigger_counts: dict[str, int] = {}
    k_values: list[int] = []
    ppo_log: list[dict[str, Any]] = []
    start = time.perf_counter()
    stop_reason = "budget_complete"
    update_requested = False
    boundary_stop = False
    boundary_checkpoint_hash: str | None = None

    def close_pending(next_value: np.ndarray, *, terminated: bool, truncated: bool,
                      reason: str, rollout_boundary: bool, update_id: int | None) -> None:
        """Close one window before any policy-version change."""
        nonlocal pending, active_window_id
        if pending is None:
            return
        sample = pending["sample"]
        reward_sum, k, discount = discounted_window(pending["rewards"], train_config.gamma)
        window = DecisionWindow(
            reward_sum, k, discount, sample["old_values"], np.asarray(next_value, dtype=np.float32),
            bool(terminated), bool(truncated), bool(rollout_boundary),
        )
        completed_samples.append(sample)
        completed_windows.append(window)
        k_values.append(k)
        window_ledger.append({
            "record_type": "window_closed", "window_protocol": WINDOW_PROTOCOL,
            "window_id": int(sample["window_id"]), "start_step": int(pending["start_step"]),
            "end_step": int(steps - 1), "k": int(k),
            "behavior_policy_version": int(sample["behavior_policy_version"]),
            "old_log_prob": float(sample["old_log_prob"]),
            "old_values": np.asarray(sample["old_values"]).tolist(),
            "preference": np.asarray(sample["preference"]).tolist(),
            "bootstrap": {"valid": not bool(terminated), "value": np.asarray(next_value).tolist(),
                          "source": reason, "policy_version": int(sample["behavior_policy_version"])},
            "terminated": bool(terminated), "truncated": bool(truncated),
            "rollout_boundary": bool(rollout_boundary), "close_reason": reason,
            "training_update_id": None if update_id is None else int(update_id),
        })
        pending = None
        active_window_id = None

    def perform_update(*, rollout_start: int) -> None:
        """Apply one update while preserving the live episode/trigger history."""
        nonlocal policy_updates, policy_hidden, world_hidden, update_requested
        nonlocal replay_forward_calls, replay_seconds, boundary_checkpoint_hash
        if not completed_windows:
            update_requested = False
            return
        update_token = budget.reserve("policy_optimizer_calls", run_id=budget_run_id) if budget is not None else None
        try:
            update = weighted_update(policy, completed_samples, completed_windows, optimizer, train_config, device)
        except BaseException as exc:
            if budget is not None and update_token is not None:
                budget.unknown(update_token, f"policy_update:{type(exc).__name__}:{exc}")
            raise
        else:
            if budget is not None and update_token is not None:
                budget.complete(update_token)
        policy_updates += 1
        replay_start = time.perf_counter()
        policy_hidden, world_hidden, replay_calls = rebuild_hidden_from_history(
            policy, world, history_observations, history_actions, history_submits, preference, device
        )
        replay_seconds += time.perf_counter() - replay_start
        replay_forward_calls += replay_calls
        update.update({"rollout_start": rollout_start, "rollout_end": steps - 1,
                       "policy_update": policy_updates, "history_replay_calls": replay_calls})
        ppo_log.append(update)
        if capture_boundary_checkpoint and boundary_checkpoint_hash is None:
            boundary_ledger_range = ledger.flush_rollout()
            boundary_counters = {"environment_steps": steps, "policy_optimizer_steps": policy_updates,
                                 "world_optimizer_steps": 0, "actor_calls": actor_calls,
                                 "continuation_steps": continuation_steps, "world_forward_calls": world_calls,
                                 "replay_forward_calls": replay_forward_calls,
                                 "replay_seconds": replay_seconds,
                                 "pending_window_k": 0 if pending is None else len(pending["rewards"]),
                                 "last_committed_step": steps - 1}
            boundary_txn_id = f"boundary-step-{steps:08d}-policy-{policy_updates:06d}-world-000000"
            boundary_path = output_dir / "boundary-update-checkpoint.pt"
            boundary_checkpoint_hash = save_runtime_checkpoint(
                boundary_path, run_id=run_id, group=group, policy=policy, world=world,
                optimizer=optimizer, env=env, obs=obs, policy_hidden=policy_hidden,
                world_hidden=world_hidden, pending=pending, preference_rng=preference_rng,
                episode_preference=preference, counters=boundary_counters,
                identity={"transaction_id": boundary_txn_id, "ledger": boundary_ledger_range,
                          "source_checkpoint": str(checkpoint_path(seed)),
                          "source_checkpoint_sha256": CHECKPOINT_HASHES[seed],
                          "group": group, "seed": seed, "boundary_checkpoint": True,
                          "training_tape_sha256": tape_sha256, "environment": asdict(env_config),
                          "training": asdict(train_config)},
                runtime_state={"history_observations": history_observations,
                               "history_actions": history_actions,
                               "history_submits": history_submits,
                               "previous_obs": previous_obs,
                               "last_action": last_action,
                               "steps_since_replan": steps_since_replan,
                               "active_window_id": active_window_id,
                               "next_window_id": next_window_id,
                               "scenario_index": scenario_index,
                               "episode_index": episode_index,
                               "prev_counts": prev_counts,
                               "prev_energy": prev_energy},
            )
        completed_samples.clear()
        completed_windows.clear()
        update_requested = False

    while steps < max_steps and time.perf_counter() - start < wall_seconds:
        rollout_start = steps
        rollout_end = min(rollout_start + train_config.rollout_steps, max_steps)
        while steps < rollout_end and time.perf_counter() - start < wall_seconds:
            current = probe(policy, world, obs, policy_hidden, world_hidden, preference, device)
            world_calls += 1
            should, reasons, conditions = (True, ["periodic_policy"], {}) if group == "P_train" else trigger_decision(obs, previous_obs, last_action, steps_since_replan)
            if should:
                if pending is not None:
                    next_value = current["evaluation"]["critic_values"][0].detach().cpu().numpy().astype(np.float32)
                    close_pending(next_value, terminated=False, truncated=False,
                                  reason="next_actor", rollout_boundary=False,
                                  update_id=(policy_updates + 1) if policy_updates < max_policy_updates else None)
                if update_requested and completed_windows:
                    perform_update(rollout_start=rollout_start)
                    if stop_after_update_boundary:
                        boundary_stop = True
                        break
                    current = probe(policy, world, obs, policy_hidden, world_hidden, preference, device)
                    world_calls += 1
                distribution = current["evaluation"]["distribution"]
                action_tensor = distribution.sample()
                action = int(action_tensor.item())
                old_log_prob = float(distribution.log_prob(action_tensor).item())
                if not bool(_mask_safe(obs["mask"])[action]):
                    raise RuntimeError("trigger-aware policy sampled illegal action")
                actor_calls += 1
                trigger_counts[reasons[0] if reasons else "periodic_policy"] = trigger_counts.get(reasons[0] if reasons else "periodic_policy", 0) + 1
                active_window_id = next_window_id
                next_window_id += 1
                pending = {
                    "sample": make_sample(current, obs, action, old_log_prob, preference,
                                          torch.zeros((1, 1, 128), dtype=torch.float32) if policy_hidden is None else policy_hidden,
                                          torch.zeros((1, 128), dtype=torch.float32) if world_hidden is None else world_hidden,
                                          episode_index, env.scenario.tape_id, active_window_id,
                                          policy_updates),
                    "rewards": [], "start_step": steps,
                }
                submit = True
                steps_since_replan = 0
            else:
                action = int(last_action)
                old_log_prob = None
                continuation_steps += 1
                submit = False
                steps_since_replan += 1
            history_observations.append(copy.deepcopy(obs))
            history_actions.append(int(action))
            history_submits.append(bool(submit))
            action_obs = obs
            step_token = budget.reserve("environment_steps", run_id=budget_run_id) if budget is not None else None
            try:
                next_obs, legacy_reward, done, info = env.step(action, submit_command=submit)
            except BaseException as exc:
                if budget is not None and step_token is not None:
                    budget.unknown(step_token, f"env.step:{type(exc).__name__}:{exc}")
                raise
            else:
                if budget is not None and step_token is not None:
                    budget.complete(step_token)
            info = dict(info)
            vector_reward, consequence, _, _ = _vector_reward(info, {"completed": 0, "expired": 0}, float(sum(env_config.uav_count * env_config.initial_energy for _ in [0])), env_config)
            # Persist the reward-difference cache across a resume boundary;
            # resetting it would change the resumed reward relative to U.
            vector_reward, consequence, prev_counts, prev_energy = _vector_reward(info, prev_counts, prev_energy, env_config)
            event_record = build_observed_event_labels(obs, next_obs, initial_energy=env_config.initial_energy,
                                                       urgent_slack=train_config.urgent_slack_steps,
                                                       low_energy_fraction=train_config.low_energy_fraction,
                                                       same_episode=True)
            if pending is not None:
                pending["rewards"].append(vector_reward.astype(np.float32))
            row = _ledger_row(run_id, group, seed, steps, episode_index, env.scenario.tape_id,
                              obs, next_obs, action, should, submit, info, preference.detach().cpu().numpy(),
                              policy_updates, 0, len(pending["rewards"]) if pending else 0,
                              vector_reward, event_record, active_window_id)
            ledger.append(row)
            transitions_logged += 1
            steps += 1
            previous_obs = obs
            obs = next_obs
            policy_hidden, world_hidden = _select_replay_hidden(current, action_obs, action, submit)
            last_action = action
            if done:
                if pending is not None:
                    terminated = bool(info.get("terminated", False))
                    truncated = bool(info.get("truncated", False))
                    if terminated:
                        next_value = np.zeros(2, dtype=np.float32)
                    else:
                        next_value = decision_value(policy, world, next_obs, policy_hidden, world_hidden, preference, device)
                    close_pending(next_value, terminated=terminated, truncated=truncated,
                                  reason="episode_end", rollout_boundary=False,
                                  update_id=policy_updates + 1)
                scenario_index = (scenario_index + 1) % len(scenarios)
                env = M10Environment(env_config, scenarios[scenario_index])
                obs = env.reset()
                previous_obs = None
                last_action = None
                steps_since_replan = MAX_REPLAN_INTERVAL
                policy_hidden = None
                world_hidden = None
                history_observations = []
                history_actions = []
                history_submits = []
                episode_index += 1
                prev_counts = {"completed": 0, "expired": 0}
                prev_energy = float(env_config.uav_count * env_config.initial_energy)
                preference = masked_normalized_preference(
                    np.asarray(fixed_preference, dtype=np.float32) if fixed_preference is not None
                    else preference_rng.dirichlet(np.ones(2)), device=device
                )
        # The rollout boundary only requests an update.  If the current window
        # is still in continuation, keep using the frozen trigger contract until
        # the next real actor boundary; perform_update() is called there before
        # sampling the new action.  A closed window can be updated immediately.
        update_requested = True
        if pending is None and completed_windows and policy_updates < max_policy_updates:
            perform_update(rollout_start=rollout_start)
        ledger_range = ledger.flush_rollout()
        counters = {"environment_steps": steps, "policy_optimizer_steps": policy_updates,
                    "world_optimizer_steps": 0, "actor_calls": actor_calls,
                    "continuation_steps": continuation_steps, "world_forward_calls": world_calls,
                    "replay_forward_calls": replay_forward_calls,
                    "replay_seconds": replay_seconds,
                    "pending_window_k": 0 if pending is None else len(pending["rewards"]),
                    "last_committed_step": steps - 1}
        txn_id = f"txn-step-{steps:08d}-policy-{policy_updates:06d}-world-000000"
        checkpoint = output_dir / "transactions" / f"{txn_id}.pt"
        checkpoint_hash = save_runtime_checkpoint(
            checkpoint, run_id=run_id, group=group, policy=policy, world=world,
            optimizer=optimizer, env=env, obs=obs, policy_hidden=policy_hidden,
            world_hidden=world_hidden, pending=pending, preference_rng=preference_rng,
            episode_preference=preference, counters=counters,
            identity={"transaction_id": txn_id, "ledger": ledger_range,
                      "source_checkpoint": str(checkpoint_path(seed)), "source_checkpoint_sha256": CHECKPOINT_HASHES[seed],
                      "group": group, "seed": seed, "trigger_contract": "public T_dispatch; max interval 1",
                      "training_tape_sha256": tape_sha256, "environment": asdict(env_config),
                      "training": asdict(train_config)},
            runtime_state={"history_observations": history_observations,
                           "history_actions": history_actions,
                           "history_submits": history_submits,
                           "previous_obs": previous_obs,
                           "last_action": last_action,
                           "steps_since_replan": steps_since_replan,
                           "active_window_id": active_window_id,
                           "next_window_id": next_window_id,
                           "scenario_index": scenario_index,
                           "episode_index": episode_index,
                           "prev_counts": prev_counts,
                           "prev_energy": prev_energy},
        )
        verified = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if verified["counters"] != counters or not finite_parameters(policy) or not finite_optimizer(optimizer):
            raise RuntimeError("trigger-aware checkpoint verification failed")
        _atomic_bytes(output_dir / "last-recovery.pt", checkpoint.read_bytes())
        ledger.publish_transaction(checkpoint_path=checkpoint, counters=counters, transaction_id=txn_id, ledger_range=ledger_range)
        if steps >= max_steps:
            stop_reason = "environment_step_budget"
            break
        if boundary_stop:
            stop_reason = "update_boundary_checkpoint"
            break
    if not (output_dir / "ledger-commit.json").is_file():
        raise RuntimeError("no committed transaction")
    if pending is not None:
        window_ledger.append({
            "record_type": "window_pending", "window_protocol": WINDOW_PROTOCOL,
            "window_id": int(pending["sample"]["window_id"]),
            "start_step": int(pending["start_step"]), "end_step": int(steps - 1),
            "k": len(pending["rewards"]),
            "behavior_policy_version": int(pending["sample"]["behavior_policy_version"]),
            "old_log_prob": float(pending["sample"]["old_log_prob"]),
            "bootstrap": {"valid": False, "value": None, "source": "pending_budget_stop"},
            "close_reason": "pending_budget_stop", "training_update_id": None,
        })
    ledger.close()
    window_ledger.close()
    untrained_closed_windows = len(completed_windows)
    summary = {
        "status": "completed" if steps >= max_steps else "stopped", "run_id": run_id,
        "group": group, "seed": seed, "environment_steps": steps,
        "policy_optimizer_steps": policy_updates, "world_optimizer_steps": 0,
        "actor_calls": actor_calls, "continuation_steps": continuation_steps,
        "world_forward_calls": world_calls, "replay_forward_calls": replay_forward_calls,
        "replay_seconds": replay_seconds, "k_count": len(k_values),
        "untrained_closed_windows": untrained_closed_windows,
        "k_distribution": {str(k): k_values.count(k) for k in sorted(set(k_values))},
        "trigger_counts": trigger_counts, "wall_seconds": time.perf_counter() - start,
        "stop_reason": stop_reason, "checkpoint_sha256": sha256(output_dir / "last-recovery.pt"),
        "boundary_checkpoint_sha256": boundary_checkpoint_hash,
        "ppo_updates": ppo_log, "source_checkpoint_sha256": CHECKPOINT_HASHES[seed],
        "world_parameter_updates": 0, "world_parameters_frozen": True,
        "decision_level_contract": {"continuation_has_log_prob": False, "gamma_power": "gamma**k",
                                    "gae_coefficient": "gamma**k*lambda", "rollout_boundary_cuts_gae": True,
                                    "window_protocol": WINDOW_PROTOCOL,
                                    "boundary_rule": "rollout boundary requests update; preserve trigger/history until next actor boundary, close old window, update, replay public prefix, then sample",
                                    "history_preserved_across_policy_update": True,
                                    "cross_policy_version_windows": 0,
                                    "window_ledger": "decision-window-ledger.jsonl"},
        "budget": None if budget is None else budget.snapshot(),
    }
    write_json(output_dir / "training-summary.json", summary)
    write_json(output_dir / "run-status.json", {"status": summary["status"], "summary": summary})
    return summary


def evaluate_checkpoint(checkpoint: Path, group: str, seed: int, scenarios: Sequence[Any],
                        env_config: M10Config, preference: tuple[float, float], mode: str) -> dict[str, Any]:
    device = torch.device("cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = JointGraphPreferencePolicy(env_config, history=True).to(device)
    world = ActionConditionedTemporalWorldModel().to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True); world.load_state_dict(payload["world_state_dict"], strict=True)
    policy.eval(); world.eval()
    p = masked_normalized_preference(preference, device=device)
    rows = []
    for scenario in scenarios:
        env = M10Environment(env_config, scenario); obs = env.reset(); previous = None; last_action = None
        policy_hidden = None; world_hidden = None; since = MAX_REPLAN_INTERVAL; done = False; steps = 0; total_reward = np.zeros(2, dtype=np.float64)
        actors = cont = 0; ks: list[int] = []; pending_k = 0; trigger_count = 0; info: Mapping[str, Any] = {}
        previous_counts = {"completed": 0, "expired": 0}; previous_energy = float(env_config.uav_count * env_config.initial_energy)
        illegal_actions = 0; noop_opportunities = 0; noop_selected = 0; command_submitted = 0; communication_messages = 0
        feedback_counts: dict[str, int] = {}; security_violations = 0
        while not done and steps < 128:
            result = probe(policy, world, obs, policy_hidden, world_hidden, p, device)
            if mode == "periodic":
                actor = True; reasons = ["periodic_policy"]
            else:
                actor, reasons, _ = trigger_decision(obs, previous, last_action, since)
            if actor:
                action = int(torch.argmax(result["evaluation"]["distribution"].logits, dim=-1).item()); actors += 1
                trigger_count += int(mode == "triggered"); since = 0
                if pending_k: ks.append(pending_k); pending_k = 0
            else:
                action = int(last_action); cont += 1; since += 1; pending_k += 1
            mask = _mask_safe(obs["mask"])
            continuation_allowed = (not actor) and action in {int(value) for value in obs.get("continuation_actions", ())}
            illegal_actions += int(not bool(mask[action]) and not continuation_allowed)
            noop_opportunities += int(bool(mask[:-1].any()))
            noop_selected += int(action == env_config.action_count - 1 and bool(mask[:-1].any()))
            next_obs, reward, done, info = env.step(action, submit_command=actor)
            vector_reward, _, previous_counts, previous_energy = _vector_reward(info, previous_counts, previous_energy, env_config)
            total_reward += vector_reward.astype(np.float64)
            command_submitted += int(bool(info.get("command_submitted", False)))
            communication_messages += len(info.get("communication_delta", []))
            feedback = str(info.get("feedback", "unknown")); feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
            security_violations += len(info.get("security_violations", [])) if isinstance(info.get("security_violations", []), (list, tuple)) else int(bool(info.get("security_violation", False)))
            previous, obs = obs, next_obs; policy_hidden = result["next_policy_hidden"]; world_hidden = result["by_action"][action]["hidden"]; last_action = action; steps += 1
        counts = info.get("counts", {})
        completion_records = info.get("completion_records", {}) if info else {}
        rows.append({"tape_id": scenario.tape_id, "condition": getattr(scenario, "communication", None).__dict__ if getattr(scenario, "communication", None) else None,
                     "completed": counts.get("completed"), "expired": counts.get("expired"), "steps": steps,
                     "actor_calls": actors, "continuation_steps": cont, "trigger_count": trigger_count,
                     "k_distribution": {str(k): ks.count(k) for k in sorted(set(ks))}, "reward_vector": total_reward.tolist(),
                     "feedback_counts": feedback_counts, "illegal_actions": illegal_actions,
                     "noop_opportunities": noop_opportunities, "noop_selected": noop_selected,
                     "command_submitted": command_submitted, "communication_proxy_messages": communication_messages,
                     "security_violations": security_violations,
                     "host_confirmed": sum(record.get("host_confirmation_time") is not None for record in completion_records.values()),
                     "host_on_time": sum(record.get("host_confirmation_before_deadline") is True for record in completion_records.values()),
                     "task_states": info.get("tasks", {}) if info else {}})
    return {"checkpoint_sha256": sha256(checkpoint), "mode": mode, "seed": seed, "group": group, "preference": preference, "episodes": rows}


def hand_fixtures(out: Path) -> dict[str, Any]:
    rows = [
        ([1.0, 0.0], 0.99), ([2.0, 1.0], 0.99), ([3.0, -1.0], 0.99),
    ]
    reward, k, discount = discounted_window([row[0] for row in rows], 0.99)
    terminal = DecisionWindow(reward, k, discount, np.zeros(2), np.zeros(2), True)
    nonterminal = DecisionWindow(reward, k, discount, np.zeros(2), np.ones(2), False, rollout_boundary=True)
    adv, ret = decision_level_gae([terminal, nonterminal], 0.99, 0.95)
    report = {"status": "passed", "k3": {"reward_sum": reward.tolist(), "k": k, "gamma_k": discount},
              "terminal_no_bootstrap": bool(np.allclose(adv[0], ret[0])),
              "gae_shapes": [list(adv.shape), list(ret.shape)],
              "contract": validate_window_contract([terminal, nonterminal], gamma=0.99)}
    write_json(out / "hand-fixtures.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("correctness", "formal", "evaluation", "all"), default="all")
    parser.add_argument("--formal-root", type=Path, default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    out = args.out.resolve(); out.mkdir(parents=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    env_config = load_config(); tapes = load_tapes()
    train_scenarios = interleave([tapes["train"][condition] for condition in CONDITIONS])
    validation = {condition: tapes["validation"][condition][:16] for condition in CONDITIONS}
    write_json(out / "protocol-and-budget.json", {"schema": "event-trigger-aware-gppo/1.0.0", "protocol": JOINT_PROTOCOL,
        "source_root": str(ROOT), "source_snapshot_manifest": "generated-after-copy", "checkpoint_hashes": CHECKPOINT_HASHES,
        "formal_tapes_sha256": TAPE_HASH, "groups": {"P_train": "periodic policy decisions", "T_train": "public T_dispatch decisions; continuation no actor sample"},
        "semi_markov": {"R_k": "sum_j gamma**j r[t+j]", "bootstrap": "gamma**k", "gae": "gamma**k*lambda", "continuation_log_prob": "not applicable"},
        "correctness_budget": {"environment_steps_total": 1024, "smoke_each": 512},
        "formal_budget": {"seeds": SEEDS, "environment_steps_each": 8192, "total_environment_steps": 49152, "wall_seconds": 4*3600},
        "validation_budget": {"episodes": 432, "max_environment_steps": 12000, "wall_seconds": 3600},
        "final_test_read": False, "environment": asdict(env_config), "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__}})
    hand_fixtures(out)
    summaries: dict[str, Any] = {}
    if args.phase in ("correctness", "all"):
        for group in ("P_train", "T_train"):
            summaries[f"smoke/{group}"] = run_training(group=group, seed=1101, output_dir=out / "correctness" / group,
                env_config=env_config, train_config=JointTrainConfig(seed=1101, rollout_steps=128), scenarios=train_scenarios,
                max_steps=512, max_policy_updates=4, wall_seconds=1800, smoke=True)
        write_json(out / "correctness-summary.json", {"status": "passed", "runs": summaries})
    if args.phase in ("formal", "all"):
        for seed in SEEDS:
            for group in ("P_train", "T_train"):
                run_dir = out / "training" / f"seed-{seed}" / group
                summaries[f"formal/{seed}/{group}"] = run_training(group=group, seed=seed, output_dir=run_dir,
                    env_config=env_config, train_config=JointTrainConfig(seed=seed, rollout_steps=128), scenarios=train_scenarios,
                    max_steps=8192, max_policy_updates=64, wall_seconds=4*3600, smoke=False)
                write_json(out / "training-progress.json", {"status": "running", "runs": summaries})
        write_json(out / "formal-summary.json", {"status": "completed", "runs": summaries})
    if args.phase in ("evaluation", "all"):
        if not all((out / "training" / f"seed-{seed}" / group / "last-recovery.pt").is_file() for seed in SEEDS for group in ("P_train", "T_train")):
            raise SystemExit("evaluation requires all six formal runs")
        eval_rows = []
        for seed in SEEDS:
            p_ck = out / "training" / f"seed-{seed}" / "P_train" / "last-recovery.pt"
            t_ck = out / "training" / f"seed-{seed}" / "T_train" / "last-recovery.pt"
            for condition in CONDITIONS:
                scen = validation[condition]
                eval_rows.append(evaluate_checkpoint(p_ck, "P_train", seed, scen, env_config, PREFERENCE, "periodic"))
                eval_rows.append(evaluate_checkpoint(p_ck, "P_train", seed, scen, env_config, PREFERENCE, "triggered"))
                eval_rows.append(evaluate_checkpoint(t_ck, "T_train", seed, scen, env_config, PREFERENCE, "triggered"))
        write_json(out / "validation-results.json", {"status": "completed", "episodes": eval_rows, "episode_count": sum(len(item["episodes"]) for item in eval_rows), "final_test_read": False})
    write_json(out / "run-status.json", {"status": "completed", "summaries": summaries, "final_test_read": False})
    print(json.dumps({"status": "completed", "out": str(out), "summaries": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
