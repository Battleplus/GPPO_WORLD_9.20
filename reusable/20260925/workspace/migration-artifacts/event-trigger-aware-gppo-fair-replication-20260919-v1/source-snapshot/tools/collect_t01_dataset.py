"""Collect a strict-split T-01 dataset from the frozen GPPO environment."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import torch

# Direct script execution places ``tools/`` rather than the repository root on
# sys.path. Add the project root before importing the local package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gppo_world.contracts import EvidenceItem, ExecutionRecord, Transition, snapshot_from_gppo
from gppo_world.dataset import SPLITS, audit_manifest, sha256_file, write_manifest
from gppo_world.recorder import TransitionRecorder
from gppo_world.registry import FEATURE_REGISTRY, SCHEMA_VERSION


PROFILES = ("normal", "single", "sequential", "overlap", "burst", "long_gap", "weak_comm")
MASTER_SEEDS = {"train": 91001, "validation": 92001, "test": 93001}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_evidence(env: Any, decision_time: float) -> tuple[EvidenceItem, ...]:
    bridge = getattr(env, "runtime_bridge", None)
    if bridge is None:
        return ()
    items: list[EvidenceItem] = []
    for event in bridge.adapter.belief.confirmed_events:
        received = event.received_at if event.received_at is not None else event.confirmed_at
        if received is None or float(received) > decision_time:
            continue
        items.append(
            EvidenceItem(
                source=str(event.source_event),
                signal_type=str(event.event_type.value),
                received_at=float(received),
                payload={
                    "event_id": str(event.event_id),
                    "affected_uavs": list(event.affected_uavs),
                    "affected_regions": list(event.affected_regions),
                    "affected_targets": list(event.affected_targets),
                    "severity": float(event.severity),
                    "status": str(event.status.value),
                    "evidence_count": len(event.evidence_ids),
                    "state_version": int(event.state_version),
                },
            )
        )
    return tuple(sorted(items, key=lambda item: (item.received_at, item.source, item.signal_type)))


def _costs(info: dict[str, Any]) -> dict[str, float]:
    after = info.get("reward_trace", {}).get("after", {})
    keys = ("uncovered", "distance", "load_gap", "switches", "recovery_delay", "constraint_violation", "total")
    return {key: float(after.get(key, 0.0)) for key in keys}


def _make_tape(RandomEventAllocationEnv, EventTape, initial_seed: int, event_seed: int, profile: str):
    if profile == "normal":
        return EventTape(initial_seed=initial_seed, event_seed=event_seed, mode="single", events=())
    mode = profile if profile in {"single", "sequential", "overlap", "burst"} else "sequential"
    env = RandomEventAllocationEnv(
        initial_seed=initial_seed,
        event_seed=event_seed,
        mode=mode,
        events_per_episode=4,
    )
    try:
        env.reset(seed=initial_seed)
        tape = env.event_tape
        assert tape is not None
    finally:
        env.close()
    if profile == "long_gap":
        events = tuple(
            replace(event, observed_at=max(event.observed_at, event.occurred_at + 8.0 + 2.0 * index))
            for index, event in enumerate(tape.events)
        )
        tape = EventTape(initial_seed=tape.initial_seed, event_seed=tape.event_seed, mode=tape.mode, events=events)
    return tape


def _train_gppo(RandomEventAllocationEnv, PPOConfig, PPOTrainer, checkpoint: Path, steps: int, seed: int):
    env = RandomEventAllocationEnv(
        initial_seed=seed,
        event_seed=seed * 1009,
        mode="sequential",
        events_per_episode=5,
        max_decisions=100,
    )
    config = PPOConfig(
        seed=seed,
        device="cpu",
        rollout_steps=64,
        minibatch_size=32,
        update_epochs=2,
        learning_rate=2e-4,
    )
    trainer = PPOTrainer(env, variant="GPPO-Adaptive", config=config)
    trainer.train(steps)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(checkpoint, extra={"purpose": "T-01 behavior coverage", "source_commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9"})
    history_path = checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(trainer.history, indent=2, sort_keys=True), encoding="utf-8")
    env.close()
    return trainer.model, history_path


def _load_or_train_gppo(RandomEventAllocationEnv, PPOConfig, PPOTrainer, checkpoint: Path, steps: int, seed: int):
    history_path = checkpoint.with_suffix(".history.json")
    if checkpoint.is_file() and history_path.is_file():
        env = RandomEventAllocationEnv(
            initial_seed=seed,
            event_seed=seed * 1009,
            mode="sequential",
            events_per_episode=5,
            max_decisions=100,
        )
        trainer, _ = PPOTrainer.load(checkpoint, env=env, device="cpu")
        model = trainer.model
        env.close()
        if model is None or trainer.total_steps != steps:
            raise RuntimeError("existing T-01 checkpoint does not match requested training steps")
        return model, history_path
    return _train_gppo(RandomEventAllocationEnv, PPOConfig, PPOTrainer, checkpoint, steps, seed)


def _collect_episode(
    *,
    RandomEventAllocationEnv,
    ActionSubmission,
    DetectorConfig,
    policy,
    behavior_policy: str,
    split: str,
    profile: str,
    tape_id: str,
    tape,
    seed: int,
    recorder: TransitionRecorder,
) -> dict[str, Any]:
    env = RandomEventAllocationEnv(
        initial_seed=tape.initial_seed,
        event_seed=tape.event_seed,
        mode=tape.mode,
        # Upstream constructs a scheduler even when a supplied tape is used;
        # it requires a positive count. The empty normal tape is still valid.
        events_per_episode=max(1, len(tape.events)),
        event_tape=tape,
        max_decisions=100,
        max_time=240.0,
    )
    graph, _ = env.reset(seed=tape.initial_seed)
    if profile == "weak_comm" and env.runtime_bridge is not None:
        env.runtime_bridge.detector.config = DetectorConfig(
            loss_rate=0.15,
            duplicate_rate=0.20,
            false_positive_rate=0.0,
            out_of_order_max_delay=0.0,
        )
    episode_id = f"{split}/{behavior_policy}/{tape_id}"
    action_counts: Counter[int] = Counter()
    event_types = Counter(event.event_type.value for event in tape.events)
    transition_count = 0
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        ctx = env.begin_decision()
        graph = ctx.graph
        decision_time = float(env.current_time)
        evidence = _safe_evidence(env, decision_time)
        if behavior_policy == "gppo":
            action, _, _, _ = policy.act(graph, deterministic=True)
        else:
            action = policy.select_action(env, graph, deterministic=behavior_policy != "random_legal")
        if not bool(graph.action_mask[action].item()):
            raise RuntimeError(f"{behavior_policy} proposed illegal action {action}")
        submission = ActionSubmission.from_decision(action, ctx)
        graph_after, reward, terminated, truncated, info = env.submit_action(submission)
        if info.get("stale_decision", False):
            raise RuntimeError("collector does not permit stale transitions")
        executed = int(info.get("repaired_action", action))
        transition = Transition(
            episode_id=episode_id,
            scenario_id=profile,
            tape_id=tape_id,
            behavior_policy=behavior_policy,
            seed=seed,
            step=transition_count,
            decision_time=decision_time,
            next_decision_time=float(env.current_time),
            graph_t=snapshot_from_gppo(graph),
            evidence_t=evidence,
            execution=ExecutionRecord(
                proposed_action=int(action),
                executed_action=executed,
                accepted=True,
                graph_version=ctx.graph_version,
                action_version=ctx.action_version,
                status="committed",
            ),
            reward=float(reward),
            costs=_costs(info),
            graph_tp1=snapshot_from_gppo(graph_after),
            continuation=not bool(terminated),
        )
        recorder.append(transition)
        transition_count += 1
        total_reward += float(reward)
        action_counts[executed] += 1
        if transition_count > 100:
            raise RuntimeError("episode exceeded decision budget")
    bridge = getattr(env, "runtime_bridge", None)
    evidence_count = 0 if bridge is None else len(bridge.adapter.belief.confirmed_events)
    observation_count = 0 if bridge is None else int(bridge.get_observation_count())
    communication_trigger_count = int(env.communication_trigger_count)
    communication_bytes = int(env.communication_bytes)
    env.close()
    return {
        "episode_id": episode_id,
        "split": split,
        "scenario_id": profile,
        "tape_id": tape_id,
        "seed": seed,
        "event_seed": tape.event_seed,
        "behavior_policy": behavior_policy,
        "transition_count": transition_count,
        "total_reward": total_reward,
        "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "event_types": dict(sorted(event_types.items())),
        "evidence_count": evidence_count,
        "observation_count": observation_count,
        "communication_trigger_count": communication_trigger_count,
        "communication_bytes": communication_bytes,
        "tape_sha256": _sha256_bytes(tape.to_bytes()),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tapes-per-profile", type=int, default=2)
    parser.add_argument("--gppo-steps", type=int, default=512)
    parser.add_argument("--training-seed", type=int, default=1101)
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        parser.error("the frozen GPPO protocol permits training only on Python 3.10/3.11")

    source_root = args.source_root.resolve()
    ppo_root = source_root / "ppo_allocation"
    os.chdir(ppo_root)
    sys.path.insert(0, str(ppo_root))
    from random_event.baselines import GreedyCostPolicy, MaskedRandomPolicy  # noqa: PLC0415
    from random_event.environment import ActionSubmission, RandomEventAllocationEnv  # noqa: PLC0415
    from random_event.events import EventTape  # noqa: PLC0415
    from random_event.runtime_bridge import DetectorConfig  # noqa: PLC0415
    from random_event.trainer import PPOConfig, PPOTrainer  # noqa: PLC0415

    output = args.output_dir.resolve()
    dataset_dir = output / "dataset"
    checkpoint = output / "checkpoints" / f"gppo_adaptive_seed{args.training_seed}_step{args.gppo_steps}.pt"
    model, history_path = _load_or_train_gppo(
        RandomEventAllocationEnv, PPOConfig, PPOTrainer, checkpoint, args.gppo_steps, args.training_seed
    )
    assert model is not None

    recorders = {split: TransitionRecorder() for split in SPLITS}
    episodes: list[dict[str, Any]] = []
    tape_records: list[dict[str, Any]] = []
    for split in SPLITS:
        rng = random.Random(MASTER_SEEDS[split])
        for profile in PROFILES:
            for tape_index in range(args.tapes_per_profile):
                initial_seed = rng.getrandbits(31)
                event_seed = rng.getrandbits(63)
                tape = _make_tape(RandomEventAllocationEnv, EventTape, initial_seed, event_seed, profile)
                tape_hash = _sha256_bytes(tape.to_bytes())
                tape_id = f"{split}-{profile}-{tape_index:02d}-{tape_hash[:12]}"
                tape_records.append(
                    {
                        "split": split,
                        "profile": profile,
                        "tape_id": tape_id,
                        "initial_seed": initial_seed,
                        "event_seed": event_seed,
                        "event_count": len(tape.events),
                        "event_types": [event.event_type.value for event in tape.events],
                        "max_observation_delay": max(
                            (float(event.observed_at - event.occurred_at) for event in tape.events),
                            default=0.0,
                        ),
                        "communication_profile": (
                            {"loss_rate": 0.15, "duplicate_rate": 0.20, "out_of_order_max_delay": 0.0}
                            if profile == "weak_comm"
                            else {"loss_rate": 0.0, "duplicate_rate": 0.0, "out_of_order_max_delay": 0.0}
                        ),
                        "sha256": tape_hash,
                    }
                )
                policies = {
                    "random_legal": MaskedRandomPolicy(seed=initial_seed ^ 0xA5A5),
                    "greedy": GreedyCostPolicy(),
                    "gppo": model,
                }
                for policy_name, policy in policies.items():
                    episodes.append(
                        _collect_episode(
                            RandomEventAllocationEnv=RandomEventAllocationEnv,
                            ActionSubmission=ActionSubmission,
                            DetectorConfig=DetectorConfig,
                            policy=policy,
                            behavior_policy=policy_name,
                            split=split,
                            profile=profile,
                            tape_id=tape_id,
                            tape=tape,
                            seed=initial_seed,
                            recorder=recorders[split],
                        )
                    )

    files = {}
    for split, recorder in recorders.items():
        path = recorder.write_jsonl(dataset_dir / f"{split}.jsonl")
        files[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "transitions": len(recorder.items),
        }
    action_coverage: Counter[str] = Counter()
    event_distribution: Counter[str] = Counter()
    for episode in episodes:
        action_coverage.update(episode["action_counts"])
    for tape in tape_records:
        event_distribution.update(tape["event_types"])
    manifest: dict[str, Any] = {
        "manifest_version": "gppo-world-dataset/0.1.0",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/Battleplus/GPPO-8.29",
        "source_commit": "2a9bb9f87b9d543df144f4d108ba970c924151f9",
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "registry_sha256": FEATURE_REGISTRY.sha256(),
        "split_master_seeds": MASTER_SEEDS,
        "tapes_per_profile": args.tapes_per_profile,
        "profiles": list(PROFILES),
        "behavior_policies": ["random_legal", "greedy", "gppo"],
        "gppo_behavior_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "training_seed": args.training_seed,
            "decision_steps": args.gppo_steps,
            "history_path": str(history_path),
            "history_sha256": sha256_file(history_path),
            "historical_50k_checkpoint": False,
        },
        "files": files,
        "tapes": tape_records,
        "episodes": episodes,
        "action_coverage": dict(sorted(action_coverage.items(), key=lambda item: int(item[0]))),
        "event_distribution": dict(sorted(event_distribution.items())),
        "truth_only_online_field_count": 0,
    }
    manifest["audit"] = audit_manifest(manifest)
    manifest_path = write_manifest(output / "dataset-manifest.json", manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "audit": manifest["audit"],
        "files": files,
    }, indent=2, sort_keys=True))
    return 0 if manifest["audit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
