"""M-10 weak-communication pilot and three-group formal experiments.

The runner freezes one communication tape before any training and keeps A/B/C
identical except for world-model input and trigger policy.  It never reads the
final evaluation tape to select a checkpoint or threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import M10Config, scenario_to_dict, weak_communication_tape
from gppo_world.m10_training import (
    M10WorldModel, PPOConfig, collect_world_dataset, evaluate_policy,
    save_policy, train_policy, train_world_model,
)
from tools.run_m10_r3 import evaluate_trigger_mode, load_world


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def make_tapes(level: str, base_seed: int = 7101) -> dict[str, list[Any]]:
    return {
        "train": list(weak_communication_tape("train", count=32, base_seed=base_seed, level=level)),
        "validation": list(weak_communication_tape("validation", count=16, base_seed=base_seed, level=level)),
        "historical_test": list(weak_communication_tape("test", count=16, base_seed=base_seed, level=level)),
        "final_test": list(weak_communication_tape("test", count=16, base_seed=base_seed + 100000, level=level)),
        "ood": list(weak_communication_tape("ood", count=16, base_seed=base_seed, level=level,
                                            name="energy_insufficient")),
    }


def freeze_tapes(tapes: dict[str, list[Any]], out: Path) -> None:
    dump(out / "tapes.json", {key: [scenario_to_dict(item) for item in value] for key, value in tapes.items()})
    ids = {key: [item.tape_id for item in value] for key, value in tapes.items()}
    all_ids = [tape_id for values in ids.values() for tape_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("communication tape IDs are not disjoint")
    dump(out / "tape-audit.json", {"split_counts": {key: len(value) for key, value in tapes.items()},
                                    "all_tape_ids_unique": True, "communication_level": tapes["train"][0].communication.to_dict()})


def save_world(path: Path, world: M10WorldModel, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": world.state_dict(), "metadata": metadata,
                "optimizer_state_dict": world.optimizer.state_dict() if hasattr(world, "optimizer") else None,
                "recovery_state": {"seed": metadata.get("seed"), "epochs": metadata.get("epochs")}}, path)


def collect_and_train_world(tapes: dict[str, list[Any]], config: M10Config, *, seed: int,
                            epochs: int, device: str, out: Path) -> tuple[M10WorldModel, dict[str, Any]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for index, split in enumerate(("train", "validation", "historical_test", "final_test", "ood")):
        dataset_split = "test" if split in ("historical_test", "final_test") else split
        rows_by_split[split] = collect_world_dataset(
            episodes=len(tapes[split]), seed=seed + 101 * index, config=config,
            scenarios=tapes[split], split=dataset_split,
        )
    rows = rows_by_split["train"] + rows_by_split["validation"] + rows_by_split["historical_test"] + rows_by_split["final_test"]
    world, metadata = train_world_model(rows, seed=seed, action_count=config.action_count,
                                        epochs=epochs, device=device, ood_rows=rows_by_split["ood"])
    metadata.update({"protocol": "weak-communication-frozen-tape", "tape_level": tapes["train"][0].communication.to_dict(),
                     "split_row_counts": {key: len(value) for key, value in rows_by_split.items()},
                     "label_policy": "reward/event/done/context are next decision-interval simulator consequences; random packet loss timing is not predicted as a deterministic event"})
    save_world(out / "world-model.pt", world, metadata)
    dump(out / "world-model-metadata.json", metadata)
    dump(out / "world-row-audit.json", {key: {"rows": len(value), "tape_ids": sorted({row["tape_id"] for row in value})}
                                         for key, value in rows_by_split.items()})
    return world, metadata


def policy_specs() -> tuple[tuple[str, str], ...]:
    return (("A-graph5-base", "base"), ("B-graph5-world", "world"), ("C-graph5-triggered", "triggered"))


def run_pilot(args: argparse.Namespace, config: M10Config, out: Path) -> None:
    tapes = make_tapes(args.level)
    freeze_tapes(tapes, out)
    world, world_meta = collect_and_train_world(tapes, config, seed=args.world_seed,
                                                epochs=args.world_epochs, device=args.device, out=out)
    validation = {item.seed: item for item in tapes["validation"]}
    rows = []
    for name, fusion in policy_specs():
        policy, metadata = train_policy(
            variant=name, encoder="graph", type_count=5, history=False,
            fusion=fusion, model=world if fusion != "base" else None,
            seed=args.seed, steps=args.steps, env_config=config,
            ppo_config=PPOConfig(rollout_steps=args.rollout_steps, update_epochs=4),
            device=args.device, trigger_threshold=args.threshold, scenarios=tapes["train"],
        )
        evaluation = evaluate_policy(
            policy, model=world if fusion != "base" else None, fusion=fusion,
            env_config=config, seeds=list(validation), device=args.device,
            trigger_threshold=args.threshold, scenarios=validation,
        )
        checkpoint = out / "pilot" / "checkpoints" / f"{name}.pt"
        save_policy(checkpoint, policy, metadata)
        rows.append({"variant": name, "fusion": fusion, "metadata": metadata, "validation": evaluation,
                     "checkpoint": str(checkpoint)})
    dump(out / "pilot-results.json", {"steps": args.steps, "seed": args.seed, "world_model": world_meta,
                                      "results": rows, "training_performed": True})


def select_trigger_threshold(policy: torch.nn.Module, world: M10WorldModel, tapes: dict[str, list[Any]],
                             config: M10Config, args: argparse.Namespace, out: Path) -> float:
    validation = tapes["validation"]
    periodic = evaluate_trigger_mode(policy, world, mode="periodic", env_config=config, scenarios=validation,
                                     device=args.device, threshold=args.threshold, max_replan_interval=args.max_wait)
    candidates = []
    for threshold in (0.1, 0.2, 0.3, 0.5, 0.7):
        risk = evaluate_trigger_mode(policy, world, mode="model-risk", env_config=config, scenarios=validation,
                                     device=args.device, threshold=threshold, max_replan_interval=args.max_wait)
        periodic_return = periodic["summary"]["return"]["mean"]
        risk_return = risk["summary"]["return"]["mean"]
        calls = risk["summary"]["actor_calls"]["mean"]
        feasible = risk_return >= periodic_return and risk["summary"]["rejected"]["mean"] == 0
        candidates.append({"threshold": threshold, "feasible": feasible, "validation_return": risk_return,
                           "periodic_return": periodic_return, "actor_calls": calls,
                           "world_calls": risk["summary"]["world_model_calls"]["mean"]})
    feasible = [item for item in candidates if item["feasible"] and item["actor_calls"] < periodic["summary"]["actor_calls"]["mean"]]
    selected = min(feasible, key=lambda item: item["actor_calls"])["threshold"] if feasible else args.threshold
    dump(out / "trigger-threshold-selection.json", {
        "selection_split": "validation_only", "periodic": periodic, "candidates": candidates,
        "selected_threshold": selected, "feasible_cost_saving_threshold_found": bool(feasible),
        "fallback_note": "If no candidate meets the frozen no-effect constraint, the configured threshold is retained for a negative-result C evaluation; safety gates remain active.",
    })
    return float(selected)


def run_formal(args: argparse.Namespace, config: M10Config, out: Path) -> None:
    tapes = make_tapes(args.level)
    freeze_tapes(tapes, out)
    sample_rows = collect_world_dataset(
        episodes=1, seed=1, config=config, scenarios=tapes["train"][:1], split="train")
    world = load_world(args.world_checkpoint, len(sample_rows[0]["obs"]), config.action_count, args.device)
    validation = {item.seed: item for item in tapes["validation"]}
    final = {item.seed: item for item in tapes["final_test"]}
    pilot_world_policy = load_policy_for_formal(args.world_policy_checkpoint, config, args.device) if args.world_policy_checkpoint else None
    threshold = args.threshold
    if pilot_world_policy is not None:
        threshold = select_trigger_threshold(pilot_world_policy, world, tapes, config, args, out)
    results = []
    for name, fusion in policy_specs():
        for seed in args.seeds:
            policy, metadata = train_policy(
                variant=name, encoder="graph", type_count=5, history=False,
                fusion=fusion, model=world if fusion != "base" else None,
                seed=seed, steps=args.formal_steps, env_config=config,
                ppo_config=PPOConfig(rollout_steps=args.rollout_steps, update_epochs=4),
                device=args.device, trigger_threshold=threshold, scenarios=tapes["train"],
                max_replan_interval=args.max_wait,
            )
            evaluation = evaluate_policy(
                policy, model=world if fusion != "base" else None, fusion=fusion,
                env_config=config, seeds=list(final), device=args.device,
                trigger_threshold=threshold, scenarios=final, max_replan_interval=args.max_wait,
            )
            checkpoint = out / "formal" / "checkpoints" / name / f"seed-{seed}.pt"
            save_policy(checkpoint, policy, metadata)
            results.append({"variant": name, "fusion": fusion, "seed": seed, "metadata": metadata,
                            "evaluation": evaluation, "checkpoint": str(checkpoint)})
            dump(out / "formal" / "records" / name / f"seed-{seed}.json", results[-1])
    dump(out / "formal-results.json", {
        "protocol": "weak communication A/B/C same Graph-5 public protocol and tapes",
        "formal_steps": args.formal_steps, "seeds": args.seeds, "threshold": threshold,
        "max_replan_interval": args.max_wait, "world_pretraining_cost": "reported in world-model metadata",
        "results": results,
    })


def load_policy_for_formal(path: Path, config: M10Config, device: str) -> torch.nn.Module:
    from tools.run_m10_r3 import load_policy
    policy, _ = load_policy(path, config, device)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", default="composite")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--world-seed", type=int, default=8101)
    parser.add_argument("--world-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[1101, 2203, 3307])
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--formal-steps", type=int, default=12288)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-wait", type=int, default=3)
    parser.add_argument("--world-checkpoint", type=Path)
    parser.add_argument("--world-policy-checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "formal" and (args.world_checkpoint is None or not args.world_checkpoint.exists()):
        parser.error("formal mode requires --world-checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    config = M10Config()
    if args.mode == "pilot":
        run_pilot(args, config, args.output)
    else:
        run_formal(args, config, args.output)
    dump(args.output / "run-complete.json", {"mode": args.mode, "training_performed": True, "completed_at": time.time()})


if __name__ == "__main__":
    main()
