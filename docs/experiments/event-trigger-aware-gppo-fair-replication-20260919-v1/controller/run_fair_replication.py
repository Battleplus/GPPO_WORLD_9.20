"""Registered local fair replication controller.

This controller owns the new experiment ledger and tapes.  It deliberately
does not import any historical run directory or budget database.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

SOURCE = Path(r"E:\Z博士\migration-artifacts\wd-event-trigger-aware-gppo-20260918\window-budget-fix-v1\source")
OUT = Path(r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1")
SEEDS = (1101, 2203, 3307)
CONDITIONS = ("I", "W1", "W2")
CONFIGS = (("A", "P_train", "periodic"), ("B", "P_train", "triggered"), ("C", "T_train", "triggered"))
PREFERENCE = (0.8, 0.2)
MAX_TRAIN_STEPS = 8192
MAX_TRAIN_UPDATES = 64
MAX_EVAL_STEPS = 48000

sys.path.insert(0, str(SOURCE))
from gppo_world.budget_executor import PersistentBudget  # noqa: E402
from gppo_world.joint_gppo import (  # noqa: E402
    ActionConditionedTemporalWorldModel, JointGraphPreferencePolicy,
    JointTrainConfig, masked_normalized_preference,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config, M10Environment, formal_three_condition_tape, scenario_from_dict,
    scenario_tape, scenario_to_dict,
)
from tools.run_event_trigger_aware_gppo import (  # noqa: E402
    _mask_safe, _vector_reward, load_config, probe, run_training, sha256,
    trigger_decision,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=json_default), encoding="utf-8")
    temp.replace(path)


def json_default(value: Any):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def source_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": hash_file(path)})
    return {"schema": "source-manifest/1.0.0", "root": str(root), "files": rows,
            "manifest_sha256": hash_payload(rows)}


def freeze_tapes() -> tuple[dict[str, list[Any]], dict[str, list[Any]], str, str]:
    train_parent = scenario_tape("train", count=64, base_seed=961001)
    eval_parent = scenario_tape("test", count=64, base_seed=971001)
    train = {condition: formal_three_condition_tape("train", count=64, base_seed=961001, condition=condition)
             for condition in CONDITIONS}
    evaluation = {condition: formal_three_condition_tape("test", count=64, base_seed=971001, condition=condition)
                  for condition in CONDITIONS}
    payload_train = {"schema": "fair-replication-training-tapes/1.0.0", "parent_count": 64,
                     "parent_generation": {"split": "train", "base_seed": 961001, "name": "mixed"},
                     "parents": [scenario_to_dict(item) for item in train_parent],
                     "derived": {key: [scenario_to_dict(item) for item in value] for key, value in train.items()},
                     "derivation": "formal_three_condition_tape; I/W1/W2 public communication event rules"}
    payload_eval = {"schema": "fair-replication-evaluation-tapes/1.0.0", "parent_count": 64,
                    "parent_generation": {"split": "test", "base_seed": 971001, "name": "mixed"},
                    "parents": [scenario_to_dict(item) for item in eval_parent],
                    "derived": {key: [scenario_to_dict(item) for item in value] for key, value in evaluation.items()},
                    "derivation": "formal_three_condition_tape; independent parent tape; no performance filtering"}
    train_path = OUT / "tapes" / "training-tapes.json"
    eval_path = OUT / "tapes" / "evaluation-tapes.json"
    atomic_json(train_path, payload_train); atomic_json(eval_path, payload_eval)
    train_hash, eval_hash = hash_file(train_path), hash_file(eval_path)
    atomic_json(OUT / "tapes" / "tape-manifest.json", {
        "schema": "fair-replication-tape-manifest/1.0.0", "training_sha256": train_hash,
        "evaluation_sha256": eval_hash, "training_parent_ids": [item.tape_id for item in train_parent],
        "evaluation_parent_ids": [item.tape_id for item in eval_parent],
        "training_derivation": "I/W1/W2 from the same 64 parent tapes",
        "evaluation_derivation": "I/W1/W2 from a different 64 parent tapes",
        "dedup_scope": [
            str(SOURCE / "historical-formal-tapes-851001-851003.json"),
            str(OUT / "tapes" / "training-tapes.json"), str(OUT / "tapes" / "evaluation-tapes.json"),
        ],
        "dedup_note": "New seed ranges 961001/971001 were compared by tape_id and canonical scenario content against the available historical formal tape; legacy development tape files without canonical content are listed as unavailable coverage.",
    })
    return train, evaluation, train_hash, eval_hash


def load_tape_payload(path: Path) -> dict[str, dict[str, tuple[Any, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"parents": tuple(scenario_from_dict(item) for item in payload["parents"]),
            **{condition: tuple(scenario_from_dict(item) for item in payload["derived"][condition]) for condition in CONDITIONS}}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
        stream.flush()


def verify_start_identity(config: M10Config) -> dict[str, Any]:
    checkpoints = {}
    expected = {
        1101: "5b010a2f7eed0af0e3c64333b5c7846f18ffd0e72aff5705719003e1791b5930",
        2203: "f1e12262bb548f5a8329fbcf993fcf7238abd50c8fe892adc1b7ef29c4763af2",
        3307: "548b848a776dc31a4ffa96be7e50daf1f6f27929e2f314f272bf141a69063941",
    }
    for seed, wanted in expected.items():
        path = Path(r"E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final") / f"run/training/seed-{seed}/WD/last-recovery.pt"
        actual = hash_file(path)
        if actual != wanted:
            raise RuntimeError(f"WD checkpoint identity mismatch for {seed}: {actual}")
        checkpoints[str(seed)] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    return {"checkpoints": checkpoints, "config": asdict(config), "config_source": str(SOURCE / "tools/run_event_trigger_aware_gppo.py")}


def train_all(deadline: float, train: dict[str, list[Any]], train_hash: str, budget: PersistentBudget, protocol: dict[str, Any]) -> dict[str, Any]:
    scenarios = []
    for i in range(64):
        for condition in CONDITIONS:
            scenarios.append(train[condition][i])
    progress = {"status": "running", "runs": {}, "order": []}
    atomic_json(OUT / "training" / "training-progress.json", progress)
    for seed in SEEDS:
        for group in ("P_train", "T_train"):
            if time.time() >= deadline:
                raise TimeoutError("absolute experiment deadline reached before training group")
            run_key = f"{group}/seed-{seed}"
            run_dir = OUT / "training" / f"seed-{seed}" / group
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            atomic_json(run_dir.parent / f"{group}.start.json", {"run": run_key, "started_at": started, "deadline": deadline})
            try:
                result = run_training(
                    group=group, seed=seed, output_dir=run_dir,
                    env_config=protocol["environment_object"],
                    train_config=JointTrainConfig(seed=seed, rollout_steps=128), scenarios=scenarios,
                    max_steps=MAX_TRAIN_STEPS, max_policy_updates=MAX_TRAIN_UPDATES,
                    wall_seconds=max(1.0, deadline - time.time()), budget=budget,
                    budget_attempt_id=f"{protocol['experiment_id']}-{run_key}",
                    tape_sha256=train_hash, fixed_preference=PREFERENCE,
                )
            except BaseException as exc:
                atomic_json(run_dir / "failure.json", {"run": run_key, "error": repr(exc), "type": type(exc).__name__, "time": time.time()})
                progress["status"] = "stopped_on_error"; progress["failure"] = {"run": run_key, "error": repr(exc)}
                atomic_json(OUT / "training" / "training-progress.json", progress)
                raise
            commit = run_dir / "ledger-commit.json"
            if not commit.is_file() or int(result.get("environment_steps", -1)) != MAX_TRAIN_STEPS:
                raise RuntimeError(f"incomplete training group {run_key}; no valid 8192-step path")
            progress["runs"][run_key] = result
            progress["order"].append(run_key)
            atomic_json(OUT / "training" / "training-progress.json", progress)
    progress["status"] = "completed"
    atomic_json(OUT / "training" / "training-summary.json", progress)
    return progress


def load_models(train_root: Path, config: M10Config) -> dict[tuple[int, str], tuple[Any, Any, str]]:
    models = {}
    for seed in SEEDS:
        for label, group, _ in CONFIGS:
            checkpoint = train_root / f"seed-{seed}" / group / "last-recovery.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            policy = JointGraphPreferencePolicy(config, history=True).cpu()
            world = ActionConditionedTemporalWorldModel().cpu()
            policy.load_state_dict(payload["policy_state_dict"], strict=True); world.load_state_dict(payload["world_state_dict"], strict=True)
            policy.eval(); world.eval()
            models[(seed, label)] = (policy, world, hash_file(checkpoint))
    return models


def evaluate_episode(model: tuple[Any, Any, str], seed: int, label: str, group: str, mode: str,
                     condition: str, scenario: Any, config: M10Config, budget: PersistentBudget,
                     deadline: float, episode_key: str) -> dict[str, Any]:
    policy, world, checkpoint_hash = model
    preference = masked_normalized_preference(PREFERENCE, device=torch.device("cpu"))
    env = M10Environment(config, scenario); obs = env.reset(); previous = None; last_action = None
    policy_hidden = world_hidden = None; since = 1; done = False; steps = 0
    actors = continuations = world_forwards = 0; total_reward = np.zeros(2, dtype=np.float64)
    previous_counts = {"completed": 0, "expired": 0}; previous_energy = float(config.uav_count * config.initial_energy)
    illegal_actions = noop_opportunities = noop_selected = security_violations = messages = 0
    latencies_ns: list[int] = []; info: dict[str, Any] = {}
    while not done and steps < 128:
        if time.time() >= deadline:
            raise TimeoutError("absolute experiment deadline reached during evaluation")
        decision_start = time.perf_counter_ns()
        result = probe(policy, world, obs, policy_hidden, world_hidden, preference, torch.device("cpu")); world_forwards += 1
        if mode == "periodic":
            actor, reasons = True, ["periodic_policy"]
        else:
            actor, reasons, _ = trigger_decision(obs, previous, last_action, since)
        if actor:
            action = int(torch.argmax(result["evaluation"]["distribution"].logits, dim=-1).item()); actors += 1; since = 0
        else:
            action = int(last_action); continuations += 1; since += 1
        latencies_ns.append(time.perf_counter_ns() - decision_start)
        mask = _mask_safe(obs["mask"])
        continuation_allowed = (not actor) and action in {int(value) for value in obs.get("continuation_actions", ())}
        if actor and not bool(mask[action]):
            raise RuntimeError(f"illegal new evaluation action {episode_key} step {steps}")
        if not actor and not continuation_allowed:
            raise RuntimeError(f"invalid evaluation continuation {episode_key} step {steps}")
        illegal_actions += int(not bool(mask[action]) and not continuation_allowed)
        noop_opportunities += int(bool(mask[:-1].any())); noop_selected += int(action == config.action_count - 1 and bool(mask[:-1].any()))
        token = budget.reserve("evaluation_environment_steps", run_id="evaluation")
        try:
            next_obs, reward, done, info = env.step(action, submit_command=actor)
        except BaseException as exc:
            budget.unknown(token, f"evaluation env.step: {type(exc).__name__}: {exc}"); raise
        else:
            budget.complete(token)
        vector_reward, _, previous_counts, previous_energy = _vector_reward(info, previous_counts, previous_energy, config)
        total_reward += vector_reward.astype(np.float64)
        security_violations += len(info.get("security_violations", [])) if isinstance(info.get("security_violations", []), (list, tuple)) else int(bool(info.get("security_violation", False)))
        messages += len(info.get("communication_delta", [])); previous, obs = obs, next_obs
        policy_hidden, world_hidden = result["next_policy_hidden"], result["by_action"][action]["hidden"]
        last_action = action; steps += 1
    records = info.get("completion_records", {}) if info else {}
    physical_on_time = sum(record.get("physical_arrival_before_deadline") is True for record in records.values())
    host_on_time = sum(record.get("host_confirmation_before_deadline") is True for record in records.values())
    counts = info.get("counts", {}) if info else {}
    return {"episode_key": episode_key, "seed": seed, "configuration": label, "group": group, "mode": mode,
            "condition": condition, "tape_id": scenario.tape_id, "parent_tape_id": scenario.tape_id.rsplit("-", 1)[0],
            "checkpoint_sha256": checkpoint_hash, "steps": steps, "actor_calls": actors,
            "continuation_steps": continuations, "world_forward_calls": world_forwards,
            "decision_latency_ns": latencies_ns, "decision_latency_boundary": "probe+trigger+action; env.step and persistence excluded",
            "physical_on_time": physical_on_time, "host_on_time": host_on_time, "task_count": len(scenario.tasks),
            "completed": counts.get("completed", 0), "expired": counts.get("expired", 0),
            "reward_vector": total_reward.tolist(), "illegal_actions": illegal_actions,
            "noop_opportunities": noop_opportunities, "noop_selected": noop_selected,
            "security_violations": security_violations, "communication_bytes": None,
            "communication_proxy_messages": messages, "trigger_reasons": reasons if 'reasons' in locals() else []}


def evaluate_all(deadline: float, evaluation: dict[str, list[Any]], config: M10Config,
                 budget: PersistentBudget, train_root: Path) -> dict[str, Any]:
    models = load_models(train_root, config)
    ledger = OUT / "evaluation" / "episodes.jsonl"
    existing = set()
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip(): existing.add(json.loads(line)["episode_key"])
    order = []
    rows = []
    for index in range(64):
        for seed in SEEDS:
            for condition in CONDITIONS:
                for label, group, mode in CONFIGS:
                    key = f"{label}|seed-{seed}|{condition}|{evaluation[condition][index].tape_id}"
                    order.append(key)
                    if key in existing: continue
                    row = evaluate_episode(models[(seed, label)], seed, label, group, mode, condition,
                                           evaluation[condition][index], config, budget, deadline, key)
                    append_jsonl(ledger, row); rows.append(row)
    all_rows = []
    if ledger.is_file():
        all_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["episode_key"] for row in all_rows}) != 1728:
        raise RuntimeError(f"evaluation incomplete or duplicate: {len(all_rows)} unique episodes")
    summary = {"status": "completed", "episode_count": len(all_rows), "order_hash": hash_payload(order),
               "budget": budget.snapshot(), "episodes_ledger": str(ledger), "optimizer_calls": 0,
               "physical_arrival_rate": {}, "host_confirmation_rate": {}, "cost": {}}
    for label, _, _ in CONFIGS:
        selected = [row for row in all_rows if row["configuration"] == label]
        summary["physical_arrival_rate"][label] = sum(row["physical_on_time"] for row in selected) / max(1, sum(row["task_count"] for row in selected))
        summary["host_confirmation_rate"][label] = sum(row["host_on_time"] for row in selected) / max(1, sum(row["task_count"] for row in selected))
        summary["cost"][label] = {"env_steps": sum(row["steps"] for row in selected), "actor_calls": sum(row["actor_calls"] for row in selected),
                                   "continuation_steps": sum(row["continuation_steps"] for row in selected), "world_forward_calls": sum(row["world_forward_calls"] for row in selected)}
    atomic_json(OUT / "evaluation" / "evaluation-summary.json", summary)
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
        print("already completed; refusing duplicate execution")
        return 0
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    prior_protocol = OUT / "protocol.json"
    if prior_protocol.is_file():
        prior = json.loads(prior_protocol.read_text(encoding="utf-8"))
        started = float(prior["started_at"]); deadline = float(prior["absolute_deadline"])
    else:
        started = time.time(); deadline = started + 8 * 3600
    config = load_config()
    identity = verify_start_identity(config)
    train, evaluation, train_hash, eval_hash = freeze_tapes()
    shutil.copytree(SOURCE, OUT / "source-snapshot", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    manifest = source_manifest(OUT / "source-snapshot")
    atomic_json(OUT / "source-manifest.json", manifest)
    budget = PersistentBudget(OUT / "budget.sqlite3", limits={"environment_steps": 49152, "evaluation_environment_steps": MAX_EVAL_STEPS,
        "policy_optimizer_calls": 384, "world_optimizer_calls": 0}, run_limits={"environment_steps": 8192, "evaluation_environment_steps": MAX_EVAL_STEPS,
        "policy_optimizer_calls": 64, "world_optimizer_calls": 0}, attempt_id="event-trigger-aware-fair-replication-controller")
    protocol = {"schema": "event-trigger-aware-gppo-fair-replication/1.0.0", "experiment_id": "event-trigger-aware-gppo-fair-replication-20260919-v1",
                "status": "registered", "started_at": started, "absolute_deadline": deadline, "deadline_hours": 8,
                "source": str(SOURCE), "source_manifest": manifest["manifest_sha256"], "training_tape_sha256": train_hash, "evaluation_tape_sha256": eval_hash,
                "seeds": SEEDS, "conditions": CONDITIONS, "configurations": CONFIGS, "preference": PREFERENCE,
                "training_limits": {"groups": 6, "environment_steps_each": 8192, "policy_optimizer_calls_each_max": 64, "environment_steps_total": 49152, "policy_optimizer_calls_total": 384, "world_updates": 0},
                "evaluation_limits": {"episodes": 1728, "environment_steps_max": 48000, "optimizer_calls": 0},
                "evaluation_order": "parent index, seed, condition, A/B/C interleaving; unique episode_key",
                "fairness_axis": "environment interaction budget; actor/window/update counts need not match",
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                "threads": {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()},
                "identity": identity, "historical_exclusions": ["formal-run-v1", "frozen-validation-20260919-v1", "budget-sqlite-v2-20260918"]}
    protocol["environment_object"] = config
    atomic_json(OUT / "protocol.json", {key: value for key, value in protocol.items() if key != "environment_object"} | {"environment": asdict(config)})
    atomic_json(status_path, {"status": "registered", "experiment_id": protocol["experiment_id"], "started_at": started, "absolute_deadline": deadline, "budget": budget.snapshot()})
    try:
        train_summary = train_all(deadline, train, train_hash, budget, protocol)
        atomic_json(status_path, {"status": "training_completed", "budget": budget.snapshot(), "training_summary": str(OUT / "training" / "training-summary.json")})
        evaluation_summary = evaluate_all(deadline, evaluation, config, budget, OUT / "training")
        atomic_json(OUT / "report-input.json", {"protocol": str(OUT / "protocol.json"), "training": train_summary, "evaluation": evaluation_summary})
        atomic_json(status_path, {"status": "completed", "budget": budget.snapshot(), "evaluation_summary": evaluation_summary})
        return 0
    except BaseException as exc:
        atomic_json(status_path, {"status": "stopped_on_error", "error": repr(exc), "type": type(exc).__name__, "budget": budget.snapshot(), "time": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
