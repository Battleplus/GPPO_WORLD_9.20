"""Evaluate frozen joint-consequence action selection on a new paired tape.

This is a bounded, no-training validation runner.  The four selectors choose
from the same public prefix before any selected-branch outcome is inspected.
The selected branches share the same exogenous key within a prefix/repeat;
the simulator's hidden state is used only after the branch for auditing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.graph5 import graph5_from_m10_observation  # noqa: E402
from gppo_world.joint_consequence_baseline import (  # noqa: E402
    PublicJointScoreConfig,
    public_joint_action_score,
    select_public_joint_action,
)
from gppo_world.joint_consequence_model import (  # noqa: E402
    Graph5JointConsequenceModel,
    JointConsequenceModelConfig,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config,
    M10Environment,
    M10Scenario,
    scenario_to_dict,
    weak_communication_profile,
    weak_communication_tape,
)
from tools.generate_m10_joint_consequence_dataset import (  # noqa: E402
    _legal_actions,
    _public_task_set,
    _task_outcome,
    arrival_config,
    digest,
)
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


PREREG_PROTOCOL = "world-gppo-9.11-joint-consequence-independent-validation/0.1.0"
METHODS = ("current_rule", "public_joint", "zero_delta", "joint_model")
PREFIX_STEPS = (1, 2, 3, 4)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def compact_step(action: int, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": int(action), "reward": float(reward), "done": bool(done),
        "time": float(info["time"]), "feedback": str(info["feedback"]),
        "counts": info.get("counts", {}),
        "communication_delta": list(info.get("communication_delta", [])),
    }


def public_prefix(scenario: M10Scenario, prefix_steps: int, prefix_key: str) -> dict[str, Any]:
    env = M10Environment(config=arrival_config(), scenario=scenario, exogenous_key=prefix_key)
    obs = env.reset()
    trace: list[dict[str, Any]] = []
    actions: list[int] = []
    for _ in range(prefix_steps):
        action = int(traditional_action(obs))
        if action not in _legal_actions(obs):
            raise RuntimeError("frozen public controller selected an illegal prefix action")
        actions.append(action)
        obs, reward, done, info = env.step(action)
        trace.append(compact_step(action, reward, done, info))
        if done:
            raise RuntimeError("planned prefix reached terminal state")
    task_set = _public_task_set(env, obs)
    return {
        "actions": actions,
        "observation": obs,
        "task_set": task_set,
        "graph": graph5_from_m10_observation(obs),
        "trace": trace,
        "trace_sha256": digest(trace),
        "public_snapshot_digest": digest({"flat": np.asarray(obs["flat"]).tolist(), "mask": np.asarray(obs["mask"]).tolist(), "version": int(obs["version"])}),
    }


def select_actions(prefix: dict[str, Any], model: Graph5JointConsequenceModel, device: torch.device, score_config: PublicJointScoreConfig) -> dict[str, Any]:
    obs = prefix["observation"]
    legal = _legal_actions(obs)
    candidate_legal = [action for action in legal if action < 24]
    rule = int(traditional_action(obs))
    public_graph = prefix["graph"]
    public = int(select_public_joint_action(public_graph, score_config))
    zero = int(min(legal))
    graph_device = public_graph.to(device)
    with torch.inference_mode():
        actions_t, prediction_t = model.predict_candidates(graph_device, len(prefix["task_set"]), legal)
    predictions = {int(action): [float(value) for value in row] for action, row in zip(actions_t.cpu().tolist(), prediction_t.cpu().tolist())}
    max_value = max(predictions[action][0] for action in legal)
    model_candidates = [action for action in legal if abs(predictions[action][0] - max_value) <= 1e-12]
    selected = {
        "current_rule": rule,
        "public_joint": public,
        "zero_delta": zero,
        "joint_model": int(min(model_candidates)),
    }
    if rule not in legal or public not in legal or zero not in legal:
        raise RuntimeError("selector returned an illegal action")
    return {
        "legal_actions": legal,
        "candidate_legal_actions": candidate_legal,
        "selected_actions": selected,
        "model_predictions_by_action": predictions,
        "model_tie_value": float(max_value),
        "tie_rule": "minimum_legal_action_id_within_1e-12",
        "public_joint_scores": {str(action): public_joint_action_score(public_graph, action, score_config) for action in legal},
    }


def run_selected_branch(
    scenario: M10Scenario,
    prefix_actions: list[int],
    selected_action: int,
    task_set: tuple[str, ...],
    prefix_key: str,
    branch_key: str,
    horizon_steps: int,
) -> dict[str, Any]:
    env = M10Environment(config=arrival_config(), scenario=scenario, exogenous_key=prefix_key)
    obs = env.reset()
    prefix_trace: list[dict[str, Any]] = []
    for action in prefix_actions:
        obs, reward, done, info = env.step(int(action))
        prefix_trace.append(compact_step(action, reward, done, info))
        if done:
            raise RuntimeError("branch prefix reached terminal state")
    if _public_task_set(env, obs) != task_set:
        raise RuntimeError("branch public task set differs from selection prefix")
    # This is the only point where the paired repeat stream is selected.
    env._exogenous_key = branch_key
    branch_trace: list[dict[str, Any]] = []
    branch_actions: list[int] = []
    started = time.perf_counter()
    done = False
    for step in range(horizon_steps):
        action = int(selected_action if step == 0 else traditional_action(obs))
        branch_actions.append(action)
        obs, reward, done, info = env.step(action)
        branch_trace.append(compact_step(action, reward, done, info))
        if done:
            break
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after_time = float(env.clock.time)
    outcomes = [_task_outcome(env, task_id, after_time) for task_id in task_set]
    communication = list(env._communication_log)
    counts = Counter(str(item.get("status", "unknown")) for item in communication)
    completed = [item for item in outcomes if item["arrival_before_deadline_physical"] is True]
    observed = [item for item in outcomes if item["deadline_observed"] is True]
    return {
        "selected_action": int(selected_action),
        "branch_actions": branch_actions,
        "task_outcomes": outcomes,
        "metrics": {
            "on_time_count": len(completed) if len(observed) == len(outcomes) else None,
            "deadline_failure_count": (len(observed) - len(completed)) if len(observed) == len(outcomes) else None,
            "observed_task_count": len(observed),
            "censored_task_count": len(outcomes) - len(observed),
            "physical_arrival_count": sum(item["physical_arrival_time"] is not None for item in outcomes),
            "decision_wall_ms": elapsed_ms,
            "communication_count": len(communication),
            "communication_status_counts": dict(counts),
            "communication_proxy_bytes": sum(len(json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")) for item in communication),
            "safety_violation_count": sum(1 for item in getattr(env.execution, "log", []) if str(item.get("result")) in {"safety_violation", "unauthorized", "fencing_bypass"}),
        },
        "label_provenance": {
            "label_source": "M10Environment.counterfactual_simulator",
            "hidden_state_used_online": False,
            "shared_exogenous_key": branch_key,
            "prefix_key": prefix_key,
            "prefix_trace_sha256": digest(prefix_trace),
            "branch_trace_sha256": digest(branch_trace),
            "continuation_controller": "frozen_public_controller_v1",
            "observation_window_ended_at": after_time,
        },
        "branch_ledger": {
            "prefix_trace": prefix_trace,
            "branch_trace": branch_trace,
            "clock_log": list(env.clock.log),
            "execution_log": list(env.execution.log),
            "communication_log": communication,
            "completion_records": {key: dict(value) for key, value in env._completion_records.items()},
            "final_observation": {"time": float(obs["time"]), "version": int(obs["version"]), "mask": np.asarray(obs["mask"]).tolist(), "flat": np.asarray(obs["flat"]).tolist()},
        },
    }


def bootstrap_ci(values: list[float], seed: int = 1101, samples: int = 10000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p05": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(samples, arr.size))
    means = arr[indices].mean(axis=1)
    return {"n": int(arr.size), "mean": float(arr.mean()), "p05": float(np.percentile(means, 2.5)), "p95": float(np.percentile(means, 97.5)), "bootstrap_seed": seed, "bootstrap_samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=293001)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.count != 16 or args.repeats != 8 or args.base_seed != 293001 or args.horizon_steps != 6:
        raise SystemExit("independent validation is frozen at count=16, base_seed=293001, repeats=8, horizon=6")
    if args.out.exists():
        raise SystemExit(f"refusing existing output directory: {args.out}")
    if not args.model.is_file():
        raise SystemExit(f"missing frozen model checkpoint: {args.model}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent device fallback")
    torch.set_num_threads(max(1, min(int(args.threads), 4)))
    device = torch.device(args.device)
    args.out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    model = Graph5JointConsequenceModel(JointConsequenceModelConfig(hidden_dim=64)).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    score_config = PublicJointScoreConfig()
    source_paths = {
        "runner": Path(__file__),
        "model": ROOT / "gppo_world" / "joint_consequence_model.py",
        "baseline": ROOT / "gppo_world" / "joint_consequence_baseline.py",
        "environment": ROOT / "gppo_world" / "m10_environment.py",
        "graph5": ROOT / "gppo_world" / "graph5.py",
        "protocol": ROOT / "docs" / "world-model" / "m10-joint-consequence-protocol-20260914.md",
        "config": ROOT / "configs" / "world-gppo-9.11-arrival-v0.1.0.json",
    }
    prereg_path = ROOT / "docs" / "plans" / "m10-joint-consequence-independent-validation-prereg-20260914.md"
    identity = {
        "schema": PREREG_PROTOCOL,
        "run_id": args.run_id,
        "status": "running",
        "started_at": started,
        "source_head_recorded_before_run": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frozen_model": {"path": str(args.model), "sha256": sha256_file(args.model), "format": checkpoint.get("format_version"), "checkpoint_epoch": checkpoint.get("epoch")},
        "protocol": {"arrival": "world-gppo-9.11-arrival/0.1.0", "task_completion_mode": "arrival_to_region", "primary_deadline_basis": "physical_arrival", "host_confirmation_secondary": True, "horizon_steps": args.horizon_steps, "continuation": "frozen_public_controller_v1", "notification_mode": "single_shot"},
        "design": {"parent_count": args.count, "prefix_steps": list(PREFIX_STEPS), "repeats": args.repeats, "conditions": {"ideal": "ideal", "weak": "composite"}, "methods": list(METHODS), "max_selected_branches": args.count * len(PREFIX_STEPS) * 2 * args.repeats * len(METHODS), "candidate_cap": 25, "base_seed": args.base_seed, "development_base_seed": 193001},
        "selectors": {"joint_model": "max predicted new_on_time_count; min legal action ID within 1e-12", "public_joint": "max public score; min action ID", "zero_delta": "min legal action ID", "current_rule": "traditional_action frozen implementation"},
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "preregistration_sha256": sha256_file(prereg_path),
        "runtime": {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version, "torch": torch.__version__, "device": str(device), "torch_threads": torch.get_num_threads(), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "budget": {"new_parent_tapes": 16, "max_simulator_selected_branches": 4096, "no_training": True, "no_tuning": True},
    }
    dump(args.out / "run-identity.json", identity)
    dump(args.out / "frozen-inputs.json", {"model_checkpoint": checkpoint.get("run_identity", {}), "score_config": score_config.__dict__, "planned_prefix_steps": list(PREFIX_STEPS), "conditions": ["ideal", "weak"], "methods": list(METHODS), "tapes": [scenario_to_dict(s) for s in weak_communication_tape("test", count=args.count, base_seed=args.base_seed, level="composite", name="mixed")]})
    selections_path = args.out / "selections.jsonl"
    branches_path = args.out / "selected-branches.jsonl"
    parent_records: dict[str, dict[str, Any]] = {}
    scenarios = weak_communication_tape("test", count=args.count, base_seed=args.base_seed, level="composite", name="mixed")
    for scenario in scenarios:
        parent_id = f"independent:{scenario.tape_id}"
        parent_records[parent_id] = {"parent_episode_id": parent_id, "tape_id": scenario.tape_id, "scenario_seed": int(scenario.seed), "conditions": {}}
        for condition_name, condition_level in (("ideal", "ideal"), ("weak", "composite")):
            condition_scenario = M10Scenario(**{**scenario.__dict__, "communication": weak_communication_profile(condition_level)})
            condition_record = {"planned_prefixes": {}, "eligible_prefix_count": 0, "selected_branch_count": 0}
            parent_records[parent_id]["conditions"][condition_name] = condition_record
            for prefix_steps in PREFIX_STEPS:
                prefix_id = f"{parent_id}:prefix-{prefix_steps}:public-rule"
                prefix_key = f"{scenario.tape_id}|{parent_id}|{prefix_id}|{condition_name}|prefix"
                prefix = public_prefix(condition_scenario, prefix_steps, prefix_key)
                if not prefix["task_set"]:
                    record = {"parent_episode_id": parent_id, "condition": condition_name, "prefix_id": prefix_id, "prefix_steps": prefix_steps, "status": "ineligible_no_public_pending_task_set", "prefix_trace_sha256": prefix["trace_sha256"], "task_set": []}
                    condition_record["planned_prefixes"][str(prefix_steps)] = record
                    with selections_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                condition_record["eligible_prefix_count"] += 1
                selections = select_actions(prefix, model, device, score_config)
                selection_record = {"parent_episode_id": parent_id, "tape_id": scenario.tape_id, "scenario_seed": int(scenario.seed), "condition": condition_name, "prefix_id": prefix_id, "prefix_steps": prefix_steps, "status": "eligible", "task_set": list(prefix["task_set"]), "prefix_actions": prefix["actions"], "prefix_trace": prefix["trace"], "prefix_trace_sha256": prefix["trace_sha256"], "public_snapshot_digest": prefix["public_snapshot_digest"], "selection": selections}
                condition_record["planned_prefixes"][str(prefix_steps)] = {"status": "eligible", "task_set": list(prefix["task_set"]), "selection": selections["selected_actions"]}
                with selections_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(selection_record, ensure_ascii=False, sort_keys=True) + "\n")
                for repeat in range(args.repeats):
                    branch_key = f"{scenario.tape_id}|{parent_id}|{prefix_id}|{condition_name}|repeat-{repeat}"
                    for method in METHODS:
                        method_action = int(selections["selected_actions"][method])
                        branch = run_selected_branch(condition_scenario, prefix["actions"], method_action, tuple(prefix["task_set"]), prefix_key, branch_key, args.horizon_steps)
                        branch_record = {"parent_episode_id": parent_id, "tape_id": scenario.tape_id, "condition": condition_name, "prefix_id": prefix_id, "prefix_steps": prefix_steps, "repeat": repeat, "method": method, "branch_key": branch_key, "selection_snapshot": selections["selected_actions"], "branch": branch}
                        with branches_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(branch_record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                        condition_record["selected_branch_count"] += 1
    dump(args.out / "parent-index.json", parent_records)
    branch_rows = [json.loads(line) for line in branches_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_condition_method: dict[str, dict[str, Any]] = {}
    parent_diffs: dict[str, dict[str, list[float]]] = {}
    for condition in ("ideal", "weak"):
        by_condition_method[condition] = {}
        for method in METHODS:
            values = [row["branch"]["metrics"]["on_time_count"] for row in branch_rows if row["condition"] == condition and row["method"] == method and row["branch"]["metrics"]["on_time_count"] is not None]
            by_condition_method[condition][method] = {"selected_branch_n": len(values), "on_time_count_mean": float(np.mean(values)) if values else None, "on_time_count_values": values}
        parent_diffs[condition] = {}
    grouped: dict[tuple[str, str, str, int], dict[str, int | None]] = {}
    for row in branch_rows:
        grouped.setdefault((row["condition"], row["parent_episode_id"], row["prefix_id"], int(row["repeat"])), {})[row["method"]] = row["branch"]["metrics"]["on_time_count"]
    parent_values: dict[str, dict[str, dict[str, list[float]]]] = {c: {m: {} for m in METHODS} for c in ("ideal", "weak")}
    for (condition, parent, prefix_id, repeat), vals in grouped.items():
        if any(vals.get(method) is None for method in METHODS):
            continue
        for method in METHODS:
            for ref in ("current_rule", "public_joint", "zero_delta"):
                if method == ref:
                    continue
                key = f"{method}_minus_{ref}"
                parent_values[condition][key].setdefault(parent, []).append(float(vals[method] - vals[ref])) if key in parent_values[condition] else parent_values[condition].setdefault(key, {}).setdefault(parent, []).append(float(vals[method] - vals[ref]))
    paired = {condition: {} for condition in ("ideal", "weak")}
    for condition in paired:
        for key, per_parent in parent_values[condition].items():
            parent_means = [float(np.mean(values)) for values in per_parent.values()]
            paired[condition][key] = {"parent_values": {parent: values for parent, values in per_parent.items()}, "parent_mean_ci": bootstrap_ci(parent_means)}
    report = {"schema": PREREG_PROTOCOL, "run_id": args.run_id, "status": "completed", "training": False, "tuning": False, "fusion": False, "independent_same_distribution_not_ood": True, "summary": by_condition_method, "paired_parent_differences": paired, "censoring": {"total_branch_rows": len(branch_rows), "censored_branch_rows": sum(row["branch"]["metrics"]["on_time_count"] is None for row in branch_rows)}, "limitations": ["parent episode is the primary statistical unit", "fixed continuation controller only", "host confirmation is not the primary joint label", "no business cost threshold was registered; no formal acceptance claim"], "completed_at": datetime.now(timezone.utc).isoformat()}
    dump(args.out / "independent-validation-report.json", report)
    identity["status"] = "completed"
    identity["completed_at"] = report["completed_at"]
    identity["actual"] = {"branch_rows": len(branch_rows), "selection_rows": len(selections_path.read_text(encoding="utf-8").splitlines()), "eligible_prefixes": sum(item["conditions"][condition]["eligible_prefix_count"] for item in parent_records.values() for condition in ("ideal", "weak"))}
    dump(args.out / "run-identity.json", identity)
    files = {}
    for path in sorted(args.out.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            files[str(path.relative_to(args.out))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dump(args.out / "artifact-manifest.json", {"schema": PREREG_PROTOCOL, "run_id": args.run_id, "files": files, "input_model_sha256": sha256_file(args.model), "branch_rows": len(branch_rows), "no_training": True})
    print(json.dumps({"run_id": args.run_id, "status": "completed", "branch_rows": len(branch_rows), "censored_branch_rows": report["censoring"]["censored_branch_rows"], "summary": {condition: {method: value["on_time_count_mean"] for method, value in methods.items()} for condition, methods in by_condition_method.items()}}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
