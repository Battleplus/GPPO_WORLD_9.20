"""Generate bounded, grouped labels for candidate-to-task-set consequences.

This is an explicit simulator-side adapter for the frozen arrival protocol.  It
is intentionally not run by the design-only stage.  Every candidate in a
prefix is evaluated from the same public prefix, with the same scenario and
exogenous repeat key; only the first action differs.  The continuation is a
fixed public controller and is recorded as part of the label provenance.

Hidden simulator state is used only after the branch to construct supervision.
It is never serialized as model input.  A non-empty output directory is
rejected so a new run cannot overwrite an earlier dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.graph5 import graph5_from_m10_observation  # noqa: E402
from gppo_world.joint_consequence import (  # noqa: E402
    DEFAULT_CONTINUATION_CONTROLLER,
    build_group_targets,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config,
    M10Environment,
    M10Scenario,
    weak_communication_tape,
)
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


SCHEMA = "gppo-arrival-joint-consequence/v1"
PROTOCOL = "world-gppo-9.11-joint-consequence/0.1.0"
PARENT_PROTOCOL = "world-gppo-9.11-arrival/0.1.0"


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrival_config() -> M10Config:
    # The notification candidate is deliberately not enabled in this route.
    return M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")


def _public_task_set(env: M10Environment, obs: dict[str, Any]) -> tuple[str, ...]:
    """Return pending task identities already allocated by the public view.

    ``view._tasks`` is the delivery-order identity table owned by
    TaskPolicyView.  The eligibility decision below uses only its public
    snapshot fields; the simulator task table is not consulted here.
    """
    rows = np.asarray(obs["tasks"], dtype=np.float32)
    result: list[str] = []
    for index, task_id in enumerate(tuple(env.view._tasks)):
        row = rows[index]
        valid = row[2::4]
        pending = float(row[20])
        deadline = float(row[8])
        if np.all(valid[:6] > 0.5) and pending > 0.5 and deadline > float(obs["time"]):
            result.append(str(task_id))
    if len(result) != len(set(result)):
        raise RuntimeError("public task identity table contains duplicates")
    return tuple(result)


def _legal_actions(obs: dict[str, Any]) -> list[int]:
    legal = [int(i) for i, allowed in enumerate(np.asarray(obs["mask"], dtype=bool)) if allowed]
    if not legal:
        raise RuntimeError("public action mask has no legal action")
    return legal


def _compact(action: int, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    communication = list(info.get("communication_delta", []))
    statuses: dict[str, int] = {}
    for event in communication:
        status = str(event.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "action": int(action),
        "reward": float(reward),
        "done": bool(done),
        "time": float(info["time"]),
        "feedback": str(info["feedback"]),
        "counts": info["counts"],
        "communication_delta": {
            "count": len(communication),
            "status_counts": statuses,
            "sha256": digest(communication),
        },
    }


def _task_outcome(env: M10Environment, task_id: str, after_time: float) -> dict[str, Any]:
    """Construct a masked physical-arrival label from simulator state."""
    task = env.clock.tasks[task_id]
    arrivals = [
        item for item in env.clock.log
        if item.get("kind") == "arrival" and str(item.get("task")) == task_id
    ]
    arrival = min(arrivals, key=lambda item: float(item["time"])) if arrivals else None
    deadline = float(task.deadline)
    observed = bool(arrival is not None or task.state.value == "expired" or after_time >= deadline)
    arrival_time = None if arrival is None else float(arrival["time"])
    on_time = None if not observed else bool(arrival is not None and arrival_time <= deadline)
    if arrival is not None:
        failure_class = None if on_time else "arrived_after_deadline"
    elif task.state.value == "expired" or observed:
        failure_class = "not_arrived_before_deadline"
    else:
        failure_class = "censored_observation_window"
    return {
        "task_id": task_id,
        "deadline": deadline,
        "physical_arrival_time": arrival_time,
        "arrival_before_deadline_physical": on_time,
        "deadline_observed": observed,
        "failure_class": failure_class,
        "label_source": "M10Environment.counterfactual_simulator",
    }


def _branch(
    scenario: M10Scenario,
    prefix_actions: list[int],
    candidate_action: int,
    *,
    task_set: tuple[str, ...],
    parent_id: str,
    prefix_id: str,
    repeat_id: str,
    prefix_exogenous_key: str,
    exogenous_key: str,
    horizon_steps: int,
) -> dict[str, Any]:
    env = M10Environment(config=arrival_config(), scenario=scenario, exogenous_key=prefix_exogenous_key)
    obs = env.reset()
    prefix_trace = []
    for action in prefix_actions:
        obs, reward, done, info = env.step(int(action))
        prefix_trace.append(_compact(action, reward, done, info))
        if done:
            raise RuntimeError("prefix reached terminal state")
    graph = graph5_from_m10_observation(obs).as_dict()
    public_task_set = _public_task_set(env, obs)
    if public_task_set != task_set:
        raise RuntimeError("branch public task set differs from prefix task set")

    # Prefix delivery must be identical across candidates and repeats.  The
    # branch key is switched only after the public decision state is frozen,
    # so repeat-specific randomness cannot change task-set membership.
    env._exogenous_key = exogenous_key

    branch_trace = []
    branch_actions = []
    for step in range(horizon_steps):
        action = int(candidate_action if step == 0 else traditional_action(obs))
        branch_actions.append(action)
        obs, reward, done, info = env.step(action)
        branch_trace.append(_compact(action, reward, done, info))
        if done:
            break
    after_time = float(env.clock.time)
    outcomes = [_task_outcome(env, task_id, after_time) for task_id in task_set]
    return {
        "schema": SCHEMA,
        "parent_episode_id": parent_id,
        "prefix_id": prefix_id,
        "repeat_id": repeat_id,
        "graph5_t": graph,
        "joint_task_set": list(task_set),
        "joint_task_outcomes": outcomes,
        "target": {
            "action": int(candidate_action),
            "prediction_horizon_steps": int(horizon_steps),
            "reference_policy": "traditional_public_controller_v1",
        },
        "exogenous_key": exogenous_key,
        "label_provenance": {
            "label_source": "M10Environment.counterfactual_simulator",
            "hidden_state_online": False,
            "shared_exogenous_randomness": True,
            "scenario_tape_id": scenario.tape_id,
            "scenario_seed": int(scenario.seed),
            "prefix_action_digest": digest(prefix_actions),
            "branch_action_digest": digest(branch_actions),
            "prefix_trace_sha256": digest(prefix_trace),
            "branch_trace_sha256": digest(branch_trace),
            "post_branch_control": DEFAULT_CONTINUATION_CONTROLLER,
            "observation_window_ended_at": after_time,
        },
        "branch_ledger": {
            "prefix_trace": prefix_trace,
            "branch_trace": branch_trace,
            "actions": branch_actions,
        },
    }


def _prefix_actions(scenario: M10Scenario, prefix_length: int, key: str) -> tuple[list[int], dict[str, Any]]:
    env = M10Environment(config=arrival_config(), scenario=scenario, exogenous_key=key)
    obs = env.reset()
    actions: list[int] = []
    for _ in range(prefix_length):
        action = int(traditional_action(obs))
        if action not in _legal_actions(obs):
            raise RuntimeError("public continuation selected an illegal action")
        actions.append(action)
        obs, _, done, _ = env.step(action)
        if done:
            raise RuntimeError("frozen public prefix reached terminal state")
    return actions, obs


def make_split(
    split: str,
    count: int,
    base_seed: int,
    prefix_steps: list[int],
    horizon_steps: int,
    max_candidates: int,
    exogenous_repeats: int,
    scenario_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scenario in weak_communication_tape(
        split, count=count, base_seed=base_seed, level="composite", name=scenario_name
    ):
        parent_id = f"{split}:{scenario.tape_id}"
        for prefix_length in prefix_steps:
            prefix_id = f"{parent_id}:prefix-{prefix_length}:public-rule"
            prefix_key = f"{scenario.tape_id}|{parent_id}|{prefix_id}"
            prefix_actions, prefix_obs = _prefix_actions(scenario, prefix_length, prefix_key)
            probe = M10Environment(config=arrival_config(), scenario=scenario, exogenous_key=prefix_key)
            probe.reset()
            for action in prefix_actions:
                probe.step(action)
            task_set = _public_task_set(probe, prefix_obs)
            legal = _legal_actions(prefix_obs)
            legal = sorted(
                legal,
                key=lambda action: digest({"parent": parent_id, "prefix": prefix_id, "action": action}),
            )[:max_candidates]
            reference_action = int(traditional_action(prefix_obs))
            if reference_action not in legal:
                legal = sorted(set(legal + [reference_action]))
            if not task_set:
                skipped.append({
                    "parent_episode_id": parent_id,
                    "prefix_id": prefix_id,
                    "reason": "no_pending_public_task_set",
                })
                continue
            for repeat in range(exogenous_repeats):
                repeat_id = str(repeat)
                exogenous_key = f"{scenario.tape_id}|{parent_id}|{prefix_id}|repeat-{repeat_id}"
                raw = [
                    _branch(
                        scenario,
                        prefix_actions,
                        action,
                        task_set=task_set,
                        parent_id=parent_id,
                        prefix_id=prefix_id,
                        repeat_id=repeat_id,
                        prefix_exogenous_key=prefix_key,
                        exogenous_key=exogenous_key,
                        horizon_steps=horizon_steps,
                    )
                    for action in legal
                ]
                rows.extend(build_group_targets(
                    raw,
                    reference_action=reference_action,
                    observation_window_steps=horizon_steps,
                    continuation_controller=DEFAULT_CONTINUATION_CONTROLLER,
                ))
    return rows, skipped


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_spec(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = sorted(
        json.dumps(
            {
                "parent_episode_id": row["parent_episode_id"],
                "prefix_id": row["prefix_id"],
                "repeat_id": row["repeat_id"],
                "action": row["target"]["action"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    )
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate parent/prefix/repeat/action identity")
    return {
        "path": path.name,
        "records": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity_sha256": hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
        "parents": len({row["parent_episode_id"] for row in rows}),
        "prefixes": len({(row["parent_episode_id"], row["prefix_id"]) for row in rows}),
        "repeats": len({(row["parent_episode_id"], row["prefix_id"], row["repeat_id"]) for row in rows}),
        "candidate_actions": len({row["target"]["action"] for row in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--ood-count", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=193001)
    parser.add_argument("--prefix-steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--max-candidates-per-prefix", type=int, default=25)
    parser.add_argument("--exogenous-repeats", type=int, default=3)
    args = parser.parse_args()
    counts = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count, "ood": args.ood_count}
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    if any(value < 0 for value in counts.values()) or not any(counts.values()):
        raise SystemExit("split counts must be nonnegative and at least one split must be nonzero")
    if any(step <= 0 for step in args.prefix_steps) or any(step > 4 for step in args.prefix_steps):
        raise SystemExit("prefix steps must be in [1, 4]")
    if args.horizon_steps <= 0 or args.max_candidates_per_prefix <= 0 or args.exogenous_repeats <= 0:
        raise SystemExit("horizon, candidate cap, and repeat count must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    skipped: dict[str, Any] = {}
    for split, offset, scenario_name in (
        ("train", 0, "mixed"),
        ("validation", 1_000_000, "mixed"),
        ("test", 2_000_000, "mixed"),
        ("ood", 3_000_000, "energy_insufficient"),
    ):
        if counts[split] == 0:
            continue
        rows, skipped_rows = make_split(
            split,
            counts[split],
            args.base_seed,
            args.prefix_steps,
            args.horizon_steps,
            args.max_candidates_per_prefix,
            args.exogenous_repeats,
            scenario_name,
        )
        path = args.out / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        files[split] = _file_spec(path, rows)
        skipped[split] = skipped_rows

    manifest = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "status": "generated_for_development_only",
        "task_completion_mode": "arrival_to_region",
        "primary_deadline_basis": "physical_arrival",
        "secondary_deadline_basis": "host_confirmation_not_in_joint_target",
        "observation_contract": "m10-graph5-5type-25action-global27",
        "joint_task_set_contract": "public_pending_tasks_at_prefix; identical across candidate branches",
        "label_contract": {
            "primary": "new_on_time_count_for_fixed_public_task_set",
            "secondary": "deadline_failure_count_for_fixed_public_task_set",
            "paired": "candidate_minus_predeclared_traditional_reference",
            "future_tasks_excluded": True,
            "full_episode_q_value": False,
            "censored_not_failure": True,
        },
        "continuation": {
            "controller": DEFAULT_CONTINUATION_CONTROLLER,
            "implementation": "traditional_public_controller_v1",
            "first_step": "candidate_action",
            "following_steps": "traditional_action_from_public_observation",
        },
        "shared_exogenous_randomness": {
            "scenario_tape_shared": True,
            "key_shared_within_prefix_repeat": True,
            "hidden_state_online": False,
            "note": "simulator key and branch trace digests are recorded; equality of a string alone is not treated as proof",
        },
        "generation": {
            "base_seed": args.base_seed,
            "counts": counts,
            "prefix_steps": args.prefix_steps,
            "horizon_steps": args.horizon_steps,
            "max_candidates_per_prefix": args.max_candidates_per_prefix,
            "exogenous_repeats": args.exogenous_repeats,
            "scenario_names": {"train": "mixed", "validation": "mixed", "test": "mixed", "ood": "energy_insufficient"},
            "result_filtering": "none; deterministic public eligibility only",
        },
        "files": files,
        "skipped_prefixes": skipped,
        "source": {
            "generator": "tools/generate_m10_joint_consequence_dataset.py",
            "schema_adapter": "gppo_world.joint_consequence",
            "protocol": PROTOCOL,
            "sha256": {
                "generator": file_sha256(Path(__file__)),
                "schema_adapter": file_sha256(ROOT / "gppo_world" / "joint_consequence.py"),
                "baseline": file_sha256(ROOT / "gppo_world" / "joint_consequence_baseline.py"),
                "protocol_config": file_sha256(ROOT / "configs" / "world-gppo-9.11-joint-consequence-v0.1.0.json"),
            },
        },
    }
    _write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"out": str(args.out), "files": files, "skipped_prefixes": skipped}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
