"""Read-only analysis of the completed fair-replication ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1")
CONFIGS = ("A", "B", "C")
SEEDS = (1101, 2203, 3307)
CONDITIONS = ("W1", "W2")
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260919


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    rows = [json.loads(line) for line in (ROOT / "evaluation" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row["condition"] in CONDITIONS]
    if len(selected) != 1152:
        raise RuntimeError(f"expected 1152 W1/W2 rows, got {len(selected)}")
    rates = {(row["configuration"], int(row["seed"]), row["condition"], row["parent_tape_id"]): row["physical_on_time"] / row["task_count"] for row in selected}
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for label in CONFIGS:
            per_seed[str(seed)][label] = {condition: float(np.mean([rates[(label, seed, condition, p)] for p in sorted({k[3] for k in rates if k[0] == label and k[1] == seed and k[2] == condition})])) for condition in CONDITIONS}
        per_seed[str(seed)]["C_minus_B"] = float(np.mean([per_seed[str(seed)]["C"][condition] - per_seed[str(seed)]["B"][condition] for condition in CONDITIONS]))
        per_seed[str(seed)]["C_minus_A"] = float(np.mean([per_seed[str(seed)]["C"][condition] - per_seed[str(seed)]["A"][condition] for condition in CONDITIONS]))

    parents = sorted({row["parent_tape_id"] for row in selected})
    parent_values = {"C_minus_B": [], "C_minus_A": []}
    for parent in parents:
        c = np.mean([rates[("C", seed, condition, parent)] for seed in SEEDS for condition in CONDITIONS])
        b = np.mean([rates[("B", seed, condition, parent)] for seed in SEEDS for condition in CONDITIONS])
        a = np.mean([rates[("A", seed, condition, parent)] for seed in SEEDS for condition in CONDITIONS])
        parent_values["C_minus_B"].append(float(c - b)); parent_values["C_minus_A"].append(float(c - a))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {}
    for name, values in parent_values.items():
        values = np.asarray(values, dtype=np.float64)
        draws = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
        samples = values[draws].mean(axis=1)
        bootstrap[name] = {"point": float(values.mean()), "ci95_lower": percentile(samples, 2.5), "ci95_upper": percentile(samples, 97.5),
                           "parent_count": len(values), "bootstrap_count": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED}

    latencies = [lat for row in rows for lat in row["decision_latency_ns"]]
    by_seed = {str(seed): [lat for row in rows if int(row["seed"]) == seed for lat in row["decision_latency_ns"]] for seed in SEEDS}
    by_condition = {condition: [lat for row in rows if row["condition"] == condition for lat in row["decision_latency_ns"]] for condition in ("I", "W1", "W2")}
    latency = {"pooled_count": len(latencies), "pooled_p95_ns": percentile(latencies, 95), "pooled_p99_ns": percentile(latencies, 99),
               "boundary": "probe+trigger+action; env.step and persistence excluded",
               "by_seed": {key: {"count": len(value), "p95_ns": percentile(value, 95), "p99_ns": percentile(value, 99)} for key, value in by_seed.items()},
               "by_condition": {key: {"count": len(value), "p95_ns": percentile(value, 95), "p99_ns": percentile(value, 99)} for key, value in by_condition.items()}}

    cost = {}
    for label in CONFIGS:
        picked = [row for row in rows if row["configuration"] == label]
        cost[label] = {"environment_steps": sum(row["steps"] for row in picked), "actor_calls": sum(row["actor_calls"] for row in picked),
                       "continuation_steps": sum(row["continuation_steps"] for row in picked), "world_forward_calls": sum(row["world_forward_calls"] for row in picked),
                       "optimizer_calls": 0}
    training = {}
    for seed in SEEDS:
        for group in ("P_train", "T_train"):
            path = ROOT / "training" / f"seed-{seed}" / group / "training-summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            training[f"{group}/seed-{seed}"] = {key: payload.get(key) for key in ("environment_steps", "policy_optimizer_steps", "world_optimizer_steps", "actor_calls", "continuation_steps", "world_forward_calls", "replay_forward_calls", "checkpoint_sha256", "source_checkpoint_sha256")}

    report = {"schema": "fair-replication-analysis/1.0.0", "episode_count": len(rows), "selected_condition_scope": CONDITIONS,
              "physical_arrival_rate": {label: float(np.mean([rates[(label, seed, condition, parent)] for seed in SEEDS for condition in CONDITIONS for parent in parents])) for label in CONFIGS},
              "per_seed": per_seed, "bootstrap": bootstrap, "latency": latency, "evaluation_cost": cost,
              "training": training, "safety": {"max_security_violations": max(int(row["security_violations"]) for row in rows), "illegal_actions": sum(int(row["illegal_actions"]) for row in rows), "noop_selected": sum(int(row["noop_selected"]) for row in rows)},
              "acceptance": {"main_comparison": "C_minus_B", "threshold": 0.01, "improvement_points": bootstrap["C_minus_B"]["point"],
                             "at_least_two_of_three_seed_strict": sum(per_seed[str(seed)]["C_minus_B"] > 0 for seed in SEEDS) >= 2,
                             "bootstrap_lower_gt_zero": bootstrap["C_minus_B"]["ci95_lower"] > 0,
                             "no_new_security_violations": max(int(row["security_violations"]) for row in rows) == 0,
                             "passed": bootstrap["C_minus_B"]["point"] >= 0.01 and sum(per_seed[str(seed)]["C_minus_B"] > 0 for seed in SEEDS) >= 2 and bootstrap["C_minus_B"]["ci95_lower"] > 0 and max(int(row["security_violations"]) for row in rows) == 0}}
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "analysis" / "fair-replication-analysis.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"physical_arrival_rate": report["physical_arrival_rate"], "bootstrap": bootstrap, "acceptance": report["acceptance"], "latency": latency}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
