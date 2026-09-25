"""Summarize the read-only frozen P/T validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SEEDS = (1101, 2203, 3307)
CONDITIONS = ("I-ideal", "W1-light", "W2-moderate")
MODELS = ("P_train_periodic", "P_train_triggered", "T_train_triggered")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parent_id(tape_id: str) -> str:
    return re.sub(r"-(?:I|W1|W2)$", "", tape_id)


def aggregate(rows: list[dict]) -> dict:
    tasks = sum(int(r["total_tasks"]) for r in rows)
    completed = sum(int(r["completed"]) for r in rows)
    return {
        "episodes": len(rows),
        "environment_steps": sum(int(r["steps"]) for r in rows),
        "tasks": tasks,
        "completed": completed,
        "completion_rate": completed / tasks if tasks else None,
        "host_on_time": sum(int(r["host_on_time"]) for r in rows),
        "host_confirmed": sum(int(r["host_confirmed"]) for r in rows),
        "expired": sum(int(r["expired"]) for r in rows),
        "undecided": sum(int(r["undecided"]) for r in rows),
        "energy_used_sum": sum(float(r["energy_used"]) for r in rows),
        "energy_used_mean_episode": float(np.mean([r["energy_used"] for r in rows])) if rows else None,
        "actor_calls": sum(int(r["actor_calls"]) for r in rows),
        "continuation_steps": sum(int(r["continuation_steps"]) for r in rows),
        "world_calls": sum(int(r["world_calls"]) for r in rows),
        "candidate_calls": sum(int(r["candidate_calls"]) for r in rows),
        "event_calls": sum(int(r["event_calls"]) for r in rows),
        "command_submitted": sum(int(r["command_submitted"]) for r in rows),
        "command_accepted": sum(int(r["command_accepted"]) for r in rows),
        "noop_opportunities": sum(int(r["noop_opportunities"]) for r in rows),
        "noop_selected": sum(int(r["noop_selected"]) for r in rows),
        "noop_rate_when_available": (
            sum(int(r["noop_selected"]) for r in rows) /
            sum(int(r["noop_opportunities"]) for r in rows)
            if sum(int(r["noop_opportunities"]) for r in rows) else None
        ),
        "illegal_new_actions": sum(int(r["illegal_new_actions"]) for r in rows),
        "illegal_continuations": sum(int(r["illegal_continuations"]) for r in rows),
        "security_violations": sum(int(r["security_violations"]) for r in rows),
        "communication_messages": sum(int(r["communication_messages"]) for r in rows),
        "communication_proxy_bytes": (
            sum(int(r["communication_proxy_bytes"]) for r in rows)
            if rows and all(r["communication_bytes_observed"] for r in rows) else None
        ),
        # The evaluator stores per-episode timing summaries, not raw decision samples.
        "decision_timing_episode_summary": {
            key: float(np.mean([r["decision_chain_timing"][key] for r in rows])) if rows else None
            for key in ("mean_ms", "p95_ms", "p99_ms")
        },
        "model_timing_episode_summary": {
            key: float(np.mean([r["model_forward_timing"][key] for r in rows])) if rows else None
            for key in ("mean_ms", "p95_ms", "p99_ms")
        },
        "reward_vector_sum": [
            float(sum(float(r["reward_vector"][i]) for r in rows)) for i in range(2)
        ],
    }


def bootstrap_parent(rows: list[dict], left: str, right: str, conditions: tuple[str, ...]) -> dict:
    # First average seed and condition effects inside each parent tape; only then resample parents.
    by_parent: dict[str, dict[tuple[str, int], dict]] = {}
    for row in rows:
        if row["model"] not in (left, right) or row["condition"]["name"] not in conditions:
            continue
        key = (row["model"], int(row["seed"]))
        by_parent.setdefault(parent_id(row["tape_id"]), {})[key + (row["condition"]["name"],)] = row
    parent_values = []
    for pid, entries in sorted(by_parent.items()):
        diffs = []
        for seed in SEEDS:
            for condition in conditions:
                l = entries.get((left, seed, condition))
                r = entries.get((right, seed, condition))
                if l is None or r is None:
                    continue
                diffs.append(l["completed"] / l["total_tasks"] - r["completed"] / r["total_tasks"])
        if diffs:
            parent_values.append(float(np.mean(diffs)))
    if not parent_values:
        return {"parents": 0, "mean": None, "ci95": [None, None], "values": []}
    values = np.asarray(parent_values, dtype=np.float64)
    rng = np.random.default_rng(20260919)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "parents": len(values), "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_resamples": 10000,
        "values": values.tolist(),
        "unit": "parent tape; seed and W1/W2 effects averaged within parent before resampling",
        "does_not_cover_training_seed_uncertainty": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = read_rows(args.episodes)
    table = {}
    for model in MODELS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                selected = [r for r in rows if r["model"] == model and int(r["seed"]) == seed and r["condition"]["name"] == condition]
                table[f"{model}|{seed}|{condition}"] = aggregate(selected)
    summary = {
        "schema": "frozen-pt-validation-summary/1.0.0",
        "episodes_sha256": hashlib.sha256(args.episodes.read_bytes()).hexdigest(),
        "episodes": len(rows),
        "expected_episodes": 432,
        "models": MODELS,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "preference": [0.8, 0.2],
        "table": table,
        "comparisons": {
            "main_T_triggered_minus_P_triggered_W1_W2_equal_weight": bootstrap_parent(rows, "T_train_triggered", "P_train_triggered", ("W1-light", "W2-moderate")),
            "secondary_T_triggered_minus_P_periodic_W1_W2_equal_weight": bootstrap_parent(rows, "T_train_triggered", "P_train_periodic", ("W1-light", "W2-moderate")),
        },
        "timing_limit": "P95/P99 are averages of per-episode timing summaries; raw decision-level samples were not stored, so exact pooled quantiles are unavailable.",
        "independent_test_read": False,
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
