"""Read-only schema and aggregate inspection for preference-pair ledgers."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = sorted(root.glob("training/seed-*/[PW]/training-ledger.jsonl"))
    for path in paths:
        rows = 0
        first = None
        last = None
        conditions = Counter()
        preferences = Counter()
        tapes = Counter()
        episode_groups = {}
        feedback = Counter()
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            rows += 1
            if first is None:
                first = row
            last = row
            for key in ("condition", "communication_condition", "tape_condition"):
                if key in row:
                    conditions[str(row[key])] += 1
            for key in ("preference", "episode_preference"):
                if key in row:
                    pref = row[key]
                    if isinstance(pref, list) and len(pref) == 2:
                        preferences["task_dominant" if pref[0] > 0.75 else "energy_dominant" if pref[1] > 0.75 else "mixed"] += 1
                    else:
                        preferences[str(pref)] += 1
            tape_id = str(row.get("tape_id", "unknown"))
            tapes[tape_id] += 1
            group_key = (tape_id, str(row.get("episode_index", "unknown")))
            episode_groups[group_key] = row
            ef = row.get("execution_feedback")
            if isinstance(ef, dict):
                for key in ("status", "reason", "action_type"):
                    if key in ef:
                        feedback[f"{key}={ef[key]}"] += 1
                if ef.get("command_submitted") is True:
                    feedback["command_submitted=true"] += 1
        selected = {}
        if first is not None:
            selected = {k: first.get(k) for k in (
                "run_id", "group", "seed", "condition", "communication_condition",
                "tape_id", "scenario_id", "episode_id", "episode_index", "step",
                "action", "reward", "preference", "terminated", "truncated",
            ) if k in first}
        print(json.dumps({
            "path": str(path),
            "rows": rows,
            "episodes": len(episode_groups),
            "tapes": dict(tapes),
            "first_selected": selected,
            "last_step": None if last is None else last.get("step"),
            "top_level_keys": [] if first is None else sorted(first),
            "condition_fields": dict(conditions),
            "preference_fields": dict(preferences),
            "episode_end_totals": [
                {
                    "tape_id": k[0],
                    "episode_index": k[1],
                    "physical_on_time_total": v.get("physical_on_time_total"),
                    "physical_deadline_failure_total": v.get("physical_deadline_failure_total"),
                    "host_on_time_total": v.get("host_on_time_total"),
                    "host_confirmed_total": v.get("host_confirmed_total"),
                    "time": v.get("time"),
                }
                for k, v in sorted(episode_groups.items(), key=lambda item: (item[0][0], item[0][1]))
            ],
            "feedback_fields": feedback.most_common(20),
        }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
