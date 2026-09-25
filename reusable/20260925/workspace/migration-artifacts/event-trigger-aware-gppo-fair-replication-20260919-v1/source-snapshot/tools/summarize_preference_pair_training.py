"""Stream the six preference-pair ledgers into compact, read-only summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pref_bin(pref: object) -> str:
    if isinstance(pref, list) and len(pref) == 2:
        if float(pref[0]) >= 0.75:
            return "task_dominant"
        if float(pref[1]) >= 0.75:
            return "energy_dominant"
    return "mixed"


def add_value(target: dict[str, object], key: str, value: object) -> None:
    bucket = target.setdefault(key, {})
    if isinstance(value, dict):
        for name, amount in value.items():
            if isinstance(amount, (int, float)):
                bucket[name] = float(bucket.get(name, 0.0)) + float(amount)
    elif isinstance(value, (int, float)):
        bucket["scalar"] = float(bucket.get("scalar", 0.0)) + float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    outputs: list[dict[str, object]] = []
    for path in sorted(root.glob("training/seed-*/[PW]/training-ledger.jsonl")):
        episodes: dict[tuple[str, str], dict[str, object]] = {}
        counts = defaultdict(lambda: {
            "episodes": 0, "tasks": 0, "physical_on_time": 0,
            "physical_deadline_failure": 0, "host_on_time": 0,
            "host_confirmed": 0, "reward_sum": {}, "training_reward_sum": {},
            "decision_rows": 0, "noop_rows": 0, "rows_with_non_noop_legal": 0,
            "command_submitted": 0, "proxy_messages": 0,
        })
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            key = (str(row.get("tape_id", "unknown")), str(row.get("episode_index", "unknown")))
            episodes[key] = row
        for (tape_id, _episode_index), row in episodes.items():
            condition = tape_id.rsplit("-", 1)[-1]
            pref = pref_bin(row.get("preference"))
            for bucket in ("all", condition, f"{condition}/{pref}"):
                item = counts[bucket]
                item["episodes"] += 1
                item["tasks"] += 6
                item["physical_on_time"] += int(row.get("physical_on_time_total", 0) or 0)
                item["physical_deadline_failure"] += int(row.get("physical_deadline_failure_total", 0) or 0)
                item["host_on_time"] += int(row.get("host_on_time_total", 0) or 0)
                item["host_confirmed"] += int(row.get("host_confirmed_total", 0) or 0)
                add_value(item, "reward_sum", row.get("raw_environment_reward", 0.0))
                add_value(item, "training_reward_sum", row.get("training_reward", 0.0))
            # Decision-level counters are accumulated separately below.
        for path_line in path.open(encoding="utf-8"):
            row = json.loads(path_line)
            condition = str(row.get("tape_id", "unknown")).rsplit("-", 1)[-1]
            for bucket in ("all", condition):
                item = counts[bucket]
                item["decision_rows"] += 1
                if int(row.get("action", -1)) == 0:
                    item["noop_rows"] += 1
                if int(row.get("legal_action_count", 0) or 0) > 1:
                    item["rows_with_non_noop_legal"] += 1
                feedback = row.get("execution_feedback")
                if isinstance(feedback, dict) and feedback.get("command_submitted") is True:
                    item["command_submitted"] += 1
                item["proxy_messages"] += int(row.get("communication_proxy_messages", 0) or 0)
        outputs.append({
            "path": str(path),
            "seed": path.parts[-3],
            "group": path.parts[-2],
            "buckets": {k: dict(v) for k, v in sorted(counts.items())},
        })
    print(json.dumps(outputs, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
