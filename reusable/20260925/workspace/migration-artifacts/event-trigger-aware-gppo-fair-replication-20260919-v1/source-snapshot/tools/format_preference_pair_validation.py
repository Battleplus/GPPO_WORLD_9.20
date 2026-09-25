"""Print compact P/W frozen-validation summaries without loading final-test."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def aggregate(rows):
    out = {}
    for condition in ("I", "W1", "W2"):
        selected = [row for row in rows if str(row.get("tape_id", "")).endswith("-" + condition)]
        keys = ("tasks_total", "physical_on_time", "host_on_time", "host_confirmed_final", "deadline_failed", "unresolved", "illegal_action_count", "explicit_safety_violations_recorded")
        block = {key: sum(int(row.get(key, 0) or 0) for row in selected) for key in keys}
        block["episodes"] = len(selected)
        block["energy_used_mean"] = (sum(float(row.get("energy_used", 0.0) or 0.0) for row in selected) / len(selected)) if selected else None
        metrics = [row.get("decision_metrics", {}) for row in selected]
        block["actor_calls"] = sum(int(m.get("actor_calls", 0) or 0) for m in metrics)
        block["world_forward_calls"] = sum(int(m.get("world_forward_calls", 0) or 0) for m in metrics)
        block["communication_proxy_bytes"] = sum(int(m.get("communication_proxy_bytes", 0) or 0) for m in metrics)
        latencies = [float(value) for metric in metrics for value in metric.get("complete_decision_latency_ms", [])]
        block["decision_latency_mean_ms"] = (sum(latencies) / len(latencies)) if latencies else None
        block["decision_latency_p95_ms"] = (sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None)
        out[condition] = block
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    for seed, groups in data["groups"].items():
        for group, payload in groups.items():
            for pref, block in payload["preferences"].items():
                summary = block["summary"]
                print(json.dumps({
                    "seed": seed, "group": group, "preference": pref,
                    "overall": {key: summary.get(key) for key in ("episodes", "tasks_total", "physical_on_time", "physical_on_time_rate", "host_on_time", "host_confirmed_final", "deadline_failed", "unresolved", "energy_used_mean", "illegal_actions", "explicit_safety_violations_recorded", "communication_proxy_messages", "communication_proxy_bytes")},
                    "by_condition": aggregate(block["episodes"]),
                }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
