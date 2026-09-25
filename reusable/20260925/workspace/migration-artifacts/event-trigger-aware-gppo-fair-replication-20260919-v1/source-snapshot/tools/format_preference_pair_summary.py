"""Format the remote preference-pair audit JSON into a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(item: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {
        "seed": item["seed"],
        "group": item["group"],
        "path": item["path"],
        "buckets": {},
    }
    buckets = item.get("buckets", {})
    selected = ["all", "I", "W1", "W2", "I/task_dominant", "I/energy_dominant", "I/mixed",
                "W1/task_dominant", "W1/energy_dominant", "W1/mixed",
                "W2/task_dominant", "W2/energy_dominant", "W2/mixed"]
    for key in selected:
        if key not in buckets:
            continue
        value = dict(buckets[key])
        value["physical_rate"] = None if not value.get("tasks") else value["physical_on_time"] / value["tasks"]
        value["host_rate"] = None if not value.get("tasks") else value["host_on_time"] / value["tasks"]
        value["noop_rate"] = None if not value.get("decision_rows") else value["noop_rows"] / value["decision_rows"]
        out["buckets"][key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    for item in data:
        print(json.dumps(compact(item), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
