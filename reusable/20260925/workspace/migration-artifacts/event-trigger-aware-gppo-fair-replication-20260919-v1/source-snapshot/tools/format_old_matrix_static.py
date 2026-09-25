"""Compact the preserved old formal matrix for the main task preference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    for condition in ("I", "W1", "W2"):
        for seed in ("1101", "2203", "3307"):
            rows = []
            for group in ("A", "B", "C", "D"):
                key = "not_applicable" if group == "A" else "(0.8,0.2)"
                item = data["groups"][condition][seed][group][key]
                rows.append({
                    "condition": condition, "seed": seed, "group": group,
                    "physical": f"{item['physical_on_time']}/{item['tasks_total']}",
                    "host_on_time": item["host_on_time"],
                    "noop_ratio": item["noop_ratio"],
                    "energy_used_mean": item["energy_used_mean"],
                    "safety": item["safety"], "illegal": item["illegal"],
                })
            print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
