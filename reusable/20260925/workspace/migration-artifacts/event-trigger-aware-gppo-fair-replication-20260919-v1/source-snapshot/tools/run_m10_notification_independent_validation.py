"""Independent paired validation of original vs bounded completion notification.

This runner intentionally excludes the rejected scheduling and combined variants.
It reuses the frozen environment and episode runner; it does not train or alter
the notification mechanism. The parent tapes are disjoint from the known
development regression tapes by using base_seed=193001.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import weak_communication_tape  # noqa: E402
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402
from tools.run_m10_targeted_improvements import run_episode  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_variant(level: str, count: int, base_seed: int, *, reliable: bool) -> dict:
    episodes = []
    for scenario in weak_communication_tape("test", count=count, base_seed=base_seed, level=level, name="mixed"):
        episodes.append(run_episode(level, scenario, traditional_action, reliable=reliable))
    rows = [row for episode in episodes for row in episode["tasks"]]
    summaries = [episode["summary"] for episode in episodes]
    wall_samples = [item["wall_ms"] for episode in episodes for item in episode["steps"]]
    communication_status_by_link: dict[str, dict[str, int]] = {}
    communication_proxy_bytes_by_link: dict[str, int] = {}
    for episode in episodes:
        for item in episode["communication_log"]:
            link = str(item.get("link", "unknown"))
            status = str(item.get("status", "unknown"))
            communication_status_by_link.setdefault(link, {})[status] = communication_status_by_link.setdefault(link, {}).get(status, 0) + 1
            encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            communication_proxy_bytes_by_link[link] = communication_proxy_bytes_by_link.get(link, 0) + len(encoded)
    return {
        "episodes": episodes,
        "tasks": rows,
        "summary": {
            "episodes": len(episodes),
            "tasks": len(rows),
            "physical_arrival_events": sum(row["physical_arrival_time"] is not None for row in rows),
            "on_time_physical": sum(row["physical_arrival_before_deadline"] is True for row in rows),
            "host_confirmation_events": sum(row["host_confirmation_time"] is not None for row in rows),
            "on_time_host_confirmation": sum(row["host_confirmation_before_deadline"] is True for row in rows),
            "physical_status": dict(Counter(row["physical_status"] for row in rows)),
            "host_status": dict(Counter(row["host_status"] for row in rows)),
            "primary_failure_causes": dict(Counter(row["primary_failure_cause"] for row in rows if row["primary_failure_cause"])),
            "host_confirmation_causes": dict(Counter(row["host_confirmation_cause"] for row in rows if row["host_confirmation_cause"])),
            "observed_safety_violations": sum(item["observed_safety_violations"] for item in summaries),
            "guard_rejections": sum(item["guard_rejections"] for item in summaries),
            "communication_status_by_link": communication_status_by_link,
            "communication_proxy_bytes_by_link": communication_proxy_bytes_by_link,
            "return_mean": sum(float(episode["return"]) for episode in episodes) / len(episodes),
            "decision_wall_ms": {
                "n": len(wall_samples),
                "mean": sum(wall_samples) / len(wall_samples) if wall_samples else None,
                "max": max(wall_samples) if wall_samples else None,
                "p95": sorted(wall_samples)[max(0, int(0.95 * len(wall_samples)) - 1)] if wall_samples else None,
            },
            "explicit_task_rows": len(rows),
            "unresolved_rows": sum(row["physical_status"] == "pending_at_cutoff" or row["host_status"] == "pending_at_cutoff" for row in rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=193001)
    args = parser.parse_args()
    if args.out.exists() or args.count != 16:
        raise SystemExit("output must be new and count must remain the frozen value 16")
    args.out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    variants: dict[str, dict] = {}
    for name, reliable in (("original", False), ("notification", True)):
        variants[name] = {level: run_variant(level, args.count, args.base_seed, reliable=reliable) for level in ("ideal", "composite")}
    result = {
        "run_id": args.run_id,
        "protocol": "world-gppo-9.11-arrival/0.1.0",
        "task_completion_mode": "arrival_to_region",
        "deadline_basis_primary": "physical_arrival",
        "training_performed": False,
        "world_model_used": False,
        "independent_parent_tape_set": True,
        "development_parent_tape_base_seed": 93001,
        "independent_parent_tape_base_seed": args.base_seed,
        "count_per_level_per_variant": args.count,
        "variant_scope": ["original", "notification"],
        "rejected_variants_not_run": ["scheduling", "combined"],
        "parameters": {"notification_retry_interval": 1.0, "notification_max_retries": 2, "notification_retention": 3.0},
        "source": {"runner_sha256": sha256_file(Path(__file__)), "targeted_runner_sha256": sha256_file(ROOT / "tools" / "run_m10_targeted_improvements.py"), "environment_sha256": sha256_file(ROOT / "gppo_world" / "m10_environment.py"), "config_sha256": sha256_file(ROOT / "configs" / "world-gppo-9.11-arrival-v0.1.0.json")},
        "runtime": {"started_at": started, "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "variants": variants,
    }
    (args.out / "notification-independent-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({name: {level: data["summary"] for level, data in levels.items()} for name, levels in variants.items()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
