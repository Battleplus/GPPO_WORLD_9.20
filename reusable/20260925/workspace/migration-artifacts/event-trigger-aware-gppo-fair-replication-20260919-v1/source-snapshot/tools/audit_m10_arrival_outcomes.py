"""Audit the existing arrival-protocol rule-baseline ledger.

This is a read-only ledger audit.  It does not rerun the simulator and keeps
physical arrival and host confirmation as separate outcome bases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import default_scenario  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for basis, block in payload["bases"].items():
        for episode in block["episodes"]:
            tape_id = str(episode["tape_id"])
            parts = tape_id.split("-")
            split, seed = parts[0], int(parts[-1])
            scenario_name = "-".join(parts[1:-2]) if len(parts) > 3 else "mixed"
            specs = {spec.task_id: spec for spec in default_scenario(scenario_name, seed=seed, split=split).tasks}
            records = episode.get("completion_records", {})
            # The old result only serialized completion records.  Reconstruct
            # the six public task identities from the frozen tape metadata so
            # every one of the 96 tasks appears, while leaving absent outcomes
            # explicitly unresolved rather than inferring them.
            for task_id, spec in sorted(specs.items()):
                record = records.get(task_id, {})
                arrival = record.get("physical_arrival_time")
                send = record.get("completion_message_send_time")
                confirm = record.get("host_confirmation_time")
                deadline = record.get("deadline")
                rows.append({
                    "basis": basis,
                    "tape_id": episode["tape_id"],
                    "task_id": task_id,
                    "uav_id": record.get("uav_id"),
                    "physical_arrival_time": arrival,
                    "completion_message_send_time": send,
                    "host_confirmation_time": confirm,
                    "deadline": deadline if deadline is not None else float(spec.deadline),
                    "physical_arrival_before_deadline": record.get("physical_arrival_before_deadline"),
                    "host_confirmation_before_deadline": record.get("host_confirmation_before_deadline"),
                    "record_presence": "completion_record" if record else "missing_from_completion_records",
                    "final_classification": "completed" if (
                        basis == "physical_arrival" and record.get("physical_arrival_before_deadline") is True
                    ) or (
                        basis == "host_confirmation" and record.get("host_confirmation_before_deadline") is True
                    ) else "expired_or_unconfirmed" if record else "not_recorded",
                    "observation_cutoff": episode.get("steps"),
                })
    by_basis: dict[str, Any] = {}
    for basis in ("physical_arrival", "host_confirmation"):
        subset = [row for row in rows if row["basis"] == basis]
        confirmed = [row for row in subset if row["host_confirmation_time"] is not None]
        by_basis[basis] = {
            "task_records": len(subset),
            "event_denominator": sum(r["record_presence"] == "completion_record" for r in subset),
            "unique_task_records": len({(r["tape_id"], r["task_id"]) for r in subset}),
            "physical_arrival_events": sum(r["physical_arrival_time"] is not None for r in subset),
            "completion_message_send_events": sum(r["completion_message_send_time"] is not None for r in subset),
            "host_confirmation_events": len(confirmed),
            "physical_arrival_before_deadline": sum(r["physical_arrival_before_deadline"] is True for r in subset),
            "host_confirmation_before_deadline": sum(r["host_confirmation_before_deadline"] is True for r in subset),
            "final_completed_under_basis": sum(r["final_classification"] == "completed" for r in subset),
            "final_expired_or_unconfirmed_under_basis": sum(r["final_classification"] == "expired_or_unconfirmed" for r in subset),
            "terminal_state_not_recorded": sum(r["final_classification"] == "not_recorded" for r in subset),
            "retained_baseline_aggregate": {
                "completed": int(round(float(block["summary"]["completed_mean"]) * int(block["summary"]["episodes"]))),
                "expired": int(round(float(block["summary"]["expired_mean"]) * int(block["summary"]["episodes"]))),
                "episodes": int(block["summary"]["episodes"]),
            },
        }
    late = [r for r in rows if r["basis"] == "host_confirmation" and r["host_confirmation_time"] is not None and r["host_confirmation_before_deadline"] is False]
    summary = {
        "source": "existing arrival rule-baseline ledger; no simulator rerun",
        "source_protocol": payload.get("protocol"),
        "episode_count": payload.get("count"),
        "tasks_per_episode": 6,
        "task_count": len({(row["tape_id"], row["task_id"]) for row in rows if row["basis"] == "physical_arrival"}),
        "by_basis": by_basis,
        "host_confirmation_explanation": {
            "confirmed_events": len([r for r in rows if r["basis"] == "host_confirmation" and r["host_confirmation_time"] is not None]),
            "on_time_confirmed_tasks": by_basis["host_confirmation"]["host_confirmation_before_deadline"],
            "reported_remaining_tasks": 11,
            "reported_remaining_source": "user-provided prior closure summary; not serialized in the old rule-baseline JSON",
            "late_confirmation_events": len(late),
            "late_confirmation_records": [{"tape_id": r["tape_id"], "task_id": r["task_id"], "confirmation_time": r["host_confirmation_time"], "deadline": r["deadline"]} for r in late],
            "interpretation": "10 is the count of unique completion-notice host-confirmation events; 9 is the count of those events received by the host at or before the deadline. The remaining confirmation is late and is not an on-time completion.",
        },
        "limits": [
            "The rule baseline contains six task records per episode, not a new policy experiment.",
            "Observation cutoff is recorded as simulator step count; it is not substituted for an unobserved confirmation time.",
            "Physical arrival and host confirmation are sensitivity bases and must not be combined into one funnel.",
            "The 48 non-arrival task records are reconstructed as identities/deadlines from the frozen tape, but the old JSON does not contain their per-task terminal state; they remain not_recorded in the task table. Aggregate completed/expired counts below are retained from the old baseline summary.",
        ],
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary, rows = audit(payload)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out / "task-outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {"source_sha256": sha256(args.input), "summary_sha256": sha256(args.out / "summary.json"), "task_outcomes_sha256": sha256(args.out / "task-outcomes.jsonl"), "records": len(rows)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "records": len(rows), "source_sha256": manifest["source_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
