#!/usr/bin/env python3
"""Summarize the already-run weak-communication boundary audit.

This tool is intentionally evaluation-only: it never runs an environment or
checkpoint and it never changes the source audit.  It separates strict event
recovery from the audit's four-way episode comparison, because an episode can
finish without the injected fault having interrupted the task.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _strict_event_class(summary: dict[str, Any]) -> str:
    """Return a conservative class for an actually interrupted task.

    Positive slack is not silently converted into "enough time".  A row that
    misses the deadline after knowledge is reported as time/physical-budget
    failure, but the output retains the measured positive slack so readers do
    not mistake it for a zero-slack proof.
    """

    if not summary.get("interrupted"):
        return "not_an_interrupted_recovery_event"
    if summary.get("legal_event_knowledge") is None:
        return "information_missing_or_stale"
    if summary.get("first_public_candidate_time_after_knowledge") is None:
        return "information_missing_or_stale"
    if summary.get("recovery_completion_success"):
        return "information_and_execution_sufficient"
    if "not_completed_before_deadline" in summary.get("failure_reasons", []):
        return "time_or_physical_budget_insufficient_after_knowledge"
    if not summary.get("recovery_service_success"):
        return "information_present_but_execution_or_resource_not_successful"
    return "other_or_insufficient_evidence"


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    ref = row["reference"]
    return {
        "split": row.get("split"),
        "tape_id": row.get("tape_id"),
        "category": row.get("category"),
        "factor": row.get("factor"),
        "scenario": row.get("scenario"),
        "event": ref.get("event"),
        "affected_task": ref.get("affected_task"),
        "interrupted": bool(ref.get("interrupted")),
        "knowledge_time": ref.get("legal_knowledge_time"),
        "knowledge_basis": ref.get("knowledge_basis"),
        "has_legal_knowledge": ref.get("legal_event_knowledge") is not None,
        "first_public_candidate_time": ref.get(
            "first_public_candidate_time_after_knowledge"
        ),
        "deadline_slack_at_knowledge": ref.get("deadline_slack_at_knowledge"),
        "accepted_alternate_count": len(ref.get("accepted_alternate_commands", [])),
        "service_recovered": bool(ref.get("recovery_service_success")),
        "task_completed": bool(ref.get("task_completed")),
        "strict_recovery_completion": bool(ref.get("recovery_completion_success")),
        "failure_reasons": list(ref.get("failure_reasons", [])),
        "security_rejections": list(ref.get("security_rejections", [])),
        "scope_class": _strict_event_class(ref),
    }


def summarize(audit: dict[str, Any]) -> dict[str, Any]:
    full = [_row_summary(row) for row in audit["full_pool"]["rows"]]
    scan = [_row_summary(row) for row in audit["boundary_factor_scan"]["rows"]]
    interrupted = [row for row in full if row["interrupted"]]
    scan_interrupted = [row for row in scan if row["interrupted"]]

    def counters(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(row["scope_class"] for row in rows).items()))

    def metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "interrupted_events": len(rows),
            "legal_knowledge": sum(row["has_legal_knowledge"] for row in rows),
            "public_candidate": sum(
                row["first_public_candidate_time"] is not None for row in rows
            ),
            "service_recovered": sum(row["service_recovered"] for row in rows),
            "deadline_completed": sum(row["strict_recovery_completion"] for row in rows),
            "task_completed_anyway": sum(row["task_completed"] for row in rows),
            "security_rejection_rows": sum(bool(row["security_rejections"]) for row in rows),
        }

    by_split: dict[str, Any] = {}
    for split in sorted({row["split"] for row in full}):
        split_rows = [row for row in interrupted if row["split"] == split]
        by_split[split] = {
            "metrics": metrics(split_rows),
            "scope_classes": counters(split_rows),
        }

    factors: dict[str, Any] = {}
    for factor in sorted({row["factor"] for row in scan}):
        factor_rows = [row for row in scan_interrupted if row["factor"] == factor]
        factors[factor] = {
            "metrics": metrics(factor_rows),
            "scope_classes": counters(factor_rows),
            "rows": factor_rows,
        }

    return {
        "format": "m10-weak-communication-scope-summary/1.0.0",
        "source": {
            "audit_format": audit.get("format"),
            "audit_scope": audit.get("scope"),
            "server_audit_sha256": "fddf511dc182cc86c881779d060eb08a70e4e838816b415de60473b4df0793cd",
            "server_root": "/home/user1/m10-runs/20260908-recovery-boundary-v3",
            "rerun": False,
            "training_started": False,
            "B_started": False,
            "C_started": False,
        },
        "definitions": {
            "strict_recovery": "interrupted task + legal post-fault knowledge + public candidate + post-event alternate service + completion before deadline + zero security rejection",
            "information_missing_or_stale": "no legal knowledge or no public candidate after legal knowledge; this is not a claim that every such row has a publisher bug",
            "time_or_physical_budget_insufficient_after_knowledge": "a legal candidate existed, but the row did not complete before deadline; positive slack at knowledge is retained and does not prove zero time",
            "information_present_but_execution_or_resource_not_successful": "legal candidate existed but no post-event alternate service was recorded, without a separate deadline failure reason",
            "not_an_interrupted_recovery_event": "episode completed or failed without the reference execution being interrupted; excluded from event-recovery denominators",
        },
        "full_pool": {
            "rows": len(full),
            "four_way_episode_counts": dict(
                sorted(Counter(row["category"] for row in full).items())
            ),
            "interrupted_event_metrics": metrics(interrupted),
            "scope_classes": counters(interrupted),
            "by_split": by_split,
        },
        "boundary_scan": {
            "rows": len(scan),
            "interrupted_event_metrics": metrics(scan_interrupted),
            "scope_classes": counters(scan_interrupted),
            "by_factor": factors,
        },
        "learning_decision": {
            "candidate_visible_old_policy_failures": 0,
            "training_recovery_samples": "unknown_training_rollout_not_exported",
            "A_only": "stopped_start_condition_unmet",
            "B_C": "not_started",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    result = summarize(audit)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "format": result["format"]}))


if __name__ == "__main__":
    main()
