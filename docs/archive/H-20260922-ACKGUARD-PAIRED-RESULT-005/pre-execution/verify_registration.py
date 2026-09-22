"""Pure registration checks for H-005; never imports models or environments."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


PACKAGE = Path(__file__).resolve().parent
WORKTREE = PACKAGE.parents[2]
RUNNER = WORKTREE / "tools" / "run_ack_known_task_guard_corrected_20260922.py"
HISTORICAL_ENV = Path(
    r"E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot\gppo_world\m10_environment.py"
)
HISTORICAL_BUDGET = WORKTREE / "runs" / "finite-communication-ack-lease-fix-20260920" / "ack-known-task-guard-baseline-v1" / "budget.sqlite3"


def load(name: str) -> Any:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function_segment(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            end = getattr(node, "end_lineno", node.lineno)
            return "\n".join(lines[node.lineno - 1:end])
    raise AssertionError(f"missing function: {name}")


def check() -> dict[str, Any]:
    manifest = load("branch-manifest.json")
    protocol = load("protocol.json")
    auth = load("authorization-template.json")
    status = load("run-status.json")
    gate = load("first-branch-gate.json")
    rules = load("analysis-and-stop-rules.json")
    source = load("source-input-identities.json")
    budget_proposal = load("budget-proposal.json")
    hashes = load("hashes.json")

    rows = manifest["rows"]
    assert len(rows) == 24
    assert len({row["branch_id"] for row in rows}) == 24
    assert {row["condition"] for row in rows} == {"W1"}
    assert {int(row["model_seed"]) for row in rows} == {1101}
    assert {int(row["prefix_index"]) for row in rows} == {0}
    assert {int(row["repeat"]) for row in rows} == {0, 1, 2}
    assert all(row["outcome_blind_selection"] is True for row in rows)
    assert manifest["old_two_guard_branches_excluded"] is True

    assert protocol["dynamic_execution_called"] is False
    assert protocol["hard_counts_at_phase1"] == rules["hard_counts_required_at_registration"]
    assert status["dynamic_execution_called"] is False
    assert status["hard_counts"] == {
        "environment_steps": 0,
        "model_forward": 0,
        "new_formal_attempt": 0,
        "offline_updates": 0,
        "optimizer_updates": 0,
        "world_updates": 0,
    }
    assert auth["status"] == "pending"
    assert auth["template_only"] is True
    assert auth["cannot_authorize_here"] is True
    assert budget_proposal["status"] == "proposal_only_not_applied"
    assert budget_proposal["minimum_migration_design"]["apply_now"] is False
    bridge = source["r_control_bridge_inputs"]
    expected_bridge_names = {
        "schema_helper",
        "audit_loader",
        "bridge_evidence",
        "bridge_qualified_compatibility",
        "bridge_strict_compatibility",
        "remote_receipt",
    }
    assert set(bridge) == expected_bridge_names
    assert all(value["exists"] is True and len(value["sha256"]) == 64 for value in bridge.values())
    assert auth["source"]["r_control_bridge"] == bridge
    assert protocol["historical_R_compatibility"]["bridge_recomputed_at_authorization"] is True
    assert protocol["historical_R_compatibility"]["bridge_saved_24_of_24_boolean_trusted"] is False
    analysis = load("analysis-and-stop-rules.json")
    assert analysis["analysis_implementation"]["path"] == "analyze_results.py"
    assert analysis["analysis_implementation"]["execution_contract"].startswith("pure standard-library")
    expected_package_files = hashes["package_files"]
    actual_package_files = {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_file() and path.name not in {"hashes.json", "hashes.json.sha256"}
    }
    assert set(expected_package_files) == actual_package_files
    assert all(sha256(PACKAGE / name) == digest for name, digest in expected_package_files.items())

    strict = load("historical-R-compatibility.json")
    assert strict["summary"]["pairs"] == 24
    assert strict["summary"]["reusable_R_control_pairs"] == 23
    assert strict["summary"]["insufficient_pairs"] == 1
    assert gate["historical_R_evidence"]["bridge_qualified_pairs"] == 24
    assert gate["historical_R_evidence"]["strict_compatibility_pairs"] == 23
    assert gate["status"] == "pending_dynamic_gate"

    runner_text = RUNNER.read_text(encoding="utf-8")
    default_adapters = _function_segment(runner_text, "_default_adapters")
    execute_branch = _function_segment(runner_text, "_execute_branch_impl")
    assert "evaluate_bridge" not in default_adapters
    assert "env.step" in execute_branch
    assert "task_decision_bridge" in runner_text
    assert "historical_compatibility_with_pinned_bridge" in runner_text
    assert "_load_current_bridge_rows" in runner_text
    assert "saved_qualified_boolean_trusted" in runner_text
    historical_text = HISTORICAL_ENV.read_text(encoding="utf-8")
    assert "TaskDecisionBridge" in historical_text
    assert "self.bridge.submit" in historical_text
    assert source["dynamic_execution_called"] is False

    uri = str(HISTORICAL_BUDGET.resolve()).replace("\\", "/")
    con = sqlite3.connect(f"file:{uri}?mode=ro", uri=True)
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        stage = con.execute(
            "SELECT limit_amount,reserved,verified,unknown FROM stages WHERE stage='environment_steps'"
        ).fetchone()
        assert stage == (384, 20, 20, 0)
        assert con.execute("SELECT COUNT(*) FROM attempts WHERE run_id='ackguard-corrected-protocol-20260922'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM attempts WHERE attempt_id='ackguard-corrected-protocol-20260922-attempt-0001'").fetchone()[0] == 0
    finally:
        con.close()

    return {
        "schema": "ackguard-matrix-registration-verification/1.0.0",
        "handoff_id": "H-20260922-ACKGUARD-PAIRED-RESULT-005",
        "status": "passed",
        "dynamic_execution_called": False,
        "environment_steps": 0,
        "model_forward": 0,
        "optimizer_updates": 0,
        "world_updates": 0,
        "offline_updates": 0,
        "new_formal_attempt": 0,
        "manifest_rows": len(rows),
        "strict_R_reusable_pairs": strict["summary"]["reusable_R_control_pairs"],
        "bridge_qualified_R_pairs": gate["historical_R_evidence"]["bridge_qualified_pairs"],
        "budget_integrity": "ok",
        "budget_identity_sha256": sha256(HISTORICAL_BUDGET),
        "runner_sha256": sha256(RUNNER),
        "bridge_statement": "native task bridge is inside historical env.step; audit evaluate_bridge is not a dynamic runner entry",
    }


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True, indent=2))
