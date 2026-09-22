from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile

import pytest

import tools.run_ack_known_task_guard_corrected_20260922 as RUNNER
from tools.run_ack_known_task_guard_corrected_20260922 import (
    AuthorizationError,
    ProtocolError,
    build_branch_manifest,
    historical_compatibility_with_pinned_bridge,
)


def _bridge_inputs() -> dict:
    return copy.deepcopy(RUNNER.r_control_bridge_identities())


@pytest.fixture
def test_workspace():
    path = Path(tempfile.mkdtemp(prefix="ackguard-bridge-test-", dir=Path.cwd()))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_current_saved_inputs_recompute_bridge_and_qualify_all_controls():
    """The authorization bridge is recomputed from current saved evidence."""
    result = historical_compatibility_with_pinned_bridge(
        build_branch_manifest(),
        _bridge_inputs(),
    )

    assert result["recomputed_from_current_inputs"] is True
    assert result["saved_qualified_boolean_trusted"] is False
    assert result["strict"]["summary"]["reusable_R_control_pairs"] == 23
    assert result["qualified"]["summary"]["reusable_R_control_pairs"] == 24
    assert result["status"] == "reusable_as_R_control"
    assert len(result["bridge_evaluations"]) == 1


def test_bridge_hash_mismatch_rejects_before_saved_evidence_load(monkeypatch):
    supplied = _bridge_inputs()
    supplied["bridge_evidence"]["sha256"] = "0" * 64
    called = []

    def forbidden(*args, **kwargs):
        called.append("load")
        raise AssertionError("bridge evidence must not load after a pin mismatch")

    monkeypatch.setattr(RUNNER, "_load_current_bridge_rows", forbidden)
    with pytest.raises(ProtocolError, match="input gate failed"):
        historical_compatibility_with_pinned_bridge(build_branch_manifest(), supplied)
    assert called == []


def test_missing_bridge_provenance_rejects_before_snapshot_restore(monkeypatch):
    monkeypatch.setattr(RUNNER, "_bridge_provenance_rows", lambda payload: {})
    with pytest.raises(ProtocolError, match="contains no branch-bound provenance row"):
        historical_compatibility_with_pinned_bridge(build_branch_manifest(), _bridge_inputs())


def test_stale_label_digest_provenance_rejects_without_execution():
    manifest = build_branch_manifest()
    evidence = RUNNER.read_json(RUNNER.R_CONTROL_BRIDGE_EVIDENCE)
    provenance = RUNNER._bridge_provenance_rows(evidence)
    key = next(iter(provenance))
    stale = copy.deepcopy(provenance)
    stale[key]["label_digest"] = "0" * 64

    with pytest.raises(ProtocolError, match="label_digest"):
        RUNNER._load_current_bridge_rows(manifest, stale)


def test_validate_authorization_calls_bridge_before_budget_read(test_workspace: Path, monkeypatch):
    """The bridge gate is reached before the budget identity is inspected."""
    from tests.test_ackguard_corrected_protocol_20260922 import _authorized_gate_fixture

    auth_path, manifest_path, _, _ = _authorized_gate_fixture(test_workspace, monkeypatch)
    events = []
    real_budget_read = RUNNER.read_budget_identity

    def bridge_spy(manifest, supplied):
        events.append("bridge")
        # This test checks ordering. The direct test above exercises the real
        # bridge reconstruction against the pinned snapshot and labels.
        return {
            "status": "reusable_as_R_control",
            "qualified": {"summary": {"reusable_R_control_pairs": 24}},
        }

    def budget_spy(path, run_id=RUNNER.RUN_ID):
        events.append("budget")
        return real_budget_read(path, run_id=run_id)

    monkeypatch.setattr(RUNNER, "historical_compatibility_with_pinned_bridge", bridge_spy)
    monkeypatch.setattr(RUNNER, "read_budget_identity", budget_spy)
    RUNNER.validate_authorization(
        auth_path,
        manifest_path=manifest_path,
        out_dir=test_workspace / "fresh-output",
    )

    assert events[:2] == ["bridge", "budget"]


def test_missing_bridge_source_is_rejected_before_budget_factory(test_workspace: Path, monkeypatch):
    from tests.test_ackguard_corrected_protocol_20260922 import _authorized_gate_fixture

    auth_path, manifest_path, auth, _ = _authorized_gate_fixture(test_workspace, monkeypatch)
    auth["source"].pop("r_control_bridge")
    auth_path.write_text(json.dumps(auth, sort_keys=True), encoding="utf-8")
    called = []

    def forbidden_budget(*args, **kwargs):
        called.append("budget")
        raise AssertionError("budget must not be read after a missing bridge source")

    monkeypatch.setattr(RUNNER, "historical_compatibility_with_pinned_bridge", forbidden_budget)
    with pytest.raises(AuthorizationError, match="pre-execution authorization gate failed"):
        RUNNER.run_matrix(
            auth_path,
            manifest_path=manifest_path,
            out_dir=test_workspace / "fresh-output",
            budget_factory=forbidden_budget,
        )
    assert called == []
