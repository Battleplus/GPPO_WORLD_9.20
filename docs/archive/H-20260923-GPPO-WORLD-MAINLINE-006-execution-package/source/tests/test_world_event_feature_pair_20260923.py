from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_world_event_feature_pair_20260923.py"
spec = importlib.util.spec_from_file_location("world_event_feature_pair", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _feature(arm: str, *, pair: str = "p", hidden: str = "h", off: bool = False) -> dict:
    raw = [[float(action + column) for column in range(17)] for action in range(25)]
    actor = [row[:12] + ([0.0] * 5 if off else row[12:]) for row in raw]
    return {
        "schema": runner.FEATURE_SCHEMA,
        "branch_id": f"{pair}|{arm}",
        "pair_id": pair,
        "arm": arm,
        "exogenous_key": "exo-p",
        "step": 1,
        "public_observation_sha256": "obs",
        "policy_hidden_before_sha256": hidden,
        "world_hidden_before_sha256": "world",
        "candidate_features_raw_25x17": raw,
        "candidate_features_actor_25x17": actor,
        "event_features_before_25x5": [row[12:] for row in raw],
        "event_features_after_25x5": [row[12:] if not off else [0.0] * 5 for row in raw],
        "by_action_hidden_sha256": {str(action): f"by-{action}" for action in range(25)},
        "next_policy_hidden_sha256": "next",
        "base_logits": [0.0] * 25,
        "preference_logits": [0.0] * 25,
        "candidate_logits": [0.0] * 25,
        "logits": [0.0] * 25,
        "probabilities": [1.0 / 25.0] * 25,
        "timing": {},
        "model_calls": {"policy_encode": 1, "world_candidate_batch": 1, "actor_readout": 1},
        "model_call_status": {
            "policy_encode": {"attempted": 1, "completed": 1},
            "world_candidate_batch": {"attempted": 1, "completed": 1},
            "actor_readout": {"attempted": 1, "completed": 1},
        },
    }


def _template(tmp_path: Path) -> Path:
    payload = runner.authorization_template(runner.package_validation())
    payload["status"] = "authorized"
    payload["review"] = {"reviewed_by": "test", "reviewed_at": "2026-09-23T00:00:00Z", "accepted_interpretation": True}
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_static_package_has_fixed_pairs_and_native_source_pin():
    result = runner.package_validation()
    assert result["ok"] is True
    assert result["manifest"]["row_count"] == 48
    assert result["manifest"]["pair_count"] == 24
    assert result["native_source_manifest"]["ok"] is True
    assert result["native_source_manifest"]["file_count"] == 44
    assert result["budget_readonly"]["stages"]["environment_steps"]["limit"] == 404
    assert result["hard_counts"]["env_step"] == 0


def test_manifest_rejects_duplicate_or_wrong_order():
    rows = runner._manifest_rows(runner.MANIFEST)
    broken = [dict(row) for row in rows]
    broken[1]["arm"] = "normal"
    result = runner.validate_manifest(broken)
    assert result["ok"] is False
    assert any("one normal/one event_features_off" in reason for reason in result["failures"])

    broken = [dict(row) for row in rows]
    broken[0]["order"], broken[1]["order"] = broken[1]["order"], broken[0]["order"]
    result = runner.validate_manifest(broken)
    assert result["ok"] is False
    assert any("order" in reason for reason in result["failures"])


def test_first_pair_gate_requires_two_arms_and_preserves_non_event_columns():
    normal = _feature("normal")
    off = _feature("event_features_off", off=True)
    gate = runner.compare_first_pair([normal, off])
    assert gate["ok"] is True
    assert gate["checks"]["same_raw_candidate_features"] is True
    assert gate["checks"]["same_by_action_hidden"] is True
    assert gate["checks"]["same_candidate_columns_0_12"] is True
    assert runner.compare_first_pair([normal, _feature("normal", off=True)])["ok"] is False

    broken = _feature("event_features_off", off=True, hidden="different")
    assert runner.compare_first_pair([normal, broken])["ok"] is False


def test_feature_schema_detects_missing_required_logs():
    row = _feature("normal")
    assert runner.required_feature_fields(row) == []
    del row["candidate_logits"]
    assert runner.required_feature_fields(row) == ["candidate_logits"]


def test_authorization_missing_and_wrong_source_hash_fail_closed(tmp_path: Path):
    with pytest.raises(runner.AuthorizationError, match="explicit reviewed authorization"):
        runner.validate_authorization(tmp_path / "missing.json", output_dir=tmp_path / "out")

    path = _template(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"]["runner_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.AuthorizationError, match="source hash mismatch"):
        runner.validate_authorization(path, output_dir=tmp_path / "out")


def test_authorization_wrong_budget_identity_and_limit_fail_closed(tmp_path: Path):
    path = _template(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["budget"]["sha256"] = "1" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.AuthorizationError, match="budget SQLite hash mismatch"):
        runner.validate_authorization(path, output_dir=tmp_path / "out")


def test_model_call_sidecar_separates_attempts_from_completed_calls():
    counts = runner.ModelCallCounters()
    counts.start("world_candidate_batch")
    assert counts.as_dict()["attempted"]["world_candidate_batch"] == 1
    assert counts.as_dict()["completed"]["world_candidate_batch"] == 0
    counts.finish("world_candidate_batch")
    assert counts.as_dict()["completed"]["world_candidate_batch"] == 1
