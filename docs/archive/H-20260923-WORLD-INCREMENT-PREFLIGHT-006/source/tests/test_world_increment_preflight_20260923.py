from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "analyze_world_increment_preflight_20260923.py"
spec = importlib.util.spec_from_file_location("world_increment_preflight", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_candidate_categories_preserve_noop_and_empty_contract():
    category, candidates, non_noop, noop = module.candidate_class({"candidate_actions": [0, 7, 24]})
    assert category == "two_or_more_non_noop"
    assert candidates == [0, 7, 24]
    assert non_noop == [0, 7]
    assert noop is True
    assert module.candidate_class({"candidate_actions": [24]})[0] == "noop_only"
    assert module.candidate_class({"candidate_actions": []})[0] == "empty_anomaly"


def test_identity_and_step_projection_are_read_only_metadata():
    row = {
        "branch_id": "parent-00|W1|seed-1101|prefix-0|repeat-0|guard-on",
        "step": 1,
        "candidate_actions": [0, 6, 24],
        "legal_mask": [int(i in (0, 1, 6, 24)) for i in range(25)],
        "excluded_actions": [1],
        "original_action": 1,
        "final_action": 0,
        "triggered": True,
        "original_probability": 0.5,
        "final_probability": 0.4,
    }
    projected = module.decision_projection(row)
    assert projected["parent_id"] == "parent-00"
    assert projected["repeat"] == 0
    assert projected["original_vs_final_changed"] is True
    assert projected["noop_retained"] is True
    assert projected["evidence_limits"]["candidate_features_saved"] is False


def test_protocol_is_event_sensitivity_not_total_world_ablation():
    plan = module.protocol()
    assert plan["decision"].startswith("A_scoped_event_feature_intervention")
    assert plan["required_new_work"]["branches"] == 48
    assert plan["required_new_work"]["new_environment_step_worst_case"] == 768
    assert "not a no-world-model" in plan["interpretation_limit"]


def saved_fixture():
    row = json.loads(module.LEDGER.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads((module.H005 / "branch-manifest.json").read_text(encoding="utf-8"))["rows"]
    return row, [m for m in manifest if m["branch_id"] == row["branch_id"]]


def test_saved_public_contract_and_readonly_validation():
    row, manifest = saved_fixture()
    original = copy.deepcopy(row)
    assert module.validate_rows([row], manifest)["contract_mismatches"] == 0
    assert row == original


@pytest.mark.parametrize("fault", ["duplicate", "illegal", "noop", "identity", "selection", "missing"])
def test_invalid_evidence_fails_closed(fault):
    row, manifest = saved_fixture()
    rows = [row]
    if fault == "duplicate":
        rows.append(copy.deepcopy(row))
    elif fault == "illegal":
        row["candidate_actions"].append(2)
    elif fault == "noop":
        row["candidate_actions"].remove(24)
    elif fault == "identity":
        row["action_task_ids"][0][1] = "wrong-task"
    elif fault == "selection":
        row["final_action"] = 24
    else:
        del row["candidate_actions"]
    with pytest.raises((ValueError, KeyError)):
        module.validate_rows(rows, manifest)
