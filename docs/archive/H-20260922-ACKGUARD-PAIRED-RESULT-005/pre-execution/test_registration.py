from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_registration.py")
SPEC = importlib.util.spec_from_file_location("ackguard_matrix_registration_verify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_registration_is_static_and_pending():
    result = MODULE.check()
    assert result["status"] == "passed"
    assert result["dynamic_execution_called"] is False
    assert result["environment_steps"] == 0
    assert result["model_forward"] == 0
    assert result["new_formal_attempt"] == 0


def test_fixed_matrix_and_control_counts():
    result = MODULE.check()
    assert result["manifest_rows"] == 24
    assert result["strict_R_reusable_pairs"] == 23
    assert result["bridge_qualified_R_pairs"] == 24


def test_budget_is_read_only_clean():
    result = MODULE.check()
    assert result["budget_integrity"] == "ok"
    assert len(result["budget_identity_sha256"]) == 64
