from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_model_rule_pair_v1 as RUNNER
import tools.model_rule_branch_runtime_v1 as RUNTIME


UAV_FIELDS = ("x", "y", "energy", "alive", "connected", "idle")
TASK_FIELDS = ("x", "y", "deadline", "remaining_service", "priority", "pending", "region_id", "target_id")


def _channel(value: float, *, known: bool = True, valid: bool = True, age: float = 0.0) -> list[float]:
    return [value, int(known), int(valid), age]


def _row(fields, values):
    return [item for field in fields for item in _channel(float(values[field]))]


def _observation(step: int = 0) -> dict:
    uavs = []
    for index in range(4):
        uavs.append(_row(UAV_FIELDS, {"x": float(index), "y": 0.0, "energy": 10.0, "alive": 1.0, "connected": 1.0, "idle": 1.0}))
    tasks = []
    for index in range(6):
        tasks.append(_row(TASK_FIELDS, {"x": float(index), "y": 0.0, "deadline": 30.0, "remaining_service": 1.0, "priority": 1.0, "pending": 1.0, "region_id": 0.0, "target_id": 0.0}))
    return {
        "flat": [float(step)],
        "time": float(step),
        "version": step,
        "uavs": uavs,
        "tasks": tasks,
        "mask": [True] * 25,
        "public_entity_ids": {"uavs": [f"uav-{i}" for i in range(4)], "tasks": [f"task-{i}" for i in range(6)]},
        "continuation_actions": [],
        "trigger_flags": {},
    }


def test_static_package_is_zero_step_and_manifest_is_canonical():
    result = RUNNER.package_validation()
    assert result["ok"] is True
    assert result["hard_counts"]["env_step"] == 0
    assert result["hard_counts"]["model_forward"] == 0
    rows = RUNNER._manifest_rows(RUNNER.MANIFEST)
    assert len(rows) == 48
    assert [int(row["order"]) for row in rows] == list(range(1, 49))
    for index, row in enumerate(rows):
        expected = RUNNER.expected_arm(row["parent_id"], int(row["repeat"]))[index % 2]
        assert row["arm"] == expected


def test_public_rule_probe_uses_actual_rule_and_identity_hidden():
    item = {"branch_id": "b", "pair_id": "p", "exogenous_key": "test-exogenous"}
    recorder = RUNNER._ProbeRecorder(item, "public_rule", RUNNER.ModelCallCounters(), Path("selector-ledger.jsonl"))
    hidden_policy = {"policy": [1.0]}
    hidden_world = {"world": [2.0]}
    rule_module = RUNNER._load_module(RUNNER.RULE, "gppo_world.public_dispatch_rule_test_v1")
    result = recorder.rule_probe(rule_module.select_public_dispatch_action, _observation(), hidden_policy, hidden_world)
    assert result["model_calls"] == {name: 0 for name in RUNNER.ModelCallCounters.NAMES}
    assert result["next_policy_hidden"] == hidden_policy
    assert result["by_action"][result["original_action"]]["hidden"] == hidden_world
    assert recorder.rule_calls == 1
    assert recorder.records[0]["input_kind"] == "public_observation_only"
    assert recorder.records[0]["model_call_status"]["actor_readout"] == {"attempted": 0, "completed": 0}


def test_runtime_adapter_declared_calls_control_actual_counters():
    counters = RUNTIME.Counters()
    adapter = RUNTIME.RuntimeAdapters(
        load_snapshots=lambda ids: {}, load_runtime=lambda snapshots: {}, probe=lambda *args: {},
        reward=lambda *args: {}, runtime_digest=lambda runtime: "stub",
        model_calls_per_probe={"policy_encode": 0, "world_candidate_batch": 0, "actor_readout": 0},
    )
    assert adapter.declared_model_calls() == {"policy_encode": 0, "world_candidate_batch": 0, "actor_readout": 0}
    assert counters.as_dict()["actor_forward"] == 0


@dataclass
class _Resource:
    energy: float = 36.0


class _Env:
    def __init__(self, step: int = 0):
        self.config = SimpleNamespace(task_capacity=6, uav_count=4, initial_energy=9.0, action_count=25)
        self.clock = SimpleNamespace(resources={"uav-0": _Resource()})
        self.step_index = step
        self._last_completed = 0
        self._last_expired = 0

    def step(self, action: int, *, submit_command: bool = True):
        self.step_index += 1
        return _observation(self.step_index), 0.0, True, {
            "terminated": True,
            "truncated": False,
            "episode_end_reason": "stub_terminal",
            "feedback": "accepted",
            "action": int(action),
            "submit_command": bool(submit_command),
            "counts": {"completed": 0, "expired": 0},
            "energy": {"uav-0": 36.0},
        }


class _Budget:
    def __init__(self):
        self.reserved = 0
        self.verified = 0
        self.path = None

    def reserve(self, stage, amount):
        assert stage == "environment_steps" and amount == 1
        self.reserved += 1
        return {"reservation_id": f"r-{self.reserved}"}

    def complete(self, token):
        self.verified += 1

    def unknown(self, token, reason):
        raise AssertionError("stub does not expect unknown finalization")

    def reservation_status(self, token):
        return "verified"

    def run_totals(self, run_id):
        return {"environment_steps": {"reserved": self.reserved, "verified": self.verified, "unknown": 0, "pending": 0}}

    def export_snapshot(self, path):
        Path(path).write_text("{}", encoding="utf-8")


def _stub_gate(tmp_path: Path):
    rows = RUNNER._manifest_rows(RUNNER.MANIFEST)
    auth = {"budget": {"path": "stub", "global_limit": 1625, "run_limit": 768}, "run": {"run_id": "stub-run", "attempt_id": "stub-attempt"}}
    return {"authorization": auth, "manifest": rows, "budget": {}}


def _stub_adapters():
    def load_snapshots(prefix_ids):
        return {prefix: {"env": _Env(), "obs": _observation(), "policy_hidden": {"p": [1.0]}, "world_hidden": {"w": [2.0]}, "probe": {"prefix": prefix}} for prefix in prefix_ids}

    def load_runtime(snapshots):
        return {"loaded": True}

    def probe(runtime, obs, policy_hidden, world_hidden):
        return {"probabilities": [0.0] * 24 + [1.0], "original_action": 24, "by_action": {24: {"hidden": {"w": [3.0]}}}, "next_policy_hidden": {"p": [4.0]}}

    def reward(info, before, after, env):
        return {"vector": [0.0, 0.0], "recomputed_vector": [0.0, 0.0], "consequence": [0.0, 0.0], "next_counts": {"completed": 0, "expired": 0}, "next_energy": 36.0}

    return SimpleNamespace(
        load_snapshots=load_snapshots, load_runtime=load_runtime, probe=probe, reward=reward,
        runtime_digest=lambda runtime: "stub-runtime", task_capacity=lambda env: 6,
        close=lambda: None,
    )


def test_execute_entry_uses_actual_branch_runtime_and_stops_at_first_failed_reservation(monkeypatch, tmp_path):
    gate = _stub_gate(tmp_path)
    monkeypatch.setattr(RUNNER, "validate_authorization", lambda *args, **kwargs: copy.deepcopy(gate))
    budget = _Budget()
    monkeypatch.setattr(RUNNER, "_load_module", lambda path, name: RUNTIME if Path(path) == RUNNER.RUNTIME else __import__("gppo_world.public_dispatch_rule_v1", fromlist=["*" ]))
    output = tmp_path / "run"
    status = RUNNER.execute_authorized(
        tmp_path / "auth.json", manifest_path=RUNNER.MANIFEST, output_dir=output,
        adapters=_stub_adapters(), budget_factory=lambda runtime, auth: budget,
    )
    assert status["status"] == "completed"
    assert status["hard_counts"]["branches_completed"] == 48
    assert status["hard_counts_per_arm"]["public_rule"]["actor_forward"] == 0
    assert status["hard_counts_per_arm"]["frozen_model"]["actor_forward"] == 24
    assert status["model_call_counts"]["attempted"] == {name: 24 for name in RUNNER.ModelCallCounters.NAMES}
    selectors = RUNNER.read_jsonl(output / "selector-ledger.jsonl")
    assert len(selectors) == 48
    assert all(row["model_call_status"]["actor_readout"]["attempted"] == (1 if row["arm"] == "frozen_model" else 0) for row in selectors)
    assert json.loads((output / "first-pair-gate.json").read_text(encoding="utf-8"))["ok"] is True


def test_failed_reservation_is_retained_and_matrix_does_not_continue(monkeypatch, tmp_path):
    gate = _stub_gate(tmp_path)
    monkeypatch.setattr(RUNNER, "validate_authorization", lambda *args, **kwargs: copy.deepcopy(gate))

    class FailingBudget(_Budget):
        def reserve(self, stage, amount):
            raise RuntimeError("synthetic reservation refusal")

    monkeypatch.setattr(RUNNER, "_load_module", lambda path, name: RUNTIME if Path(path) == RUNNER.RUNTIME else __import__("gppo_world.public_dispatch_rule_v1", fromlist=["*" ]))
    output = tmp_path / "failed-run"
    with pytest.raises(RUNTIME.RuntimeTechnicalStop):
        RUNNER.execute_authorized(
            tmp_path / "auth.json", manifest_path=RUNNER.MANIFEST, output_dir=output,
            adapters=_stub_adapters(), budget_factory=lambda runtime, auth: FailingBudget(),
        )
    assert not (output / "branch-results.jsonl").exists()
    failures = RUNNER.read_jsonl(output / "failure-ledger.jsonl")
    assert failures
    assert all(row.get("branch_id") == RUNNER._manifest_rows(RUNNER.MANIFEST)[0]["branch_id"] for row in failures if row.get("branch_id"))
    assert not (output / "progress.jsonl").exists()


def test_validate_authorization_rejects_manifest_or_budget_identity(monkeypatch, tmp_path):
    check = RUNNER.package_validation()
    auth = RUNNER.authorization_template(check)
    auth["status"] = "authorized"
    auth["review"] = {"reviewed_by": "test", "reviewed_at": "2026-09-23", "accepted_interpretation": True}
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth), encoding="utf-8")
    with pytest.raises(RUNNER.AuthorizationError):
        RUNNER.validate_authorization(path, manifest_path=RUNNER.MANIFEST, output_dir=tmp_path / "fresh")
    auth["budget"]["sha256"] = "0" * 64
    path.write_text(json.dumps(auth), encoding="utf-8")
    with pytest.raises(RUNNER.AuthorizationError):
        RUNNER.validate_authorization(path, manifest_path=RUNNER.MANIFEST, output_dir=tmp_path / "fresh2")
