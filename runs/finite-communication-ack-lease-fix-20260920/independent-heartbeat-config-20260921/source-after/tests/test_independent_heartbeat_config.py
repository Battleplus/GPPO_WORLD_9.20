from __future__ import annotations

from dataclasses import replace

import pytest

from gppo_world.m10_contract_v1 import ContractV1Config
from gppo_world.m10_environment import (
    M10Config,
    M10Environment,
    M10Scenario,
    M10TaskSpec,
    m10_config_from_dict,
    m10_config_to_dict,
)


@pytest.fixture(autouse=True)
def forbid_environment_steps(monkeypatch):
    def forbidden_step(*args, **kwargs):
        raise AssertionError("M10Environment.step must not be called")

    monkeypatch.setattr(M10Environment, "step", forbidden_step)


def _scenario() -> M10Scenario:
    return M10Scenario(
        "heartbeat-fixture",
        (M10TaskSpec("task-0", 0.0, 1.0, 1.0, 10.0, 1.0, 1.0),),
    )


def _stable_observation_values(env: M10Environment) -> None:
    """Suppress value-change reports so only heartbeat triggers are observed."""
    values = {}
    for uid, resource in env.clock.resources.items():
        occupied = any(task.assigned_uav == uid for task in env.clock.tasks.values())
        for field, value in {
            "x": resource.position[0], "y": resource.position[1],
            "energy": resource.energy, "alive": float(resource.alive),
            "connected": float(resource.connected), "idle": float(not occupied),
        }.items():
            values[("uav", uid, field)] = (float(value),)
    for task_id, task in env.clock.tasks.items():
        spec = env._task_by_id[task_id]
        for field, value in {
            "x": spec.x, "y": spec.y, "deadline": task.deadline,
            "remaining_service": max(0.0, task.required_service - task.service),
            "priority": task.priority, "pending": float(task.state.value == "pending"),
            "region_id": float(spec.region_id), "target_id": float(spec.target_id),
        }.items():
            values[("task", task_id, field)] = (float(value),)
    env._contract_last_reported = values


def _heartbeat_events(env: M10Environment, times: list[float]) -> list[tuple[float, str, str]]:
    events: list[tuple[float, str, str]] = []

    def capture(kind, entity, field, value, *, message_kind="telemetry", completion_task_id=None):
        events.append((float(env.clock.time), kind, message_kind))
        return f"{kind}|{entity}|{field}|{env.clock.time:.9f}"

    env._send = capture
    env._contract_pending_sends.clear()
    env._contract_budget.queue_depth["uplink"] = 0
    env._contract_last_uav_heartbeat = float("-inf")
    env._contract_last_task_telemetry_heartbeat = float("-inf")
    _stable_observation_values(env)
    for now in times:
        env.clock.time = now
        env._deliver_observations()
    return events


def test_legacy_config_inherits_both_stream_intervals_and_reads_old_payload():
    old_payload = {
        "version": "finite-communication-control-reserve-v1",
        "reporting_mode": "event",
        "heartbeat_interval": 3.5,
    }
    contract = ContractV1Config.from_dict(old_payload)
    assert contract is not None
    assert contract.uav_heartbeat_interval is None
    assert contract.task_telemetry_heartbeat_interval is None
    assert contract.effective_uav_heartbeat_interval == 3.5
    assert contract.effective_task_telemetry_heartbeat_interval == 3.5
    assert contract.to_dict()["heartbeat_interval_sources"] == {
        "uav": "heartbeat_interval", "task_telemetry": "heartbeat_interval",
    }


def test_explicit_stream_overrides_round_trip_independently():
    contract = ContractV1Config(
        reporting_mode="event", heartbeat_interval=2.0,
        uav_heartbeat_interval=4.0,
        task_telemetry_heartbeat_interval=2.0,
    )
    restored = ContractV1Config.from_dict(contract.to_dict())
    assert restored is not None
    assert restored.uav_heartbeat_interval == 4.0
    assert restored.task_telemetry_heartbeat_interval == 2.0
    assert restored.effective_uav_heartbeat_interval == 4.0
    assert restored.effective_task_telemetry_heartbeat_interval == 2.0
    assert restored.to_dict()["heartbeat_interval_sources"] == {
        "uav": "uav_heartbeat_interval",
        "task_telemetry": "task_telemetry_heartbeat_interval",
    }


def test_environment_config_round_trip_preserves_stream_overrides():
    config = M10Config(contract_v1=ContractV1Config(
        reporting_mode="event", heartbeat_interval=2.0,
        uav_heartbeat_interval=4.0,
        task_telemetry_heartbeat_interval=2.0,
    ))
    restored = m10_config_from_dict(m10_config_to_dict(config))
    assert restored.contract_v1 is not None
    assert restored.contract_v1.effective_uav_heartbeat_interval == 4.0
    assert restored.contract_v1.effective_task_telemetry_heartbeat_interval == 2.0
    assert restored.contract_v1.to_dict() == config.contract_v1.to_dict()


def test_default_schedule_matches_independent_legacy_shared_heartbeat_reference():
    contract = ContractV1Config(reporting_mode="event", heartbeat_interval=2.0)
    env = M10Environment(
        M10Config(uav_count=2, task_capacity=1, contract_v1=contract), _scenario()
    )
    actual = _heartbeat_events(env, [0.0, 1.0, 2.0, 3.0, 4.0])

    # Independent reference for the pre-split implementation: one shared
    # marker and one legacy interval drove both message streams.
    expected = []
    last = float("-inf")
    for now in [0.0, 1.0, 2.0, 3.0, 4.0]:
        due = now - last >= 2.0
        if due:
            expected.extend([(now, "uav", "event")] * (2 * 6))
            expected.extend([(now, "task", "telemetry")] * 8)
            last = now
    assert actual == expected


@pytest.mark.parametrize(
    ("uav_interval", "task_interval", "expected_counts"),
    [
        (4.0, 2.0, {0.0: {"uav": 12, "task": 8}, 2.0: {"uav": 0, "task": 8}, 4.0: {"uav": 12, "task": 8}}),
        (2.0, 4.0, {0.0: {"uav": 12, "task": 8}, 2.0: {"uav": 12, "task": 0}, 4.0: {"uav": 12, "task": 8}}),
    ],
)
def test_stream_heartbeat_triggers_are_independent(uav_interval, task_interval, expected_counts):
    contract = ContractV1Config(
        reporting_mode="event", heartbeat_interval=2.0,
        uav_heartbeat_interval=uav_interval,
        task_telemetry_heartbeat_interval=task_interval,
    )
    env = M10Environment(
        M10Config(uav_count=2, task_capacity=1, contract_v1=contract), _scenario()
    )
    events = _heartbeat_events(env, [0.0, 2.0, 4.0])
    for now, counts in expected_counts.items():
        observed = {"uav": 0, "task": 0}
        for event_time, kind, _ in events:
            if event_time == now:
                observed[kind] += 1
        assert observed == counts


def test_zero_environment_step_guard_is_active():
    contract = ContractV1Config(reporting_mode="event", uav_heartbeat_interval=4.0,
                                task_telemetry_heartbeat_interval=2.0)
    env = M10Environment(
        M10Config(uav_count=1, task_capacity=1, contract_v1=contract), _scenario()
    )
    _heartbeat_events(env, [0.0, 2.0, 4.0])


def test_nonheartbeat_value_change_keeps_event_trigger_semantics():
    contract = ContractV1Config(
        reporting_mode="event", heartbeat_interval=4.0,
        uav_heartbeat_interval=4.0,
        task_telemetry_heartbeat_interval=4.0,
    )
    env = M10Environment(
        M10Config(uav_count=1, task_capacity=1, contract_v1=contract), _scenario()
    )
    env._contract_pending_sends.clear()
    env._contract_budget.queue_depth["uplink"] = 0
    env._contract_last_uav_heartbeat = 0.0
    env._contract_last_task_telemetry_heartbeat = 0.0
    _stable_observation_values(env)
    captured: list[tuple[str, str]] = []

    def capture(kind, entity, field, value, *, message_kind="telemetry", completion_task_id=None):
        captured.append((kind, field))
        return "fixture"

    env._send = capture
    env.clock.resources["uav-0"].position = (1.0, 0.0)
    env.clock.time = 1.0
    env._deliver_observations()
    assert ("uav", "x") in captured
