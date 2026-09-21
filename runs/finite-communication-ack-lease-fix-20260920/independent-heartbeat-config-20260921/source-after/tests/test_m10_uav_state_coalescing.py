from __future__ import annotations

from gppo_world.m10_contract_v1 import ContractBudgetConfig, ContractV1Config
from gppo_world.m10_communication import CommunicationProfile
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.telemetry import Telemetry


def _env(*, enabled: bool, queue_capacity: int = 4) -> M10Environment:
    contract = ContractV1Config(
        version="finite-communication-control-reserve-v1",
        reporting_mode="event",
        queue_admission_policy="priority_replace_merge",
        enable_uav_state_coalescing=enabled,
        budget=ContractBudgetConfig(
            uplink_rate=0.0,
            uplink_capacity=1.0,
            downlink_rate=0.0,
            downlink_capacity=1.0,
            uplink_queue_capacity=queue_capacity,
            downlink_queue_capacity=queue_capacity,
        ),
    )
    env = M10Environment(
        M10Config(uav_count=1, task_capacity=1, horizon=4.0, contract_v1=contract),
        M10Scenario(
            "coalescing-test",
            (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0),),
            communication=CommunicationProfile(name="ideal"),
        ),
    )
    env._contract_pending_sends.clear()
    env._contract_budget.queue_depth["uplink"] = 0
    env._contract_budget.tokens["uplink"] = 0.0
    env._contract_budget.last_time = env.clock.time
    env._communication_log.clear()
    return env


def test_enabled_coalesces_same_uav_state_and_preserves_age_depth_and_tokens():
    env = _env(enabled=True)
    env.clock.time = 1.0
    first_id = env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    old_depth = env._contract_budget.queue_depth["uplink"]
    old_tokens = env._contract_budget.tokens["uplink"]
    old_age = env._contract_pending_sends[0]["enqueued_at"]

    env.clock.time = 2.0
    second_id = env._send("uav", "uav-0", "x", 2.0, message_kind="event")

    item = env._contract_pending_sends[0]
    assert first_id != second_id
    assert item["message"].message_id == second_id
    assert item["message"].value == 2.0
    assert item["message"].measured_at == 2.0
    assert item["enqueued_at"] == old_age
    assert env._contract_budget.queue_depth["uplink"] == old_depth
    assert env._contract_budget.tokens["uplink"] == old_tokens
    merge = [x for x in env._communication_log if x["status"] == "coalesced_unsent"]
    assert merge and merge[-1]["original_enqueued_at"] == old_age
    assert merge[-1]["old_measured_at"] == 1.0
    assert merge[-1]["new_measured_at"] == 2.0


def test_scope_and_semantic_boundaries_do_not_mix():
    env = _env(enabled=True, queue_capacity=8)
    env.clock.time = 1.0
    env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    env._send("uav", "uav-1", "x", 2.0, message_kind="event")
    env._send("uav", "uav-0", "y", 3.0, message_kind="event")
    env._send("task", "task-0", "x", 4.0, message_kind="event")
    env._send("task", "task-0", "x", 5.0, message_kind="emergency_announcement")
    assert len(env._contract_pending_sends) == 5
    assert not any(x["status"] == "coalesced_unsent" for x in env._communication_log)


def test_task_telemetry_path_is_identical_with_coalescing_enabled_or_disabled():
    disabled = _env(enabled=False, queue_capacity=8)
    enabled = _env(enabled=True, queue_capacity=8)
    for env in (disabled, enabled):
        env.clock.time = 1.0
        env._send("task", "task-0", "x", 1.0, message_kind="telemetry")
        env.clock.time = 2.0
        env._send("task", "task-0", "x", 2.0, message_kind="telemetry")
        env.clock.time = 3.0
        env._send("task", "task-0", "y", 3.0, message_kind="telemetry")

    def pending_signature(env):
        return [
            (
                item["kind"], item["message_kind"], item["message"],
                item["enqueued_at"], item.get("direction"),
                item.get("service_sort_key"),
            )
            for item in env._contract_pending_sends
        ]

    assert pending_signature(enabled) == pending_signature(disabled)
    assert enabled._contract_budget.snapshot() == disabled._contract_budget.snapshot()
    assert enabled._communication_log == disabled._communication_log


def test_task_telemetry_cannot_replace_completion_or_be_replaced_by_completion():
    env = _env(enabled=True, queue_capacity=8)
    env.clock.time = 1.0
    completion_id = env._send(
        "task", "task-0", "pending", 0.0,
        message_kind="completion", completion_task_id="task-0",
    )
    telemetry_id = env._send("task", "task-0", "pending", 1.0, message_kind="telemetry")
    assert completion_id != telemetry_id
    assert [item["message_kind"] for item in env._contract_pending_sends] == [
        "completion", "telemetry",
    ]
    assert env._contract_pending_sends[0]["message"].message_id == completion_id

    reverse = _env(enabled=True, queue_capacity=8)
    reverse.clock.time = 1.0
    reverse_telemetry_id = reverse._send(
        "task", "task-0", "pending", 1.0, message_kind="telemetry",
    )
    reverse_completion_id = reverse._send(
        "task", "task-0", "pending", 0.0,
        message_kind="completion", completion_task_id="task-0",
    )
    assert reverse_telemetry_id != reverse_completion_id
    assert [item["message_kind"] for item in reverse._contract_pending_sends] == [
        "telemetry", "completion",
    ]
    assert reverse._contract_pending_sends[0]["message"].message_id == reverse_telemetry_id


def test_repeated_task_telemetry_updates_preserve_completion_slot():
    env = _env(enabled=True, queue_capacity=8)
    env.clock.time = 1.0
    completion_id = env._send(
        "task", "task-0", "pending", 0.0,
        message_kind="completion", completion_task_id="task-0",
    )
    env.clock.time = 2.0
    first_telemetry_id = env._send("task", "task-0", "pending", 1.0, message_kind="telemetry")
    env.clock.time = 3.0
    second_telemetry_id = env._send("task", "task-0", "pending", 2.0, message_kind="telemetry")

    assert [item["message_kind"] for item in env._contract_pending_sends] == [
        "completion", "telemetry",
    ]
    assert env._contract_pending_sends[0]["message"].message_id == completion_id
    assert env._contract_pending_sends[1]["message"].message_id == second_telemetry_id
    assert first_telemetry_id != second_telemetry_id
    assert env._contract_budget.queue_depth["uplink"] == 2


def test_older_or_duplicate_measurement_does_not_replace_newer_pending_state():
    env = _env(enabled=True)
    env.clock.time = 1.0
    env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    current = env._contract_pending_sends[0]["message"]
    stale = Telemetry("uav-0", "x", 0.5, 0.5, 0.5, current.sequence, "uav|uav-0|x|1|0.500000000")
    assert env._priority_admit_contract_pending(kind="uav", message=stale, message_kind="event", now=2.0)
    assert env._contract_pending_sends[0]["message"] == current
    assert any(x["status"] == "coalesced_stale_ignored" for x in env._communication_log)


def test_sent_state_is_not_replaceable_and_ttl_uses_new_measurement_time():
    env = _env(enabled=True)
    env._contract_budget.tokens["uplink"] = 1.0
    env.clock.time = 1.0
    sent_id = env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    assert not env._contract_pending_sends
    env.clock.time = 2.0
    queued_id = env._send("uav", "uav-0", "x", 2.0, message_kind="event")
    assert queued_id != sent_id
    assert env._contract_pending_sends[0]["message"].message_id == queued_id
    env.clock.time = 4.6
    env._expire_contract_pending_sends(env.clock.time)
    assert not env._contract_pending_sends
    expiry = [x for x in env._communication_log if x["status"] == "expired_in_queue"]
    assert expiry and expiry[-1]["measured_at"] == 2.0


def test_disabled_switch_preserves_original_event_behavior():
    env = _env(enabled=False)
    env.clock.time = 1.0
    env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    env.clock.time = 2.0
    env._send("uav", "uav-0", "x", 2.0, message_kind="event")
    assert len(env._contract_pending_sends) == 2
    assert not any(x["status"] == "coalesced_unsent" for x in env._communication_log)


def test_coalesced_state_reaches_original_send_and_receive_path():
    env = _env(enabled=True)
    env._contract_budget.tokens["uplink"] = 0.0
    env.clock.time = 1.0
    env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    env.clock.time = 2.0
    latest_id = env._send("uav", "uav-0", "x", 2.0, message_kind="event")
    env._contract_budget.tokens["uplink"] = 1.0
    env._flush_contract_pending_sends()
    assert any(x.get("status") == "sent" and x.get("identity") == latest_id
               for x in env._communication_log if x.get("link") == "budget")
    assert any(x.get("status") == "received" and x.get("message_id") == latest_id
               for x in env._communication_log)
