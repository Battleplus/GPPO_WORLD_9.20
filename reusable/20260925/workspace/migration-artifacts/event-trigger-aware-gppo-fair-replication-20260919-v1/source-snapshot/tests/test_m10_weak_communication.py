from __future__ import annotations

from gppo_world.m10_communication import CommunicationProfile
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec, scenario_from_dict, scenario_to_dict


def one_task(profile: CommunicationProfile) -> M10Scenario:
    return M10Scenario(
        "weak-link",
        (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 6.0, 1.0, 1.0),),
        (), 17, "test", "weak-link-test", profile,
    )


def test_tape_roundtrip_preserves_frozen_communication_protocol():
    scenario = one_task(CommunicationProfile(name="composite", telemetry_loss_probability=0.2,
                                              telemetry_outage_intervals=((2.0, 4.0),)))
    restored = scenario_from_dict(scenario_to_dict(scenario))
    assert restored == scenario


def test_telemetry_is_not_visible_before_delivery_and_audit_distinguishes_links():
    scenario = one_task(CommunicationProfile(name="delay", telemetry_extra_delay=1.0))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1), scenario)
    assert not bool(env.action_mask()[0])
    env.step(1)
    assert bool(env.action_mask()[0])
    statuses = {item["status"] for item in env._communication_log}  # audit-only inspection
    assert "sent" in statuses and "received" in statuses
    assert {item["link"] for item in env._communication_log} == {"telemetry", "command"}


def test_overdue_telemetry_is_recorded_as_expired_and_not_exposed():
    scenario = one_task(CommunicationProfile(name="expiry", telemetry_extra_delay=1.0))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, telemetry_max_age=0.1), scenario)
    env.step(1)
    assert any(item["link"] == "telemetry" and item["status"] == "expired"
               for item in env._communication_log)
    assert not bool(env.action_mask()[0])


def test_out_of_order_old_telemetry_cannot_overwrite_newer_sequence():
    scenario = one_task(CommunicationProfile(name="reorder-test", telemetry_reorder_window=2.0))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1), scenario)
    env._send("uav", "uav-0", "energy", 8.0)
    env._send("uav", "uav-0", "energy", 7.0)
    env._send("uav", "uav-0", "energy", 6.0)
    env.step(1)
    env.step(1)
    stale = [item for item in env._communication_log if item["status"] == "stale_or_duplicate"]
    assert stale
    latest = env.view._uavs._latest[("uav-0", "energy")]  # audit-only inspection
    assert latest.sequence >= 4


def test_command_loss_never_reaches_bridge_or_service_clock():
    scenario = one_task(CommunicationProfile(name="command-loss", command_loss_probability=1.0))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1), scenario)
    _, _, _, info = env.step(0)
    assert info["feedback"] == "command_lost"
    assert info["task_service"]["task-0"] == 0.0
    assert any(item["link"] == "command" and item["status"] == "dropped" for item in info["communication_delta"])


def test_lost_ack_can_execute_once_but_duplicate_submission_is_execution_rejected():
    scenario = one_task(CommunicationProfile(name="ack-loss", ack_loss_probability=1.0))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=8.0), scenario)
    _, _, _, first = env.step(0)
    first_service = first["task_service"]["task-0"]
    assert first["feedback"] == "accepted"
    assert any(item["link"] == "ack" and item["status"] == "dropped" for item in first["communication_delta"])
    _, _, _, second = env.step(0)
    assert second["feedback"] in {"masked", "resource_busy", "task_unavailable", "stale"}
    assert second["task_service"]["task-0"] >= first_service
    assert len([item for item in env.execution.log if item["result"] == "accepted"]) == 1


def test_telemetry_audit_has_stable_message_ids_and_duplicate_delivery_ordinals():
    scenario = one_task(CommunicationProfile(
        name="duplicate", telemetry_duplicate_probability=1.0,
        telemetry_extra_delay=0.25,
    ))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=3.0), scenario)
    env.step(1)
    telemetry = [item for item in env._communication_log if item["link"] == "telemetry"]
    assert telemetry
    assert all(item.get("message_id") for item in telemetry)
    sent_ids = {item["message_id"] for item in telemetry if item["status"] == "sent"}
    delivery_ids = {item["message_id"] for item in telemetry
                    if item["status"] in {"received", "stale_or_duplicate", "expired"}}
    assert delivery_ids <= sent_ids
    assert any(item.get("delivery_ordinal") == 1 and item["status"] == "stale_or_duplicate"
               for item in telemetry)
