from dataclasses import replace

from gppo_world.m10_contract_v1 import (
    ContractBudgetConfig,
    ContractTimeLedger,
    ContractV1Config,
    SendBudgetExecutor,
)
from gppo_world.m10_communication import CommunicationProfile
from gppo_world.m10_environment import (
    M10Config, M10Environment, M10Scenario, M10TaskSpec,
    m10_config_from_dict, m10_config_to_dict,
)


def _contract(**kwargs):
    budget = kwargs.pop("budget", ContractBudgetConfig())
    return ContractV1Config(budget=budget, **kwargs)


def test_priority_admission_replaces_and_merges_only_unsent_telemetry():
    contract = _contract(
        queue_admission_policy="priority_replace_merge",
        budget=ContractBudgetConfig(
            uplink_rate=0.0, uplink_capacity=1.0, downlink_rate=0.0,
            downlink_capacity=1.0, uplink_queue_capacity=1,
        ),
    )
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=3.0,
                                    contract_v1=contract),
                         M10Scenario("admission", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0),)))
    env._contract_pending_sends.clear()
    env._contract_budget.queue_depth["uplink"] = 0
    env._send("uav", "uav-0", "x", 1.0, message_kind="telemetry")
    env._send("uav", "uav-0", "y", 2.0, message_kind="telemetry")
    env._send("task", "urgent-0", "x", 3.0, message_kind="emergency_announcement")
    assert any(item["message"].entity == "urgent-0" for item in env._contract_pending_sends)
    assert not any(item["message"].entity == "uav-0" and item["message"].field == "y"
                   for item in env._contract_pending_sends)
    assert any(item.get("status") == "replaced_in_queue" for item in env._communication_log)

    env2 = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=3.0,
                                     contract_v1=contract),
                          M10Scenario("merge", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0),)))
    env2._contract_pending_sends.clear()
    env2._contract_budget.queue_depth["uplink"] = 0
    env2._send("uav", "uav-0", "x", 1.0, message_kind="telemetry")
    old_depth = env2._contract_budget.queue_depth["uplink"]
    env2._send("uav", "uav-0", "x", 2.0, message_kind="telemetry")
    assert env2._contract_budget.queue_depth["uplink"] == old_depth
    assert env2._contract_pending_sends[0]["message"].value == 2.0
    assert any(item.get("status") == "coalesced_unsent" for item in env2._communication_log)


def test_emergency_entry_replaces_when_only_control_reserve_remains():
    """The real emergency announcement path must honor the control floor."""
    contract = _contract(
        version="finite-communication-control-reserve-v1",
        queue_admission_policy="priority_replace_merge",
        budget=ContractBudgetConfig(
            uplink_rate=0.0, uplink_capacity=8.0, downlink_rate=0.0,
            downlink_capacity=8.0, uplink_queue_capacity=1,
            control_reserve_uplink=2.0, control_reserve_downlink=1.0,
        ),
    )
    emergency = M10TaskSpec(
        "urgent-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0,
        emergency=True, source="external_task_publisher",
    )
    env = M10Environment(
        M10Config(uav_count=1, task_capacity=1, horizon=3.0, contract_v1=contract),
        M10Scenario("emergency-entry", (emergency,)),
    )
    env._contract_pending_sends.clear()
    env._contract_budget.queue_depth["uplink"] = 0
    env._contract_budget.tokens["uplink"] = 2.0
    env._contract_budget.last_time = 0.0
    env._send("uav", "uav-0", "x", 1.0, message_kind="event")
    # Keep the low-priority item in the real pending queue.  Suppress the
    # unrelated UAV event reports so the task publisher is the next sender.
    from gppo_world.task_lifecycle import TaskState
    env.clock.tasks["urgent-0"].state = TaskState.PENDING
    env.clock.time = 1.0
    env._contract_published_emergencies.clear()
    resource = env.clock.resources["uav-0"]
    for field, value in {
        "x": resource.position[0], "y": resource.position[1],
        "energy": resource.energy, "alive": float(resource.alive),
        "connected": float(resource.connected), "idle": 1.0,
    }.items():
        env._contract_last_reported[("uav", "uav-0", field)] = (float(value),)
    env._deliver_observations()
    assert any(item.get("status") == "replaced_in_queue"
               and item.get("message_kind") == "emergency_announcement"
               for item in env._communication_log)
    assert any(item["message_kind"] == "emergency_announcement"
               and item["message"].entity == "urgent-0"
               for item in env._contract_pending_sends)


def test_priority_overload_remains_budget_rejected_and_sent_messages_are_not_replaced():
    contract = _contract(
        queue_admission_policy="priority_replace_merge",
        budget=ContractBudgetConfig(
            uplink_rate=0.0, uplink_capacity=1.0, downlink_rate=0.0,
            downlink_capacity=1.0, uplink_queue_capacity=1,
        ),
    )
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=3.0,
                                    contract_v1=contract),
                         M10Scenario("overload", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0),)))
    env._contract_pending_sends.clear()
    env._contract_budget.queue_depth["uplink"] = 0
    env._send("task", "urgent-0", "x", 1.0, message_kind="emergency_announcement")
    env._send("task", "urgent-1", "x", 2.0, message_kind="emergency_announcement")
    assert any(item.get("status") == "budget_rejected" and item.get("reason") == "queue_full"
               for item in env._communication_log)


def test_budget_distinguishes_send_queue_and_rejection():
    budget = SendBudgetExecutor(ContractBudgetConfig(
        uplink_rate=0.0, uplink_capacity=1.0, downlink_rate=0.0,
        downlink_capacity=1.0, uplink_queue_capacity=1,
    ))
    assert budget.request(0.0, "uplink", "telemetry").status == "sent"
    assert budget.request(0.0, "uplink", "event").status == "queued"
    assert budget.request(0.0, "uplink", "telemetry").status == "budget_rejected"
    assert budget.attempts["uplink"] == 1
    assert budget.queue_depth["uplink"] == 1


def test_control_reserve_protects_ack_and_renewal_tokens_in_same_bucket():
    budget = SendBudgetExecutor(ContractBudgetConfig(
        uplink_capacity=2.0, downlink_capacity=1.0,
        control_reserve_uplink=2.0, control_reserve_downlink=1.0,
    ))
    telemetry = budget.reserve(0.0, "uplink", "telemetry")
    assert telemetry.status == "queued"
    ack = budget.reserve_control(0.0, "uplink", "ack")
    renewal_ack = budget.reserve_control(0.0, "uplink", "ack")
    renewal = budget.reserve_control(0.0, "downlink", "lease_renewal")
    assert ack.status == renewal_ack.status == renewal.status == "sent"
    assert budget.attempts == {"uplink": 2, "downlink": 1}
    assert budget.tokens == {"uplink": 0.0, "downlink": 0.0}


def test_zero_control_reserve_keeps_legacy_v1_bucket_behavior():
    budget = SendBudgetExecutor(ContractBudgetConfig(uplink_capacity=1.0))
    assert budget.reserve(0.0, "uplink", "telemetry").status == "sent"


def test_send_loss_consumes_token_and_is_not_refunded():
    profile = CommunicationProfile(name="all-loss", telemetry_loss_probability=1.0)
    scenario = M10Scenario(
        "loss", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0),),
        communication=profile,
    )
    contract = _contract(budget=ContractBudgetConfig(
        uplink_rate=0.0, uplink_capacity=100.0, downlink_rate=0.0, downlink_capacity=100.0,
    ))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=2.0,
                                   contract_v1=contract), scenario)
    assert env._contract_budget.attempts["uplink"] > 0
    assert any(item.get("status") == "dropped_after_send" for item in env._communication_log)
    assert env._contract_budget.tokens["uplink"] < 100.0


def test_emergency_task_is_hidden_before_arrival_and_visible_only_after_announcement():
    emergency = M10TaskSpec(
        "urgent-0", 1.0, 0.0, 0.0, 4.0, 1.0, 2.0,
        emergency=True, source="external_task_publisher",
    )
    scenario = M10Scenario("urgent", (emergency,))
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=3.0,
                                   contract_v1=_contract(reporting_mode="event")), scenario)
    before = env._observation()
    assert not any(row[1] > 0.5 for row in before["tasks"])
    env.step(env.config.action_count - 1)
    after = env._observation()
    assert any(row[1] > 0.5 for row in after["tasks"])
    assert any(item["event_type"] == "emergency_task_announcement"
               for item in env._contract_event_log)
    assert env._contract_ledger["urgent-0"].first_legal_awareness is not None


def test_contract_time_ledger_keeps_null_reason_and_separate_recovery():
    ledger = ContractTimeLedger("task-0")
    ledger.mark("actual_arrival", 2.0)
    ledger.mark("host_confirmation", None, reason="ack_lost")
    ledger.add_interruption("task-0|interruption|1", 4.0, "disconnect")
    ledger.add_recovery("task-0|interruption|1", 5.0, "uav-1")
    assert ledger.host_confirmation is None
    assert ledger.null_reasons["host_confirmation"] == "ack_lost"
    assert ledger.replacement_execution_start[0]["interruption_id"] == "task-0|interruption|1"


def test_legacy_default_has_no_contract_budget_side_effect():
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=2.0),
                         M10Scenario("legacy", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 5.0, 1.0, 1.0),)))
    assert env.config.contract_v1 is None
    assert env._contract_budget is None
    _, _, _, info = env.step(env.config.action_count - 1)
    assert info["contract_version"] == "legacy"
    assert info["contract_budget"] is None


def test_contract_config_roundtrip_keeps_version_and_budget():
    config = M10Config(contract_v1=_contract(reporting_mode="event"))
    restored = m10_config_from_dict(m10_config_to_dict(config))
    assert restored.contract_v1.to_dict() == config.contract_v1.to_dict()


def test_deadline_equality_and_damaged_uav_do_not_get_free_confirmation():
    profile = CommunicationProfile(name="ideal")
    task = M10TaskSpec("task-0", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    scenario = M10Scenario("boundary", (task,), communication=profile)
    env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=1.0,
                                   contract_v1=_contract(reporting_mode="event", budget=ContractBudgetConfig(
                                       uplink_rate=8.0, uplink_capacity=100.0,
                                       downlink_rate=8.0, downlink_capacity=100.0)),
                                   task_completion_mode="arrival_to_region",
                                   deadline_basis="physical_arrival"), scenario)
    env.step(0)
    ledger = env._contract_ledger["task-0"]
    assert ledger.physical_arrival is not None
    assert ledger.physical_arrival <= 1.0

    damaged = M10Scenario(
        "damaged", (M10TaskSpec("task-0", 0.0, 1.0, 0.0, 5.0, 1.0, 1.0),),
        events=(), communication=profile,
    )
    # The opt-in path never synthesizes a damage message from an already-dead
    # resource; this fixture checks the local-reporting guard directly.
    damaged_env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=2.0,
                                            contract_v1=_contract(reporting_mode="event", budget=ContractBudgetConfig(
                                                uplink_rate=8.0, uplink_capacity=100.0,
                                                downlink_rate=8.0, downlink_capacity=100.0))), damaged)
    damaged_env.clock.resources["uav-0"].alive = False
    before = len(damaged_env._communication_log)
    damaged_env._deliver_observations()
    after = damaged_env._communication_log[before:]
    assert not any(item.get("entity") == "uav-0" for item in after)
