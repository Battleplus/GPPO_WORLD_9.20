from gppo_world.m10_communication import CommunicationProfile
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.service_clock import ServiceEvent


def parallel_env(*, events=(), communication=CommunicationProfile()):
    config = M10Config(
        uav_count=2, task_capacity=2, horizon=8.0, lease_ttl=2.0,
        initial_energy=20.0,
    )
    scenario = M10Scenario(
        "parallel-lease",
        (
            M10TaskSpec("task-0", 0.0, 0.0, 0.0, 7.0, 4.0, 1.0),
            M10TaskSpec("task-1", 0.0, 0.0, 0.0, 7.0, 4.0, 1.0),
        ),
        tuple(events), 101, "test", "parallel-lease-101", communication,
    )
    env = M10Environment(config, scenario)
    env.reset()
    return env


def test_two_acknowledged_leases_survive_new_assignment_and_noop_steps():
    env = parallel_env()
    # action 0 = uav-0/task-0; action 3 = uav-1/task-1; action 4 = NOOP
    _, _, _, first = env.step(0)
    _, _, _, second = env.step(3)
    assert first["feedback"] == second["feedback"] == "accepted"
    assert len(second["active_continuations"]) == 2

    for _ in range(4):
        _, _, _, info = env.step(4, submit_command=True)

    assert info["tasks"] == {"task-0": "completed", "task-1": "completed"}
    assert info["task_service"] == {"task-0": 4.0, "task-1": 4.0}
    renewal_commands = {
        item["command_id"] for item in env.execution.log
        if item["result"] == "renewed"
    }
    assert len(renewal_commands) == 2
    assert all(item["kind"] == "lease_renewal" for item in env._communication_log
               if item.get("kind") == "lease_renewal")


def test_fault_only_interrupts_its_lease_and_other_task_completes():
    env = parallel_env(events=(ServiceEvent(2.5, "uav-0", "damage"),))
    env.step(0)
    env.step(3)
    _, _, _, info = env.step(4, submit_command=True)
    assert info["tasks"]["task-0"] == "pending"
    assert info["tasks"]["task-1"] == "serving"
    _, _, _, info = env.step(4, submit_command=True)
    _, _, _, info = env.step(4, submit_command=True)
    assert info["tasks"] == {"task-0": "pending", "task-1": "completed"}
    assert env.execution.status["m10-parallel-lease-0-cmd-00001"] == "revoked"
    assert env.execution.status["m10-parallel-lease-0-cmd-00002"] == "completed"


class DropFirstRenewalProfile(CommunicationProfile):
    def renewal(self, *, seed, identity):
        fate = super().renewal(seed=seed, identity=identity)
        if "cmd-00001|renew|" in identity:
            return {**fate, "dropped": True}
        return fate


class DropFirstRenewalAckProfile(CommunicationProfile):
    def ack_delivered(self, *, seed, identity):
        if "|renew|" in identity:
            return False
        return super().ack_delivered(seed=seed, identity=identity)


def test_expired_lease_does_not_clear_other_active_lease():
    env = parallel_env(communication=DropFirstRenewalProfile())
    env.step(0)
    env.step(3)
    _, _, _, info = env.step(4, submit_command=False)
    assert info["lease_renewals"]
    assert info["tasks"]["task-0"] == "pending"
    assert info["tasks"]["task-1"] == "serving"
    assert env.execution.status["m10-parallel-lease-0-cmd-00001"] == "lease_expired"
    assert env.execution.status["m10-parallel-lease-0-cmd-00002"] == "executing"
    assert any(item.get("kind") == "lease_renewal" and item["status"] == "dropped"
               for item in info["communication_log"])


def test_lost_renewal_ack_keeps_last_control_knowledge_without_hidden_sync():
    env = parallel_env(communication=DropFirstRenewalAckProfile())
    env.step(0)
    _, _, _, info = env.step(0, submit_command=False)
    command_id = "m10-parallel-lease-0-cmd-00001"
    assert info["lease_renewals"][command_id] == "ack_lost"
    assert info["active_continuations"][0]["command_id"] == command_id
    assert env.clock.tasks["task-0"].service == 2.0


def test_delayed_duplicate_renewal_is_idempotent_and_audited():
    profile = CommunicationProfile(
        renewal_extra_delay=0.25,
        renewal_duplicate_probability=1.0,
        renewal_reorder_window=0.5,
    )
    config = M10Config(uav_count=1, task_capacity=1, horizon=7.0, lease_ttl=2.0,
                       initial_energy=20.0)
    scenario = M10Scenario(
        "renewal-transport",
        (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 6.0, 4.0, 1.0),), (),
        303, "test", "renewal-transport-303", profile,
    )
    env = M10Environment(config, scenario)
    env.reset()
    env.step(0)
    env.step(0, submit_command=False)
    env.step(0, submit_command=False)
    _, _, _, info = env.step(0, submit_command=False)
    assert info["tasks"]["task-0"] == "completed"
    renewal_log = [item for item in env._communication_log
                   if item.get("kind") == "lease_renewal"]
    assert any(item["status"] == "duplicate_received" for item in renewal_log)
    assert env.clock.tasks["task-0"].service == 4.0
