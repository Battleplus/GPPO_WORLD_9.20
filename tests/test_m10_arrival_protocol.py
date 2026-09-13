from gppo_world.m10_communication import CommunicationProfile
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.task_lifecycle import TaskState


def scenario(*, communication=None):
    return M10Scenario(
        name="arrival-functional",
        tasks=(M10TaskSpec("task-0", 0.0, 1.0, 0.0, 3.0, 2.0, 1.0),),
        seed=4101,
        tape_id="arrival-functional-4101",
        communication=communication or CommunicationProfile(),
    )


def config(**kwargs):
    return M10Config(
        uav_count=1, task_capacity=1, region_count=1, target_count=1,
        event_capacity=1, horizon=4.0, **kwargs,
    )


def test_arrival_completion_records_physical_send_and_host_times():
    env = M10Environment(config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival"), scenario())
    env.reset()
    env.step(env.config.action_count - 1)
    _, _, _, info = env.step(0)
    record = info["completion_records"]["task-0"]
    assert record["physical_arrival_time"] == 2.0
    assert record["completion_message_send_time"] == 2.0
    assert record["host_confirmation_time"] == 2.0
    assert info["counts"]["completed"] == 1
    assert env.clock.tasks["task-0"].state == TaskState.COMPLETED


def test_host_confirmation_deadline_is_separate_from_physical_arrival():
    env = M10Environment(
        config(task_completion_mode="arrival_to_region", deadline_basis="host_confirmation", telemetry_delay=1.0),
        scenario(),
    )
    env.reset()
    env.step(env.config.action_count - 1)
    _, _, _, info = env.step(0)
    record = info["completion_records"]["task-0"]
    assert record["physical_arrival_time"] == 2.0
    assert record["host_confirmation_time"] is None
    assert info["counts"]["completed"] == 0
    _, _, _, info = env.step(env.config.action_count - 1)
    record = info["completion_records"]["task-0"]
    assert record["host_confirmation_time"] == 3.0
    assert record["host_confirmation_before_deadline"] is True
    assert info["counts"]["completed"] == 1


def test_arrival_deadline_basis_can_mark_late_host_confirmation_expired():
    late = M10Scenario(
        name="arrival-late-confirmation",
        tasks=(M10TaskSpec("task-0", 0.0, 1.0, 0.0, 2.5, 2.0, 1.0),),
        seed=4102,
        tape_id="arrival-late-confirmation-4102",
        communication=CommunicationProfile(),
    )
    env = M10Environment(
        config(task_completion_mode="arrival_to_region", deadline_basis="host_confirmation", telemetry_delay=1.0),
        late,
    )
    env.reset()
    env.step(env.config.action_count - 1)
    env.step(0)
    _, _, _, info = env.step(env.config.action_count - 1)
    record = info["completion_records"]["task-0"]
    assert record["physical_arrival_before_deadline"] is True
    assert record["host_confirmation_before_deadline"] is False
    assert info["counts"]["completed"] == 0
    assert info["counts"]["expired"] == 1


def test_arrival_energy_failure_has_no_physical_completion_notice():
    env = M10Environment(
        config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival", initial_energy=0.2),
        scenario(),
    )
    env.reset()
    _, _, _, info = env.step(0)
    assert info["completion_records"] == {}
    assert info["counts"]["completed"] == 0
