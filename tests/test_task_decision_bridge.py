from gppo_world.task_decision_bridge import TaskDecisionBridge
from gppo_world.task_execution import TaskExecution, TaskCommand
from gppo_world.task_lifecycle import TaskLifecycle
from gppo_world.service_clock import ServiceClock, ServiceResource
from gppo_world.task_policy_view import TaskPolicyView
from gppo_world.telemetry import Telemetry


def bridge(energy=10):
    view = TaskPolicyView(('u',), task_capacity=2, max_age=2)
    for kind, entity, fields in (
        ('uav', 'u', dict(x=0, y=0, energy=10, alive=1, connected=1, idle=1)),
        ('task', 't', dict(x=0, y=0, deadline=10, remaining_service=3, priority=1, pending=1)),
    ):
        for field, value in fields.items():
            view.receive(kind, Telemetry(entity, field, value, 0, 0, 0), 0)
    clock = ServiceClock({'t': TaskLifecycle('t', 0, 10, 3)},
                         {'u': ServiceResource(energy, 1, 1)}, [], disconnect_interrupts=True)
    return TaskDecisionBridge(view, TaskExecution(clock, command_ttl=1, lease_ttl=5))


def test_hidden_energy_changes_execution_but_not_policy_observation():
    a, b = bridge(10), bridge(0)
    x, y = a.observe(), b.observe()
    assert x == y and x.mask[0]
    assert isinstance(a.submit(0, version=x.version, command_id='c'), TaskCommand)
    assert b.submit(0, version=y.version, command_id='c') == 'energy'
    assert a.observe() == b.observe()  # no immediate truth feedback injected


def test_received_change_invalidates_previously_issued_decision():
    b = bridge()
    old = b.observe()
    b.view.receive('uav', Telemetry('u', 'energy', 9, 0, 0, 1), 0)
    assert b.submit(0, version=old.version, command_id='c') == 'stale_snapshot'
    assert not b.execution.commands


def test_proposal_ack_and_physical_completion_form_one_path():
    b = bridge()
    snapshot = b.observe()
    command = b.submit(0, version=snapshot.version, command_id='c')
    assert b.execution.clock.tasks['t'].service == 0
    assert b.execution.acknowledge('c', 'u', command.token) == 'accepted'
    b.execution.advance(4)
    assert b.execution.clock.tasks['t'].completed_at == 3
    assert b.execution.clock.resources['u'].energy == 7
    assert b.execution.status['c'] == 'completed'


def test_padding_and_noop_never_create_assignment():
    b = bridge()
    snapshot = b.observe()
    assert b.submit(1, version=snapshot.version, command_id='c') == 'masked'
    assert b.submit(2, version=snapshot.version, command_id='d') == 'noop'
    assert not b.execution.commands
