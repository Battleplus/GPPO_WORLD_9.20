import pytest
from gppo_world.task_lifecycle import TaskLifecycle, TaskState
from gppo_world.service_clock import ServiceClock, ServiceEvent, ServiceResource


def clock(events=(), energy=10, service=8, deadline=20, disconnect=True):
    return ServiceClock({'t': TaskLifecycle('t', 0, deadline, service)},
                        {'u': ServiceResource(energy, 1, 1)}, list(events),
                        disconnect_interrupts=disconnect)


def test_damage_splits_interval_and_stops_service():
    c = clock([ServiceEvent(2, 'u', 'damage')])
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].service == 2 and c.resources['u'].energy == 8
    assert c.tasks['t'].state == TaskState.PENDING
    with pytest.raises(ValueError):
        c.assign('t', 'u')


def test_exhaustion_stops_before_end_without_negative_energy():
    c = clock(energy=2)
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].service == 2 and c.resources['u'].energy == 0
    assert c.tasks['t'].state == TaskState.PENDING


def test_completion_does_not_charge_remaining_interval():
    c = clock(service=2)
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].completed_at == 2 and c.resources['u'].energy == 8


def test_reconnect_requires_reassignment_if_interrupted():
    c = clock([ServiceEvent(2, 'u', 'disconnect'), ServiceEvent(4, 'u', 'reconnect')])
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].service == 2
    c.assign('t', 'u')
    c.advance(6)
    assert c.tasks['t'].service == 3


def test_link_policy_is_explicit_autonomous_service_may_continue():
    c = clock([ServiceEvent(2, 'u', 'disconnect')], disconnect=False)
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].service == 5


def test_completion_at_damage_and_deadline_boundary():
    c = clock([ServiceEvent(2, 'u', 'damage')], service=2, deadline=2)
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].state == TaskState.COMPLETED
    assert c.tasks['t'].completed_at == 2


def test_partial_service_expires_at_deadline():
    c = clock(deadline=3)
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].state == TaskState.EXPIRED
    assert c.tasks['t'].service == 3 and c.resources['u'].energy == 7


def test_near_co_located_task_does_not_create_zero_duration_travel_boundary():
    c = ServiceClock(
        {'t': TaskLifecycle('t', 0, 20, 8)},
        {'u': ServiceResource(10, 1, 1)},
        [],
        disconnect_interrupts=True,
        task_positions={'t': (8.881784197001252e-16, 0.0)},
    )
    c.assign('t', 'u')
    c.advance(5)
    assert c.tasks['t'].service == 5
    assert c.resources['u'].energy == 5
