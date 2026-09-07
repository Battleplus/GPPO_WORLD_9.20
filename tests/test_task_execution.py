from gppo_world.service_clock import ServiceClock, ServiceResource, ServiceEvent
from gppo_world.task_lifecycle import TaskLifecycle
from gppo_world.task_execution import TaskExecution, TaskCommand


def gate(events=()):
    c = ServiceClock({'t': TaskLifecycle('t', 0, 20, 10)},
                     {'u': ServiceResource(20, 1, 1), 'v': ServiceResource(20, 1, 1)},
                     list(events), disconnect_interrupts=True)
    return TaskExecution(c, command_ttl=1, lease_ttl=2)


def propose(g, cid='c', uid='u'):
    return g.propose(cid, 't', uid, version=g.version, proposal_allowed=True)


def test_assignment_waits_for_matching_ack_and_duplicate_never_reexecutes():
    g = gate()
    c = propose(g)
    assert isinstance(c, TaskCommand)
    assert g.clock.tasks['t'].assigned_uav is None
    assert g.acknowledge('c', 'v', c.token) == 'ack_identity'
    assert g.acknowledge('c', 'u', c.token + 1) == 'ack_identity'
    assert g.acknowledge('c', 'u', c.token) == 'accepted'
    assert g.acknowledge('c', 'u', c.token) == 'inactive_command'
    assert propose(g) == 'duplicate_or_empty_id'
    assert g.clock.resources['u'].energy == 20


def test_stale_ack_retry_rechecks_actual_energy():
    g = gate()
    c = propose(g)
    g.observe_version(1)
    assert g.acknowledge('c', 'u', c.token) == 'stale'
    g.clock.resources['u'].energy = 0  # execution-only disturbance before retry
    assert propose(g, 'retry') == 'energy'
    assert g.clock.tasks['t'].assigned_uav is None


def test_ack_expiry_boundary_is_rejected_without_assignment():
    g = gate()
    c = propose(g)
    g.advance(1)
    assert g.acknowledge('c', 'u', c.token) == 'ack_timeout'
    assert g.clock.tasks['t'].service == 0


def test_competing_pending_commands_have_one_holder():
    g = gate()
    a, b = propose(g, 'a'), propose(g, 'b', 'v')
    assert g.acknowledge('b', 'v', b.token) == 'accepted'
    assert g.acknowledge('a', 'u', a.token) == 'fenced'
    assert g.clock.tasks['t'].assigned_uav == 'v'


def test_lease_expiration_stops_service_inside_long_step():
    g = gate()
    c = propose(g)
    g.acknowledge('c', 'u', c.token)
    g.advance(5)
    assert g.clock.tasks['t'].service == 2
    assert g.clock.resources['u'].energy == 18
    assert g.status['c'] == 'lease_expired'
    assert g.renew('c', 'u', c.token) == 'inactive_lease'
    retry = propose(g, 'retry', 'v')
    assert retry.token > c.token
    assert g.acknowledge('retry', 'v', retry.token) == 'accepted'


def test_renewal_extends_execution_but_disconnect_revokes_it():
    g = gate([ServiceEvent(2.5, 'u', 'disconnect')])
    c = propose(g)
    g.acknowledge('c', 'u', c.token)
    g.advance(1.5)
    assert g.renew('c', 'u', c.token) == 'renewed'
    g.advance(4)
    assert g.clock.tasks['t'].service == 2.5
    assert g.status['c'] == 'revoked'
    assert g.renew('c', 'u', c.token) == 'inactive_lease'
