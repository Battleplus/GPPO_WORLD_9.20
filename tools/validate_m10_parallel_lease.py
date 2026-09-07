"""Direct server smoke validation for the parallel lease contract.

This intentionally uses assertions instead of pytest because the designated
server environment does not install pytest.  It is a contract audit, not a
claim of the full test suite.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_communication import CommunicationProfile
from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.service_clock import ServiceEvent


def env(*, events=(), communication=CommunicationProfile()):
    config = M10Config(uav_count=2, task_capacity=2, horizon=8.0,
                       lease_ttl=2.0, initial_energy=20.0)
    scenario = M10Scenario(
        "parallel-lease",
        (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 7.0, 4.0, 1.0),
         M10TaskSpec("task-1", 0.0, 0.0, 0.0, 7.0, 4.0, 1.0)),
        tuple(events), 101, "server-validation", "parallel-lease-server-101", communication,
    )
    result = M10Environment(config, scenario)
    result.reset()
    return result


def main():
    e = env()
    e.step(0)
    e.step(3)
    for _ in range(4):
        _, _, _, info = e.step(4, submit_command=True)
    assert info["tasks"] == {"task-0": "completed", "task-1": "completed"}
    assert info["task_service"] == {"task-0": 4.0, "task-1": 4.0}
    assert len([x for x in e.execution.log if x["result"] == "renewed"]) >= 2

    e = env(events=(ServiceEvent(2.5, "uav-0", "damage"),))
    e.step(0)
    e.step(3)
    e.step(4, submit_command=True)
    e.step(4, submit_command=True)
    _, _, _, info = e.step(4, submit_command=True)
    assert info["tasks"] == {"task-0": "pending", "task-1": "completed"}
    assert e.execution.status["m10-parallel-lease-0-cmd-00001"] == "revoked"
    assert e.execution.status["m10-parallel-lease-0-cmd-00002"] == "completed"

    print({
        "parallel_completion": info["tasks"],
        "active_leases_after_completion": len(e.execution.leases),
        "renewal_messages": sum(item.get("kind") == "lease_renewal" for item in e._communication_log),
        "duplicate_or_unauthorized": sum(item["result"] in {"duplicate_or_empty_id", "ack_identity", "fenced"}
                                          for item in e.execution.log),
    })


if __name__ == "__main__":
    main()
