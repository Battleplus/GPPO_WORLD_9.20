from dataclasses import replace

from gppo_world.m10_communication import weak_communication_profile
from gppo_world.m10_environment import M10Config
from tools.diagnose_m10_weak_comm import (
    minimal_config, minimal_scenarios, run_legal_scheduler,
)


def test_legal_scheduler_completes_minimal_task_under_ideal_link():
    scenario = replace(minimal_scenarios()["one_uav_one_task"],
                       communication=weak_communication_profile("ideal"))
    result = run_legal_scheduler(scenario, minimal_config("one_uav_one_task"), trace=True)
    assert result["completed"] == 1
    assert result["accepted_commands"] == 1
    assert result["communication"]["telemetry"]["conservation"]["original_wire_attempts"] > 0


def test_legal_scheduler_recovers_with_other_uav_after_disconnect():
    scenario = replace(minimal_scenarios()["one_uav_recovery"],
                       communication=weak_communication_profile("recovery"))
    result = run_legal_scheduler(scenario, minimal_config("one_uav_recovery"), trace=True)
    assert result["completed"] == 1
    assert result["accepted_commands"] >= 2
    accepted = [item["command_id"] for item in result["execution_log"]
                if item["result"] == "accepted"]
    assert len(accepted) == len(set(accepted))
