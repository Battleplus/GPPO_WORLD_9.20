"""Run paired end-to-end M-10 causal, message and execution checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec, default_scenario, scenario_tape
from gppo_world.m10_training import M10ActorCritic, M10WorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []

    def record(name: str, passed: bool, **details):
        results.append({"name": name, "passed": bool(passed), **details})

    base = M10Scenario("causal", default_scenario("normal", seed=1).tasks, ())
    future_damage = M10Scenario("causal", base.tasks, (default_scenario("uav_damage", seed=1).events[0],))
    left = M10Environment(M10Config(), base).public_snapshot_digest()
    right = M10Environment(M10Config(), future_damage).public_snapshot_digest()
    record("future_event_not_visible", left == right, initial_snapshot_equal=left == right)

    tape_left, tape_right = scenario_tape("train", count=2, base_seed=8101), scenario_tape("test", count=2, base_seed=8101)
    record("seeded_tape_is_distinct", tape_left[0].tape_id != tape_right[0].tape_id and (tape_left[0].tasks != tape_right[0].tasks or tape_left[0].events != tape_right[0].events), train_id=tape_left[0].tape_id, test_id=tape_right[0].tape_id)

    delayed = M10Environment(M10Config(telemetry_delay=1.0, telemetry_max_age=2.0), base)
    delayed.reset()
    before_delivery = bool(delayed.action_mask()[0])
    delivered_obs, _, _, _ = delayed.step(delayed.config.action_count - 1)
    record("delayed_message_delivery", (not before_delivery) and bool(delivered_obs["mask"][0]), before_delivery=before_delivery, after_delivery=bool(delivered_obs["mask"][0]))

    stale = M10Environment(M10Config(), base)
    stale_obs = stale.reset()
    stale._send("task", "task-0", "priority", 1.1)
    stale_result = stale.bridge.submit(0, version=stale_obs["version"], command_id="stale-command")
    record("stale_snapshot_rejected", stale_result == "stale_snapshot", result=str(stale_result))

    ack_scenario = M10Scenario("ack-lease", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 8.0, 4.0, 1.0),), ())
    ack_env = M10Environment(M10Config(uav_count=1, task_capacity=1, lease_ttl=1.5, horizon=5.0), ack_scenario)
    _, _, _, ack_info = ack_env.step(0)
    _, _, _, lease_info = ack_env.step(1)
    record("ack_and_lease_progression", ack_info["feedback"] == "accepted" and lease_info["tasks"]["task-0"] == "pending", ack_feedback=ack_info["feedback"], after_lease=lease_info["tasks"]["task-0"])

    hidden_energy = M10Environment(M10Config(), base)
    hidden_energy.reset()
    visible_energy_mask = bool(hidden_energy.action_mask()[0])
    hidden_energy.clock.resources["uav-0"].energy = 0.0
    _, _, _, energy_info = hidden_energy.step(0)
    record("execution_truth_rejection", energy_info["feedback"] == "energy" and visible_energy_mask, feedback=energy_info["feedback"], policy_mask_before_truth_change=visible_energy_mask)

    event_env = M10Environment(M10Config(horizon=10.0), default_scenario("composite", seed=8))
    event_env.reset()
    event_kinds = []
    for _ in range(10):
        _, _, done, info = event_env.step(event_env.config.action_count - 1)
        event_kinds.extend(item["kind"] for item in info["new_events"] if item["kind"] in ("damage", "disconnect", "reconnect"))
        if done:
            break
    record("scheduled_event_dispatch", set(("damage", "disconnect", "reconnect")) <= set(event_kinds), event_kinds=event_kinds)

    complete_scenario = M10Scenario("complete", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 4.0, 1.0, 1.0),), ())
    complete_env = M10Environment(M10Config(uav_count=1, task_capacity=1, horizon=5.0), complete_scenario)
    _, _, _, complete_info = complete_env.step(0)
    record("completion_classification", complete_info["tasks"]["task-0"] == "completed", task_state=complete_info["tasks"]["task-0"])

    graph_env = M10Environment(M10Config(), scenario_tape("train", count=1, base_seed=8201)[0])
    graph_obs = graph_env.reset()
    graph_model = M10ActorCritic(uav_count=4, task_capacity=6, action_count=25, encoder="graph", type_count=5, history=False)
    graph_model.eval()
    original = torch.tensor(graph_obs["flat"], dtype=torch.float32)
    swapped = original.clone()
    task_start = (4 + graph_env.config.region_count + graph_env.config.target_count) * 32
    swapped[task_start:task_start + 32], swapped[task_start + 32:task_start + 64] = original[task_start + 32:task_start + 64].clone(), original[task_start:task_start + 32].clone()
    relation_start = (4 + graph_env.config.region_count + graph_env.config.target_count + graph_env.config.task_capacity + graph_env.config.event_capacity) * 32
    relation_end = relation_start + 4 * 6 * 4
    relation = swapped[relation_start:relation_end].clone().reshape(4, 6, 4)
    relation[:, :2, :] = relation[:, [1, 0], :].clone()
    swapped[relation_start:relation_end] = relation.reshape(-1)
    with torch.no_grad():
        original_logits = graph_model(original[None, :])[0]
        swapped_logits = graph_model(swapped[None, :])[0]
    graph_ok = torch.allclose(original_logits[0, [0, 1]], swapped_logits[0, [1, 0]], atol=1e-5) and torch.allclose(original_logits[0, [6, 7]], swapped_logits[0, [7, 6]], atol=1e-5)
    record("candidate_identity_permutation", graph_ok)

    world_model = M10WorldModel(obs_dim=4, action_count=3)
    try:
        world_model.context_for(torch.zeros((1, 4)))
        context_contract = False
    except ValueError:
        context_contract = True
    fallback, fallback_active, fallback_risk = world_model.context_for(torch.ones((1, 4)) * 100.0, action_mask=torch.tensor([True, False, False]))
    record("world_context_action_and_ood_fallback", context_contract and not fallback_active and fallback_risk == 1.0 and tuple(fallback.shape) == (1, 8))

    summary = {"format": "m10-causal-acceptance/0.2.0", "checks": results, "passed": sum(item["passed"] for item in results), "total": len(results), "all_passed": all(item["passed"] for item in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
