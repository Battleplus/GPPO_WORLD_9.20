from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, M10TaskSpec
from gppo_world.m10_training import (
    EVENT_NAMES, M10WorldModel, _policy_input, _policy_input_bundle,
    evaluate_world_model_rows,
)


def test_one_snapshot_world_context_is_reused_for_forced_trigger_context():
    config = M10Config(uav_count=1, task_capacity=1, horizon=5.0)
    scenario = M10Scenario("r3", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 10.0, 3.0, 1.0),), ())
    env = M10Environment(config, scenario)
    obs = env.reset()
    model = M10WorldModel(len(obs["flat"]), config.action_count)
    with patch.object(model, "context_for", wraps=model.context_for) as context_for:
        gated, active, risk, full = _policy_input_bundle(
            env, obs, model, fusion="triggered", device=torch.device("cpu"), trigger_threshold=1.1,
        )
        forced, forced_active, forced_risk = _policy_input(
            env, obs, model, fusion="triggered", device=torch.device("cpu"),
            trigger_threshold=1.1, force_context=True,
        )
    assert context_for.call_count == 2
    assert not active and forced_active
    assert forced_risk == risk
    assert len(gated) == len(obs["flat"])
    assert len(forced) == len(obs["flat"]) + model.context_dim
    np.testing.assert_allclose(forced, full)


def test_prediction_metrics_include_rare_event_pr_auc_and_split_rates():
    config = M10Config(uav_count=1, task_capacity=1, horizon=3.0)
    scenario = M10Scenario("r3", (M10TaskSpec("task-0", 0.0, 0.0, 0.0, 10.0, 2.0, 1.0),), ())
    env = M10Environment(config, scenario)
    obs = env.reset()
    rows = []
    for index, target in enumerate((0.0, 1.0, 0.0)):
        rows.append({
            "obs": obs["flat"].copy(), "action": 1, "reward": float(index),
            "event_target": np.asarray([target, 0.0, 0.0, 0.0], dtype=np.float32),
            "persistence_event_target": np.asarray([target, 0.0, 0.0, 0.0], dtype=np.float32),
            "done": float(index == 2), "split": "validation", "tape_id": f"r3-{index}",
        })
    model = M10WorldModel(len(obs["flat"]), config.action_count)
    metrics = evaluate_world_model_rows(model, rows, baseline_rows=rows)
    assert tuple(metrics["event"]) == EVENT_NAMES
    assert metrics["event"]["damage"]["positive_count"] == 1.0
    assert metrics["event"]["damage"]["pr_auc"] >= 0.0
    assert metrics["split_counts"]["validation"]["rows"] == 3
    assert "persistence_event" in metrics["baselines"]
