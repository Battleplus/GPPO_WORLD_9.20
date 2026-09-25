from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gppo_world.calibration import ShadowCalibration, fit_shadow_calibration
from gppo_world.data import STATE_DIM
from gppo_world.model import EventAwareGraphWorldModel
from gppo_world.shadow import ShadowRequest, ShadowRuntime, graph_snapshot_sha256

from test_contracts import make_graph
from test_events import _schema_and_episode


def _calibration(**overrides) -> ShadowCalibration:
    values = dict(
        format_version="gppo-shadow-calibration/0.1.0",
        source_split="test-fixture",
        source_transition_count=1,
        state_change_temperature=1.0,
        continuation_temperature=1.0,
        state_variance_scale=1.0,
        reward_variance_scale=1.0,
        cost_variance_scale=1.0,
        input_mean=(0.0,) * STATE_DIM,
        input_std=(1.0,) * STATE_DIM,
        ood_score_threshold=100.0,
        uncertainty_threshold=100.0,
        latency_p95_budget_ms=25.0,
        latency_p99_budget_ms=50.0,
        timeout_ms=50.0,
    )
    values.update(overrides)
    return ShadowCalibration(**values)


def _runtime(calibration=None):
    schema, _ = _schema_and_episode()
    model = EventAwareGraphWorldModel(event_schema=schema)
    return ShadowRuntime(model, calibration or _calibration(), model_version="fixture")


def _request(step=0, graph=None):
    graph = graph or make_graph(7 + step)
    return ShadowRequest(
        episode_id="episode-1",
        step=step,
        graph=graph,
        executed_action=3,
        evidence=(),
        action_version=step,
        decision_time=float(step),
        execution_accepted=True,
        expected_post_graph_version=graph.graph_version + 1,
        expected_post_action_version=step + 1,
    )


def test_shadow_valid_inference_is_read_only():
    runtime = _runtime()
    request = _request()
    before = graph_snapshot_sha256(request.graph)
    result = runtime.observe(request)
    after = graph_snapshot_sha256(request.graph)
    assert result.valid
    assert before == after
    assert len(result.latent) == 88
    assert all(
        runtime.counters[name] == 0
        for name in (
            "belief_write_count",
            "action_mask_write_count",
            "graph_version_write_count",
            "action_version_write_count",
            "action_submission_count",
        )
    )


def test_stale_before_and_after_never_commit_latent():
    runtime = _runtime()
    request = _request()
    stale_before = runtime.observe(request, version_reader=lambda: (999, 0))
    assert not stale_before.valid and stale_before.fallback_reason == "stale_before"
    calls = iter(
        (
            (request.expected_post_graph_version, request.expected_post_action_version),
            (request.expected_post_graph_version + 1, request.expected_post_action_version),
        )
    )
    stale_after = runtime.observe(request, version_reader=lambda: next(calls))
    assert not stale_after.valid and stale_after.fallback_reason == "stale_after"
    assert stale_after.latent == (0.0,) * 88


def test_timeout_exception_and_ood_use_zero_context():
    timeout_runtime = _runtime()
    timeout = timeout_runtime.observe(_request(), latency_injection_ms=100.0)
    assert not timeout.valid and timeout.fallback_reason == "timeout"
    exception = _runtime().observe(_request(), force_exception=True)
    assert not exception.valid and exception.fallback_reason == "exception"
    graph = make_graph()
    graph.nodes["uav"].fill_(10.0)
    ood = _runtime(_calibration(ood_score_threshold=0.1)).observe(_request(graph=graph))
    assert not ood.valid and ood.fallback_reason == "ood"
    assert ood.latent == (0.0,) * 88


def test_rejected_execution_is_skipped_and_future_evidence_is_rejected():
    runtime = _runtime()
    rejected = replace(_request(), executed_action=None, execution_accepted=False)
    result = runtime.observe(rejected)
    assert not result.valid and result.fallback_reason == "rejected_execution"
    with pytest.raises(ValueError, match="future evidence"):
        replace(
            _request(),
            evidence=({"received_at": 1.0, "payload": {}},),
            decision_time=0.0,
        )


def test_failed_step_forces_history_reset_on_recovery():
    runtime = _runtime()
    assert runtime.observe(_request(step=0)).valid
    assert not runtime.observe(_request(step=1), latency_injection_ms=100.0).valid
    recovered = runtime.observe(_request(step=2))
    assert recovered.valid
    assert recovered.history_reset


def test_calibration_is_fitted_only_from_train_and_validation():
    schema, episode = _schema_and_episode()
    model = EventAwareGraphWorldModel(event_schema=schema)
    calibration = fit_shadow_calibration(model, [episode], [episode])
    assert "validation" in calibration.source_split
    assert "test" not in calibration.source_split
    assert len(calibration.input_mean) == STATE_DIM
    assert calibration.ood_score_threshold >= 0.0
    assert calibration.uncertainty_threshold > 0.0
