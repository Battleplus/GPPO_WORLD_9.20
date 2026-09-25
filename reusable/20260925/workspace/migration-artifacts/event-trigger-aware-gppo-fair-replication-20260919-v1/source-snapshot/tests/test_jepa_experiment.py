import torch
from gppo_world.data import STATE_DIM
from tools.run_j01_experiment import batches, fit_probe, predict_probe, task_errors, cluster_interval, collapse_stats


def test_batches_preserve_every_row_and_no_singleton():
    for size in (64, 65, 129, 664):
        parts = batches(size, 64)
        assert torch.equal(torch.cat(parts), torch.arange(size))
        assert min(map(len, parts)) > 1


def test_ridge_uses_train_scaling_and_recovers_linear_relationship():
    torch.manual_seed(10)
    x = torch.randn(80, 4, dtype=torch.float64)
    v = torch.randn(20, 4, dtype=torch.float64) + 1
    w = torch.randn(4, STATE_DIM + 9, dtype=torch.float64)
    y, vy = x @ w + 3, v @ w + 3
    scale = y.std(0, unbiased=False).clamp_min(0.05)
    active = torch.ones(STATE_DIM, dtype=torch.bool)
    result = fit_probe(x, y, v, vy, scale, active, [0.001, 10])
    assert result["ridge"] == 0.001
    assert torch.equal(result["mean"], x.mean(0))
    assert (predict_probe(result, v) - vy).abs().max() < 0.001


def test_cluster_not_transition_weighted_and_collapse_detected():
    result = cluster_interval([1, 1, 1, 3], ["a", "a", "a", "b"])
    assert result["mean"] == 2 and result["tapes"] == 2
    collapsed = collapse_stats(torch.zeros(100, 32, dtype=torch.float64))
    assert collapsed["mean_std"] == 0 and collapsed["effective_rank_covariance_entropy"] == 0


def test_macro_tasks_equal_weight_not_dimension_weight():
    truth = torch.zeros(3, STATE_DIM + 9)
    pred = truth.clone()
    pred[:, STATE_DIM] = 2
    active = torch.ones(STATE_DIM, dtype=torch.bool)
    tasks, macro = task_errors(pred, truth, torch.ones(STATE_DIM + 9), active)
    assert torch.equal(macro, torch.ones(3))
    assert torch.equal(tasks[:, 1], torch.full((3,), 4.0))
