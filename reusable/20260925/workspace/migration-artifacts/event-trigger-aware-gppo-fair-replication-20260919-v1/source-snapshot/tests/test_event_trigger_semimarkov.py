import numpy as np

from gppo_world.event_trigger_semimarkov import DecisionWindow, decision_level_gae, discounted_window, validate_window_contract


def test_discounted_window_k_one_and_k_three():
    reward, k, discount = discounted_window([[1.0, 2.0]], 0.9)
    np.testing.assert_allclose(reward, [1.0, 2.0])
    assert k == 1 and abs(discount - 0.9) < 1e-12
    reward, k, discount = discounted_window([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 0.9)
    np.testing.assert_allclose(reward, [1.0 + 0.9 * 3.0 + 0.9**2 * 5.0, 2.0 + 0.9 * 4.0 + 0.9**2 * 6.0])
    assert k == 3 and abs(discount - 0.9**3) < 1e-12


def test_terminal_truncation_and_rollout_boundary():
    terminal = DecisionWindow(np.asarray([2.0, 0.0]), 2, 0.81, np.asarray([1.0, 1.0]), np.zeros(2), True)
    trunc = DecisionWindow(np.asarray([1.0, 0.0]), 1, 0.9, np.asarray([0.0, 0.0]), np.asarray([2.0, 0.0]), False, True)
    boundary = DecisionWindow(np.asarray([1.0, 0.0]), 1, 0.9, np.asarray([0.0, 0.0]), np.asarray([2.0, 0.0]), False, rollout_boundary=True)
    validate_window_contract([terminal, trunc, boundary], gamma=0.9)
    advantages, returns = decision_level_gae([terminal, trunc, boundary], 0.9, 0.95)
    assert np.isfinite(advantages).all() and np.isfinite(returns).all()
    assert np.allclose(advantages[0], returns[0] - terminal.value)


def test_boundary_cuts_recursive_gae_but_keeps_bootstrap():
    boundary = DecisionWindow(np.asarray([1.0, 0.0]), 2, 0.25, np.zeros(2), np.ones(2), False, rollout_boundary=True)
    advantages, _ = decision_level_gae([boundary], 0.5, 0.95)
    np.testing.assert_allclose(advantages[0], [1.25, 0.25])

