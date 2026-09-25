from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest
import torch

from gppo_world.joint_gppo import (
    ActionConditionedTemporalWorldModel,
    EVENT_NAMES,
    JointGraphPreferencePolicy,
    build_observed_event_labels,
    masked_event_bce,
    preco_logit_direction,
    preference_similarity_coefficients,
)
from gppo_world.m10_environment import M10Config, M10Environment, default_scenario
from gppo_world.joint_training import (
    GROUPS, JointTrainConfig, _batch_encode, _gae_vector, _joint_logprob_rows,
    _masked_smooth_l1, _preference_after_episode, _preference_at_rollout_boundary,
    _TrainingLedger, _atomic_torch_save, verify_commit_pointer,
)
from gppo_world.m10_training import masked_distribution


def _obs(task_ids=("t0", "t1"), *, order=(0, 1), task_pending=(1.0, 1.0),
         task_remaining=(2.0, 2.0), task_deadlines=(3.0, 8.0), time=0.0):
    uavs = np.zeros((4, 24), dtype=np.float32)
    # UAV field layout is x,y,energy,alive,connected,idle; each has value/known/valid/age.
    for i in range(4):
        for field, value in enumerate((0.0, 0.0, 9.0, 1.0, 1.0, 1.0)):
            uavs[i, field * 4:field * 4 + 4] = (value, 1.0, 1.0, 0.0)
    tasks = np.zeros((6, 32), dtype=np.float32)
    for row, original in enumerate(order):
        for field, value in enumerate((1.0, 1.0, task_deadlines[original], task_remaining[original], 1.0,
                                       task_pending[original], 0.0, 0.0)):
            tasks[row, field * 4:field * 4 + 4] = (value, 1.0, 1.0, 0.0)
    return {
        "flat": np.zeros(1, dtype=np.float32), "uavs": uavs, "tasks": tasks,
        "time": float(time), "mask": np.ones(25, dtype=np.bool_),
        "public_entity_ids": {"uavs": tuple(f"uav-{i}" for i in range(4)), "tasks": tuple(task_ids[i] for i in order)},
    }


def test_public_event_labels_are_identity_aligned_and_deterministic():
    before = _obs(task_ids=("A", "B"), task_remaining=(2.0, 2.0))
    after = _obs(task_ids=("A", "B"), task_remaining=(1.0, 2.0), time=1.0)
    first = build_observed_event_labels(before, after, initial_energy=9.0)
    second = build_observed_event_labels(before, after, initial_energy=9.0)
    assert np.array_equal(first["labels"], second["labels"])
    assert np.array_equal(first["mask"], second["mask"])
    assert first["labels"][EVENT_NAMES.index("public_task_state_change")] == 1.0


def test_future_truth_not_in_event_label_or_online_model_signature():
    before = _obs()
    after = _obs(time=1.0)
    first = build_observed_event_labels(before, after, initial_energy=9.0)
    before_private = {**before, "hidden_truth_damage": True}
    after_private = {**after, "hidden_truth_damage": False}
    second = build_observed_event_labels(before_private, after_private, initial_energy=9.0)
    assert np.array_equal(first["labels"], second["labels"])
    assert "next_observation" not in inspect.signature(ActionConditionedTemporalWorldModel.forward).parameters
    assert "labels" not in inspect.signature(ActionConditionedTemporalWorldModel.forward).parameters


def test_task_slot_permutation_does_not_break_entity_matching():
    before = _obs(task_ids=("A", "B"), task_remaining=(2.0, 2.0), order=(0, 1))
    after = _obs(task_ids=("A", "B"), task_remaining=(1.0, 2.0), order=(1, 0), time=1.0)
    result = build_observed_event_labels(before, after, initial_energy=9.0)
    assert result["labels"][EVENT_NAMES.index("public_task_state_change")] == 1.0
    assert result["mask"][EVENT_NAMES.index("public_task_state_change")]


def test_new_identity_is_not_misread_as_same_slot_task_change():
    before = _obs(task_ids=("A", "B"))
    after = _obs(task_ids=("B", "C"), order=(1, 0), time=1.0)
    result = build_observed_event_labels(before, after, initial_energy=9.0)
    assert result["labels"][EVENT_NAMES.index("public_task_state_change")] == 0.0
    assert result["mask"][EVENT_NAMES.index("public_task_state_change")]


def test_episode_boundary_masks_every_event_label_even_if_slots_and_ids_repeat():
    before = _obs(task_ids=("A", "B"), task_remaining=(2.0, 2.0))
    after = _obs(task_ids=("A", "B"), task_remaining=(0.0, 0.0), time=0.0)
    result = build_observed_event_labels(before, after, initial_energy=9.0, same_episode=False)
    assert not result["mask"].any()
    assert not result["labels"].any()
    assert all(reason["episode_boundary"] == 1 for reason in result["reasons"].values())


def test_urgent_task_requires_new_public_identity_and_valid_fields():
    before = _obs(task_ids=("A", "B"), task_deadlines=(2.0, 8.0))
    after = _obs(task_ids=("A", "C"), task_deadlines=(2.0, 1.5), time=1.0)
    result = build_observed_event_labels(before, after, initial_energy=9.0, urgent_slack=1.0)
    idx = EVENT_NAMES.index("urgent_task_first_public")
    assert result["mask"][idx] and result["labels"][idx] == 1.0


def test_masked_event_loss_ignores_invalid_fields_and_is_zero_if_all_masked():
    logits = torch.zeros((2, 5), requires_grad=True)
    labels = torch.ones((2, 5))
    mask = torch.zeros((2, 5), dtype=torch.bool)
    loss, count = masked_event_bce(logits, labels, mask)
    assert count == 0 and loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_masked_world_regression_returns_zero_when_transition_is_censored():
    prediction = torch.randn((2, 4), requires_grad=True)
    target = torch.randn((2, 4))
    loss, count = _masked_smooth_l1(prediction, target, torch.zeros(2, dtype=torch.bool))
    assert count == 0 and loss.item() == 0.0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_action_conditioned_world_predictions_are_per_candidate_and_recursive():
    model = ActionConditionedTemporalWorldModel()
    observation = {"mask": np.ones(25, dtype=np.bool_), "graph": {"relations": np.zeros((4, 6, 4), dtype=np.float32)}}
    features, outputs = model.predict_all_candidates(torch.ones((1, 128)), None, observation)
    assert features.shape == (1, 25, 17)
    assert not torch.allclose(outputs[0]["latent"], outputs[1]["latent"])
    recursive = model.recursive_rollout(torch.ones((1, 128)), [0, 24])
    assert len(recursive) == 2 and recursive[-1]["next_state"].shape == (1, 128)


def test_event_loss_has_gradient_into_shared_graph_history_encoder():
    torch.manual_seed(1)
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    policy = JointGraphPreferencePolicy(env_config)
    world = ActionConditionedTemporalWorldModel()
    obs = torch.randn((2, policy.base.base_obs_dim))
    features, _, _ = policy.encode(obs)
    output = world(features, torch.tensor([0, 24]), torch.zeros((2, 4)))
    labels = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss, count = masked_event_bce(output["event_logits"], labels, mask)
    gradients = torch.autograd.grad(loss, tuple(policy.base.token_encoder.parameters()), allow_unused=True)
    assert count == 10
    assert any(gradient is not None and float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_preco_combines_two_objective_gradients_and_masked_policy_is_legal():
    logits = torch.randn((3, 25), requires_grad=True)
    mask = torch.zeros((3, 25), dtype=torch.bool)
    mask[:, [0, 4, 24]] = True
    distribution = masked_distribution(logits, mask)
    actions = distribution.sample()
    assert all(mask[index, action] for index, action in enumerate(actions))
    losses = [-(logits[:, 0].mean()), -(logits[:, 4].mean())]
    similarity = -(distribution.probs[:, :5].mean())
    direction, weight = preco_logit_direction(losses, similarity, logits, strength=0.1)
    assert 0.0 <= weight <= 1.0 and torch.isfinite(direction).all()


def test_preference_similarity_uses_detached_rollout_returns_and_moves_hand_fixture_by_preference():
    # Two actions: action 0 earns only task utility; action 1 earns only energy utility.
    # At uniform behavior, the observed return target maps to utility [0.5, 0.5].
    raw_return_target = torch.tensor([[0.0, -0.5]])

    def one_update_direction(preference):
        logits = torch.zeros((1, 2), requires_grad=True)
        probabilities = torch.softmax(logits, dim=-1)
        coeff, utility, clipped = preference_similarity_coefficients(
            torch.tensor([preference], dtype=torch.float32), raw_return_target,
        )
        assert torch.equal(utility, torch.tensor([[0.5, 0.5]]))
        assert clipped == 0
        # These are local policy-surrogate objectives, not a critic/Q prediction.
        objective_losses = [-probabilities[:, 0].mean(), -probabilities[:, 1].mean()]
        similarity_local = (coeff * probabilities).sum(dim=-1).mean()
        direction, _ = preco_logit_direction(objective_losses, similarity_local, logits, strength=0.1)
        return direction.detach()

    task_direction = one_update_direction((0.9, 0.1))
    energy_direction = one_update_direction((0.1, 0.9))
    assert task_direction[0, 0] > task_direction[0, 1]
    assert energy_direction[0, 1] > energy_direction[0, 0]
    assert not torch.allclose(task_direction, energy_direction)


def test_preference_critic_is_state_value_not_candidate_action_q():
    torch.manual_seed(19)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    policy = JointGraphPreferencePolicy(config)
    features = torch.randn((2, 128))
    pair = torch.randn((2, 4, 6, 64))
    preference = torch.tensor([[0.8, 0.2], [0.15, 0.85]])
    mask = torch.ones((2, 25), dtype=torch.bool)
    first = policy.evaluate_encoded(features, pair, preference, torch.zeros((2, 25, 17)), mask)
    second = policy.evaluate_encoded(features, pair, preference, torch.randn((2, 25, 17)), mask)
    assert first["state_values"].shape == (2, 2)
    assert torch.equal(first["state_values"], second["state_values"])
    assert not hasattr(policy, "vector_q")


def test_vector_critic_fits_raw_return_without_driving_policy_similarity_gradient():
    torch.manual_seed(23)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    policy = JointGraphPreferencePolicy(config)
    features = torch.randn((1, 128))
    pair = torch.randn((1, 4, 6, 64))
    preference = torch.tensor([[0.75, 0.25]])
    mask = torch.ones((1, 25), dtype=torch.bool)
    output = policy.evaluate_encoded(features, pair, preference, torch.zeros((1, 25, 17)), mask)
    action_loss = -output["distribution"].log_prob(torch.tensor([3])).mean()
    critic_loss = torch.nn.functional.mse_loss(output["state_values"], torch.tensor([[0.2, -0.3]]))
    critic_parameters = tuple(policy.vector_value.parameters())
    action_grads = torch.autograd.grad(action_loss, critic_parameters, allow_unused=True, retain_graph=True)
    critic_grads = torch.autograd.grad(critic_loss, critic_parameters, allow_unused=True)
    assert all(gradient is None for gradient in action_grads)
    assert any(gradient is not None and float(gradient.abs().sum()) > 0 for gradient in critic_grads)
    coefficients, utility, _ = preference_similarity_coefficients(
        preference, torch.tensor([[0.2, -0.3]], requires_grad=True),
    )
    assert not coefficients.requires_grad and not utility.requires_grad


def test_each_observed_event_head_has_a_positive_fixture_and_stale_identity_is_masked():
    before = _obs(task_ids=("A", "B"))

    urgent_after = _obs(task_ids=("A", "C"), task_deadlines=(2.0, 0.5), time=0.0)
    state_after = _obs(task_ids=("A", "B"), task_remaining=(1.0, 2.0), time=1.0)

    def changed_uav(*, alive=None, connected=None, energy=None, ids=None, field_valid=True):
        result = _obs(time=1.0)
        result["public_entity_ids"]["uavs"] = tuple(ids or (f"uav-{i}" for i in range(4)))
        for row in result["uavs"]:
            if alive is not None:
                row[3 * 4] = alive
            if connected is not None:
                row[4 * 4] = connected
            if energy is not None:
                row[2 * 4] = energy
            if not field_valid:
                row[2 * 4 + 2] = 0.0
        return result

    cases = (
        (0, before, urgent_after),
        (1, before, state_after),
        (2, before, changed_uav(alive=0.0)),
        (3, before, changed_uav(connected=0.0)),
        (4, before, changed_uav(energy=0.5)),
    )
    for index, old, new in cases:
        result = build_observed_event_labels(old, new, initial_energy=9.0, urgent_slack=1.0)
        assert result["mask"][index], EVENT_NAMES[index]
        assert result["labels"][index] == 1.0, EVENT_NAMES[index]

    stale = build_observed_event_labels(before, changed_uav(energy=0.5, ids=("new-0", "new-1", "new-2", "new-3")),
                                        initial_energy=9.0)
    assert not stale["mask"][EVENT_NAMES.index("uav_crossed_low_energy_threshold")]


def test_event_head_negative_and_invalid_mask_fixtures_are_separate():
    before = _obs()
    unchanged = _obs(time=1.0)
    result = build_observed_event_labels(before, unchanged, initial_energy=9.0)
    assert np.all(result["mask"])
    assert not np.any(result["labels"])

    invalid_after = _obs(time=1.0)
    invalid_after["uavs"][:, 2 * 4 + 2] = 0.0
    invalid = build_observed_event_labels(before, invalid_after, initial_energy=9.0)
    energy_idx = EVENT_NAMES.index("uav_crossed_low_energy_threshold")
    assert not invalid["mask"][energy_idx]
    assert invalid["reasons"][EVENT_NAMES[energy_idx]]["field_invalid"] > 0


def test_sampling_old_logprob_and_cached_prediction_recompute_share_distribution_definition():
    torch.manual_seed(7)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    policy = JointGraphPreferencePolicy(config)
    observation = torch.randn((1, policy.base.base_obs_dim))
    features, pair, _ = policy.encode(observation)
    candidate_features = torch.randn((1, 25, 17))
    mask = torch.zeros((1, 25), dtype=torch.bool)
    mask[:, [0, 7, 24]] = True
    preference = torch.tensor([[0.7, 0.3]])
    before = policy.evaluate_encoded(features, pair, preference, candidate_features, mask)
    action = before["distribution"].sample()
    old_log_probability = before["distribution"].log_prob(action)
    # This is the same training-time path: cached candidate predictions, same
    # mask and preference, with the current actor/critic distribution function.
    after = policy.evaluate_encoded(features, pair, preference, candidate_features, mask)
    new_log_probability = after["distribution"].log_prob(action)
    assert torch.equal(old_log_probability, new_log_probability)
    assert bool(mask[0, action.item()])


def test_ppo_behavior_logprob_replay_uses_per_decision_cuda_encoding_path():
    # Batched CUDA GEMMs can differ slightly from the batch-1 forward used
    # during collection. Recovery and PPO replay must reproduce that path.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2718)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2718)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    policy = JointGraphPreferencePolicy(config, history=True).to(device)
    policy.train()
    obs_dim = policy.base.base_obs_dim
    generator = torch.Generator(device="cpu").manual_seed(1618)
    observations = torch.randn((32, obs_dim), generator=generator).numpy().astype(np.float32)
    hidden_rows = np.zeros((32, 128), dtype=np.float32)
    mask = np.ones(25, dtype=np.bool_)
    preference = np.asarray((0.65, 0.35), dtype=np.float32)
    candidates = np.zeros((25, 17), dtype=np.float32)
    transitions = []
    expected_log_probs = []
    for index in range(len(observations)):
        obs_tensor = torch.as_tensor(observations[index:index + 1], dtype=torch.float32, device=device)
        hidden_tensor = torch.zeros((1, 1, 128), dtype=torch.float32, device=device)
        pref_tensor = torch.as_tensor(preference[None, :], dtype=torch.float32, device=device)
        candidate_tensor = torch.as_tensor(candidates[None, :, :], dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        with torch.no_grad():
            features, pair, _ = policy.encode(obs_tensor, hidden_tensor)
            distribution = policy.evaluate_encoded(
                features, pair, pref_tensor, candidate_tensor, mask_tensor,
            )["distribution"]
            action = int(index % 25)
            expected_log_probs.append(float(distribution.log_prob(torch.tensor([action], device=device)).item()))
        transitions.append({
            "obs": observations[index],
            "policy_hidden_before": hidden_rows[index],
            "mask": mask,
            "preference": preference,
            "candidate_features": candidates,
            "action": action,
        })
    replayed = _joint_logprob_rows(policy, transitions, device).cpu().numpy()
    assert np.max(np.abs(replayed - np.asarray(expected_log_probs, dtype=np.float32))) <= 1e-6


def test_identity_sidecar_does_not_change_flat_features_or_mask_or_protocol_gates():
    scenario = default_scenario("normal", seed=33)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    env = M10Environment(config, scenario)
    obs = env.reset()
    assert tuple(obs["public_entity_ids"]["uavs"]) == tuple(env.uav_ids)
    assert tuple(obs["public_entity_ids"]["tasks"]) == env.view.public_task_ids
    assert np.array_equal(obs["mask"], env.action_mask())
    before_flat = obs["flat"].copy()
    before_mask = obs["mask"].copy()
    invalid = int(np.flatnonzero(~before_mask)[0]) if np.any(~before_mask) else -1
    if invalid >= 0:
        next_obs, _, _, info = env.step(invalid)
        assert info["feedback"] in ("masked", "rejected", "invalid", "expired", "fenced", "unknown") or "reject" in str(info["feedback"]).lower()
        assert np.array_equal(obs["flat"], before_flat)
        assert np.array_equal(obs["mask"], before_mask)
        assert "public_entity_ids" in next_obs


def test_environment_snapshot_is_pickle_recoverable_at_exact_public_state():
    env = M10Environment(
        M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival"),
        default_scenario("normal", seed=41),
    )
    env.reset()
    restored = pickle.loads(pickle.dumps(env))
    assert restored.public_snapshot_digest() == env.public_snapshot_digest()


def test_group_matrix_keeps_bcd_rewards_and_event_ablation_explicit():
    assert GROUPS["A"].vector_reward is False
    assert all(GROUPS[group].preference and GROUPS[group].vector_reward for group in ("B", "C", "D"))
    assert GROUPS["B"].world_model is False
    assert GROUPS["C"].world_model and not GROUPS["C"].event_auxiliary
    assert GROUPS["D"].world_model and GROUPS["D"].event_auxiliary


def test_preference_history_and_gae_are_correct_across_rollout_boundary():
    rng = np.random.default_rng(471001)
    preference = torch.as_tensor(rng.dirichlet(np.ones(2)), dtype=torch.float32)
    rng_state_at_cut = rng.bit_generator.state
    first_rollout_pref = _preference_at_rollout_boundary(preference)
    second_rollout_pref = _preference_at_rollout_boundary(first_rollout_pref)
    assert first_rollout_pref is preference and second_rollout_pref is preference
    assert rng.bit_generator.state == rng_state_at_cut

    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    torch.manual_seed(71)
    policy = JointGraphPreferencePolicy(config, history=True).eval()
    obs1 = torch.randn((1, policy.base.base_obs_dim))
    obs2 = torch.randn((1, policy.base.base_obs_dim))
    with torch.no_grad():
        _, _, hidden1 = policy.encode(obs1, None)
        _, _, hidden_split = policy.encode(obs2, hidden1)
        _, _, hidden_full = policy.encode(obs2, policy.encode(obs1, None)[2])
    assert torch.equal(hidden_split, hidden_full)

    train = JointTrainConfig(gamma=0.9, gae_lambda=0.8)
    transitions = [
        {"vector_reward": np.asarray([1., 0.]), "old_values": np.asarray([0., 0.]),
         "next_values": np.asarray([0., 0.]), "terminated": False, "truncated": False},
        {"vector_reward": np.asarray([0., 0.]), "old_values": np.asarray([0., 0.]),
         "next_values": np.asarray([10., 0.]), "terminated": False, "truncated": True},
    ]
    advantages, _ = _gae_vector(transitions, train, torch.device("cpu"))
    assert torch.allclose(advantages[:, 0], torch.tensor([7.48, 9.0]), atol=1e-6, rtol=0)

    # A batch spanning two episodes keeps each row's own preference.
    rows = [{
        "obs": np.zeros(policy.base.base_obs_dim, dtype=np.float32),
        "policy_hidden_before": np.zeros(128, dtype=np.float32),
        "mask": np.ones(25, dtype=np.bool_),
        "preference": np.asarray(pref, dtype=np.float32),
        "candidate_features": np.zeros((25, 17), dtype=np.float32),
    } for pref in ((0.9, 0.1), (0.2, 0.8))]
    encoded = _batch_encode(policy, rows, torch.device("cpu"))
    assert torch.equal(encoded[3], torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
    next_episode_preference = _preference_after_episode(
        preference, rng, torch.device("cpu"), enabled=True,
    )
    assert not torch.equal(preference, next_episode_preference)


def test_transaction_crash_boundaries_keep_last_commit_and_preserve_tail(tmp_path):
    for stage in ("before_update", "after_update_before_checkpoint", "checkpoint_before_commit"):
        run_dir = tmp_path / stage
        run_dir.mkdir()
        ledger = _TrainingLedger(run_dir / "training-ledger.jsonl", run_id=f"run-{stage}")
        ledger.append({"step": 0, "run_id": f"run-{stage}"})
        first_range = ledger.flush_rollout()
        first_counters = {"environment_steps": 1, "policy_optimizer_steps": 1,
                          "world_optimizer_steps": 0}
        first_id = "txn-0001"
        first_checkpoint = run_dir / "transactions" / f"{first_id}.pt"
        _atomic_torch_save(first_checkpoint, {
            "run_id": f"run-{stage}", "counters": first_counters,
            "persistence": {"transaction_id": first_id},
        })
        ledger.publish_transaction(checkpoint_path=first_checkpoint, counters=first_counters,
                                   transaction_id=first_id, ledger_range=first_range)
        old_pointer = (run_dir / "ledger-commit.json").read_bytes()

        # Simulate the next rollout tail and crash at each requested boundary.
        ledger.append({"step": 1, "run_id": f"run-{stage}"})
        ledger.flush_rollout()
        if stage == "after_update_before_checkpoint":
            (run_dir / "uncommitted-update-marker.json").write_text('{"live_update":true}')
        elif stage == "checkpoint_before_commit":
            orphan = run_dir / "transactions" / "txn-0002.pt"
            _atomic_torch_save(orphan, {
                "run_id": f"run-{stage}",
                "counters": {"environment_steps": 2, "policy_optimizer_steps": 2,
                             "world_optimizer_steps": 0},
                "persistence": {"transaction_id": "txn-0002"},
            })
        verified = verify_commit_pointer(run_dir)
        assert verified["last_committed_step"] == 0
        assert verified["last_committed_policy_update"] == 1
        assert verified["ledger_tail_bytes_uncommitted"] > 0
        assert (run_dir / "ledger-commit.json").read_bytes() == old_pointer
        assert (run_dir / "training-ledger.jsonl").read_bytes().endswith(b'{"run_id": "run-' + stage.encode() + b'", "step": 1}\n')
        ledger.close()


def test_atomic_torch_save_fsync_roundtrip(tmp_path):
    path = tmp_path / "atomic.pt"
    payload = {"tensor": torch.arange(5), "tag": "windows-fsync"}
    _atomic_torch_save(path, payload)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(restored["tensor"], payload["tensor"])
    assert restored["tag"] == payload["tag"]
