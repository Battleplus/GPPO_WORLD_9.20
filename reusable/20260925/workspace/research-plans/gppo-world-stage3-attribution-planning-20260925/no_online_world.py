"""Proposed inference intervention; no imports, loads or execution on import.

Retains the trained public-history encoder and preference actor. Removes only
the candidate_actor contribution and all online world computation. This is
NOT a policy trained without a world model.
"""

def actor_without_candidate(policy, features, pair_messages, preference, mask):
    import torch
    from gppo_world.joint_gppo import _base_logits
    from gppo_world.m10_training import masked_distribution
    if policy.training or torch.is_grad_enabled():
        raise RuntimeError('Frozen eval/no_grad required')
    if preference.ndim == 1:
        preference = preference[None, :].expand(features.shape[0], -1)
    logits = _base_logits(policy.base, features, pair_messages)
    logits = logits + policy.preference_actor(torch.cat((features, preference), dim=-1))
    distribution = masked_distribution(logits, mask)
    return {'logits': logits, 'probabilities': distribution.probs}


def native_reference_without_candidate(policy, features, pair_messages, preference, mask):
    """Independent native evaluation with the candidate head output neutralized.

    Used only in authorized A validation, never advertised as a natural
    no-world training distribution. Still pays the native unused computation.
    """
    import torch
    if policy.training or torch.is_grad_enabled():
        raise RuntimeError('Frozen eval/no_grad required')
    handle = policy.candidate_actor.register_forward_hook(
        lambda module, args, output: torch.zeros_like(output))
    try:
        zeros = features.new_zeros((features.shape[0], 25, 17))
        return policy.evaluate_encoded(features, pair_messages, preference, zeros, mask)
    finally:
        handle.remove()
