"""Candidate inference path only. Does not change or load any model weights.

Torch is imported only inside calls. This module is not integrated into the
frozen native source and has not yet passed numerical or runtime validation.
"""

def world_for_decision(world,history,hidden,observation,*,use_events=True):
    import torch
    if world.training or torch.is_grad_enabled():
        raise RuntimeError('Candidate is inference-only: eval and no_grad required')
    if tuple(history.shape)!=(1,128) or (hidden is not None and tuple(hidden.shape)!=(1,128)):
        raise ValueError('Frozen single-observation hidden contract required')
    device=history.device
    mask=torch.as_tensor(observation['mask'],dtype=torch.bool,device=device).reshape(-1)
    if tuple(mask.shape)!=(25,):raise ValueError('Expected 25 actions')
    relation_rows=torch.as_tensor(observation['graph']['relations'],dtype=torch.float32,device=device).reshape(24,4)
    actions=torch.arange(25,dtype=torch.long,device=device)
    relations=torch.cat((relation_rows,torch.zeros((1,4),dtype=torch.float32,device=device)),dim=0)
    histories=history.expand(25,-1)
    states=histories.new_zeros((25,128)) if hidden is None else hidden.expand(25,-1)
    # Same native modules, argument order, concatenation and recurrent cell.
    encoded_action=world.action_embedding(actions.long())
    encoded_relation=world.relation_encoder(relations)
    latent=world.temporal(torch.cat((histories,encoded_action,encoded_relation),dim=-1),states)
    reward=world.vector_reward(latent)
    consequences=world.task_consequence(latent)
    event_logits=world.event_head(latent)
    policy_feature=world.policy_feature(latent)
    features=torch.cat((policy_feature,reward,consequences,torch.sigmoid(event_logits)),dim=-1)
    if not use_events:features=torch.cat((features[:,:12],torch.zeros_like(features[:,12:])),dim=-1)
    features=torch.where(mask[:,None],features,torch.zeros_like(features))
    # Keep every candidate's hidden, including masked actions, as in native.
    return features[None,:,:],{i:{'hidden':latent[i:i+1]} for i in range(25)}

def actor_for_decision(policy,features,pair_messages,preference,candidate_features,mask):
    import torch
    from gppo_world.joint_gppo import _base_logits
    from gppo_world.m10_training import masked_distribution
    if policy.training or torch.is_grad_enabled():
        raise RuntimeError('Candidate is inference-only: eval and no_grad required')
    if preference.ndim==1:preference=preference[None,:].expand(features.shape[0],-1)
    if candidate_features.shape!=(features.shape[0],25,17):raise ValueError('Expected [B,25,17]')
    logits=_base_logits(policy.base,features,pair_messages)
    logits=logits+policy.preference_actor(torch.cat((features,preference),dim=-1))
    logits=logits+policy.candidate_actor(candidate_features).squeeze(-1)
    distribution=masked_distribution(logits,mask)
    return {'logits':logits,'probabilities':distribution.probs}
