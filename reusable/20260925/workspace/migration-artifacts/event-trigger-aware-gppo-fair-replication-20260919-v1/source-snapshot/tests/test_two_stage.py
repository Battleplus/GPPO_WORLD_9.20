from copy import deepcopy
import torch
from gppo_world.jepa_two_stage import NodeEncoder,ActionDynamics,MaskedPretrainer,masked_graph
from gppo_world.jepa import batch_graphs
from test_contracts import make_graph
from tools.run_j02_development import branch_regret,fit_readout,readout,macro
from gppo_world.data import STATE_DIM


def test_mask_clears_duplicate_relations_and_preserves_source():
    g=batch_graphs([make_graph(),make_graph()]);old=deepcopy(g)
    for x in g['edge_attr'].values():x.fill_(1)
    masked,mask=masked_graph(g,torch.Generator().manual_seed(12))
    assert mask.sum(1).min()>=3
    assert all(torch.count_nonzero(x)==0 for x in masked['edge_attr'].values())
    assert all(torch.all(x==1) for x in g['edge_attr'].values())
    assert all(torch.equal(g['nodes'][k],v) for k,v in old['nodes'].items())


def arguments():
    torch.manual_seed(8)
    return dict(tokens=torch.randn(2,4,11,64),valid=torch.ones(2,4,dtype=torch.bool),
        past_actions=torch.tensor([[1,2,3,-1],[2,1,0,-1]]),past_gaps=torch.zeros(2,4),
        candidate_edges=torch.tensor([[(u,r) for u in range(4) for r in range(4)]]*2),
        edge_attributes=torch.randn(2,16,5),action=torch.tensor([1,2]),evidence=torch.zeros(2,14))


def test_no_action_ablation_does_not_read_any_candidate_information():
    model=ActionDynamics(with_action=False);a=arguments();b=deepcopy(a)
    b['action']=torch.tensor([16,8]);b['candidate_edges'].fill_(999);b['edge_attributes'].normal_()
    assert torch.equal(model(**a),model(**b))


def test_endpoint_mapping_noop_and_padding():
    model=ActionDynamics();a=arguments();a['valid'][:,:2]=False
    b=deepcopy(a);b['tokens'][:,:2].normal_(100,10);b['past_actions'][:,:2]=7;b['past_gaps'][:,:2]=999
    assert torch.equal(model(**a),model(**b))
    b=deepcopy(a);b['action']=torch.tensor([16,16]);c=deepcopy(b);c['edge_attributes'].fill_(999)
    assert torch.equal(model(**b),model(**c))
    b=deepcopy(a);b['action']=torch.tensor([8,15]);assert not torch.equal(model(**a),model(**b))


def test_pretraining_target_no_grad_and_graph_shapes():
    model=MaskedPretrainer();g=batch_graphs([make_graph(),make_graph()])
    loss=model.loss(g,torch.Generator().manual_seed(7));loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder(g).shape==(2,11,64)
    assert all(p.grad is None for p in model.target.parameters())


def test_cuda_indices_share_device_when_available():
    if not torch.cuda.is_available():
        return
    device=torch.device('cuda:0')
    g={part:{key:value.to(device) for key,value in values.items()} for part,values in batch_graphs([make_graph(),make_graph()]).items()}
    pre=MaskedPretrainer().to(device)
    assert torch.isfinite(pre.loss(g,torch.Generator().manual_seed(7)))
    model=ActionDynamics().to(device)
    values={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in arguments().items()}
    assert model(**values).device==device


def test_branch_regret_uses_true_same_origin_rewards_and_tie_rule():
    rows=[{'episode_id':'a','tape_id':'t','scenario_id':'s','step':0,'action':a,'reward':r} for a,r in [(1,2.),(3,5.)]]
    pred=torch.zeros(2,STATE_DIM+8)
    result=branch_regret(pred,rows,1e-6)
    assert result['immediate_reward_regret']==3.
    pred[1,STATE_DIM]=1
    assert branch_regret(pred,rows,1e-6)['immediate_reward_regret']==0


def test_probe_normalization_is_train_only():
    torch.manual_seed(7);x=torch.randn(80,4);y=x@torch.randn(4,3)
    p=fit_readout(x,y,y.std(0).clamp_min(.05))
    assert torch.equal(p['xmean'],x.double().mean(0))
    assert ((readout(p,x)-y)**2).mean()<.01
