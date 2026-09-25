from copy import deepcopy
import os
from pathlib import Path
import sys
import pytest
from gppo_world.phase_data import model_input, validate_record, native_fingerprint, observed_environment, digest, canonical
from tools.collect_j02_phases import capture
from gppo_world.phase_data import history_inputs


@pytest.fixture
def baseline():
    path=Path(os.environ.get('GPPO_BASELINE_ROOT',str(Path(__file__).resolve().parents[2]/'GPPO-8.29-baseline')))
    if not path.is_dir(): pytest.skip('Pinned GPPO baseline not available')
    sys.path.insert(0,str(path))
    from ppo_allocation.random_event.environment import RandomEventAllocationEnv,ActionSubmission
    return RandomEventAllocationEnv,ActionSubmission


@pytest.mark.parametrize('mode',['single','sequential','overlap','burst'])
def test_native_full_state_and_factual_replay(baseline,mode):
    Base,Submission=baseline
    obs=observed_environment(Base)
    a=obs(initial_seed=8171,event_seed=982181,mode=mode)
    b=Base(initial_seed=8171,event_seed=982181,mode=mode)
    c=obs(initial_seed=8171,event_seed=982181,mode=mode)
    for env in (a,b,c):env.reset()
    for step in range(10):
        assert native_fingerprint(a)==native_fingerprint(b)==native_fingerprint(c)
        ctx=a.begin_decision();action=int(ctx.graph.action_mask.nonzero()[0])
        row,_=capture(a,action,{'step':step})
        replay,_=capture(c,action,{'step':step})
        result=b.submit_action(Submission.from_decision(action,b.begin_decision()))
        assert row==replay
        assert row['audit']['native_return']==digest(canonical(result))
        assert native_fingerprint(a)==native_fingerprint(b)
        assert validate_record(row)
        if row['terminated'] or row['truncated']:break


def test_future_excluded_and_future_evidence_rejected(baseline):
    Base,_=baseline;env=observed_environment(Base)(initial_seed=8171,event_seed=982181);env.reset()
    action=int(env.begin_decision().graph.action_mask.nonzero()[0])
    row,_=capture(env,action,{})
    original=model_input(row)
    changed=deepcopy(row)
    for key in ['commit','next','audit','reward','costs']:changed[key]={'future_truth':'poison'}
    assert model_input(changed)==original
    row['pre']['evidence'].append({'received_at':row['pre']['time']+1})
    with pytest.raises(ValueError,match='Future evidence'):model_input(row)


def test_fingerprint_catches_hidden_rng_and_queue_changes(baseline):
    Base,_=baseline;env=Base();env.reset();old=native_fingerprint(env)
    env.runtime_bridge.detector.rng.random()
    assert native_fingerprint(env)!=old
    old=native_fingerprint(env);env.next_event_index+=1
    assert native_fingerprint(env)!=old


def test_history_padding_and_current_action_separation(baseline):
    Base,_=baseline;env=observed_environment(Base)(initial_seed=8171,event_seed=982181);env.reset()
    action=int(env.begin_decision().graph.action_mask.nonzero()[0])
    row,_=capture(env,action,{'episode_id':'one','step':0})
    second=deepcopy(row);second['step']=1
    other=deepcopy(row);other['episode_id']='two'
    values=history_inputs([row,second,other])
    assert values[0]['valid']==[False,False,False,True]
    assert values[1]['valid']==[False,False,True,True]
    assert values[1]['history'][-2]['past_executed_action']==action
    assert values[1]['history'][-1]['past_executed_action'] is None
    assert values[2]['valid']==values[0]['valid']
    with pytest.raises(ValueError,match='contiguous'):history_inputs([second])
