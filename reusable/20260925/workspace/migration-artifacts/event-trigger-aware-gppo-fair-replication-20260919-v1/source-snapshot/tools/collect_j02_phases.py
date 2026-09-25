"""Collect development three-phase trajectories and replay every legal branch."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from gppo_world.phase_data import (FORMAT, canonical, digest, native_fingerprint,
    observed_environment, rng_fingerprints, validate_record)
from gppo_world.contracts import snapshot_from_gppo
from gppo_world.recorder import graph_to_dict, evidence_to_dict
from tools.collect_t01_dataset import _make_tape, _safe_evidence, _costs
from tools.run_d02_adapter_diagnostics import write, sha, git


def capture(env, action, identity):
    from ppo_allocation.random_event.environment import ActionSubmission
    ctx = env.begin_decision()
    before = native_fingerprint(env)
    pre = {'graph': graph_to_dict(snapshot_from_gppo(ctx.graph)), 'time': float(env.current_time),
        'action_version': int(ctx.action_version), 'decision_duration': float(env.decision_duration),
        'evidence': [evidence_to_dict(e) for e in _safe_evidence(env, env.current_time)]}
    if before != native_fingerprint(env):
        raise ValueError('Pre observation mutation')
    pre_rng = rng_fingerprints(env)
    result = env.submit_action(ActionSubmission.from_decision(action, ctx))
    graph, reward, terminated, truncated, info = result
    if info.get('invalid_action') or info.get('stale_decision') or info['repaired_action'] != action:
        raise ValueError('Rejected or repaired branch')
    row = {'format': FORMAT, **identity, 'pre': pre, 'action': action,
        'commit': deepcopy(env._phase_commit),
        'next': {'graph': graph_to_dict(snapshot_from_gppo(graph)), 'time': float(env.current_time)},
        'reward': float(reward), 'costs': _costs(info), 'terminated': bool(terminated), 'truncated': bool(truncated),
        'execution': {'accepted': True, 'executed_action': action},
        'audit': {'before': before, 'after': native_fingerprint(env), 'pre_rng': pre_rng,
                  'next_rng': rng_fingerprints(env), 'native_return': digest(canonical(result))}}
    validate_record(row)
    if row['commit']['rng'] != pre_rng:
        raise ValueError('Commit consumes randomness; action branches require revised coupling protocol')
    return row, result


def execute(args):
    protocol = json.loads((ROOT / 'nodes/J-02/data-contract.json').read_text(encoding='utf-8'))
    if git(ROOT, 'status', '--porcelain') or git(args.baseline, 'status', '--porcelain'):
        raise ValueError('Freeze clean source before collecting')
    if git(args.baseline, 'rev-parse', 'HEAD') != protocol['baseline_commit']:
        raise ValueError('Wrong baseline')
    sys.path.insert(0, str(args.baseline))
    from ppo_allocation.random_event.environment import RandomEventAllocationEnv, ActionSubmission
    from ppo_allocation.random_event.events import EventTape
    from ppo_allocation.random_event.baselines import GreedyCostPolicy, MaskedRandomPolicy
    from ppo_allocation.random_event.runtime_bridge import DetectorConfig
    from ppo_allocation.random_event.trainer import PPOTrainer
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observed = observed_environment(RandomEventAllocationEnv)
    started = time.monotonic()
    checkpoint = args.old_data / 'behavior.pt'
    old_manifest = json.loads((args.old_data / 'dataset-manifest.json').read_text(encoding='utf-8'))
    if sha(checkpoint) != old_manifest['gppo_behavior_checkpoint']['sha256']:
        raise ValueError('Old behavior checkpoint mismatch')
    trainer, _ = PPOTrainer.load(checkpoint, env=RandomEventAllocationEnv(), device='cpu')
    policy = trainer.model.eval()
    forbidden = set()
    refs = [args.old_data / 'dataset-manifest.json', ROOT/'nodes/T-05/evidence/server-test-bank-manifest.json',
            ROOT/'nodes/D-02/evidence/development-bank.json']
    def scan(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if 'seed' in k and isinstance(v, int): forbidden.add(v)
                scan(v)
        elif isinstance(x, list):
            for v in x: scan(v)
    for ref in refs: scan(json.loads(ref.read_text(encoding='utf-8')))
    write(args.output/'run-manifest.json', {'protocol': protocol, 'source_commit': git(ROOT,'rev-parse','HEAD'),
        'baseline_commit': protocol['baseline_commit'], 'behavior_checkpoint_sha256': sha(checkpoint),
        'references': [{'path':str(p),'sha256':sha(p)} for p in refs]})
    def fresh(tape, profile, observer=True):
        cls = observed if observer else RandomEventAllocationEnv
        env = cls(initial_seed=tape.initial_seed,event_seed=tape.event_seed,event_tape=tape,
            mode=tape.mode,events_per_episode=max(1,len(tape.events)),max_decisions=100,max_time=240.)
        env.reset(seed=tape.initial_seed)
        if profile == 'weak_comm':
            env.runtime_bridge.detector.config = DetectorConfig(loss_rate=.15,duplicate_rate=.20,
                false_positive_rate=0.,out_of_order_max_delay=0.)
        return env
    counts = Counter()
    manifest = []
    for split, master in protocol['development']['master_seeds'].items():
        rng = random.Random(master)
        with (args.output/f'{split}.jsonl').open('x',encoding='utf-8') as out, (args.output/f'{split}-branches.jsonl').open('x',encoding='utf-8') as bout:
            for profile in protocol['profiles']:
                for index in range(protocol['development']['tapes_per_profile'][split]):
                    initial,event = rng.getrandbits(31),rng.getrandbits(63)
                    if initial in forbidden or event in forbidden: raise ValueError('Seed overlap')
                    forbidden.update((initial,event))
                    tape = _make_tape(RandomEventAllocationEnv,EventTape,initial,event,profile)
                    tape_id=f'j02-{split}-{profile}-{index}-{digest(json.loads(tape.to_json()))[:12]}'
                    write(args.output/'tapes'/f'{tape_id}.json',json.loads(tape.to_json()))
                    manifest.append({'split':split,'profile':profile,'tape_id':tape_id,'initial_seed':initial,'event_seed':event})
                    for behavior in ('random_legal','greedy','gppo'):
                        env, plain = fresh(tape,profile),fresh(tape,profile,False)
                        sampler = MaskedRandomPolicy(seed=initial ^ 0xA5A5) if behavior=='random_legal' else GreedyCostPolicy()
                        actions=[]
                        for step in range(100):
                            if time.monotonic()-started > protocol['max_wall_seconds']: raise TimeoutError('Collection budget exceeded')
                            if native_fingerprint(env)!=native_fingerprint(plain): raise ValueError('Native/observed state mismatch')
                            graph=env.begin_decision().graph
                            action=int(policy.act(graph,deterministic=True)[0] if behavior=='gppo' else sampler.select_action(env,graph,deterministic=behavior!='random_legal'))
                            identity={'split':split,'tape_id':tape_id,'scenario_id':profile,'behavior':behavior,'episode_id':f'{tape_id}/{behavior}','step':step}
                            row,result=capture(env,action,identity)
                            ctx=plain.begin_decision()
                            native=plain.submit_action(ActionSubmission.from_decision(action,ctx))
                            if digest(canonical(native))!=row['audit']['native_return'] or native_fingerprint(plain)!=row['audit']['after']:
                                raise ValueError('Observer altered native transition')
                            out.write(json.dumps(row,sort_keys=True)+'\n')
                            counts['transitions']+=1
                            counts['commit_next_different']+=row['commit']['graph']!=row['next']['graph']
                            counts['history_four_available']+=step>=3
                            legal=graph.action_mask.nonzero().flatten().tolist()
                            counts['multi_action_origins']+=len(legal)>1
                            # Replay entire history for every branch, including the factual action.
                            if step < protocol['branch_origins_per_episode']:
                                for candidate in legal:
                                    branch=fresh(tape,profile)
                                    for previous in actions:
                                        ctx=branch.begin_decision()
                                        branch.submit_action(ActionSubmission.from_decision(previous,ctx))
                                    if native_fingerprint(branch)!=row['audit']['before']: raise ValueError('History replay mismatch')
                                    branch_row,_=capture(branch,int(candidate),identity)
                                    if candidate==action and branch_row!=row: raise ValueError('Factual branch mismatch')
                                    branch_row['factual_action']=action
                                    branch_row['history_actions']=actions.copy()
                                    # G_next is factual per-branch, not common-random-number counterfactual truth.
                                    branch_row['branch_scope']='commit_only; next observed but stochastic streams may diverge'
                                    bout.write(json.dumps(branch_row,sort_keys=True)+'\n')
                                    counts['branches']+=1
                                    counts['factual_replays']+=candidate==action
                                    branch.close()
                            actions.append(action)
                            if row['terminated'] or row['truncated']: break
                        counts['episodes']+=1
                        env.close();plain.close()
                    print(json.dumps({'split':split,'profile':profile,'tape':index,'counts':dict(counts)}),flush=True)
    write(args.output/'dataset-manifest.json',{'tapes':manifest,'counts':dict(counts),'protocol':protocol,
        'source_commit':git(ROOT,'rev-parse','HEAD'),'elapsed_seconds':time.monotonic()-started,
        'native_observer_mismatches':0,'factual_replay_mismatches':0,'commit_rng_changes':0,
        'next_branch_coupling_claim':False,'status':'collected_development_only'})
    write(args.output/'inventory.json',[{'path':p.relative_to(args.output).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(args.output.rglob('*')) if p.is_file()])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--old-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try: execute(args)
    except Exception:
        write(args.output/'failure.json',{'traceback':traceback.format_exc()});raise
