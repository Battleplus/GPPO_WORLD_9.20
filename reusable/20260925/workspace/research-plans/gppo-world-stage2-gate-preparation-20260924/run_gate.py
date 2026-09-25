"""Explicitly authorized technical-gate runner. Imports models only after gates.

This entry point intentionally cannot launch the formal 172800-step stage.
No approval file exists by default. The template is not an approval.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

from public_controller import ContractError, PublicMemory, PublicPlanner, public_copy

ROOT=Path(__file__).resolve().parent
PLAN=ROOT.parent/'gppo-world-decision-freeze-20260924'
LIMITS={'environment_steps':72,'policy_encode':36,'world_candidate_batch':36,'actor_readout':36}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def primitive(x):
    if hasattr(x,'tolist'): return primitive(x.tolist())
    if isinstance(x,dict): return {str(k):primitive(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [primitive(v) for v in x]
    if isinstance(x,(str,int,bool)) or x is None: return x
    if isinstance(x,float):
        if not math.isfinite(x): raise ContractError('Nonfinite audit value')
        return x
    # Never silently drop unknown outcome objects.
    raise ContractError(f'Unsupported audit type {type(x).__name__}')


def canonical(x):
    return json.dumps(primitive(x),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(x): return hashlib.sha256(canonical(x).encode('utf-8')).hexdigest()


def write(path,value):
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:
        f.write(canonical(value)+'\n'); f.flush(); os.fsync(f.fileno())


def append(path,value):
    with Path(path).open('a',encoding='utf-8',newline='\n') as f:
        f.write(canonical(value)+'\n'); f.flush(); os.fsync(f.fileno())


def authorize(path):
    """No mutation, native import, model load or scenario generation here."""
    if not path or not Path(path).is_file(): raise PermissionError('Explicit technical-gate authorization required')
    auth=json.loads(Path(path).read_text(encoding='utf-8'))
    binding=json.loads((ROOT/'binding.json').read_text(encoding='utf-8'))
    if (auth.get('approved') is not True or auth.get('stage')!='technical_gate'
        or auth.get('limits')!=LIMITS or auth.get('updates')!=0
        or auth.get('binding_sha256')!=sha(ROOT/'binding.json')
        or not auth.get('user_approval_reference') or not auth.get('user_approval_text')):
        raise PermissionError('Approval does not bind this stage, code and exact caps')
    for name,expected in binding['execution_files'].items():
        if sha(ROOT/name)!=expected: raise ContractError(f'Execution source changed: {name}')
    for name,expected in binding['plan_files'].items():
        if sha(PLAN/name)!=expected: raise ContractError(f'Frozen plan changed: {name}')
    if auth.get('output_directory')!=str(ROOT/'runs'/'technical-gate-v1'):
        raise PermissionError('Unbound output or alternative attempt directory')
    out=Path(auth['output_directory'])
    if out.exists(): raise FileExistsError('Attempt already exists; no retry or overwrite')
    if auth.get('formal_stage_authorized') is not False:
        raise PermissionError('This entry point cannot authorize the formal stage')
    return auth,binding,out


def checked_sources(binding):
    for row in binding['native_inputs']:
        if sha(row['path'])!=row['sha256']: raise ContractError(f'Native input changed: {row["path"]}')
    for relative,expected in binding['native_python_files'].items():
        if sha(Path(binding['native_root'])/relative)!=expected:
            raise ContractError(f'Native dependency changed before import: {relative}')
    if any(k=='gppo_world' or k.startswith('gppo_world.') for k in sys.modules):
        raise ContractError('Native package was imported before authorization/isolation')


def peak_rss():
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in
            ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    c=Counters(); c.cb=ctypes.sizeof(c)
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]; fn.restype=ctypes.c_int
    if not fn(ctypes.c_void_p(-1),ctypes.byref(c),ctypes.sizeof(c)):
        raise OSError('Cannot audit RSS')
    return c.peak


class DurableCalls:
    def __init__(self,budget,path):
        self.budget,self.path=budget,path
        self.audit_wall=0.; self.audit_cpu=0.
    def call(self,stage,fn,*args,**kwargs):
        full_wall=time.perf_counter(); full_cpu=time.process_time()
        token=self.budget.reserve(stage,1)
        inner_wall=inner_cpu=0.
        try:
            append(self.path,{'kind':'call_begin','stage':stage,'reservation':token})
            inner_start_wall=time.perf_counter();inner_start_cpu=time.process_time()
            value=fn(*args,**kwargs)
            inner_wall=time.perf_counter()-inner_start_wall;inner_cpu=time.process_time()-inner_start_cpu
            append(self.path,{'kind':'call_return','stage':stage,'reservation':token})
            self.budget.complete(token)
            return value
        except BaseException as exc:
            try: self.budget.unknown(token,f'{type(exc).__name__}: {exc}')
            except BaseException: pass
            raise
        finally:
            self.audit_wall+=max(0.,time.perf_counter()-full_wall-inner_wall)
            self.audit_cpu+=max(0.,time.process_time()-full_cpu-inner_cpu)


def f32(v): return struct.unpack('<f',struct.pack('<f',v))[0]


def verify_vector(info,previous_counts,previous_energy,native,config):
    reward,_,counts,energy=native(info,previous_counts,previous_energy,config)
    if len(reward)!=2: raise ContractError('Reward must contain exactly two components')
    independent=[f32((max(0,int(info['counts']['completed'])-previous_counts['completed'])
                      -max(0,int(info['counts']['expired'])-previous_counts['expired']))/6),
                 f32(-max(0.,previous_energy-sum(float(v) for v in info['energy'].values()))/36)]
    if any(not math.isfinite(float(a)) or abs(float(a)-b)>1e-6 for a,b in zip(reward,independent)):
        raise ContractError('Reward contract mismatch')
    return independent,counts,energy


def ending(done,info):
    if type(info.get('terminated')) is not bool or type(info.get('truncated')) is not bool:
        raise ContractError('Missing native termination labels')
    terminated,truncated=info['terminated'],info['truncated']
    if terminated and truncated or bool(done)!=(terminated or truncated):
        raise ContractError('Native termination flags inconsistent')
    return {'terminated':terminated,'truncated':truncated,
            'episode_end_reason':info.get('episode_end_reason')}


def run(auth,binding,out):
    checked_sources(binding)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); budget=None; summaries=[]; initials={}; sources={}
    try:
        write(out/'authorization.json',auth); write(out/'binding.json',binding)
        src=Path(binding['native_root']).resolve()
        sys.path.insert(0,str(src))
        # All imports below occur only after explicit authorization and identity checks.
        import torch
        if tuple(sys.version_info[:3])!=(3,14,4) or torch.__version__!='2.13.0+cpu':
            raise ContractError('Frozen Python/PyTorch runtime mismatch')
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        from gppo_world.m10_environment import M10Config,M10Environment,formal_three_condition_tape,scenario_to_dict
        from gppo_world.joint_gppo import JointGraphPreferencePolicy,ActionConditionedTemporalWorldModel,masked_normalized_preference
        from gppo_world.joint_training import _obs_tensor,_vector_reward
        from gppo_world.budget_executor import PersistentBudget
        for name,module in list(sys.modules.items()):
            if name=='gppo_world' or name.startswith('gppo_world.'):
                path=Path(module.__file__).resolve()
                if not path.is_relative_to(src): raise ContractError('Mixed native package import')
                relative=path.relative_to(src).as_posix()
                expected=binding['native_python_files'].get(relative)
                if expected is None or sha(path)!=expected: raise ContractError('Imported unpinned module')
                sources[name]={'path':str(path),'sha256':expected}
        config_payload=json.loads(Path(binding['config']).read_text(encoding='utf-8-sig'))
        config=M10Config(**config_payload['environment'])
        if (config.action_count!=25 or config.completion_notice_mode!='single_shot'
            or config.task_completion_mode!='arrival_to_region' or config.deadline_basis!='physical_arrival'
            or config.horizon!=18 or config.decision_interval!=1): raise ContractError('Wrong frozen configuration')
        # Generate only this explicitly authorized two-parent gate; never the formal/test matrix.
        scenarios=formal_three_condition_tape('train',count=2,base_seed=924100000,condition='W1',name='mixed')
        tapes=[scenario_to_dict(s) for s in scenarios]
        occupied=set(binding['historical_seed_ids'])
        if any(int(t['seed']) in occupied for t in tapes): raise ContractError('Historical seed collision')
        write(out/'scenario-manifest.json',{'tapes':tapes,'hashes':[digest(t) for t in tapes],
                                           'role':'technical_gate_not_formal_evaluation'})
        device=torch.device('cpu'); load_start=time.perf_counter()
        payload=torch.load(binding['checkpoint'],map_location=device,weights_only=False)
        if payload.get('group')!='P_train': raise ContractError('Wrong checkpoint group')
        policy=JointGraphPreferencePolicy(config,history=True).to(device)
        world=ActionConditionedTemporalWorldModel().to(device)
        policy.load_state_dict(payload['policy_state_dict'],strict=True)
        world.load_state_dict(payload['world_state_dict'],strict=True)
        policy.eval();world.eval()
        for m in [policy,world]:
            for p in m.parameters(): p.requires_grad_(False)
        load_seconds=time.perf_counter()-load_start
        pref=masked_normalized_preference((.8,.2),device=device)
        def state_hash():
            h=hashlib.sha256()
            for prefix,m in [('policy',policy),('world',world)]:
                for k,v in sorted(m.state_dict().items()):
                    h.update((prefix+k).encode());h.update(v.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        model_before=state_hash()
        budget=PersistentBudget(out/'budget.sqlite3',limits=LIMITS,run_limits=LIMITS,
            attempt_id='native-public-history-gate-v1-attempt-0001',run_id='native-public-history-gate-v1')
        calls=DurableCalls(budget,out/'calls.jsonl')
        write(out/'runtime-start.json',{'module_sources':sources,'model_state_hash':model_before,
            'checkpoint_hash':sha(binding['checkpoint']),'model_load_seconds':load_seconds,
            'initial_budget':budget.snapshot(),'torch':torch.__version__,'python':sys.version})
        for parent_index,scenario in enumerate(scenarios):
            parent=f'gate-{parent_index:03d}'
            for arm in (['M','R'] if parent_index%2==0 else ['R','M']):
                key=f'frozen-native-public-history-allocation-v1|{parent}|repeat-0'
                env=M10Environment(config,copy.deepcopy(scenario),exogenous_key=key)
                obs=env.reset()
                if env.view._completed_tasks: raise ContractError('Unexpected bounded completion marker')
                # Evaluation-only access: initial energy/counts are never supplied to the controller.
                previous_counts={'completed':0,'expired':0}
                previous_energy=sum(float(r.energy) for r in env.clock.resources.values())
                first=public_copy(obs); initial_hash=digest(first)
                if parent in initials and initials[parent]!=initial_hash: raise ContractError('Paired reset public state mismatch')
                initials[parent]=initial_hash
                memory=PublicMemory();planner=PublicPlanner();ph=wh=None;utility=0.;timings=[];done=False
                episode_start=time.perf_counter()
                for step in range(18):
                    if time.perf_counter()-started>3600 or peak_rss()>4*1024**3:
                        raise ContractError('Technical gate resource cap')
                    wall=time.perf_counter();cpu=time.process_time()
                    audit_wall_before=calls.audit_wall; audit_cpu_before=calls.audit_cpu
                    public=public_copy(obs);memory.observe(public);candidates=memory.candidates(public)
                    by_action=None;diagnostic={}
                    if arm=='M':
                        with torch.no_grad():
                            features,pair,next_ph=calls.call('policy_encode',policy.encode,_obs_tensor(public,device),ph)
                            feature,by_action=calls.call('world_candidate_batch',world.predict_all_candidates,
                                features,wh,public,use_events=True)
                            evaluation=calls.call('actor_readout',policy.evaluate_encoded,features,pair,pref,feature,
                                torch.as_tensor(public['mask'],dtype=torch.bool,device=device)[None,:])
                            probs=evaluation['probabilities'][0].detach().cpu().tolist()
                            if not all(math.isfinite(x) for x in probs): raise ContractError('Nonfinite policy')
                            action=max(candidates,key=lambda a:(probs[a],-a))
                            if action not in by_action: raise ContractError('Selected world hidden absent')
                            ph=next_ph.detach();wh=by_action[action]['hidden'].detach()
                            diagnostic={'probabilities':probs}
                    else:
                        action,diagnostic=planner.choose(public,memory)
                    if action not in candidates: raise ContractError('Illegal final selection')
                    memory.submitted(public,action)
                    raw_cpu=time.process_time()-cpu;raw_wall=time.perf_counter()-wall
                    audit_cpu=calls.audit_cpu-audit_cpu_before;audit_wall=calls.audit_wall-audit_wall_before
                    timing={'controller_cpu_seconds':max(0.,raw_cpu-audit_cpu),
                            'controller_wall_seconds':max(0.,raw_wall-audit_wall),
                            'raw_controller_cpu_with_audit':raw_cpu,'raw_controller_wall_with_audit':raw_wall,
                            'excluded_audit_cpu_seconds':audit_cpu,'excluded_audit_wall_seconds':audit_wall}
                    timings.append(timing)
                    if arm=='M':
                        # This audit-only hash is outside the controller timer.
                        diagnostic['selected_world_hidden_sha256']=hashlib.sha256(wh.cpu().numpy().tobytes()).hexdigest()
                    # env.step reservation stays pending until labels and durable step evidence succeed.
                    token=budget.reserve('environment_steps',1)
                    try:
                        append(out/'decisions.jsonl',{'parent':parent,'repeat':0,'arm':arm,'step':step,
                            'observation':public,'candidates':candidates,'action':action,'submit_command':True,
                            'diagnostic':diagnostic,'timing':timing,'reservation':token})
                        obs,scalar,done,info=env.step(action,submit_command=True)
                        native_end=ending(done,info)
                        vec,counts,energy=verify_vector(info,previous_counts,previous_energy,_vector_reward,config)
                        utility += .99**step*(.4*vec[0]+.2*vec[1])
                        if env.view._completed_tasks: raise ContractError('single_shot marker path changed')
                        append(out/'steps.jsonl',{'parent':parent,'repeat':0,'arm':arm,'step':step,
                            'vector_reward':vec,'scalar_reward':float(scalar),'info':info,'done':bool(done),'reservation':token})
                        budget.complete(token)
                        previous_counts,previous_energy=counts,energy
                    except BaseException as exc:
                        try: budget.unknown(token,f'{type(exc).__name__}: {exc}')
                        except BaseException: pass
                        raise
                    if done: break
                if not done: raise ContractError('Non-native truncation at cap')
                summary={'parent':parent,'repeat':0,'arm':arm,'environment_steps':step+1,'utility':utility,
                    'initial_public_sha256':initial_hash,'task_counts':previous_counts,'final_energy':previous_energy,
                    'energy_used':36-previous_energy,'last_info':info,'timings':timings,
                    'wall_seconds':time.perf_counter()-episode_start,**native_end}
                summaries.append(summary);append(out/'episodes.jsonl',summary)
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>1024**3:
                    raise ContractError('Technical gate disk cap')
        final=budget.snapshot()
        if any(v['unknown'] or v['pending'] for v in final['stages'].values()): raise ContractError('Unresolved budget')
        if state_hash()!=model_before: raise ContractError('Frozen weights changed')
        for row in binding['native_inputs']:
            if sha(row['path'])!=row['sha256']: raise ContractError('Native source/checkpoint changed during run')
        write(out/'run-status.json',{'status':'technical_gate_passed','episodes':len(summaries),
            'budget':final,'updates':0,'native_model_state_unchanged':True,'peak_rss_bytes':peak_rss(),
            'wall_seconds':time.perf_counter()-started,'formal_stage_authorized':False,
            'interpretation':'Interface gate only; no algorithm benefit claim or automatic formal run.'})
    except BaseException as exc:
        failure={'status':'technical_stop','error':f'{type(exc).__name__}: {exc}',
                 'completed_episodes':len(summaries),'formal_stage_authorized':False,'automatic_retry':False}
        if budget is not None:
            try: failure['budget']=budget.snapshot()
            except BaseException as e: failure['budget_error']=str(e)
        write(out/'run-status.json',failure)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args(argv)
    auth,binding,out=authorize(args.authorization)
    run(auth,binding,out)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(json.dumps({'status':'stopped','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False))
        sys.exit(2)
