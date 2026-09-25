"""Owned worker: all model/environment entry points require a bound permit."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from runtime_support import *
from cost_replay import groups,replay_batch
from decision_only_inference import world_for_decision,actor_for_decision

class Worker:
    def __init__(self,permit,binding,out):
        self.permit=permit;self.binding=binding;self.out=out;self.kind=permit['kind'];self.stage=permit['stage']
        self.counts={k:0 for k in ['environment_steps','environment_resets','model_loads','policy_encode','world_candidate_batch','actor_readout','rule_decisions']}
        native_package(binding)
        from gppo_world.m10_environment import M10Config
        self.config=M10Config(**json.loads(Path(binding['config']).read_text(encoding='utf-8-sig'))['environment'])
        if (self.config.action_count!=25 or self.config.horizon!=18 or self.config.decision_interval!=1
            or self.config.completion_notice_mode!='single_shot' or self.config.task_completion_mode!='arrival_to_region'
            or self.config.deadline_basis!='physical_arrival'):raise RuntimeError('Native configuration mismatch')
        self.reward=load_reward_function(binding);self.planner=PublicPlanner()
        if self.kind!='rule':
            import torch
            self.torch=torch
            if torch.__version__!='2.13.0+cpu' or tuple(sys.version_info[:3])!=(3,14,4):raise RuntimeError('Runtime version mismatch')
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            from gppo_world.joint_gppo import JointGraphPreferencePolicy,ActionConditionedTemporalWorldModel,masked_normalized_preference
            self.counts['model_loads']+=1
            payload=torch.load(binding['checkpoint'],map_location='cpu',weights_only=False)
            if payload.get('group')!='P_train':raise RuntimeError('Checkpoint group mismatch')
            self.policy=JointGraphPreferencePolicy(self.config,history=True);self.world=ActionConditionedTemporalWorldModel()
            self.policy.load_state_dict(payload['policy_state_dict'],strict=True);self.world.load_state_dict(payload['world_state_dict'],strict=True)
            for m in [self.policy,self.world]:
                m.eval()
                for p in m.parameters():p.requires_grad_(False)
            self.pref=masked_normalized_preference((.8,.2),device=torch.device('cpu'))
            self.before=self.weight_hash()
        else:
            if 'torch' in sys.modules:raise RuntimeError('Rule worker imported Torch')
        self.sources=verify_imports(binding)

    def weight_hash(self):
        h=hashlib.sha256()
        for tag,m in [('policy',self.policy),('world',self.world)]:
            for k,v in sorted(m.state_dict().items()):h.update((tag+k).encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def choose(self,public,memory,state,candidates):
        if self.kind=='rule':
            self.counts['rule_decisions']+=1
            action,diagnostic=self.planner.choose(public,memory)
            return action,None,diagnostic
        t=self.torch;ph,wh=(None,None) if state is None else state
        with t.no_grad():
            self.counts['policy_encode']+=1
            f,pair,nextph=self.policy.encode(t.as_tensor(public['flat'],dtype=t.float32).reshape(1,-1),ph)
            self.counts['world_candidate_batch']+=1
            cf,by=(self.world.predict_all_candidates(f,wh,public,use_events=True) if self.kind=='reference'
                else world_for_decision(self.world,f,wh,public,use_events=True))
            self.counts['actor_readout']+=1;mask=t.as_tensor(public['mask'],dtype=t.bool)[None,:]
            ev=(self.policy.evaluate_encoded(f,pair,self.pref,cf,mask) if self.kind=='reference'
                else actor_for_decision(self.policy,f,pair,self.pref,cf,mask))
            probs=ev['probabilities'][0].detach().cpu().tolist()
            if not all(math.isfinite(x) for x in probs):raise RuntimeError('Nonfinite probabilities')
            action=max(candidates,key=lambda a:(probs[a],-a));selected=by[action]['hidden'].detach()
            bundle={'selected':selected,'probabilities':ev['probabilities']}
            if self.stage=='A':
                bundle.update(features=f,pair=pair,policy_hidden=nextph,candidate_features=cf,logits=ev['logits'],
                    **{f'world_hidden_{a}':by[a]['hidden'] for a in range(25)})
            return action,(nextph.detach(),selected),bundle

    def tensor_record(self,bundle):
        result={}
        for k,v in bundle.items():
            if not self.torch.isfinite(v).all():raise RuntimeError('Nonfinite equivalence tensor')
            result[k]={'shape':list(v.shape),'dtype':str(v.dtype),'bytes_hex':v.detach().cpu().contiguous().numpy().tobytes().hex()}
        return result

    def replay(self):
        rows=[json.loads(x) for x in (PREP/'runs/technical-gate-v1/decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        traces=groups(rows,'M');expected={(r['parent'],r['step']):r for e in traces for r in e}
        reference=read(self.out/'A/reference-tensors.json') if self.kind=='candidate' else None
        batches=[]
        for index in range(21):
            stats,records=replay_batch(traces,self.choose)
            actual=[]
            for parent,step,action,bundle,latency in records:
                row=expected[(parent,step)]
                if action!=row['action']:raise RuntimeError('Replay action mismatch')
                data=self.tensor_record(bundle)
                if hashlib.sha256(bytes.fromhex(data['selected']['bytes_hex'])).hexdigest()!=row['diagnostic']['selected_world_hidden_sha256']:
                    raise RuntimeError('Saved hidden mismatch')
                actual.append({'parent':parent,'step':step,'action':action,'tensors':data})
            if reference is None:
                reference=actual;write(self.out/'A/reference-tensors.json',reference)
            if actual!=reference:raise RuntimeError('Bitwise tensor equivalence failed')
            stats.update(index=index,cold=index==0,decision_wall_seconds=[r[4] for r in records]);batches.append(stats)
        if self.weight_hash()!=self.before:raise RuntimeError('Weights changed')
        result={'kind':self.kind,'batches':batches,'counts':self.counts,'equivalence':True,'weights_unchanged':True,'peak_rss':peak_rss()}
        write(self.out/('A/'+self.kind+'-result.json'),result)
        return result

    def generate(self,args):
        from gppo_world.m10_environment import formal_three_condition_tape,scenario_to_dict
        tapes=[scenario_to_dict(s) for s in formal_three_condition_tape('train',count=args['count'],base_seed=args['seed'],condition='W1',name='mixed')]
        return {'tapes':tapes,'hashes':[digest(t) for t in tapes]}

    def reset(self,args):
        from gppo_world.m10_environment import M10Environment,scenario_from_dict
        self.counts['environment_resets']+=1
        self.env=M10Environment(self.config,scenario_from_dict(args['tape']),exogenous_key=args['key'])
        self.obs=self.env.reset();self.memory=PublicMemory();self.state=None;self.step_index=0;self.done=False
        self.prevcounts={'completed':0,'expired':0};self.energy=sum(float(r.energy) for r in self.env.clock.resources.values())
        if self.env.view._completed_tasks:raise RuntimeError('Unexpected bounded-completion marker')
        self.utility=0.;public=public_copy(self.obs)
        return {'initial_public':public,'initial_hash':digest(public),'counts':dict(self.counts)}

    def one_step(self):
        if self.done or self.step_index>=18:raise RuntimeError('Invalid extra environment step')
        def control():
            public=public_copy(self.obs);self.memory.observe(public);candidates=self.memory.candidates(public)
            action,state,diagnostic=self.choose(public,self.memory,self.state,candidates)
            self.state=state;self.memory.submitted(public,action)
            return public,candidates,action,diagnostic
        (public,candidates,action,diag),controller=timed(control)
        if self.kind!='rule':
            diag={'probabilities':diag['probabilities'].detach().cpu().tolist(),
                'selected_world_hidden_sha256':hashlib.sha256(diag['selected'].cpu().numpy().tobytes()).hexdigest()}
        self.counts['environment_steps']+=1
        (self.obs,scalar,self.done,info),environment=timed(self.env.step,action,submit_command=True)
        native_end=ending(self.done,info)
        (vector,counts,energy),labels=timed(verify_vector,info,self.prevcounts,self.energy,self.reward,self.config)
        self.prevcounts,self.energy=counts,energy;self.utility+=.99**self.step_index*(.4*vector[0]+.2*vector[1])
        if self.env.view._completed_tasks:raise RuntimeError('single_shot marker changed')
        row={'step':self.step_index,'public':public,'candidates':candidates,'action':action,'submit_command':True,
            'diagnostic':diag,'controller':controller,'environment':environment,'labels':labels,
            'vector_reward':vector,'scalar_reward':float(scalar),'done':bool(self.done),'info':info,'counts':dict(self.counts)}
        self.step_index+=1
        if self.step_index==18 and not self.done:raise RuntimeError('Non-native truncation')
        if self.done:
            records=info['completion_records']
            row['episode']={'steps':self.step_index,'utility':self.utility,'energy_used':36-self.energy,'task_counts':counts,
                'physical_on_time':sum(v['physical_arrival_before_deadline'] is True for v in records.values())/6,
                'host_on_time_observed':sum(v['host_confirmation_before_deadline'] is True for v in records.values())/6,
                'physical_completion_without_host_observed':sum(v['host_confirmation_time'] is None for v in records.values()),**native_end}
        return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--authorization',required=True,type=Path);p.add_argument('--name',required=True);args=p.parse_args()
    a,b,r=authorize(args.authorization);out=ROOT/'runs/consolidated-v1';permit=read(out/(args.name+'-permit.json'))
    write(out/(args.name+'-claim.json'),{'pid':os.getpid(),'permit':permit})
    worker=None
    try:
        worker=Worker(permit,read(PREP/'binding.json'),out)
        print(canonical({'ok':True,'ready':True,'counts':worker.counts,'sources':worker.sources,'initializer':'export-only shim','torch_loaded':'torch' in sys.modules,'startup_cpu':time.process_time()}),flush=True)
        for line in sys.stdin:
            request=json.loads(line);cmd=request['command']
            if cmd=='replay':value=worker.replay()
            elif cmd=='generate':value=worker.generate(request)
            elif cmd=='reset':value=worker.reset(request)
            elif cmd=='step':value=worker.one_step()
            elif cmd=='shutdown':
                if worker.kind!='rule' and worker.weight_hash()!=worker.before:raise RuntimeError('Weights changed')
                value={'counts':worker.counts,'weights_unchanged':True,'peak_rss':peak_rss()}
            else:raise ValueError('Unknown worker command')
            print(canonical({'ok':True,'result':value}),flush=True)
            if cmd=='shutdown':return
    except BaseException as exc:
        print(canonical({'ok':False,'error':f'{type(exc).__name__}: {exc}','counts':worker.counts if worker else None}),flush=True)
        raise

if __name__=='__main__':main()
