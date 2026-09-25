"""No experiment imports: fake workers exercise resource and stop gates."""
import copy
import importlib.abc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy','gppo_world'):raise AssertionError('Experimental import forbidden')
sys.meta_path.insert(0,NoModels())
import run_diagnostic as runner
import attribution_contract as contract
from analysis_contract import decide
from analyze_runtime_necessity import analyze,percentile

class FakeMonitor:
    def __init__(self,out):self.out=out;self.children=[];self.stage=None
    def enter(self,stage):self.stage=stage
    def check(self,**kwargs):pass
    def snapshot(self):return {'peak_conservative_rss_sum':100}
    def measure(self,category,fn,*args,**kwargs):return fn(*args,**kwargs)

class FakeReservations:
    def __init__(self):self.pending=[];self.totals={}
    def reserve(self,amounts):
        if self.pending:raise AssertionError('Unresolved reservation')
        self.pending=[True]
        for k,v in amounts.items():self.totals[k]=self.totals.get(k,0)+v
    def complete(self):self.pending=[]

class FakeWorker:
    mismatch=False
    def __init__(self,name,stage,kind,auth,monitor):
        self.kind=kind;self.counts=dict.fromkeys(runner.COUNTERS,0);self.counts['model_loads']=1
        self.ready={'counts':dict(self.counts)};monitor.children.append(self)
    def call(self,command,**kwargs):
        if command!='replay_once':raise AssertionError('Only fixed replay permitted')
        self.counts['policy_encode']+=25;self.counts['actor_readout']+=25
        if self.kind=='full':self.counts['world_candidate_batch']+=25
        rows=[{'parent':'fixture','step':i,'action':0,'candidates':[0,24],
            'logits_sha256':'fixed','probabilities_sha256':'fixed','policy_hidden_sha256':'fixed',
            'wall_seconds':.004} for i in range(25)]
        if self.kind=='candidate' and self.mismatch:rows[0]['action']=24
        return {'counts':dict(self.counts),'records':rows,
            'stats':{'decisions':25,'cpu_seconds':.2 if self.kind=='full' else .1,'wall_seconds':.1}}
    def close(self):pass

class FakeMemory:
    def observe(self,p):pass
    def candidates(self,p):return [0,24]
    def submitted(self,p,a):pass

class EpisodeWorker(FakeWorker):
    def call(self,command,**kwargs):
        public={'mask':[True]*25}
        if command=='generate':
            tapes=[{'fixture':1},{'fixture':2}]
            return {'tapes':tapes,'hashes':[runner.digest(t) for t in tapes]}
        if command=='reset':
            self.counts['environment_resets']+=1
            return {'initial_public':public,'initial_hash':runner.digest(public),'counts':dict(self.counts)}
        if command=='step':
            for k in ('environment_steps','policy_encode','actor_readout'):self.counts[k]+=1
            counts={'completed':0,'expired':0};info={'counts':counts,'energy':{'one':36},
                'terminated':True,'truncated':False,'episode_end_reason':'terminated','completion_records':{}}
            return {'step':0,'submit_command':True,'public':public,'candidates':[0,24],'action':0,
                'info':info,'vector_reward':[0.,-0.],'diagnostic':{'world_enabled':False},
                'controller':{'cpu':.001,'wall':.001},'environment':{'cpu':.001,'wall':.001},
                'labels':{'cpu':.001,'wall':.001},'done':True,'counts':dict(self.counts),
                'episode':{'steps':1,'utility':0.,'energy_used':0.,'physical_on_time':0.,'host_on_time_observed':0.}}
        raise AssertionError(command)

class RunnerTests(unittest.TestCase):
    def test_A_exact_count_and_all_batches(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'OwnedWorker',FakeWorker):
            reservations=FakeReservations();m=FakeMonitor(Path(d))
            result=runner.replay_gate(None,m,reservations)
            self.assertEqual(reservations.totals,{'model_loads':3,'policy_encode':1575,'actor_readout':1575,'world_candidate_batch':525})
            self.assertAlmostEqual(result['paired_replay_cpu_ratio'],.5)
            self.assertEqual(len(result['batches']['candidate']),21)
            self.assertTrue((Path(d)/'A/status.json').is_file())
    def test_A_mismatch_stops_first_batch(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'OwnedWorker',FakeWorker),patch.object(FakeWorker,'mismatch',True):
            reservations=FakeReservations()
            with self.assertRaisesRegex(RuntimeError,'mismatch'):runner.replay_gate(None,FakeMonitor(Path(d)),reservations)
            self.assertEqual(reservations.totals['policy_encode'],75)
            self.assertFalse((Path(d)/'A/status.json').exists())
    def test_no_authorization_no_runs_or_ledger(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'ROOT',Path(d)):
            with self.assertRaises(PermissionError):runner.main([])
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_budget_cannot_borrow_stages(self):
        request=runner.read(runner.ROOT/'BUDGET_REQUEST.json')
        actual=contract.limits(request)
        self.assertEqual(actual['C/environment_steps'],86400)
        self.assertEqual(actual['C/world_candidate_batch'],0)
        request['stage_caps']['C']['world_candidate_batch']=1
        with self.assertRaises(ValueError):contract.limits(request)
    def test_unexpected_world_forward_counter_fails(self):
        before=dict.fromkeys(runner.COUNTERS,0);after=dict(before)
        after['environment_steps']=1;after['policy_encode']=1;after['actor_readout']=1;after['world_candidate_batch']=1
        with self.assertRaisesRegex(RuntimeError,'counter'):runner.exact_delta(before,after,{'environment_steps':1,'policy_encode':1,'actor_readout':1})
    def test_full_frozen_matrix_and_initial_pairing(self):
        request=runner.read(runner.ROOT/'BUDGET_REQUEST.json');rows=runner.read(runner.ROOT/'sample-manifest.json')
        old=runner.read(request['inputs']['baseline_episodes']);tapes=runner.read(request['inputs']['scenario_manifest'])
        runner.validate_matrix(rows,old,tapes)
        with self.assertRaises(RuntimeError):runner.validate_matrix(rows[:-1],old,tapes)
        bad=copy.deepcopy(rows);bad[0]['exogenous_key']='wrong'
        with self.assertRaisesRegex(RuntimeError,'exogenous'):runner.validate_matrix(bad,old,tapes)
        bad=copy.deepcopy(rows);bad[0]['initial_public_hash']='wrong'
        with self.assertRaisesRegex(RuntimeError,'pairing'):runner.validate_matrix(bad,old,tapes)
    def test_no_actual_model_import(self):
        self.assertFalse(any(k in sys.modules for k in ('torch','numpy','gppo_world')))
    def test_B_with_synthetic_worker_records_and_verifies(self):
        request=runner.read(runner.ROOT/'BUDGET_REQUEST.json')
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'OwnedWorker',EpisodeWorker),patch.object(runner,'PublicMemory',FakeMemory):
            reservations=FakeReservations()
            episodes,cost=runner.environments('B',None,FakeMonitor(Path(d)),reservations,request['inputs'])
            self.assertEqual(len(episodes),2);self.assertEqual(cost['compressed_rows'],2)
            self.assertEqual(reservations.totals,{'model_loads':1,'environment_resets':2,'environment_steps':2,'policy_encode':2,'actor_readout':2})
            self.assertEqual(reservations.pending,[])
    def test_step_rejects_illegal_action_and_false_reward(self):
        w=EpisodeWorker('fake','B','candidate',None,FakeMonitor(Path('.')))
        row=w.call('step');row['action']=7
        with self.assertRaisesRegex(RuntimeError,'guard'):runner.validate_step(row,0,FakeMemory(),{'completed':0,'expired':0},36)
        row=w.call('step');row['vector_reward']=[1,0]
        with self.assertRaisesRegex(RuntimeError,'Reward'):runner.validate_step(row,0,FakeMemory(),{'completed':0,'expired':0},36)
    def test_analysis_rejects_partial_or_nonfinite_matrix_before_statistics(self):
        with self.assertRaisesRegex(ValueError,'matrix'):analyze([],[],{}, {}, {})
        rows=[{'parent':f'eval-{p:04d}','repeat':r,'arm':a,'utility':float('nan'),
            'physical_on_time':0.,'host_on_time_observed':0.,'energy_used':0.}
            for p in range(1600) for r in range(3) for a in ('M','R','N')]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):analyze(rows[:9600],rows[9600:],{}, {}, {})
        self.assertEqual(percentile([0.,1.,2.],.5),1.)
    def test_real_historical_snapshot_schema_and_nonzero_totals(self):
        request=runner.read(runner.ROOT/'BUDGET_REQUEST.json')
        actual=runner.read(request['inputs']['historical_status'])['segment_budget']
        totals=runner.verified_totals(actual,request['history_counters'])
        self.assertEqual(totals['environment_steps'],13399)
        self.assertEqual(totals['policy_encode'],6329)
        self.assertEqual(totals['model_loads'],1)
        invalid=copy.deepcopy(actual);invalid['stages']['C/environment_steps']['pending']=1
        with self.assertRaisesRegex(RuntimeError,'Unresolved'):runner.verified_totals(invalid,request['history_counters'])

if __name__=='__main__':unittest.main(verbosity=2)
