"""No native imports, no model loading, no environment execution."""
import ast
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','gppo_world'}:
            raise AssertionError('Runtime import in zero-step test: '+fullname)
sys.meta_path.insert(0,RejectRuntime())
import recovery_contract as contract
import run_recovery as runner
from build_recovery_inventory import recompute_episode, planned

class RecoveryPreparationTests(unittest.TestCase):
    def test_unauthorized_cannot_create_budget_or_worker(self):
        with patch.object(runner,'budget_class',side_effect=AssertionError('budget')),patch.object(runner,'OwnedWorker',side_effect=AssertionError('worker')):
            with self.assertRaises(PermissionError):runner.main([])

    def test_source_payload_recomputes_all_fields(self):
        rows=json.loads((contract.ROOT/'missing-episode-source-steps.json').read_text(encoding='utf-8'))
        result=recompute_episode(rows,'synthetic-initial-hash')
        self.assertEqual(result['steps'],12)
        broken=copy.deepcopy(rows);broken[3]['vector_reward'][0]+=.1
        with self.assertRaises(AssertionError):recompute_episode(broken,'x')
        with self.assertRaises(AssertionError):recompute_episode(rows[1:],'x')

    def test_suffix_is_exact_and_disjoint(self):
        matrix=[{'parent':'p','repeat':i,'arm':a} for i in range(3) for a in ['M','R']]
        contract.reject_duplicate_or_wrong_order(matrix[:2],matrix[2:],matrix)
        with self.assertRaises(RuntimeError):contract.reject_duplicate_or_wrong_order(matrix[:2],matrix[1:],matrix)
        with self.assertRaises(RuntimeError):contract.reject_duplicate_or_wrong_order(matrix[:2],list(reversed(matrix[2:])),matrix)

    def test_original_frozen_pair_order(self):
        tapes={'hashes':['h']*1600};matrix=planned(tapes)
        self.assertEqual(len(matrix),9600)
        self.assertEqual([matrix[4034][k] for k in ['parent','repeat','arm']],['eval-0672',1,'R'])
        self.assertEqual(sum(r['arm']=='M' for r in matrix[4034:]),2783)

    def test_cumulative_never_refunds_history_and_load_extension_explicit(self):
        old={'C/environment_steps':{'limit':172800,'reserved':56381,'verified':56381,'unknown':0},'C/model_loads':{'limit':1,'reserved':1,'verified':1,'unknown':0}}
        seg={'stages':{'C/environment_steps':{'reserved':100188,'verified':100187,'unknown':1},'C/model_loads':{'reserved':1,'verified':1,'unknown':0}}}
        result=contract.cumulative_snapshot(old,seg)
        self.assertEqual(result['C/environment_steps']['reserved'],156569)
        self.assertEqual(result['C/environment_steps']['unknown'],1)
        self.assertEqual(result['C/model_loads']['authorized_limit_including_explicit_extension'],2)
        seg['stages']['C/environment_steps']['reserved']=172800
        with self.assertRaises(RuntimeError):contract.cumulative_snapshot(old,seg)

    def test_resource_caps_fit_remaining_stage_and_total(self):
        r=contract.read(contract.ROOT/'RECOVERY_BUDGET_REQUEST.json');c=r['caps'];h=r['historical_resources'];hold=r['shutdown_holdback']
        self.assertEqual(c['environment_steps'],5566*18)
        self.assertEqual(c['policy_encode'],2783*18)
        self.assertLessEqual(h['C_wall_seconds']+hold['wall_seconds']+c['wall_seconds'],129600)
        self.assertLessEqual(h['C_cpu_seconds']+hold['cpu_seconds']+c['total_process_cpu_seconds'],518400)
        self.assertLessEqual(h['total_wall_seconds']+hold['wall_seconds']+c['wall_seconds'],134400)
        self.assertLessEqual(h['total_cpu_seconds']+hold['cpu_seconds']+c['total_process_cpu_seconds'],536400)
        self.assertLessEqual(h['artifact_bytes_conservative']+c['artifact_bytes'],40*1024**3)

    def test_worker_and_decision_code_unchanged(self):
        for name in ['stage_worker.py','decision_only_inference.py','analyze_formal.py']:
            self.assertEqual((contract.ROOT/name).read_bytes(),(contract.OLD/name).read_bytes())

    def test_actual_recovery_loop_visits_only_5566_suffix_keys_with_stubs(self):
        matrix=contract.read(contract.ROOT/'frozen-matrix.json')
        retained=[{**r,'initial_public_hash':'h'} for r in matrix[:4034]]
        calls=[];writes=[];workers=[]
        class FakeMemory:
            def observe(self,p):pass
            def candidates(self,p):return [24]
            def submitted(self,p,a):pass
        class FakeWorker:
            def __init__(self,name,stage,kind,auth,monitor):
                self.name=name;self.arm='M' if kind=='candidate' else 'R'
                self.counts={k:0 for k in [*runner.MODEL_KEYS,'environment_steps','rule_decisions','environment_resets','model_loads']}
                self.counts['model_loads']=int(self.arm=='M')
                self.ready={'counts':dict(self.counts),'torch_loaded':self.arm=='M'};workers.append(self)
            def call(self,command,**kw):
                if command=='reset':
                    calls.append((kw['key'],self.arm));self.counts['environment_resets']+=1
                    return {'initial_hash':'h','counts':dict(self.counts)}
                self.counts['environment_steps']+=1
                for k in (runner.MODEL_KEYS if self.arm=='M' else ['rule_decisions']):self.counts[k]+=1
                return {'step':0,'counts':dict(self.counts),'public':{'mask':[False]*24+[True]},'candidates':[24],
                    'action':24,'info':{'energy':{'u':36.},'counts':{'completed':0,'expired':0},'communication_delta':[]},
                    'vector_reward':[0.,0.],'controller':{'cpu':.001,'wall':.001},'environment':{'cpu':.001,'wall':.001},
                    'labels':{'cpu':.001,'wall':.001},'done':True,'episode':{'utility':0.,'steps':1}}
            def close(self):pass
        class FakeLogger:
            def __init__(self,p):
                self.raw=types.SimpleNamespace(closed=False);self.rows=0;self.digest=types.SimpleNamespace(hexdigest=lambda:'stub')
            def add(self,r):self.rows+=1
            def close(self):self.raw.closed=True
        def read(p):
            if p.name=='retained-episodes.json':return retained
            if p.name=='not-executed.json':return matrix[4034:]
            if p.name=='frozen-matrix.json':return matrix
            if p.name=='scenario-manifest.json':return {'tapes':[{}]*1600,'hashes':['h']*1600}
            raise AssertionError(p)
        with tempfile.TemporaryDirectory() as folder:
            mon=Mock();mon.out=Path(folder);mon.measure.side_effect=lambda label,fn,*a:fn(*a)
            mon.snapshot.side_effect=lambda:{'workers':[{'name':w.name,'cpu_seconds':100.,'ready':{'startup_cpu':0.}} for w in workers]}
            res=Mock();res.pending=[]
            with patch.object(runner,'read',side_effect=read),patch.object(runner,'digest',return_value='h'),patch.object(runner,'OwnedWorker',FakeWorker),patch.object(runner,'PublicMemory',FakeMemory),patch.object(runner,'CompressedRecords',FakeLogger),patch.object(runner,'ending'),patch.object(runner,'verify_compressed'),patch.object(runner,'append',side_effect=lambda p,v:writes.append(v)),patch.object(runner,'write'),patch.object(runner,'publish_progress'):
                episodes,cost=runner.stages_environment('C',Path('synthetic'),mon,res,{})
        self.assertEqual(calls,[(r['random_key'],r['arm']) for r in matrix[4034:]])
        self.assertEqual(len(episodes),5566);self.assertEqual(cost['interface_steps'],5566)
        self.assertEqual([w.counts['environment_resets'] for w in workers],[2783,2783])

    def test_reset_assignments_clear_episode_state_using_stub_only(self):
        tree=ast.parse((contract.ROOT/'stage_worker.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Worker')
        reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
        fake=types.ModuleType('gppo_world.m10_environment');envs=[]
        class FakeEnv:
            def __init__(self,config,tape,exogenous_key):
                self.key=exogenous_key;self.clock=types.SimpleNamespace(resources={'x':types.SimpleNamespace(energy=36.)});self.view=types.SimpleNamespace(_completed_tasks=set());envs.append(self)
            def reset(self):return {'synthetic_stub':True}
        fake.M10Environment=FakeEnv;fake.scenario_from_dict=lambda t:dict(t)
        ns={'PublicMemory':lambda:object(),'public_copy':copy.deepcopy,'digest':lambda x:'stub-hash'}
        exec(compile(ast.Module(body=[reset],type_ignores=[]),'<exact-reset-method-with-stubs>','exec'),ns)
        w=types.SimpleNamespace(config=object(),counts={'environment_resets':0},state='old',memory='old',step_index=99,done=True,utility=12.)
        with patch.dict(sys.modules,{'gppo_world':types.ModuleType('gppo_world'),'gppo_world.m10_environment':fake}):
            ns['reset'](w,{'tape':{},'key':'a'});m=w.memory;ns['reset'](w,{'tape':{},'key':'b'})
        self.assertEqual(len(envs),2);self.assertIsNot(envs[0],envs[1]);self.assertIsNot(m,w.memory)
        self.assertIsNone(w.state);self.assertEqual(w.step_index,0);self.assertFalse(w.done);self.assertEqual(w.utility,0.)
        self.assertEqual(w.counts['environment_resets'],2)

    def test_progress_failure_does_not_abort_simulation(self):
        mon=types.SimpleNamespace(out=Path('not-created-test-path'))
        with patch.object(Path,'open',side_effect=PermissionError('simulated reader')):runner.publish_progress(mon,1,('p',0,'M'))
        self.assertEqual(mon.observer_publish_errors,1)

    def test_cost_merge_keeps_all_old_and_new_timings(self):
        old={'latencies':{'M':[{'cpu':10.,'wall':2.}],'R':[{'cpu':3.,'wall':4.}]},'worker_parts':{},'communication_events':{},'resources':{'wall_seconds':9.}}
        new=copy.deepcopy(old);new['latencies']['M'][0]['cpu']=1.
        combined=runner.merge_cost(old,new,{})
        self.assertEqual([v['cpu'] for v in combined['latencies']['M']],[10.,1.])
        self.assertEqual(len(old['latencies']['M']),1)

    def test_rss_counts_suspended_workers(self):
        request=contract.read(contract.ROOT/'RECOVERY_BUDGET_REQUEST.json')
        with tempfile.TemporaryDirectory() as temp,patch.object(runner,'peak_rss',return_value=1024**3),patch.object(runner,'process_cpu',return_value=0):
            mon=runner.Monitor(request,Path(temp));worker=Mock();worker.p.poll.return_value=None;worker.p._handle=11;worker.suspended=True;mon.children=[worker]
            with patch.object(runner,'process_memory',return_value=(4*1024**3,4*1024**3)):
                with self.assertRaisesRegex(RuntimeError,'Combined RSS cap'):mon.check()

    def test_no_dynamic_imports(self):
        for name in ['torch','numpy','gppo_world']:self.assertNotIn(name,sys.modules)
        for p in contract.ROOT.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')

if __name__=='__main__':unittest.main(verbosity=2)
