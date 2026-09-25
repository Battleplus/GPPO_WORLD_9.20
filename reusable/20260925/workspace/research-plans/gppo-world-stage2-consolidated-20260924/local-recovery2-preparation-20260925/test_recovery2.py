"""Zero-environment tests. Native simulation and model imports are prohibited."""
import copy
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from unittest.mock import patch

class NoExperimentImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy','gppo_world'):
            raise AssertionError('Forbidden experiment import: '+fullname)
sys.meta_path.insert(0,NoExperimentImports())

import recovery_contract as contract
import run_recovery as runner
from session_budget import single_writer_budget

HERE=Path(__file__).resolve().parent
NATIVE=Path(r'E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot\gppo_world\budget_executor.py')
spec=importlib.util.spec_from_file_location('test_only_budget',NATIVE)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
Session=single_writer_budget(native.PersistentBudget)

def load(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def normalize(s):
    s=copy.deepcopy(s);s.pop('db_path')
    for row in s['history']:row.pop('reservation_id')
    return s

class LedgerTests(unittest.TestCase):
    def test_original_session_equivalence_caps_and_unknown(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-ledger-') as d:
            records=[]
            for name,cls in [('original',native.PersistentBudget),('session',Session)]:
                b=cls(Path(d)/(name+'.sqlite3'),limits={'test':6},run_limits={'test':4},attempt_id='test',run_id='r1')
                a=b.reserve('test',2);b.complete(a);b.complete(a)
                z=b.reserve('test');b.unknown(z,'injected')
                with self.assertRaises(native.BudgetStateConflict):b.complete(z)
                with self.assertRaises(native.BudgetExhausted):b.reserve('test',2)
                p=b.reserve('test');self.assertEqual(b.run_totals('r1')['test']['pending'],1)
                b.complete(p);q=b.reserve('test',2,run_id='r2');b.complete(q)
                with self.assertRaises(native.BudgetExhausted):b.reserve('test',run_id='r3')
                self.assertEqual(b.integrity_check()['integrity_check'],'ok')
                records.append(normalize(b.snapshot()))
                if name=='session':
                    self.assertEqual(b.connection_open_count,1);b.close()
                    with self.assertRaises(RuntimeError):b.snapshot()
                    Path(d,name+'.sqlite3').rename(Path(d,'released.sqlite3'))
            self.assertEqual(*records)

    def test_rollback_on_exception_and_cross_thread_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-ledger-') as d:
            b=Session(Path(d)/'temporary.sqlite3',limits={'test':2},attempt_id='test')
            try:
                with self.assertRaisesRegex(ValueError,'primary'):
                    with closing(b._connect()) as c:
                        c.execute('BEGIN IMMEDIATE');c.execute('UPDATE stages SET reserved=2')
                        raise ValueError('primary')
                self.assertEqual(b.snapshot()['stages']['test']['reserved'],0)
                errors=[]
                def other_thread():
                    try:b.snapshot()
                    except RuntimeError as e:errors.append(str(e))
                t=threading.Thread(target=other_thread);t.start();t.join(5)
                self.assertEqual(errors,['Ledger single-writer thread changed'])
            finally:b.close()

    def test_abrupt_exit_preserves_committed_pending(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-crash-') as d:
            path=Path(d)/'temporary.sqlite3'
            script="""import importlib.util,os,sys
from session_budget import single_writer_budget
s=importlib.util.spec_from_file_location('budget',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=single_writer_budget(m.PersistentBudget)(sys.argv[2],limits={'test':4},attempt_id='test')
b.complete(b.reserve('test'));b.reserve('test');os._exit(19)
"""
            p=subprocess.run([sys.executable,'-B','-c',script,str(NATIVE),str(path)],cwd=HERE,timeout=20,capture_output=True)
            self.assertEqual(p.returncode,19,p.stderr)
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as c:
                self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0],'ok')
                self.assertEqual(c.execute('SELECT reserved,verified,unknown FROM stages').fetchone(),(2,1,0))

class ContractTests(unittest.TestCase):
    def test_exact_suffix_and_envelopes(self):
        completed=load('retained-episodes.json');remaining=load('not-executed.json');matrix=load('frozen-matrix.json')
        contract.reject_duplicate_or_wrong_order(completed,remaining,matrix)
        self.assertEqual((len(completed),len(remaining)),(8637,963))
        self.assertEqual(sum(r['arm']=='M' for r in remaining),481)
        with self.assertRaises(RuntimeError):contract.reject_duplicate_or_wrong_order(completed,remaining[::-1],matrix)
        request=load('RECOVERY_BUDGET_REQUEST.json');history=load('historical-budget-readonly.json')
        contract.validate_resource_envelope(request,history)
        request['caps']['environment_resets']+=1
        with self.assertRaises(RuntimeError):contract.validate_resource_envelope(request,history)

    def test_unauthorized_never_creates_run_or_worker(self):
        with patch.object(runner,'OwnedWorker',side_effect=AssertionError('worker created')):
            with self.assertRaises(PermissionError):runner.main([])
        self.assertFalse((HERE/'runs').exists())

    def test_explicit_amendments_and_hash_binding(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-auth-') as d:
            root=Path(d);request=load('RECOVERY_BUDGET_REQUEST.json');request['preparation_ready']=True
            (root/'RECOVERY_BUDGET_REQUEST.json').write_text(json.dumps(request))
            (root/'historical-budget-readonly.json').write_text(json.dumps(load('historical-budget-readonly.json')))
            bound=root/'bound.txt';bound.write_text('frozen')
            (root/'recovery-manifest.json').write_text(json.dumps({'files':[{'path':str(bound),'sha256':contract.sha(bound)}]}))
            a={'approved':True,'attempt':'consolidated-v1-recovery-2-local-once',
               'recovery_manifest_sha256':contract.sha(root/'recovery-manifest.json'),
               'budget_request_sha256':contract.sha(root/'RECOVERY_BUDGET_REQUEST.json'),
               'caps':request['caps'],'user_approval_reference':'synthetic-test-only','user_approval_text':'synthetic-test-only',
               'accepted_amendments':request['required_authorization_amendments']}
            auth=root/'authorization.json';auth.write_text(json.dumps(a))
            with patch.object(contract,'ROOT',root):
                contract.authorize(auth)
                for field in a['accepted_amendments']:
                    broken=copy.deepcopy(a);del broken['accepted_amendments'][field];auth.write_text(json.dumps(broken))
                    with self.assertRaises(PermissionError):contract.authorize(auth)
                auth.write_text(json.dumps(a));bound.write_text('changed')
                with self.assertRaises(RuntimeError):contract.authorize(auth)

    def test_exact_prefix_behavior_but_not_timing(self):
        original=load('partial-prefix.json')['steps'][0];a=copy.deepcopy(original)
        a['controller']={'cpu':999,'wall':999};contract.verify_prefix_step(a,original)
        for field in contract.PREFIX_FIELDS:
            a=copy.deepcopy(original);a[field]=None
            with self.assertRaises((RuntimeError,ValueError)):contract.verify_prefix_step(a,original)

    def test_timing_merge_never_selects_faster_replay(self):
        old={'latencies':{'M':[{'cpu':2,'wall':2}],'R':[{'cpu':3,'wall':3}]*7},'worker_parts':{},'communication_events':{},'resources':{}}
        new={'latencies':{'M':[{'cpu':4,'wall':4}],'R':[{'cpu':5,'wall':5}]},'worker_parts':{},'communication_events':{},'resources':{},'replay_overhead_latencies':[{'cpu':1,'wall':1}]*7}
        result=runner.merge_cost(old,new,{})
        self.assertEqual(len(result['latencies']['R']),8)
        self.assertEqual(result['latencies']['R'][:7],old['latencies']['R'])
        self.assertEqual(len(result['replay_overhead_latencies']),7)

class OrchestratorTests(unittest.TestCase):
    def exercise(self,mismatch=False):
        retained=load('retained-episodes.json');remaining=load('not-executed.json');matrix=load('frozen-matrix.json');prefix=load('partial-prefix.json')
        initials={(r['parent'],r['repeat']):r['initial_public_hash'] for r in retained}
        current=[None];seen=[];workers=[];recorded=[];verified=[]
        class Memory:
            def observe(self,p):pass
            def candidates(self,p):return current[0]['candidates']
            def submitted(self,p,a):pass
        class Worker:
            def __init__(self,name,stage,kind,auth,monitor):
                self.name=name;self.arm=name[-1];self.ready={'torch_loaded':False,'startup_cpu':0,'counts':{k:0 for k in ['environment_steps','environment_resets','model_loads','policy_encode','world_candidate_batch','actor_readout','rule_decisions']}}
                self.ready['counts']['model_loads']=int(self.arm=='M');self.counts=self.ready['counts'].copy();workers.append(self)
            def call(self,command,**kw):
                if command=='reset':
                    parts=kw['key'].split('|');self.parent=parts[1];self.repeat=int(parts[2].split('-')[-1]);self.step=0;self.uc=0
                    self.partial=(self.parent,self.repeat,self.arm)==tuple(prefix['key']);self.counts['environment_resets']+=1
                    seen.append((self.parent,self.repeat,self.arm))
                    return {'initial_hash':initials.get((self.parent,self.repeat),'same'),'counts':self.counts.copy()}
                if self.partial and self.step<7:
                    row=copy.deepcopy(prefix['steps'][self.step])
                    if mismatch and self.step==3:row['diagnostic']={'injected':'mismatch'}
                else:
                    info=copy.deepcopy(prefix['steps'][-1]['info']) if self.partial else {'counts':{'completed':0,'expired':0},'energy':{'all':36},'communication_delta':[]}
                    row={'step':self.step,'public':{'mask':[True]},'candidates':[0],'action':0,'info':info,'vector_reward':[0.,0.],'done':True}
                    row['episode']={'utility':self.uc,'steps':self.step+1}
                for k in ['environment_steps',*(['policy_encode','world_candidate_batch','actor_readout'] if self.arm=='M' else ['rule_decisions'])]:self.counts[k]+=1
                row['counts']=self.counts.copy()
                for label in ['controller','environment','labels']:row[label]={'cpu':.001,'wall':.001}
                self.uc+=.99**self.step*(.4*row['vector_reward'][0]+.2*row['vector_reward'][1]);self.step+=1;current[0]=row
                return row
            def close(self):pass
        class Monitor:
            def __init__(self,out):self.out=out
            def enter(self,s):pass
            def measure(self,label,fn,*a,**k):return fn(*a,**k)
            def check(self,**k):pass
            def snapshot(self):return {'workers':[{'name':w.name,'ready':w.ready,'cpu_seconds':1000} for w in workers]}
        class Reservations:
            pending=[]
            def reserve(self,a):self.pending=['synthetic']
            def complete(self):self.pending=[];verified.append(1)
        class Records:
            def __init__(self,path):self.raw=type('Raw',(),{'closed':False})();self.digest=__import__('hashlib').sha256();self.rows=0
            def add(self,row):self.rows+=1;recorded.append(copy.deepcopy(row))
            def close(self):self.raw.closed=True
        data={'retained-episodes.json':retained,'not-executed.json':remaining,'frozen-matrix.json':matrix,'partial-prefix.json':prefix,
              'scenario-manifest.json':{'tapes':[{'i':i} for i in range(1600)],'hashes':[runner.digest({'i':i}) for i in range(1600)]}}
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-orchestrator-') as d:
            with patch.multiple(runner,OwnedWorker=Worker,PublicMemory=Memory,CompressedRecords=Records,ending=lambda *a:None,
                    read=lambda p:copy.deepcopy(data[Path(p).name]),verify_compressed=lambda *a:None):
                if mismatch:
                    with self.assertRaisesRegex(RuntimeError,'Partial prefix mismatch'):
                        runner.stages_environment('C',Path('synthetic'),Monitor(Path(d)),Reservations(),{})
                    self.assertEqual(seen,[tuple(prefix['key'])]);self.assertEqual(len(recorded),4)
                    self.assertEqual(len(verified),6) # load, reset, four recorded steps
                else:
                    episodes,cost=runner.stages_environment('C',Path('synthetic'),Monitor(Path(d)),Reservations(),{})
                    self.assertEqual(seen,[(r['parent'],r['repeat'],r['arm']) for r in remaining])
                    self.assertEqual(len(episodes),963);self.assertEqual(cost['interface_steps'],970)
                    self.assertEqual(len(cost['replay_overhead_latencies']),7)
                    self.assertEqual(len(cost['latencies']['M']),481);self.assertEqual(len(cost['latencies']['R']),482)
        self.assertFalse(any(n in sys.modules for n in ['torch','numpy','gppo_world']))
    def test_full_suffix_using_stubs_only(self):self.exercise()
    def test_prefix_mismatch_stops_before_next_episode(self):self.exercise(True)

if __name__=='__main__':unittest.main(verbosity=2)
