"""Public fixtures and stub calls only. Native/model imports are blocked."""
import ast
import copy
import importlib.abc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','gppo_world'}:
            raise AssertionError('Forbidden model/environment import during pure tests: '+fullname)


sys.meta_path.insert(0,RejectNative())
from public_controller import ContractError,PublicMemory,PublicPlanner,public_copy,travel_cost
import run_gate


def observation(now=0.,tasks=2):
    def row(values): return [x for v in values for x in (v,1.,1.,0.)]
    return {'time':now,'version':1,'uavs':[row((0,0,9,1,1,1)) for _ in range(4)],
            'tasks':[row((j+1,0,14,1,1,1,0,0)) if j<tasks else [0.]*32 for j in range(6)],
            'mask':[j<tasks for u in range(4) for j in range(6)]+[True],
            'public_entity_ids':{'uavs':[f'u{i}' for i in range(4)],'tasks':[f't{j}' for j in range(tasks)]},
            'continuation_actions':[],'flat':[],'graph':{}}


class GuardTests(unittest.TestCase):
    def test_private_keys_not_forwarded(self):
        o=observation();o['env']=object();o['info']={'future_truth':999}
        p=public_copy(o);self.assertNotIn('env',p);self.assertNotIn('info',p)
        self.assertEqual(o['mask'],p['mask'])

    def test_shared_ack_filter_task_and_resource(self):
        o=observation();o['continuation_actions']=[0];before=copy.deepcopy(o)
        m=PublicMemory();m.observe(o);c=m.candidates(o)
        self.assertNotIn(1,c);self.assertNotIn(6,c);self.assertIn(7,c);self.assertIn(24,c)
        self.assertEqual(o,before)

    def test_handle_removal_releases_without_hidden_lease(self):
        o=observation();o['continuation_actions']=[0];m=PublicMemory();m.observe(o)
        n=observation(1);m.observe(n);self.assertIn(0,m.candidates(n))

    def test_pending_not_ack_and_old_receipts_do_not_release(self):
        o=observation();m=PublicMemory();m.observe(o);m.submitted(o,0)
        n=observation(1)
        for rows in [n['uavs'],n['tasks']]:
            for r in rows:
                for i in range(3,len(r),4):r[i]=1.
        m.observe(n)
        self.assertEqual(m.active,());self.assertNotIn(0,m.candidates(n));self.assertNotIn(1,m.candidates(n))
        self.assertIn(7,m.candidates(n))
        self.assertEqual(m.get('uavs','u0','idle')['measured_at'],0.)

    def test_both_new_measurements_required(self):
        o=observation();m=PublicMemory();m.observe(o);m.submitted(o,0)
        n=observation(1);n['tasks'][0][23]=1.;m.observe(n)
        self.assertNotIn(0,m.candidates(n))
        q=observation(2);m.observe(q);self.assertIn(0,m.candidates(q))

    def test_expiry_not_acceptance_or_permission(self):
        o=observation();m=PublicMemory();m.observe(o);m.submitted(o,0)
        for now in [3.99,4.]:
            n=observation(now)
            for rows in [n['uavs'],n['tasks']]:
                for r in rows:
                    for i in range(3,len(r),4):r[i]=now
            m.observe(n)
            self.assertEqual(0 in m.candidates(n),now>=4.)
            self.assertEqual(m.active,())

    def test_ack_resolves_pending_to_known_busy(self):
        o=observation();m=PublicMemory();m.observe(o);m.submitted(o,0)
        n=observation(1);n['continuation_actions']=[0];m.observe(n)
        self.assertFalse(m.pending);self.assertNotIn(0,m.candidates(n))

    def test_noop_never_added_if_illegal(self):
        o=observation();o['mask']=[False]*25;m=PublicMemory();m.observe(o)
        with self.assertRaises(ContractError):m.candidates(o)

    def test_future_and_undelivered_rejected(self):
        o=observation();o['uavs'][0][3]=-.1
        with self.assertRaises(ContractError):public_copy(o)
        o=observation();o['mask'][5]=True
        with self.assertRaises(ContractError):public_copy(o)

    def test_both_arms_have_isolated_identical_logic(self):
        o=observation();a=PublicMemory();b=PublicMemory();a.observe(o);b.observe(o)
        self.assertEqual(a.candidates(o),b.candidates(o));a.submitted(o,0)
        self.assertIn(0,b.candidates(o));self.assertNotIn(0,a.candidates(o))


class PlannerTests(unittest.TestCase):
    def test_current_mask_cannot_be_revived_by_prediction(self):
        o=observation();o['mask']=[False]*24+[True]
        m=PublicMemory();m.observe(o)
        action,d=PublicPlanner(node_cap=40).choose(o,m)
        self.assertEqual(action,24);self.assertEqual(d['nodes'],0)

    def test_finite_search_deterministic_and_state_unchanged(self):
        o=observation();before=copy.deepcopy(o);m=PublicMemory();m.observe(o)
        r=PublicPlanner(node_cap=50)
        a,da=r.choose(o,m);b,db=r.choose(o,m)
        self.assertEqual((a,da),(b,db));self.assertIn(a,m.candidates(o))
        self.assertLessEqual(da['nodes'],50);self.assertEqual(before,o)

    def test_completion_at_deadline_and_energy_bound(self):
        o=observation(tasks=1);o['tasks'][0][8]=1.
        m=PublicMemory();m.observe(o);p=PublicPlanner(node_cap=50)
        jobs,en,n=p.initial(o,m)
        self.assertIsNotNone(p.child(n,0,jobs,root=True))
        o['uavs'][0][8]=.35;m=PublicMemory();m.observe(o);jobs,en,n=p.initial(o,m)
        self.assertIsNone(p.child(n,0,jobs,root=True))

    def test_known_busy_prediction_not_free_now(self):
        o=observation();o['continuation_actions']=[0]
        m=PublicMemory();m.observe(o);p=PublicPlanner(node_cap=40)
        jobs,en,n=p.initial(o,m)
        self.assertNotIn(0,jobs);self.assertGreater(n.vehicles[0].ready,0)
        self.assertIsNone(p.child(n,1,jobs,root=True))

    def test_public_age_changes_estimate_not_mask(self):
        o=observation(tasks=1);o['uavs'][0][3]=2.;o['uavs'][0][7]=2.
        m=PublicMemory();m.observe(o);p=PublicPlanner(node_cap=40);jobs,en,n=p.initial(o,m)
        child=p.child(n,0,jobs,root=True)
        self.assertEqual(dict(child.completions)[0],3.)
        self.assertTrue(o['mask'][0])

    def test_travel_energy_not_double_idle_and_capped(self):
        self.assertAlmostEqual(travel_cost(((0,0.,1.),),0.,1.,(9.,0.,0.,0.)),.35)
        self.assertAlmostEqual(travel_cost(((0,0.,1.),),0.,1.,(.1,0.,0.,0.)),.1)

    def test_unserved_deadline_is_failure_not_missing(self):
        o=observation(tasks=1);m=PublicMemory();m.observe(o);p=PublicPlanner(node_cap=40)
        jobs,en,n=p.initial(o,m)
        noop=p.child(n,24,jobs);served=p.child(n,0,jobs,root=True)
        self.assertGreater(p.score(served,jobs,0.,en),p.score(noop,jobs,0.,en))


class RunnerTests(unittest.TestCase):
    def test_missing_authorization_no_native_import(self):
        with self.assertRaises(PermissionError):run_gate.authorize(None)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('gppo_world',sys.modules)

    def test_pending_template_not_approval(self):
        path=Path(__file__).parent/'authorization-template.json'
        with self.assertRaises(PermissionError):run_gate.authorize(path)

    def test_call_failure_unknown_once_no_retry(self):
        events=[]
        class StubBudget:
            def reserve(self,stage,amount):events.append('reserve');return {'stage':stage}
            def unknown(self,token,reason):events.append('unknown')
            def complete(self,token):events.append('complete')
        def fail():events.append('call');raise ValueError('failure')
        with patch.object(run_gate,'append',lambda *a:None):
            calls=run_gate.DurableCalls(StubBudget(),Path('not-written'))
            with self.assertRaises(ValueError):calls.call('policy_encode',fail)
        self.assertEqual(events,['reserve','call','unknown'])

    def test_call_success_reserved_then_verified(self):
        events=[]
        class StubBudget:
            def reserve(self,s,n):events.append('reserve');return {}
            def complete(self,t):events.append('verified')
            def unknown(self,t,r):self.fail('unexpected')
        with patch.object(run_gate,'append',lambda *a:None):
            c=run_gate.DurableCalls(StubBudget(),Path('not-written'))
            self.assertEqual(c.call('world_candidate_batch',lambda:7),7)
        self.assertEqual(events,['reserve','verified']);self.assertGreaterEqual(c.audit_cpu,0)

    def test_independent_reward_contract_detects_mismatch(self):
        info={'counts':{'completed':1,'expired':0},'energy':{'u0':8.,'u1':9.,'u2':9.,'u3':9.}}
        def good(*args):return [run_gate.f32(1/6),run_gate.f32(-1/36)],[],info['counts'],35.
        v,_,_=run_gate.verify_vector(info,{'completed':0,'expired':0},36.,good,None)
        self.assertEqual(v,[run_gate.f32(1/6),run_gate.f32(-1/36)])
        with self.assertRaises(ContractError):
            run_gate.verify_vector(info,{'completed':0,'expired':0},36.,lambda *a:([0.,0.],[],{},35.),None)

    def test_no_top_level_native_or_sqlite_import(self):
        for filename in ['run_gate.py','public_controller.py']:
            tree=ast.parse((Path(__file__).parent/filename).read_text(encoding='utf-8'))
            for node in tree.body:
                if isinstance(node,ast.Import):names=[n.name for n in node.names]
                elif isinstance(node,ast.ImportFrom):names=[node.module or '']
                else:continue
                self.assertFalse(any(n.split('.')[0] in {'gppo_world','torch','numpy','sqlite3'} for n in names))

    def test_native_time_limit_is_not_termination(self):
        result=run_gate.ending(True,{'terminated':False,'truncated':True,'episode_end_reason':'time_limit'})
        self.assertFalse(result['terminated']);self.assertTrue(result['truncated'])
        with self.assertRaises(ContractError):run_gate.ending(False,{'terminated':True,'truncated':False})
        with self.assertRaises(ContractError):run_gate.ending(True,{})

    def test_durable_log_failure_marks_call_unknown(self):
        events=[]
        class StubBudget:
            def reserve(self,s,n):events.append('reserve');return {}
            def complete(self,t):events.append('verified')
            def unknown(self,t,r):events.append('unknown')
        def fn():events.append('forward');return 3
        with patch.object(run_gate,'append',side_effect=[None,OSError('disk full')]):
            with self.assertRaises(OSError):run_gate.DurableCalls(StubBudget(),Path('not-written')).call('actor_readout',fn)
        self.assertEqual(events,['reserve','forward','unknown'])


if __name__=='__main__':unittest.main(verbosity=2)
