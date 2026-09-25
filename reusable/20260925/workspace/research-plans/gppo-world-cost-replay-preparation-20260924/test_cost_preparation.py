import ast
import importlib.abc
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','gppo_world'}:raise AssertionError('Unauthorized native import')
sys.meta_path.insert(0,RejectNative())
import cost_replay as c

def obs():
    return {'time':0.,'uavs':[[0.]*24 for _ in range(4)],'tasks':[[0.]*32 for _ in range(6)],
        'mask':[False]*24+[True],'public_entity_ids':{'uavs':['u0','u1','u2','u3'],'tasks':[]},'continuation_actions':[]}

class PreparationTests(unittest.TestCase):
    def test_missing_approval_denies(self):
        with self.assertRaises(PermissionError):c.authorize(None)
    def test_pending_template_denies(self):
        with self.assertRaises(PermissionError):c.authorize(Path(__file__).parent/'authorization-template.json')
    def test_incomplete_trace_denies(self):
        with self.assertRaises(ValueError):c.groups([], 'M')
    def test_batch_uses_aggregate_cpu_and_isolates_episodes(self):
        traces=[[{'parent':p,'step':0,'observation':obs(),'action':24}] for p in ['a','b']]
        calls=[]
        def choose(public,memory,state,candidates):
            self.assertIsNone(state);self.assertEqual(candidates,(24,));self.assertEqual(len(memory.history),1)
            calls.append(1);return 24,'unshared',None
        cpu=iter([10.,10.5]);wall=iter([1.,1.1,1.2,1.3,1.4,2.])
        with patch.object(c,'save',side_effect=AssertionError('IO inside batch')):
            stat,records=c.replay_batch(traces,choose,lambda:next(wall),lambda:next(cpu))
        self.assertEqual(stat['cpu_seconds'],.5);self.assertEqual(stat['decisions'],2);self.assertEqual(len(calls),2)
        c.verify_records(traces,records)
    def test_mismatch_not_silently_accepted(self):
        traces=[[{'parent':'a','step':0,'action':0}]]
        with self.assertRaises(RuntimeError):c.verify_records(traces,[('a',0,24,None,.1)])
    def test_hidden_mismatch_stops(self):
        traces=[[{'parent':'a','step':0,'action':24,'diagnostic':{'selected_world_hidden_sha256':'expected'}}]]
        with self.assertRaises(RuntimeError):c.verify_records(traces,[('a',0,24,None,.1)],lambda x:'wrong')
    def test_caps_include_first_pass(self):
        self.assertEqual(c.REPETITIONS,21)
        self.assertEqual(c.LIMITS['actor_readout'],25*21);self.assertEqual(c.LIMITS['rule_decisions'],31*21)
        self.assertEqual(c.LIMITS['environment_steps'],0)
    def test_no_environment_calls_or_optimizers_in_ast(self):
        tree=ast.parse((Path(__file__).parent/'cost_replay.py').read_text(encoding='utf-8'))
        calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in {'step','reset','backward'} for n in calls))
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='M10Environment' for n in calls))
        self.assertNotIn('torch',sys.modules);self.assertNotIn('gppo_world',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
