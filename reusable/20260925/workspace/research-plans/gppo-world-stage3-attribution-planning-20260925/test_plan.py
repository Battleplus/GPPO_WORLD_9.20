import ast
import importlib.abc
import sys
import unittest
from pathlib import Path
from analysis_contract import decide

class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy','gppo_world'):
            raise AssertionError('No experimental import authorized')
sys.meta_path.insert(0,RejectModels())
import no_online_world

class DecisionTests(unittest.TestCase):
    def args(self):
        return dict(n_minus_m_ci=[-.003,.002],n_minus_r_ci=[.05,.07],physical_difference=0,
            host_difference=0,n_cpu_ms=6,n_wall_p95_ms=5,rss_sum_bytes=400000000,
            paired_replay_cpu_ratio=.7,historical_formal_cpu_ratio=.7)
    def test_retention_and_actual_cost_required(self):
        a=self.args();r=decide(**a)
        self.assertIn('removal_supported',r['decision']);self.assertFalse(r['world_independent_contribution_proven'])
        a['paired_replay_cpu_ratio']=.9
        self.assertEqual(decide(**a)['decision'],'task_retention_supported_cost_saving_not_accepted')
    def test_ood_loss_does_not_prove_world(self):
        a=self.args();a['n_minus_m_ci']=[-.07,-.02];r=decide(**a)
        self.assertEqual(r['decision'],'direct_removal_not_supported_no_independent_world_claim')
        self.assertFalse(r['world_independent_contribution_proven']);self.assertFalse(r['automatic_training'])
    def test_uncertain_and_boundary_not_pass(self):
        a=self.args();a['n_minus_m_ci']=[-.01,.02]
        self.assertEqual(decide(**a)['decision'],'uncertain_stop_without_more_samples_or_training')
        a=self.args();a['n_minus_r_ci']=[.01,.02]
        self.assertEqual(decide(**a)['decision'],'uncertain_stop_without_more_samples_or_training')
    def test_task_guard_cannot_be_bought_with_cpu(self):
        a=self.args();a['physical_difference']=-.00001
        self.assertFalse(decide(**a)['task_guard_pass'])
    def test_nonfinite_or_negative_results_fail(self):
        for field,value in [('n_cpu_ms',float('nan')),('rss_sum_bytes',-1),('paired_replay_cpu_ratio',float('inf'))]:
            a=self.args();a[field]=value
            with self.assertRaises(ValueError):decide(**a)
        a=self.args();a['n_minus_m_ci']=[.1,-.1]
        with self.assertRaises(ValueError):decide(**a)
    def test_implementation_does_not_call_world_or_candidate_head(self):
        tree=ast.parse(Path(no_online_world.__file__).read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='actor_without_candidate')
        attrs={n.attr for n in ast.walk(fn) if isinstance(n,ast.Attribute)}
        self.assertNotIn('candidate_actor',attrs);self.assertNotIn('predict_all_candidates',attrs)
        self.assertNotIn('world_for_decision',attrs);self.assertIn('preference_actor',attrs)
        self.assertFalse(any(n in sys.modules for n in ('torch','numpy','gppo_world')))

if __name__=='__main__':unittest.main(verbosity=2)
