import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_diagnostic as runner
import process_support
from attribution_contract import limits
from analysis_contract import decide
from selection_contract import selected_keys
from analyze_runtime_necessity import analyze

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def method(path,cls,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    return ast.dump(next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)

class StaticTests(unittest.TestCase):
    def args(self):
        return dict(utility_ci=[-.005,.005],rule_ci=[.02,.04],physical_ci=[-.01,.01],host_ci=[-.01,.01],
                    loss_family_ci=[[-.008,.008],[-.015,.015],[-.015,.015]],n_cpu_ms=12,n_wall_p95_ms=4,
                    rss_sum_bytes=100000000,replay_ratio=.7932,formal_decision_ratio=.9,formal_episode_ratio=.9)
    def test_task_pass_never_means_cost_pass(self):
        x=decide(**self.args())
        self.assertEqual(x['decision'],'support_preparing_simplification_task_basis_only_cost_not_qualified')
        self.assertFalse(x['practical_acceptance_pass']);self.assertFalse(x['absolute_cost_pass_current_C_only'])
        a=self.args();a.update(n_cpu_ms=5,formal_decision_ratio=.5,formal_episode_ratio=.5)
        x=decide(**a);self.assertFalse(x['practical_acceptance_pass']);self.assertTrue(x['prior_A_cost_failure_preserved'])
    def test_loss_and_uncertainty_remain_distinct(self):
        a=self.args();a.update(utility_ci=[-.03,-.02],loss_family_ci=[[-.035,-.015],[-.015,.015],[-.015,.015]])
        self.assertEqual(decide(**a)['decision'],'clear_loss_direct_removal_not_supported')
        a=self.args();a['utility_ci']=[-.012,.004]
        self.assertEqual(decide(**a)['decision'],'uncertain_pause_no_more_samples')
    def test_strict_threshold_and_task_guard(self):
        a=self.args();a['utility_ci']=[-.01,.02]
        self.assertFalse(decide(**a)['utility_retention_pass'])
        a=self.args();a['host_ci']=[-.02,.03]
        self.assertFalse(decide(**a)['task_benefit_retention_pass'])
    def test_same_selection_and_full_pair_identity(self):
        req=read(ROOT/'BUDGET_REQUEST.json');rows=read(ROOT/'sample-manifest.json')
        self.assertEqual([(r['parent'],r['repeat']) for r in rows],selected_keys())
        runner.validate_matrix(rows,read(Path(req['inputs']['baseline_episodes'])),read(Path(req['inputs']['scenario_manifest'])))
        self.assertEqual((ROOT/'sample-manifest.json').read_bytes(),(ROOT.parent/'gppo-world-runtime-minimal-paired-20260925/sample-manifest.json').read_bytes())
    def test_budget_no_A_no_world_or_rule(self):
        req=read(ROOT/'BUDGET_REQUEST.json');lim=limits(req)
        self.assertEqual(set(req['stage_caps']),{'B','C'})
        self.assertEqual(sum(v for k,v in lim.items() if k.endswith('/environment_steps')),14436)
        self.assertEqual(lim['B/model_loads'],1);self.assertEqual(lim['C/model_loads'],1)
        for k,v in lim.items():
            if k.endswith(('/world_candidate_batch','/rule_decisions','/optimizer_updates','/world_updates','/offline_updates')):self.assertEqual(v,0)
    def test_counter_mismatch_cannot_be_verified(self):
        counts=dict.fromkeys(runner.COUNTERS,0);after=dict(counts,world_candidate_batch=1)
        with self.assertRaisesRegex(RuntimeError,'counter mismatch'):runner.exact_delta(counts,after,{})
    def test_n_inference_and_environment_methods_unchanged(self):
        old=ROOT.parent/'gppo-world-runtime-minimal-paired-20260925'
        for name in ['choose','_select_action','one_step','reset','_step_diagnostic','_install_inactive_world_guard']:
            self.assertEqual(method(ROOT/'runtime_worker.py','Worker',name),method(old/'runtime_worker.py','Worker',name))
        for name in ['base_worker.py','no_online_world.py','decision_only_inference.py','safe_observer.py','session_budget.py']:
            self.assertEqual((ROOT/name).read_bytes(),(old/name).read_bytes())
    def test_no_A_execution_path(self):
        self.assertFalse(hasattr(runner,'replay_gate'))
        worker=(ROOT/'runtime_worker.py').read_text(encoding='utf-8')
        self.assertIn('raise RuntimeError("A replay is not authorized")',worker)
        self.assertIn('Only N in B/C is authorized',worker)
    def test_preparation_reserve_is_charged_to_B_not_borrowed_from_C(self):
        monitor=object.__new__(process_support.Monitor)
        monitor.request=read(ROOT/'BUDGET_REQUEST.json');monitor.children=[];monitor.peak_sum=0
        monitor.started=0;monitor.stage_start=0;monitor.stage_cpu_start=0;monitor.total_cpu=lambda:1750
        with patch.object(process_support.time,'perf_counter',return_value=100),patch.object(process_support,'peak_rss',return_value=0):
            monitor.stage='B'
            with self.assertRaisesRegex(RuntimeError,'CPU cap'):monitor.check()
            monitor.stage='C';monitor.check()
    def test_partial_matrix_never_analyzed(self):
        with self.assertRaisesRegex(ValueError,'matrix'):
            analyze([],[],{},{},{},read(ROOT/'sample-manifest.json'),{})
    def test_authorization_needed_before_outputs(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'ROOT',Path(folder)):
            with self.assertRaises(PermissionError):runner.main([])
            self.assertEqual(list(Path(folder).iterdir()),[])
    def test_no_model_or_environment_imports(self):
        self.assertFalse(any(k in sys.modules for k in ['torch','numpy','gppo_world']))

if __name__=='__main__':unittest.main(verbosity=2)
