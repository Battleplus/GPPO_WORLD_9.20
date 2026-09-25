"""Delta tests only; never imports a model or runs an environment."""
import copy
import importlib.abc
import math
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy','gppo_world'):raise AssertionError('No model/environment imports')
sys.meta_path.insert(0,RejectModels())
from analysis_contract import decide
from analyze_runtime_necessity import analyze
from prepare_minimal import ROOT,read,selected_keys
from attribution_contract import limits
import run_diagnostic as runner

class MinimalTests(unittest.TestCase):
    def args(self):
        return dict(utility_ci=[-.003,.002],rule_ci=[.04,.06],physical_ci=[-.01,.005],host_ci=[-.01,.005],
            loss_family_ci=[[-.004,.003],[-.015,.01],[-.015,.01]],n_cpu_ms=6,n_wall_p95_ms=5,rss_sum_bytes=400000000,
            replay_ratio=.7,formal_decision_ratio=.7,formal_episode_ratio=.7)
    def test_three_decisions_and_not_training_claim(self):
        a=self.args();self.assertEqual(decide(**a)['decision'],'support_simplification_of_frozen_inference')
        a['utility_ci']=[-.014,-.005];a['loss_family_ci'][0]=[-.02,-.003]
        self.assertEqual(decide(**a)['decision'],'uncertain_or_practical_gate_not_met_stop')
        a['utility_ci']=[-.03,-.02];a['loss_family_ci'][0]=[-.04,-.015]
        r=decide(**a);self.assertEqual(r['decision'],'clear_loss_direct_removal_not_supported')
        self.assertFalse(r['world_training_contribution_proven']);self.assertFalse(r['automatic_more_samples'])
    def test_zero_task_mean_not_required_and_boundary_uncertain(self):
        a=self.args();a['physical_ci']=[-.015,-.001]
        self.assertTrue(decide(**a)['task_guard_pass'])
        a['physical_ci']=[-.02,.01]
        self.assertEqual(decide(**a)['decision'],'uncertain_or_practical_gate_not_met_stop')
    def test_total_episode_cost_cannot_hide_more_decisions(self):
        a=self.args();a['formal_episode_ratio']=1.1
        self.assertFalse(decide(**a)['cpu_saving_pass'])
    def test_nonfinite_not_a_pass(self):
        a=self.args();a['utility_ci']=[float('nan'),.01]
        with self.assertRaises(ValueError):decide(**a)
    def test_fixed_selection_independent_of_outcomes(self):
        keys=selected_keys();self.assertEqual(len(keys),800);self.assertEqual(len(set(p for p,r in keys)),800)
        manifest=read(ROOT/'sample-manifest.json');self.assertEqual(keys,[(r['parent'],r['repeat']) for r in manifest])
        req=read(ROOT/'BUDGET_REQUEST.json');baseline=read(req['inputs']['baseline_episodes']);tapes=read(req['inputs']['scenario_manifest'])
        runner.validate_matrix(manifest,baseline,tapes)
        # Replace all outcomes: selection and pairing validation use identity only.
        for r in baseline:r['utility']=12345
        runner.validate_matrix(manifest,baseline,tapes)
        bad=copy.deepcopy(manifest);bad[0]['repeat']=(bad[0]['repeat']+1)%3
        with self.assertRaises(RuntimeError):runner.validate_matrix(bad,baseline,tapes)
    def test_stage_sum_and_no_old_credit(self):
        req=read(ROOT/'BUDGET_REQUEST.json');lim=limits(req)
        self.assertEqual(sum(v for k,v in lim.items() if k.endswith('/environment_steps')),14436)
        self.assertEqual(req['caps']['policy_encode'],16011)
        self.assertEqual(req['caps']['environment_resets'],802)
        self.assertFalse(req['approved']);self.assertFalse(req['historical_balance_reuse'])
    def test_sensitivity_reference_not_new_variance(self):
        d=read(ROOT/'sample-size-evidence.json');self.assertFalse(d['N_minus_M_variance_known'])
        r=next(r for r in d['sensitivity'] if r['independent_parents']==800 and r['assumed_sd_of_parent_paired_difference']==.1)
        expected=NormalDist().cdf(.01*math.sqrt(800)/.1-NormalDist().inv_cdf(.975))
        self.assertAlmostEqual(r['approx_power_NI_at_true_difference_zero'],expected)
        self.assertAlmostEqual(expected,.8074295788138213)
    def test_cost_pairs_cover_exact_selection(self):
        c=read(ROOT/'selected-historical-M-cost.json')
        self.assertEqual({(r['parent'],r['repeat']) for r in c['rows']},set(selected_keys()))
        self.assertEqual(c['steps'],sum(r['steps'] for r in c['rows']))
        self.assertGreater(c['cpu_seconds'],0)
        timings=read(ROOT/'historical-M-timing-manifest.json')
        self.assertEqual(len(timings),c['steps'])
        self.assertEqual(len({(r['parent'],r['repeat'],r['step']) for r in timings}),c['steps'])
        self.assertAlmostEqual(sum(r['cpu'] for r in timings),c['cpu_seconds'])
    def test_partial_matrix_never_analyzed(self):
        selection=read(ROOT/'sample-manifest.json')
        with self.assertRaisesRegex(ValueError,'matrix'):analyze([],[],{},{},{},selection,{})
    def test_complete_synthetic_matrix_costs_and_decision(self):
        selection=read(ROOT/'sample-manifest.json');old=[];new=[]
        for row in selection:
            for arm in ('M','R','N'):
                record={'parent':row['parent'],'repeat':row['repeat'],'arm':arm,'steps':1,
                    'utility':.1 if arm=='R' else .2,'physical_on_time':1.,'host_on_time_observed':1.,'energy_used':1.}
                (new if arm=='N' else old).append(record)
        historical={'rows':[{'parent':r['parent'],'repeat':r['repeat']} for r in selection],'steps':800,'cpu_seconds':8.}
        cost={'latencies':[{'cpu':.006,'wall':.003} for _ in selection]}
        result=analyze(old,new,cost,{'paired_replay_cpu_ratio':.6},{'peak_conservative_rss_sum':1000000},selection,historical)
        self.assertEqual(result['decision'],'support_simplification_of_frozen_inference')
        self.assertEqual(result['parent_count'],800);self.assertEqual(result['reused_episodes'],1600)
        self.assertAlmostEqual(result['N_controller_cost']['per_episode_ratio_to_matched_M'],.6)
        self.assertAlmostEqual(result['summary']['N-M']['utility']['mean_difference'],0)
    def test_no_approval_creates_nothing(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'ROOT',Path(d)):
            with self.assertRaises(PermissionError):runner.main([])
            self.assertEqual(list(Path(d).iterdir()),[])
        self.assertFalse(any(k in sys.modules for k in ('torch','numpy','gppo_world')))

if __name__=='__main__':unittest.main(verbosity=2)
