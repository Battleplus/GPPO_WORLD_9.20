import ast
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'torch','numpy','gppo_world'}:raise AssertionError('Native call during preparation test')
sys.meta_path.insert(0,RejectNative())
import runtime_support as support
import run_pipeline as runner
from analyze_formal import analyze

class PreparationTests(unittest.TestCase):
    def test_no_authorization_before_budget_or_worker(self):
        with patch.object(runner,'budget_class',side_effect=AssertionError('Budget created')),patch.object(runner,'OwnedWorker',side_effect=AssertionError('Spawned')):
            with self.assertRaises(PermissionError):runner.main([])
    def test_limits_sum_to_requested_caps(self):
        r=support.read(support.ROOT/'BUDGET_REQUEST.json');limits=runner.limits_for(r)
        for k in ['environment_steps','environment_resets','model_loads','rule_decisions',*runner.MODEL_KEYS]:
            self.assertEqual(sum(v for key,v in limits.items() if key.endswith('/'+k)),r['caps'][k])
    def test_partial_reservation_remains_unknown_not_refunded(self):
        b=Mock();b.reserve.side_effect=[{'stage':'A/policy_encode'},RuntimeError('quota')]
        mon=Mock();mon.stage='A';mon.measure.side_effect=lambda category,fn,*a:fn(*a)
        res=runner.Reservations(b,mon)
        with self.assertRaises(RuntimeError):res.reserve({'policy_encode':1,'world_candidate_batch':1})
        self.assertEqual(len(res.pending),1);res.unknown('stop');b.unknown.assert_called_once();b.complete.assert_not_called()
    def test_exact_boundary_and_no_faster_subset_selection(self):
        row={'decisions':25,'cpu_seconds':.25,'decision_wall_seconds':[.01]*25}
        r={'batches':[dict(row) for _ in range(21)],'equivalence':True,'weights_unchanged':True,'peak_rss':100}
        self.assertTrue(runner.summarize_replay(r)['passed'])
        r['batches'][1]['cpu_seconds']=.50
        self.assertFalse(runner.summarize_replay(r)['passed'])
    def test_lossless_compressed_public_and_truth_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.gz';writer=support.CompressedRecords(path)
            writer.add({'id':'公开任务','null':None,'mask':[True,False],'energy':1.25});writer.close()
            support.verify_compressed(path,writer.digest.hexdigest(),1)
            with self.assertRaises(RuntimeError):support.verify_compressed(path,'0'*64,1)
    def test_shutdown_does_not_suspend_exiting_process(self):
        w=runner.OwnedWorker.__new__(runner.OwnedWorker);w.resume=Mock();w.pause=Mock();w.receive=Mock(return_value={'result':{'ok':1}})
        w.p=Mock();w.p.poll.return_value=0;w.p.returncode=0;w.err=Mock();w.monitor=Mock()
        w.close();w.pause.assert_not_called();self.assertEqual(w.final,{'ok':1})
    def test_cost_failure_stops_before_environment_stages(self):
        req=support.read(support.ROOT/'BUDGET_REQUEST.json')
        with tempfile.TemporaryDirectory() as folder:
            mon=Mock();mon.snapshot.return_value={};b=Mock();b.snapshot.return_value={}
            with patch.object(runner,'ROOT',Path(folder)),patch.object(runner,'authorize',return_value=({}, {},req)),patch.object(runner,'Monitor',return_value=mon),patch.object(runner,'budget_class',return_value=lambda *a,**kw:b),patch.object(runner,'stage_a',return_value=False),patch.object(runner,'stages_environment') as env:
                runner.main(['--authorization','synthetic-unit-test-only.json']);env.assert_not_called()
            status=json.loads((Path(folder)/'runs/consolidated-v1/status.json').read_text())
            self.assertEqual(status['status'],'research_stop_A_cost')
    def test_formal_decisions_and_parent_unit(self):
        rows=[]
        for p in range(2):
            for r in range(3):
                for arm in ['M','R']:
                    rows.append({'parent':f'eval-{p:04d}','repeat':r,'arm':arm,'steps':1,'utility':.1 if arm=='M' else 0.,'physical_on_time':.5,'host_on_time_observed':.5,'energy_used':1.})
        cost={'latencies':{a:[{'cpu':.001,'wall':.001}]*6 for a in ['M','R']},'resources':{'peak_conservative_rss_sum':100}}
        a=analyze(rows,cost,parent_count=2,bootstrap_count=20)
        self.assertEqual(a['parent_count'],2);self.assertTrue(a['decision'].startswith('practical_gain_pass'))
        cost['latencies']['M']=[{'cpu':.02,'wall':.001}]*6
        self.assertEqual(analyze(rows,cost,parent_count=2,bootstrap_count=20)['decision'],'simplify_or_pause_current_candidate')
        with self.assertRaises(ValueError):analyze(rows+[rows[0]],cost,parent_count=2,bootstrap_count=20)
    def test_rss_includes_paused_workers_and_parent(self):
        req=support.read(support.ROOT/'BUDGET_REQUEST.json')
        gib=1024**3
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'peak_rss',return_value=gib),patch.object(runner,'process_cpu',return_value=0):
            mon=runner.Monitor(req,Path(folder))
            paused=Mock();paused.p.poll.return_value=None;paused.p._handle=11;paused.suspended=True
            active=Mock();active.p.poll.return_value=None;active.p._handle=12;active.suspended=False
            mon.children=[paused,active]
            with patch.object(runner,'process_memory',return_value=(gib,gib)) as mem:
                mon.check();self.assertEqual(mon.peak_sum,3*gib);self.assertEqual(mem.call_count,2)
            with patch.object(runner,'process_memory',return_value=(2*gib,2*gib)):
                with self.assertRaisesRegex(RuntimeError,'Combined RSS cap'):mon.check()
    def test_candidate_and_runner_compile_without_native(self):
        for path in support.ROOT.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules);self.assertNotIn('gppo_world',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
