"""Create the separately authorized B/C-only package, without experimental calls."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'gppo-world-runtime-minimal-paired-20260925'
ATTEMPT = 'independent-task-benefit-20260925-once'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(text, before, after):
    if text.count(before) != 1: raise ValueError('Expected unique source replacement: '+before[:100])
    return text.replace(before, after)

def main():
    if (ROOT/'runs').exists(): raise RuntimeError('Never prepare over an attempted run')
    names = ['runtime_support.py','base_worker.py','runtime_worker.py','no_online_world.py',
             'decision_only_inference.py','process_support.py','safe_observer.py','session_budget.py',
             'sqlite_diagnostics.py','run_diagnostic.py','attribution_contract.py','analyze_runtime_necessity.py',
             'sample-manifest.json','selected-historical-M-cost.json']
    for name in names:
        target = ROOT/name
        if target.exists(): raise RuntimeError('Existing prepared file: '+name)
        target.write_bytes((OLD/name).read_bytes())
    p=ROOT/'attribution_contract.py';t=p.read_text(encoding='utf-8')
    t=replace(t,"ATTEMPT='stage3-runtime-minimal-paired-v2-once'",f"ATTEMPT='{ATTEMPT}'");p.write_text(t,encoding='utf-8')
    p=ROOT/'process_support.py';t=p.read_text(encoding='utf-8')
    t=replace(t,"self.stage=None;self.enter('A')","self.stage=None;self.enter('B')");p.write_text(t,encoding='utf-8')
    p=ROOT/'runtime_worker.py';t=p.read_text(encoding='utf-8')
    t=replace(t,'        kind = permit.get("kind")\n        stage = permit.get("stage")',
              '        kind = permit.get("kind")\n        stage = permit.get("stage")\n        if kind != "candidate" or stage not in ("B", "C"):\n            raise ValueError("Only N in B/C is authorized; A replay is forbidden")')
    t=replace(t,'out = ROOT / "runs/runtime-necessity-v1"','out = ROOT / "runs/task-benefit-v1"')
    t=replace(t,'                value = worker.replay_once()','                raise RuntimeError("A replay is not authorized")');p.write_text(t,encoding='utf-8')
    p=ROOT/'run_diagnostic.py';t=p.read_text(encoding='utf-8')
    start=t.index('def semantic(');end=t.index('def validate_matrix(')
    t=t[:start]+t[end:]
    t=replace(t,'    from prepare_minimal import selected_keys','    from selection_contract import selected_keys')
    t=replace(t,"out=ROOT/'runs/runtime-necessity-v1'","out=ROOT/'runs/task-benefit-v1'")
    t=replace(t,"monitor.enter('A');budget=None","monitor.enter('B');budget=None")
    t=replace(t,"        a=replay_gate(args.authorization,monitor,reservations)",
              "        a=read(request['inputs']['prior_A_result'])\n        if not a['bitwise_reference_candidate_match'] or not a['full_historical_match'] or a['N_mean_cpu_ms']<=10:\n            raise RuntimeError('Prior A evidence/failure identity mismatch')\n        write(out/'prior-A-evidence-reference.json',{'path':request['inputs']['prior_A_result'],'sha256':sha(request['inputs']['prior_A_result']),'cost_failure_preserved':True,'replayed':False})")
    t=replace(t,"        history=read(request['inputs']['historical_verification'])",
              "        history=read(request['inputs']['historical_verification'])\n        prior=read(request['inputs']['prior_A_review'])")
    t=replace(t,"'stage2_plus_this_diagnostic_totals':{k:history['cumulative_A_B_C'][k]+v for k,v in new_totals.items()},",
              "'prior_failed_A_totals':prior['new_resource_totals'],\n            'stage2_plus_failed_A_plus_diagnostic_totals':{k:prior['stage2_plus_this_attempt_totals'][k]+v for k,v in new_totals.items()},")
    t=replace(t,"'historical_resources':history['resources'],'new_resources':monitor.snapshot(),",
              "'historical_resources':history['resources'],'prior_failed_A_resources':prior['new_resources'],\n            'prior_guard_charged_wall_bound':prior['stage2_plus_attempt_guard_charged_wall_bound'],\n            'prior_guard_charged_CPU_bound':prior['stage2_plus_attempt_guard_charged_CPU_bound'],\n            'prior_A_cost_failure_preserved':True,'practical_acceptance_pass':False,'new_resources':monitor.snapshot(),")
    t=replace(t,"'resource_scope':'stage2 plus this diagnostic only; earlier research not reset or refunded',",
              "'resource_scope':'stage2 plus failed A plus separate task diagnostic; no historical cost reset or refund',")
    t=replace(t,"        write(out/'status.json',status)",
              "        status['cumulative_guard_charged_wall_bound']=status['prior_guard_charged_wall_bound']+status['new_resources']['wall_seconds']+request['recovery_shutdown_holdback']['wall_seconds']\n        status['cumulative_guard_charged_CPU_bound']=status['prior_guard_charged_CPU_bound']+status['new_resources']['total_cpu_seconds']+request['recovery_shutdown_holdback']['cpu_seconds']\n        write(out/'status.json',status)")
    t=replace(t,'"""One bound A -> B -> C runtime diagnostic; no training or retry."""',
              '"""Separately authorized task-only B -> C diagnostic; never rerun A."""')
    t=replace(t,'# Frozen runtime diagnostic','# Independent task-benefit diagnostic (prior cost failure retained)')
    t=replace(t,'World-independent contribution is not established by this removal test.',
              'Prior A cost failure remains; practical acceptance is false regardless of these task outcomes. World-independent contribution is not established by this removal test.')
    p.write_text(t,encoding='utf-8')
    old_req=read(OLD/'BUDGET_REQUEST.json');req={k:old_req[k] for k in ['recovery_shutdown_holdback','inputs','history_counters']}
    req['stage_caps']={k:old_req['stage_caps'][k] for k in ['B','C']}
    req['caps']={k:sum(req['stage_caps'][s].get(k,0) for s in ['B','C']) for k in old_req['history_counters']}
    req['caps'].update(wall_seconds=11400,total_process_cpu_seconds=45600,resident_process_rss_sum_bytes=4*1024**3,artifact_bytes=3758096384)
    req.update(attempt=ATTEMPT,approved=True,purpose='independent task-benefit diagnostic; original cost failure permanent',A_replay_authorized=False)
    req['inputs'].update(prior_A_result=str(OLD/'runs/runtime-necessity-v1/A/result.json'),prior_A_review=str(OLD/'runs/runtime-necessity-v1/final-review/decision.json'))
    save(ROOT/'BUDGET_REQUEST.json',req)
    diffs={}
    for name in names:
        if name.endswith('.py') and sha(ROOT/name)!=sha(OLD/name):
            diffs[name]=''.join(difflib.unified_diff((OLD/name).read_text(encoding='utf-8').splitlines(True),(ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='prior/'+name,tofile='new/'+name))
    save(ROOT/'source-differences.json',diffs)
    for p in ROOT.glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    save(ROOT/'preparation-cost.json',{'wall_seconds':time.perf_counter()-START,'process_cpu_seconds':time.process_time(),'env_steps':0,'model_forwards':0,'loads':0,'updates':0})
    print('B/C-only package prepared; no model import or environment execution.')

if __name__=='__main__':main()
