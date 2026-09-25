"""Freeze revised planning package; no experimental calls or original writes."""
import ast
import difflib
import json
from pathlib import Path
from prepare_minimal import ROOT,OLD,ATTEMPT,read,save,sha

def main():
    old_manifest=read(OLD/'execution-manifest.json')
    files={}
    for row in old_manifest['files']:
        p=Path(row['path'])
        if p.is_relative_to(OLD):continue
        # Reuse pinned identity; only equality check, no dynamic revalidation.
        if sha(p)!=row['sha256']:raise RuntimeError('Reused identity changed: '+str(p))
        files[str(p)]=row
    reused=[]
    for name in ['base_worker.py','decision_only_inference.py','safe_observer.py','session_budget.py',
        'sqlite_diagnostics.py','runtime_support.py','runtime_worker.py','no_online_world.py','process_support.py']:
        matched=sha(ROOT/name)==sha(OLD/name)
        if not matched:raise RuntimeError('Unexpected change to reused component: '+name)
        reused.append({'name':name,'sha256':sha(ROOT/name),'old_sha256':sha(OLD/name),'byte_identical':matched})
    for name in ['run_diagnostic.py','attribution_contract.py','analysis_contract.py','analyze_runtime_necessity.py']:
        diff=''.join(difflib.unified_diff((OLD/name).read_text(encoding='utf-8').splitlines(True),
            (ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='old/'+name,tofile='new/'+name))
        (ROOT/(name+'.diff')).write_text(diff,encoding='utf-8')
    for p in ROOT.glob('*.py'):compile(ast.parse(p.read_text(encoding='utf-8')),str(p),'exec')
    ledger_checks=[]
    for row in read(OLD/'stage2-budget-readonly.json')['segments']:
        actual=sha(row['path'])
        if actual!=row['sha256']:raise RuntimeError('Historical budget changed')
        ledger_checks.append({'path':row['path'],'sha256':actual,'unchanged':True})
    req=read(ROOT/'BUDGET_REQUEST.json')
    if req['approved'] or (ROOT/'runs').exists():raise RuntimeError('Preparation must not authorize or start a run')
    # Include new inputs and prior evidence supporting direct reuse.
    for p in [*map(Path,req['inputs'].values()),OLD/'zero-step-validation.json',OLD/'execution-manifest.json',
              OLD/'BUDGET_REQUEST.json',OLD/'stage2-budget-readonly.json']:
        files[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    excluded={'execution-manifest.json','authorization-template.json','hashes.json','validation.json'}
    for p in ROOT.iterdir():
        if p.is_file() and p.name not in excluded:
            files[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    save('execution-manifest.json',{'attempt':ATTEMPT,'files':sorted(files.values(),key=lambda r:r['path']),
        'previous_4800_plan_authorized':False,'previous_plan_unchanged':True})
    a=read(ROOT/'authorization-template.json');a['manifest_sha256']=sha(ROOT/'execution-manifest.json')
    a['budget_sha256']=sha(ROOT/'BUDGET_REQUEST.json');save('authorization-template.json',a)
    save('validation.json',{'status':'minimum_fixed_plan_prepared_not_executed','reused_components':reused,
        'prior_interface_and_ledger_tests_reused_not_rerun':True,'old_ledgers_unchanged':ledger_checks,
        'new_environment_steps':0,'new_environment_resets':0,'new_model_forwards':0,'checkpoint_deserializations':0,
        'optimizer_updates':0,'world_updates':0,'offline_updates':0,'new_experimental_attempts':0,
        'formal_sqlite_created':False,'selected_parent_count':800,'new_formal_episode_plan':800,
        'manifest_sha256':sha(ROOT/'execution-manifest.json'),'budget_sha256':sha(ROOT/'BUDGET_REQUEST.json'),
        'dynamic_N_interfaces_not_yet_verified':True,'incremental_tests':'test-summary.json'})
    save('hashes.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='hashes.json']})
    print(json.dumps({'status':'prepared_no_execution','frozen_files':len(files),'selected_parents':800,
        'new_environment_steps':0,'new_model_forwards':0,'budget_sha256':sha(ROOT/'BUDGET_REQUEST.json')}))

if __name__=='__main__':main()
