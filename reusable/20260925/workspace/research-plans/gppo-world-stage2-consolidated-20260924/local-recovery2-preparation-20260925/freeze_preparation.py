"""Freeze a reviewable request only. Never creates an approved authorization."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FIRST=ROOT.parent/'recovery-preparation-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (ROOT/'runs').exists(),'Formal run exists; do not refreeze identity'
    prior=read(FIRST/'recovery-manifest.json');files={}
    for row in prior['files']:
        p=Path(row['path']);assert sha(p)==row['sha256'],str(p)
        files[str(p)]=row['sha256']
    histories=read(ROOT/'historical-ledgers-readonly.json')
    for row in histories:
        p=Path(row['path']);assert sha(p)==row['sha256'],str(p)
        files[str(p)]=row['sha256']
    unchanged={}
    for name in ['stage_worker.py','decision_only_inference.py','analyze_formal.py','runtime_support.py','safe_observer.py']:
        assert (ROOT/name).read_bytes()==(FIRST/name).read_bytes(),name
        unchanged[name]=sha(ROOT/name)
    request=read(ROOT/'RECOVERY_BUDGET_REQUEST.json')
    request.update(preparation_ready=True,execution_ready=False,approved=False,
        reason_not_ready='Awaiting explicit one-time authorization including all listed amendments',
        test_result={'passed':14,'environment_steps':0,'model_forwards':0,'formal_attempts_created':0},
        prior_authorized_C_model_load_limit=2)
    request['conditions'][0]='Execution requires explicit acceptance of unresolved historical I/O cause with tested mitigation, plus listed reset/load cap amendments'
    request['conditions'][-1]='Stop on any identity, prefix, accounting, I/O or resource failure; no automatic retry'
    save('RECOVERY_BUDGET_REQUEST.json',request)
    from recovery_contract import validate_resource_envelope,reject_duplicate_or_wrong_order
    history=read(ROOT/'historical-budget-readonly.json');validate_resource_envelope(request,history)
    retained=read(ROOT/'retained-episodes.json');remaining=read(ROOT/'not-executed.json');matrix=read(ROOT/'frozen-matrix.json')
    reject_duplicate_or_wrong_order(retained,remaining,matrix)
    diff=''.join(difflib.unified_diff((FIRST/'run_recovery.py').read_text(encoding='utf-8').splitlines(True),
                 (ROOT/'run_recovery.py').read_text(encoding='utf-8').splitlines(True),
                 fromfile='recovery1/run_recovery.py',tofile='recovery2/run_recovery.py'))
    (ROOT/'orchestrator-diff.patch').write_text(diff,encoding='utf-8')
    for name in ['recovery_contract.py','session_budget.py','sqlite_diagnostics.py']:
        before=(FIRST/name).read_text(encoding='utf-8') if (FIRST/name).exists() else ''
        diff+=''.join(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text(encoding='utf-8').splitlines(True),
                      fromfile='recovery1/'+name,tofile='recovery2/'+name))
    (ROOT/'source-diff.patch').write_text(diff,encoding='utf-8')
    checks={'prior_manifest_files_verified':len(prior['files']),'historical_databases_unchanged':len(histories),
        'unchanged_execution_and_analysis':unchanged,'retained_complete':len(retained),'remaining':len(remaining),
        'matrix':len(matrix),'first_remaining':remaining[0],
        'C_before':{k:v['verified'] for k,v in history.items() if k.startswith('C/')},
        'projected_C_env_steps':history['C/environment_steps']['reserved']+request['caps']['environment_steps'],
        'projected_C_resets':history['C/environment_resets']['reserved']+request['caps']['environment_resets'],
        'cumulative_resource_envelope_pass':True,'partial_effect_analysis':False,'environment_steps':0,'model_forwards':0,
        'training_updates':0,'formal_attempts':0,'execution_authorized':False}
    save('preparation-verification.json',checks)
    # Bind old status/monitor evidence and the raw prefix source without changing them.
    additional=[FIRST/'recovery-manifest.json',FIRST/'final-review/verification.json',
        FIRST/'final-review/terminal-budget-and-resources.json',FIRST/'observer-test-output-final.txt']
    provenance=read(ROOT/'input-provenance.json');raw=Path(provenance['partial_source'])
    assert sha(raw)==provenance['partial_gzip_sha256'];files[str(raw)]=provenance['partial_gzip_sha256']
    for p in additional:files[str(p)]=sha(p)
    excluded={'hashes.json','recovery-manifest.json','authorization-template.json','freeze-output.txt'}
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.name not in excluded:files[str(p)]=sha(p)
    save('recovery-manifest.json',{'schema':'fixed-original-matrix-local-recovery2-v1',
        'attempt':request['proposed_new_attempt'],'files':[{'path':p,'sha256':h} for p,h in sorted(files.items())]})
    save('authorization-template.json',{'approved':False,'attempt':request['proposed_new_attempt'],
        'recovery_manifest_sha256':sha(ROOT/'recovery-manifest.json'),
        'budget_request_sha256':sha(ROOT/'RECOVERY_BUDGET_REQUEST.json'),
        'caps':request['caps'],'accepted_amendments':request['required_authorization_amendments'],
        'user_approval_reference':None,'user_approval_text':None})
    save('hashes.json',{'schema':'recovery2-output-hashes-v1','files':[
        {'path':p.name,'sha256':sha(p)} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in ('hashes.json','freeze-output.txt')]})
    print(json.dumps({'frozen_files':len(files),'retained':len(retained),'remaining':len(remaining),
          'historical_databases_unchanged':len(histories),'execution_authorized':False,'env_step':0},ensure_ascii=False))

if __name__=='__main__':main()
