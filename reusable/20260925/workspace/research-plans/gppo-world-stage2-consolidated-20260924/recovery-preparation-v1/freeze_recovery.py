"""Bind prepared recovery without granting authority or creating an attempt."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (ROOT/'runs').exists() and not (ROOT/'authorization-approved.json').exists()
    inventory=read(ROOT/'inventory-decision.json')
    assert inventory['retained']==4034 and inventory['remaining']==5566 and inventory['remaining_step_upper_bound']==100188
    request=read(ROOT/'RECOVERY_BUDGET_REQUEST.json')
    assert request['approved'] is False
    files={}
    for row in read(OLD/'execution-manifest.json')['files']:
        p=Path(row['path']);assert sha(p)==row['sha256'];files[str(p)]=row['sha256']
    for name,value in read(ROOT/'source-hashes.json').items():
        p=Path(name);assert sha(p)==value;files[name]=value
    for name in ['run_recovery.py','runtime_support.py','stage_worker.py','decision_only_inference.py','analyze_formal.py',
                 'recovery_contract.py','safe_observer.py','RECOVERY_PROTOCOL.md','RECOVERY_BUDGET_REQUEST.json',
                 'frozen-matrix.json','completed-with-summary.json','completed-missing-summary.json','not-executed.json',
                 'retained-episodes.json','retained-cost-inputs.json','historical-budget-readonly.json',
                 'derived-summary-provenance.json','inventory-decision.json','continuation-boundary.md']:
        p=ROOT/name;files[str(p)]=sha(p)
        if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),str(p),'exec')
    manifest={'schema':'fixed-original-matrix-recovery-v1','files':[{'path':n,'sha256':h} for n,h in files.items()],
              'original_execution_manifest_sha256':sha(OLD/'execution-manifest.json'),
              'retained':4034,'remaining':5566,'authorized':False,'next_key':inventory['next_key']}
    save(ROOT/'recovery-manifest.json',manifest)
    save(ROOT/'authorization-template.json',{'approved':False,'recovery_manifest_sha256':sha(ROOT/'recovery-manifest.json'),
        'budget_request_sha256':sha(ROOT/'RECOVERY_BUDGET_REQUEST.json'),'caps':request['caps'],
        'user_approval_reference':'','user_approval_text':''})
    print(json.dumps({'bound_files':len(files),'authorized':False,'new_environment_steps':0,'new_model_forwards':0}))

if __name__=='__main__':main()
