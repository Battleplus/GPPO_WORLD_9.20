"""Bind executable bytes and existing inputs, without importing native code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PREP=ROOT.parent/'gppo-world-stage2-gate-preparation-20260924'
COST=ROOT.parent/'gppo-world-cost-replay-preparation-20260924'
def sha(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (ROOT/'runs').exists(), 'Never rebind after an attempt exists'
    assert not (ROOT/'authorization-approved.json').exists(), 'Never rebind approved execution'
    request=read(ROOT/'BUDGET_REQUEST.json')
    request['protocol_sha256']=sha(ROOT/'PROTOCOL.md');request['formal_runner_assembled']=True
    save('BUDGET_REQUEST.json',request)
    native=read(PREP/'binding.json');paths=[]
    for name in ['decision_only_inference.py','runtime_support.py','stage_worker.py','run_pipeline.py','analyze_formal.py','PROTOCOL.md','BUDGET_REQUEST.json']:
        paths.append(ROOT/name)
        if name.endswith('.py'):compile((ROOT/name).read_text(encoding='utf-8'),name,'exec')
    for name,h in native['native_python_files'].items():
        p=Path(native['native_root'])/name;assert sha(p)==h;paths.append(p)
    for row in native['native_inputs']:
        p=Path(row['path']);assert sha(p)==row['sha256'];paths.append(p)
    for name,h in native['execution_files'].items():
        p=PREP/name;assert sha(p)==h;paths.append(p)
    for name,h in native['plan_files'].items():
        p=ROOT.parent/'gppo-world-decision-freeze-20260924'/name;assert sha(p)==h;paths.append(p)
    oldhashes=read(COST/'hashes.json');assert sha(COST/'cost_replay.py')==oldhashes['cost_replay.py']
    paths += [PREP/'binding.json',PREP/'runs/technical-gate-v1/decisions.jsonl',COST/'cost_replay.py']
    manifest={'schema':'consolidated-execution-v1','files':[{'path':str(p),'sha256':sha(p)} for p in dict.fromkeys(paths)],
        'native_initializer':'export-only namespace shim, same for both arms','runner':'run_pipeline.py',
        'new_environment_steps':0,'new_model_forwards':0,'authorization_pending':True}
    save('execution-manifest.json',manifest)
    save('authorization-template.json',{'approved':False,'caps':request['caps'],'stage_caps':request['stage_caps'],
        'execution_manifest_sha256':sha(ROOT/'execution-manifest.json'),'budget_request_sha256':sha(ROOT/'BUDGET_REQUEST.json'),
        'user_approval_reference':'','user_approval_text':''})
    print(json.dumps({'bound_files':len(manifest['files']),'budget_authorized':False,'runner_assembled':True,'new_model_forwards':0}))

if __name__=='__main__':main()
