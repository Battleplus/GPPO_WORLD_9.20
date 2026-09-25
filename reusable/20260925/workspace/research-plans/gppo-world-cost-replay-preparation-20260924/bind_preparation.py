"""Bind only bytes/JSON. This script does not import native/model code."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PREP=ROOT.parent/'gppo-world-stage2-gate-preparation-20260924'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    native=read(PREP/'binding.json')
    for name,h in native['execution_files'].items():assert sha(PREP/name)==h
    raw=read(PREP/'technical-gate-review-v1/input-hashes.json')
    for name,h in raw.items():assert sha(PREP/name)==h
    inputs=[ROOT/'cost_replay.py',ROOT/'protocol.md',PREP/'public_controller.py',PREP/'run_gate.py',PREP/'binding.json',PREP/'runs/technical-gate-v1/decisions.jsonl']
    for name,h in native['native_python_files'].items():
        path=Path(native['native_root'])/name;assert sha(path)==h;inputs.append(path)
    for row in native['native_inputs']:
        path=Path(row['path']);assert sha(path)==row['sha256'];inputs.append(path)
    write('binding.json',{'schema':'fixed-trace-cost-diagnostic-v1',
        'decisions':str(PREP/'runs/technical-gate-v1/decisions.jsonl'),
        'budget_executor':str(Path(native['native_root'])/'gppo_world/budget_executor.py'),
        'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs]})
    write('authorization-template.json',{'approved':False,'stage':'fixed_trace_cost_diagnostic',
        'limits':{'environment_steps':0,'policy_encode':525,'world_candidate_batch':525,'actor_readout':525,'rule_decisions':651},
        'updates':0,'binding_sha256':sha(ROOT/'binding.json'),'user_approval_reference':'','user_approval_text':'',
        'formal_stage_authorized':False,'wall_seconds_max':600,'rss_bytes_max':4*1024**3,'artifact_bytes_max':512*1024**2})
    print(json.dumps({'bound_inputs':len(inputs),'approved':False,'model_forwards':0,'environment_steps':0}))

if __name__=='__main__':main()
