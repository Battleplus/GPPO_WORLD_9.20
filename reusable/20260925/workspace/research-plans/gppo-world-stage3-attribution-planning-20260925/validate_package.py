"""Zero experimental operations; standard-library tests and read-only hashes."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'gppo-world-stage2-consolidated-20260924/local-recovery2-preparation-20260925'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    files=list(ROOT.glob('*.py'))
    for f in files:compile(ast.parse(f.read_text(encoding='utf-8')),str(f),'exec')
    command=[sys.executable,'-B','-X','utf8','-m','unittest','-v','test_plan','test_runner','test_runtime_worker_static']
    test=subprocess.run(command,cwd=ROOT,text=True,encoding='utf-8',capture_output=True,timeout=60)
    (ROOT/'test-output.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Zero-step tests failed; see test-output.txt')
    reused={}
    for name,oldname in [('base_worker.py','stage_worker.py'),('decision_only_inference.py','decision_only_inference.py'),
        ('safe_observer.py','safe_observer.py'),('session_budget.py','session_budget.py'),('sqlite_diagnostics.py','sqlite_diagnostics.py')]:
        reused[name]={'source':str(OLD/oldname),'sha256':sha(ROOT/name),'byte_identical':sha(ROOT/name)==sha(OLD/oldname)}
    if not all(x['byte_identical'] for x in reused.values()):raise RuntimeError('Unexpected reused source change')
    old_ledgers=read(ROOT/'stage2-budget-readonly.json')['segments']
    unchanged=[{'path':r['path'],'sha256':sha(Path(r['path'])),'matches_readonly_audit':sha(Path(r['path']))==r['sha256']} for r in old_ledgers]
    if not all(r['matches_readonly_audit'] for r in unchanged):raise RuntimeError('Historical ledger changed')
    manifest=read(ROOT/'execution-manifest.json')
    errors=[r['path'] for r in manifest['files'] if sha(Path(r['path']))!=r['sha256']]
    if errors:raise RuntimeError('Frozen file mismatch: '+str(errors))
    template=read(ROOT/'authorization-template.json')
    if template['approved'] is not False:raise RuntimeError('Template must not authorize execution')
    if (ROOT/'runs').exists():raise RuntimeError('Formal run directory unexpectedly exists during preparation')
    binding=read(ROOT.parent/'gppo-world-stage2-gate-preparation-20260924/binding.json')
    technical_collision=sorted({925310000,925310001}&set(binding['historical_seed_ids']))
    if technical_collision:raise RuntimeError('Technical parent seed collision')
    summary={'status':'zero_step_preparation_pass','python_sources_compiled':len(files),
        'test_command':command,'test_output_sha256':sha(ROOT/'test-output.txt'),
        'experimental_environment_steps':0,'model_forwards':0,'environment_resets':0,'checkpoint_loads':0,
        'optimizer_updates':0,'world_updates':0,'offline_updates':0,'new_formal_attempts':0,
        'formal_budget_created':False,'reused_sources':reused,'old_ledgers_unchanged':unchanged,
        'technical_seeds_vs_bound_historical_seed_index_collision':technical_collision,
        'frozen_files_verified':len(manifest['files']),'dynamic_A_B_pass_claimed':False,
        'manifest_sha256':sha(ROOT/'execution-manifest.json'),'budget_sha256':sha(ROOT/'BUDGET_REQUEST.json')}
    save('zero-step-validation.json',summary)
    save('hashes.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.name!='hashes.json' and '__pycache__' not in p.parts]})
    print(json.dumps({'status':summary['status'],'frozen_files_verified':summary['frozen_files_verified'],
        'environment_steps':0,'model_forwards':0,'budget_created':False}))

if __name__=='__main__':main()
