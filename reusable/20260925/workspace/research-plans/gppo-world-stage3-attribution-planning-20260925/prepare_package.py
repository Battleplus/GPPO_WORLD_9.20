"""Planning artifacts only: JSON and hashes; never imports experiment runtime."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PLAN=ROOT.parent
BASE=PLAN/'gppo-world-stage2-consolidated-20260924'
FINAL=BASE/'local-recovery2-preparation-20260925'
ATTEMPT='stage3-runtime-necessity-v1-once'
KEYS=['environment_steps','environment_resets','policy_encode','actor_readout','world_candidate_batch',
      'rule_decisions','model_loads','optimizer_updates','world_updates','offline_updates']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,d):
    (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def prepare():
    def cap(env,resets,enc,world,loads,wall,cpu,gib):
        return {**dict.fromkeys(KEYS,0),'environment_steps':env,'environment_resets':resets,
            'policy_encode':enc,'actor_readout':enc,'world_candidate_batch':world,'model_loads':loads,
            'wall_seconds':wall,'cpu_seconds':cpu,'artifact_bytes':gib*1024**3}
    stages={'A':cap(0,0,1575,525,3,1200,3600,1),'B':cap(36,2,36,0,1,3600,14400,1),
            'C':cap(86400,4800,86400,0,1,64800,259200,12)}
    caps={k:sum(s[k] for s in stages.values()) for k in KEYS}
    caps.update(wall_seconds=69600,total_process_cpu_seconds=277200,
                resident_process_rss_sum_bytes=4*1024**3,artifact_bytes=14*1024**3)
    inputs={'baseline_episodes':str(FINAL/'runs/consolidated-v1/C/combined-episodes.json'),
        'baseline_analysis':str(FINAL/'runs/consolidated-v1/C/analysis.json'),
        'historical_verification':str(FINAL/'final-review/verification.json'),
        'historical_status':str(FINAL/'runs/consolidated-v1/status.json'),
        'scenario_manifest':str(BASE/'runs/consolidated-v1/C/scenario-manifest.json'),
        'replay_decisions':str(PLAN/'gppo-world-stage2-gate-preparation-20260924/runs/technical-gate-v1/decisions.jsonl')}
    history=read(inputs['historical_verification'])
    save('BUDGET_REQUEST.json',{'schema':'frozen-runtime-diagnostic-budget-v1','attempt':ATTEMPT,
        'approved':False,'purpose':'frozen runtime simplification; not independent training attribution',
        'caps':caps,'stage_caps':stages,'recovery_shutdown_holdback':{'wall_seconds':30,'cpu_seconds':600},
        'shutdown_policy':'inside each stage cap; stop new work before reserve; no credit transfer',
        'inputs':inputs,'history_counters':KEYS,'stage2_historical_consumed':history['cumulative_A_B_C'],
        'historical_cost':history['resources'],'historical_balances_authorize_nothing':True,
        'sequence':['A numerical and absolute cost gates','B native interface gates','C full fixed matrix and analysis'],
        'automatic_stage_progression_after_pass':True,'automatic_retry':False,'training_authorized':False,
        'server_authorized':False,'git_publication_authorized':False,'further_experiments_authorized':False,
        'model_loading_note':'five new loads; inactive world weights retained for identity and immutable-weight audit',
        'resource_definition':'full parent CPU plus every worker; RSS sum of parent and all resident workers, including suspended; no historical cost refunds'})
    old=read(inputs['baseline_episodes']);by={(r['parent'],r['repeat'],r['arm']):r for r in old}
    if len(old)!=9600 or len(by)!=9600:raise ValueError('Historical matrix not complete')
    tapes=read(inputs['scenario_manifest']);rows=[]
    for p in range(1600):
        for r in range(3):
            name=f'eval-{p:04d}';initial=by[(name,r,'M')]['initial_public_hash']
            if initial!=by[(name,r,'R')]['initial_public_hash']:raise ValueError('Historical pair mismatch')
            rows.append({'parent':name,'repeat':r,'arm':'N','tape_index':p,'tape_hash':tapes['hashes'][p],
                'initial_public_hash':initial,'exogenous_key':f'frozen-native-public-history-allocation-v1|{name}|repeat-{r}'})
    save('sample-manifest.json',rows)
    save('authorization-template.json',{'approved':False,'attempt':ATTEMPT,'caps':caps,'stage_caps':stages,
        'manifest_sha256':'FILLED_AFTER_FREEZE','budget_sha256':sha(ROOT/'BUDGET_REQUEST.json'),
        'user_approval_text':None,'user_approval_reference':None})

def freeze():
    r=read(ROOT/'BUDGET_REQUEST.json');binding=read(PLAN/'gppo-world-stage2-gate-preparation-20260924/binding.json')
    # Bind all local implementation/protocol inputs; mutable execution and audit outputs excluded.
    excluded={'execution-manifest.json','authorization-template.json','zero-step-validation.json','hashes.json'}
    files={p.resolve() for p in ROOT.iterdir() if p.is_file() and p.name not in excluded and not p.name.startswith('test-output')}
    files.update(Path(v).resolve() for v in r['inputs'].values())
    files.add((PLAN/'gppo-world-stage2-gate-preparation-20260924/binding.json').resolve())
    files.update((Path(binding['native_root'])/p).resolve() for p in binding['native_python_files'])
    files.update(Path(binding[k]).resolve() for k in ('checkpoint','config'))
    for name in ['public_controller.py','run_gate.py']:
        files.add((PLAN/'gppo-world-stage2-gate-preparation-20260924'/name).resolve())
    files.add((PLAN/'gppo-world-cost-replay-preparation-20260924/cost_replay.py').resolve())
    files.update(p.resolve() for p in (ROOT/'remote').rglob('*') if p.is_file())
    save('execution-manifest.json',{'attempt':ATTEMPT,'scope':'all execution sources and frozen comparison inputs',
        'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]})
    a=read(ROOT/'authorization-template.json');a['manifest_sha256']=sha(ROOT/'execution-manifest.json')
    a['budget_sha256']=sha(ROOT/'BUDGET_REQUEST.json');save('authorization-template.json',a)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:prepare()
