"""Freeze preparation identities using bytes/JSON only; never imports native code."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PLAN=ROOT.parent/'gppo-world-decision-freeze-20260924'
ART=Path(r'E:\Z博士\migration-artifacts')
FAIR=ART/'event-trigger-aware-gppo-fair-replication-20260919-v1'
WORK=Path(r'E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt')
MANIFEST=WORK/'runs/finite-communication-ack-lease-fix-20260920/model-rule-stage1-execution-package-v1/native-source-manifest.json'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(name,data):(ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    original=read(PLAN/'hashes.json')
    for name,expected in original.items():
        assert sha(PLAN/name)==expected, f'Frozen plan mismatch: {name}'
    manifest=read(MANIFEST);src=Path(manifest['source_root'])
    native={name:row['sha256'] for name,row in manifest['files'].items()
            if name.startswith('gppo_world/') and name.endswith('.py')}
    for name,expected in native.items():assert sha(src/name)==expected,name
    training=FAIR/'tapes/training-tapes.json'
    protocol=ART/'replan-value-feature-collection-20260919-v1/protocol.json'
    tape=read(training);old=read(protocol)
    occupied=set()
    def seeds(value):
        if isinstance(value,dict):
            if isinstance(value.get('seed'),int):occupied.add(value['seed'])
            for v in value.values():seeds(v)
        elif isinstance(value,list):
            for v in value:seeds(v)
    seeds(tape)
    occupied.update(range(old['base_seed'],old['base_seed']+old['parent_count']))
    planned=read(PLAN/'planned-parent-manifest.json')['parents']
    assert not occupied.intersection(row['seed'] for row in planned)
    config=ART/'preference-weighted-wm-event-cpu-20260917/final/run/training/seed-1101/WD/resolved-config.json'
    checkpoint=FAIR/'training/seed-1101/P_train/last-recovery.pt'
    assert sha(checkpoint)=='bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a'
    binding={'schema':'technical-gate-preparation-binding-v1','native_root':str(src),
        'config':str(config),'checkpoint':str(checkpoint),
        'execution_files':{name:sha(ROOT/name) for name in ['public_controller.py','run_gate.py']},
        'plan_files':{name:sha(PLAN/name) for name in ['report.md','protocol.md','decision.json','evidence-index.json','planned-parent-manifest.json']},
        'native_python_files':native,
        'native_inputs':[{'path':str(p),'sha256':sha(p)} for p in [config,checkpoint,MANIFEST,training,protocol]],
        'historical_seed_ids':sorted(occupied),
        'seed_identity_provenance':{'observed_training_metadata':str(training),'reserved_protocol_range':str(protocol),
            'scope':'Permitted training and reserved collection identities only; no heldout outcomes read, no global unseen-data uniqueness claim.'},
        'planned_seed_collisions':0,'approved':False}
    write('binding.json',binding)
    write('authorization-template.json',{'approved':False,'stage':'technical_gate',
        'limits':{'environment_steps':72,'policy_encode':36,'world_candidate_batch':36,'actor_readout':36},
        'updates':0,'binding_sha256':sha(ROOT/'binding.json'),'user_approval_reference':'','user_approval_text':'',
        'output_directory':str(ROOT/'runs/technical-gate-v1'),'formal_stage_authorized':False,
        'wall_seconds_max':3600,'rss_bytes_max':4*1024**3,'artifact_bytes_max':1024**3})
    print(json.dumps({'status':'bound_not_authorized','native_python_files':len(native),
                      'planned_parents':len(planned),'seed_collisions':0,'environment_steps':0,'model_forwards':0}))

if __name__=='__main__':main()
