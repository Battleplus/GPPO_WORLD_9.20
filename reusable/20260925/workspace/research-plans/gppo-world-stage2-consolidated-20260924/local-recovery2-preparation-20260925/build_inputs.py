"""Prepare old evidence and accounting only; no experiment imports or effects."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FIRST=ROOT.parent/'recovery-preparation-v1'
RUN=FIRST/'runs/consolidated-v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def key(r):return r['parent'],r['repeat'],r['arm']

def main():
    assert not (ROOT/'runs').exists()
    retained=read(FIRST/'retained-episodes.json')
    with (RUN/'C/episodes.jsonl').open(encoding='utf-8') as f:retained.extend(json.loads(line) for line in f)
    matrix=read(FIRST/'frozen-matrix.json');remaining=matrix[len(retained):]
    assert len(retained)==8637 and len(remaining)==963
    assert [key(r) for r in retained+remaining]==[key(r) for r in matrix]
    assert key(remaining[0])==('eval-1439',1,'R')
    partial_key=key(remaining[0]);partial_reset=None
    with (RUN/'C/episode-starts.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if key(row)==partial_key:partial_reset=row
    assert partial_reset is not None
    old=read(FIRST/'retained-cost-inputs.json')
    current={'latencies':{'M':[],'R':[]},'worker_parts':{},'communication_events':{}}
    prefix=[];h=hashlib.sha256();count=0
    with gzip.open(RUN/'C/steps.jsonl.gz','rb') as f:
        for line in f:
            h.update(line);count+=1;row=json.loads(line);arm=row['arm']
            current['latencies'][arm].append(row['controller'])
            for label in ['controller','environment','labels']:
                values=current['worker_parts'].setdefault(arm+'/'+label,{'cpu':0.,'wall':0.})
                for field,value in row[label].items():values[field]+=value
            for event in row['info']['communication_delta']:
                name=f"{arm}/{event['link']}/{event['status']}"
                current['communication_events'][name]=current['communication_events'].get(name,0)+1
            if key(row)==partial_key:prefix.append(row)
    previous=read(FIRST/'final-review/verification.json')
    assert count==63830 and h.hexdigest()==previous['uncompressed_sha256']
    assert len(prefix)==7 and [r['step'] for r in prefix]==list(range(7)) and not any(r['done'] for r in prefix)
    assert len(current['latencies']['M'])==30075 and len(current['latencies']['R'])==33755
    for arm in ['M','R']:old['latencies'][arm].extend(current['latencies'][arm])
    for name,value in current['worker_parts'].items():
        target=old['worker_parts'].setdefault(name,{'cpu':0.,'wall':0.})
        for field,n in value.items():target[field]+=n
    for name,n in current['communication_events'].items():old['communication_events'][name]=old['communication_events'].get(name,0)+n
    old['partial_R_prefix_included']=7
    old['latency_merge_rule']='Keep original partial R prefix timings; future 7-step replay timings are separate overhead, never select faster measurements.'
    status=read(FIRST/'final-review/terminal-budget-and-resources.json')
    old['resources']=status['resources']
    stages={}
    for name,row in status['cumulative_stages'].items():
        stages[name]={k:row[k] for k in ['reserved','verified','unknown','pending']}
        stages[name]['limit']=row['authorized_limit_including_explicit_extension']
    request=read(ROOT/'budget-draft.json')
    request.update(schema='local-original-matrix-recovery2-request-v1',not_an_authorization_request_yet=False,
        execution_ready=False,preparation_ready=False,
        required_authorization_amendments={'accept_unresolved_historical_io_cause_with_tested_mitigation':True,
            'C_reset_cumulative_limit':9601,'C_model_load_cumulative_limit':3,'total_model_load_cumulative_limit':6},
        historical_resources={'total_wall_seconds':status['resources']['wall_seconds'],
            'total_cpu_seconds':status['resources']['total_cpu_seconds'],
            'C_wall_seconds':9554.473701900002+status['resources']['recovery_segment']['wall_seconds'],
            'C_cpu_seconds':8622.4375+status['resources']['recovery_segment']['total_cpu_seconds'],
            'peak_conservative_rss_sum':status['resources']['peak_conservative_rss_sum'],
            'artifact_bytes_conservative':1260904191+sum(p.stat().st_size for p in FIRST.rglob('*') if p.is_file())},
        historical_shutdown_holdback={'wall_seconds':60,'cpu_seconds':1200},
        recovery_shutdown_holdback={'wall_seconds':30,'cpu_seconds':600},
        stage_caps={'C':{'environment_steps':17334,'each_model_forward':8658,'rule_decisions':8676,
                         'wall_seconds':28800,'cpu_seconds':115200}},
        original_C_limits={'environment_steps':172800,'policy_encode':86400,'world_candidate_batch':86400,
            'actor_readout':86400,'rule_decisions':86400,'environment_resets':9600,'model_loads':1,
            'wall_seconds':129600,'total_process_cpu_seconds':518400,'artifact_bytes':40*1024**3})
    save('RECOVERY_BUDGET_REQUEST.json',request)
    save('historical-budget-readonly.json',stages)
    save('retained-episodes.json',retained);save('frozen-matrix.json',matrix);save('not-executed.json',remaining)
    save('retained-cost-inputs.json',old);save('partial-prefix.json',{'key':list(partial_key),'reset':partial_reset,'steps':prefix})
    save('input-provenance.json',{'partial_source':str(RUN/'C/steps.jsonl.gz'),
         'partial_gzip_sha256':sha(RUN/'C/steps.jsonl.gz'),'uncompressed_sha256':h.hexdigest(),
         'last_seven_records':True,'record_count':count,'all_previous_inputs_bound_by':str(FIRST/'recovery-manifest.json'),
         'historical_episode_counts':[4034,4603],'retained_complete':8637,'remaining':963,
         'new_environment_steps':0,'new_model_forwards':0,'updates':0,'partial_effect_computed':False})
    print(json.dumps({'retained':len(retained),'remaining':len(remaining),'prefix_steps':len(prefix),'cost_timing_rows':{k:len(v) for k,v in old['latencies'].items()},'new_environment_steps':0}))

if __name__=='__main__':main()
