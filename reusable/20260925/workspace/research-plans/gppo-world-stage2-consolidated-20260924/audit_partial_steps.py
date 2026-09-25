import collections,gzip,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/consolidated-v1'
def main():
    start=time.perf_counter();counts=collections.Counter();timings={a:{'cpu':0.,'wall':[]} for a in ['M','R']};last=None;digest=hashlib.sha256();done_count=0
    for stage in ['B','C']:
        n=0
        with gzip.open(OUT/stage/'steps.jsonl.gz','rb') as f:
            for line in f:
                digest.update(line);row=json.loads(line);a=row['arm'];counts[stage+'/environment_steps']+=1;counts[stage+'/'+a]+=1;n+=1
                if stage=='C':
                    timings[a]['cpu']+=row['controller']['cpu'];timings[a]['wall'].append(row['controller']['wall'])
                    done_count+=bool(row['done']);last={k:row.get(k) for k in ['parent','repeat','arm','step','done','counts']}
        print(json.dumps({'scanned_stage':stage,'rows':n}),flush=True)
    status=json.loads((OUT/'status.json').read_text(encoding='utf-8'));stages=status['budget']['stages']
    for stage in ['B','C']:
        assert counts[stage+'/environment_steps']==stages[stage+'/environment_steps']['verified']
        assert counts[stage+'/M']==stages[stage+'/policy_encode']['verified']
        assert counts[stage+'/R']==stages[stage+'/rule_decisions']['verified']
    partial={}
    for a,v in timings.items():
        t=sorted(v['wall']);pos=(len(t)-1)*.95;i=int(pos);q=t[i]+(t[min(i+1,len(t)-1)]-t[i])*(pos-i)
        partial[a]={'decisions':len(t),'mean_cpu_ms':1000*v['cpu']/len(t),'wall_p95_ms':1000*q}
    result={'step_rows_match_verified_budget':True,'counts':dict(counts),'C_terminal_step_rows':done_count,'last_step_identity':last,'partial_controller_cost':partial,'partial_cost_is_not_formal_acceptance':True,'gzip_read_to_eof_and_crc_pass':True,'combined_uncompressed_sha256':digest.hexdigest(),'wall_seconds':time.perf_counter()-start,'no_task_effect_estimate_computed':True}
    dest=ROOT/'final-review/partial-step-integrity.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
