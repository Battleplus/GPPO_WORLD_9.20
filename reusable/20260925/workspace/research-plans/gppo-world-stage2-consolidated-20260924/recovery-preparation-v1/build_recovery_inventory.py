"""Stdlib-only recovery indexing. Never import an environment or a model."""
import collections
import gzip
import hashlib
import json
import sqlite3
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent
RUN = OLD / 'runs/consolidated-v1'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def canonical(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()

def save(name, v):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2)
        f.write('\n')

def key(r):
    return r['parent'], r['repeat'], r['arm']

def planned(tapes):
    return [dict(parent=f'eval-{i:04d}', repeat=r, arm=a, ordinal=i*6+r*2+j,
                 tape_index=i, tape_hash=tapes['hashes'][i],
                 random_key=f'frozen-native-public-history-allocation-v1|eval-{i:04d}|repeat-{r}')
            for i in range(1600) for r in range(3)
            for j, a in enumerate(['M','R'] if (i+r)%2 == 0 else ['R','M'])]

def recompute_episode(rows, initial_hash):
    assert 1 <= len(rows) <= 18
    assert [r['step'] for r in rows] == list(range(len(rows)))
    assert not any(r['done'] for r in rows[:-1]) and rows[-1]['done']
    utility = 0.; previous = {'completed': 0, 'expired': 0}; energy = 36.
    f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    for index, row in enumerate(rows):
        info = row['info']; counts = {k:info['counts'][k] for k in ['completed','expired']}; now = sum(info['energy'].values())
        vector = [f32((max(0, counts['completed']-previous['completed'])-max(0, counts['expired']-previous['expired']))/6), f32(-max(0, energy-now)/36)]
        assert vector == row['vector_reward']
        utility += .99**index*(.4*vector[0]+.2*vector[1])
        previous, energy = counts, now
    end = rows[-1]; info = end['info']; records = info['completion_records']
    assert type(info['terminated']) is bool and type(info['truncated']) is bool
    assert info['terminated'] != info['truncated']
    result = {'steps': len(rows), 'utility': utility, 'energy_used': 36-energy,
              'task_counts': previous,
              'physical_on_time': sum(v['physical_arrival_before_deadline'] is True for v in records.values())/6,
              'host_on_time_observed': sum(v['host_confirmation_before_deadline'] is True for v in records.values())/6,
              'physical_completion_without_host_observed': sum(v['host_confirmation_time'] is None for v in records.values()),
              'terminated': info['terminated'], 'truncated': info['truncated'],
              'episode_end_reason': info.get('episode_end_reason')}
    differences={k:{'recomputed':v,'saved':end['episode'].get(k)} for k,v in result.items() if v!=end['episode'].get(k)}
    assert not differences, f'Saved terminal payload differs: {differences}'
    result.update(parent=end['parent'], repeat=end['repeat'], arm=end['arm'], initial_public_hash=initial_hash)
    return result

def main():
    prior_hashes = read(OLD/'final-review/hashes.json')
    paths = [RUN/'C/steps.jsonl.gz', RUN/'C/episodes.jsonl', RUN/'C/episode-starts.jsonl',
             RUN/'C/scenario-manifest.json', RUN/'budget.sqlite3', RUN/'status.json']
    sources = {str(p): sha(p) for p in paths}
    for p in paths:
        assert sources[str(p)] == prior_hashes[str(p.relative_to(OLD))]
    tapes = read(RUN/'C/scenario-manifest.json')
    assert len(tapes['tapes']) == len(tapes['hashes']) == 1600
    assert [t['seed'] for t in tapes['tapes']] == list(range(924200000,924201600))
    assert all(digest(t)==h for t,h in zip(tapes['tapes'],tapes['hashes']))
    plan = planned(tapes); plan_by = {key(r):r for r in plan}
    starts = [json.loads(s) for s in (RUN/'C/episode-starts.jsonl').read_text(encoding='utf-8').splitlines()]
    summaries = [json.loads(s) for s in (RUN/'C/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    starts_by = {key(r):r for r in starts}; summary_by = {key(r):r for r in summaries}
    assert len(starts_by)==len(starts) and len(summary_by)==len(summaries)
    for s in starts:
        p = plan_by[key(s)]
        assert s['random_key']==p['random_key'] and s['tape_hash']==p['tape_hash']
        assert digest(s['reset']['initial_public'])==s['reset']['initial_hash']
    completed=[]; retained=[]; derived=[]; current=[]; current_key=None; count=0; line_start=1
    costs={'latencies':{'M':[],'R':[]},'worker_parts':{},'communication_events':{}}
    frozen_boundary=[]
    with gzip.open(RUN/'C/steps.jsonl.gz','rb') as f:
        for raw in f:
            count+=1; row=json.loads(raw); k=key(row)
            if current_key is None:current_key=k;line_start=count
            assert k==current_key and row['step']==len(current)
            current.append(row); a=row['arm'];costs['latencies'][a].append(row['controller'])
            for label in ['controller','environment','labels']:
                d=costs['worker_parts'].setdefault(a+'/'+label,{'cpu':0.,'wall':0.})
                for x,v in row[label].items():d[x]+=v
            for event in row['info']['communication_delta']:
                name=f"{a}/{event['link']}/{event['status']}"
                costs['communication_events'][name]=costs['communication_events'].get(name,0)+1
            if row['done']:
                assert k in starts_by
                assert k==key(plan[len(completed)]), 'Executed keys are not the frozen order prefix'
                if k in summary_by:
                    ep={**row['episode'],'parent':k[0],'repeat':k[1],'arm':k[2], 'initial_public_hash':starts_by[k]['reset']['initial_hash']}
                    assert ep==summary_by[k]
                    source='original_episode_summary'
                else:
                    ep=recompute_episode(current,starts_by[k]['reset']['initial_hash'])
                    source='derived_from_complete_steps_not_original_summary'
                    derived.append({'key':list(k),'summary':ep,'source':{'gzip_path':str(RUN/'C/steps.jsonl.gz'),'gzip_sha256':sources[str(RUN/'C/steps.jsonl.gz')],'uncompressed_line_start':line_start,'uncompressed_line_end':count,'method':'recompute every float32 vector reward, discount, terminal task/energy/confirmation fields; equality to saved terminal payload'},'raw_steps':current})
                entry={**plan_by[k],'steps':len(current),'summary_source':source,'line_start':line_start,'line_end':count}
                completed.append(entry);retained.append(ep);current=[];current_key=None
    assert not current and count==56381
    assert len(completed)==len(starts)==4034 and len(summaries)==4033 and len(derived)==1
    assert [key(r) for r in starts]==[key(r) for r in completed]
    assert [key(r) for r in summaries]==[key(r) for r in completed if r['summary_source']=='original_episode_summary']
    for i in range(0,len(completed),2):
        k1,k2=key(completed[i]),key(completed[i+1])
        assert k1[:2]==k2[:2]
        assert starts_by[k1]['reset']['initial_hash']==starts_by[k2]['reset']['initial_hash']
    remaining=plan[len(completed):]; assert len(remaining)==5566
    con=sqlite3.connect((RUN/'budget.sqlite3').as_uri()+'?mode=ro',uri=True)
    stages={r[0]:dict(zip(['limit','reserved','verified','unknown'],r[1:])) for r in con.execute('select stage,limit_amount,reserved,verified,unknown from stages')};con.close()
    assert all(v['unknown']==0 and v['reserved']==v['verified'] for v in stages.values())
    assert count==stages['C/environment_steps']['verified']
    for a,kind in [('M','policy_encode'),('R','rule_decisions')]:
        assert len(costs['latencies'][a])==stages['C/'+kind]['verified']
    # The prior CRC and full audit remain authoritative; this scan adds episode boundaries and restore inputs.
    cost_source={'reuse_crc_audit':str(OLD/'final-review/partial-step-integrity.json'),'sources':sources,'no_effect_or_winner_analysis':True}
    costs['resources']=read(RUN/'status.json')['resources'];costs['provenance']=cost_source
    save('completed-with-summary.json',[r for r in completed if r['summary_source']=='original_episode_summary'])
    save('completed-missing-summary.json',[r for r in completed if r['summary_source']!='original_episode_summary'])
    save('not-executed.json',remaining);save('frozen-matrix.json',plan)
    save('retained-episodes.json',retained);save('retained-cost-inputs.json',costs)
    save('derived-summary-provenance.json',derived)
    save('source-hashes.json',sources);save('historical-budget-readonly.json',stages)
    save('inventory-decision.json',{'retained':4034,'original_summaries':4033,'derived_summaries':1,'remaining':5566,'next_key':remaining[0], 'remaining_by_arm':dict(collections.Counter(r['arm'] for r in remaining)), 'remaining_step_upper_bound':len(remaining)*18,'completed_step_rows':count,'all_started_episodes_terminated':True,'pairs_verified':2017,'source_sha256_unchanged':all(sha(Path(n))==h for n,h in sources.items()),'prior_crc_reused':True,'model_or_environment_imported':False,'partial_effect_computed':False})
    print(json.dumps({'retained':4034,'remaining':5566,'next_key':remaining[0],'max_new_environment_steps':5566*18},ensure_ascii=False))

if __name__=='__main__':main()
