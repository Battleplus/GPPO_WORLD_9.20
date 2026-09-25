"""Read saved technical-gate evidence. No native imports, model calls or steps."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct

from public_controller import PublicMemory

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/technical-gate-v1'
OUT=ROOT/'technical-gate-review-v1'

def read(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
def lines(name):return [json.loads(x) for x in (RUN/name).read_text(encoding='utf-8').splitlines()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def percentile(values,q):
    v=sorted(values);k=(len(v)-1)*q;lo=math.floor(k);hi=math.ceil(k)
    return v[lo]+(v[hi]-v[lo])*(k-lo)
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=False)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in RUN.iterdir() if p.is_file()}
    status=read('run-status.json');runtime=read('runtime-start.json');binding=read('binding.json')
    episodes=lines('episodes.jsonl');steps=lines('steps.jsonl');decisions=lines('decisions.jsonl');calls=lines('calls.jsonl')
    assert status['status']=='technical_gate_passed' and len(episodes)==4
    assert status['native_model_state_unchanged'] and status['updates']==0
    expected={('gate-000',0,'M'),('gate-000',0,'R'),('gate-001',0,'M'),('gate-001',0,'R')}
    key=lambda r:(r['parent'],r['repeat'],r['arm'])
    assert {key(r) for r in episodes}==expected
    dmap={(*key(r),r['step']):r for r in decisions}
    assert len(dmap)==len(decisions)==len(steps)==56
    for parent in ['gate-000','gate-001']:
        assert len({r['initial_public_sha256'] for r in episodes if r['parent']==parent})==1
    reservations={}
    with sqlite3.connect((RUN/'budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        assert c.execute('pragma integrity_check').fetchone()[0]=='ok'
        for stage,cap,reserved,verified,unknown in c.execute('select * from stages'):
            assert reserved==verified and unknown==0 and reserved<=cap
            snap=status['budget']['stages'][stage]
            assert snap=={'reserved':reserved,'verified':verified,'unknown':0,'pending':0}
        for rid,stage,amount,state in c.execute('select reservation_id,stage,amount,status from reservations'):
            assert state=='verified' and amount==1
            reservations[rid]=stage
        assert c.execute('select count(distinct attempt_id) from attempts').fetchone()[0]==1
    step_reservations=set()
    ep_analysis=[]
    for ep in episodes:
        rows=sorted((r for r in steps if key(r)==key(ep)),key=lambda r:r['step'])
        assert [r['step'] for r in rows]==list(range(ep['environment_steps']))
        memory=PublicMemory();counts={'completed':0,'expired':0};energy=36.;utility=0.
        for r in rows:
            d=dmap[(*key(r),r['step'])];obs=d['observation'];memory.observe(obs)
            assert list(memory.candidates(obs))==d['candidates']
            assert obs['mask'][d['action']] and d['action'] in d['candidates'] and d['submit_command'] is True
            if ep['arm']=='M':
                probs=d['diagnostic']['probabilities']
                assert d['action']==max(d['candidates'],key=lambda a:(probs[a],-a))
                assert len(d['diagnostic']['selected_world_hidden_sha256'])==64
            memory.submitted(obs,d['action'])
            info=r['info'];newenergy=sum(info['energy'].values())
            v=[f32((max(0,info['counts']['completed']-counts['completed'])-max(0,info['counts']['expired']-counts['expired']))/6),f32(-max(0,energy-newenergy)/36)]
            assert all(abs(a-b)<1e-6 for a,b in zip(v,r['vector_reward']))
            utility+=.99**r['step']*(.4*v[0]+.2*v[1]);counts=info['counts'];energy=newenergy
            assert r['done']==bool(info['terminated'] or info['truncated'])
            assert r['done']==(r is rows[-1])
            rid=r['reservation']['reservation_id'];assert reservations[rid]=='environment_steps'
            assert rid not in step_reservations;step_reservations.add(rid)
            assert d['reservation']==r['reservation']
        assert abs(utility-ep['utility'])<1e-12
        assert ep['terminated']==rows[-1]['info']['terminated'] and ep['truncated']==rows[-1]['info']['truncated']
        last=ep['last_info'];records=last['completion_records']
        physical=sum(v['physical_arrival_before_deadline'] is True for v in records.values())
        host=sum(v['host_confirmation_before_deadline'] is True for v in records.values())
        unobserved=sum(v['host_confirmation_time'] is None for v in records.values())
        ep_analysis.append({k:ep[k] for k in ['parent','repeat','arm','environment_steps','utility','energy_used','task_counts','terminated','truncated']}|{
            'on_time_physical_observed':physical,'on_time_host_observed':host,'total_tasks':len(last['tasks']),
            'physically_completed_without_host_observed_at_native_stop':unobserved,
            'host_metric_scope':'Observed by native episode end; no post-terminal delivery extrapolation.'})
    bycall=collections.defaultdict(list)
    for r in calls:bycall[r['reservation']['reservation_id']].append(r)
    assert len(bycall)==75
    for rid,rs in bycall.items():
        assert [r['kind'] for r in rs]==['call_begin','call_return']
        assert all(r['stage']==reservations[rid] for r in rs)
    cost={};feedback={};messages={}
    for arm in ['M','R']:
        ds=[r for r in decisions if r['arm']==arm];t=[r['timing'] for r in ds]
        cost[arm]={'decisions':len(ds),'mean_cpu_ms':statistics.mean(r['controller_cpu_seconds'] for r in t)*1000,
            'mean_wall_ms':statistics.mean(r['controller_wall_seconds'] for r in t)*1000,
            'wall_p95_ms':percentile([r['controller_wall_seconds'] for r in t],.95)*1000,
            'totals_seconds':{k:sum(r[k] for r in t) for k in t[0]},
            'zero_recorded_cpu_decisions':sum(r['controller_cpu_seconds']==0 for r in t),
            'first_decisions':[r['timings'][0] for r in episodes if r['arm']==arm]}
        rr=[r for r in steps if r['arm']==arm]
        feedback[arm]=dict(collections.Counter(r['info']['feedback'] for r in rr))
        messages[arm]=dict(collections.Counter(z['link']+'/'+z['status'] for r in rr for z in r['info']['communication_delta']))
    differences=[]
    for parent in ['gate-000','gate-001']:
        pair={r['arm']:r for r in ep_analysis if r['parent']==parent}
        differences.append({'parent':parent,'utility_M_minus_R':pair['M']['utility']-pair['R']['utility']})
    for row in binding['native_inputs']:assert sha(Path(row['path']))==row['sha256']
    for name,h in binding['native_python_files'].items():assert sha(Path(binding['native_root'])/name)==h
    for name,h in binding['execution_files'].items():assert sha(ROOT/name)==h
    assert all(sha(ROOT/name)==h for name,h in inputs.items())
    result={'status':'functional_gate_passed_cost_evidence_limited','episodes':ep_analysis,
        'budget':status['budget']['stages'],'sqlite_integrity':'ok','single_attempt':True,
        'public_guard_reconstruction_passed':56,'reward_recomputation_passed':56,'durable_forward_pairs':75,
        'native_weights_unchanged_runtime_check':True,'input_files_unchanged_after_review':True,
        'controller_cost':cost,'peak_process_rss_bytes':status['peak_rss_bytes'],
        'runner_wall_seconds':status['wall_seconds'],'model_load_seconds':runtime['model_load_seconds'],
        'feedback_counts':feedback,'communication_delta_event_counts':messages,
        'descriptive_only_parent_differences':differences,
        'descriptive_only_mean_difference':statistics.mean(r['utility_M_minus_R'] for r in differences),
        'observed_model_cpu_threshold_satisfied':cost['M']['mean_cpu_ms']<=10,
        'observed_model_wall_threshold_satisfied':cost['M']['wall_p95_ms']<=50,
        'formal_stage_authorized':False,'algorithm_adoption_decision':'not_established',
        'limits':['Two technical parents do not establish benefit or broad runtime cost.',
            'Windows GetProcessTimes increments show 15.625ms quantization in these records; zero entries do not imply zero CPU.',
            'Shared process includes Torch thread pool for both arms; arm-isolated deployment CPU/RSS is not measured.',
            'Total process CPU and separate environment/logging CPU were not recorded; do not invent them.',
            'Native termination leaves one physical completion per episode without observed host confirmation; eventual receipt is unknown.',
            'No rerun, tuning, new samples, or formal matrix is authorized.']}
    write('analysis.json',result);write('input-hashes.json',inputs)
    print(json.dumps({'status':result['status'],'steps':56,'each_forward':25,'cpu_ms_M':cost['M']['mean_cpu_ms'],'sqlite':'ok'}))

if __name__=='__main__':main()
