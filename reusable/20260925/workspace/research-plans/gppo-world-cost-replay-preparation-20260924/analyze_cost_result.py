"""Analyze completed saved timing records, with no native/model execution."""
from pathlib import Path
import collections
import hashlib
import json
import math
import sqlite3

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/cost-diagnostic-v1'
OUT=ROOT/'review-v1'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def percentile(values,q):
    values=sorted(values);index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)
def aggregate(batches):
    n=sum(b['decisions'] for b in batches)
    cpu=sum(b['cpu_seconds'] for b in batches);wall=sum(b['wall_seconds'] for b in batches)
    return {'decisions':n,'cpu_seconds':cpu,'wall_seconds':wall,'cpu_ms_per_decision':cpu*1000/n,
        'wall_ms_per_decision':wall*1000/n,
        'decision_wall_p95_ms':percentile([t*1000 for b in batches for t in b['decision_wall_seconds']],.95),
        'batch_cpu_ms_min':min(b['cpu_seconds']/b['decisions']*1000 for b in batches),
        'batch_cpu_ms_max':max(b['cpu_seconds']/b['decisions']*1000 for b in batches)}

def main():
    OUT.mkdir(exist_ok=False)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in RUN.iterdir() if p.is_file()}
    status=read(RUN/'status.json');binding=read(ROOT/'binding.json');auth=read(RUN/'authorization.json')
    assert status['status']=='completed' and status['updates']==0
    assert status['formal_stage_authorized'] is False and auth['approved'] is True
    assert auth['binding_sha256']==sha(ROOT/'binding.json')
    for item in binding['inputs']:assert sha(Path(item['path']))==item['sha256']
    assert [r['arm'] for r in status['workers']]==['M','R']
    assert all(r['exit_code']==0 for r in status['workers'])
    counts=collections.Counter();result={}
    for arm in ['M','R']:
        r=read(RUN/(arm+'-result.json'));bs=r['batches']
        assert len(bs)==21 and [b['index'] for b in bs]==list(range(21))
        assert bs[0]['cold'] is True and all(b['cold'] is False for b in bs[1:])
        assert all(b['decisions']==({'M':25,'R':31}[arm]) for b in bs)
        assert all(len(b['decision_wall_seconds'])==b['decisions'] for b in bs)
        assert all(math.isfinite(b['cpu_seconds']) and b['cpu_seconds']>0 and b['wall_seconds']>0 for b in bs)
        assert r['action_reproduction'] and r['updates']==0
        if arm=='M':assert r['hidden_reproduction']
        counts.update(r['counts'])
        worker=next(x for x in status['workers'] if x['arm']==arm)
        result[arm]={'cold':aggregate(bs[:1]),'warm':aggregate(bs[1:]),'all':aggregate(bs),
            'complete_process_cpu_seconds':worker['process_cpu_seconds'],
            'complete_process_wall_seconds':worker['lifetime_wall_seconds'],
            'setup_wall_seconds':r['setup_wall_seconds'],'peak_rss_bytes':r['peak_rss_bytes'],
            'action_reproduction':True,'hidden_reproduction':r['hidden_reproduction']}
        assert result[arm]['all']['cpu_seconds']<=worker['process_cpu_seconds']
    with sqlite3.connect((RUN/'budget.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
        assert c.execute('pragma integrity_check').fetchone()[0]=='ok'
        for stage,cap,reserved,verified,unknown in c.execute('select * from stages'):
            assert cap==auth['limits'][stage] and reserved==verified==counts[stage] and unknown==0
            assert status['budget']['stages'][stage]=={'reserved':reserved,'verified':verified,'unknown':0,'pending':0}
        assert c.execute("select count(*) from reservations where status!='verified'").fetchone()[0]==0
        assert c.execute('select count(distinct attempt_id) from attempts').fetchone()[0]==1
    assert all(sha(ROOT/name)==h for name,h in hashes.items())
    analysis={'result':'cost_threshold_exceeded_on_fixed_traces','arms':result,'actual_counts':dict(counts),
        'budget':status['budget']['stages'],'sqlite_integrity':'ok','single_attempt':True,
        'model_cpu_threshold_ms':10,'model_warm_cpu_pass':result['M']['warm']['cpu_ms_per_decision']<=10,
        'no_new_environment_steps':True,'updates':0,'new_training':False,'formal_matrix_executed':False,
        'decision':'pause_current_frozen_configuration_before_formal_matrix',
        'scope':'One hardware/runtime configuration and two already observed technical parents; replay repeats are not independent scenarios.',
        'task_gain':'not_established','world_attribution':'not_started','automatic_tuning_or_resampling':False}
    save('analysis.json',analysis);save('input-hashes.json',hashes)
    print(json.dumps({'decision':analysis['decision'],'M_warm_cpu_ms':result['M']['warm']['cpu_ms_per_decision'],
        'R_warm_cpu_ms':result['R']['warm']['cpu_ms_per_decision'],'verified_each_model_forward':counts['actor_readout']}))

if __name__=='__main__':main()
