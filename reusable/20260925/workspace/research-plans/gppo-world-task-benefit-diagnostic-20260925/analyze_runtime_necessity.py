"""Fixed 800-parent complete-pair analysis; no partial efficacy analysis."""
import math
import random
import statistics
from analysis_contract import decide

def percentile(values,p):
    s=sorted(values);k=(len(s)-1)*p;i=math.floor(k);j=math.ceil(k)
    return s[i]+(s[j]-s[i])*(k-i)

def analyze(old,new,cost,a_cost,resources,selection,historical_cost,check=lambda:None):
    keys=[(x['parent'],x['repeat']) for x in selection]
    if len(keys)!=800 or len({p for p,r in keys})!=800:raise ValueError('Fixed parent selection mismatch')
    all_rows=old+new;by={(r['parent'],r['repeat'],r['arm']):r for r in all_rows}
    expected={(p,r,a) for p,r in keys for a in ('M','R','N')}
    if len(by)!=len(all_rows) or set(by)!=expected:raise ValueError('Incomplete or duplicated paired matrix')
    metrics=['utility','physical_on_time','host_on_time_observed','energy_used']
    for row in all_rows:
        if not all(math.isfinite(row[k]) for k in metrics):raise ValueError('Nonfinite label')
        if not all(0<=row[k]<=1 for k in metrics[1:3]) or row['energy_used']<0:raise ValueError('Out-of-range label')
    parents=[{'parent':p,'repeat':r,'arms':{a:{k:by[(p,r,a)][k] for k in metrics} for a in ('M','R','N')}} for p,r in keys]
    diffs={c:{k:[p['arms']['N'][k]-p['arms'][base][k] for p in parents] for k in metrics} for c,base in [('N-M','M'),('N-R','R')]}
    rng=random.Random(20260925);samples={c:{k:[] for k in metrics} for c in diffs}
    for index in range(10000):
        if index%50==0:check()
        draw=rng.choices(range(800),k=800)
        for c in diffs:
            for k in metrics:samples[c][k].append(sum(diffs[c][k][i] for i in draw)/800)
    summary={c:{k:{'mean_difference':statistics.mean(v),'sd':statistics.stdev(v),
        'ci95':[percentile(samples[c][k],.025),percentile(samples[c][k],.975)]} for k,v in d.items()} for c,d in diffs.items()}
    # Any-of-three loss declarations use Bonferroni two-sided family intervals.
    loss_cis=[[percentile(samples['N-M'][k],.05/6),percentile(samples['N-M'][k],1-.05/6)] for k in metrics[:3]]
    t=cost['latencies'];steps=sum(r['steps'] for r in new)
    if len(t)!=steps or not t or not all(math.isfinite(r[k]) and r[k]>=0 for r in t for k in ('cpu','wall')):raise ValueError('Invalid cost records')
    cpu_sum=sum(r['cpu'] for r in t)
    if cpu_sum<=0:raise ValueError('Unmeasurable CPU')
    cpu_ms=cpu_sum/steps*1000;wall=percentile([r['wall']*1000 for r in t],.95)
    if {(r['parent'],r['repeat']) for r in historical_cost['rows']}!=set(keys):raise ValueError('Historical cost pairs mismatch')
    old_cpu=historical_cost['cpu_seconds'];old_steps=historical_cost['steps']
    if old_cpu<=0 or old_steps<=0:raise ValueError('Historical cost denominator')
    per_decision_ratio=(cpu_sum/steps)/(old_cpu/old_steps)
    per_episode_ratio=cpu_sum/old_cpu
    result=decide(utility_ci=summary['N-M']['utility']['ci95'],rule_ci=summary['N-R']['utility']['ci95'],
        physical_ci=summary['N-M']['physical_on_time']['ci95'],host_ci=summary['N-M']['host_on_time_observed']['ci95'],
        loss_family_ci=loss_cis,n_cpu_ms=cpu_ms,n_wall_p95_ms=wall,rss_sum_bytes=resources['peak_conservative_rss_sum'],
        replay_ratio=a_cost['paired_replay_cpu_ratio'],formal_decision_ratio=per_decision_ratio,formal_episode_ratio=per_episode_ratio)
    return {**result,'summary':summary,'loss_family_ci983333':dict(zip(metrics[:3],loss_cis)),
        'parents':parents,'parent_count':800,'repeats_per_parent':1,'new_episodes':800,'reused_episodes':1600,
        'N_controller_cost':{'decisions':steps,'mean_cpu_ms':cpu_ms,'wall_p95_ms':wall,'cpu_seconds':cpu_sum,
            'matched_M_cpu_seconds':old_cpu,'per_decision_ratio_to_matched_M':per_decision_ratio,
            'per_episode_ratio_to_matched_M':per_episode_ratio,'paired_replay_ratio':a_cost['paired_replay_cpu_ratio']},
        'bootstrap_count':10000,'bootstrap_seed':20260925,
        'scope':'one hash-selected repeat for each of800 hash-selected existing development parents; seed1101 historical M10/W1',
        'repeat_stability_not_estimated':True,'cross_run_cost_limitation':True,'training_updates':0}
