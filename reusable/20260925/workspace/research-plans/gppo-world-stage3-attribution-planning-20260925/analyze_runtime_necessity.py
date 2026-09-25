"""Complete-matrix analysis only; standard library, no outcome-based selection."""
import math
import random
import statistics
from analysis_contract import decide

def percentile(values,p):
    s=sorted(values);k=(len(s)-1)*p;i=math.floor(k);j=math.ceil(k)
    return s[i]+(s[j]-s[i])*(k-i)

def analyze(old,new,cost,a_cost,resources,check=lambda:None):
    all_rows=old+new;by={(r['parent'],r['repeat'],r['arm']):r for r in all_rows}
    expected={(f'eval-{p:04d}',r,a) for p in range(1600) for r in range(3) for a in ['M','R','N']}
    if len(by)!=len(all_rows) or set(by)!=expected:raise ValueError('Incomplete or duplicated paired matrix')
    for row in all_rows:
        if not all(math.isfinite(row[k]) for k in ('utility','physical_on_time','host_on_time_observed','energy_used')):
            raise ValueError('Nonfinite label')
        if not all(0<=row[k]<=1 for k in ('physical_on_time','host_on_time_observed')) or row['energy_used']<0:
            raise ValueError('Out-of-range label')
    metrics=['utility','physical_on_time','host_on_time_observed','energy_used'];parents=[]
    for p in range(1600):
        name=f'eval-{p:04d}';arms={a:{k:statistics.mean(by[(name,r,a)][k] for r in range(3)) for k in metrics} for a in ['M','R','N']}
        if not all(math.isfinite(v) for d in arms.values() for v in d.values()):raise ValueError('Nonfinite label')
        parents.append({'parent':name,'arms':arms})
    diffs={contrast:{k:[p['arms']['N'][k]-p['arms'][baseline][k] for p in parents] for k in metrics} for contrast,baseline in [('N-M','M'),('N-R','R')]}
    rng=random.Random(20260925);samples={contrast:{k:[] for k in metrics} for contrast in diffs}
    for draw_index in range(10000):
        if draw_index%50==0:check()
        draw=rng.choices(range(1600),k=1600)
        for contrast in diffs:
            for metric in metrics:samples[contrast][metric].append(sum(diffs[contrast][metric][i] for i in draw)/1600)
    summary={c:{k:{'mean_difference':statistics.mean(v),'ci95':[percentile(samples[c][k],.025),percentile(samples[c][k],.975)]} for k,v in d.items()} for c,d in diffs.items()}
    t=cost['latencies'];expected_decisions=sum(r['steps'] for r in new)
    if len(t)!=expected_decisions or not all(math.isfinite(row[k]) and row[k]>=0 for row in t for k in ('cpu','wall')):raise ValueError('Invalid cost records')
    if sum(row['cpu'] for row in t)<=0 or sum(row['wall'] for row in t)<=0:raise ValueError('Unmeasurable aggregate control cost')
    cpu=statistics.mean(r['cpu'] for r in t)*1000;wall=percentile([r['wall']*1000 for r in t],.95)
    # The denominator is the frozen full-matrix stage2 measurement, not a selected fast segment.
    historic_mean=cost['historical_M_mean_cpu_ms']
    if historic_mean<=0:raise ValueError('Invalid historical CPU denominator')
    decision=decide(n_minus_m_ci=summary['N-M']['utility']['ci95'],n_minus_r_ci=summary['N-R']['utility']['ci95'],
        physical_difference=summary['N-M']['physical_on_time']['mean_difference'],host_difference=summary['N-M']['host_on_time_observed']['mean_difference'],
        n_cpu_ms=cpu,n_wall_p95_ms=wall,rss_sum_bytes=resources['peak_conservative_rss_sum'],
        paired_replay_cpu_ratio=a_cost['paired_replay_cpu_ratio'],historical_formal_cpu_ratio=cpu/historic_mean)
    return {**decision,'summary':summary,'arm_means':{a:{k:statistics.mean(p['arms'][a][k] for p in parents) for k in metrics} for a in ['M','R','N']},
        'parents':parents,'parent_count':1600,'repeats':3,'new_episodes':4800,'reuse_episodes':9600,
        'N_controller_cost':{'decisions':expected_decisions,'mean_cpu_ms':cpu,'wall_p95_ms':wall,'ratio_to_historical_M':cpu/historic_mean},
        'paired_replay_cpu_ratio':a_cost['paired_replay_cpu_ratio'],'bootstrap_seed':20260925,'bootstrap_count':10000,
        'noninferiority_margin':.01,'minimum_increment_vs_rule':.01,'minimum_cpu_saving_fraction':.20,
        'scope':'Previously observed stage2 development parents; frozen seed1101 policy trained WITH world; online-readout removal only',
        'historical_cross_run_cost_limitation':True,'training_updates':0,'independent_training_comparison':False}
