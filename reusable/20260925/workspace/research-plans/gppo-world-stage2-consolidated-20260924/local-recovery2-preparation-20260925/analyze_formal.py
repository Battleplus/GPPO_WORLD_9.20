"""Frozen parent-level analysis; pure standard library, no model/runner calls."""
import math
import random
import statistics

def percentile(values,p):
    s=sorted(values);k=(len(s)-1)*p;i=math.floor(k);j=math.ceil(k)
    return s[i]+(s[j]-s[i])*(k-i)

def analyze(episodes,cost,*,parent_count=1600,repeat_count=3,bootstrap_count=10000):
    expected={(f'eval-{p:04d}',r,a) for p in range(parent_count) for r in range(repeat_count) for a in ['M','R']}
    by={(r['parent'],r['repeat'],r['arm']):r for r in episodes}
    if len(by)!=len(episodes) or set(by)!=expected:raise ValueError('Incomplete or duplicated formal matrix')
    metrics=['utility','physical_on_time','host_on_time_observed','energy_used']
    parents=[]
    for p in range(parent_count):
        name=f'eval-{p:04d}';row={'parent':name,'arms':{},'differences':{}}
        for arm in ['M','R']:
            values={k:statistics.mean(by[(name,r,arm)][k] for r in range(repeat_count)) for k in metrics}
            if not all(math.isfinite(x) for x in values.values()):raise ValueError('Nonfinite primary labels')
            row['arms'][arm]=values
        row['differences']={k:row['arms']['M'][k]-row['arms']['R'][k] for k in metrics};parents.append(row)
    rng=random.Random(20260924);samples={k:[] for k in metrics}
    for _ in range(bootstrap_count):
        draw=rng.choices(range(parent_count),k=parent_count)
        for k in metrics:samples[k].append(sum(parents[i]['differences'][k] for i in draw)/parent_count)
    summary={k:{'mean_difference':statistics.mean(p['differences'][k] for p in parents),
        'ci95':[percentile(samples[k],.025),percentile(samples[k],.975)],
        'M_mean':statistics.mean(p['arms']['M'][k] for p in parents),'R_mean':statistics.mean(p['arms']['R'][k] for p in parents)} for k in metrics}
    runtime={}
    for arm in ['M','R']:
        ts=cost['latencies'][arm]
        if len(ts)!=sum(r['steps'] for r in episodes if r['arm']==arm):raise ValueError('Missing decision timing')
        if not all(math.isfinite(v) and v>=0 for t in ts for v in t.values()):raise ValueError('Invalid timing')
        runtime[arm]={'decisions':len(ts),'mean_cpu_ms':statistics.mean(t['cpu'] for t in ts)*1000,
            'wall_p95_ms':percentile([t['wall']*1000 for t in ts],.95)}
    task_ok=summary['physical_on_time']['mean_difference']>=0 and summary['host_on_time_observed']['mean_difference']>=0
    cost_ok=runtime['M']['mean_cpu_ms']<=10 and runtime['M']['wall_p95_ms']<=50 and cost['resources']['peak_conservative_rss_sum']<=4*1024**3
    lower,upper=summary['utility']['ci95']
    if lower>.01 and task_ok and cost_ok:decision='practical_gain_pass_prepare_world_attribution_protocol_only'
    elif upper<=.01 or not task_ok or not cost_ok:decision='simplify_or_pause_current_candidate'
    else:decision='uncertain_stop_without_additional_samples'
    return {'decision':decision,'parent_count':parent_count,'repeat_count':repeat_count,'episodes':len(episodes),
        'primary':'native_discounted_utility_M_minus_R','practical_margin':.01,'summary':summary,'parents':parents,
        'controller_cost':runtime,'task_guard_pass':task_ok,'cost_guard_pass':cost_ok,'bootstrap_seed':20260924,
        'bootstrap_count':bootstrap_count,'updates':0,'independent_unit':'parent','automatic_followup_experiment':False,
        'scope':'Frozen native historical M10/W1, one model seed; no training-seed or deployment-generalization claim.'}
