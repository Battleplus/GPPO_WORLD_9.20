"""Planning only: original M/R statistics, fixed identities and budget. No inference."""
import hashlib
import json
import math
import statistics
from statistics import NormalDist
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'gppo-world-stage3-attribution-planning-20260925'
ATTEMPT='stage3-runtime-minimal-paired-v2-once'
SELECTION_NAMESPACE='runtime-necessity-minimal-v2-20260925'
KEYS=['environment_steps','environment_resets','policy_encode','actor_readout','world_candidate_batch','rule_decisions','model_loads','optimizer_updates','world_updates','offline_updates']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,d):(ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rank(parent,tag):return hashlib.sha256(f'{SELECTION_NAMESPACE}|{tag}|{parent}'.encode()).hexdigest()
def selected_keys():
    names=sorted([f'eval-{i:04d}' for i in range(1600)],key=lambda p:rank(p,'parent'))[:800]
    return [(p,min(range(3),key=lambda r:rank(p,f'repeat-{r}'))) for p in sorted(names)]

def prepare():
    previous=read(OLD/'BUDGET_REQUEST.json');inputs=dict(previous['inputs'])
    inputs['baseline_cost']=str(Path(inputs['baseline_episodes']).with_name('combined-cost.json'))
    rows=read(inputs['baseline_episodes']);stats=read(inputs['baseline_analysis']);tapes=read(inputs['scenario_manifest'])
    by={(r['parent'],r['repeat'],r['arm']):r for r in rows}
    if len(rows)!=9600 or len(by)!=9600:raise RuntimeError('Use completed stage2 matrix only')
    sd={}
    for metric in ('utility','physical_on_time','host_on_time_observed'):
        sd[metric]={'parent_mean_of_3_sd':statistics.stdev(r['differences'][metric] for r in stats['parents']),
            'single_repeat_across_parent_sd':[statistics.stdev(by[(f'eval-{p:04d}',r,'M')][metric]-by[(f'eval-{p:04d}',r,'R')][metric] for p in range(1600)) for r in range(3)]}
    z=NormalDist();sensitivity=[]
    for n in (128,256,384,512,800,1600):
        for sigma in (.04,.06,.08,.10,.12,.15,.20):
            se=sigma/math.sqrt(n)
            sensitivity.append({'independent_parents':n,'assumed_sd_of_parent_paired_difference':sigma,
                'approx_95ci_halfwidth':z.inv_cdf(.975)*se,
                'approx_power_NI_at_true_difference_zero':z.cdf(.01/se-z.inv_cdf(.975)),
                'approx_power_clear_loss_at_true_difference_minus_002_family_adjusted':z.cdf(.01/se-z.inv_cdf(1-.05/6)),
                'approx_80pct_distance_from_decision_boundary':(z.inv_cdf(.975)+z.inv_cdf(.8))*se})
    boundary=[]
    for sigma in (.08,.10,.12,.15):
        for mu in (0.,-.005,-.01,-.015,-.02,-.03):
            se=sigma/math.sqrt(800)
            boundary.append({'n':800,'assumed_sd':sigma,'assumed_true_N_minus_M':mu,
                'NI_probability_normal_approx':z.cdf((mu+.01)/se-z.inv_cdf(.975)),
                'clear_loss_probability_normal_approx_family_adjusted':z.cdf((-.01-mu)/se-z.inv_cdf(1-.05/6))})
    task_sensitivity=[{'n':800,'assumed_task_difference_sd':sigma,'margin':.02,
        'approx_95ci_halfwidth':z.inv_cdf(.975)*sigma/math.sqrt(800),
        'approx_noninferiority_power_at_true_zero':z.cdf(.02*math.sqrt(800)/sigma-z.inv_cdf(.975))}
        for sigma in (.10,.15,.20,.25,.30)]
    save('sample-size-evidence.json',{'historical_sd_M_minus_R_only':sd,'N_minus_M_variance_known':False,
        'method':'large-n normal approximation to paired-parent mean CI; sensitivity, not exact bootstrap power',
        'alpha_per_CI':.05,'bootstrap_future':10000,'target_power_reference':.8,
        'margins':{'utility':.01,'physical_rate':.02,'host_rate':.02},
        'sensitivity':sensitivity,'near_boundary_sensitivity':boundary,
        'task_guard_sensitivity':task_sensitivity,
        'minimum_n_at_assumed_SD_010_true_delta_0_normal_formula':math.ceil(((z.inv_cdf(.975)+z.inv_cdf(.8))*.1/.01)**2),
        'not_a_guaranteed_minimum':True,'joint_success_power_unknown':True,
        'no_finite_population_correction':True,'no_new_outcomes_or_observed_power_computed':True})
    selected=selected_keys();manifest=[]
    for parent,r in selected:
        i=int(parent.split('-')[1]);m=by[(parent,r,'M')];rule=by[(parent,r,'R')]
        if m['initial_public_hash']!=rule['initial_public_hash']:raise RuntimeError('Original pair mismatch')
        manifest.append({'parent':parent,'repeat':r,'arm':'N','tape_index':i,'tape_hash':tapes['hashes'][i],
            'initial_public_hash':m['initial_public_hash'],'exogenous_key':f'frozen-native-public-history-allocation-v1|{parent}|repeat-{r}'})
    save('sample-manifest.json',manifest)
    save('selection-contract.json',{'namespace':SELECTION_NAMESPACE,'algorithm':'SHA256(namespace|parent|parent_id) ascending first800; one repeat argmin SHA256(namespace|repeat-r|parent_id); execute parent ascending',
        'frame':1600,'selected':800,'repeats_per_parent':1,'input_fields_used':['parent_id','repeat_id'],
        'repeat_counts':{str(r):sum(rr==r for _,rr in selected) for r in range(3)},
        'no_reroll_no_replacement':True,'old_4800_request_superseded_not_approved':True})
    # Cost assignment uses the unchanged original per-arm execution order and exact step counts.
    costs=read(inputs['baseline_cost']);mr=[r for r in rows if r['arm']=='M']
    if [(r['parent'],r['repeat']) for r in mr]!=[(f'eval-{i:04d}',r) for i in range(1600) for r in range(3)]:raise RuntimeError('Original M cost order changed')
    if sum(r['steps'] for r in mr)!=len(costs['latencies']['M']):raise RuntimeError('Original M cost coverage mismatch')
    index=0;selected_cost=[];timing_manifest=[];keys=set(selected)
    for ep in mr:
        length=ep['steps'];chunk=costs['latencies']['M'][index:index+length]
        if (ep['parent'],ep['repeat']) in keys:
            timing_manifest.extend({'parent':ep['parent'],'repeat':ep['repeat'],'step':step,
                'original_M_latency_index':index+step,'cpu':c['cpu'],'wall':c['wall']} for step,c in enumerate(chunk))
            selected_cost.append({'parent':ep['parent'],'repeat':ep['repeat'],'steps':length,
                'source_latency_start':index,'source_latency_stop':index+length,
                'controller_cpu_seconds':sum(c['cpu'] for c in chunk),'controller_wall_seconds':sum(c['wall'] for c in chunk)})
        index+=length
    if len(selected_cost)!=800:raise RuntimeError('Selected cost missing')
    save('historical-M-timing-manifest.json',timing_manifest)
    save('selected-historical-M-cost.json',{'rows':selected_cost,'cost_source':inputs['baseline_cost'],
        'cost_source_sha256':sha(inputs['baseline_cost']),'ordering_basis':'original per-arm M parent/repeat order; validated 62979-step coverage',
        'timing_manifest_sha256':sha(ROOT/'historical-M-timing-manifest.json'),
        'cpu_seconds':sum(r['controller_cpu_seconds'] for r in selected_cost),'steps':sum(r['steps'] for r in selected_cost),
        'not_a_new_run':True,'cross_run_clock_limitation':True})
    def cap(env,resets,enc,world,loads,wall,cpu,gib):
        return {**dict.fromkeys(KEYS,0),'environment_steps':env,'environment_resets':resets,'policy_encode':enc,
            'actor_readout':enc,'world_candidate_batch':world,'model_loads':loads,'wall_seconds':wall,
            'cpu_seconds':cpu,'artifact_bytes':int(gib*1024**3)}
    stages={'A':cap(0,0,1575,525,3,1200,3600,.5),'B':cap(36,2,36,0,1,600,2400,.5),
        'C':cap(14400,800,14400,0,1,10800,43200,3)}
    caps={k:sum(s[k] for s in stages.values()) for k in KEYS}
    caps.update(wall_seconds=12600,total_process_cpu_seconds=49200,resident_process_rss_sum_bytes=4*1024**3,artifact_bytes=4*1024**3)
    save('BUDGET_REQUEST.json',{'attempt':ATTEMPT,'approved':False,'purpose':'minimum fixed paired frozen runtime simplification diagnostic',
        'caps':caps,'stage_caps':stages,'recovery_shutdown_holdback':{'wall_seconds':30,'cpu_seconds':600},
        'inputs':inputs,'history_counters':KEYS,'stage2_historical_consumed':previous['stage2_historical_consumed'],
        'historical_cost':previous['historical_cost'],'old_request':str(OLD/'BUDGET_REQUEST.json'),
        'old_request_authorized':False,'historical_balance_reuse':False,'interstage_borrowing':False,
        'sequence':['A fixed trace reference equivalence and costs','B 2 normal-reset technical episodes','C 800 fixed parent/exogenous pairs'],
        'automatic_retry':False,'automatic_sample_extension':False,'training_updates':0,'git_publication':False,
        'scope':'historical native M10/W1, seed1101, public-history frozen policy trained WITH world',
        'cost_caps_are_limits_not_duration_predictions':True})
    save('authorization-template.json',{'approved':False,'attempt':ATTEMPT,'caps':caps,'stage_caps':stages,
        'manifest_sha256':'UNFROZEN','budget_sha256':sha(ROOT/'BUDGET_REQUEST.json'),'user_approval_text':None,'user_approval_reference':None})

if __name__=='__main__':prepare()
