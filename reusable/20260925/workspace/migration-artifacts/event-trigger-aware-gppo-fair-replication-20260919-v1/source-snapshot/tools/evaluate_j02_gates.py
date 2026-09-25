"""Apply frozen development gates without modifying scores or thresholds."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_d02_adapter_diagnostics import write,sha


def evaluate(folder):
    results=json.loads((folder/'results.json').read_text())
    manifest=json.loads((folder/'run-manifest.json').read_text())
    p=manifest['protocol'];g=p['gates'];seeds=p['seeds']
    if results['updates']!=p['training']['unique_updates_total']:raise ValueError('Budget accounting failed')
    matrix={(r['group'],r['seed']):r for r in results['results']}
    if set(matrix)!={(a,s) for a in p['groups'] for s in seeds}:raise ValueError('Incomplete matrix')
    for member in json.loads((folder/'inventory.json').read_text()):
        path=folder/member['path']
        if path.stat().st_size!=member['bytes'] or sha(path)!=member['sha256']:raise ValueError('Artifact corruption')
    checks=[]
    def check(name,value,threshold,direction='min'):
        checks.append({'gate':name,'value':value,'threshold':threshold,'direction':direction,
                       'passed':value>=threshold if direction=='min' else value<=threshold})
    def avg(xs):return sum(xs)/len(xs)
    candidate=[matrix['J-fixed-action',s] for s in seeds]
    noaction=[matrix['J-fixed-noaction',s] for s in seeds]
    supervised=[matrix['S',s] for s in seeds]
    for r in candidate:
        for field,key in [('latent_persistence_improvement','all_seeds_latent_persistence_relative_improvement_min'),
                          ('latent_mean_std','all_seeds_latent_mean_std_min'),('effective_rank','all_seeds_effective_rank_min')]:
            check(f'{field}/seed{r["seed"]}',r[field],g[key])
    for task in ('state_active_nmse','reward_nmse','cost_nmse'):
        values=lambda rs:[r['task_metrics'][task]['scenario_macro'] for r in rs]
        ca,na,su=values(candidate),values(noaction),values(supervised)
        check(f'{task}/ratio_to_supervised',avg(ca)/max(avg(su),1e-20),g['mean_state_reward_cost_nmse_ratio_to_supervised_max'],'max')
        check(f'{task}/improvement_vs_noaction',1-avg(ca)/max(avg(na),1e-20),g['mean_state_reward_cost_nmse_relative_improvement_vs_noaction_min'])
        check(f'{task}/positive_seed_count',sum(a<b for a,b in zip(ca,na)),g['positive_improvement_seed_count_min'])
    reward=avg([r['task_metrics']['reward_nmse']['scenario_macro'] for r in candidate])
    check('reward/ratio_to_raw_direct',reward/max(results['raw_direct']['reward_nmse']['scenario_macro'],1e-20),g['mean_reward_nmse_ratio_to_raw_direct_max'],'max')
    regret=avg([r['immediate_branch_regret'] for r in candidate]);nr=avg([r['immediate_branch_regret'] for r in noaction])
    check('branch_regret/improvement_vs_noaction',1-regret/max(nr,1e-20),g['branch_reward_regret_relative_improvement_vs_noaction_min'])
    check('branch_regret/positive_seed_count',sum(a['immediate_branch_regret']<b['immediate_branch_regret'] for a,b in zip(candidate,noaction)),g['positive_improvement_seed_count_min'])
    check('branch_distinguishable_origins',min(r['branch_origins'] for r in candidate),g['validation_min_reward_distinguishable_origins'])
    # Origin metadata from saved per-origin regrets independently confirms per-profile tape support.
    origins=json.loads((folder/f'J-fixed-action-{seeds[0]}'/'branch-regrets.json').read_text())['per_origin']
    for profile in ('single','sequential','overlap','burst','long_gap','weak_comm'):
        check(f'tapes/{profile}',len({r['tape_id'] for r in origins if r['scenario_id']==profile}),g['validation_min_tapes_per_non_normal_profile'])
    audit=json.loads((folder/'input-audit.json').read_text())
    check('input_audit_failures',0 if audit['status']=='passed' else 1,0,'max')
    passed=all(c['passed'] for c in checks)
    return {'status':'development_passed' if passed else 'development_gate_failed','all_gates_passed':passed,
        'checks':checks,'failed_gates':[c['gate'] for c in checks if not c['passed']],
        'source_protocol_sha256':manifest['protocol_sha256'],'test_evaluated':False,'policy_training_started':False,
        'decision':'Proceed to separately frozen confirmation' if passed else 'Stop current candidate branch; preserve negative results; no J02C/P01 launch',
        'interpretation':'Development screen only; no formal statistical or stable policy benefit claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=evaluate(args.run);write(args.output,result);print(json.dumps(result))
