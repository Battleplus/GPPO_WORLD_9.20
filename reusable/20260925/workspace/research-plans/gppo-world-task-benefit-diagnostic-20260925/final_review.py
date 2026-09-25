"""Read terminated artifacts only; no inference, environment or statistical rerun."""
import time
START=time.perf_counter()
import hashlib
import json
from pathlib import Path
import sqlite3
from attribution_contract import authorize

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/task-benefit-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

def main():
    authorization,manifest,request=authorize(ROOT/'authorization-approved.json')
    if not (OUT/'status.json').exists():raise RuntimeError('Complete successful terminal status required for full matrix review')
    status=read(OUT/'status.json');analysis=read(OUT/'C/analysis.json');selection=read(ROOT/'sample-manifest.json')
    assert status['status']=='completed'
    assert analysis['prior_A_cost_failure_preserved'] and analysis['practical_acceptance_pass'] is False
    assert not (OUT/'A').exists()
    b=rows(OUT/'B/episodes.jsonl');c=rows(OUT/'C/episodes.jsonl')
    assert len(b)==2 and len(c)==800
    expected={(x['parent'],x['repeat'],'N') for x in selection}
    assert {(x['parent'],x['repeat'],x['arm']) for x in c}==expected
    assert len(expected)==800
    total_steps=sum(x['steps'] for x in b+c)
    new=status['new_totals'];assert total_steps==new['environment_steps']==new['policy_encode']==new['actor_readout']
    assert new['environment_resets']==802 and new['model_loads']==2
    assert all(new[k]==0 for k in ['world_candidate_batch','rule_decisions','optimizer_updates','world_updates','offline_updates'])
    snapshot=read(OUT/'budget-snapshot.json')
    assert all(v['unknown']==0 and v['pending']==0 and v['reserved']==v['verified'] for v in snapshot['stages'].values())
    db=OUT/'budget.sqlite3';before=sha(db)
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        integrity=connection.execute('PRAGMA integrity_check').fetchone()[0]
    assert integrity=='ok' and sha(db)==before
    stage_resources={s:read(OUT/s/'cost-and-interface.json')['resources'] for s in ['B','C']}
    for stage,r in stage_resources.items():
        cap=request['stage_caps'][stage];extra=request['preparation_charge_bound'] if stage=='B' else {'wall_seconds':0,'cpu_seconds':0}
        assert r['stage_wall_seconds']+extra['wall_seconds']<=cap['wall_seconds']
        assert r['stage_cpu_seconds']+extra['cpu_seconds']<=cap['cpu_seconds']
        assert all(w['exit_code']==0 and w['final']['weights_unchanged'] for w in r['workers'])
    r=status['new_resources'];prep=request['preparation_charge_bound'];hold=request['recovery_shutdown_holdback']
    result={
        'status':'complete_fixed_matrix_verified','decision':analysis['decision'],
        'task_benefit_retention_pass':analysis['task_benefit_retention_pass'],
        'increment_vs_rule_pass':analysis['increment_vs_rule_pass'],
        'clear_loss_family_pass':analysis['clear_loss_family_pass'],
        'prior_A_cost_failure_preserved':True,'practical_acceptance_pass':False,
        'B_episodes':2,'C_episodes':800,'old_M_R_reused':1600,
        'new_counts':new,'cumulative_counts_stage2_failedA_diagnostic':status['stage2_plus_failed_A_plus_diagnostic_totals'],
        'SQLite_integrity':integrity,'SQLite_unchanged_by_review':True,'pending':0,'unknown':0,
        'frozen_inputs_reverified':len(manifest['files']),
        'stage_resources':stage_resources,'new_execution_resources':r,
        'prior_A_mean_CPU_ms':read(Path(request['inputs']['prior_A_result']))['N_mean_cpu_ms'],
        'task_effects':analysis['summary'],'N_controller_cost':analysis['N_controller_cost'],
        'preparation_charge_bound':prep,'shutdown_tail_bound':hold,
        'no_retry':True,'new_review_environment_steps':0,'new_review_model_forwards':0,
        'training_updates':0,'world_independent_contribution_proven':False,
    }
    review=OUT/'final-review';review.mkdir(exist_ok=False)
    hashes={}
    for path in [OUT/'status.json',OUT/'budget.sqlite3',OUT/'budget-snapshot.json',OUT/'C/analysis.json',
                 OUT/'C/complete-N-episodes.json',OUT/'B/episodes.jsonl',OUT/'C/episodes.jsonl',
                 OUT/'B/cost-and-interface.json',OUT/'C/cost-and-interface.json']:
        hashes[str(path)]=sha(path)
    result['review_process_cpu_seconds_so_far']=time.process_time()
    result['review_wall_seconds_so_far']=time.perf_counter()-START
    # Preserve the reserved unmeasured shutdown tail, and add this distinct read-only process.
    wall_bound=r['wall_seconds']+prep['wall_seconds']+hold['wall_seconds']+result['review_wall_seconds_so_far']
    cpu_bound=r['total_cpu_seconds']+prep['cpu_seconds']+hold['cpu_seconds']+result['review_process_cpu_seconds_so_far']
    result['new_guard_charged_wall_bound']=wall_bound
    result['new_guard_charged_CPU_bound']=cpu_bound
    result['cumulative_guard_charged_wall_bound']=status['prior_guard_charged_wall_bound']+wall_bound
    result['cumulative_guard_charged_CPU_bound']=status['prior_guard_charged_CPU_bound']+cpu_bound
    result['actual_preparation_and_exit_tail_not_falsely_claimed_measured']=True
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    result['package_bytes_before_report']=size
    assert wall_bound<request['caps']['wall_seconds'] and cpu_bound<request['caps']['total_process_cpu_seconds']
    assert r['peak_conservative_rss_sum']<=request['caps']['resident_process_rss_sum_bytes'] and size<request['caps']['artifact_bytes']
    final_stage=stage_resources['C']
    # Include analysis through parent terminal snapshot, not merely the environment loop.
    assert r['stage_wall_seconds']+hold['wall_seconds']+result['review_wall_seconds_so_far']<request['stage_caps']['C']['wall_seconds']
    assert r['stage_cpu_seconds']+hold['cpu_seconds']+result['review_process_cpu_seconds_so_far']<request['stage_caps']['C']['cpu_seconds']
    q1='支持保留（效用非劣与两项任务护栏均通过）' if analysis['task_benefit_retention_pass'] else '尚不支持保留'
    q2='有实用任务增量' if analysis['increment_vs_rule_pass'] else '未证实实用任务增量'
    decision={'support_preparing_simplification_task_basis_only_cost_not_qualified':'支持准备简化方案，但成本实用验收仍不通过',
              'clear_loss_direct_removal_not_supported':'明确损失，否定当前冻结策略的直接移除',
              'uncertain_pause_no_more_samples':'证据不确定，暂停且不追加样本'}[analysis['decision']]
    lines=['# 独立任务收益诊断最终报告','',f'1. 移除在线世界模型后的任务收益：{q1}。',f'2. 相对强规则：{q2}。',f'3. 研究决策：{decision}。','',
           '原 A 的 10.386904761904763 ms > 10 ms 成本失败保持不变，未重试 A。以下任务判断不构成成本通过或部署合格。移除损失不证明世界模型独立贡献。','',
           '|配对终点|均值差|95%父场景CI|','|---|---:|---|']
    for contrast in ['N-M','N-R']:
        for metric,v in analysis['summary'][contrast].items():lines.append(f"|{contrast} {metric}|{v['mean_difference']:.9f}|[{v['ci95'][0]:.9f}, {v['ci95'][1]:.9f}]|")
    lines+=['',f"明确损失的家族校正区间：`{json.dumps(analysis['loss_family_ci983333'],ensure_ascii=False)}`。",'',
            '800 个原先固定父场景各一个外生 repeat；C 新增 N 800 条，复用 M/R 1600 条，B 两条技术 episode 不纳入效应。10000 次同步父场景 bootstrap，seed=20260925。没有按结果选样、扩样、训练或后续实验。','',
            f"本次实际环境步 {new['environment_steps']}，encode/actor 各 {new['policy_encode']}，reset 802，加载 2；world、规则及更新均为 0。账本 reserved=verified，无 pending/unknown，完整性 ok。",'',
            f"累计阶段2＋失败A＋本次计数：`{json.dumps(result['cumulative_counts_stage2_failedA_diagnostic'],ensure_ascii=False)}`。",'',
            f"新执行进程测得 CPU {r['total_cpu_seconds']:.6f} s、wall {r['wall_seconds']:.6f} s，合计驻留峰值保守和 {r['peak_conservative_rss_sum']} bytes；含准备、尾部上界与本次复核的计费上界 CPU {cpu_bound:.6f} s、wall {wall_bound:.6f} s。分项和总授权额度核对通过。上界不是精确测量。",'',
            f"N 控制成本：`{json.dumps(analysis['N_controller_cost'],ensure_ascii=False)}`。跨运行成本比较仍受计时范围与系统负载限制；即使某项新成本低于阈值，整体实用验收仍不通过。",'',
            '范围仅为历史 M10 25-action/W1、冻结 seed1101、已经公开的开发父场景；不支持有限通信部署或新父场景/训练seed泛化。N 保留了含 world 训练得到的权重，不是无 world 训练基线。', '',
            '输入身份在执行前后复核；weights unchanged，旧 ledger 哈希未变。详细成本谱系、输入验证及机器可读结论见 verification.json；未提交或发布 Git。']
    (review/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(review/'verification.json',result)
    hashes[str(review/'report.md')]=sha(review/'report.md');hashes[str(review/'verification.json')]=sha(review/'verification.json')
    save(review/'hashes.json',hashes)
    print(json.dumps({'decision':analysis['decision'],'task_retention':analysis['task_benefit_retention_pass'],'rule_increment':analysis['increment_vs_rule_pass'],
        'new_steps':new['environment_steps'],'prior_cost_failure_retained':True,'report':str(review/'report.md')},ensure_ascii=False))

if __name__=='__main__':main()
