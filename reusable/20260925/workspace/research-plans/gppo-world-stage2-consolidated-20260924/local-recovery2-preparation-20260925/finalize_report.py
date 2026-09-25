"""Read-only final evidence checks and report. No environment/model execution."""
import collections
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/consolidated-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def key(r):return r['parent'],r['repeat'],r['arm']
def save(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    started=time.perf_counter();cpu=time.process_time()
    status=read(RUN/'status.json');assert status['status']=='completed'
    analysis=read(RUN/'C/analysis.json');combined=read(RUN/'C/combined-episodes.json')
    retained=read(ROOT/'retained-episodes.json');remaining=read(ROOT/'not-executed.json');matrix=read(ROOT/'frozen-matrix.json')
    assert len(combined)==9600 and len({key(r) for r in combined})==9600
    assert [key(r) for r in combined]==[key(r) for r in matrix]
    assert combined[:8637]==retained
    assert [key(r) for r in combined[8637:]]==[key(r) for r in remaining]
    groups=collections.Counter(r['parent'] for r in combined)
    assert len(groups)==1600 and set(groups.values())=={6}
    assert analysis['episodes']==9600 and analysis['parent_count']==1600 and analysis['repeat_count']==3
    # Verify the frozen gate from recorded full-matrix results, without rerunning bootstrap.
    summary=analysis['summary'];cost=analysis['controller_cost']
    assert analysis['practical_margin']==.01 and analysis['bootstrap_count']==10000 and analysis['bootstrap_seed']==20260924
    tasks=summary['physical_on_time']['mean_difference']>=0 and summary['host_on_time_observed']['mean_difference']>=0
    costs=cost['M']['mean_cpu_ms']<=10 and cost['M']['wall_p95_ms']<=50 and status['resources']['peak_conservative_rss_sum']<=4*1024**3
    passed=summary['utility']['ci95'][0]>.01 and tasks and costs
    assert passed and analysis['decision']=='practical_gain_pass_prepare_world_attribution_protocol_only'
    identities=read(ROOT/'recovery-manifest.json')['files']
    for row in identities:assert sha(Path(row['path']))==row['sha256'],row['path']
    prefix=read(RUN/'C/prefix-reproduction.json')
    assert prefix['exact_behavior_match'] and prefix['verified_steps']==7 and prefix['counts_charged']
    budget=status['cumulative_stages'];new=status['segment_budget']['stages']
    assert all(r['reserved']==r['verified'] and r['pending']==r['unknown']==0 for r in budget.values())
    assert all(r['reserved']<=r['authorized_limit_including_explicit_extension'] for r in budget.values())
    for arm,stage in [('M','C/policy_encode'),('R','C/rule_decisions')]:
        sample_steps=sum(r['steps'] for r in combined if r['arm']==arm)
        assert sample_steps==cost[arm]['decisions']
        assert sample_steps+(7 if arm=='R' else 0)==budget[stage]['verified']
    assert sum(r['steps'] for r in combined)+7==budget['C/environment_steps']['verified']
    assert all(v['verified']==0 for k,v in budget.items() if 'updates' in k)
    assert status['ledger_connections']==1 and status['observer_publish_errors']==0
    assert all(w['exit_code']==0 and w['final']['weights_unchanged'] for w in status['resources']['recovery_segment']['workers'])
    ledgers=read(ROOT/'historical-ledgers-readonly.json')+[{'segment':'recovery2','path':str(RUN/'budget.sqlite3'),'sha256':sha(RUN/'budget.sqlite3')}]
    db_checks=[]
    for row in ledgers:
        p=Path(row['path']);before=sha(p);assert before==row['sha256']
        wal=Path(str(p)+'-wal');assert not wal.exists() or wal.stat().st_size==0,'Live/nonempty WAL requires different read protocol'
        with closing(sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True)) as c:
            integrity=c.execute('PRAGMA integrity_check').fetchone()[0]
            pending=c.execute("SELECT COUNT(*) FROM reservations WHERE status='pending'").fetchone()[0]
            stages={r[0]:{'reserved':r[1],'verified':r[2],'unknown':r[3]} for r in c.execute('SELECT stage,reserved,verified,unknown FROM stages')}
        assert integrity=='ok' and pending==0 and before==sha(p)
        assert all(r['unknown']==0 and r['reserved']==r['verified'] for r in stages.values())
        if row['segment']=='recovery2':
            for k,r in stages.items():assert r=={f:new[k][f] for f in r}
        db_checks.append({'segment':row['segment'],'path':str(p),'sha256':before,'integrity':integrity,'pending':pending,'unknown':0,'unchanged_by_review':True})
    caps=read(ROOT/'RECOVERY_BUDGET_REQUEST.json');res=status['resources'];seg=res['recovery_segment']
    assert seg['wall_seconds']+caps['recovery_shutdown_holdback']['wall_seconds']<=caps['caps']['wall_seconds']
    assert seg['total_cpu_seconds']+caps['recovery_shutdown_holdback']['cpu_seconds']<=caps['caps']['total_process_cpu_seconds']
    assert res['bytes']<=caps['original_C_limits']['artifact_bytes']
    assert seg['bytes']<=caps['caps']['artifact_bytes']
    assert caps['historical_resources']['C_wall_seconds']+seg['wall_seconds']+90<=caps['original_C_limits']['wall_seconds']
    assert caps['historical_resources']['C_cpu_seconds']+seg['total_cpu_seconds']+1800<=caps['original_C_limits']['total_process_cpu_seconds']
    totals={suffix:sum(v['verified'] for k,v in budget.items() if k.endswith('/'+suffix)) for suffix in ['environment_steps','environment_resets','policy_encode','world_candidate_batch','actor_readout','rule_decisions','model_loads','optimizer_updates','world_updates','offline_updates']}
    out=ROOT/'final-review';out.mkdir(exist_ok=False)
    verification={'status':'verified_complete','matrix_total':9600,'independent_parents':1600,'repeats':3,
        'retained_unchanged':8637,'new_complete':963,'prefix_replayed_exactly':7,'replay_charged_not_new_samples':True,
        'statistical_episode_steps':sum(r['steps'] for r in combined),'frozen_files_matched':len(identities),
        'ledger_checks':db_checks,'cumulative_A_B_C':totals,'new_segment':new,'resources':res,
        'algorithm_gate_pass':passed,'resource_caps_pass':True,'decision':analysis['decision'],
        'historical_io_cause_still_unresolved':True,'historical_shutdown_cost_is_reserved_bound_not_exact_measurement':True,
        'review_environment_steps':0,'review_model_forwards':0,'review_updates':0}
    save(out/'verification.json',verification)
    save(out/'decision.json',{'stage':2,'result':'practical_increment_pass_in_frozen_scope',
        'next_action':'prepare_world_model_contribution_protocol','next_experiment_authorized':False,
        'training_authorized':False,'git_publication_authorized':False,'scope':analysis['scope'],
        'utility':summary['utility'],'controller_cost':cost,'task_guard_pass':tasks,'cost_guard_pass':costs,
        'not_yet_proven':['world_model_independent_contribution','preference_control','event_cost_savings','other_training_seeds','finite_communication_generalization','deployment_readiness']})
    u=summary['utility'];p=summary['physical_on_time'];h=summary['host_on_time_observed'];e=summary['energy_used']
    report=f'''# 阶段2完整矩阵：达到冻结实用增量，进入组件归因方案准备

9600/9600 个 episode 已完成，1600 个独立父场景、每父场景3次外生重复、M/R两臂。冻结 GPPO＋世界模型相对合理公开历史强规则，达到预先定义的实用增量，任务与成本护栏均通过。此结论只适用于历史 M10 25-action / W1、当前 P_train seed-1101 权重，不能外推到当前有限通信系统、其他训练seed或生产部署。

## 任务结果

| 指标（父场景等权） | 模型 M | 强规则 R | M−R，95% CI |
|---|---:|---:|---|
| 原生折扣效用 | {u['M_mean']:.9f} | {u['R_mean']:.9f} | {u['mean_difference']:.9f} [{u['ci95'][0]:.9f}, {u['ci95'][1]:.9f}] |
| 按时物理到达率 | {p['M_mean']*100:.4f}% | {p['R_mean']*100:.4f}% | +{p['mean_difference']*100:.4f} pp [{p['ci95'][0]*100:.4f}, {p['ci95'][1]*100:.4f}] |
| 按时主机确认率 | {h['M_mean']*100:.4f}% | {h['R_mean']*100:.4f}% | +{h['mean_difference']*100:.4f} pp [{h['ci95'][0]*100:.4f}, {h['ci95'][1]*100:.4f}] |
| 每episode能耗（原生单位） | {e['M_mean']:.6f} | {e['R_mean']:.6f} | +{e['mean_difference']:.6f} [{e['ci95'][0]:.6f}, {e['ci95'][1]:.6f}] |

主比较实用阈值为 0.01 原生效用单位，**不是1个百分点**。CI下界 {u['ci95'][0]:.6f} > 0.01；两项任务护栏都改善。模型消耗更多能量，收益由冻结的任务—能耗效用综合判断，不声称同时节能。

使用原分析器，父场景是独立单位，先平均三次重复，再等权汇总1600父场景；bootstrap 10000次、seed=20260924。场景、纳入规则、指标和停止条件未按模型表现调整。所有恢复前保留项与原文件逐条一致；没有部分矩阵提前揭盲。

## 成本与实际消耗

| 控制器成本 | 模型 M | 强规则 R | 模型冻结门槛 |
|---|---:|---:|---:|
| 平均CPU / decision | {cost['M']['mean_cpu_ms']:.6f} ms | {cost['R']['mean_cpu_ms']:.6f} ms | ≤10 ms |
| wall p95 / decision | {cost['M']['wall_p95_ms']:.6f} ms | {cost['R']['wall_p95_ms']:.6f} ms | ≤50 ms |

全部驻留进程RSS保守峰值合计 {res['peak_conservative_rss_sum']/1024**2:.3f} MiB < 4 GiB。暂停worker同样计入。CPU与wall是不同口径，不可互换；模型本地CPU线程为4/1。成本门通过，不代表任意机器、训练全过程或部署全生命周期都更便宜。

本次恢复执行963项，新增环境步13399、reset963、encode/world/actor各6329、规则决策7070、模型加载1，训练更新全0。新段采样wall {seg['wall_seconds']:.6f}秒，完整父/子进程CPU {seg['total_cpu_seconds']:.6f}秒，均远低于8小时/32 CPU小时上限。单预算连接，无观察发布错误。

原段＋恢复1＋恢复2的A/B/C累计实际计数：环境步 {totals['environment_steps']}、reset {totals['environment_resets']}、encode/world/actor各 {totals['policy_encode']}、规则决策 {totals['rule_decisions']}、模型加载 {totals['model_loads']}，训练更新全0。C累计环境步133610、reset9601；矩阵统计仅133603步，差额7步为中断前缀重放，不计新样本。所有账本unknown/pending=0，完整性检查均ok，旧账本未改写。

历史累计采样wall {res['wall_seconds']:.6f}秒（{res['wall_seconds']/3600:.4f}小时）、CPU {res['total_cpu_seconds']:.6f}秒（{res['total_cpu_seconds']/3600:.4f} CPU小时）。保留所有历史启动、加载、失败和恢复开销；历史收尾未精确测量，按批准预留累计90秒wall/1800秒CPU保守占用，得到 {res['guard_charged_wall_seconds']:.6f}秒 / {res['guard_charged_cpu_seconds']:.6f}秒。预留不是实测值。累计制品保守计数 {res['bytes']} bytes（终态写入前采样），最终审查新增开销另记于 review-overhead.json；没有选择两段中更好看的时间。

## 恢复与解释边界

8637项完整结果保持不变，只执行963个原清单剩余键。eval-1439/repeat-1/R 的7步以原reset、外生键精确重现；所有比较字段一致，重放计资源、不重复计统计。保留原7步timing，新7步timing作为恢复开销单列。两个worker正常退出，权重未变，129项冻结身份匹配。

此前监控共享冲突和SQLite I/O技术失败保留为技术失败，不冒充算法负结果。本次单连接缓解顺利完成，但未反向证明历史I/O根因已定位。用户批准了这一不确定性下的一次恢复，以及累计reset9601和模型加载6的例外额度。

本阶段模型有实际决策增量，**尚不能将增量归因给世界模型本身**。历史记忆、策略训练及世界预测的作用需要公平区分；也未证明偏好响应、事件成本收益或有限通信泛化。

## 决定

进入世界模型独立贡献的协议准备，不训练、不运行阶段3—5、不自动加样或调参。下一步唯一主要问题：在匹配公开历史、动作约束和训练/推理成本后，世界模型是否提供强规则或无世界模型策略不能解释的实际增量。不得仅以零置/打乱造成的分布外损伤证明其价值；任何新实验需单独预算授权。

证据：本目录 verification.json、decision.json；上级 runs/consolidated-v1/C/analysis.json、combined-episodes.json、combined-cost.json、prefix-reproduction.json、steps.jsonl.gz；runs/consolidated-v1/status.json、budget.sqlite3；上级 recovery-manifest.json、authorization-approved.json、RECOVERY_PROTOCOL.md。checkpoint SHA-256 为 bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a。

本轮未提交或发布Git，未连接服务器，未运行后续阶段。
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    # Reporting itself is not a model run; disclose its additional cost and artifact size.
    overhead={'wall_seconds_before_final_small_writes':time.perf_counter()-started,
        'cpu_seconds_before_final_small_writes':time.process_time()-cpu,
        'scope':'this read-only post-run review process; small final metadata writes excluded',
        'total_directory_bytes_before_final_small_writes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        'environment_steps':0,'model_forwards':0,'training_updates':0}
    assert overhead['wall_seconds_before_final_small_writes']+seg['wall_seconds']+30<28800
    assert overhead['cpu_seconds_before_final_small_writes']+seg['total_cpu_seconds']+600<115200
    assert overhead['total_directory_bytes_before_final_small_writes']<8*1024**3
    save(out/'review-overhead.json',overhead)
    sources=[RUN/'status.json',RUN/'budget.sqlite3',RUN/'C/analysis.json',RUN/'C/combined-episodes.json',RUN/'C/combined-cost.json',RUN/'C/prefix-reproduction.json',RUN/'C/steps.jsonl.gz',ROOT/'recovery-manifest.json',ROOT/'authorization-approved.json',ROOT/'finalize_report.py']
    sources.extend(sorted(out.iterdir()))
    save(out/'hashes.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in sources]})
    print(json.dumps({'decision':analysis['decision'],'episodes':9600,'retained':8637,'recovery_steps':13399,
          'cumulative_steps':totals['environment_steps'],'ledger_integrity':'ok','identity_matches':len(identities),
          'report':str(out/'report.md'),'postrun_review':overhead},ensure_ascii=False))

if __name__=='__main__':main()
