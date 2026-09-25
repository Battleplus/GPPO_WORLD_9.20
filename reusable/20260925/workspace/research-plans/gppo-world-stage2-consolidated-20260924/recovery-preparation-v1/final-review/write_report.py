"""Publish the terminal technical-stop decision; no partial effect computation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / 'runs/consolidated-v1'
PLANS = ROOT.parent.parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')

v = read(HERE / 'verification.json')
s = read(HERE / 'terminal-budget-and-resources.json')
request = read(ROOT / 'RECOVERY_BUDGET_REQUEST.json')
assert v['complete_total'] == 8637 and v['remaining_not_complete'] == 963
assert not v['formal_analysis_exists'] and not v['partial_effect_computed']
res = s['resources']
seg = res['recovery_segment']
cum = s['cumulative_stages']
total_counts = {}
for name, row in cum.items():
    metric = name.split('/', 1)[1]
    total_counts[metric] = total_counts.get(metric, 0) + row['verified']
assert total_counts['environment_steps'] == 120263
assert total_counts['model_loads'] == 5
assert all(r['pending'] == r['unknown'] == 0 and r['reserved'] <= r['authorized_limit_including_explicit_extension'] for r in cum.values())
assert seg['wall_seconds'] + 30 < request['caps']['wall_seconds']
assert seg['total_cpu_seconds'] + 600 < request['caps']['total_process_cpu_seconds']
assert seg['wall_seconds'] + request['historical_resources']['C_wall_seconds'] + 60 < request['original_C_limits']['wall_seconds']
assert seg['total_cpu_seconds'] + request['historical_resources']['C_cpu_seconds'] + 1200 < request['original_C_limits']['total_process_cpu_seconds']

input_paths = [ROOT/'authorization-approved.json', ROOT/'RECOVERY_BUDGET_REQUEST.json', ROOT/'recovery-manifest.json',
              ROOT/'retained-episodes.json', ROOT/'derived-summary-provenance.json', RUN/'status.json',
              RUN/'budget.sqlite3', RUN/'ledger-lineage.json', RUN/'C/episodes.jsonl',
              RUN/'C/episode-starts.jsonl', RUN/'C/steps.jsonl.gz', RUN/'C/scenario-manifest.json',
              RUN/'observer-progress.jsonl']
write(HERE/'input-hashes.json', [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in input_paths])
artifact_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
decision = {
    'attempt': v['attempt'], 'terminal_status': 'technical_stop',
    'decision': 'pause_due_to_technical_stop_not_algorithm_negative',
    'practical_gain': 'not_established_incomplete_frozen_matrix',
    'practical_cost_acceptance': 'not_evaluated_for_complete_matrix',
    'recorded_resource_caps': 'within_limits_including_shutdown_allowances',
    'error': s['error'], 'underlying_io_cause': 'not_established',
    'retained_complete_episodes': 8637, 'incomplete_episodes': 1, 'not_started_episodes': 962,
    'remaining_not_complete': 963,
    'recovery_steps': 63830, 'C_cumulative_steps': 120211,
    'A_B_C_cumulative_verified': total_counts,
    'pending': 0, 'unknown': 0, 'sqlite_integrity': 'ok',
    'complete_matrix_analyzed': False, 'partial_effect_computed': False,
    'world_attribution_authorized': False, 'automatic_retry': False,
    'new_training_or_tuning_or_samples': False, 'git_published': False,
    'artifact_bytes_before_final_report': artifact_bytes,
    'historical_plus_current_artifact_bytes_before_final_report': artifact_bytes + request['historical_resources']['artifact_bytes_conservative'],
    'cost_limitations': ['Process snapshots precede final status serialization and OS exit; shutdown allowances are not measured exact tails.',
                         'Post-stop verification/reporting and external observation helpers are outside the frozen experiment parent+worker boundary; not silently treated as zero.',
                         'No partial controller-latency performance gate or utility comparison was calculated.'],
    'scope': 'Frozen historical native M10 25-action/W1, seed-1101 checkpoint, original 1600-parent matrix.',
    'next_action': 'Stop. Any further execution requires a separate concrete repair/recovery proposal and new explicit authorization.'
}
write(HERE/'research-decision.json', decision)

report = f'''# 一次性恢复终态：技术停止，暂停

attempt：`consolidated-v1-recovery-1-once`。

**GPPO＋世界模型是否具有相对强规则的实用增量：尚不能判断。完整矩阵未完成，未计算部分主效应、置信区间或模型胜负。**

**成本是否达标：记录到的资源消耗在额度内；完整矩阵的模型 CPU、wall p95 与收益联合验收未完成，不能报告实用成本验收通过。**

**去留：暂停。此次为技术失败，不能据此宣称算法负结果，也不足以进入世界模型组件归因或作出算法简化结论。**

## 停止事实与边界

恢复进程返回 `OperationalError: disk I/O error`，终态为 `technical_stop`，shell exit code 为 1。父调度器已处置其两个 worker，核对时无残留 Python 实验进程。没有自动重试、模型重载、追加样本、训练、后续阶段或 Git 发布。

本次仅通过 `safe_observer.py` 读取独立 `observer-progress.jsonl`，没有读取正在写入的正式 episodes、steps 或 SQLite 作为进度来源。此错误不能未经证据归因为原来的文件共享冲突。终态没有保存足以定位 SQLite 底层 I/O 错误的完整堆栈/扩展错误码；只读 integrity_check=ok 也不能反证运行时错误没有发生。停止后 E 盘可用空间检查约 56.34 GiB；这不是根因证明。

## 工作保全

| 项目 | 数量 |
|---|---:|
| 原有完整 episode，全部保留 | 4034 |
| 本次完整执行且汇总存在 | 4603 |
| 本次完整执行但缺汇总 | 0 |
| 合计完整 episode | **8637 / 9600** |
| 本次已 reset 但未完整终止 | **1** |
| 尚未开始 | **962** |
| 尚未完整完成 | **963** |

最后完整项是 `eval-1439 / repeat-1 / M`。最后中断项是 `eval-1439 / repeat-1 / R`，逐步记录含 step 0..6 共 **7 步**，末步 `done=false`。不能将该项补成完整汇总，也未填零、猜测、重新执行或伪装成完成项。另 962 项未 reset。

原 4034 项含先前明确标注来源的 1 份派生汇总；其 provenance 与原始 12 步证据继续保留。本次没有新增派生 episode 汇总或改写任何原始日志。完整项及已开始项分别是冻结矩阵的精确前缀，没有重复键、替换场景或改变执行顺序。

新 steps gzip 可完整读至 EOF，CRC 通过，共 **63830** 条；逐条核对 step 顺序、动作 mask/候选资格、预算 reservation 身份和 verified 状态，终止 payload 与 4603 份原始汇总对应字段一致。累计 **4319** 对已 reset 的 M/R 公开初始哈希一致（包括末尾尚未完整完成的那对，不能将其算成完整结果对）。

91 项绑定文件哈希全部匹配，原源码、checkpoint、场景清单、恢复入口和旧账本保持冻结身份。恢复新账本只读核对前后 SHA-256 不变；所有 192415 条 reservation 为 verified，pending=unknown=0，integrity_check=ok。零步核验未调用环境、模型或训练。

## 实际消耗与累计额度

| 计数 | 恢复实际 | 原 C＋恢复累计 | A/B/C 总累计 |
|---|---:|---:|---:|
| 环境步 | 63830 | 120211 | **120263** |
| encode / world / actor，各 | 30075 | 56650 | **57722** |
| 规则决策 | 33755 | 63561 | **63591** |
| reset | 4604 | 8638 | **8642** |
| 模型加载 | 1 | 2 | **5** |
| optimizer / world / offline update | 0 | 0 | 0 |

本次环境步上限100188、各模型前向/规则决策上限50094、reset上限5566均未超额。原 C 环境步172800及各项累计上限也未超额。新增一次模型加载授权已消耗，累计5次已达到批准上限；余额不代表可再次执行的授权。A/B未重跑，历史失败消耗没有退款或清零。

| 资源 | 本次恢复采样 | 含历史累计采样 |
|---|---:|---:|
| wall | {seg['wall_seconds']:.6f} 秒（{seg['wall_seconds']/3600:.4f} 小时） | {res['wall_seconds']:.6f} 秒（{res['wall_seconds']/3600:.4f} 小时） |
| 父进程＋全部 worker CPU | {seg['total_cpu_seconds']:.6f} 秒（{seg['total_cpu_seconds']/3600:.4f} 小时） | {res['total_cpu_seconds']:.6f} 秒（{res['total_cpu_seconds']/3600:.4f} 小时） |
| 监测的保守 RSS 峰值和 | {seg['peak_conservative_rss_sum']} bytes | {res['peak_conservative_rss_sum']/1024**2:.3f} MiB |

RSS 包括暂停但仍驻留的 worker，上限4 GiB。累计保守计费另保留 wall 60秒/CPU 1200秒（旧、新收尾各30/600），对应 wall **{res['guard_charged_wall_seconds']:.6f}秒**、CPU **{res['guard_charged_cpu_seconds']:.6f}秒**；仍在批准的本段及原 C 累计额度内。预算预留不是已测量的精确关机开销；终态序列化及进程退出尾部不能伪称已完整重建。

终态快照制品计费为本段 {seg['bytes']} bytes、历史累计 {res['bytes']} bytes。快照之后状态文件仍会写出，因此另作停止后文件扫描：本包截至最终报告生成前为 **{artifact_bytes} bytes**（含准备与停止核验制品），加历史制品 debit 为 **{artifact_bytes+request['historical_resources']['artifact_bytes_conservative']} bytes**，远低于新增38 GiB和原 C 40 GiB；最终输出文件体积列入 `output-hashes.json`。

恢复启动、模型加载、逐步日志及预算写入开销均留在运行成本内，没有仅拼接较好的推理时间。恢复父进程记录的 ledger 写入 CPU {seg['parent_timed_parts']['C/ledger']['cpu']:.6f}秒、wall {seg['parent_timed_parts']['C/ledger']['wall']:.6f}秒属于上述全进程成本的组成部分，不能再次加总。

停止后独立核验脚本成功执行耗时 {v['verification_wall_seconds']:.6f}秒、CPU {v['verification_cpu_seconds']:.6f}秒，属于单独的研究审查开销。其首次解析 reservation 结构的纯数据检查失败后修正了审查脚本（未改运行源码），其他短时查看/报告与观察辅助进程未逐一 CPU 计量；这些不是环境重试，也不纳入冻结的实验父进程＋worker口径，不能宣称端到端研究开销已精确计量。

## 研究决定

原实用效用阈值0.01、任务护栏、1600父场景统计单位、三次外生重复、bootstrap及成本门保持不变。没有按部分模型表现改变场景、样本量或继续条件。

本次没有生成 `C/analysis.json` 或完整矩阵模型成本验收；其原因是明确的技术停止，不能用8637条部分样本替代原9600矩阵。当前只支持保全和消耗事实，不支持 GPPO/world 正负收益结论。

暂停执行，保留原失败与本次失败现场。若将来考虑恢复，需先形成针对 I/O 失败与7步中断项的可审计处理方案并获得新的明确授权；本报告不自动启动该工作，也不申请扩大统计样本。

## 制品索引

- `verification.json`：完整性、计数与零环境步检查。
- `completed-keys.json`：8637个完整项；`incomplete-episode-keys.json`：1个7步中断项。
- `not-started-keys.json`：962个尚未开始项；`complete-missing-summary-keys.json`：空。
- `terminal-budget-and-resources.json`：终态计费与原 C 累计，去除巨大逐条history但不改原status。
- `frozen-input-verification.json`：91项身份；`input-hashes.json`：停机现场输入哈希。
- `research-decision.json`：机器可读去留结论；`output-hashes.json`：审查输出哈希。
- 原始运行现场：`../runs/consolidated-v1/`，未改写。
'''
write(HERE/'report.md', report)

state_path = PLANS/'CURRENT_RESEARCH_STATE.json'
old_state = read(state_path)
write(HERE/'previous-current-research-state.json', old_state)
write(HERE/'previous-current-research-state.md', (PLANS/'CURRENT_RESEARCH_STATE.md').read_text(encoding='utf-8'))
new_md = f'''# 当前研究状态：一次性恢复技术停止，暂停

attempt `consolidated-v1-recovery-1-once` 因 `OperationalError: disk I/O error` 终止。未自动重试，原A/B未重跑；无训练或Git发布。

完整保留 **8637/9600 episode**：旧4034＋恢复4603。另1项仅完成7步且done=false，962项尚未开始。未完成共963项，不能把中断项当完整汇总。

GPPO＋世界模型相对强规则的实用增量尚不能判断：完整矩阵未完成，未计算部分主效应/胜负。记录资源消耗未超过额度；完整矩阵成本验收未完成。决定暂停，不进入组件归因，也不将技术失败当作模型负结果。

本次63830环境步、各模型前向30075、规则33755、reset4604、模型加载1；A/B/C累计环境步120263、各模型前向57722、规则63591、reset8642、加载5。pending=unknown=0，SQLite integrity=ok，训练更新0。91项冻结身份一致，新gzip完整CRC通过。旧账本与原始记录保留。

累计采样wall {res['wall_seconds']:.3f}秒、完整实验进程CPU {res['total_cpu_seconds']:.3f}秒；收尾保守预留另外wall60/CPU1200秒，不能伪称精确测量。合计RSS监测峰值{res['peak_conservative_rss_sum']/1024**2:.3f}MiB。停止核验与报告开销独立说明。

本次授权已消耗且终止；模型加载累计5次达到上限。任何进一步运行需新的具体修复/恢复方案及明确授权，当前不启动后续阶段、重试或扩样。

终态报告：{HERE/'report.md'}
机器决策：{HERE/'research-decision.json'}
旧状态已完整存入同目录previous-current-research-state.md/.json。
'''
(PLANS/'CURRENT_RESEARCH_STATE.md').write_text(new_md, encoding='utf-8')
state = dict(old_state)
state.update(decision=decision['decision'], reason=s['error'], stage2='C_recovery_technical_stop', overall_goal_achieved=False,
             formal_matrix_authorized=False, current_execution_authorized=False,
             final_result={'path':str(HERE/'report.md'),'sha256':sha(HERE/'report.md')},
             historical_final_result=old_state.get('final_result'),
             terminal_recovery=decision,
             new_consumption_this_status_update={'environment_steps':0,'model_forwards':0,'updates':0,'attempts':0})
state['recovery_execution'] = {**old_state.get('recovery_execution',{}), 'status':'technical_stop', 'authorization_consumed_no_retry':True,
                             'new_complete_episodes':4603,'complete_total':8637,'incomplete_episodes':1,'not_started':962}
state['evidence'] = old_state.get('evidence',[]) + [{'path':str(HERE/'research-decision.json'),'sha256':sha(HERE/'research-decision.json')}]
for item in state['evidence']:
    if item.get('path') == str(PLANS/'CURRENT_RESEARCH_STATE.md'):
        item['sha256'] = sha(PLANS/'CURRENT_RESEARCH_STATE.md')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
write(HERE/'output-hashes.json', [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='output-hashes.json'] +
      [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [PLANS/'CURRENT_RESEARCH_STATE.md',state_path]])
print(json.dumps({'report':str(HERE/'report.md'),'decision':decision['decision'],'total_counts':total_counts,
                  'artifact_bytes_before_report':artifact_bytes,'outputs_hashed':len(read(HERE/'output-hashes.json'))},ensure_ascii=False))
