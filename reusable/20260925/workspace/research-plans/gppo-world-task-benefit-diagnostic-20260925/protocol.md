# 独立任务收益诊断：已授权，执行前冻结

唯一 attempt：`independent-task-benefit-20260925-once`。用户于本任务明确批准零步实现、测试、冻结及 B→C 条件执行，无需再次逐项询问。本实验不是原失败运行的恢复。原 `stage3-runtime-minimal-paired-v2-once` A 平均控制 CPU 10.386904761904763 ms > 10 ms，失败及实际消耗全部保留。

## 问题、固定身份与证据复用

回答冻结策略移除在线 world 和 candidate_actor 输出后是否保留任务收益，以及相对强规则是否仍有实用增量。N 曾接受含 world 的训练，直接移除造成损失不能证明 world 训练或预测语义的独立贡献。

原 800 父场景、各自唯一 repeat、执行顺序及外生键直接复制原 sample-manifest.json，按原 namespace 的身份散列复核，不重新选样。固定原生历史 M10 25-action/W1、seed1101 checkpoint。复用对应旧 M/R 1600 条结果。不是新盲测，不估计父内重复稳定性或训练 seed 泛化。

原 A 的 21 批数值等价与 full 历史回放证据只读引用，禁止重试 A。base_worker、模型数学函数、N actor、原生环境、奖励、记账与 safe observer 直接复用。runtime_worker 的 choose/one_step/reset 方法不变；只增加入口授权限制（仅 N/B/C，禁止 replay），改变输出目录。本包可改的部分限编排、授权、资源记录及任务与成本结论分离。

## 条件执行

B：原技术生成器与 base_seed=925310000，两个 W1/mixed 正常 reset episode，各最多 18 步。仅检查初态与 hidden/公共记忆独立初始化、原生合法动作和提交路径、无 world 调用、奖励标签独立复算、原生终止、真实计数及账本。不以效用或 10 ms 决定继续。B 技术通过自动 C。

C：原固定 800 个 N episode，各正常 reset，与同 parent/repeat 的旧 M/R 初始公开状态必须匹配，各最多 18 步。全部完整才分析；中途只观察独立进度文件及资源，不揭盲部分效果。不重跑、换场景、补重复、调参或训练。

任何身份、初态、配对、接口、公开 guard、标签、账本、日志、终止或资源异常立即停止，保留已花费与 unknown；不自动重试或恢复。不得继续成本 A，但已知 10 ms 失败不触发本次任务诊断中止。资源总量硬门仍随调用检查。

## 统计与研究决策

效用沿原 float32 vector reward 与折扣：Σ0.99^k(0.4*r_task+0.2*r_energy)。800 父场景为独立重抽单位，同一次配对重抽用于各终点，10000 次 bootstrap，seed=20260925。

- 保留效用：N−M 95% CI 下界 >−0.01。
- 相对强规则实用增量：N−R 95% CI 下界 >+0.01。
- 任务护栏：N−M 物理按时和主机按时 95% CI 下界各 >−0.02。
- 明确损失：效用/物理/主机三项沿用 Bonferroni 98.3333% 双侧 CI，任一上界低于 −0.01/−0.02/−0.02。

任务保留、护栏、规则增量全部成立则支持“准备简化方案的任务依据”，不表示简化方案实用验收或部署通过。明确损失则否定直接移除；其余不确定后暂停。不得通过单项显著代替联合条件。

原成本阈值全部不变：10 ms CPU、50 ms wall p95、4 GiB RSS、三个 N/M CPU 比值各 ≤0.8。继续记录 C 本身及匹配历史 M 的成本，A 比值复用原日志；跨运行系统负载和计时范围差异必须披露。无论 C 成本怎样，`prior_A_cost_failure_preserved=true`、`practical_acceptance_pass=false`，不能把任务通过称为成本通过。完整进程 CPU/wall 包括启动、加载、检查、记录和分析。

## 分项预算（不可互借）

|资源|B|C|总计|
|---|---:|---:|---:|
|环境步 / encode / actor 各|36|14400|14436|
|reset|2|800|802|
|模型加载|1|1|2|
|world / rule / 所有更新|0|0|0|
|wall|600 s|10800 s|11400 s|
|完整进程 CPU|2400 s|43200 s|45600 s|
|新增制品|0.5 GiB|3 GiB|3.5 GiB|

RSS 为父进程及全部同时驻留 worker 合计 4 GiB，暂停 worker 也计入；用各驻留进程峰值相加作保守限制。每阶段含 wall 30 s / CPU 600 s 关机预留，正常运行在预留前停止。禁止历史余额抵扣。旧阶段 2 + 原失败 A + 本次诊断累计报告；测得进程时间和尾部预留上界分开，不把上界当精确测量。零步准备/测试成本另列，计入总资源保守核算。

## 运行环境与调用

复用已验证的 Windows / `C:\Python314\python.exe` 3.14.4、torch 2.13.0+cpu，intra=4/inter=1；不重建环境，不安装依赖，不执行额外 kernel/model witness。绑定、权重与原生源码按旧 manifest 复核。其他项目进程不由本任务管理，未保证独占机器。

零步验证（标准库测试，不导入 torch/numpy/环境、不加载 checkpoint）：

```powershell
C:\Python314\python.exe -B -X utf8 -m unittest discover -s E:\Z博士\research-plans\gppo-world-task-benefit-diagnostic-20260925 -p test_task_diagnostic.py -v
```

授权和执行身份冻结后仅一次调用：

```powershell
C:\Python314\python.exe -B -X utf8 E:\Z博士\research-plans\gppo-world-task-benefit-diagnostic-20260925\run_diagnostic.py --authorization E:\Z博士\research-plans\gppo-world-task-benefit-diagnostic-20260925\authorization-approved.json
```

运行期间只能用复用的 safe_observer（Windows READ/WRITE/DELETE 共享）读取独立 `observer-progress.jsonl`，不得用普通共享不明的读法访问正式 episode/step/SQLite 文件。待进程终止后只读做最终核验、完整报告。无 Git 发布、后续实验或训练授权。

零步准备命令不另借预算：B 内保守扣留 wall 60 秒 / CPU 120 秒用于准备、测试、身份冻结进程开销（非精确测量；不把用户/助手等待计作计算进程时间）。运行监控先扣留此数，再保留关机余量；不得将此上界说成实际测量。
