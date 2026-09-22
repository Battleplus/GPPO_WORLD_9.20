# H-20260922-ACKGUARD-RESUME-001

状态：APPROVED。默认同意仅限既有明确范围。批准依据为用户当前活动目标：runner 最小修复、原预算有条件续行、证据审查与选择性进度归档。执行者为自定义 luna_worker / gpt-5.6-luna / max。主代理负责独立审查与 GitHub 归档。

## 唯一目标和路径

修复 ACK 已知任务基线 runner，核验已有两条 guard-on，只有完整性证据足够时完成原 24 条 guard-on 与 24 条历史 R 配对矩阵。

工作树 `E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt`；runner `tools/run_ack_known_task_guard_baseline_v1.py`；阶段 `runs/finite-communication-ack-lease-fix-20260920/ack-known-task-guard-baseline-v1`。使用阶段 manifest 的历史 M10 25-action source/config/snapshot/frozen policy/world/reward 身份；checkpoint SHA-256 `bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a`。不切换到当前有限通信环境或 D-02 模型。

输入为阶段 protocol、branch/control-reuse/source-checkpoint/snapshot manifest、原始 decision/step/branch ledger、first-pair 检查、budget.sqlite3 和哈希清单；先检查适用 AGENTS、当前进程和后续完成制品，避免重复运行。

## 最小修复

1. 首对是一条 guard-on 与对应历史 R，而不是两个动态 guard 分支。
2. 分支预算增量、阶段累计和完成分支总账分别核验。
3. 快照按 prefix_id 获取并比较真实语义状态；不使用遗留变量或字符串非空检查。
4. 每分支前向调用用计数增量；累计计数单列，不直接相加。原始字段保留，更正视图附来源。
5. 首步反馈关联实际 step。
6. 续行按唯一 branch_id 跳过已完成身份；部分记录或未决预留不能静默跳过。

只允许 runner、必要回归测试、独立修复/续行证据目录的变更；你不独自在代码库，不撤销他人编辑。保留失败现场，不覆盖旧报告。纯规则、样本、环境、模型、奖励及统计合同不变。

## 核验和条件续行

先完成不调用真实 env.step 或模型 forward 的回归测试，覆盖首对、累计与增量、快照隔离、前向计数、重复/缺失身份、缺标签、缺对照和续行跳过逻辑。

对 parent-00 repeat-0/1 独立核验身份、初始状态、逐步合法动作、实际动作 hidden 对应、标签复算、终止及预算。重新加载快照未变不等于旧进程当时未变；不得补写历史证据。证据不足或真实污染则保存阻断报告并停止，不重跑。证据足够才自动续行剩余 22 条。

## 硬预算

使用原阶段 budget.sqlite3，总上限 384。最近历史消费 20 verified，unknown/pending=0，必须现场核对。20 步不退款。剩余 22 条每条 <=16，新消费 <=352。历史 R 对照 24 条/299 步仅复用，不计新增。不得 reset、前缀重放、新建预算、转移额度、补样或自动重试。所有实际 step 逐步预留核销；真实续行 attempt 如实登记。所有 optimizer/world/offline update=0；模型前向只用于规定分支执行。

## 停止条件

已有后续完成制品或冲突进程；身份/账本无法对应；unknown/pending 未解决；已有两条污染、错误 hidden 或完整性无法核验；需重跑/额外预算/改变合同；非法动作、状态污染、标签缺失、计费错误。正常通信拒绝和任务失败是有效结果，不因结果差而停止、换样或重试。

## 产物与验收

新建修复续行证据目录，包含 repair-report.md、minimal-diff.patch、回归测试输出、completed-branch-integrity.json、resume-manifest.json、续行逐步/分支记录、paired-analysis.json、预算前后快照、final-report.md、hashes.json；不能执行时交付具体阻断和不适用产物说明。

完整矩阵验收：24 个唯一 guard 分支完整，正确对应 24 个历史 R，标签与原账本一致，原20与新消费全部计入，unknown/pending=0，更新0。主指标为 (0.8,0.2) 下 guard-R 父场景宏平均效用，先平均每个 parent 三重复，再对8个parent等权。拒绝、命令、NOOP、任务完成/过期、物理到达、主机确认、能耗与计算分别报告。

只作 W1 同批开发性结论，不称世界模型增量、偏好学习成功或生产收益。回传必须带本 handoff_id、路径/哈希、实际消费、失败/未知项。主代理审查后 CLOSED（完整矩阵或技术阻断）或 NEEDS_USER_DECISION；不自动下一阶段。

## 归档授权与分工

执行者不提交、不push、不连服务器。主代理独占独立进度工作树，在 Battleplus/GPPO_WORLD_9.20 的 archive/research-progress 上选择性提交本单文档、报告、必要diff、索引/哈希。当前用户第十一项明确授权该范围，优先于历史“禁止push”的一般限制。不修改main、不force、不Release、不上传原始大制品/权重/凭据/无关修改。不改全局Git配置或关闭既有签名。失败准确记录；只有远端提交和文件回读一致才报告归档核验成功。
