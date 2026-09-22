# 实际分配决策后续协议草案

状态：仅立项审查草案；未创建预算、未运行环境、未调用模型。

## 研究问题

在同一个真实 M10 决策前缀、相同公开 observation、相同合法 mask 和相同执行合同下，至少两个合法非 NOOP UAV-Task 分配动作的任务、能耗、deadline 后果是否不同且可重复；若不同，决策时公开信息或候选后果模型是否能预测这种差异，world model 是否在公开信息基线之上提供增量。

## 唯一环境与输入合同

- 只使用历史 R/K 采集对应的 M10 25-action source snapshot：`E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot`。
- action mapping 固定为 4 UAV × 6 task slots + NOOP；NOOP=24。禁止接入 D-02/T-05 的 17-action policy 或 mask。
- 保持原任务集合、通信 profile、TTL、rate/capacity/reserve、ACK/lease/fencing、completion、mask、安全约束、horizon=18 秒、decision interval=1 秒。
- 公开输入只能使用动作发生前实际可见的 observation、字段 age/valid、公开 mask、历史公开状态和已收到的合法消息。未来结果、hidden truth、未来能耗、外生 key、branch outcome 不进入在线动作输入。
- 行为策略、world/context、偏好和续行合同必须以运行前 source manifest 与 checkpoint manifest 冻结。现有 global latent adapter 不得冒称候选后果模型。

## 四道独立门

### 门 0：候选资格（零环境步）

在查看任何分支结果前冻结身份和候选选择规则，对已有可恢复公开前缀重建公开 mask。只保留至少两个合法非 NOOP UAV-Task action 的状态；合法动作包含 NOOP 的计数单独保存，不能把“NOOP + 一个分配”当作多候选。候选资格不足则停止，不放宽 mask、TTL、安全规则或通信容量。

有限通信诊断已显示 I/W1 的保存决策行中至少两个合法候选为 0、W2 没有公开 pending；这只能作为压力层和预警，不能替代针对历史 M10 原生数据的资格预检。

### 门 1：真实后果差异

对每个冻结前缀选择两个合法非 NOOP action。以同一初始状态、任务集合、公开前缀和 exogenous key 配对，只干预首个分配动作；后续沿相同冻结策略续行。记录动作依赖的 RNG 消耗差异，不假设两个分支使用相同的后续随机流。

使用已验证的逐步二维奖励记录器：

`vector_reward_k = [(new_success - new_deadline_failure) / task_capacity, -used_energy / (uav_count * initial_energy)]`

`G = sum(0.99 ** k * vector_reward_k)`，第一步 `k=0`，包含终止步；terminated 与 truncated 分开，截断不补零、不调用未经登记的 critic。固定评价偏好仍为 `(0.2,0.8)`, `(0.5,0.5)`, `(0.8,0.2)`，任务 scale=0.5，能耗 scale=1.0。行为执行不因评价偏好改变。

门 1 只要求预登记的任务/能耗/deadline 向量差异超过容差并能由逐步标签复算；不把标量 reward、R/K hindsight 或最终完成比例当作充分证据。

### 门 2：重复稳定

同一前缀至少使用 3 个预登记、身份不同的 exogenous key。优先用父场景内配对重复，按 parent 聚类汇总。通过条件必须在运行前写明，例如：至少 2/3 重复在同一主要差异方向，且父场景级方向不能由单一 parent 删除后完全反转；具体幅度容差应在数据读取前冻结。若只有一条重复或只在单个 parent 出现，不称为稳定。

### 门 3：预测增量

只有门 1/2 通过后才评价预测：

1. 公开基线：当前公开 observation、mask、动作身份和可合法使用的历史公开特征；
2. 候选后果模型：与公开基线相同输入边界，增加明确的 action-conditioned outcome head；
3. world model 增量：只在 world/context 身份与 M10 输入合同兼容且不泄漏未来信息时加入冻结 world 特征。

三者使用同一父场景隔离、同一标签和同一评价分母。公开基线没有候选时不能凭 truth 补候选。现有 D-02 global latent adapter 若不能按 M10 25-action 和 candidate consequence 接口对齐，直接标记不可用，不补零、不裁剪、不重新编号。

## 分支执行合同

每个分支保存：prefix identity、parent、condition、repeat/exogenous key、源码/checkpoint/config identity、公开 observation hash、mask hash、首 action、首 action 的 UAV/task identity、任务计数前后值、逐步 vector reward、energy 前后值、deadline/host confirmation、terminated/truncated/end reason、gamma/scale、后续 action、hidden/cache 摘要和通信审计字段。

首动作之外不做动作置换，不改变 hidden/cache 推进，不主动查询，不改变通信服务，不写入公开观测，不使用 offline truth。R/K 只作已封存路线的参考，不与新分支混合为独立候选样本。

## 选择、分层与隔离

先冻结全体候选池的身份清单和排序规则，再执行资格门。至少需要 8 个 parent-prefix 单元、每单元 3 个 exogenous repeats、每重复 2 个首动作分支；不得按收益、翻转、模型输出或后果标签筛选。W1/W2 若候选资格不足，只报告压力层，不强行纳入主要排序层。

开发/评价隔离采用 parent-level holdout；同一 parent 的前缀、重复、模型 seed 不能跨边界。训练预测器的任何后续阶段必须另行登记，不得把本草案视作训练授权。

## 预算建议（非账本）

若每条完整分支最多 18 个决策步，8 个 parent-prefix × 3 repeats × 2 branches 的保守上界为 `8 × 3 × 2 × 18 = 864` environment steps。若需要前缀重放，必须将重放步数逐条加入；若可审计 snapshot restore，只能把实际恢复与分支续行分开计数。正式预算必须根据冻结清单的实际 `prefix_replay + branch_continuation` 最坏值另行登记，不能借用 heartbeat、主账本或 pilot 余额。本轮预算消耗为 0。

## 指标与停止条件

主指标是配对的任务—能耗—deadline 后果差异；同时报告候选覆盖、A/B/C/D 分类、重复方向、父场景宏平均、普通/紧急任务、物理到达、主机确认、未完成/EXPIRED/damage/disconnect、token/排队/发送/接收和完整失败分母。多候选不自动等于真实 trade-off，hindsight 不等于部署收益。

任一情况停止：候选无法由公开 mask 稳定复算；两个 action 不是合法分配；后果差异由 truth 泄漏或过宽 mask 造成；重复方向不稳定；公开基线不存在但只靠 truth 造候选；动作、hidden、通信、ACK/lease/fencing/completion 合同变化；标签缺失；预测只改善 hindsight 子集而完整分母恶化；world 特征无法与 M10 合同对齐；需要改 TTL、reserve、mask、重试或通信权限。

