# M-10 到达语义世界模型计划

## 主张与反主张

- 主张：在到达区域即完成的弱通信合同下，候选后果先验能在合法公开历史之外改善动作选择。
- 支持主张：预测到达时间/截止前到达和执行失败，比系统总能耗更直接解释任务选择。
- 反主张：规则调度器已经能从公开距离、能量和 deadline 准确排序，学习模型只增加成本。

## 冻结合同

协议：`world-gppo-9.11-arrival/0.1.0`；Graph-5、25 action、周期决策、continuous weak communication、ACK/lease/fencing/energy gates 全部保持。旧 `continuous_service_until_deadline` 不改写。物理到达和主机确认 deadline 分开运行，主口径由会议确认后再冻结。

新后果标签从同一合法公开前缀分支得到：`arrival_time`、`arrival_before_deadline_physical`、`arrival_before_deadline_host`、`execution_or_energy_failure`；未到达窗口、拒绝、NOOP 和未确认分别 mask/分类。系统总能耗仅为辅助字段。隐藏状态只进仿真标签，不进 Graph-5 或 history。

## 实验区块

### B0：功能与规则基线（已完成）

- 16 个共享 composite tape；传统调度器只读公开信息。
- 两种 deadline basis 分开记录，未训练。
- 物理到达平均 3.0000 个任务/episode；主机确认平均 0.5625；安全违规 0。

### B1：新标签质量（MUST-RUN）

- train/validation/test/OOD 按父 episode/前缀分组；不得跨 split。
- 预测指标：到达时间 MAE/删失覆盖、deadline Brier/校准、执行失败有效分母、同状态候选 pairwise accuracy 和 selection regret。
- 基线：公开距离/速度、剩余能量、任务 deadline 的物理估计；持久性只在语义适用时使用。
- 通过条件：至少两个核心目标超过最佳简单基线，且候选排序与规则差距有稳定改善；否则停止融合。

### B2：先验策略隔离（条件 MUST-RUN）

- 合法传统调度器、GPPO、GPPO-History、GPPO-History+候选后果先验。
- 保持原奖励，先验只进入候选评分/辅助特征；不平均候选，不加入事件触发或奖励塑形。
- shared history、动作预算、tape 和安全门禁一致；预测模型冻结。

## 执行顺序与预算

| 阶段 | 内容 | 预算 | 决策门 |
|---|---|---:|---|
| M0 | 新语义小场景、两 deadline basis、无训练规则基线 | 已完成；16 tape | 合同字段与安全账本完整 |
| M1 | 新数据小 pilot 与标签审计 | 先登记；不超过 1 个 seed、4096 模型更新、1 小时 | 失败/泄漏/非有限即停止 |
| M2 | 新后果模型 train/validation 选模，冻结后评估 test/OOD | 1 seed 1101；最多 8 epochs、patience 3、4096 updates、1 h | 不超过简单基线则停止 |
| M3 | 单 seed 融合 pilot | 先另行登记，最多 512 环境步/学习组 | 仅在 B1 通过后执行 |
| M4 | 策略矩阵 | 新预算，不重置旧预算；3 seed×8192/学习组 | 仅在 M3 通过后执行 |

本轮没有启动 M1–M4 训练。旧总能耗残差路线和旧失败门槛作为历史负结果保留。

## 当前结论

新语义已经具备可执行状态转移和账本，但区域半径、主 deadline basis、完成消息责任方仍待确认。规则基线已显示两种口径差异很大，不能先选有利口径。新后果模型尚未证明额外候选排序价值，不能写成世界模型改善调度。
