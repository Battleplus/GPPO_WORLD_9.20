# Experiment Plan

**Problem**: 到达区域即完成的无人机弱通信任务调度  
**Method Thesis**: 让世界模型预测合法 UAV–Task 动作的到达和执行后果，并作为候选先验接受严格的排序与任务收益检验。  
**Date**: 2026-09-13

## Claim Map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1：新候选后果预测超越公开物理基线 | 核心标签误差/校准和同状态排序同时改善 | B1 |
| C2：预测先验带来任务调度收益 | 在相同奖励、tape、history 和预算下稳定超过 History/规则基线 | B2 |

## Paper Storyline

- Main paper must prove：新任务合同下的候选排序是否真的改善调度。
- Appendix can support：通信、确认延迟、故障和能量失败分层账本。
- Intentionally cut：系统总能耗残差路线、事件触发和奖励塑形，直到核心假设成立。

## Blocks

### B0 功能与规则基线（MUST-RUN，已完成）

- 新到达语义，两种 deadline basis 分开；16 个共享 composite tape。
- 结果：物理到达 3.0000 完成任务/episode；主机确认 0.5625；安全违规 0。

### B1 标签与预测质量（MUST-RUN）

- 按父 episode/前缀分组，公开物理估计、均值/发生率和适用持久性基线。
- 到达时间、deadline Brier/校准、执行失败、删失比例、同状态排序和 selection regret。
- 一 seed；最多 8 epochs、patience 3、4096 optimizer updates、3600 秒。
- 若核心目标或候选排序不超过简单基线，停止策略融合。

### B2 先验策略隔离（条件 MUST-RUN）

- 传统、GPPO、GPPO-History、GPPO-History+候选后果先验。
- 原奖励和安全门禁不变；预测按候选使用，不平均成全局 context。
- 单 seed 512 步 pilot 通过后，才登记 3 seed×8192 环境步矩阵。

## Run Order and Milestones

| Milestone | Goal | Decision Gate | Status |
|---|---|---|---|
| M0 | 合同/账本/规则 | 两口径可区分且安全字段完整 | DONE |
| M1 | 新反事实数据 | split/删失/失败/NOOP 审计通过 | TODO |
| M2 | 预测质量 | 超过简单基线且排序有价值 | TODO |
| M3 | 策略先验 | 单 seed pilot 无安全/预算异常 | TODO |
| M4 | 正式比较 | 逐 seed 与共享 tape 报告 | TODO |

## Compute and Data Budget

- 新模型：1 seed，最多 4096 updates、1 小时；不自动追加。
- 策略：只有 M2/M3 通过后另行冻结；旧预算不重置。
- 所有 split 按父 episode/前缀分组；正式 test/OOD 不用于选择。
- 最大瓶颈：主 deadline basis 和区域半径尚未确认。

## Final Checklist

- [ ] 主张由新任务合同直接支持
- [x] 传统规则基线已执行
- [ ] 候选后果预测超过简单基线
- [ ] 预测先验带来策略收益
- [x] 系统总能耗残差路线已明确切除
