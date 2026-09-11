# Experiment Plan

**Problem**: 在无人机弱通信任务调度中，世界模型是否能在合法公开信息和历史信息基线之外，预测候选 UAV–Task 动作后果并改善调度。  
**Method Thesis**: 保留候选动作差异，预测移动/服务/能量/deadline 后果及不确定性，将其作为原 GPPO 的辅助候选特征。  
**Date**: 2026-09-11

## Claim Map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1：候选后果预测提供历史信息之外的调度价值 | 同协议、同动作/通信/安全门禁和同策略预算下，GPPO＋历史＋预测在主要任务指标上优于 GPPO＋历史，且置信区间/逐 seed 证据不被额外成本抵消 | B1, B2 |
| C2：收益来自可校准的候选后果，而非更多计算或筛选偏差 | 各预测头超过持久性/均值基线，test/OOD 无泄漏，预测成本和总成本单列，消融去掉任一后果头后价值下降或有明确边界 | B2, B3, B4 |

## Paper Storyline

- 主线必须证明：传统合法调度器、GPPO、历史信息、候选后果预测在同一新合同下的可比性。
- 附录支持：逐事件 ledger、校准曲线、OOD 失败类型、硬件延迟分解。
- 有意切除：首轮事件触发、reward shaping、搜索参数扩展、额外 checkpoint 选择；避免多因素同时变化。

## Experiment Blocks

### B0：合同与数据 sanity（MUST-RUN）

- Claim tested：任务、标签和信息隔离正确。
- Compared systems：数据 schema、协议 validator、标签泄漏检查。
- Metrics：合法动作率、标签有限性、episode-disjoint、共享外生随机 key 一致性。
- Setup：不训练；单机测试和小型固定 tape。
- Success criterion：0 泄漏、0 非法动作、0 非有限标签；否则停止。

### B1：同合同策略矩阵（MUST-RUN）

- Claim tested：候选后果预测是否改善调度。
- Compared systems：合法传统调度器、GPPO、GPPO＋历史、GPPO＋历史＋候选后果预测。
- Setup：相同环境、通信、ACK/lease/fencing、能量、安全门禁、tape、每 seed actor budget=8192；3 seeds=1101/2203/3307；世界模型数据采样/预训练成本单列。
- Metrics：完成率、deadline 违约、回报、能耗、恢复事件/时间、正确拒绝、安全违规、通信代理量、actor/world 调用、完整决策延迟。
- Success criterion：主任务效果不劣于历史基线且在预先冻结的成本约束下有明确改善；否则报告负结果。
- Failure interpretation：预测可能无额外价值，或标签目标与控制目标不一致；不得自动追加预算。

### B2：预测质量与校准（MUST-RUN）

- Claim tested：模型真的预测候选后果，而非只提供装饰性 context。
- Metrics：travel/service/energy 的 MAE、RMSE、Gaussian NLL、覆盖率；deadline risk 的 BCE、PR-AUC、Brier、ECE、precision/recall；与均值、持久性基线比较。
- Setup：固定 train/validation/test/OOD，test/OOD 只在选择冻结后评估；报告标签阳性率、尺度、缺失 mask、实际 optimizer updates。
- Success criterion：至少一个与控制目标对应的核心头超过简单基线，且 OOD 退化被如实报告；不要求所有头都由同一种损失解决。

### B3：机制消融与总成本（MUST-RUN）

- Compared systems：候选后果全头、去掉 travel/service/energy/risk 各一项、随机/均值候选特征、历史信息但无预测。
- Metrics：效果—成本前沿、world calls、actor calls、总训练/采样/推理时间、完整链路 latency。
- Success criterion：额外效果不能只由更多采样或搜索预算解释；若成本不值得，结论为不采用。

### B4：OOD 与失败分析（NICE-TO-HAVE，若 B0–B3 通过）

- Data：未按旧策略失败筛选的 OOD 通信延迟、丢包、重排、能源/损毁组合。
- Metrics：分布外校准、过期/拒绝/安全行为、deadline 失败类型和逐事件差异。
- Gate：任何安全违规或信息泄漏立即停止，不以任务回报抵消。

## Run Order and Milestones

| Milestone | Goal | Budget | Decision gate |
|---|---|---:|---|
| M0 | B0 合同、标签、兼容性和小型过拟合 sanity | 本地；不训练完整模型 | 任一泄漏/非法动作/NaN：停止 |
| M1 | 合法传统调度器与 GPPO/History 单 seed pilot | 512 actor decisions/系统 | 计时、optimizer updates、结果记录完整才进入 M2 |
| M2 | 四组 3-seed 训练/评估 | 8192 actor decisions/seed；最多 128 PPO updates/seed | 预先冻结；不按结果追加预算 |
| M3 | 后果模型质量与机制消融 | 与 M2 分开计预算 | 核心预测头不超过简单基线则不做 B4 |
| M4 | OOD/汇报归档 | 只在 M2/M3 通过后 | 生成报告、ledger、hash、Release；不改历史 |

## Compute and Data Budget

- 学习组统一策略交互预算：3 seeds × 8192 actor decisions；传统方法报告相同 tape/任务预算，不访问隐藏真值。
- 世界模型采样、预训练和在线推理成本分开；预训练最多 30 epochs、validation patience=5，实际更新数必须记录。
- 服务器要求：可记录 GPU/CPU、Torch/CUDA/依赖版本；本机只做测试和静态检查。
- 最大风险：标签语义（下一窗口后果）与真正需要重规划的后果不一致。

## Risks and Mitigations

- 泄漏：隐藏状态进入线上特征；用 schema validator、字段白名单和 twin-branch audit 阻断。
- 标签偏差：只从旧策略失败轨迹取样；按完整冻结 tape 采样并保留真实负样本。
- 成本误判：只看 actor calls；单列 world calls、总计算和完整 latency。
- 任务语义漂移：到达即完成与持续服务混用；协议版本化，首轮保持旧奖励。
- 安全问题：违规、错误 ACK、失效 lease 继续执行；立即停止并保留 ledger。

## Final Checklist

- [ ] 合同版本和动作/观测/归一化已冻结
- [ ] train/validation/test/OOD episode-disjoint
- [ ] 预测与仿真标签分离，隐藏状态未进入线上策略
- [ ] 传统、GPPO、History、Consequence 四组同预算
- [ ] 每个损失头、校准和简单基线单独报告
- [ ] 总成本和新硬件延迟单列
- [ ] 无新结果时不写成通过；负结果完整保留
