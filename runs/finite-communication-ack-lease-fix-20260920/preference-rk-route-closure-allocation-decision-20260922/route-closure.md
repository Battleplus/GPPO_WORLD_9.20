# 偏好 R/K 路线封存与实际分配决策立项

日期：2026-09-22

## 封存结论

`preference-branch-evaluation-v1`、`preference-vector-label-pilot-v1`、`preference-vector-label-train-v1` 和 `preference-repeat-stability-audit-v1` 共同完成了 R/K 分支标签路线的审计闭环：

- 训练奖励合同、逐步二维标签和历史分支复现已经通过；
- train 覆盖 154 个 prefix、16 个 parent、924 条 R/K 分支、11,659 个标签步；
- R/K 之间存在任务—能耗取舍信号，并有 17 个严格跨固定偏好的平均效用翻转 prefix，覆盖 9 个 parent；
- 但只有 6/17 个平均翻转 prefix 在三次重复中均严格翻转。留出重复相对始终 R 的 parent-macro 差值为 `(0.000980, 0.000984, 0.001037)`，对应偏好 `(0.2,0.8)`, `(0.5,0.5)`, `(0.8,0.2)`；后两档 bootstrap 区间跨零；
- 因此跨重复稳定性没有达到预登记推进条件，停止 validation 补采和 R/K 选择器训练。

保留能耗偏重档的小幅探索信号，但不升级为确认性收益。`hindsight - better constant` 与 `hindsight - leave-one-repeat` 是不同量：前者分别为 `0.00243473565685/0.00409426177686/0.00582719340518`，后者分别为 `0.00145452578905/0.00310986567515/0.00478984307778`；前者不能被当作留出策略收益。

公开信息是否能够预测 R/K 选择尚未检验，不能写成“不可预测”。本路线停止的是 validation 补采和 R/K 选择器训练，不是对偏好学习、世界模型或 GPPO 整体作否定。

## 已关闭路线

- R/K 选择器：封存。原因是重复稳定性不足，而不是标签不可用。
- 事件触发训练原配方：保留公平复现负结果，不再按同一配方扩 seed 或调参。
- `candidate-conditioned-feedback-authorization-v1`：保留独立原型和测试，但完整环境适用机会确认数为 0；I/W1 仍有无法确认记录，不能写成所有机会已被证明不存在。正式环境未集成。
- D-02 冻结 latent adapter 的 candidate residual 解释：撤回。它是 T-05 原生 17-action 的 global latent actor residual，不是按 UAV-Task 对齐的候选后果模型，不能用于 M10 25-action 候选收益结论。

## 仍未回答的问题

R/K 只表示“重新调用冻结策略”与“合法续行一个控制周期”，不是全部 UAV-Task 分配候选规划。尚未有同一真实 M10 决策前缀中两个合法非 NOOP 分配动作的严格配对证据，能够同时回答：

1. 两个动作的任务、能耗或 deadline 后果是否真实不同；
2. 差异能否跨外生重复保持；
3. 决策时公开信息能否预测差异；
4. 一个候选后果模型或 world model 相比公开信息基线是否有增量。

这四个问题必须分门验收，前一门通过不等于后一门通过。

