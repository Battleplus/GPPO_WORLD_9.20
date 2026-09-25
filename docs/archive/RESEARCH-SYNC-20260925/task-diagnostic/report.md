# 独立任务收益诊断最终报告

1. 移除在线世界模型后的任务收益：支持保留（效用非劣与两项任务护栏均通过）。
2. 相对强规则：有实用任务增量。
3. 研究决策：支持准备简化方案，但成本实用验收仍不通过。

原 A 的 10.386904761904763 ms > 10 ms 成本失败保持不变，未重试 A。以下任务判断不构成成本通过或部署合格。移除损失不证明世界模型独立贡献。

|配对终点|均值差|95%父场景CI|
|---|---:|---|
|N-M energy_used|0.003560195|[-0.005919660, 0.012146958]|
|N-M host_on_time_observed|-0.000833333|[-0.002083333, 0.000208333]|
|N-M physical_on_time|0.000625000|[0.000000000, 0.001458333]|
|N-M utility|0.000410361|[-0.000031782, 0.000995263]|
|N-R energy_used|0.964285333|[0.906091396, 1.021916922]|
|N-R host_on_time_observed|0.089583333|[0.078750000, 0.100416667]|
|N-R physical_on_time|0.091041667|[0.082083333, 0.100208333]|
|N-R utility|0.063441822|[0.056856860, 0.070185049]|

明确损失的家族校正区间：`{"host_on_time_observed": [-0.0022916666666666675, 0.00041666666666666686], "physical_on_time": [0.0, 0.0016666666666666663], "utility": [-6.0896082880856354e-05, 0.0011427323151727612]}`。

800 个原先固定父场景各一个外生 repeat；C 新增 N 800 条，复用 M/R 1600 条，B 两条技术 episode 不纳入效应。10000 次同步父场景 bootstrap，seed=20260925。没有按结果选样、扩样、训练或后续实验。

本次实际环境步 10540，encode/actor 各 10540，reset 802，加载 2；world、规则及更新均为 0。账本 reserved=verified，无 pending/unknown，完整性 ok。

累计阶段2＋失败A＋本次计数：`{"actor_readout": 76166, "environment_resets": 10407, "environment_steps": 144202, "model_loads": 11, "offline_updates": 0, "optimizer_updates": 0, "policy_encode": 76166, "rule_decisions": 70661, "world_candidate_batch": 64576, "world_updates": 0}`。

新执行进程测得 CPU 966.093750 s、wall 622.531865 s，合计驻留峰值保守和 286212096 bytes；含准备、尾部上界与本次复核的计费上界 CPU 1686.828125 s、wall 713.269506 s。分项和总授权额度核对通过。上界不是精确测量。

N 控制成本：`{"cpu_seconds": 86.78125, "decisions": 10514, "matched_M_cpu_seconds": 98.6875, "mean_cpu_ms": 8.253875784668063, "paired_replay_ratio": 0.7931818181818182, "per_decision_ratio_to_matched_M": 0.8803576593734365, "per_episode_ratio_to_matched_M": 0.8793540215326155, "wall_p95_ms": 3.8244999988819472}`。跨运行成本比较仍受计时范围与系统负载限制；即使某项新成本低于阈值，整体实用验收仍不通过。

范围仅为历史 M10 25-action/W1、冻结 seed1101、已经公开的开发父场景；不支持有限通信部署或新父场景/训练seed泛化。N 保留了含 world 训练得到的权重，不是无 world 训练基线。

输入身份在执行前后复核；weights unchanged，旧 ledger 哈希未变。详细成本谱系、输入验证及机器可读结论见 verification.json；未提交或发布 Git。
