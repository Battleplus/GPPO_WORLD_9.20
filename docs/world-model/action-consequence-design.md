# 候选动作后果世界模型设计（v1）

## 动机

旧 M-10 世界模型主要预测下一步 reward/event/done/context，并在候选动作之间形成平均 context；R3 已显示 event/done 质量不足以支撑可靠触发。本阶段改为显式预测“当前可见快照下选择某个合法 UAV–Task 动作后的任务后果”，首轮保持周期决策，不引入新的事件触发因素。

## 在线输入与输出

实现：`gppo_world/consequence_model.py`。模型复用冻结图编码器，但对每个合法候选保留独立的 action embedding 和候选行，不把候选 context 平均成一个动作无关向量。

每个候选在固定 `horizon_steps` 窗口内输出：

1. `travel_time`：移动/到达所需时间；
2. `service_progress`：服务推进量；
3. `energy_delta`：移动、等待/服务相关能耗变化；
4. `deadline_risk`：窗口内 deadline 违约概率。

前三项输出均值和 log-variance，第四项输出 logit 并经 sigmoid 得到风险；不确定性作为独立记录。候选评分只是辅助特征，调用方仍保留原 GPPO reward 和执行安全门禁；不实现 reward shaping。

## 标签与反事实数据

`ConsequenceTarget` 只记录 episode/decision/action、预测窗口、共享外生随机流标识和后果标签。隐藏状态可以用于仿真分支生成标签及离线核验，但不进入线上特征。事实与反事实分支复用同一外生随机条件，并分别记录 `source=simulator-counterfactual`；模型预测不能回写为监督标签。

新阶段的线上输入采用 `m10-graph5-5type-25action`：UAV、Region、Target、Task、Event 五类公共节点，24 个 UAV–Task 候选和 NOOP。`gppo_world/graph5.py` 是独立适配器；旧 `GraphSnapshot` 的 3 类节点/17 动作接口及 checkpoint 不可直接复用。观察窗口未覆盖完整到达或 deadline 时，对应连续/风险标签通过 `label_masks` 标为删失，不把删失当作完成或零风险。

数据分为 episode-disjoint 的 train/validation/test/OOD。test/OOD 不用于选择阈值、权重或 checkpoint。损坏、错误、泄漏、协议不兼容样本隔离并记录原因；真实超时、损毁、能量不足仍保留为负样本。

## 损失与质量检查

连续后果分别使用异方差 Gaussian NLL，deadline risk 单独使用 BCE-with-logits；报告每个头的损失、标签分布、尺度、校准、RMSE/MAE、PR-AUC/Brier/ECE 和持久性/均值基线。不得因为某个头使用 BCE 或 MSE 就预设所有问题属于同一损失原因。每个 run 保存实际 optimizer 更新数与数据样本数。

## 控制接口

首轮策略为：原 GPPO logits + 固定系数的辅助后果分数；系数在 validation 冻结，不能用 test/OOD 调参。若不确定性过高，走原有合法拒绝/安全门禁路径，不自动获得执行许可。未来若研究后果驱动重规划，另开单因素协议，不能把本模块的辅助评分写成已有触发能力。
