# 训练监督与损失接线审计

本轮仅离线读取已保存的 train lean labels、相应 58 个小型 gzip 窗口轨迹、预测/门结果作五例参照及冻结源码。没有加载 checkpoint、执行 forward、采集或训练。原 `offline_prediction_gate_stop` 保持不变；已查看评价父场景仅为开发材料。

## 结论

没有发现这五类主要后果的系统性标签缺失或配对错位：训练 split 共 96 个窗口，其中 58 个有机会窗口、38 个 no_opportunity；仅 58 个机会窗口产生 257 个合法候选标签，分布在 24 个父场景。257 个候选与原始分支轨迹完成逐候选核验。共同配对身份（公开状态、前缀、隐藏环境、RNG/exogenous key、采样器、续行规则和环境摘要）匹配；全部 trace 从原窗口时点起始，并用相同固定 continuation 结束或到 horizon。逐候选 utility、目标及其他任务生命周期差值、能耗/电量/损毁/竞争标签与原始分支记录一致，unknown 保持 unknown。no_opportunity 不作为失败标签或监督候选。

因此五例应判为**已有全局增量监督、并形成同窗排序监督时仍出现的预测错误**；但“当前任务获益同时其他任务受损”覆盖只有 `3` 个候选、`3` 窗口、`3` 父场景，特定外部性机制样本较稀。该证据不能证明梯度实际推动方向，也不能排除覆盖广度或目标尺度问题。冻结权重均值有候选数效应，但无逐头训练梯度/优化 loss 记录，不能据此认定 loss 权重是已证实根因。

## 原始后果到 loss

见 `outcome_to_loss_mapping.csv`。关键合同如下：

| 项目 | 源合同与核验 | 标签/mask/loss |
|---|---|---|
| 同初态和配对 | 训练 58 窗口的原 gzip hash 均匹配清单；同一 `pair_identity` 绑定 candidate 与透明分支的 public/prefix/hidden/RNG/exogenous/sampler/environment 摘要及 `public-joint-opportunity-rule/1`。257/257 分支 `restore_verified` 且 trace hash 与 label 对应 | 候选身份按 action index 对齐；透明 baseline 的全局增量为 0 |
| 结束、折扣 | 每个 trace 时间连续、从窗口时点开始，固定续行最多 18 步，合法结束为 horizon 或 terminated；两支按分支起点以相同 `GAMMA=.99` 折扣 | `utility = sum(.99^t * (.4*reward_A + .2*reward_B))`; 保存的全局差值逐项复算 257/257 通过 |
| 任务结果 | lifecycle 按 task_id 对齐；on-time/expired 独立；未结算是 censored unknown。host confirmation 未观察到为 unknown，不记负；collision 由于模拟器未实现，257/257 为 unknown | task_id 经 `slots` 回填六槽；NOOP 的当前任务头 mask=false，其他任务和全局标签仍有效 |
| 聚合后果 | 其他任务完成/过期增量由逐任务 paired lifecycle 差求和；energy/battery/damage/competition 按同一 candidate-baseline 分支差值 | 连续头 valid mask；unknown 数值占位 0 不进 loss |
| 目标和 reduction | 冻结权重：binary 1、continuous 1、per-task 1、ranking 1、latent 1、public 0.1、anticollapse 0.01 | binary BCE、其余监督头 SmoothL1(beta=1)，各自先对有效标量均值后相加；无 class balancing、无全局/逐任务一致性正则 |
| 排序 | 同一 parent/window 中全局真值有效且不相等才成 pair，方向为 sign(truth_i-truth_j)，严格平局跳过 | softplus ranking 对 batch 中全部有效不等 pair 平均；pair 身份键不含 repeat，但本 train split 无 parent/window 重复，故本次无跨重复混 pair |

## 训练覆盖

分母以候选计数描述标签覆盖，同时给出唯一窗口和父场景簇；候选不是独立样本。完整明细按时点及 NOOP 在 `training_coverage_by_time_branch.csv`，按二元结果时点在 `training_binary_by_time.csv`。

| 训练监督 | 候选 | 窗口 | 父场景 | 同窗有更好候选 | 说明 |
|---|---:|---:|---:|---:|---|
| 按时完成→过期的目标任务 | 9 | 8 | 7 | 9 | 每个都形成全局排序对 |
| 目标任务完成增量>0 且其他任务过期增量>0 | 3 | 3 | 3 | 3 | 联合机制覆盖稀少；共形成 17 个严格更优对 |
| 全局增量≤−0.1 | 25 | 17 | 14 | 25 | 形成 80 个更优比较，坏候选都可由更好动作排序 |
| 任一其他任务过期增量>0 | 21 | 14 | 13 | — | 这是该事件单独的监督覆盖，不要求同时有目标收益 |

target `on_time` 有效正/负为 `184/15` 个标签，分别分布于 `24/10` 父场景；`expired` 正/负为 `15/184`，分别来自 `10/24` 父场景。另 58 个 NOOP target 是正确 unknown，而非负例。Collision 全部 unknown；host-confirmation 没有有效 false 标签，这是观测删失合同而非错标。

训练有 `257` 个候选、窗口候选数从 `2` 到 `9`，均值 `4.43`。flatten 后的回归标量按候选数线性加权；排序共有 `499` 个不等真值 pair，单窗口从 `1` 到 `36` 对，窗口权重随候选数近似二次增长。candidate-count 分布见 `training_candidate_weighting.csv`。

## 五例参照与下一步

五个已查看评价重大损失中，3 个是其他任务新增过期，2 个是目标任务从按时完成变为过期；全局增量都被高估。训练集具备相同标签种类，但“目标正收益+其他过期”仅 3 个候选/3 个父场景，“目标转过期”9 个候选/7 个父场景。标签类似不等于同一场景机制。25 个训练严重负增量候选都有更好动作配对，因此不是全局坏标签完全缺失；模型却仍有泛化/拟合错误。各头等权也不等于梯度等影响；可见到按候选和 pair 归一化的不均，但没有保存 optimizer-step 的逐头 loss/gradient 记录。`head-metrics.json` 是离线预测指标，不是训练梯度证据。

**唯一建议：预注册一次“按窗口等权的 pairwise ranking”小型消融。** 保持原数据、种子、架构、其余损失和训练预算不变；只将当前 batch 全部 candidate pair 平均改为先对每个窗口内有效 pair 求均值、再对窗口均值求平均。先在既有已查看父场景上作开发诊断，不据此改门；最终判断必须使用未查看的评价父场景。这个实验检验已证实的窗口 pair 数不均是否影响排序，不预设会改善 major loss，不追加 tail loss、阈值或数据筛选。若不具备未查看评价父场景，则只报告开发消融，不声称确认。

该建议是对已确认的 loss reduction 不均衡作最小因果检验，不表示现有门失败已被此机制解释。原 `offline_prediction_gate_stop` 及全部负结果保持原样。
