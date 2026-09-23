# H-20260923 世界模型增量可检验性预检

本轮为零环境步静态审查，范围是 H-005 已观察的 W1 开发矩阵。未加载 checkpoint、未构造环境、未调用 reset/env.step、未进行模型前向或 hidden replay，未修改 SQLite。H-005 数据不是 validation/heldout。

## 判定

判定为 **A（限定为 event-feature 输入关闭敏感性）**，完整 no-world-model 对照仍缺乏依据，不属于该 A 判定的范围。原生 H-005 路径存在可隔离的 candidate-conditioned world 到 actor logits 接口；guard 后仍有足够选择空间。现有保存记录没有逐动作 candidate features 或分量 logits，因此旧记录只能证明选择空间，不能归因旧动作排序。

世界模型不应被称为全局 latent actor residual：H-005 使用 `ActionConditionedTemporalWorldModel` 的 25-action candidate rows，并在 `JointGraphPreferencePolicy.evaluate_encoded` 中直接进入 `candidate_actor`。selected `by_action` hidden 只负责下一决策状态推进。

## 决策空间

H-005 immutable decision ledger 共 **279** 行、24 个分支。guard 后分类为：仅 NOOP **94**，恰一个非 NOOP **7**，至少两个非 NOOP **178**，空集合异常 **0**。含至少一个非 NOOP 的窗口为 **185**；NOOP 在 **279/279** 行保留。原始动作与 guard 最终动作不同 **120** 行，guard 触发 **120** 行。

| parent | decisions | NOOP-only | one non-NOOP | >=2 non-NOOP | guard changed |
|---|---:|---:|---:|---:|---:|
| parent-00 | 30 | 9 | 3 | 18 | 18 |
| parent-01 | 39 | 10 | 1 | 28 | 22 |
| parent-02 | 38 | 17 | 0 | 21 | 9 |
| parent-03 | 31 | 9 | 2 | 20 | 17 |
| parent-04 | 31 | 9 | 1 | 21 | 12 |
| parent-05 | 33 | 9 | 0 | 24 | 12 |
| parent-06 | 39 | 20 | 0 | 19 | 13 |
| parent-07 | 38 | 11 | 0 | 27 | 17 |

公开合法候选、执行器接受和实际完成是三个不同层次。本表只统计保存的公开 mask 与 guard 候选集合；没有对未执行动作推断 accepted、物理完成或 host confirmation。

## 数据通路

- `joint_gppo.py:100-101,125-128`：public observation 与 policy hidden 产生 actor features、pair messages 和 next policy hidden。
- `joint_gppo.py:131-185`：world 对每个 action 使用 action embedding、relation 和 world hidden，产生 policy feature、vector reward、task consequence、event probability 与 by-action hidden；17 维 candidate row 由此组成。
- `joint_gppo.py:103-120`：`candidate_actor(candidate_features)` 直接加到 base/preference logits，再使用公开 mask 得到 action distribution。
- `run_replan_value_experiment.py:205-215`：probe 执行上述完整路径并记录 probabilities/original action/by_action；`:218-222` 选择 action 对应 hidden。
- corrected runner `run_ack_known_task_guard_corrected_20260922.py:2221-2222`：selected world hidden 只推进下一决策状态。

## 可登记协议

可登记两个新执行臂：`normal` 使用 `predict_all_candidates(use_events=True)`；`event-features-off` 使用已有 API 的 `use_events=False`，只将五个 event-probability feature 列置零。两臂固定同一 guard、checkpoint、mask、通信/ACK/lease/fencing、奖励、偏好、gamma、终止、父场景和外生 repeat。该干预只能解释为 event-feature 敏感性，不能解释为完整世界模型净收益，因为其余 policy_feature/vector_reward/task_consequence 没有训练定义的中性值。

正式执行提案为 48 个新分支（8 parent × 3 repeat × 2 arm），最坏新增环境步 **768**（每分支16步）；optimizer/world/offline update 必须为0。H-005 的轨迹不自动复用为新臂对照，需另行通过 exact interface/state/randomness gate。permutation 不列入正式臂。

## 资源与停止

本轮 `env.step=0`、reset/replay=0、model forward=0、optimizer/world/offline update=0、新增 attempt=0。正式账本只读状态为 H-005 总上限404、reserved=verified=299、unknown=0、pending=0，剩余105；`PRAGMA integrity_check=ok`，当前 SQLite SHA-256 为 `9ff13f0e18ab8ce56f418f00eac786f08a47e6107a2b8d31f7184b2502c9d420`，与 H-005 最终权威摘要一致。本轮不使用剩余额度、不建替代账本。身份不符、非法动作、mask/guard mismatch、状态污染、标签缺失、预算 unknown/pending 或任一更新计数非零时立即停止。

## 证据边界

H-005 的 full probability ledger、public observation digest、policy/world hidden digest 和 selected-action hidden digest 已保留；但没有保存 candidate feature tensor、candidate actor component logits 或未选动作的 outcome。哈希证明身份，不等于可重放 hidden 或 world 归因。

后续方案仍待批准：同一 SQLite 总上限由404提至1067（保留299已用，新增阶段最多768，扩额663），本轮不改库、不使用现有105步。动态执行前仍须完成实施接口检查；A 不证明实际排名改变或收益。

## 集中审查补充

最终单一判定 A，仅限事件特征关闭敏感性。置零依据是既有接口约定，不是无事件的校准预测，也未证明处于该 checkpoint 的训练分布。P_train 源码使用 use_events=True，须登记分布外风险。其余12维、同输入下原始 world输出、by_action及policy hidden保持相同；后续动作分叉可导致后续hidden正常分叉。next_state不直接进入17维actor特征，辅助训练贡献未被隔离。

本次新增严格输入检查覆盖279个唯一决策、24条登记分支、外生键、mask、候选减排除、公开任务映射、概率及argmax与NOOP保留。保存metadata的一致性不等同于重新证明未保存的公开观测内容。测试不导入模型或环境。

完整两臂清单和顺序见 paired-manifest-proposal.json；主效用为从恢复后首步k=0开始的sum(0.99^k*(0.4*r_task+0.2*r_energy))，原生终止，无终端价值补项。8父场景等权，父级10000次bootstrap，seed20260923，线性分位点0.025/0.975。所有场景已观察，属于开发性敏感性评价。

模型调用提案：每轨迹决策1次policy encode、1次world25候选batch、1次actor readout，各最多768；首对相同输入门复用已有编码/world输出，最多额外2次actor readout。保存完整17维特征、分量logits、概率和hidden摘要；同输入差异与分叉后轨迹比较分开。详细门控、费用计数、停止规则以protocol.json为准，全部尚未执行。
