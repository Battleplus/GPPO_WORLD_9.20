# M-10 到达协议结局核对与后果模型结果

状态：本地开发/正式数据冻结与单 seed 有界训练已完成；策略融合未启动。

源码 HEAD：`f34743f3ab376f5cddd977861c83eb6cf10d5f64`（工作分支 `world-model-consequence-v1`）。本轮新增修改尚未用旧协议结果替代任何历史资产。

## 1. 任务结局核对

核对输入是已有 `m10-arrival-rule-baseline-20260913-v1/rule-baseline.json`，脚本只读取账本，没有重跑旧诊断。16 个 episode、每个 6 个任务，共 96 个任务；物理到达和主机确认分别作为独立口径。

| 口径 | 任务/事件分母 | 物理到达事件 | 完成消息发送 | 主机确认事件 | deadline 前完成 | 原汇总过期/未完成 |
|---|---:|---:|---:|---:|---:|---:|
| physical_arrival | 96 任务；48 条到达记录 | 48 | 48 | 10 | 48 | 48 |
| host_confirmation | 96 任务；48 条到达记录 | 48 | 48 | 10 | 9 | 76；另有 11 个状态未在旧账本明确记录 |

10 次确认与 9 个完成任务并不矛盾：9 次在 deadline 前被主机合法确认，1 次确认发生在 deadline 之后（`test-mixed-seed-2093015/task-4`，确认 16.0，deadline 14.9283962219），因此不算按时完成。旧 JSON 仅序列化了到达任务的 `completion_records`；其余 48 个任务的身份和 deadline 已由冻结 tape 重建，但逐任务终态缺失，核对表将其标为 `not_recorded`，不擅自分配为过期或等待确认。原汇总的 48/48、9/76/11 保留为 aggregate-only 证据。

本轮研发主口径采用 `physical_arrival`：物理进入目标区域即完成；`host_confirmation` 仅作通信敏感性口径。最终会议验收口径仍待确认。消息发送、合法收到、确认、物理到达和 deadline 分类在账本中分别保留，不能把等待确认当成物理失败或完成。

逐任务文件：`E:\Z博士\9.2日\WORLD-GPPO_9.11-local-runs\m10-arrival-outcome-audit-20260913-v3\task-outcomes.jsonl`。

## 2. 正式数据冻结

协议：`world-gppo-9.11-arrival/0.1.0`；Graph-5/5-type/25-action/global-27；预测窗口 6 steps；同一父 episode 的 4 个前缀和候选均留在同一 split；候选共享同一外生随机 key；隐藏状态只用于仿真标签。`OOD` 使用 `energy_insufficient` 场景。NOOP、未到达、删失、执行/能量失败保留，窗口内未观察到不被写成永不完成。

冻结目录：`E:\Z博士\9.2日\m10-arrival-consequence-formal-20260913-v2`。

| split | 父 episode | 前缀 | 候选记录 | 到达时间有效 | deadline 有效 | failure 有效/正例 | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 64 | 256 | 1521 | 924 | 1156 | 1187 / 31 | `3b22ed91f1dde18b1a0031c119208378e900a23c96706b1222e593100afb6d60` |
| validation | 32 | 128 | 815 | 494 | 619 | 652 / 33 | `da1547175e25216a436e6363983335ede145c8c68ded5e1b4c5abbd16e3655dd` |
| test | 64 | 256 | 1562 | 906 | 1182 | 1235 / 53 | `066287060f67adc8aa96b2733ad2a2caa5883e8f276b6d4c76ab37b9b82884bc` |
| OOD | 32 | 128 | 812 | 482 | 606 | 642 / 36 | `fdf57466c12be69c5d2e7e84956baea12e7f4d42880ffa2c30b2e83148065a85` |

manifest SHA-256：`00fa37338185581803ba9c81c26534ab2e830186ef69da4873c5575e175c8e47`。实际内容审计通过，无跨 split 父 episode/prefix。生成成本为本地 simulator 分支展开，不是线上网络流量。
完整本地关联及所有关键 SHA-256 见 `docs/results/m10-arrival-artifact-manifest-20260913.json`；test/OOD 评估 JSON SHA-256 为 `3d22e064cba201896c859019c8e710cd53cb33a1e2d7d6058ac02dc6e8499b70`。

## 3. 有界训练与恢复

训练入口：`tools/train_m10_arrival_consequence_model.py`；单 seed `1101`；CPU、4 threads、单进程、batch 16；最多 8 epochs、patience 3、4096 updates、3600 seconds。实际 7 个 epoch、670 次 optimizer update、70.344 秒，停止原因 `validation_patience`。训练目标是到达时间（有效标签才计算）、physical deadline BCE 和 execution/energy failure BCE；总系统能耗未作为本轮主训练目标。

best 推理 checkpoint 与 last recovery checkpoint 分开保存。恢复状态包含 model/optimizer、epoch、next index、数据顺序、实际更新数、RNG、早停状态和 elapsed time；另以 smoke run 验证了 1 update 中断后同 run-id 显式 `--resume` 的续跑链路。全流程未调用 backward 以外的训练入口，未更新冻结模型或旧优化器。

制品：

- best：`...\arrival-consequence-formal-train-20260913-v2\checkpoints\best-inference.pt`，SHA-256 `a5acd173b3c57301fd744cb1e6105789278de8f58438f0e42e7535a23e500b86`；
- last recovery：同目录 `checkpoints\last-recovery.pt`，SHA-256 `f7309f41d1c56a749a90fe89a96d11057034ac309e6d5545a089090895bf557`；
- training summary：SHA-256 `9cf6a702184ea6009df2b81447a63d8abde2c88147c37fc45013305d179e93a0`。

## 4. 预测质量与决策价值

模型由 validation total 选择；test/OOD 只在冻结后评估。结果如下，所有记录按父 episode/prefix 分组，候选分支不是独立 episode。

| split | 到达时间 MAE：模型 / 距离速度 | deadline Brier：模型 / 发生率 / 公开物理 | failure Brier：模型 / 发生率 / 公开号称能量 |
|---|---|---|---|
| test | 0.2973 / 0.3896（n=906） | 0.1644 / 0.1801 / 0.2335（n=1182） | 0.0389 / 0.0414 / 0.0429（n=1235） |
| OOD | 0.2908 / 0.5175（n=482） | 0.1627 / 0.1628 / 0.2013（n=606） | 0.0504 / 0.0538 / 0.0561（n=642） |

候选排序结果：test 的到达选择 regret 为模型 0.01395、距离规则 0.01477；OOD 为模型 0.03971、距离规则 0.04093。模型零 regret 率 test 96.95% 对 96.34%，OOD 90.59% 对 89.41%。但 deadline 选择 test 持平（模型/规则均 0.7829），OOD 模型 0.6860 低于公开规则 0.7907。由此只能说到达时间与 failure 头有局部预测质量，不能说候选调度价值稳定成立；不启动融合矩阵。

推理测量为本机 CPU、2 threads、无预热、一次离线评估；test 1562 候选 3.51 s，OOD 812 候选 1.68 s，分别约 2.25 ms/2.07 ms 每候选。它不代表 GPU 性能、实时保证或实际网络延迟。

## 5. 结论与边界

1. 新到达协议功能与数据链路已完成有限验证；物理到达和主机确认口径仍需会议最终确认。
2. 后果模型已完成一次有界训练和 best/last 恢复制品归档，训练链路状态为 completed-with-early-stop，而不是“预测有效”或“策略通过”。
3. test/OOD 的局部误差改善不足以证明稳定候选选择收益，尤其 OOD deadline 选择退化；本轮不执行 GPPO 融合。
4. 旧持续服务协议下的 GPPO/History 排名不迁移到新协议；旧 raw ledger、真实网络流量和原服务器状态仍是历史缺口。
5. “弱通信可用性通过”未成立。后续若继续，应在新协议下先重新建立合法规则、GPPO、GPPO-History 与融合组对照，并预先登记新预算；不得把本轮换任务定义带来的完成率变化称为算法收益。
