# Experiment Tracker

本阶段只完成迁移、设计、代码和测试；没有启动训练。

| Run ID | Milestone | Purpose | System | Split | Priority | Status |
|---|---|---|---|---|---|---|
| R911-000 | M0 | 合同/标签/模型接口 sanity | consequence schema + unit tests | fixed toy | MUST | READY |
| R911-001 | M1 | 单 seed 计时与基线闭环 | traditional legal scheduler | frozen pilot | MUST | BLOCKED: server not supplied |
| R911-002 | M1 | GPPO baseline | GPPO | frozen pilot | MUST | BLOCKED: server not supplied |
| R911-003 | M1 | 历史信息基线 | GPPO + history | frozen pilot | MUST | BLOCKED: server not supplied |
| R911-004 | M1 | 后果预测基线 | GPPO + history + consequence | frozen pilot | MUST | BLOCKED: server not supplied |
| R911-005 | M2 | 四组 3-seed 正式矩阵 | all four systems | train/val/test/OOD | MUST | TODO after M1 gate |
| R911-006 | M3 | 预测质量/消融/总成本 | head and cost ablations | test/OOD | MUST | TODO after R911-005 |

服务器运行前不得把 `BLOCKED` 改成 `RUNNING`；同一 run-id 只有在无存活进程且已有输出完整性通过后才允许 resume。
