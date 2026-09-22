# H-005 执行授权申请（待批准）

目标：H-20260922-ACKGUARD-PAIRED-RESULT-005。只进行冻结规则与历史 R 的配对评价，不训练、调参或扩展样本。

待批准动作：将原 ack-known-task-guard-baseline-v1/budget.sqlite3 的环境步总上限从 384 调整为 404。历史 reserved=verified=20 全部保留；新增固定 run 上限 384。禁止替代账本、覆盖原消耗或自动重试。

执行固定 8 个 parent × 3 个外生 repeat，共 24 条新规则分支，复用对应的24条历史 R/299步，不重跑对照。旧两条 guard 分支不纳入新处理组。单条最多16步，合计新增最多384步。冻结 GPPO/world、原25-action环境、guard、偏好(0.8,0.2)、奖励尺度(0.5,1.0)、gamma=0.99及源随机键。

批准后：保存原预算一致性备份和摘要，在同一SQLite事务中仅调整授权上限并记录迁移，复核原20步、无unknown/pending及新run/attempt为空；生成绑定批准记录和最新预算哈希的authorization，运行既定矩阵。不会逐分支请求确认。出现身份/动作/污染/门控/标签/预算异常立即技术停止，保留全部消耗，不自动重跑。

主报告按parent内3次repeat平均，再8个parent等权汇总guard-R效用；父场景bootstrap 10000次、seed=20260922、95%区间。报告命令接受/拒绝/task_unavailable、任务与主机确认、能耗、guard触发/动作变化、合法NOOP保留和资源调用。未完成矩阵不能形成研究结果。

结果形成后选择性归档到 GPPO_WORLD_9.20 / archive/research-progress，远端回读，作出是否固定简单基线的明确决策。之后停止。

审批回复建议：同意同一SQLite扩额至404，并按H-005固定矩阵执行。

本文件为待审批材料，不构成批准。当前正式SQLite未修改，没有新增环境步、模型前向或attempt。
