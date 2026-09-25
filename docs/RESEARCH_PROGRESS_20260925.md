# 研究进度同步：2026-09-25

本次仅同步已有结果、解释边界与基线准备范围。环境步、模型前向、训练更新及新增实验attempt均为0。没有启动主线程实验。

## 当前判断

GPPO＋世界模型＋偏好条件策略保留为主研究候选。阶段2已经证明：在历史原生M10/W1、当前单一模型seed下，完整系统优于一个明确实现的公开历史有限搜索规则。尚未证明优于标准PPO、滚动匹配等代表性传统方法，也未证明world训练或偏好机制的独立贡献。

“在线移除world”的后续诊断是组件必要性检查，不是必须删除world的决定。它保留原训练权重，不能视为无world训练基线。下一步优先补齐代表性基线与公平协议，正式运行另需授权。

## 已完成结果

| 工作 | 结果 | 证据范围 |
|---|---|---|
| 阶段2完整M/R矩阵 | 9600/9600 episodes，1600独立parent×3 repeats×2 arms；效用差+0.066018427，95% CI [0.062341284, 0.069742127]，超过δ=0.01；该阶段任务和成本门通过 | 原生M10 25-action/W1、seed1101，不是有限token通信环境 |
| 阶段2任务与能耗 | 物理按时86.9236% vs 77.4722%；主机按时69.7326% vs 60.4410%；能耗6.443457 vs 5.453014 | 增加能耗换取更高任务收益，不声称同时节能 |
| 后续A成本回放 | N平均CPU 10.386905ms > 10ms，失败保留，未重试A | 10ms是研发标准，非已证明现场SLA；不同计时范围不能直接互换 |
| 独立任务收益诊断 | 固定800个旧开发parent各1 repeat，新增800个N；N−M效用+0.000410361，CI [-0.000031782,0.000995263]，非劣及两项任务护栏通过；N−R效用+0.063441822，CI [0.056856861,0.070185049] | 支持任务保留，不是world训练无价值或部署通过 |
| 诊断成本 | N正式段8.253876ms/决策；相对匹配M成本比约0.880，未达0.8；原A失败仍有效 | 任务通过不能包装为成本通过，简化实用验收未通过 |
| 基线范围冻结 | 最近距离贪心、最早截止贪心、滚动匈牙利、已有搜索规则、标准PPO、无world训练GPPO、完整系统 | 新基线仅登记规范，未完成实现或新训练比较 |

## 资源与恢复历史

阶段2 A/B/C累计环境步133662、reset9605、encode/world/actor各64051、规则决策70661、模型加载6；训练更新0。矩阵统计与技术失败/恢复开销分列，历史监控共享冲突和SQLite I/O失败保留，不以新账本清零。

后续A新增模型前向包含在累计记录；任务诊断新增环境步10540、encode/actor各10540、reset802、加载2，world/规则/更新0。阶段2＋失败A＋任务诊断的**这一条研究链累计**环境步144202、encode/actor各76166、world64576、规则70661、reset10407、加载11。不是项目全部历史实验的全生命周期总额。

细节、失败及恢复开销、实测值与保守预留区别，以随包原报告和verification为准。本次没有重新运行实验或改写SQLite。

## 仍需解决

1. 与标准PPO、贪心、滚动匹配的同合同公平比较尚缺；已有搜索规则不能代表所有传统算法。
2. 无world从头训练与完整训练谱系的成本对齐尚缺；在线移除不能替代该比较。
3. 主结果只有一个训练seed、一个生成器与W1条件；偏好适应、事件成本收益、跨任务家族及有限通信泛化未通过。
4. 原生任务是到达即完成；priority不直接进入向量任务奖励，主机确认独立作护栏。不得把现有仿真说成完整救援/巡检服务或经济利润验证。

## 原始证据与后续入口

- [阶段2报告](archive/RESEARCH-SYNC-20260925/stage2/report.md)、[完整分析](archive/RESEARCH-SYNC-20260925/stage2/analysis.json)、[核验](archive/RESEARCH-SYNC-20260925/stage2/verification.json)。
- [成本可比性审计](archive/RESEARCH-SYNC-20260925/cost-audit/report.md)。该报告形成于任务诊断之前，其“尚未检验”是当时状态。
- [任务收益诊断](archive/RESEARCH-SYNC-20260925/task-diagnostic/report.md)、[核验与累计成本](archive/RESEARCH-SYNC-20260925/task-diagnostic/verification.json)。
- [基线范围冻结](BASELINE_COMPARISON_FREEZE_20260925.md)、[下一步](NEXT_EXECUTION.md)。
- [原文件来源与哈希](archive/RESEARCH-SYNC-20260925/source-index.json)、[同步文件哈希](archive/RESEARCH-SYNC-20260925/hashes.json)。
- 更早阶段参见[原进度归档分支](https://github.com/Battleplus/GPPO_WORLD_9.20/tree/archive/research-progress)、[R/K封存](https://github.com/Battleplus/GPPO_WORLD_9.20/tree/archive/preference-rk-route-closure-20260922)、[阶段2历史技术停止](https://github.com/Battleplus/GPPO_WORLD_9.20/tree/archive/stage2-recovery-technical-stop-20260925)。它们保留历史状态，不能取代本页最新判断。

本包同步文档和关键机器可读结果；不包含完整源码、checkpoint、大体量逐步日志或SQLite，不是可独立重跑的完整部署包。原报告按字节保留，其中本地路径和“未发布”指其形成时状态。后续引用以相应归档路径及哈希为准。

## 可复用资产补充归档

[源码、基线、测试、配置及提交权重入口](REUSABLE_ASSETS_20260925.md)。新基线仍按已实现/待实现区分；完整原始日志和SQLite另列未上传清单，本次归档不启动实验。
