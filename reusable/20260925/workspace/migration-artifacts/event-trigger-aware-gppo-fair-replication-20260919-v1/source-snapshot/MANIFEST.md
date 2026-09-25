# WORLD-GPPO_9.11 交付清单

## 本阶段新增：联合任务集合后果设计（2026-09-14）

- 协议与标签定义：`docs/world-model/m10-joint-consequence-protocol-20260914.md`、`configs/world-gppo-9.11-joint-consequence-v0.1.0.json`
- 标签 schema：`schemas/m10-joint-consequence-v1.schema.json`
- 现有反事实生成器适配入口：`tools/generate_m10_joint_consequence_dataset.py`；开发 pilot v4 数据位于本地 runs 目录
- 新 schema 数据 loader、轻量模型和有界训练入口：`gppo_world/joint_consequence_data.py`、`gppo_world/joint_consequence_model.py`、`tools/train_m10_joint_consequence_model.py`
- 合法公开联合机会成本基线：`gppo_world/joint_consequence_baseline.py`
- 固定 fixture 测试：`tests/test_joint_consequence.py`（本阶段 6 passed）
- 实现说明与 pilot 登记：`docs/plans/m10-joint-consequence-implementation-20260914.md`
- 实验计划与 tracker：`refine-logs/EXPERIMENT_PLAN-m10-joint-consequence-20260914.md`、`refine-logs/EXPERIMENT_TRACKER-m10-joint-consequence-20260914.md`
- pilot 启动前登记：`docs/results/m10-joint-consequence-pilot-preregistration-20260914.md`
- 开发 pilot 结果：`docs/results/m10-joint-consequence-development-pilot-20260914.md`、对应 JSON

本阶段仅完成设计、schema、接口和测试；没有新数据、模型训练或策略融合运行。通知/调度工程候选、旧到达时间/系统能耗模型和负结果保持原版本。

## 本提交

- 会议要求与新任务合同差距：`docs/contracts/m10-arrival-protocol-gap-analysis-20260913.md`、`configs/world-gppo-9.11-arrival-v0.1.0.json`
- 到达语义实现：`gppo_world/task_lifecycle.py`、`gppo_world/service_clock.py`、`gppo_world/m10_environment.py`
- 新到达反事实生成器：`tools/generate_m10_arrival_consequence_dataset.py`
- 新协议规则基线：`tools/run_m10_arrival_rule_baseline.py`
- 新阶段计划与 tracker：`docs/plans/m10-arrival-world-model-plan-20260913.md`、`refine-logs/EXPERIMENT_PLAN-20260913.md`、`refine-logs/EXPERIMENT_TRACKER-20260913.md`
- 会议纪要草稿：`docs/meeting/无人机调度模型优化-会议纪要草稿-20260913.md`
- 新到达协议结局核对与后果模型报告：`docs/results/m10-arrival-protocol-outcome-and-model-20260913.md`、对应 JSON
- 新到达后果模型代码：`gppo_world/arrival_consequence_model.py`、`gppo_world/arrival_consequence_data.py`
- 正式数据生成、结局审计与 test/OOD 评估入口：`tools/generate_m10_arrival_consequence_dataset.py`、`tools/audit_m10_arrival_outcomes.py`、`tools/evaluate_m10_arrival_consequence.py`

- 来源迁移与版本边界：`docs/transition/legacy-provenance.md`
- 任务合同差异：`docs/contracts/world-gppo-9.11-task-contract.md`
- 候选后果世界模型：`gppo_world/consequence_model.py`
- 反事实标签加载与泄漏审计：`gppo_world/consequence_data.py`
- 服务器训练入口（本阶段未执行）：`tools/train_m10_consequence_model.py`
- 数据/标签/损失设计：`docs/world-model/action-consequence-design.md`
- 实验计划与 tracker：`docs/plans/EXPERIMENT_PLAN.md`、`docs/plans/EXPERIMENT_TRACKER.md`
- 新协议配置：`configs/world-gppo-9.11-consequence-v0.1.0.json`
- 本地测试证据：`docs/verification/local-tests-20260911.md`、`docs/verification/consequence-tool-tests-20260912.md`
- 历史结果与制品索引：`docs/results/historical-evidence.md`、`docs/provenance/artifact-index.json`
- 大文件 Release staging：`release-staging/world-gppo-9.11-m10-legacy-v1/`

逐文件 SHA-256：`docs/provenance/file-sha256-20260912.json`；`docs/provenance/file-sha256-20260911.json` 作为兼容索引保留，最新清单以 20260912 文件为准。该清单不包含 pytest 临时目录或 Python 缓存；大 checkpoint、world model、optimizer/recovery state、训练日志和数据仍关联旧项目已核验归档，待新远端可达后以独立 Release 上传。

## 当前状态

- 新仓库远端：`world-model-consequence-v1` 已 push；基础对照 Release `m10-baseline-stage-20260913-v1` 已创建并独立下载核验。
- 本地新仓库分支：`world-model-consequence-v1`。
- 新到达协议：16 episode/96 task 旧账本核对完成；物理到达为研发主口径，主机确认独立报告。
- 系统可用性闭环：`docs/results/m10-arrival-system-availability-closure-20260914.md`、对应 JSON 和制品清单；同一 16 条 tape 的 ideal/composite 规则配对 replay 已完成，旧 `not_recorded` 不变。
- 针对性改进：`docs/results/m10-arrival-targeted-improvements-20260914.md`、对应 JSON 和 `tools/run_m10_targeted_improvements.py`；通知/调度/合并三变体共96个新 episode 已完成，原规则结果复用旧 ledger，三种新方案均未采纳为默认。
- 通知候选选型与阶段结论：`docs/results/m10-arrival-notification-candidate-selection-20260914.md`、对应 JSON、`docs/meeting/m10-arrival-phase-progress-20260914.md` 和制品清单；明确 9→21、48→32、48→31 均来自同一开发回归父 tape，独立验证尚未完成；当前仅保留原调度器 + 有界通知为仿真工程候选。
- 通知方案独立验证：`docs/results/m10-notification-independent-validation-20260914.md`、对应 JSON、`docs/meeting/m10-arrival-overall-status-20260914.md` 和制品清单；使用新父 tape `2193001–2193016` 完成原系统/通知两变体的64 episode配对验证。结论为原系统继续默认，通知保留为未定效果—成本权衡候选；不再扩展通知/调度机制。
- 新后果模型：正式数据 v2 已冻结；单 seed CPU 有界训练 670 updates 后因 validation patience 停止；test/OOD 局部预测改善但候选调度价值未建立，融合未启动。
- 基础对照训练：已完成 49152 环境步；融合矩阵未启动。
- 旧服务器：未连接，历史中断状态不变。
- 历史负结果：保留，不改写。
- 联合任务集合后果方向：开发 pilot 已完成；结果为有限开发信号，尚无独立泛化或策略增益结论。数据/模型制品已发布 Release `m10-joint-consequence-development-pilot-20260914-v1` 并完成独立下载校验。

## 当前新增：到达协议 GPPO 候选后果融合接口（2026-09-14）

- 可执行方案：`docs/plans/m10-arrival-gppo-fusion-executable-20260914.md`
- 接口合同：`docs/contracts/m10-arrival-gppo-fusion-interface-v1-20260914.md`
- 候选级冻结模型加载、逐动作特征、GPPO prior 与合法动作选择：`gppo_world/arrival_gppo_fusion.py`
- 定向测试：`tests/test_arrival_gppo_fusion.py`；与到达协议回归合计 12 passed。
- 研究边界：旧持续服务 policy、旧联合任务集合 checkpoint 和动作前平均 context 均不作为新 arrival 融合输入；本轮未训练、未生成数据、未启动 GPPO 融合。
- 计划与追踪：`refine-logs/EXPERIMENT_PLAN-m10-arrival-gppo-fusion-20260914.md`、`refine-logs/EXPERIMENT_TRACKER-m10-arrival-gppo-fusion-20260914.md`，固定副本为 `refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md`。

## 当前新增：到达协议融合单 seed pilot（2026-09-14）

- 真实 PPO runner：`tools/run_m10_arrival_gppo_fusion_pilot.py`
- 保存/加载核验：`tools/verify_m10_arrival_gppo_fusion_pilot.py`
- Pilot 报告：`docs/results/m10-arrival-gppo-fusion-pilot-20260914.md`、对应 JSON
- run-id：`m10-arrival-gppo-fusion-pilot-20260914-seed1101`；本地输出为 `E:\Z博士\9.2日\WORLD-GPPO_9.11-local-runs\m10-arrival-gppo-fusion-pilot-20260914-v2`
- 到达模型：外部本地归档路径由 `run-identity.json` 登记，SHA-256 `a5acd173b3c57301fd744cb1e6105789278de8f58438f0e42e7535a23e500b86`；本轮冻结、未更新。
- 实际训练：GPPO、GPPO-History、GPPO-History-CandidateArrival 均 512 environment steps、8 次 optimizer updates；融合训练期间 512 次 world calls；规则组不训练。
- 结果边界：规则 53/96、GPPO 41/96、History 42/96、融合 44/96 按时物理到达；单 seed/16 validation tape，只是探索性 pilot，不构成稳定增益或弱通信可用性通过。
- checkpoint 首决策加载复核通过；大型 pilot 输出保留在本地，未在本清单中伪称已上传 Release。
- Release：`m10-arrival-gppo-fusion-pilot-20260914-v1` 已创建，资产 3,645,759 bytes，SHA-256 `728884d396f88d12168a3a337699be11ab06af54dc33f02065872672c312bd62`；GitHub API 摘要一致。独立下载因超时未通过，未将其记为独立下载校验成功。
