# WORLD-GPPO_9.11 交付清单

## 本提交

- 会议要求与新任务合同差距：`docs/contracts/m10-arrival-protocol-gap-analysis-20260913.md`、`configs/world-gppo-9.11-arrival-v0.1.0.json`
- 到达语义实现：`gppo_world/task_lifecycle.py`、`gppo_world/service_clock.py`、`gppo_world/m10_environment.py`
- 新到达反事实生成器：`tools/generate_m10_arrival_consequence_dataset.py`
- 新协议规则基线：`tools/run_m10_arrival_rule_baseline.py`
- 新阶段计划与 tracker：`docs/plans/m10-arrival-world-model-plan-20260913.md`、`refine-logs/EXPERIMENT_PLAN-20260913.md`、`refine-logs/EXPERIMENT_TRACKER-20260913.md`
- 会议纪要草稿：`docs/meeting/无人机调度模型优化-会议纪要草稿-20260913.md`

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
- 新到达协议：功能测试与规则基线已完成；新后果模型训练未启动。
- 基础对照训练：已完成 49152 环境步；融合矩阵未启动。
- 旧服务器：未连接，历史中断状态不变。
- 历史负结果：保留，不改写。
