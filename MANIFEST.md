# WORLD-GPPO_9.11 交付清单

## 本提交

- 来源迁移与版本边界：`docs/transition/legacy-provenance.md`
- 任务合同差异：`docs/contracts/world-gppo-9.11-task-contract.md`
- 候选后果世界模型：`gppo_world/consequence_model.py`
- 数据/标签/损失设计：`docs/world-model/action-consequence-design.md`
- 实验计划与 tracker：`docs/plans/EXPERIMENT_PLAN.md`、`docs/plans/EXPERIMENT_TRACKER.md`
- 新协议配置：`configs/world-gppo-9.11-consequence-v0.1.0.json`
- 本地测试证据：`docs/verification/local-tests-20260911.md`
- 历史结果与制品索引：`docs/results/historical-evidence.md`、`docs/provenance/artifact-index.json`
- 大文件 Release staging：`release-staging/world-gppo-9.11-m10-legacy-v1/`

逐文件 SHA-256：`docs/provenance/file-sha256-20260911.json`。该清单不包含未提交的 pytest 临时目录或 Python 缓存；大 checkpoint、world model、optimizer/recovery state、训练日志和数据仍关联旧项目已核验归档，待新远端可达后以独立 Release 上传。

## 当前状态

- 新仓库远端：连接重置，尚未 push 或创建 Release。
- 本地新仓库分支：`world-model-consequence-v1`。
- 新训练：未启动。
- 本机训练：未启动。
- 旧服务器：未连接，历史中断状态不变。
- 历史负结果：保留，不改写。
