# 交付索引

本包采用 experiment-plan 工作流，将五阶段研究目标落实为单一阶段一比较。

- EXPERIMENT_PLAN.md：目标、证据、矩阵、统计及停止条件。
- EXPERIMENT_TRACKER.md：五阶段状态，整体目标保持 active。
- protocol.json / paired-manifest.json：冻结比较与 48 条分支身份。
- COMMANDS.md：测试、批准后预算变更、执行与分析命令。
- user-approval-PENDING.json：仅为待批准模板，不能据此执行。
- ACCEPTANCE.md：集中验收结果；report.md：最终执行包报告。
- native-source-manifest.json / source-manifest.json：原生环境和实际执行源码身份。
- schema.json：selector、首对 gate、runtime/status 和 analyzer 输入合同。
- runner-source.py / runtime-source.py / analyzer-source.py / authorizer-source.py / public-rule-source.py / guard-source.py：冻结入口源码副本。
- budget-readonly.json：正式数据库只读证据；测试临时库不代表正式预算。
- hashes.json：本包、入口源码副本、关键本地源码和固定模型输入的 SHA-256 清单（不包含自身）。

源码入口：tools/run_model_rule_pair_v1.py、tools/model_rule_branch_runtime_v1.py、
tools/analyze_model_rule_pair_v1.py、tools/authorize_model_rule_pair_v1.py、
gppo_world/public_dispatch_rule_v1.py；对应 tests/test_*_v1.py。

本包不提供任何新的实验结果；旧观察过的前缀不称为独立泛化数据。
