# 授权前与批准后的固定命令

工作目录：`E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt`。

授权前可运行下列纯/stub测试；其中临时SQLite仅用于单元测试，不是正式预算：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
C:\Python314\python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp 'E:\Z博士\research-reviews\world-event-feature-package-006\approval-test-fresh' tests/test_world_event_feature_pair_20260923.py tests/test_analyze_world_event_feature_pair_20260923.py tests/test_authorize_world_event_feature_pair_20260923.py
```

以下命令仅在用户明确批准本包范围后执行。先复制 `user-approval-PENDING.json` 到下列 approval 路径，在副本中填写真实用户批准记录、时间、批准包 hashes.json 的SHA-256；status改为approved，approved_by=user，accepted_event_sensitivity_only=true。不得把本说明或自动继续视为批准；不得修改已冻结的pending模板。

```powershell
$stage='runs/finite-communication-ack-lease-fix-20260920'
C:\Python314\python.exe -X utf8 tools/authorize_world_event_feature_pair_20260923.py --apply --approval "$stage/world-event-feature-approval-20260923/user-approval.json" --authorization-out "$stage/world-event-feature-approval-20260923/authorization.json" --receipt-dir "$stage/world-event-feature-approval-20260923/migration"
C:\Python314\python.exe -X utf8 tools/run_world_event_feature_pair_20260923.py --execute --authorization "$stage/world-event-feature-approval-20260923/authorization.json" --manifest "$stage/world-event-feature-execution-package-20260923/paired-manifest.json" --out "$stage/world-event-feature-authorized-run-20260923"
C:\Python314\python.exe -X utf8 tools/analyze_world_event_feature_pair_20260923.py --manifest "$stage/world-event-feature-execution-package-20260923/paired-manifest.json" --run "$stage/world-event-feature-authorized-run-20260923" --sqlite "$stage/ack-known-task-guard-baseline-v1/budget.sqlite3" --run-id world-event-feature-pair-20260923 --attempt-id world-event-feature-pair-20260923-attempt-0001 --out "$stage/world-event-feature-authorized-run-20260923/analysis.json"
```

按顺序执行，每条返回非零则停止，保留证据，不自动运行下一条或重试。迁移命令只扩原库404→1067，不启动实验；runner才登记新run并恢复模型/快照。任一动态失败都不重跑，不用新账本绕过。运行器故障证据仍可零步审查，但不完整矩阵不报告主效应。

整体pickle恢复的限制：历史加载器将整个快照包反序列化后筛选8个prefix。未使用validation/heldout内容选择或分析；这不是严格的逐对象隔离读取。当前准备阶段没有反序列化。

冻结后不要运行会重写包文件的 `--check-package`；只读检查可调用 `package_validation()`。正式授权器也用只读检查。
