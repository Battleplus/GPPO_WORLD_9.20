# 冻结后命令与授权边界

工作目录：`E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt`。Python `C:\Python314\python.exe -X utf8`。

本轮只允许静态/纯和stub测试，不执行下列动态命令。测试路径使用workspace临时目录；临时测试SQLite不是实验预算。

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
C:\Python314\python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp 'E:\Z博士\research-reviews\model-rule-stage1-v1\future-test-fresh' tests/test_public_dispatch_rule_v1.py tests/test_model_rule_pair_v1.py tests/test_analyze_model_rule_pair_v1.py tests/test_authorize_model_rule_pair_v1.py
```

用户批准后，另建批准记录副本，填写本包hashes.json的精确SHA、实际用户消息引用和时间；不改变冻结pending模板，不把自动继续视为批准。

下列命令按顺序各运行一次，每条非零立即停止，禁止自动重试、改名复跑、退款或新建替代库：

```powershell
$stage='runs/finite-communication-ack-lease-fix-20260920'
C:\Python314\python.exe -X utf8 tools/authorize_model_rule_pair_v1.py --apply --approval "$stage/model-rule-stage1-approval-v1/user-approval.json" --authorization-out "$stage/model-rule-stage1-approval-v1/authorization.json" --receipt-dir "$stage/model-rule-stage1-approval-v1/migration"
C:\Python314\python.exe -X utf8 tools/run_model_rule_pair_v1.py --execute --authorization "$stage/model-rule-stage1-approval-v1/authorization.json" --manifest "$stage/model-rule-stage1-execution-package-v1/paired-manifest.json" --out "$stage/model-rule-stage1-authorized-run-v1"
C:\Python314\python.exe -X utf8 tools/analyze_model_rule_pair_v1.py --manifest "$stage/model-rule-stage1-execution-package-v1/paired-manifest.json" --run "$stage/model-rule-stage1-authorized-run-v1" --sqlite "$stage/ack-known-task-guard-baseline-v1/budget.sqlite3" --run-id model-rule-stage1-v1 --attempt-id model-rule-stage1-v1-attempt-0001 --out "$stage/model-rule-stage1-authorized-run-v1/analysis.json"
```

完整矩阵才输出主结果；技术停止交付已完成样本、真实消耗及异常，不运行补样。源码/模型/协议身份不一致停止，不能替换成另一版本。

CPU单进程，模型intra4/inter1，模型和规则共享同一固定环境合同。模型加载只为model臂；规则不读取hidden或world预测。快照pickle加载器整体反序列化再筛选prefix的边界限制保持披露，不使用heldout内容或标签。
