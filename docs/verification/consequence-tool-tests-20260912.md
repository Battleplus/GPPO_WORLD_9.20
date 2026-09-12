# 候选后果工具验证（2026-09-12）

本次只验证代码，不启动训练、不加载训练数据、不连接服务器。

```text
python tools/train_m10_consequence_model.py --help
命令帮助正常输出

python -m pytest --basetemp .pytest-local-20260912-consequence-2 -q \
  tests/test_consequence_model.py tests/test_consequence_data.py \
  tests/test_world_model.py tests/test_contracts.py
27 passed in 2.08s

python -m pytest --basetemp .pytest-local-20260912-full -q
160 passed in 6.88s
```

覆盖：候选动作差异、预测不确定性、连续 NLL 与 deadline BCE 分离、反事实标签来源/窗口、隐藏状态/未来字段拒绝、非法动作拒绝，以及旧 world model/合同回归。

训练入口的实际服务器运行、optimizer/recovery checkpoint 恢复、逐事件数据质量和新协议结果仍待可用服务器完成。
