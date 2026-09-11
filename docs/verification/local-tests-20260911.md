# 本地代码验证记录（2026-09-11）

本记录只证明代码/合同单元测试，不证明新训练或新实验结果。

## 环境

- OS：Windows（本机）
- Python：3.14.4
- Torch：2.13.0+cpu
- NumPy：2.5.0
- 设备：CPU；未启动训练

## 命令与结果

```text
python -m pytest tests/test_consequence_model.py -q
4 passed in 2.17s

python -m pytest tests/test_consequence_model.py tests/test_world_model.py tests/test_contracts.py -q
22 passed in 2.22s

python -m pytest --basetemp .pytest-local-20260911-full -q
155 passed in 7.19s
```

第一次全套测试使用系统临时目录时有 7 个 `tmp_path` setup 权限错误；改用新仓库内独立临时目录后 155 项全部通过。该权限差异不改变代码结果。测试覆盖候选动作差异、连续后果 NLL 与 deadline BCE 分离、标签来源/窗口验证、历史世界模型和原有合同回归。

## 未验证事项

未执行服务器训练、完整新协议 episode、checkpoint 恢复、GPU 延迟、真实通信流量或生产安全验证；这些只能在可用服务器和冻结数据/环境上完成。
