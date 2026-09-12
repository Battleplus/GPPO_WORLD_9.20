# WORLD-GPPO_9.11 服务器运行边界与复现入口（draft v1）

本地已完成代码和测试准备；本机不训练。正式训练必须在用户已有授权服务器上进行。服务器不可用时，只保留本运行包和阻塞说明，不购买云资源、不自行选择付费服务。

## 运行前

1. 记录主机、CPU/GPU、内存、OS、Python、Torch、CUDA、NumPy、pytest 及依赖锁/实际版本。
2. 核对本仓库提交、`docs/provenance/artifact-index.json`、协议版本、tape、模型兼容性及每个输入 SHA-256。
3. 检查是否已有同一 run-id 存活进程或完整输出；确认同一 run 没有存活进程后才可 resume。
4. 先跑数据/合同/模型加载 gate，再跑小预算 sanity；任何泄漏、非法动作、非有限损失、安全违规或预算超限立即停止。

## 计划入口

```powershell
python tools/train_m10_consequence_model.py `
  --protocol configs/world-gppo-9.11-consequence-v0.1.0.json `
  --data <episode-disjoint-data> `
  --out <unique-run-directory> `
  --device cuda `
  --run-id <unique-run-id>
```

The checked-in configuration declares `m10-graph5-5type-25action`.  The entry
uses the explicit `Graph5Snapshot`/Graph-5 consequence model for that protocol;
the older `gppo-graph-3type-17action` path remains a separate compatibility
mode and requires a compatible base checkpoint.  Do not change the protocol
string to force a run.  A resumed run must reuse the same inputs and run id and
add `--resume`; the entry refuses a live or identity-mismatched run.

该入口在本阶段尚未启动；正式实现必须保存逐事件 rollout ledger、逐分支标签 provenance、optimizer/recovery state、runtime、run-status、停止原因及输入哈希。不得把规划文件当作已有训练结果。

## 结果归档

完成后先校验输出和模型哈希，再生成逐 seed、逐 episode、预测质量、任务效果、通信代理量、安全审计和完整决策延迟报告；新硬件延迟单列。大型 checkpoint、optimizer/recovery state、训练日志和数据只上传新仓库的独立 Release，不写入 Git。
