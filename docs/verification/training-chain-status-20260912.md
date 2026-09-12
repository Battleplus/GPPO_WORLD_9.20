# 9.11 后果模型训练链路状态（2026-09-12）

## 已落地

- `gppo_world/consequence_data.py` 使用统一的 `files.<split>` 对象：必须包含 `path`、`sha256`、`records` 和 `identity_sha256`。审计实际 JSONL 内容、记录数、SHA-256、重复身份以及 `(parent_episode_id, prefix_id)` 的跨 split 泄漏。
- 记录身份由父 episode、前缀、episode、决策序号、候选动作和共享外生随机 key 组成；严格训练加载要求 `parent_episode_id` 与 `prefix_id`。
- `tools/train_m10_consequence_model.py` 已具备显式 `--resume`、运行身份检查、逐 optimizer step 恢复点、独立 `best-inference.pt` / `last-recovery.pt`、Python/NumPy/Torch RNG、数据顺序、早停状态、实际更新数、墙钟预算和非有限检查。
- 图、历史、目标和模型在推理/训练前统一到请求设备；请求 CUDA 但 CUDA 不可用时直接失败。

## 当前硬闸门

新阶段协议明确声明目标为 `m10-graph5-5type-25action`，而 M-10 公共观测包含 UAV、Region、Target、Task、Event 五类节点和 24 个 UAV–Task 候选加 NOOP。旧 `GraphWorldModel/GraphSnapshot` 不能直接复用；入口把 Graph-5 模型作为独立架构，旧 checkpoint 只保留在 3 类型/17 动作兼容模式，不能通过重命名字段互换。

因此，当前尚未声称“世界模型已训练”或“预测有效”。显式 Graph-5/25-action adapter、版本化 JSONL 生成器和共享外生随机分支账本已落地，但尚未在服务器 pilot 上执行；本地冒烟数据仅用于链路审计，不是目标实验数据。旧的 3 类型/17 动作测试仅证明兼容模式的代码回归。

## 尚未执行

- 本机未训练；未连接原训练服务器；未购买或申请新算力。
- 没有已生成并冻结的正式后果数据；生成器会按完整 tape 采样并记录共享外生随机键、分支账本和删失标签。
- 因 Graph-5 adapter、数据和服务器均未齐备，未启动 pilot，也未启动策略矩阵。

## 服务器解除阻塞后的顺序

1. 在服务器确认 Python、Torch、CUDA 和依赖版本，检查数据真实内容与 split 审计。
2. 先运行 Graph-5 adapter 的小规模共享随机反事实 pilot，保存逐分支轨迹和标签来源。
3. 训练预测模型，核验 best/last 加载和续跑、每个预测头误差/校准、候选排序、失败与不确定性。
4. 只有预测 pilot 有价值时，按 1101/2203/3307 和 8192 actor decisions/seed 运行四组策略对照；否则封存负结果并停止融合训练。
