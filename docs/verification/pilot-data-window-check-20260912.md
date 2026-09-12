# Pilot 数据窗口检查（2026-09-12）

本文件只记录服务器 pilot 前的本地开发数据检查，不是模型训练或策略评估结果。

## 配置与结果

| 配置 | train | validation | 结论 |
|---|---:|---:|---|
| `prefix_steps=2, horizon_steps=1` | 36 条；service_progress 全 0；deadline 有效标签 0 | 20 条；service_progress 全 0；deadline 有效标签 0 | 仅可作 CUDA/加载链路 smoke data |
| `prefix_steps=2, horizon_steps=6` | 36 条；service_progress 范围 0–2.1662；deadline 有效 24/36，正/负 19/17 | 20 条；service_progress 范围 0–2.2132；deadline 有效 16/20，正/负 8/12 | 可作为开发窗口候选，尚未冻结为正式数据 |

两次生成均为每 split 4 个父 episode；同一父前缀的候选分支不跨 split。horizon=6 数据的实际 manifest 审计通过：文件 SHA-256、记录数、记录身份和跨 split 前缀审计均无错误。

## 标签边界

- `service_progress` 是窗口内服务推进，不等于最终完成时间。
- 观察窗口未覆盖到达或 deadline 时，`travel_time`/`deadline_risk` 通过 `label_masks` 标为删失；不当作“永不到达”、成功或零风险。
- 生成器的分支账本保存共享外生 key、前缀和逐步公开执行轨迹；隐藏状态只用于离线标签生成。
- 候选数受当时公开合法 mask 约束，不能把同一前缀的多个候选计作多个 episode。

## 状态

`horizon_steps=6` 仅是开发窗口候选。正式训练前仍需在服务器固定数据采样上限、检查 Graph-5 兼容性、运行 CUDA pilot、建立简单基线并冻结 train/validation/test/OOD。当前没有预测模型误差、校准、候选排序或策略增益证据。
