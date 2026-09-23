# H-20260923 世界模型增量评价授权申请

本文件只申请后续动态执行，不代表已获批。本轮预检已停止，未调用环境或模型。

申请批准以下固定两臂：

1. `normal`：guard 开启，冻结 world `predict_all_candidates(use_events=True)`。
2. `event-features-off`：同一冻结 world/policy/guard，调用已有 `use_events=False`，仅将 event-probability 五列置零。该臂是 event-feature 敏感性，不是完整 no-world-model 对照。

两臂均使用 H-005 的 8 个 W1 development parent × 3 个外生 repeat；新执行 48 个分支，最坏 768 个环境步。两臂不自动复用 H-005 轨迹。保持 checkpoint、源码、mask、动作映射、通信/ACK/lease/fencing、奖励、偏好、gamma、终止和外生键完全一致。

主指标为 normal 减 event-features-off 的原登记偏好效用；每 parent 先平均 repeat，再 8 parent 等权；父场景级 bootstrap 10,000 次，seed 20260923。次指标包括动作排序/选择、接受/拒绝/task_unavailable、物理/主机完成、能耗、NOOP、guard 变化和调用计数。

批准前置条件：

- 仅提出同一 SQLite 总上限404→1067（299已用+768新阶段；增加663）的待批准方案，不新建替代账本；本轮不修改 SQLite（当前 SHA-256 `9ff13f0e18ab8ce56f418f00eac786f08a47e6107a2b8d31f7184b2502c9d420`）。
- 确认 event-only 敏感性解释及其不能代表完整 world-model 净收益的限制。
- 首分支检查 exact identity、mask、guard、snapshot/hidden 状态隔离和世界模型输出齐全；失败立即停止，不重试、不退款。
- optimizer/world/offline update 必须为 0；unknown/未能最终核销的pending、非法动作、状态污染、标签缺失或重复/部分分支均立即停止。

不申请 permutation 臂、不申请训练、不申请 validation/heldout、不申请当前剩余额度使用。
