# H-006 阶段一执行包报告

状态：动态授权待定，未执行环境、快照恢复、模型前向、训练或正式 attempt。

本包将检验冻结 GPPO/world 在同一 guard、同一公开状态和同一外生键下，对 actor 事件候选特征的指定敏感性。normal 使用原生事件特征；event_features_off 使用现有 `use_events=False` 接口，仅将 actor-facing 候选特征第 12 至 16 列置零，保留 raw world 输出、by_action hidden 和 hidden 推进。该干预不能解释为无世界模型对照或世界模型净收益。

固定矩阵为 parent-00..07、W1、seed-1101、prefix-0、每 parent 三个 repeat、两臂共 48 分支，最多 768 个环境步。主指标为原登记偏好 `(0.8,0.2)` 下 normal 减 event_features_off 的配对效用；先在 parent 内平均 repeat，再做 8 parent bootstrap（10000 次，seed 20260923）。若证据不完整，只报告技术停止，不给算法负结果。

原正式 SQLite 当前 404 上限、299 verified/reserved、unknown/pending=0，剩余 105。本包提出批准后同库扩至 1067，增加 663，新阶段最多使用 768；本轮未修改 SQLite。授权工具仅在用户明确批准、包哈希和批准时间齐备时执行迁移，且只修改 environment_steps 的 limit_amount。

纯/stub测试最终 `30 passed`。运行器首对 gate 只读取每臂共同初始 step=1，并登记两次额外 actor readout；manifest 路径、transitive local dependencies、source/checkpoint/snapshot/config 和预算身份均需精确绑定。未进行 validation/heldout 或后续阶段。
