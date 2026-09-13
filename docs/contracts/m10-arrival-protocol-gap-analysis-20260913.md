# 会议要求与实现差距表：到达区域即完成

| 会议要求 | 原实现（旧协议） | 新实现/当前状态 | 证据或边界 |
|---|---|---|---|
| 到达指定坐标区域即完成 | 到达后仍需持续服务 | 已增加 `arrival_to_region`；默认半径 0.0，保留旧默认 | `gppo_world/task_lifecycle.py`、`gppo_world/service_clock.py` |
| 物理进入区域时间 | 无独立字段 | `arrival` clock event 与 `physical_arrival_time` | 需要确认区域半径 |
| 完成消息发送时间 | 仅有普通 telemetry | 新增 completion notice 与发送时间 | 消息丢失保持未确认 |
| 主机合法收到/确认时间 | 无独立字段 | `host_confirmation_time`，受 telemetry delay/loss 影响 | 仍是仿真通信，不是真实网络 |
| deadline 口径 | 旧服务完成口径 | `physical_arrival` 与 `host_confirmation` 可配置 | 主口径待会议确认，禁止混用 |
| 故障、ACK、lease、fencing、能量门禁 | 已实现 | 新协议沿用 | 新增功能测试覆盖能量不足与确认延迟 |
| 后果目标 | travel/service/系统 energy/deadline | arrival time、deadline 前到达、执行/能量失败；系统能耗辅助 | 新模型训练尚未开始 |
| 反事实共享随机性 | 旧持续服务分支 | 新生成器同一父前缀共享 key，保留 NOOP 与失败样本 | 样例仅为开发数据，不是盲测 |
| 规则基线 | 旧任务语义结果 | 新协议两种 deadline basis 各 16 tape 初步结果 | 尚非策略学习结论 |

## 新协议初步规则结果

在相同 16 个 composite test tape 上，合法传统调度器的平均结果为：

| deadline basis | 完成任务/episode | 过期任务/episode | 物理到达事件 | 主机确认事件 |
|---|---:|---:|---:|---:|
| physical_arrival | 3.0000 | 3.0000 | 48 | 10 |
| host_confirmation | 0.5625 | 4.7500 | 48 | 10 |

两行是不同 deadline 定义，不能合并成一个结果。主口径仍待确认。
