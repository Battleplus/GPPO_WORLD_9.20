# 独立 heartbeat 配置：零环境步兼容性验证

## 结论

本轮没有重复既有 heartbeat 实验；现有目录只有原先的单一 heartbeat 检查，未有独立配置实现或完成结果。已完成最小配置解耦和零环境步组件验证，具备进入 `UAV=4、task=2` 的后续开发验证条件，但本轮没有运行该环境验证。

## 1. 实施范围

保留旧字段 `heartbeat_interval`，新增可选字段：

- `uav_heartbeat_interval`
- `task_telemetry_heartbeat_interval`

未指定新字段时，两条流分别继承旧字段；显式指定时只覆盖对应流。序列化保存原始覆盖值、两个有效值和继承来源；旧协议只含 `heartbeat_interval` 时可正常读取。

环境中把原共享 `_contract_last_heartbeat` 拆为：

- `_contract_last_uav_heartbeat`
- `_contract_last_task_telemetry_heartbeat`

`_deliver_observations` 的 UAV 和 task heartbeat 触发分别使用对应有效间隔。非 heartbeat 的值变化、event、消息类别、测量时间、TTL、合并、队列、token、completion、ACK/lease/fencing 路径未改。

## 2. 独立默认等价证据

默认配置仍为 `heartbeat_interval=2.0`，两个覆盖值均为 `None`。测试使用修改前快照语义实现的独立旧参考：单一间隔和单一共享标记在 `t=0,1,2,3,4` 上驱动两条流，并逐事件比较当前实现的 UAV/task 消息顺序与类别。该测试通过，未用新实现互相比较来宣称兼容。

独立触发 fixture 还验证了：

- `UAV=4、task=2` 时 task 在 `t=2` 单独触发，UAV 在 `t=4` 触发；
- `UAV=2、task=4` 时 UAV 在 `t=2` 单独触发，task 在 `t=4` 触发。

## 3. 测试

所有本轮测试均有 pytest 自动 fixture，调用 `M10Environment.step` 会立即抛出失败；实际 `env.step=0`。

准确命令和原始输出保存在：

`E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\independent-heartbeat-config-20260921\test-commands.txt`

结果：

- `test_independent_heartbeat_config.py test_m10_uav_state_coalescing.py`: 17 passed
- 合同预算/队列/控制保留定向测试: 9 passed, 3 deselected
- completion 候选及集成定向测试: 16 passed

需要真实 episode 的既有测试未运行，避免消耗环境预算。

## 4. 预算与身份

本轮没有创建、修改或预留任何环境预算。只读 SQLite 核对结果：

- `E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\budget.sqlite3`: limit 256, reserved 246, verified 246, unknown 0, pending 0；integrity `ok`。
- `...\small-multiscene-baseline-20260920\stage-budget.sqlite3`: limit 3072, reserved/verified 1408, unknown/pending 0；integrity `ok`。
- `...\uplink-rate-sensitivity-20260920\stage-budget.sqlite3`: limit 3072, reserved/verified 0, unknown/pending 0；integrity `ok`。

历史累计 1668 步仅作阶段上下文，不称为项目全部历史消耗。

源码修改前后快照、精确 diff、预算 JSON、运行时身份、测试输出和完整哈希位于本报告同目录。修改前核心哈希：`m10_contract_v1.py=87389e82a69d108e28f77e36d4e06fecde607f932f244882c71357c9dd1cab54`，`m10_environment.py=f939a7c4697fc64a14c99f5640b1cb34fd422dcdc3e39e6e76dfba73c368c3d4`。修改后核心哈希：`m10_contract_v1.py=ce9ba291fcd1b1a53dafbc82b378b38f73ac4f94879b1527ae641cb7506e5731`，`m10_environment.py=26a4b185951565b7d636fbb74eefc2b5e2dbea0217f98e1ed6f72c33c7fe35c0`。

## 5. 是否可进入后续验证

可以进入，但仅限后续另行登记的 `event` 环境开发验证，配置为 `uav_heartbeat_interval=4.0`、`task_telemetry_heartbeat_interval=2.0`，并继续冻结其他通信和执行合同。本轮不把组件通过写成环境级可行性或任务收益结论。
