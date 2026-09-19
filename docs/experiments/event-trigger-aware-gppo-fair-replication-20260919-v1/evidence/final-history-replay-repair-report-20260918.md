# T_train 历史重放语义修复及原预算续行

日期：2026-09-18。运行范围为本机 CPU、单进程 4 线程；未连接服务器，未重置截止时间。

## 根因与失败前缀

失败前缀来自 `T_train/seed-1101` 的冻结 128 步 checkpoint，episode 8，场景
`train-mixed-seed-851003-W2`。第一处分歧是 step 3：`action=1`、
`actor_decision=false`、`submit_command=false`、`command_id=null`，反馈
`reuse_existing`；当时新命令 mask 为 false，但 `continuation_actions=[1]`
且续行合法。step 12 重现同类情况：`action=22`、续行集合 `[5,22]`，新命令
mask 为 false，续行合法，反馈仍为 `reuse_existing`，命令身份为 null。

因此实际第一处分歧正是：合法 `submit_command=false` 续行被历史重放错误地用
新命令 mask 拒绝。不是把 mask 失效误判成续行，也不是环境执行反馈分歧。

## 修复后的合同

历史前缀逐步保存 `actor_decision`、`command_submitted/submit_command`、action、
公开 mask、续行集合、命令身份和执行反馈。新 actor 动作必须通过当前公开的新
命令 mask；续行只需通过当时公开的续行资格，并且必须存在于 world 的
`by_action`，不使用 NOOP fallback。旧前缀只有从配对账本恢复出明确决策类型时才
迁移；否则拒绝迁移并要求更早的兼容完整事务。

在线推进和重放共用 `_select_replay_hidden` 的动作分支语义。重放不调用
`env.step`、不重新采样动作、不读取未来真值、不调用 optimizer；environment
安全 mask 仍保持权威，非法新分配和失效续行仍拒绝。

## 无训练验证

真实失败前缀只读重放：`replay-prefix-verification-v4.json`，15 个前缀步，识别
step 3 和 12，policy/world hidden 最大绝对差均为 `0.0`，离散合同完全一致，
`environment_steps=0`、`optimizer_calls=0`、重放 forward 15 次。

针对性 fixture：5 passed。覆盖新分配合法、新 mask=false 且续行合法、非法新分配拒绝、
失效续行拒绝，以及 episode 边界保存/恢复决策类型。Python 编译检查通过；未重做
旧 SQLite 压力测试，未重跑完整微型训练。

## 预算与有效训练量

SQLite 主账仍为同一数据库和稳定 `run_id`。总环境预算 `49152`，总策略调用预算
`384`，world 更新预算为 0。最终账本为环境 `49152 reserved / 49151 verified /
1 unknown / 0 pending`，策略 `379/384`，world `0`。

| 组 | 实际/预留环境步 | 完整事务逻辑步 | 策略更新 | actor / continuation | 差异说明 |
|---|---:|---:|---:|---:|---|
| P_train/1101 | 8192 | 7707 | 58 | 7707 / 0 | 已封存；485 步为未提交尾部，不能补足 |
| T_train/1101 | 8192 | 8191 | 62 | 7551 / 640 | 保留 1 步 pending window，不伪造提交 |
| P_train/2203 | 8192 | 8192 | 63 | 8192 / 0 | 完整 |
| T_train/2203 | 8192 | 8192 | 64 | 7539 / 653 | 完整 |
| P_train/3307 | 8192 | 8192 | 63 | 8192 / 0 | 完整 |
| T_train/3307 | 8192 reserved（8191 verified，1 unknown） | 8122 | 62 | 7437 / 685 | 从 step 7168 完整事务恢复；旧 67 步未提交尾部保留，最终 window pending |

累计已用环境资源为 `49152`，其中可确认的完整逻辑训练量为 `48496`；差异为
P1101 的 485、T1101 的 1、T3307 的 70。差异属于既有未提交尾部、pending window
和未知预留，不能写成完全等预算，也不通过重放或补环境步修正。旧失败、旧
checkpoint、未提交尾部和 unknown reservation 均保留；没有 refund。

T_train/3307 的恢复事务为 logical step 8122、policy update 62、world update 0，
checkpoint `txn-step-00008122-policy-000062-world-000000.pt`。原实验的后续四组
协议没有在本修复阶段擅自启动；剩余资源按主账为 0 环境步可分配，故不启动 validation。

## 哈希

修复源码：

| 文件 | SHA-256 |
|---|---|
| `source/tools/run_event_trigger_aware_gppo.py` | `3f66c7f43a1f279b04f052b2b305593917b016563ba6cc7536d625dc70dfbf9a` |
| `source/gppo_world/budget_executor.py` | `5a2a7536a0ebb0078638a1fd45491a5d13cd9a719c9f012f158b74f5050fd29b` |
| `source/tests/test_event_trigger_history_replay.py` | `17a6eb2af97b288a089e1ba5c31a6a38623c3d0b0ecc3b5401cb882d8eee3330` |
| `source/tools/verify_t_replay_prefix.py` | `dfd91819c9bf65b58553bf4091c8ade15a7c3e9ef6f70db2552c74ac0c1d599b` |

验证及账本制品：

| 制品 | SHA-256 |
|---|---|
| `failure-prefix-contract-v1.json` | `7eca61a1eaae73fe0a274a10d10b9716c7a0e85dfa80648f10a685052b6d11f6` |
| `replay-prefix-verification-v4.json` | `c238b8aa2b116d82ace2a721aecde547412e2ba1d66a43bf901ac825270f8f9d` |
| `budget-sqlite-v2-20260918/persistent-budget.sqlite3` | `2d034909750db09c27867fc73388a44169e4072e32140a1889d5f5edfd71d3f6` |
| `T_train/3307` final checkpoint | `a5c0440003b4b19c99726c8f6f38ab4b6012946ffff13e0adc94442861c590e2` |
| `P_train/1101` sealed checkpoint | `0cde0624199964746b4d7739f1d5238d1339df34407215679a2b3367c10d1d5e` |

## 发布状态

旧 Git 历史未改写，未执行签名重试。修复源码快照、测试、验证输出和本报告已准备
进入 `WORLD-GPPO_9.11` 的独立归档目录；不上传 SQLite、checkpoint、完整训练日志
或 `__pycache__`。Git 发布仅记录可审阅源码和证据哈希。
