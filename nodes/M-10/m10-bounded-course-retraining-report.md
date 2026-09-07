# M-10 并行租约修复后的有界课程补训报告

日期：2026-09-08  
代码基线：远端分支 `execute-r02-20260905`，最终本轮提交 `7f19b1d1fc27192d5e23598968b1e8f464bbee69`  
修复提交：`3d5df1c794443e4431ada7b319517cf7cf345afc`  
范围：阶段一 Graph-5 Base 基础学习能力、阶段二 A 组弱通信课程；本轮未启动 B/C，也未读取最终测试 tape。

## 训练前核对

- 主工作树仍为 `6da0a3b`，其 staged/修改/未跟踪文件未触碰；并发提交进程未终止。
- 隔离工作区当前完整收集为 145 项，使用工作区内 basetemp 后 `145 passed`；本轮相关针对性集合为 `33 passed`。
- 此前 171 项来自主工作树。修复隔离快照缺少 `test_m10_environment.py`（4 项）、`test_motion_clock.py`（7 项）、`test_task_lifecycle.py`（10 项）、`test_task_policy_view.py`（6 项）、`test_telemetry.py`（4 项），合计 31 项；同时新增 `test_m10_parallel_lease.py` 5 项，因此净差 `171 - 31 + 5 = 145`。这不是通过删除必要测试制造通过。
- GAE bootstrap/历史边界、触发语义、通信账本、候选身份、因果输入隔离和并行 lease 回归均在当前源码/测试中保留。
- `train_policy`、`collect_rollout`、`evaluate_policy` 和 benchmark 均直接实例化 `M10Environment`，由同一 `TaskExecution`、通信和 ACK/renewal 合同执行；续租不是测试入口特判。
- 服务器专用 `m10-test-venv` 原先无 NumPy/Torch。GPU wheel 因外部 `pypi.nvidia.com` 超时未完成安装，未使用共享环境；仅在专用 venv 安装 `numpy 2.2.6` 和 CPU-only `torch 2.14.0+cpu`，因此训练设备条件为 CPU。

## 阶段一：基础学习能力

运行目录：`/home/user1/m10-runs/20260907-bounded-course-phase1-v2`，运行进程 `2060033`。阶段一从头初始化 Graph-5 Base，三个训练 seed 为 1101、2203、3307，每 seed 8192 环境步；阶段边界为单任务 2048、并行任务 4096、无通信故障随机任务 8192。每个验证集合 64 条独立 tape，训练/验证分离，未读最终测试。

| seed | 单任务完成率 | 并行任务完成率 | 合法调度器并行完成率 | 完成差异 | 安全错误 |
|---:|---:|---:|---:|---:|---:|
| 1101 | 100% | 100% | 100% | 0 pp | 0 |
| 2203 | 100% | 100% | 100% | 0 pp | 0 |
| 3307 | 100% | 100% | 100% | 0 pp | 0 |

阶段一三 seed 全部通过，满足进入弱通信课程的实验性前置条件。checkpoint 在 2048/4096/6144/8192 边界保存，optimizer recovery state 保留。

另有一次被中止的 v1 尝试：因初版 runner 每 256 步错误评估且服务器源快照缺少 `run_m10_r3.py` 依赖而停止；该目录/日志保留，不计入正式结果。修复后 v2 重新从头执行。

## 阶段二：弱通信课程

运行目录：`/home/user1/m10-runs/20260907-bounded-course-phase2-v1`，Python 进程 `2245120`。阶段二从阶段一各 seed 的 8192 步 checkpoint 恢复，新增训练每 seed 16384 步，累计每 seed 24576 步；课程固定为 telemetry-delay → random-loss → burst-loss → reorder → recovery → composite。每一级使用 64 条独立验证 tape；最终测试未使用。

最终 composite gate：

| seed | 完成率相对合法调度器差异 | deadline 差异 | 可恢复事件成功率 | 可恢复事件数 | 安全错误 | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1101 | +4.95 pp | -4.95 pp | 12.5% | 8 | 0 | recovery gate fail |
| 2203 | +0.78 pp | -0.78 pp | 0% | 11 | 0 | recovery gate fail |
| 3307 | +7.81 pp | -7.81 pp | 25% | 12 | 0 | recovery gate fail |

可恢复事件定义已冻结为：事件在受影响 UAV 上中断任务，之后由另一 UAV 合法接受该任务；任务最终完成才计为恢复。分母只包含可审计的事件—任务接管对，未观察到可恢复对的事件不被静默删除。完成率与 deadline 门槛和安全门槛均通过，但三个 seed 的 recovery gate 均未达到 80%，因此 runner 判定 `all_seeds_passed=false` 并停止。

阶段二没有启动 B/C，没有世界模型训练，没有增加 16384 步上限，也没有通过关闭 NOOP、版本、ACK、lease 或 fencing 门禁制造完成率。

## 制品与哈希

- 阶段一 `course-results.json`：`49d004b2275bde9327d00412a49c9450f98b50b6035b31c000f4c733f77affba`
- 阶段一 `run-complete.json`：`6e0e14ce54000a52c1bb450bad82ce6dd91933359495ca7aaa02b00698db94be`
- 阶段一 `protocol.json`：`941f9d54ed442727b710caf0690d810c70ca9578c12135ce199195d498f5c351`
- 阶段二 `course-results.json`：`f1796a5b17d69341553c71034a63e1bf95aad1d61079baa60178f96cc5df18e7`
- 阶段二 `run-complete.json`：`a156d97ea1fbb14f91a948314fcf7a5d2464580d6c8cc9442dfe91ec166e2d26`
- 阶段二 `protocol.json`：`1950f7f85ba44251b7400322397ec95f0baccc23ccc7b7bee025a9612bf9c5ea`
- 服务器阶段二 tar 包：`8cadc2c217aa8eccb46b6c2ecf355cb3b5587f2eb73657694c746fe6325b458b`

本地下载目录：`server-bounded-course-phase1-v2`、`server-bounded-course-phase2-v1-download2`。阶段二包含 24 个逐边界 checkpoint、逐 seed 结果、训练/验证 tape、协议和日志。

## 决策

- A 已通过基础单任务/并行学习能力门槛。
- A 未通过弱通信课程的可恢复事件门槛；当前不进入 B/C 公平扩训。
- 这证明修复后策略在明确可行基础场景中可以学习有效分配，但不证明弱通信下恢复能力达到可用标准，也不证明世界模型收益。
- 当前停止并保留结果。下一研究假设应优先解释跨通信课程的恢复失败（事件获知、重分配时机、续租/接管信用分配），另立预算和停止条件；不得把本轮负结果重命名为稳定收益。
- 返航、换电、充电、真实控制周期和人工汇报活动仍是范围外/待确认事项，不混入本轮训练结论。
