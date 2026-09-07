# M-10 并行 lease/continuation 修复

日期：2026-09-07  
基线：`3d743426e660d11e67073758f17b8fd6587a65a6`（NOOP/action-capacity 诊断提交）  
范围：执行合同修复与针对性验收；本轮无训练。

## 修复内容

`gppo_world/m10_environment.py`：

- 将控制端已确认的 continuation 从单个 `_active_command/_active_action` 改为按 `command_id` 保存的活动集合；旧单值属性仅作为兼容性诊断视图，不再参与控制。
- 每个环境步最多仍提交一个新分配；新分配或 NOOP 不会清除其他已确认 lease。
- 每个周期为所有既有 ACK-confirmed lease 各发一次续租；新 lease 从下一周期开始续租。
- 续租失败只移除对应 command，完成、损毁、断联、能量耗尽和 lease 到期不会清理其他 command。
- 删除了基于 `execution.leases` 的环境侧隐藏真值轮询。控制端只在收到续租响应后更新自己的活动集合；执行端接受但 ACK 丢失时保留最后控制端知识，后续通信再收敛。
- 新增 `continuation_actions`、`lease_renewals`、`lease_renewal_delivery_results` 和 `active_continuations` 审计字段；不作为隐藏真值输入策略。

`gppo_world/m10_communication.py`：

- 续租使用独立的确定性 renewal identity 和 command-loss预算；可配置续租额外延迟、重复概率和重排窗口，默认均为零以保持旧 tape 行为。
- 续租请求进入 command 通道并记录 sent/dropped/received/duplicate_received；送达后通过 ACK 通道确认，ACK 丢失不会被本地伪造为成功或失败。
- 延迟续租在执行时钟内按送达时间处理，晚于 lease expiry 的续租不能复活执行；重复续租对同一 fenced command 是幂等的。

旧训练、旧 Release 和旧失败结果没有改写。旧 A/B/C 的 checkpoint 文件仍可加载，但旧的 24/96 环境结果属于单活动 lease 合同；修复后的配对复评才可用于新的执行合同结论。旧模型的公共输入/动作参数并未自动变成新合同下的训练结果。

## 冻结执行合同

控制器负责在每个决策周期最多提出一个新 UAV–Task 命令，并在下一周期维护其已收到 ACK 的 continuation 集合。每个续租请求有唯一 `renewal_id`，沿 command link 发送；传输成功后执行端按原 command、UAV 和 fencing token 校验，随后返回 renewal ACK。控制端只采纳收到的 ACK。

执行端的 lease 仍是每个 command 独立的 `lease_ttl`。某一 command 的 renewal 丢失、ACK 丢失、stale、失效或到期只影响该 command。旧 token、旧 ACK 和迟到续租不能夺回任务或追溯补服务。长失联自主执行、返航、换电和充电仍不在本轮范围。

## 本地验证

在隔离工作区 `E:\Z博士\9.2日\GPPO-WORLD-9.2-lease-fix`，当前完整隔离测试通过：

```text
145 passed in 7.34s
```

覆盖：双 UAV 同时服务、NOOP/新命令期间的全部 lease 续租、单 UAV 损毁只中断对应任务、单 lease 到期不影响另一 lease、续租 command 丢失、续租 ACK 丢失、续租延迟、重复、重排、幂等、服务/能耗不重复，以及既有 ACK、版本、fencing、GAE/触发记账回归。

直接验证脚本 `tools/validate_m10_parallel_lease.py` 也通过，输出确认另一任务完成、活动 lease 收敛为 0、产生 18 条续租审计消息、重复/越权计数为 0。该脚本是服务器无 pytest 时使用的合同审计，不冒充 pytest 全套。

## 结果解释

## 服务器修复后配对复评

服务器专用目录：`/home/user1/m10-runs/20260907-parallel-lease-fix-v1`。直接合同审计在服务器通过；服务器没有 pytest，因此该结果不冒充 pytest 全套。一次初始复现因源快照漏带 `gppo_world/m10_training.py` 产生了保留的失败日志，补齐同一源快照后复评成功。

使用同一冻结 final-test tape（16 episodes、96 tasks）和既有 A/B/C checkpoint，未训练、未扩大预算。结果文件：`parallel-lease-old-checkpoints-v3-20260907.json`，下载后 SHA-256 为 `083738c6d92f3b8a85d69f9b9254959d7655e741e985ea461f273d91c3827c12`，与服务器文件一致。

| policy | episodes/tasks | completed | expired | accepted commands | max active leases | actor calls | NOOP on valid candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| A Graph-5 Base | 48 / 288 | 29 | 259 | 72 | 3 | 835 | 292 / 420 (69.52%) |
| B Graph-5 World | 48 / 288 | 25 | 263 | 74 | 3 | 838 | 292 / 421 (69.36%) |
| C Graph-5 Triggered | 48 / 288 | 25 | 263 | 74 | 3 | 838 | 292 / 421 (69.36%) |
| legal scheduler | 16 / 96 | 23 | 73 | 80 | 2 | — | — |
| all-NOOP | 16 / 96 | 0 | 96 | 0 | 0 | — | — |

按策略 seed，A 的完成数为 `0 / 1.8125 / 0`（seed-1101/2203/3307，每 seed 16 episodes）；B/C 为 `0 / 1.5625 / 0`。这仍显示旧 checkpoint 尚未稳定学会有效分配，不能把修复后的执行能力误报成策略收益。

审计中 A/B/C 的 `duplicate_or_empty_id=0`、ACK identity 错误为 0、fencing 错误为 0；三者各有 13 个 lease-expired 事件。合法调度器为 10 个 lease-expired 事件。续租账本分别记录 A `renewed=107, ack_lost=18, command_lost=14, inactive_lease=53`，B/C `101/16/13/53`，规则调度器 `99/12/11/60`。消息丢失、ACK 丢失和活动集合清理均按 command 独立归属，未发现跨任务清理。

同一修复环境下，双任务容量探针的两个任务最终均为 `completed`；这是并行 lease 合同的直接证据。它不等价于完整压力 tape 的可用性通过。修复前诊断中的 24/96 以及更早的 42/96 来自不同版本/实现，继续作为历史对照，不与本次 23/96、29/288 等结果混合。

## 结论与剩余范围

本轮确认并修复了单活动 lease/continuation 对并行服务的实现缺口，并完成本地 145 项测试、服务器直接合同审计和旧 checkpoint 的同 tape 配对复评。修复后旧策略的完成率仍低且跨 seed 不稳定，故当前不能声称弱通信任务处理可用，也没有证据要求本轮启动补训。进入补训前的条件已满足“执行合同不再是已知单活动句柄缺陷”，但仍需以冻结场景中的合法有效轨迹、奖励/探索学习证据和明确预算/停止条件为前置；本轮没有训练。

本轮不改奖励、不禁用 NOOP、不放宽版本、ACK、lease 或 fencing 门禁；返航、换电、充电和长期失联自主执行仍不在范围内。修复通过只证明并行执行合同在小规模和通信边界下工作，不证明完整弱通信场景可用，也不证明策略已经学会分配。
