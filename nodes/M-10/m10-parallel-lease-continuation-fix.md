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

在隔离工作区 `E:\Z博士\9.2日\GPPO-WORLD-9.2-lease-fix`，通过：

```text
24 passed
```

覆盖：双 UAV 同时服务、NOOP/新命令期间的全部 lease 续租、单 UAV 损毁只中断对应任务、单 lease 到期不影响另一 lease、续租 command 丢失、续租 ACK 丢失、续租延迟、重复、重排、幂等、服务/能耗不重复，以及既有 ACK、版本、fencing、GAE/触发记账回归。

直接验证脚本 `tools/validate_m10_parallel_lease.py` 也通过，输出确认另一任务完成、活动 lease 收敛为 0、产生 18 条续租审计消息、重复/越权计数为 0。该脚本是服务器无 pytest 时使用的合同审计，不冒充 pytest 全套。

## 结果解释

修复通过只证明并行执行合同在小规模和通信边界下工作，不证明完整弱通信场景可用，也不证明策略已经学会分配。本轮不训练、不扩大预算；服务器配对复评需使用相同冻结 tape 和旧 checkpoint，并与修复前 24/96 结果分开报告。
