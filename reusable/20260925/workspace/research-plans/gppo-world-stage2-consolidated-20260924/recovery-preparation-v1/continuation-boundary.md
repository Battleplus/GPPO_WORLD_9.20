# Episode续行边界

## 可保留的语义

历史M10/25-action/P_train/W1身份不变。原stage_worker.py逐字节复制，Worker.choose/reset/one_step未改；decision_only_inference.py与analyze_formal.py也逐字节一致。恢复适配只改授权、路径、调度后缀、计账与独立观察输出。

stage_worker.py:112 的reset每次新建M10Environment与scenario_from_dict结果；:116 调用正常reset并新建PublicMemory，state=None、step_index=0、done=False；:117–119 重置前计数、能量与utility。原生m10_environment.py:302重建ServiceClock/TaskExecution/TaskPolicyView/bridge、序列、事件游标、消息队列、活动命令及确认集合；:659公开reset再进入_reset_state。

policy/world在Worker.__init__严格加载完整state_dict、eval且requires_grad=False；choose:58使用no_grad且确定性max(probability,-action)，不sample。policy GRU状态和world latent均从显式state传入，episode开始为None。检查原生joint_gppo和M10ActorCritic：实际路径无随机抽样、dropout、BatchNorm运行统计更新或按调用次数改变输出的cache；GRU内部权重布局缓存不作为语义状态，权重未训练。构造时的初始化随机数被严格checkpoint加载覆盖，不参与决策随机键。

PublicPlanner仅保存固定node_cap/horizon；choose的队列、serial、best、节点和计划都为局部变量。所有公开历史都在每episode新建的PublicMemory中；没有跨episode规划缓存。

CommunicationProfile._uniform(m10_communication.py:86)直接以scenario seed、link与语义identity做SHA256；M10Environment._random_identity(:297)以冻结exogenous_key映射，不依赖全局随机流或前一episode调用数。场景已冻结保存，恢复不调用生成器。新进程不能通过改变生成器RNG改变下一场景。

因此按当前源码依赖，重建两个worker不会丢失下一episode所需的历史状态；只需重载同一模型和固定配置。23项零步检查包含原reset方法的纯桩隔离测试、实际恢复循环5566项纯桩遍历、源文件字节比较与RNG路径静态检查。纯桩测试不是一次真实环境运行；本轮没有数值重放或跨进程动态等价新证据，复用A/B及既有身份/公开reset核验。

## 配对与边界

完整冻结矩阵按parent→repeat→交替arm顺序列出9600键。逐步索引证明前4034键均完整终止，恰好2017个M/R配对；所有已开始episode都已结束，4034条reset记录与4034个终止段一一对应。每对reset公开哈希一致，外生键、tape hash与原scenario清单一致。

下一键是eval-0672/repeat-1/R（ordinal4034），然后同pair M；恢复必须精确遍历not-executed.json的5566键，不接受跳过、追加或调序。

## 代价与限制

任务结果/统计合同可作为原固定矩阵的分段续行保留；不把它称为未中断运行。新增进程启动、checkpoint加载、恢复身份读取、日志和分析全部属于恢复成本。控制器时序合并所有旧C决策和所有新C决策，不删首次/慢样本，不只拼较好部分。新进程/冷热状态可能影响时间，不改变任务分配语义；额外启动成本单列并累计。

旧进程status.json的资源快照发生在最终写文件/退出之前，尾部完整CPU无法事后精确恢复。已测旧总CPU8649.34375秒、总wall9572.7176203秒保留；预算另冻结旧尾部30秒wall/600秒CPU的保守占用，不伪装成实测或已证明上界。恢复也在申请额度内预留相同收尾空间。若未来必须精确获得旧进程退出后的完整CPU，本次零步恢复无法补出该历史值；最终报告必须保留该缺口，不能声称精确全生命周期成本已追溯修复。

该成本计量限制不允许放宽10ms模型控制器CPU均值、50ms wall p95或4GiB合计RSS门。新段仍按完整进程采样、保守收尾占用和原累计上限执行。预算批准应明确接受这里的分段执行和成本边界。
