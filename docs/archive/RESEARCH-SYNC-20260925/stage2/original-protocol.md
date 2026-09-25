# 正式验证协议：frozen-native-public-history-allocation-v1

状态：方案冻结，待预算授权；不是现成 runner 已验收的声明。执行 AI 可按下述确定规则实现隔离 evaluator，并在获批技术门验收；不得自由选择基线、场景或指标。唯一主比较 M−R，不实施完整消融矩阵。

## 1. 身份与隔离

原生源码根目录：E:\Z博士\migration-artifacts\event-trigger-aware-gppo-fair-replication-20260919-v1\source-snapshot。

模型文件：同制品根 training/seed-1101/P_train/last-recovery.pt，SHA-256 bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a。policy_state_dict/world_state_dict 均由此文件 strict=True 加载，group=P_train；不加载 adapter 权重。

原配置：E:\Z博士\migration-artifacts\preference-weighted-wm-event-cpu-20260917\final\run\training\seed-1101\WD\resolved-config.json 的 environment/training。它提供继承合同，不表示 P_train 就是 WD 训练身份。原生代码与当前 worktree 模块不可混导；独立进程中逐一检查实际模块 __file__ 和 evidence-index 的 hash。单进程 CPU、intra=4/inter=1、模型 eval/no_grad、禁 optimizer。

关键合同：4 UAV、6 slots、25 actions；action=6*uav_slot+task_slot，NOOP=24；horizon18、dt1；arrival_to_region，physical_arrival，arrival_radius0；初始每机能量9、速度1、travel_power .35、idle_power .05；single_shot 完成通知、重试0、command TTL .5、lease TTL2、telemetry_max_age2.5；偏好(.8,.2)、gamma .99、任务/能耗尺度(.5,1)。不使用当前有限通信 coalescing/control reserve 扩展。

证据索引中的完整 native-source-manifest 是依赖身份基准，执行前对会导入的依赖核对；未导入的旧工具不用全面重审。不得应用 proposed-mask-index.patch；若配置切到 bounded_retry，立即停止而非继续。

## 2. 场景清单和随机性

使用原生 formal_three_condition_tape('train', count=N, base_seed=B, condition='W1', name='mixed')，不修改生成器。train 仅是原生成器的 split 枚举；正式新数据在实验元数据标为 independent_development_eval，不回流训练。

- 技术门 N=2、B=924100000，parent gate-000/001，各 repeat=0。
- 正式 N=1600、B=924200000，parent eval-0000..1599，各 repeat=0/1/2。
- 每臂 exogenous_key='frozen-native-public-history-allocation-v1|'+parent_id+'|repeat-'+r；在 M10Environment 构造时传入，相同 parent/repeat 两臂相同。
- 按 parent、repeat 升序；(parent_index+repeat)%2=0 时 M 先，否则 R 先。每 episode 新对象，正常 reset 一次，无历史 pickle、无历史 hidden 续接。
- 用原 CommunicationProfile._uniform(seed,link,identity) 和 _random_identity，不改变消息随机语义。同一 seed 不意味着分叉后的消息流逐条相同，配对依据是相同父 tape、重复键和原身份寻址机制。

生成规则保留原6任务到达/位置/deadline抖动、W1随机地址2秒断联及独立损伤分布。随机 tape、种子、故障表只能进入环境，不能给控制器；规则可知原运动常量，不允许读取本场景未来事件、未公告任务或私有时钟对象。

在任何模型/环境执行前将全部计划 seed 和父 tape hash 写 manifest，并与已允许读取的历史 train/公开开发身份清单去重。若真正重复则技术停止，报告身份，不根据结果补选新场景。不能读旧盲测内容来查重；只用已有获准身份索引。最终 test 保留，不生成、不分析。

技术门 parent 不纳入正式主估计。无候选、丢失、全失败都保留，不按胜负、候选窗口或任务成功筛场景。正式中途不查看聚合效应来决定样本量。

## 3. 控制器信息和共同执行过滤

白名单：当时 obs 中的 flat/graph/uavs/tasks/mask/time/version、public_entity_ids、trigger_flags、event_signal、continuation_actions；同一 episode 过去这些 obs、自己的历史动作。允许的 ACK-known 占用由公开 continuation identity 得到。禁止直接传入 env、clock、execution、scenario、随机键、完整 info、reward 真值、离线 truth、未来消息或其他臂结果。评估器用 info 评分，控制器与评分器必须隔离。

两臂均保存公开历史至18步。共同过滤器先尊重原 mask，再由当前 continuation_actions 映射已 ACK 确认的 task/UAV：删除同任务或同 UAV 的新提交候选，保留合法 NOOP；不编辑原 mask。只据公开句柄，不能假定未确认命令已接受；句柄不再公开即撤销该句柄的过滤，不靠隐藏 lease 状态续期。两臂相同过滤，使已知忙/重复提议不被包装为模型预测能力。

自己的已提交但尚未看到ACK句柄的分配也不能被遗忘：共同过滤器维护 (task,UAV,submit_time) 未确认记录，在对应公开句柄出现时转为上述ACK过滤；若没有句柄，在 UAV idle 与任务 pending 的已收到测量时间均严格晚于submit_time时解除未确认记录，再尊重新的公开mask。否则最多抑制同task/同UAV再次新提交至 submit_time+4.0 秒（原command_ttl .5＋W1 telemetry delay1＋max_age2.5）；到界解除只表示停止本地抑制，绝不推定接受/失败或伪造权限。若一字段始终未收到，按4秒界处理，NOOP仍保留。只记实际本臂提交过的非NOOP，不把模型建议或预测登记成命令。该固定保守规则两臂相同，不能成为模型相对弱基线的优势来源。

这是相对旧实验仅按任务过滤的显式公平性加强；新比较作为新协议，不与旧差值直接合并。纯函数测试需覆盖同任务异UAV、同UAV异任务、句柄消失、无ACK不得宣称接受、非法slot、仅NOOP、原mask不变。

纯函数测试还须覆盖未确认提交、ACK丢失而不能知真实接受、旧测量重复接收不解除抑制、两个新测量到齐解除、4秒边界、不同任务/资源不受影响以及两臂使用完全相同代码。

每决策两臂均选择新提议并使用原 submit_command=True 路径，包括 NOOP；原 env.step 自行维护所有 ACK-known lease。无事件触发跳过 actor、无新增提交接口。如此保持 P_train 的逐步 actor 使用范式，续租语义由原生环境保持；不将 NOOP 当作新有效任务命令。

## 4. 系统 M

按已核验 load_models/probe 路径：policy.encode；world.predict_all_candidates(use_events=True)；policy.evaluate_encoded。从共同过滤后的候选选概率最大动作，同分选 action ID 小者。不采样。

policy hidden 按 next_policy_hidden；world hidden 必须按最终动作 by_action[action]['hidden'] 推进，不能按过滤前 argmax 推进。无候选或 by_action 缺失则技术停止，不补 NOOP。每步三类调用分别在调用前/成功后计数。

不调用 recursive_rollout；不新增 head、校准、微调或 offline updates。维持同一 policy/world checkpoint，模型版本全程不变。

## 5. 系统 R：公开历史有限搜索规划器

R 用同一白名单，不读取 M 的 hidden/预测。它结合已收执行身份和位置/能量历史，安排当前已知任务的连续执行，避免采用忽略 ACK 或只按 priority 的弱基线。

### 5.1 输入状态估计

按公开 entity ID 维护每字段最近已到达的 value/valid/age 和观测时间。输入中 age 与 time 给出测量时点；之后重复看到相同旧测量不能刷新年龄。只把当前 mask/共同过滤中合法的动作作为现实首步。

未来计划只纳入当前已公开且必要字段 valid、pending=1 的任务。ACK-known 执行中的任务不作为待重新分配工作；其公开目标位置可用于估计该 UAV 的释放状态。对只因遥测陈旧而不知道是否可执行的 UAV，当前首步仍不得开放。

用于计划的 UAV 最后公开位置 q、能量 e 取最近合法已收字段，位置年龄 a=max(age_x,age_y)。首段名义距离 d=Euclidean(q,target)，保守余量 b=v*a；到达估计 tau=t_dispatch+(d+b)/v。首段能量估计 .35*(d+b)/v，扣除已知 idle 时间 .05*duration；估计不是安全许可。

已知忙 UAV 若拥有公开任务目标和最近位置，预测释放为 t+(d+b)/v，位置为目标，能量扣相应旅行；无这些字段则整个本次搜索中不可用。当前公开 alive/connected=0 的 UAV 不预测其未来自动恢复；其后收到新的合法恢复信息才能重纳。第一段之后计划状态使用自己的预测位置、年龄0，但只在规划器内部，不能回写公开观测。

在当前公开有空闲可分配 UAV 时，首先处理它们；其它已知忙 UAV 可在预测释放后进入后续计划。全部未来分配仍需能量>旅行消耗且 tau≤deadline；否则不算成功。

### 5.2 确定性搜索

对每个共同过滤后的首动作（含NOOP）建立根分支。节点包含下一提交时刻 g、各机预计可用时刻/位置/能量、尚待分配的已知任务、预计完成时刻表、首动作和后续序列。每次最多提交一个任务；提交后 g 加1。忙 UAV 的后续提交时刻取 max(g,ceil(ready_time))。到达后释放；energy/位置按上节更新。任何 task 只安排一次，过deadline任务从待分配集合去掉并计失败。

NOOP 将 g 加1，可等待已知忙 UAV 释放；不假设未来未公告任务出现。任务数≤6，搜索到 horizon18。分支按 (首动作ID, 后续(action,dispatch_time)序列) 升序生成。按下述已实现预测效用 J 的降序 best-first 展开，同分按序列字典序；一次控制调用最多5000个展开节点。达到上限返回目前最优完整计划；节点未到终局的估价将尚未完成任务全部视为deadline失败，因而每节点都有确定保守终局评分。不调用模拟器 clone/step 作为预测器。

计划评分只计算当前已公开任务相对候选的后续项：预测按时完成在预计到达所在决策区间加 .4/6，未完成且deadline在horizon内在到期区间减 .4/6；预测能耗每决策区间按实际预测 travel/idle 分段积分乘 −.2/36，预测能量耗尽则不继续安排该机。已经在执行的已知任务及释放前能耗对每个根一致地纳入，不能只在某分支扣除。某区间末端发生完成/到期时，区间索引 k=max(0,ceil(event_time−t_now)−1)，折扣 .99^k。计划不加入 priority 权重，不把尚未到horizon截止的未完成任务提前判过期。

只选择最高 J 的计划的首动作，J 差≤1e−12视为同分按首动作小者，之后序列字典序。J为规则的预测评分，不是评价标签。最终动作再校验当前共同合法集合。

如果输入足以形成合法分配但搜索逻辑异常/非有限数，技术停止，不悄悄换基线。若正常无可行完成计划，各根仍有终局评分，选择最高 J（可能NOOP）；没有任何非NOOP候选时直接合法NOOP。记录节点数、截断、J、首步、估计年龄/释放信息；5000上限不按结果改大。规则不拟合新参数、不偷看未来故障概率结果。

实现时用标准化单位与 float64 规则计算；不可改变模型 float32 前向。纯函数测试只验证算法规范和输入权限，不能称为任务收益。

## 6. 技术门和正式运行

授权前仅可做 runner/规则实现与不触及模型/环境的静态、纯函数检查，且不能借此修改正式环境。本协议不自动授权这些之外的任何动态调用。

技术门单独申请：gate-000/001 各 M/R，max72步。正常reset、相同初始公开输入/父tape hash、重复键、真实模块路径、动作合法、奖励独立复算、模型hidden选择、预算事务、原生终止、输出与CPU计时必须通过。无需模型赢、无需出现完成/多候选；缺字段/违规/非原生截断则停止。原 single_shot 下的 _completed_tasks 始终为空可作为无模型/环境扰动的断言；若不为空，说明身份或合同不匹配，停止，不直接打补丁。

技术门通过后提交结果和原协议hash，等待正式预算授权。不得拿72步结果选seed、调整规划器或改变阈值；实现缺陷需要修正时保留失败，重新明确授权，不自动重跑。

正式矩阵一次性1600×3×2=9600 episodes，每个最多18步、原生done停止、不做bootstrap尾值。不复用技术门或历史结果。每步先预留，再执行，再独立奖励复算并持久化，再verified；失败保留unknown/pending，不退款、不补执行。

使用新阶段标识和用户明确授权对应的预算，原SQLite只读保留。新建阶段账本仅在批准后，不借用214或其他历史余额。保存每个parent/重复/臂的唯一键，不允许双attempt重复消费。预算达限即停止；未完成矩阵不填0、不冒称完整。

## 7. 主指标、统计与任务护栏

逐步 r_task=float32((new_completed−new_expired)/6)，r_energy=float32(−max(0,E_prev−E_now)/36)。标签从info/资源账本生成，控制器不得接触此真值通道。独立复算与原 _vector_reward 一致，绝对差≤1e−6，否则停止。

U=Σ(k=0..T−1) .99^k(.4*r_task[k]+.2*r_energy[k])，正常reset后的第一步k=0，原生done后无尾项；单位为原生偏好效用，非百分点。主差 d_parent=mean_3repeats(U_M−U_R)，全部1600 parent等权。

主区间：parent percentile bootstrap 10000次，random.Random(20260924)，每次有放回抽1600个完整parent差，取线性插值2.5%和97.5%。只有一个主比较。报告mean、CI、所有parent差及重复分布；不做按结果选子组或同时比较多个checkpoint。bootstrap有限样本覆盖并非保证。

实用下限δ=.01。在同单位下，单个净新增完成项不计避免过期和能耗约 .4/6×.99^k=.056～.067；.01相当于每约6 episode多一次该项的净收益量级。这是前置研发采用门槛，不是经济估值；当未完成转为完成同时避免过期时贡献更大，不能反推固定任务次数。不得改尺度/终止后照搬δ。

全部任务完成、按deadline物理到达、按deadline主机确认、最终确认、过期和累计能耗均独立报告。物理按时完成率及按时主机确认率的parent宏平均，M都须≥R作为采用护栏；均值护栏不构成非劣证明，仍报告其区间。主效用通过但护栏失败不通过总体采用门。EXPIRED明确为失败、缺失明确标缺失；缺主要标签技术停止。

规模依据见 analysis.json：历史parent SD=.093821345，1.5倍敏感性SD=.140732018；N1600时近似CI半宽.006895742，对真实Δ=.02或0相对δ=.01的正确去留概率约81.1%。未知新分布方差或Δ靠近δ时会不确定，计划不承诺一定得到结论。预算未获批到这一规模则不运行缩小版并声称同等功效；另作新决策。

补充敏感性：SD为旧值2倍时，Δ=.02的上述功效约56.8%，Δ=.03约98.9%。不把1.5倍选择称为有保证的方差上界。独立parent只支持此生成器分布下的新抽样，不宣称跨任务家族泛化。

## 8. 性能—成本联合判断

同机顺序交替臂；控制器计时从白名单输入构造到最终动作/hidden提交结束，包含历史维护、共同过滤、规划和三类模型前向。环境、日志、SQLite另计，启动/模型加载/首次决策单列；不得用环境开销掩盖控制器开销。计process_time CPU、perf_counter wall、峰值进程RSS，保留逐决策原始时间。

M控制器平均CPU≤10ms/决策、wall p95≤50ms，峰值进程RSS≤4GiB；阈值对应1秒周期的1% CPU/5%时延研发包络，非现场SLA。R同样测量，报告M/R比值但不假装等计算预算。若业务经济价值或历史训练成本未知，明确尚未支持完整生命周期经济优越性。

分项报 env.step、encode/world/actor、rule调用与展开节点、optimizer/world/offline update、加载/总wall/CPU、消息sent/received/dropped；原合同无真实token账本就不报token节省或“通信成本相等”。模型输出准确率只能作为诊断，不作为通过标准。

## 9. 授权资源表与停止

| 阶段 | env.step | encode | world批量前向 | actor | 更新 | CPU wall | 制品 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 技术门 | 72 | 36 | 36 | 36 | 0 | 1h | 1GiB |
| 正式 | 172800 | 86400 | 86400 | 86400 | 0 | 36h | 40GiB |
| 条件合计 | 172872 | 86436 | 86436 | 86436 | 0 | 37h | 41GiB |

所有冒烟/失败调用包含在上限内，不额外预热episode；实际早停剩余不转用。三类前向合计上限259308，不把world内部25候选误写为25个独立环境步。GPU-hours=0。

停止条件：未授权、哈希/配置变更、错误action空间、模块混导、私有信息泄漏、首门失败、非法动作、非有限值、hidden选择错误、奖励/标签缺失、真实随机机制不匹配、SQLite预留不一致或unknown/pending、重复attempt、预算/时间/内存/制品上限、非原生截断。保存状态和失败尾部，不自动重试、不覆盖旧账本。

最终：CI下界>.01且任务/成本门通过→只建议新预算world归因；CI上界≤.01或任务/成本门失败→简化/停止当前冻结模型；跨阈值→明确不确定并停止。不得自动扩训、调参、更换主指标或追加parent。记录无安全违规也不意味着生产安全验收。

## 10. 给执行AI的完整指令

只执行用户明确批准的阶段。先读本包report/protocol/decision/evidence-index，核对包hash及许可，不读取盲测。复用已验证原生模型加载和奖励合同，按第3～5节实现共同公开过滤与确定性R，不改变正式环境/通信/模型。完成纯函数/身份/模块隔离检查，冻结实际evaluator源码hash与parent清单；不声称代码完成就是研究完成。

获批技术门后仅运行指定4个episode，上限72步/三类前向各36，报告接口、预算、全成本和失败情况，不按胜负挑场景。技术门通过后停下，正式阶段须另获172800步/各86400前向的明确授权。正式获批则按冻结1600 parent清单一次执行、全量评分，不中途改规则或看聚合效应决定继续。完成后依据第9节给唯一去留结论，保留负结果和不确定性。无额外授权不commit/push/Release、不向其他任务发送消息，不使用用户对话中的任何凭据。
