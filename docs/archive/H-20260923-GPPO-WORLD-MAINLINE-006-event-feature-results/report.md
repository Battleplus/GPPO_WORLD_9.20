# H-006：冻结事件特征干预实验结果

固定实验完成：48/48新分支，24/24完整配对，8个已观察开发parent×3repeat×2arm；W1、seed1101、原生M10 25-action。未重跑、补样、更新模型或改动冻结源码。研究主线整体仍未完成。

## 主结论与研究决策

normal 和 event_features_off 的父场景宏平均主偏好效用均为 **0.27682090172392726**。normal−off=**0.0**，冻结10000次父级bootstrap(seed20260923)的95%CI为**[0.0, 0.0]**。24对差值和8个parent均为零，所以区间退化；按冻结规则判为 **inconclusive**，不能宣称等效、没有一般作用或世界模型整体无效。

**决策：结束并封存本固定开发矩阵的事件特征关闭实验，不补样追正、不改偏好/指标。当前没有证据支持事件通道带来动作或效用增量；为可比性保留现有normal冻结基线，不据此删除训练组件。**

唯一建议的下一步：静态定义同原生25-action、同公开信息、同合法NOOP和同guard的**无模型规则对照**，使之后能区分模型整体贡献和guard贡献。只提出此方向；本轮没有实施或启动下一实验，动态比较仍需具体协议及单独批准。H005比较中的基线含冻结GPPO/world，不能把本结果当作已经胜过无模型规则。

## 授权与资源

用户明确批准修正包v2，归档 e1506c99b344447c3a50c4b93443eb5e6128922f；包SHA-256 `5ef0d32def14fa399e49ea38856b4504de69db48c7cd5f381b6820a576d9c8f3`。复用43项测试，执行前仅核对固定身份、源和账本，不重跑历史实验。

- 原同一SQLite上限404→1067，既有299步保留，历史reservation逐行与迁移前备份一致。
- 新环境步：558/768；全局reserved=verified=857/1067；pending=unknown=0；integrity=ok。
- global和本run剩余210，不自动使用；没有退款或替代账本。
- policy encode：558 attempted/558 completed；world candidate batch：558/558（25行/batch）；actor readout：560/560（含首对2额外读出）。
- optimizer/world/offline update：0。legacy model_forward=558表示probe次数，不是全部模型组件调用总和；3类调用合计1676。
- 正式执行exit code=0，分析exit code=0；无动态failure-ledger。首对唯一step1门通过。

SQLite最终SHA-256：`ea42090b92b9ecea9ab1c487742dbb059d4f95c2087f90f8d66dbfbb6be38d50`。

## 每个父场景

同一parent内先平均3repeat，然后8parent等权；奖励为原生float32 vector_reward，gamma=.99，主效用=sum(.99**k*(.4*r_task+.2*r_energy))，恢复后首步k=0，原生终止、不加价值bootstrap。

| parent | normal效用 | off效用 | normal−off | 三repeat差值 |
|---|---:|---:|---:|---|
| parent-00 | 0.348778686594 | 0.348778686594 | 0 | [0, 0, 0] |
| parent-01 | 0.184646395305 | 0.184646395305 | 0 | [0, 0, 0] |
| parent-02 | 0.189815565354 | 0.189815565354 | 0 | [0, 0, 0] |
| parent-03 | 0.307692706116 | 0.307692706116 | 0 | [0, 0, 0] |
| parent-04 | 0.343552379727 | 0.343552379727 | 0 | [0, 0, 0] |
| parent-05 | 0.351020020011 | 0.351020020011 | 0 | [0, 0, 0] |
| parent-06 | 0.187222137204 | 0.187222137204 | 0 | [0, 0, 0] |
| parent-07 | 0.301839323480 | 0.301839323480 | 0 | [0, 0, 0] |

以上为均值展示；完整未舍入值、身份、分支步数见parent-results.json、sample-index.json和原analysis.json。

## 任务、能耗和命令

以下是每臂24分支的累计计数，含3次repeat，不作为独立样本量；两臂各279步。

| 指标 | normal | event_features_off |
|---|---:|---:|
| 完成任务增量 | 130 | 130 |
| 原登记cohort按时物理到达 | 48 | 48 |
| 原登记cohort按时主机确认 | 48 | 48 |
| 所有记录任务按时物理到达 | 130 | 130 |
| 所有记录任务按时主机确认 | 106 | 106 |
| 能耗总量（原合同单位） | 145.00367481243035 | 145.00367481243035 |
| 命令接受 | 129 | 129 |
| 拒绝/未接受反馈 | 56 | 56 |
| task_unavailable | 0 | 0 |
| submit_command=true | 279 | 279 |
| NOOP | 94 | 94 |
| guard触发/改变动作 | 120/120 | 120/120 |

56个非接受反馈细分：resource_busy37、resource_unavailable11、command_lost8。另94个NOOP是noop反馈，不是物理命令接受。cohort与所有记录任务集合不同，不能混用分母；submit标记不等于实际有效命令。

## 干预是否实际生效

首对两臂唯一step1的公开输入、policy/world hidden、raw candidate、by_action hidden及next policy hidden一致，actor-facing仅末5维事件输入改变；额外两次读出复用encode/world输出。原首对数值分析：概率TV=0.00243583507835865，top-1和guard选择均未改变。

附加零推理描述性核对：先逐记录验证输入/hidden/raw candidate一致，再比较对应实际轨迹；279/279输入匹配，230/279概率分布发生变化，7/279完整合法动作排序变化，**0/279 top-1变化、0/279最终选择变化**。平均TV=0.002623131772416467，最大0.009145542979240417。279步的实际动作/反馈/奖励/能耗/终止字段全部配对一致。

这说明事件通道确实影响分数和概率，但在此冻结确定性选择+guard矩阵中没有跨过改变动作的边界。该补充描述不改变预登记首对指标、主效用、样本或统计规则；无反事实分支、无新模型前向。

## 实际计时与测量边界

- setup：6.7015747秒；runner内部总wall：133.9375956秒，未含外部分析耗时。
- normal分支累计59.6793875秒；off59.6059882秒，不主张off更快。
- feature-ledger累计：policy encode0.5096547秒、world batch0.3980958秒、主要actor readout0.2947368秒、额外readout0.0013574秒。
- feature-ledger probe总时长7.8779894秒；branch timing中的早期probe总计1.4283958秒。两者采样位置不同，后者尚未包括大部分tensor转JSON记录处理，**不可把它们当成同一计时量**。
- branch_seconds−早期probe包含环境、guard、预算、奖励复算及部分日志开销；不是独立env.step耗时。单次CPU短运行和序列化开销不支持计算节省结论。两臂均实际计算world，off并不节省world调用。

## 异常与解释限制

无动态技术停止、非法动作或账本未结清证据；冻结分析通过，source与模型权重摘要未变。此前包v1入口错误已在v2勘误且旧提交保留，未在错误包上运行实验。Luna此次未按要求在开始/首对阶段主动发送进度，只在结束返回；这是协作可见性不足，不补造中间通知。PID/命令/时间和退出码见真实command记录。

置零是接口关闭，未证明训练分布内中性；其余12维世界候选输入和训练影响保留。因此这不是完整no-world消融。使用已观察开发场景和单一模型seed，不能声称独立泛化。整体pickle恢复后才筛选prefix，反序列化边界限制与执行包相同；未用heldout选择、分析或调参。

## 审计与制品

本地原始制品保留于 `E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\world-event-feature-authorized-run-20260923`；批准、命令及迁移记录位于 `E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\world-event-feature-approval-20260923`。review.json绑定当前账本、分析输入哈希、完整样本、模型计数与历史299步一致性；artifact-manifest.json列出大文件及中间staging的完整哈希。大型checkpoint、pickle、SQLite及原始大ledger不上传，远端仅选择性归档必要小文件和证据索引。

本报告编制新增env.step/model forward/update/attempt均0。完整配对结果已得出，本轮局部研究完成，整体GPPO＋世界模型各组件独立价值仍未验证，不标记整体目标完成。
