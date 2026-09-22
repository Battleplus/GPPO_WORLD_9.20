# 修正后 ACK guard 协议：主代理最终验收

handoff_id: H-20260922-ACKGUARD-PROTOCOL-003

状态：第一阶段交付 CLOSED；第二阶段 NEEDS_EVIDENCE_AND_BUDGET_AUTHORIZATION。完成的是零环境步协议、执行入口、模拟验证和证据审查，动态验收门尚未通过。未进入第二或第三阶段。

Luna（gpt-5.6-luna / max）执行，主代理等待完成后集中审查。原代理被用户要求中断并重建，替代代理接续已有成果；未重复正式实验。主审拒收的中间结果和目录保留，未覆盖历史失败证据。

## 最终范围及验证

- 新增独立runner及其测试，冻结NOOP guard与旧runner保持不变；真实最小diff包含这两个新增文件，逐字重建比对通过。
- 固定8父场景 × 3外生重复 = 24新guard-on，与24历史R/299历史步配对；旧两条guard/20步排除，不重跑、不退款、不拼接。
- 动态入口在模型/环境/预算对象构造前检查显式授权、源码及输入身份、完整矩阵、历史R兼容性、同账本足额及unknown/pending、已有attempt/run和部分输出。
- 分支注入实际外生键，按最终所选action推进world hidden；保留全候选概率、mask、反馈、奖励标签、原生终止与每分支计数增量。独立副本及基础snapshot前后摘要覆盖env/obs/policy-hidden/world-hidden/probe，失败也记录摘要与真实预算状态。
- 独立复跑81 passed（15.85秒）：含授权门逐项拒绝、合成24分支入口、两步selected-hidden推进、预算最后token竞争、写入/complete前后故障、pending保留、重复与部分执行防护、源隔离、预算副本迁移。全部为纯函数、stub或临时test-only SQLite；没有真实环境/模型调用。
- 16项当前制品/源码哈希、21项原阶段制品哈希匹配；正式SQLite只读integrity=ok，SHA保持700b0c9173c8debd5cacb8066328dd2359bb913a128f3018710c8129de5e6223，reserved=verified=20/384，unknown=pending=0，历史attempt总数1。本轮真实env.step/model forward/全部更新/正式attempt增量均0。

## 历史R判定

23/24通过当前原R对照复用审查，1/24证据不足。完整24矩阵门槛因此阻断，不丢弃异常对或缩为23对。

待核对项：parent-00|W1|seed-1101|prefix-0|repeat-0|mode-R，来源reused_from_pilot。R首步公开观测摘要为029fdda5e21097ca33a4fdf738f9a0a5ada2a80d35440e4d3ea9e95100e25a87；登记prefix摘要为969682e664ab722f982fd6e10af98291705ed2c311e155ae894d71733918e64f。

摘要字符串不一致已经证实；是否仅为pilot/后续采集摘要schema差异尚未证实。不能由此断言真实初态一定不同或受污染，也不能自比较snapshot记录冒称通过。此项需要追溯既有pilot源码、observer-equivalence、原观测及摘要定义，不能重跑补证据。

R是原策略对照，不要求与新guard动作一致；缺少全候选概率也不自动否决原R对照。旧两条guard规则改变与hidden/cache历史缺证据是独立问题，仍不得复用为新规则轨迹。

## 预算方案及停止条件

24×16=384新增步；旧20必须保留，最坏总量404。原384上限仅余364，不足完整矩阵。建议未来明确批准同SQLite总上限384→404（增加20上限），新run封顶384，旧消费与历史记录不变。迁移仅在临时副本测试，正式账本未写入，authorization-template仍pending。

独立的动态阻断：R兼容性未全通过；预算扩额及动态执行未授权。两个条件均解决前不得运行。预算批准不能替代R证据门，测试成功也不能替代真实动态接受证据。

主指标冻结为各parent先平均3repeat，再8parent等权计算guard−R；U=0.8×0.5×G_task+0.2×1.0×G_energy，gamma=0.99，终止步包含。任务完成、能耗、拒绝、NOOP、主机确认和计算调用同时报告。负结果、通信失败及任务失败不重试。

下一步只建议先对上述单个pilot复用项做零环境步摘要口径审查；不增加样本，不启动矩阵、训练或冻结模型对照。若既有证据不能解决，保留证据不足并另行决策，不自动重采。

## 归档与身份

runner SHA-256: 83fead84240311a0518a9b965b06497e553083ee3fa779b38ee3e2f43f328789

原完整evidence-index SHA-256: 34a499236f16614fd7984744558a09c7e7a40869b9cc90bbb862fb09b45b21e8

完整逐对兼容性JSON本地保留；远端归档其紧凑审查摘要、路径和完整文件SHA，未上传权重或原始大制品。主代理归档行为另计，不改写执行者“未push”的历史陈述。最终远端提交和回读记录另存final-remote-verification.json。
