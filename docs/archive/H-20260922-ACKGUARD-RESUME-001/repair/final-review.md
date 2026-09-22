# ACK guard 修复：最终独立审查

handoff_id: H-20260922-ACKGUARD-RESUME-001

本单交付及审查 CLOSED（技术阻断交付）；实验为 incomplete_technical_stop，下一阶段 NEEDS_USER_DECISION。未完成矩阵，不作效用或算法收益结论。

执行者为 luna_worker（gpt-5.6-luna，max）。主代理在执行者完成后集中验收。runner 修复首对配对、累计与分支消费区分、逐前缀语义摘要、前向增量、实际首步反馈及已有记录防重跑。动态 resume_run 明确禁用，未实现可运行的续行循环；不能称为续行功能完成。

主代理独立回归：27 passed（两个定向测试文件，pytest 禁用 cacheprovider）。原制品21项、新修复制品21项 SHA-256 全部匹配。最小差异仅包含 runner 与新增回归测试；guard 保持不变。

原两条分支共20条 step 的保存身份、动作合法性、排除任务身份、hidden 摘要链、float32 奖励独立复算、终止与预算核验通过。这是已保存记录范围的核验，不是完整历史状态无污染证明。历史 env_state_before_sha256 实际在 step 后计算，不作为前状态证据；历史 safety_violation 仅为最终标记，不作为逐步安全证明。记录未保存所有候选概率，不能完整独立复算 argmax。

两个阻断：

1. 冻结 guard 会在存在分配候选时丢弃合法 NOOP。纯函数反例中，无 continuation，action 0/24 概率为0.1/0.9，guard仍选择0。原20条记录未发现该NOOP覆盖，不能证明未来分支不受影响。修复需要改变本单明确冻结的 guard。proposed-guard-fix.patch 仅为未应用建议。
2. 原运行未保存结束后的基础快照 hidden/cache 摘要，历史未污染状态为 not_recorded。当前重新载入不能补成历史证明。

曾产生的 first-pair-interface-check.json 错将缺失摘要检查写为 true，已拒收但保留。权威更正是 first-pair-interface-check-erratum.json：base_snapshot_unchanged=null，checks_known_passed=true，gate_status=insufficient_evidence，passed=false。review-erratum.md 说明更正，不能引用旧 passed=true 授权续行。

资源：原阶段上限384，reserved=verified=20，unknown=pending=0，integrity=ok；唯一历史 attempt 不变。原历史 R 对照24条/299步未重跑。本单新增 env.step=0、模型前向=0、optimizer/world/offline update=0、新增 attempt=0，SQLite 未写入。guard-on 仍为2/24，22条未执行。

身份：

- runner: da768e30403c68041c01db1d32f9dd248c03c8f82d200f241506bf629b41c71b
- 新测试: c0bbb1932f7688cc9b9e3f76cadbd50022e9903d9670da51cafb7c5f6cbea81b
- guard: 6e73323f60647ea48fe05bd841aaa3e0631bd870eb786ad082a07ce007bb207b
- 原 SQLite: 700b0c9173c8debd5cacb8066328dd2359bb913a128f3018710c8129de5e6223
- checkpoint: bf10d2685a4a3e9da036689f5028b022330e86e922e09c95a7dd0a929df9bb1a
- 最小差异: 27c8a344f5c08450c6bea0a25dbbe9c89c088ae372625e8287a32deb7a1899c1
- 新证据 hashes.json: 22b346ce9449d8b570a2375eb9c0bf3f7cc75e1ae8a0a43d043f51245fa0f9b0

下一步建议另立零环境步执行单：授权修正 guard 保留合法NOOP并审查旧两条能否在新规则下复用。单独修改guard不意味着已有完整性缺口消失，不自动授权22条续行、新预算或重跑。

归档仅包含报告、勘误、测试输出、最小差异和小型索引；不含权重或原始逐步大制品。最终远端提交与回读结果另保存在 final-remote-verification.json，避免自引用哈希。
