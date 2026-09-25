# 独立审查及处理

审查者：`luna_worker`，gpt-5.6-luna/max，当前任务内的 `/root/luna_world_attribution_review`。采用 ablation-planner 技能要求的独立审查；用户的 Luna 偏好优先。未向其他用户任务发送信息，未运行模型、环境或训练。

研究设计审查意见：

1. 阶段 2 正结果不能证明 world 独立贡献。采纳，协议和所有决策输出明确 `world_independent_contribution_proven=false`。
2. 严格训练归因需匹配公开历史、初始化、数据/训练预算与优化器，区分无 world、隔离梯度和共享梯度。采纳为后续尚未授权设计要求，不将当前移除检验伪称完整归因。
3. 置零/移除使性能下降可能由分布外输入或共适应造成。采纳；仅当保留任务收益且有实际成本下降时支持推理简化。
4. 历史 WB/WD/P_train 不能直接组成当前同预算三臂。已用训练身份清单解释，不运行这组混杂比较。
5. CI 未通过不等于已证明无效。采纳，保留明确的“不确定后停止”结论，不自动追加数据。

实现审查发现与修复：

- `budget.snapshot()` 返回带 schema/limits/stages/attempts/history 的顶层对象。原新 runner 把顶层当作阶段记录遍历，会在分析后报错。修复：集中由 `verified_totals()` 读取 `snapshot['stages']`，拒绝 pending/unknown；新增真实旧 recovery2 快照测试，确认 environment_steps=13399、policy_encode=6329、model_loads=1，避免错误汇总为零。
- 全局 wall 起点原在部分导入之后。修复：正式入口开头记录 ENTRY_WALL_STARTED；父进程完整 CPU 从进程 CPU 零点计入 A；A 重复 enter 不会重置初始资源基数。
- 分析的资源检查回调需要与调用签名一致。已一致；bootstrap 每 50 批检查资源。
- 最终大账本快照单独持久化并记录哈希，完整性检查完成后关闭连接；最终状态保留阶段汇总，避免复制大 history 后遗漏内存和写入成本。快照持久化属于 C 成本，之后再次检查资源。

动态数值等价和真实环境接口不属于零步审查可证明内容，仍必须通过授权后 A/B。未声称代码审查等同于任务收益验收。
