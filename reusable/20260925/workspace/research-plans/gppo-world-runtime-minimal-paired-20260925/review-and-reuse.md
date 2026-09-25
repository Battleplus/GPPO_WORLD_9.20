# 独立审查、复用与修订范围

Luna/max 的独立设计审查认可 800父×1固定哈希repeat 作为一次有界开发诊断；512父×3可降低父内重复噪声但花费1536个新episode，旧1600父×3的4800新episode目前没有必须运行的证据。采用 statistical-power 技能的效应界与方差敏感性方法；只用标准库正态近似，未安装依赖，未生成模型或环境数据。

独立审查要求及落实：

- 不以M−R代替N−M方差。报告明确历史参考、假设和未知量，补充效用、任务护栏与接近决策界的检测能力；不承诺精确最低n或联合通过概率。
- 同一parent和同一repeat配对。manifest只依赖身份哈希，N不与M/R三repeat均值错配，不按结果重抽。
- 护栏失败不是明确损失。以任务非劣CI替代旧未获批草案的样本符号门，明确2pp容忍界及其待授权性质；明确损失使用三项家族校正CI。
- 总任务控制开销也必须下降。最终要求每决策CPU和每episode控制CPU均至少降20%，比仅要求总量不增加更严格；不自动通过只优化单次耗时的方案。
- 历史成本需有身份。新增 `(parent,repeat,step)` 与原扁平M成本索引的派生manifest，冻结匹配800键分母，完整62979步覆盖核验。
- 同步更新执行与分析中的1600×3硬编码。新包的矩阵验收、bootstrap、配额、历史对照切片和报告均改为固定800父×1；旧包不修改、旧执行申请不生效。

原 stage3-attribution-planning 包中的身份/原生接口/账本/safe observer 核验结果直接复用。以下文件字节复用：base_worker.py、decision_only_inference.py、safe_observer.py、session_budget.py、sqlite_diagnostics.py、runtime_support.py、runtime_worker.py、no_online_world.py、process_support.py。未重跑原23项测试或阶段2A/B实验；仅对新选择规则、资源总和、决策边界、分析完整性和成本身份做增量标准库测试。待授权A/B仍是新N实现尚缺的实际数值/环境验证，不得声称已通过。

attribution_contract.py只更换attempt身份；run_diagnostic.py只改变固定矩阵与配对成本/分析输入；新analysis_contract.py明确三类研究决策，新增分析脚本不重算/覆盖原阶段2结论。prepare_minimal.py为零推理规划生成器，不能启动实验。

旧SQLite哈希与旧制品来源manifest由上一轮核验复用，本轮只补确认未修改。旧数据库无写连接，新正式SQLite不存在。所有训练、推理、环境step、reset、checkpoint反序列化为0。
