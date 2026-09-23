# H-006 阶段一执行包 v2

状态：修正后的执行包准备完成，等待用户明确批准动态运行；整体研究目标仍在进行中。没有实验效果结果。

前次归档 dd2a542 的文件回读一致，但运行入口缺陷未被30项测试覆盖，因此旧完成声明和动态执行申请已撤回。详见 erratum-v2.md 与真实 implementation-correction.diff。Luna 首轮修复曾发生服务端并发限制中断，未完成代码被过早验收；本次恢复既有工作并修复具体缺陷，不重复实验。

固定首对 gate 只允许每臂唯一 step1；合法后续分叉不作同输入比较。固定首对两臂各额外一次 actor readout，复用同一次 encode/world 输出；记录 attempted/completed，并由分析器复算特征、logits、概率和输入不变性。入口stub覆盖固定矩阵与故障停止；纯测试不代表真实环境动态接受已通过。

模型干预仅清零17维候选输入末5维事件通道，保留raw world/by_action/hidden。接口来自原生 use_events=False，但零没有训练分布中性保证；结论只限指定敏感性，不能宣称整个世界模型净收益。H005基线也含冻结GPPO/world。

固定8个已观察train开发parent、W1、seed1101、prefix0、各3repeat、两臂共48新分支。原始同一exogenous key配对；动作分叉后的后续状态可不同。保留所有失败及真实消耗，不补样追正。

原奖励/偏好保持：gamma=.99，utility=sum(.99**k*(.4*r_task+.2*r_energy))，k从恢复后首步0开始。先parent内3repeat平均，再8parent等权，10000父级bootstrap seed20260923。完整矩阵和统计协议见 paired-manifest.json/protocol.json。

待批准资源：同一正式SQLite从404上限扩为1067，增加663；历史299保留，新run最多768环境步。最多768 policy encode、768 world候选batch（每batch25行）、770 actor readout，所有更新0。现有105步不能当作新阶段授权。迁移工具只在用户批准记录及包哈希验证后使用原库，不新建替代账本。

本轮 env.step/reset/replay/checkpoint加载/模型前向/全部更新/新增正式attempt均0，正式SQLite SHA和299/404状态未改变；临时测试SQLite不属于实验预算。未分析validation/heldout，未自动推进其他研究阶段。

最终测试输出见 test-output-v2.txt；源码/配置和44个原生模块身份由runner-package-check及native-source-manifest绑定。完整运行命令及pickle读取限制见 COMMANDS.md。只读审查、纯测试不能替代批准后的首对动态门，首对失败即停止。
