# 可复用基线与研究资产归档（2026-09-25）

本包保留实际源码字节、配置、测试、分析/恢复工具、历史协议和已提交模型权重。不是训练或重试授权，也不表示所有计划中的基线已实现。

## 从哪里开始

- [基线状态索引](baseline-index.json)：已实现搜索规则、完整GPPO/world、在线移除诊断；贪心/匈牙利/标准PPO等待补齐状态分别登记。
- [源码和输入清单](source-manifest.json)：每个文件的工作区相对路径、原始SHA-256、归档SHA-256及压缩方式。
- [权重身份](checkpoint-index.json)：每个checkpoint别名均与ledger-commit所记哈希一致；原transaction文件存在时一并保存，缺失时明确标记，不伪造原文件存在。P_train与T_train以及WD分开，未按文件名大小选择。额外seed为历史资产，不是新的多seed收益证据。
- [未纳入的大制品](excluded-large-evidence.json)：完整逐步日志、SQLite、原始训练ledger及未选中间checkpoint未上传。本包不是完整原始数据备份或保证可从任意中断点续训的包。
- [最新研究结果](../../docs/RESEARCH_PROGRESS_20260925.md)及[基线冻结规范](../../docs/BASELINE_COMPARISON_FREEZE_20260925.md)。

## 资产内容

`workspace/migration-artifacts/event-trigger-aware-gppo-fair-replication-20260919-v1/source-snapshot/`：历史原生25-action M10环境、模型、训练模块、测试、配置和工具。它不是后来有限token通信扩展。

`workspace/research-plans/gppo-world-stage2-gate-preparation-20260924/public_controller.py`：正式使用的公开历史搜索规划器与公共记忆/过滤组件。其当前合同最多5000展开节点，不能改名为所有传统算法的代表。

`workspace/research-plans/`：阶段2执行、成本回放、监控与SQLite恢复、在线world移除诊断、统计分析、协议及失败记录。旧恢复目录是已关闭历史attempt，不是新的执行入口；其中proposed/rejected补丁不自动应用。

`workspace/migration-artifacts/.../controller/`及`source-snapshot/tools/`：已有训练/公平复现工具。随包旧authorization和预算只记历史，不授权再次启动。

## 零模型、零环境的文件核验

在本目录运行 `python verify_archive.py`，仅核对文件哈希、压缩内容和Python语法，不导入torch、环境或实验runner。它不加载checkpoint，不执行原测试，不读取SQLite。

历史文件大于1MiB的文本以确定性gzip无损保存，文件名附加`.gz`；源路径和原字节哈希写入manifest。下载后若需恢复原布局，可在新的独立目录按manifest解压，不能覆盖正在使用的工作区。

## 复用限制与依赖

模型正式运行身份是Python3.14.4、torch2.13.0+cpu、CPU intra4/inter1；原pyproject的最低依赖范围不等于这套冻结身份。依赖安装、模型加载和动态验证本轮未执行。历史runner部分包含Windows进程计量和绝对路径，不能宣称跨平台开箱即跑。

原文件中的`E:/Z博士/...`保持原样以保全哈希。将来执行前须在隔离副本重新绑定源码/checkpoint/config路径并冻结新身份；保持公共信息权限、奖励、动作、统计标准和预算边界。没有完整原始日志/账本时不得声称旧恢复脚本已具备全部输入，更不能用新账本清零旧成本。

本次静态归档通过文件/提交身份和语法检查，不重复历史测试或实验；测试源码及已有测试输出按原样归档。模型推理、env.step、训练更新和实验attempt均新增0。

研究结论继续保留：阶段2相对指定搜索规则通过；后续N任务收益保留，但成本实用验收不通过。标准PPO、无world从头训练、偏好响应及world独立贡献仍待公平验证。
