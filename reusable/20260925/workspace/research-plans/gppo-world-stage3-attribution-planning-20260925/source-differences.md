# 实现及身份边界

- 原生 M10 source-snapshot、模型 checkpoint、环境配置和历史制品：只读，不改动。
- base_worker.py、decision_only_inference.py、safe_observer.py、session_budget.py、sqlite_diagnostics.py：从已完成阶段 2 recovery2 包复用；字节比较结果列入 zero-step-validation.json。
- runtime_support.py：复用既有公开控制器、原生导入检查、Windows 进程核算、压缩日志和 reward 核验；调整当前包路径与授权入口。
- process_support.py：抽出既有 Monitor/OwnedWorker/Reservations；worker 入口切换 runtime_worker.py；每阶段制品按进入时基数累计；仍核算全部同时驻留 worker，包括暂停者。保留 single-writer SQLite 与异常信息，不声称历史 I/O 根因修复。
- runtime_worker.py：继承原模型加载、环境生成与 reset；full 为原在线模型路径，reference 为原生 candidate 末端输出零化，candidate 为移除该加性分支；N 不产生伪造 world hidden 摘要。全体冻结 eval/no_grad，权重始末校验。
- no_online_world.py：新增 N 的优化读出和独立原生参考；参考末端输出 hook 不等同于仅将输入 features 置零，因为 candidate 层存在 bias。
- run_diagnostic.py：仅新增本诊断顺序和授权/配额门；复用原已通过的 reward、公开 guard、终止及压缩日志核验，不修改既有控制任务。
- analysis_contract.py、analyze_runtime_necessity.py：新增冻结非劣、相对强规则实用差及双重成本节省判断；拒绝不完整矩阵，不将破坏后变差宣称为独立 world 贡献。
- sample-manifest.json：从完整旧矩阵身份构造全部 4800 个 N 键，无效果筛选；只读取已完成结果中的身份和初态字段以生成清单。

本包新增实现未在真实模型或环境运行；其动态有效性由待授权 A/B 验证。单元测试使用标准库、AST、伪 worker 和临时文件，不加载 checkpoint。新正式 SQLite 尚未创建。
