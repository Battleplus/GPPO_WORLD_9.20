# 2026-09-24 更新：最新比较与路线

最新本地证据目录：
`E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt\runs\finite-communication-ack-lease-fix-20260920\model-rule-stage1-cli-recovery-run-v1`

关键文件：report.md、analysis.json、run-status.json、runtime-costs.json、completion-audit.json、research-decision.json、final-hashes.json。

原正式数据库仍在上述 runs 根目录的 `ack-known-task-guard-baseline-v1/budget.sqlite3`；最近报告 1411/1625，剩余 214。这是原阶段余额，不是下一阶段授权。

当前文档更新只迁移五阶段路线、研究状态与下一目标，不迁移源码、checkpoint、SQLite 或大型原始制品。不修改实验工作树或旧归档分支。执行 AI 应先核对实际源码、模型、协议和 SQLite 身份。

以下为旧证据索引，路径用于定位历史；其中“当前预算”“下一步”及迁移状态以各历史日期为准，不能当作新的执行授权。

---

# 证据位置与迁移边界

## 当前源码位置

`E:\Z博士\9.2日\WORLD-GPPO_9.11-replan-value-20260919-wt`

用户报告分支：replan-value-event-trigger-20260919；历史基线HEAD：fe94441f510e51d45fb36408068cbcb4a62a06f2。当前有未提交修改，不能仅以HEAD代表运行源码。

最新报告：`docs/finite-communication-emergency-admission-regression-20260920.md`。
本次交接实际读取了该报告，核对了上述准入修复、字段发送时间、预算204以及探针断点；没有重跑实验。

最新报告SHA-256（用户提供，交接归档另核验）：
`7A4C2BC78351E37E04F11E0BD797C013FF448D6C330E3500CCAA50A388A9A0E5`

关键源码用户报告哈希：
- gppo_world/m10_environment.py：D8AA3CA0AC758D84AF676F21A80A9EE6872200A11DF3024178797B33098538E5
- tests/test_m10_contract_v1.py：C132F670626E8B54E56C99714CD8D7E7EF8F620BD464E9FFD1BCA7B2BBE293B8

运行根目录：`runs/finite-communication-ack-lease-fix-20260920`
- budget.sqlite3：原共享预算，必须实际读取，不能从文档新建替代。
- emergency-probe：前一次拒绝记录。
- emergency-admission-regression：本次公告准入成功、探针NOOP记录。

SQLite历史哈希仅代表当时文件快照，不能当作后续运行必须不变的值；拷贝活跃SQLite应使用一致性备份，不只复制主文件忽略WAL。

## 历史报告索引（同源码目录docs下）

- finite-communication-emergency-contract-v1.md
- finite-communication-admission-fix-20260920.md
- finite-communication-execution-diagnosis-20260920.md
- finite-communication-ack-lease-fix-20260920.md
- finite-communication-emergency-contract-acceptance-20260920.md
- finite-communication-emergency-admission-regression-20260920.md

旧基线制品：`E:\Z博士\migration-artifacts\finite-communication-task-closed-loop-baseline-20260920-v2`

旧准入修复制品：`E:\Z博士\migration-artifacts\finite-communication-task-closed-loop-admission-fix-20260920`

## 接手执行时必须核实

1. 读取工作树实际状态、适用AGENTS.md及当前源码哈希，不清理用户文件。
2. 确认最新报告之后是否又有实验，读取预算数据库及运行状态。
3. 在执行指令记录实际Python路径/版本和源码manifest。
4. 不把本仓库文档视作完整模型/源代码包。
5. 跨机器时如缺源码、模型或原预算账本，先明确迁移缺项，不凭历史描述重建后声称同一实验。

本次新仓库上传不包含凭据、训练模型、大型原始ledger或数据库。后续若迁移代码/制品，需要单独列清单并保持当前未提交源码的真实身份。
