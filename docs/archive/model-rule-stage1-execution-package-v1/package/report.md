# 阶段一执行包：冻结模型对公开规则

状态：动态执行未授权，正式预算未扩额。

本包固定比较同一原生 25-action M10 前缀上的两条分支：冻结 GPPO+world（保留原 guard）与只使用公开观测的 `public-priority-edf-nearest-v1`（同一 guard，零模型调用）。矩阵为 8 个开发父场景 × 3 个原生外生重复 × 2 个臂，共 48 条新分支，最多 768 环境步。模型选择、规则选择、环境执行、预算登记和分析均要求逐条审计。

规则排序使用公开任务 priority、deadline、UAV-task 欧氏距离、公开 energy、action id；保留 NOOP。rank carrier 仅是共同 guard 接口的排序载体，已归一化但不作校准概率解释。原始 mask 若允许分配动作而必要公开字段不完整或 ready 条件矛盾，必须技术停止；masked unknown/padding 行可保留原生占位值。

当前纯验证：规则及合同测试 48 项通过；父侧分析器与授权器测试 23 项通过，最新合计 52 项通过。测试仅使用替身数据或临时 SQLite，未调用环境、模型、checkpoint/snapshot、正式 SQLite 或新增 attempt。

正式执行仍需 runner 实际入口、selector ledger、运行成本、首对 gate 与分析器逐条复算证据通过集中验收，并由用户明确批准。批准后才允许将正式账本 environment_steps 上限从 1067 调至 1625；历史 857 步保留，新增上限为 768 步，所有更新为 0。
