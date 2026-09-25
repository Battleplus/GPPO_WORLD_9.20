# Stage 2：一次性恢复技术停止归档（2026-09-25）

**决定：暂停；完整矩阵未完成，尚不能判断 GPPO＋世界模型相对强规则的实用增量。**

- 已完整保留 **8637/9600** 个 episode：原有4034＋本次4603。
- 1个 episode 完成7步但未终止，962个尚未开始；未完成共963个。
- 本次恢复63830环境步；A/B/C累计120263步，训练更新0。
- 技术停止：`OperationalError: disk I/O error`。未重试；未计算部分模型胜负。
- 已记录资源在授权额度内；完整矩阵成本验收尚未完成。

## 阅读入口

- [终态报告](final-review/report.md)
- [研究决定](final-review/research-decision.json)
- [完整性核验](final-review/verification.json)
- [预算与资源](final-review/terminal-budget-and-resources.json)
- [完整项唯一键](final-review/completed-keys.json)、[中断项](final-review/incomplete-episode-keys.json)、[未开始项](final-review/not-started-keys.json)
- [当前研究状态快照](CURRENT_RESEARCH_STATE.md)
- [恢复准备材料与代码压缩包](recovery-support.zip)、[压缩包内容/逐文件哈希](support-contents.json)
- [原始本地输入哈希](final-review/input-hashes.json)、[冻结身份核验](final-review/frozen-input-verification.json)
- [复制来源](source-copy-manifest.json)、[远端包文件哈希](archive-hashes.json)

## 归档边界

这是证据、协议与恢复实现归档，不是完整可部署环境或完整原始数据上传。压缩包保留授权、冻结矩阵、准备报告、恢复/监控代码、测试证据、旧故障报告和派生汇总来源；内容以 support-contents.json 为准。

大型逐步日志、原始 SQLite、checkpoint 和完整原生源码仍在本地，路径及 SHA-256 见 input-hashes.json 与冻结清单。所有复制文件保留原字节，历史报告中的相对/绝对本地路径和当时的“未授权”等状态表述不改写；实际批准文件与后续终态优先用于理解时间线。历史 output-hashes.json 指向原始本地路径，远端包请使用 archive-hashes.json 核验。

本次只作 GitHub 归档，没有启动环境、模型推理、训练或恢复尝试。后续实验与再次恢复未获授权。
