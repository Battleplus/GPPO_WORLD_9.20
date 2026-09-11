# WORLD-GPPO_9.11 M-10 legacy artifact Release staging

状态：待新仓库远端可达后上传；本目录不包含 checkpoint 二进制，仅关联旧项目已核验归档与 SHA-256。

待上传到新仓库独立 Release 的类别：

- 9 个策略 checkpoint；
- 对应冻结 world model；
- optimizer/recovery state（字段存在已核验，恢复加载尚未验证）；
- training log、逐 seed/逐 episode 记录和运行时清单。

完整路径、大小、SHA-256、来源提交和协议版本见 `docs/provenance/artifact-index.json`。原始训练 raw ledger 缺失、旧服务器状态未知，不能以 replay 结果补齐。

上传前必须在新 Release 独立回读资产并下载核验摘要；不得覆盖旧项目 Release、不得 force push。
