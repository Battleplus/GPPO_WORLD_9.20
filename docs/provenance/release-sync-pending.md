# 9.11 Release 同步状态

截至 2026-09-11，新仓库远端探测返回连接重置，尚未读取远端 HEAD，也未创建或上传 Release。为避免在网络异常时重复操作，本地源码、计划、测试、制品索引和 Release staging 已保留。

网络恢复后的唯一同步路径：

1. 在 `world-model-consequence-v1` 检查提交和凭据扫描；
2. 推送普通分支/提交，不 force push；
3. 创建独立 Release，上传大型 checkpoint、optimizer/recovery state、训练日志和数据；
4. 通过 API/Release 页面回读资产大小和摘要；
5. 独立下载到新目录，按 `docs/provenance/artifact-index.json` 核验 SHA-256；
6. 若任一摘要不一致，停止发布并保留失败证据。
