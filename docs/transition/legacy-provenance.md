# WORLD-GPPO_9.11 历史迁移与来源记录

日期：2026-09-11  
目标仓库：`Battleplus/WORLD-GPPO_9.11`  
本地工作区：`E:\Z博士\9.2日\WORLD-GPPO_9.11`  
分支：`world-model-consequence-v1`

## 来源

| 来源 | 本地来源 | 完整提交/版本 | 用途 |
|---|---|---|---|
| GPPO-WORLD-9.2 lease-fix | `E:\Z博士\9.2日\GPPO-WORLD-9.2-lease-fix` | `05702ae861e460e667acf8f0184e9a3709ef6fd4` | 新仓库代码基线；含 M-10 R3 代码与历史文档 |
| GPPO-WORLD-9.2 clean handoff | `E:\Z博士\9.2日\GPPO-WORLD-9.2` | `117652e264a08f05a580e672e0a8755c6f4d422b` | 旧项目 clean handoff 参考 |
| GPPO-8.29-baseline | `E:\Z博士\9.2日\GPPO-8.29-baseline` | 见其本地 Git 历史 | 仅作为传统方法历史背景；不与 9.2 弱通信结果混合统计 |

旧 lease-fix 工作树存在未提交修改，因此本仓库基于其完整提交建立；未提交文件不被静默纳入来源提交。需要纳入的新实现均在本仓库以新提交记录。

## 远端状态

已按一次有限检查访问新仓库远端，但 GitHub 连接被重置，未能读取新仓库 HEAD、分支或 Release。未继续重试，不把网络失败当成远端内容不存在；本地提交和 Release staging 保留待网络恢复后同步。

## 版本边界

- 旧仓库、旧 Release、失败记录和负结果继续保留。
- 旧完成率、旧协议和旧延迟只属于对应版本；本仓库不跨协议拼接统计。
- `m10-local-cpu-replay-20260910-v1` 是旧项目冻结 replay 的关联 Release，不是 9.11 新实验结果。
- 本阶段没有新训练、没有本机训练、没有旧服务器连接。
