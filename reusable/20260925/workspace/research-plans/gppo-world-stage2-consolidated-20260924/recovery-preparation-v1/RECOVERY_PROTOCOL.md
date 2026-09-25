# 原9600矩阵一次性恢复协议 v1（未授权）

仅恢复gppo-world-stage2-consolidated-20260924的C段。旧失败attempt不删除、不重命名、不追加日志、不修改SQLite。当前准备授权不启动恢复。

1. 保留4033份原汇总与1份基于12步完整记录确定性复算的派生汇总；派生汇总附gzip SHA256、解压行号、原reset身份、原终态payload一致性。retained-episodes.json为供合并用的新制品，原episodes.jsonl保持。
2. 以frozen-matrix.json、completed-with-summary.json、completed-missing-summary.json和not-executed.json分区，前4034后5566无重复无缺口。新恢复入口run_recovery.py只允许后5566项，先R的eval-0672/repeat-1开始。每个剩余episode上限18步：horizon18/dt1正常reset，原native done不变；最多100188新环境步，M/R各2783项。
3. 原worker、模型读出及分析器字节不变；原配置/checkpoint/场景/随机键/偏好/公开信息权限/控制器/奖励/统计标准不变。不得读未授权validation/heldout；不分析部分主效应，不按胜负选样。
4. A/B证据复用，不再运行。重载一次原模型，严格load_state_dict与eval/no_grad和源身份门保持；恢复只在episode边界启动，不恢复中途hidden。
5. 新预算是带旧账本SHA256和全部历史stage余额的追加分段账本；它不清零历史。每次先预留、证据持久化后verified；新旧C额度相加不得超过原阶段上限。唯一额度扩展是模型加载总次数4→5，须本次单独明确批准。B余额不转C，历史失败也不退款。
6. 新段上限：100188步；encode/world/actor各50094；规则50094；reset5566；模型加载1；wall33小时；完整进程CPU141小时；制品38GiB；父调度器和所有驻留worker（含暂停）合计RSS4GiB；训练/GPU0。精确机器可读额度见RECOVERY_BUDGET_REQUEST.json。旧尾部30s wall/600s CPU另保守占用，恢复尾部相同余量包含在本申请上限内。保守累计仍低于原C及总资源上限，除明示的一次新增模型加载。
7. 进度仅由调度器追加到observer-progress.jsonl；只用safe_observer.py显式FILE_SHARE_READ|WRITE|DELETE读取这个独立文件。禁止System.IO.File.ReadLines，禁止观察正式写入中的episodes/steps/SQLite。进度读取或写入失败只记观察错误，不能导致实验重启或正式数据写入失败。不得用观察失败/超时判定实验终止；以实际进程handle和终态为准。
8. 模型隐藏状态、动态环境和规则行为未改；不因恢复增加免费预热、模型前向或episode。进程启动、加载、身份校验、日志、合并、bootstrap和收尾计入新段资源。最终合并所有旧/新逐决策时延及全部资源，不剔除慢样本；报告段别、额外启动和历史未测尾部。
9. 只有4034+5566=9600唯一键全部完成且账本/身份/配对/标签核验通过后，才调用原analyze_formal。保留1600父场景、3重复、10000 bootstrap、seed20260924、原效用/0.01增量及任务护栏/成本门。
10. 源/模型/数据身份变化、错误配对、非法动作、标签缺失、非原生截断、unknown/pending未解、额度或资源门失败即停止，保留新旧段；本次拟申请仅一次恢复，没有再次自动重试、换候选、调参或补样。
11. 完成后只报告原问题的实用增量/成本/去留，不自动阶段3—5、训练、Git发布或外部转发。

## 调用（仅正式批准并生成绑定authorization-approved.json后）

C:\Python314\python.exe -B -X utf8 run_recovery.py --authorization authorization-approved.json

观察命令：C:\Python314\python.exe -B -X utf8 safe_observer.py runs/consolidated-v1/observer-progress.jsonl

本包没有authorization-approved.json。授权模板approved=false；没有新实验SQLite或attempt。恢复动态接口未实际执行，23项测试均为纯数据/桩或临时文件I/O。
