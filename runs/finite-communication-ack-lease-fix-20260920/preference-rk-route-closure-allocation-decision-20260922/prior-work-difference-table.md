# 既有工作与新协议差异表

| 工作 | 原生环境/动作 | 已回答 | 质量门槛或失败 | 新协议的实质差异 |
|---|---|---|---|---|
| `preference-branch-evaluation-v1` | 历史 M10 采集包；R/K 分支 | 未折扣任务—累计能耗二维差异可描述 | 旧包没有逐步 vector reward，原训练效用不能可靠重建 | 使用已验证记录器合同，但研究真实分配动作而非 R/K 标签复算 |
| `preference-vector-label-pilot-v1` | 原采集 M10；25 actions | 逐步二维标签可复算，记录器开关不改行为，历史轨迹复现 | 4 个 prefix 没有固定偏好翻转；不是收益评价 | 不把 pilot 结果当候选后果证据；只复用记录合同与审计规则 |
| `preference-vector-label-train-v1` | 原采集 M10；25 actions | 154 prefix、924 分支、17 个平均偏好翻转 | 只是 hindsight/描述性 opportunity，未做预测器或 validation | 新协议的分支变量是首个合法非 NOOP 分配 action，而不是 R/K |
| `preference-repeat-stability-audit-v1` | 同上，train 三重复 | 可量化重复稳定性与 hindsight 乐观增量 | 仅 6/17 平均翻转 prefix 三次均翻转；后两档相对 R CI 跨零 | 新协议必须把“真实后果差异”和“跨重复稳定”拆开验收 |
| J-02A/J-02B | T-05 原生 17-action | 数据合同、latent 工程接入和 branch regret 等工程/模型检查 | J-02B 任务质量门槛失败；不是 M10 25-action 分配后果实验 | 新协议禁止混用 17-action，直接在 M10 25-action 合同上定义候选门 |
| `legal-candidate-causality-20260921` | 当前有限通信 M10；25 actions | 公开 mask 可复算，I/W1/W2 的字段阻断可解释 | I/W1 公开 pending 行中至少两个合法候选为 0；W2 无公开 pending | 不能用有限通信无候选结果证明排序无效；先做身份一致的候选资格预检 |
| `frozen-latent-adapter-offline-diagnostic-20260921` | D-02/T-05；17 actions | global residual 可改变部分合法动作排序 | 不是 candidate-wise consequence residual，不能推出真实后果改善 | 新协议若使用 world model，必须是候选后果接口并单独与公开基线比较 |
| `research-mainline-protocol-closure-20260921` | 研究主线汇总 | 关闭重复路线并提出候选后果方向 | 原推荐三臂 candidate residual 方案后来被身份审计撤回 | 本协议撤回 candidate-wise latent 干预，改为真实分支后果测量与预测门控 |

## 身份边界

新协议唯一的执行合同是历史 M10 采集对应的 25-action 原生环境：4 UAV、6 task capacity、`4 * 6 + 1 = 25`，NOOP index 24，decision interval 1 秒，horizon 18 秒。D-02/J-02 的 T-05 17-action（NOOP index 16）只作历史差异对照，不得作为本协议输入、模型或结果。

## 为什么不是换报表

新问题改变了干预对象、因果配对和验收顺序：只在同一公开前缀的合法 mask 内改变首个分配动作，随后使用同一冻结策略和 hidden 推进，直接保存逐步任务/能耗/deadline 标签；先问后果是否不同，再问是否稳定，再问公开信息和 world model 是否可预测。增加 seed、改模型名、只换 hindsight 报表都不能替代这些差异。

