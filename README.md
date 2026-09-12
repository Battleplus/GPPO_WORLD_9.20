# WORLD-GPPO_9.11

本仓库是后续研究的唯一归档入口（目标远端：`Battleplus/WORLD-GPPO_9.11`）。内容从旧项目 `GPPO-WORLD-9.2` 的提交 `05702ae861e460e667acf8f0184e9a3709ef6fd4` 迁移而来；旧仓库、旧 Release、失败记录和负结果不被覆盖，且不跨协议混合统计。

当前阶段主题是“无人机弱通信任务调度中的世界模型优化”：首轮保持周期决策，新增候选 UAV–Task 后果预测设计与实现，但尚未启动新训练。请先阅读 [迁移来源](docs/transition/legacy-provenance.md)、[任务合同](docs/contracts/world-gppo-9.11-task-contract.md)、[后果模型设计](docs/world-model/action-consequence-design.md) 和 [实验计划](docs/plans/EXPERIMENT_PLAN.md)。

候选后果标签校验入口为 `gppo_world/consequence_data.py`，服务器训练入口为 `tools/train_m10_consequence_model.py`；当前仅完成接口与测试，未执行训练。

新仓库远端在 2026-09-11 探测时连接被重置；大型 checkpoint、optimizer/recovery state、训练日志和数据仅在 `release-staging/` 中登记，待远端可达后上传独立 Release 并回读核验。

本仓库用于把“事件感知世界模型”迁移到 GPPO 动态任务分配系统，并保存从设计、数据、模型、联调到实验验收的完整证据。

一句话概括最终目标：

> 让世界模型学习“在当前 belief 图中实际执行某个动作后，系统可能怎样变化”，再把经过验证的预测 latent 提供给 GPPO；GPPO 仍然是唯一动作选择器，真实 action mask 和执行安全链始终拥有最终权威。

本项目以 [`Battleplus/GPPO-8.29@2a9bb9f`](https://github.com/Battleplus/GPPO-8.29/commit/2a9bb9f87b9d543df144f4d108ba970c924151f9) 为固定设计基线，参考 EAWM 的自动事件、Event Predictor 和 GES 思想，但针对 UAV–Region–Target 异构图重新实现，不直接照搬 Atari 图像模型。

> 2026-09-05 重新分析：T-00～T-05 已按各自已执行协议验收，四组 × 三 seeds 消融已封存；原始目标要求的 GPPO-History 对照仍未完成，不能等同于原始完整验收全部满足。稳定策略增益尚未建立。阅读 [重新分析](docs/09-project-reassessment-20260905.md)、[修订规划](docs/10-revised-experiment-plan-20260905.md) 与 [原始要求覆盖表](nodes/requirements-status.json)。历史结果见 [T-05 报告](nodes/T-05/evidence/final-report.md)。

当前执行入口：[9 月中旬分阶段交付计划](nodes/M-09/README.md)。按会议要求优先基础功能、延迟验证和汇报交付，再推进世界模型研究。

## 2026-09-06 最新交付

M-09 S5 已形成候选交付包并完成服务器独立复现、PPTX/PDF 渲染检查与归档校验：[S5 交付目录](nodes/M-09/S5/README.md)。默认 A 为纯 GPPO + 可关闭只读 Shadow；历史 B 为 EAWM-GPPO adapter 消费复现。S3 明确跳过且未训练；实际彩排/评审尚未举行，9/13 冻结与 9/15 汇报仍待确认。GitHub Release：`m09-s5-delivery-freeze-v1-20260906`（候选）。

2026-09-05 已完成 R-02、J-02A 和服务器 J-02B 四配置三 seed 实验。J-02B 开发门槛失败，完整训练文件已发布至 [GitHub Release](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/j02b-server-archive-20260905)。原始 GPPO-History 对照和稳定策略增益仍未完成；原始 R-02 大文件尚待独立 Release。

## 9 月中旬汇报准备度

截至 2026-09-06，面向会议交付的累计准备度评估为 **40%～50%（当前加权检查点 46/100）**；M-09 的正式阶段验收为 **1/6**。两个数字回答不同问题：46/100 计入 T-05、R-02、J-02A/J-02B 的既有框架、服务器训练与归档；1/6 只统计按本轮会议标准重新验收通过的阶段。

| 已形成的基础 | 为什么计入进度 | 为什么还不能算会议交付完成 |
|---|---|---|
| GPPO 基础任务分配、合法动作与执行安全链 | 已有可运行代码、checkpoint 和测试 | 尚未按本轮六场景 60 条轨迹重新验收 |
| T-05 世界模型接入和四组服务器训练 | 证明训练链、Shadow、adapter 和回退可运行 | 没有证明稳定策略增益，也不是会议所述五类型版本 |
| J-02A/J-02B 数据合同和服务器训练 | 证明防未来泄漏、合法分支和完整训练归档 | J-02B 开发门槛失败，不能直接进入后续策略扩展 |
| S0 版本与术语核查 | 已锁定可复用三类型基线并明确缺口 | 五类型实现、四类扰动完整闭环仍未找到/未验收 |

当前真正阻塞中旬演示的是：会议所述五类型源码与模型缺失；紧急任务、能量/换电和复合扰动没有完整可控实现；“慢 3～4 倍”缺少同硬件、同负载的公平复测；汇报稿和演示包尚未冻结。完整的计算方法、证据链、风险与后续顺序见 [会议准备度说明](nodes/M-09/readiness.md)，机器可读评估见 [readiness.json](nodes/M-09/readiness.json)。

## 为什么需要世界模型

新增独立路线：[J-01 Graph-JEPA 实验](nodes/J-01/README.md)已完成三组 × 三 seed 离线训练，接入门槛未通过，未启动 JEPA-GPPO 训练。它与原 EAWM 并列，不替换 GPPO。[完整结果](nodes/J-01/evidence/final-report.md) / [下一版草案](nodes/J-02/README.md)。

当前 GPPO 能读取实时 belief 图，在 16 条 UAV–Region 候选边和一个 NOOP 中自主选择合法动作，并通过 graph/action version、ACK、lease 和 fencing 保证执行安全。

它的主要局限是：决策以当前图为主，没有一个经过训练的内部模型显式表示“这个动作执行以后可能发生什么”。世界模型要补充的正是这一层能力：

- 从多步可见历史中保留系统变化信息；
- 区分不同 executed action 导致的不同后果；
- 预测下一图/状态差分、reward、cost、continuation 和不确定度；
- 通过自动事件监督，让 latent 更关注关键状态变化；
- 在确认安全、校准和兼容性后，把冻结 latent 提供给 GPPO。

世界模型不是新的动作控制器，也不直接向环境下达命令。

## 最终系统是什么

```text
历史可见 belief 异构图 G_t
+ decision_time 前已到达的 evidence/message
+ 实际确认执行的动作 a_t
+ 时间、版本和有效性 mask
                 │
                 ▼
       动作条件异构图世界模型
  Graph encoder + action encoder + dynamics
                 │
        temporal latent [h_t, z_t]
      ┌──────────┼──────────┬──────────┬──────────┐
      ▼          ▼          ▼          ▼          ▼
 下一图/差分  自动事件   reward/cost continuation uncertainty
                 │
                 ▼
         可关闭的 frozen latent adapter
                 │
                 ▼
               GPPO
                 │
                 ▼
  真实 action mask + version + ACK/lease/fencing
```

### 世界模型输入

- 决策时刻可见的 UAV、Region、Target 异构图；
- `decision_time` 前已经收到的证据和消息；
- 实际执行并确认的 UAV–Region 动作或 NOOP；
- graph/action version、当前 `decision_time` 和有效性/padding mask；动作后才知道的下一决策时间差禁止作为输入。

### 世界模型输出

- 时序 latent `[h_t, z_t]`；
- 下一图或节点/边状态差分；
- ordinal、nominal、structural、evidence 自动事件概率；
- reward、cost vector 和 continuation 预测；
- epistemic/aleatoric uncertainty；
- model/input version 和 `valid` 状态。

### GPPO 如何使用输出

T-05 主实验只把冻结 latent 经过可选 adapter 加入 actor/critic。自动事件 logits 默认不直接进入 actor。模型异常、超时、版本不一致或不确定度过高时，系统使用 zero context，恢复原始 no-WM GPPO 路径。

## GPPO 与世界模型的职责边界

| 能力 | GPPO/现有安全链 | 世界模型 |
|---|---|---|
| 在合法候选动作中选择具体动作 | 唯一负责 | 禁止直接选择或提交 |
| 定义 16 条候选边和 NOOP | 权威合同 | 只读取 |
| 维护真实 action mask | 权威状态 | 禁止写入 |
| 维护 belief 和 graph/action version | 权威状态 | 禁止写入 |
| ACK、lease、fencing 和 stale 拦截 | 权威执行链 | 禁止绕过 |
| 预测动作后的状态、事件和成本 | 不负责 | 负责 |
| 输出预测 latent 和不确定度 | 不负责 | 负责 |
| 模型故障时继续运行 | 原 GPPO 路径 | 必须允许无损关闭 |

## T-00～T-06 每一步的意义

| 节点 | 要解决的问题 | 主要实现/产物 | 通过以后意味着什么 | 当前状态 |
|---|---|---|---|---|
| [T-00](nodes/T-00/README.md) | 模型到底能读取什么，怎样保证不偷看未来 | 因果 Transition schema、字段注册表、future/truth denylist、统一 recorder、基线测试 | 输入输出和安全边界已经冻结，可以可信采集数据 | **passed** |
| [T-01](nodes/T-01/README.md) | 世界模型用什么真实轨迹训练，怎样防止 train/test 泄漏 | random legal、greedy、GPPO 三类轨迹；完整 episode/tape/seed split；数据/策略 checkpoint 和 SHA-256 | 已有可复现、可审计的数据，可以开始训练世界模型 | **passed** |
| [T-02](nodes/T-02/README.md) | 不考虑事件监督时，模型能否学到“动作导致的后果” | Graph encoder、action encoder、temporal dynamics、next-state/reward/cost/continuation/uncertainty heads | 得到第一个真实 Graph-WM checkpoint，并用合法动作反事实证明模型使用动作 | **passed** |
| [T-03](nodes/T-03/README.md) | 自动事件和 GES 是否让 latent 更关注关键变化 | 自动事件生成器、按模态 Event Heads、hard/smooth GES、WM/EA-noGES/EAWM 消融 | 得到事件感知世界模型，并能分离 Event Head 与 GES 的贡献 | **passed** |
| [T-04](nodes/T-04/README.md) | 模型在线运行是否可信、校准、及时且不污染系统 | 只读 Shadow runtime、ID/OOD 校准、risk-coverage、延迟和安全回退报告 | 世界模型可以在线观察和预测，但仍不影响正式动作 | **passed** |
| [T-05](nodes/T-05/README.md) | 世界模型 latent 对 GPPO 是否有真实增量价值 | frozen latent adapter、zero-context fallback、旧 checkpoint 兼容、四组公平实验 | 完成世界模型基础迁移，并以负结果严谨判断当前版本未带来稳定增益 | **passed** |
| [T-06](nodes/T-06/README.md) | 短期 imagined rollout 是否有额外价值 | GPPO 合法候选动作的 1～3 步 rollout、不确定度截断、真实环境验证 | 可选的预测规划扩展；失败时保留 T-05，不影响基础迁移 | planned / optional |

### T-00：冻结合同，而不是先写网络

意义是防止后面训练出一个“指标很好但偷看未来”的模型。本节点把在线可见字段、未来 target、proposal 与 executed action、版本语义和统一记录方式明确分开。

已验证：原 GPPO 核心/训练/并发/事件桥接 50 项测试通过；本仓库因果合同、动作合法性、版本和 recorder 测试通过。证据见 [T-00 节点](nodes/T-00/README.md)。

### T-01：建立可训练、不可泄漏的数据

意义是让世界模型同时看到不同策略和不同压力场景，而不是只学习单一 GPPO 的窄分布。数据按完整 `scenario/tape/seed` 分组切分，同一个事件带绝不能跨 train/validation/test。

当前已封存：

- 126 个 episode、502 条 transition；
- random legal、greedy、GPPO 三类行为策略；
- normal、single、sequential、overlap、burst、long gap、weak communication；
- 17 个动作全部覆盖；
- split overlap 为 0；
- 在线 truth-only 字段为 0。

数据和采集用 checkpoint 位于 [T-01 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t01-data-v0.1.0)。其中 512-step GPPO 只用于扩大数据覆盖，不是历史 50k 正式模型，也不是世界模型。

### T-02：先证明基础世界模型真的理解动作

第一版只训练动作条件 Graph-WM，不加入 Event Predictor/GES，也不修改 GPPO。核心验证不是单纯看 loss，而是进行：

- 正确 action；
- action shuffle；
- no-action；
- last-value/frequency；
- summary-vector GRU；
- Graph World Model。

当前 T-02 已通过并封存于 [Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t02-base-wm-v0.1.0)。
合法动作反事实的 state/reward/cost 均显著退化，checkpoint 可独立恢复；同时透明保留了 Flat-GRU
状态误差更低、no-action state CI 跨 0 的负结果。T-02 只证明基础模型链路成立，不声称下游 GPPO 增益。

### T-03：迁移 EAWM 的事件感知思想

自动事件来自相邻可见图变化，而不是人工指定“发生 UAV_DAMAGE 就选某动作”。事件分为：

- ordinal：连续量 DOWN/SAME/UP；
- nominal：类别 SAME/CHANGED；
- structural：节点、关系、候选边和合法 support 变化；
- evidence：新证据、重复、冲突、确认和过期。

Event Predictor 是世界模型辅助头，GES 用于调节高事件密度边界对训练的影响。它们服务于 latent 表示学习，不替代 GPPO，也不等同于人类偏好学习。

当前 T-03 已完成三 seed 固定预算消融并封存于 [T-03 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t03-eawm-v0.1.0)。
EAWM-hard 的 macro-F1 为 `0.4668±0.0050`、macro-AUPRC 为 `0.4320±0.0136`；基础 state/reward/cost
预测的最大逐 seed 相对退化均低于冻结的 5% 上限。TTL 缺失、失败配置和测试集已查看的协议限制均已显式保留，
详见 [T-03 节点证据](nodes/T-03/README.md)。

### T-04：先 Shadow，再允许策略读取

世界模型在线维护 latent、记录预测和实际结果，但不修改动作。只有以下门禁全部通过才进入 T-05：

- belief/action mask/version 写入次数为 0；
- stale hidden state 不提交；
- 异常、超时、OOD 高风险能够回退；
- ECE、Brier、risk-coverage 达到冻结标准；
- P50/P95/P99 延迟满足预算。

T-04 已通过并封存于 [T-04 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t04-shadow-v0.1.0)。
真实基线环境的 belief、action mask、graph/action versions 及动作提交接口均保持零写入；完整 observe 的 P95/P99
为 `6.99/8.05 ms`。合成 OOD 的范围和 8.59% ID 假阳性率已透明记录，不作生产级 OOD 泛化声明。

### T-05：冻结 latent 接入 GPPO

保持动作空间、mask、奖励、PPO 预算、场景和 seed 一致，至少比较：

1. GPPO；
2. WM-GPPO；
3. EA-noGES-GPPO；
4. EAWM-GPPO。

必要时增加 GPPO-History，以排除“只是多看历史”的解释。只有真实 held-out 环境、多个 seed 和安全指标共同支持，才能声称世界模型对 GPPO 有增益。

T-05 已完成：冻结 `[h,z]` residual adapter、post-action Shadow hook、逐 transition versioned latent sidecar、旧 checkpoint 无损回退，以及四组 × 3 seeds × 50k 的正式 GPU 训练。固定 50k checkpoints 全部在同一有序 100-tape Test bank 上评估；12/12 runs、24 checkpoints、12/12 evaluations 和 1,200 traces 均完成哈希复核。真实环境/belief/mask/version/动作提交写入为 0，世界模型冻结且延迟 Gate 全部通过。结果没有证明稳定性能增益，详见 [最终报告](nodes/T-05/evidence/final-report.md)。

### T-06：可选想象规划

T-06 不属于当前基础迁移完成条件。它只允许对 GPPO 提出的合法候选动作做 1～3 步短期 rollout，并根据不确定度截断。若只改善 predicted return、没有改善真实 held-out 结果，本节点应标记失败并回退 T-05。

## 最终会交付什么

完成 T-00～T-05 后，仓库应当包含：

1. **可运行的世界模型代码**：图编码器、动作编码器、时序 dynamics、预测 heads、loss、训练和评估入口；
2. **真实 checkpoint**：基础 Graph-WM、EA-noGES、EAWM+GES，附带不可变下载链接和 SHA-256；
3. **自动事件系统**：事件生成、模态注册、Event Predictor、GES 和相应测试；
4. **只读 Shadow runtime**：校准、OOD、延迟、版本一致性和故障回退；
5. **GPPO latent adapter**：可配置关闭、zero-context parity、旧 GPPO checkpoint 兼容；
6. **公平实验报告**：逐 seed 结果、置信区间、失败 run、业务指标、安全指标和资源成本；
7. **完整证据链**：代码 commit、配置、数据 manifest、split hash、seed、checkpoint、日志、指标和节点结论。

## 什么才算“迁移完成”

以下条件必须全部满足：

- 动作条件异构图世界模型与严格数据切分真实实现；
- state/event/reward-cost/continuation/calibration 均有独立评估；
- GPPO 可以读取冻结 latent，并能无损关闭世界模型；
- belief、action mask、version、ACK/lease/fencing 不受污染；
- 至少完成 GPPO、WM-GPPO、EA-noGES-GPPO、EAWM-GPPO 四组公平消融；
- 每项结论都能追溯到真实 checkpoint、配置、seed、日志和源码提交。

计划文件、未保存的本地模型、单个最佳 seed 或模型内部 predicted return 都不能作为“完成”的依据。

## 明确不做什么

- 不让世界模型取代 GPPO；
- 不让预测图覆盖真实 belief；
- 不绕过 action mask、版本、ACK、lease 或 fencing；
- 不把 event logits 默认直接输入 actor；
- 不声称能准确预测不可观测的外生随机事件；
- 不在单步预测和校准未通过前开展长时域规划；
- 不把自动事件监督包装成人类偏好学习；
- 不在没有实际测量时声称降低延迟或计算量。

## 仓库结构

```text
GPPO-WORLD-9.2/
├── gppo_world/            # 合同、数据、世界模型与后续运行模块
├── tools/                 # 采集、审计、训练和评估入口
├── tests/                 # 因果性、模型、安全和兼容性测试
├── docs/                  # 总体设计、架构、执行与验收规范
├── nodes/T-00...T-06/    # 每个节点的状态、Gate 和真实证据
├── nodes/status.json      # 机器可读的权威节点状态
└── README.md              # 项目总入口
```

## 文档阅读顺序

1. [范围、分工与安全边界](docs/00-scope-and-boundaries.md)
2. [架构与数据合同](docs/01-architecture-and-contracts.md)
3. [T-00～T-06 执行规划](docs/02-execution-plan.md)
4. [节点、checkpoint 与证据保存规范](docs/03-checkpoint-and-evidence-policy.md)
5. [实验矩阵与验收定义](docs/04-experiment-and-acceptance.md)
6. [当前任务进度](docs/05-current-progress.md)
7. [T-05 服务器训练 AI 接力说明](docs/06-server-ai-handoff.md)
8. [T-05 正式服务器 Campaign 实时接力存档](docs/07-t05-live-server-campaign-handoff.md)
9. [节点总索引](nodes/README.md)
10. [T-05 封存后的诊断与下一步](docs/08-post-t05-diagnostics-and-next-plan.md)

节点的权威当前状态以 [`nodes/status.json`](nodes/status.json) 为准。README 负责解释路线，节点证据负责证明结果；文档中的计划不能替代真实实验。

## 快速验证

在安装 PyTorch、NumPy 和 pytest 的 Python 环境中运行：

```powershell
python -m pytest -q
```

T-01 数据、采集用 GPPO checkpoint 和原始 manifest：

- [Release 页面](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t01-data-v0.1.0)
- [T-01 节点证据](nodes/T-01/README.md)

T-02 基础世界模型 checkpoint、训练日志与指标：

- [T-02 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t02-base-wm-v0.1.0)
- [T-02 节点证据](nodes/T-02/README.md)

T-03 事件感知模型、逐 seed 消融、失败 run 与指标：

- [T-03 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t03-eawm-v0.1.0)
- [T-03 节点证据](nodes/T-03/README.md)

T-04 Shadow、校准、真实基线零写入审计与回退记录：

- [T-04 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t04-shadow-v0.1.0)
- [T-04 节点证据](nodes/T-04/README.md)

## 方法来源

- [GPPO-8.29](https://github.com/Battleplus/GPPO-8.29)
- [世界模型任务目标与改进目标](https://github.com/Battleplus/GPPO-8.29/blob/2a9bb9f87b9d543df144f4d108ba970c924151f9/docs/world-model/current/%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E4%BB%BB%E5%8A%A1%E7%9B%AE%E6%A0%87%E4%B8%8E%E6%94%B9%E8%BF%9B%E7%9B%AE%E6%A0%87.md)
- [EAWM 官方实现](https://github.com/MarquisDarwin/EAWM)

## 当前能力声明

T-00～T-05 已有封存证据，世界模型基础迁移完成。T-05 正式消融、checkpoint/日志/指标、兼容与安全 Gate 均已通过；仓库保留失败尝试和负结果。当前不支持“世界模型稳定提升 GPPO”的声明，T-06 仍是未开始的可选研究项。
