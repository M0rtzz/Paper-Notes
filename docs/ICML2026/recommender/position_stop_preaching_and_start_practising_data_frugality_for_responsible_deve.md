---
title: >-
  [论文解读] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI
description: >-
  [ICML 2026][推荐系统][数据节俭] 这篇 position paper 指出 ML 社区在"数据节俭"(data frugality)上长期"只说不做"——大家口头承认 coreset 能省能耗，却几乎没人真去汇报能耗和碳排放…
tags:
  - "ICML 2026"
  - "推荐系统"
  - "数据节俭"
  - "coreset"
  - "碳排放"
  - "责任性 AI"
  - "子集选择"
---

# Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI

**会议**: ICML 2026  
**arXiv**: [2602.19789](https://arxiv.org/abs/2602.19789)  
**代码**: https://github.com/saintslab/data-frugality  
**领域**: AI 安全 / 可持续 AI / 数据高效  
**关键词**: 数据节俭, coreset, 碳排放, 责任性 AI, 子集选择

## 一句话总结
这篇 position paper 指出 ML 社区在"数据节俭"(data frugality)上长期"只说不做"——大家口头承认 coreset 能省能耗，却几乎没人真去汇报能耗和碳排放，并以 ImageNet-1K 为例算出下游训练 + 存储约 5.82 GWh / 2589 tCO2e 的保守下限，呼吁把数据节俭从口号变成可度量、可执行、可奖励的工程实践。

## 研究背景与动机

**领域现状**：当前主流 AI 训练沿着 Kaplan 与 Hoffmann 的 scaling law 走，把"更大数据集"等同于"更好模型"，benchmark 和 leaderboard 也奖励在更大语料上跑实验。与此同时，存在一条平行的支流呼吁"scaling down"——例如 coreset selection、dataset condensation 等数据子集方法。

**现有痛点**：作者对 10 篇有代表性的 coreset 论文做了系统调查，发现 8/10 篇把"computational efficiency"写进动机、3/10 篇提到"energy efficiency"，但**只有 1/10 篇真的报告了能耗节省**，6/10 篇报告了时间节省。也就是说，数据节俭的"宣传"远远跑在"实践"前面。

**核心矛盾**：碳排放并不能从计算量或数据规模线性推导——25% 的数据裁剪在不同架构上得到 24%–40% 的时间收益和 24%–33% 的能耗收益，不是简单成比例。这意味着不把能耗、碳排实测出来，所有"省能耗"的口号都只是修辞，无法用于责任性 AI 的真正决策。同时，碳排估计还高度依赖电网碳强度（全球极差超过 60 倍），同一份训练在 Turkmenistan 与 Lesotho 之间的碳排会差几十倍。

**本文目标**：(i) 把 ImageNet-1K 这种"国民级"数据集的下游训练 + 存储碳成本定量算出来，给出可引用的下限；(ii) 用实证证明 coreset 既不损精度又能省电；(iii) 把诉求落地到面向 People / Platforms / Policies 三层的具体建议。

**切入角度**：作者把 ML 实验室自己（SAINTS Lab）的 Carbontracker 测量栈、OpenReview ICLR 元数据爬取、Hugging Face 下载量等可获得证据拼起来，刻意做"保守下限"——他们想说明哪怕用最少的假设，数据节俭的"机会成本"也已经触目惊心。

**核心 idea**：把数据节俭从"价值观—行动鸿沟"(value-action gap) 的活动主义问题，转译成 ML 社区可以直接对齐的报告与评审规范——若一个方法以能耗为卖点，就**必须**测能耗。

## 方法详解

这是一篇 position paper，"方法"由三块互相支撑的论证组成：碳成本核算、coreset 实证、可执行的行动框架。

### 整体框架

论文沿着 lifecycle 视角组织：数据生命周期（采集、清洗、存储、分发）与模型生命周期（训练、选择、部署）并行存在，coreset / subset selection 同时作用于这两条生命周期的多个阶段。在此基础上：

1. **量化基线**：用 ICLR 2017–2022 的 OpenReview 元数据估计有多少篇论文真的"在 ImageNet-1K 上从随机初始化训练模型"，再用 dimensions.ai 关键词索引把这个比例放大到全 ML 文献，并向 2023–2025 线性外推。
2. **实证收益**：在 A100 上跑 ResNet-34 / ResNet-50 / Swin-T，用 Carbontracker 实测 full vs. 25% pruned 的能耗 / 时间；引用 Dyn-Unc (He et al., 2024) 与 InfoMax (Tan et al., 2025) 的 SOTA 子集选择曲线说明精度不掉。
3. **公平性副作用**：在 Coloured-MNIST (99% 多数色) 上对比 random / reweighted / balanced 三种 sampling，证明 coreset 还能顺手做去偏。
4. **行动框架**：把上面三类证据归并为面向 People / Platforms / Policies 的呼吁。

### 关键设计

1. **ImageNet-1K 下游碳成本的可复现下限估计**:

    - 功能：把"数据集本身的环境成本"从抽象口号变成可引用、可质疑的数字 — 5.82 GWh 能耗、2589 tCO2e 碳排（全球均值碳强度）。
    - 核心思路：训练成本 = 估计训练次数 $N \approx 46{,}179 \pm 1{,}154$ × ResNet-50 单 epoch 能耗 $\approx 0.394$ kWh × 300 epoch，得 $5.46 \pm 0.14$ GWh；存储成本 = $N \times 130$ GB × 60 kWh/TB/yr $\approx 360 \pm 9$ MWh。$N$ 由 ICLR OpenReview 中"从零训 ImageNet"的论文占比线性外推+dimensions.ai 关键词索引得到，错误率 2.5% 来自 LLM 标注校验。
    - 设计动机：作者刻意只用公开元数据，避免依赖任何厂商内部数据；同时坦言这是"下限"——Hugging Face Hub 上 214 个 ImageNet-1K 派生数据集合计下载量超过 250 万，比论文计数大 55 倍，说明真实成本远高于此。这个保守姿态让结论更难被反驳。

2. **Coreset 收益的双轴实测：精度曲线 × 能耗表**:

    - 功能：用一张精度-pruning ratio 曲线和一张能耗表共同回答"少用数据到底能省多少、掉多少点"。
    - 核心思路：精度侧借用 Dyn-Unc 在 Swin-T 上、InfoMax 在 ResNet-34 上的 SOTA 曲线，说明 ImageNet-1K 可裁掉 25%–35% 而 Top-1 精度不降；能耗侧自己跑 ResNet-34/50 + Swin-T 各 10 epoch×3 次（同 A100 / DGX 配置），用 Carbontracker 同时记录 CPU 与 GPU 能耗。结果是 25% pruning 给出 ResNet-34: 32% 时间 + 29% 能耗、ResNet-50: 40% + 33%、Swin-T: 24% + 24% 的节省。
    - 设计动机：作者特别强调 25% 数据 $\neq$ 25% 能耗，反例式地戳破"data size 当能耗 proxy"的常见假设；并诚实地把 coreset 构造的一次性成本算成"3–4 次训练即可摊销"，避免田忌赛马式比较。

3. **CARAML 之前：面向 People / Platforms / Policies 三层的可操作动议**:

    - 功能：把"道德呼吁"翻译成可以写进 conference call、leaderboard 规则、funding 条款的具体动作，避免论文沦为又一篇高调宣言。
    - 核心思路：(People) 用"Data-Pareto"——精度/数据量同时画——替代单一精度；要求"motivate 什么就 measure 什么"。(Platforms) 借鉴 CVPR 2026 强制 compute reporting 表与 BabyLM / E2MIP 这类 data-efficient 挑战赛，把节俭嵌入投稿与排行榜激励。(Policies) 标准化能耗 / 碳排报告、推动共享数据中心（如瑞典 Berzelius）以减少冗余本地副本、提出"data sunset laws"——大数据使用须像生物医学数据一样审批并设废止期。
    - 设计动机：作者认为价值-行动鸿沟在气候行动史上反复出现，根因是"想比做容易，且外部约束限制行动"；只有把节俭从"个人美德"升级到"institutional default"，才能跨越鸿沟。

### 损失函数 / 训练策略

这是一篇 position paper，无主模型训练 loss。能耗测量栈 = Carbontracker + CodeCarbon，覆盖训练；存储采用 60 kWh/TB/yr 的能耗强度系数（Selvan, 2025）。

## 实验关键数据

### 主实验：ImageNet-1K 下游环境成本估计

| 维度 | 估计值 | 等价碳排（445 g/kWh） | 等价人均年碳足迹 |
|------|--------|----------------------|------------------|
| 训练（46.2k 次 × 300 ep × 0.394 kWh） | 5.46 ± 0.14 GWh | 2429 ± 61 tCO2e | ~514 ± 13 人 |
| 存储（46.2k 副本 × 130 GB × 60 kWh/TB/yr） | 360 ± 9 MWh | 160 ± 4 tCO2e | ~34 ± 1 人 |
| 合计 | 5.82 ± 0.15 GWh | 2589 ± 65 tCO2e | ~548 人 |
| 同样能耗 @ Turkmenistan (1310 g/kWh) | — | 7624 ± 191 tCO2e | — |
| 同样能耗 @ Lesotho (21 g/kWh) | — | 122 ± 3 tCO2e | — |

### 消融：25% 数据裁剪对训练时间 / 能耗的影响（A100，单卡）

| 模型 | 参数 | min/epoch (full → 25%) | 时间节省 | kWh/epoch (full → 25%) | 能耗节省 |
|------|------|------------------------|----------|------------------------|----------|
| ResNet-34 | 21.8M | 35.2 → 23.8 | 32% | 0.2798 → 0.1989 | 29% |
| ResNet-50 | 25.6M | 40.7 → 24.3 | 40% | 0.3940 → 0.2645 | 33% |
| Swin-T | 28.3M | 58.7 → 44.6 | 24% | 0.7002 → 0.5300 | 24% |

### 关键发现
- **"25% 数据 ≠ 25% 能耗"**：不同架构能耗-数据缩减比从 24% 到 33% 不等，证明数据规模不能当能耗 proxy，必须实测。
- **Dyn-Unc 与 InfoMax 在 ImageNet-1K 上分别可裁 25% / 35% 而精度不降**；按 1.4–1.9 GWh 估算，相当于 621–854 tCO2e 的潜在节省。
- **Coreset 构造一次性成本可在 3–4 次训练后摊销完**，即使是与一次 full 训练同量级的"昂贵" coreset 方法（如 Dyn-Unc）也是值得的。
- **Coloured-MNIST 实验**显示，用 balanced / reweighted 采样替代 random sampling，可在 99% 多数色偏置下显著拉高 conflicted accuracy；这意味着数据节俭可以"顺手"做去偏。
- **冰山效应**：Hugging Face Hub 上 214 个 ImageNet-1K 衍生数据集合计下载超 250 万次，比论文级计数的 46k 大 55 倍——真实下游成本远高于本文给出的下限。

## 亮点与洞察
- **"先做下限再呼吁"**：与多数 sustainability 立场文章不同，作者刻意只用公开元数据算保守下限，把"批评者反驳空间"压到最小，论证更稳。
- **"motivate 什么就 measure 什么"**这条单句行动准则可以直接拷进 ML 会议的 review form——是少见的把价值观转译成 reviewer 工具的 position 写法。
- **Data-Pareto 报告范式**（精度 vs. 数据量同图）对其他子领域有迁移性：模型压缩可以画"精度 vs. FLOPs"、RLHF 可以画"对齐分 vs. 标注 token"，都是把单维 leaderboard 升级为多维证据。
- **把"shared dataset infrastructure"列为政策建议**是个被低估的视角——大多数节能讨论只盯训练侧，而本文指出每个实验室各自存一份 ImageNet 的冗余本身就是几十 MWh 量级的浪费。

## 局限与展望
- 估计仅覆盖 ImageNet-1K、电力侧、训练 + 存储，**未含数据采集 / 清洗 / 网络传输**的 embodied 成本与未发表训练；作者也承认这是下限。
- 全部实证都是**图像分类**——生成式模型对长尾、稀有模式更敏感，已有 coreset 难直接迁移；生成模型上的数据节俭仍是开放问题。
- **Rebound effect** 风险：效率提升常被消化为"做更多实验"而非净减能耗；论文承认这点但未给量化方案。
- **政策建议落地性参差**：carbon cap、data sunset law 等高级建议依赖跨国监管，短期内难推动；可执行性最强的还是 People 层的报告范式。

## 相关工作与启发
- **vs Kandpal & Raffel (2025)**: 他们从经济角度论证训练数据的"人工劳动价值"被低估，倡导给数据生产者付费；本文从环境角度论证数据"被当作免费输入"的代价，二者互补。
- **vs Wang et al. (2025) / Goel et al. (2025)**: 他们呼吁 LLM "scale down"；本文把同一主张降到具体子集选择方法 + 实证 + 报告规范层面。
- **vs McCoy et al. (2025)**: 他们提出 capability-per-resource 作为新指标；本文给出 Data-Pareto 报告范式，是该指标的具体落地。
- **vs Strubell et al. (2019) 与 Luccioni et al. (2023)**: 前作算的是单次训练 / 单模型的能耗；本文上升到**数据集级**的下游 aggregate 成本估计。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是新算法，但"先算下限再呼吁"+ Data-Pareto 报告范式 + People/Platforms/Policies 三层框架的组合很少见。
- 实验充分度: ⭐⭐⭐⭐ 数字保守可复现；Carbontracker + 3 个架构 + 公开 SOTA 曲线 + Coloured-MNIST 去偏实验，证据链完整。
- 写作质量: ⭐⭐⭐⭐⭐ 结构干净（preach vs. practise 贯穿全文），论证克制不煽情，limitation 段落主动认错。
- 价值: ⭐⭐⭐⭐ 数据节俭迟早会被纳入 conference 与 funding 规范，本文给出可直接复用的报告模版与行动清单。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)
- [\[ICML 2025\] Position: The Right to AI](../../ICML2025/recommender/the_right_to_ai.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](../../ACL2026/recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)
- [\[AAAI 2026\] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data](../../AAAI2026/recommender/multitab_a_scalable_foundation_for_multitask_learning_on_tabular_data.md)

</div>

<!-- RELATED:END -->
