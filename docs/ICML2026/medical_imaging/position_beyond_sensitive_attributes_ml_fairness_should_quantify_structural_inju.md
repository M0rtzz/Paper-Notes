---
title: >-
  [论文解读] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants
description: >-
  [ICML 2026][医学图像][算法公平] 这是一篇 ICML 立场论文：作者主张 ML 公平性研究不能只盯着 race/sex 这类"敏感属性"，而必须把"社会决定因素"（neighborhood、ADI、学校经费、医疗可及性等情境变量）也纳入审计…
tags:
  - "ICML 2026"
  - "医学图像"
  - "算法公平"
  - "结构性不公"
  - "社会决定因素"
  - "敏感属性"
  - "因果公平"
---

# Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants

**会议**: ICML 2026  
**arXiv**: [2508.08337](https://arxiv.org/abs/2508.08337)  
**代码**: 无（立场论文）  
**领域**: AI 安全 / 算法公平 / 立场论文  
**关键词**: 算法公平、结构性不公、社会决定因素、敏感属性、因果公平

## 一句话总结
这是一篇 ICML 立场论文：作者主张 ML 公平性研究不能只盯着 race/sex 这类"敏感属性"，而必须把"社会决定因素"（neighborhood、ADI、学校经费、医疗可及性等情境变量）也纳入审计，并用大学录取理论模型 + 美国人口普查数据 + 乳腺癌筛查半合成实验，证明只围绕敏感属性的缓解策略反而可能制造新的结构性不公。

## 研究背景与动机

**领域现状**：当前 ML 公平性文献几乎把"不公"等价于"沿敏感属性的歧视"——大多数公平度量（Demographic Parity、Equal Opportunity、Conditional Demographic Parity、因果路径效应等）都是先指定 $A$（race/sex/age），再要求预测/决策与 $A$ 解耦或满足某种条件独立。Adult、Folktables、Communities and Crimes 等基准数据集甚至会主动丢掉 address、geolocation 这类情境字段。

**现有痛点**：跨学科文献（政治哲学、社会学、公共卫生）早已指出，真正塑造个体机会与结局的，是 social determinants——neighborhood deprivation、学校经费、空气污染、医院距离、社区资源等情境变量。这些变量在同一族群内部产生异质性（同样是 African American 女性，住在不同 PUMA 的中位年收入从 $38k 跌到 $18.8k），又在不同族群之间产生共同负担（贫困区的非 URM 申请者和 URM 申请者面对同样的社区劣势）。只看敏感属性会把这两种结构性信号同时抹掉。

**核心矛盾**：敏感属性是个体级的、（准）稳态的内禀标识；社会决定因素是情境级的、随空间/时间漂移的结构性变量。现有的个体级因果图（$A \to Y$、$A \to M \to Y$）与去敏感化损失，根本无法承载"邻里-个体"双向影响、社区聚合统计量这种 community-level 结构，把情境当噪声 normalize 掉了。

**本文目标**：把"社会决定因素"作为一个 first-class 的审计对象，回答三个问题：(i) 概念上如何把社会决定因素和敏感属性、敏感属性 proxy 清晰区分？(ii) 现有技术范式为何无法承载？(iii) 只围绕敏感属性的缓解策略到底会带来什么新的结构性不公？

**切入角度**：从一个具体场景出发——历史 redlining 把黑人家庭驱赶进特定社区，使 race、zip code、社区族裔构成长期高度相关；但这三者在公平涵义上完全不同（zip code 是行政标签，不能"改进"；学校经费、空气质量是真正的可干预结构变量）。作者用三个判据（context-level 定义 / social-structural content / exogenous stratification）把它们干净地分类。

**核心 idea**：审计必须先于缓解（auditing must precede mitigation）——在动手"修"模型之前，先要把结构性不公通过社会决定因素显式量化出来，否则盲目地按 race 配额很可能把更弱势的子群（贫困区的非 URM）推向更糟的位置。

## 方法详解

这是一篇立场论文，"方法"由三块构成：(1) 一套定义体系，把社会决定因素从敏感属性 / proxy / 行政标签里切出来；(2) 一个大学录取的闭式理论模型，证明配额式 affirmative action 在何种条件下会反过来伤害贫困区非 URM 申请者；(3) 用 U.S. Census PUMS + OSF HealthCare 真实乳腺癌筛查数据做的半合成实验，验证社会决定因素带来的实证差距。

### 整体框架
输入：跨学科关于结构性不公的概念基础 + ML 公平性现有度量 + 真实人口普查/医疗数据。  
中间：(I) 概念差距分析 → (II) 现有技术范式的三条不适配理由 → (III) 闭式理论模型（quota-based admission）→ (IV) Census 人口异质性 + 半合成乳腺癌筛查 → (V) 直面两种反对意见（"社会决定因素只是另一种敏感属性" / "敏感属性的因果效应已经包含了"）。  
输出：三条 actionable pillar——数据治理、动态公平度量（Social Determinant Parity）、把社会决定因素当作显式干预目标的因果框架。

### 关键设计

1. **社会决定因素的三判据定义（Definition 2.2）**：

    - 功能：把 social determinant 从 sensitive attribute、proxy、行政标签里严格切出来，避免后续讨论被术语模糊吞掉。
    - 核心思路：变量 $S$ 同时满足三条才算社会决定因素——(a) **Context-level definition**：定义在某个情境（neighborhood / 机构 / 司法辖区）上，多个个体共享同一个 $S$ 值；(b) **Social-structural content**：跨情境的差异主要由资源配置、机构政策、系统性投资塑造（学校经费 ✓，zip code 标签 ✗）；(c) **Exogenous stratification**：聚合所用的边界（neighborhood / 邮政区）外生地划定，而不是按被刻画群体的特征 endogenously 划定。基于这三条，table 1 给出干净分类：race=敏感属性；zip code=非社会决定因素（行政标签）；HOLC redlining 区的族裔构成=敏感属性的 proxy（边界 endogenous）；zip code 区的族裔构成 / 学校经费=社会决定因素。
    - 设计动机：以前文献常把"用 neighborhood 当代理跑 race 的歧视"和"审计 neighborhood 本身的结构条件"混为一谈，前者是 redlining 的延伸，后者是审计结构性不公；不区分就会把改善学校经费这种可干预动作排除在视野外。

2. **配额式录取的结构性不公定理（Theorem 4.5）**：

    - 功能：用闭式不等式形式化"只按 race 设配额的 affirmative action 何时会伤害贫困区的非 URM 申请者"，从而证明敏感属性中心化缓解 ≠ 推进结构正义。
    - 核心思路：在 4 个假设下（区域族裔分布失衡 + Academic Preparedness $\perp$ Race $\mid$ Region + 富区分数 CDF 随机占优贫困区 + 选拔性大学有限名额 $g$），把 URM 配额写成 $\eta_{\mathrm{quota}} \cdot \frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n} g$。定理给出反例不等式：只有当 $\max_q \frac{F^{(\mathrm{rich})}(q)}{F^{(\mathrm{poor})}(q)} \ge \frac{\eta_{\mathrm{quota}}}{1+(1-\eta_{\mathrm{quota}})\frac{n_a^{(\mathrm{poor})}+n_a^{(\mathrm{rich})}}{n_{a'}^{(\mathrm{poor})}+n_{a'}^{(\mathrm{rich})}}}$ 时，贫困区非 URM 申请者面对的分数阈值才不会被推得比富区 URM 申请者更高。
    - 设计动机：揭示一个反直觉悖论——结构性不公越严重（左侧随机占优比越大），不等式越容易成立、配额伤害越小；反过来结构正义改善时，沿用同样的配额反而越可能制造新的不公；而且 $\eta_{\mathrm{quota}}$ 越大、右侧门槛越高，"越激进的敏感属性中心化缓解越会放大对贫困区非 URM 的伤害"。这把"为什么必须审计社会决定因素"从直觉上升为可证明命题。

3. **乳腺癌筛查半合成实验（Section 5.2）**：

    - 功能：把上面那套理论落到一个高风险医疗场景，证明即使统一执行筛查指南，社会决定因素仍然制造系统性差距，而干预社会决定因素能拿到可量化的早检收益。
    - 核心思路：用 OSF HealthCare 2012–2022 约 5.4 万次筛查 / 4.5 万患者的真实记录，把贫困区（ADI ∈ [75,100)）和富区（ADI ∈ [0,25)）白人女性的"首次筛查年龄"分布画出来——同一指南、同一族群，均值差 >3 年，中位数差近 5 年。然后用 100k 粒子模拟 + SEER 年龄别发病率采样癌症 onset，把 10k 个筛查名额按"现状分布 vs 改进分布"（贫困区改用富区的首次筛查年龄分布）+ "全分给贫困区 vs 两区均分"四种政策组合各跑 500 次，统计"首次筛查年龄 ≤ onset 年龄 = 早检"次数。结果：在贫困区采用改进型筛查模式后，早检数从 $1367 \pm 33$ 升到 $1461 \pm 36$。
    - 设计动机：(a) 实证回应"fairness through unawareness"——同一族群、同一指南都消不掉差距，必须把社会决定因素显式纳入审计；(b) 量化"干预社会决定因素 vs 调整敏感属性配额"的差别，给 Pillar 3 的"把社会决定因素当干预目标"提供经验佐证。

### 损失函数 / 训练策略
立场论文不涉及训练目标。但作者在 Pillar 2 提出新度量 **Social Determinant Parity**：把现有 Demographic Parity 的条件变量从族裔换成 area deprivation index、基础设施可及性、政策暴露等结构变量；其纵向版本进一步要求度量随情境变量时变追踪。Pillar 3 进一步主张引入多层因果模型 + causal representation learning，让社会决定因素成为显式干预节点而不是 mediator。

## 实验关键数据

### 主实验

| 场景 | 关键数据 | 说明 |
|------|----------|------|
| Census PUMS, 加州 African American 女性年收入中位数 | 低 ADI \$38,000 / 中 ADI \$23,800 / 高 ADI \$18,800 | 同一族裔 × 性别交集，社会决定因素仍带来 >2× 的中位收入差距 |
| OSF HealthCare, 白人女性首次乳腺癌筛查年龄 | 富区 vs 贫困区：均值差 >3 年，中位数差 ≈5 年 | 同一统一筛查指南，差距只能归因于结构性条件 |
| 乳腺癌半合成模拟（10k 名额全分给贫困区，500 次） | 现状模式 $1367 \pm 33$ → 改进模式 $1461 \pm 36$ 次早检 | 仅靠改善"首次筛查年龄分布"这一个社会决定因素相关代理就拿到约 +7% 的早检收益 |

### 消融 / 政策对比

| 配额倍率 $\eta_{\mathrm{quota}}$ | 不等式 (1) 右侧门槛 | 贫困区非 URM 受损概率 |
|---|---|---|
| $\eta=1$（自然比例） | 右侧 = 1 | 与结构性不公严重度直接挂钩 |
| $\eta$ 增大 | 右侧单调升高 | 越大越易违反 → 贫困区非 URM 越被挤压 |
| 结构性不公改善（CDF 比下降） | 左侧下降 | 同 $\eta$ 下反而更易制造新不公 |

### 关键发现
- **同族同性别也能差 2 倍**：Figure 1 直接打脸"交集敏感属性已经足够"——African American 女性这一最常被讨论的交集群体，内部因 ADI 不同收入差距巨大。
- **统一指南消不掉差距**：OSF 数据里富区/贫困区白人女性走的是同一筛查指南，差距完全来自结构性条件（交通、可及性、信任度），说明"指南不感知社会决定因素"本身就是不公的来源。
- **配额悖论**：理论模型证明，结构正义越好的时候，配额反而越容易反噬贫困区非 URM；激进配额放大伤害——这是对"affirmative action 是不是越多越好"的形式化警告。
- **可干预性**：半合成实验显示仅"改进首次筛查年龄分布"就能换来 +94 次早检/10k 筛查，说明社会决定因素是真正可作为政策杠杆操作的，而 race 不是。

## 亮点与洞察
- **三判据定义切得极干净**：用 context-level / social-structural / exogenous stratification 三条 yes/no，能把 race / zip code / HOLC 族裔构成 / zip code 族裔构成 / 学校经费 摆进截然不同的格子，立刻让以后任何"我把 neighborhood 当 sensitive attribute 加进去就完事"的偷懒做法暴露问题，工具性极强。
- **配额悖论的反直觉**：业界长期讨论"affirmative action 是否公平"几乎都停留在哲学/价值判断；这篇用一个 4 假设 + 1 不等式把"何时配额会反过来伤害最弱势子群"变成可验算的条件，把哲学争论拉回数学，是这篇 position paper 最让人"啊哈"的地方。
- **审计先于缓解（auditing must precede mitigation）**：这一句方法论口号可迁移到任何"先评估后干预"的责任 AI 场景——比如把它套到 RLHF 的 reward 数据治理、医疗算法部署，都能直接复用 Pillar 1–3 的三层框架。
- **把医疗 SDoH 拉回 ML 公平**：Obermeyer 等人 2019 年那篇黑人医疗算法的工作之后，ML 社区一直缺一个"如何系统化地把 SDoH 接进公平框架"的入口，这篇把 OSF 真实数据 + 半合成模拟做出来，给后续工作铺了具体的实验范式。

## 局限与展望
- 作者承认：理论模型只刻画了"区域间"结构性不公，没考虑同一学校/机构内部的种族歧视；半合成乳腺癌实验也只覆盖首次筛查年龄这一个杠杆，不能视为对结构性壁垒的因果断言。
- 三判据定义在落地时还有灰色地带——例如"institutional membership 是否 exogenous"在很多就业、教育场景里其实并不清楚（college 录取本身就是 endogenous 选择），实操中如何稳定分类需要更多 case study。
- Social Determinant Parity 作为度量只在概念层提出，尚未给出具体可优化的微分形式，也没和现有 in-processing / post-processing 公平算法做实证对比。
- 多层因果模型 + causal representation learning 的设想要求观测足够多 community-level 协变量，且需要解决 SUTVA / no interference 失效问题，工程化路径仍待后续工作。

## 相关工作与启发
- **vs Conditional Demographic Parity (Žliobaite et al., 2011; Wachter et al., 2021)**: 后者用 region 当 mediator 解释 race 与 outcome 的残差依赖，本质仍是"race 视角"；本文反过来主张把 region 上承载的结构性变量本身设为审计目标，并明确撇清自己不是 Conditional Demographic Parity 的变种。
- **vs Path-specific Causal Fairness (Zhang & Bareinboim, 2018a; Chiappa, 2019; Wu et al., 2019)**: 这些方法虽然可以把社会决定因素塞进 race → SD → Y 的路径，但默认 SD 是 race 的下游 mediator；本文指出环境变量并非 race 的 ancestor，且把 SD 当 mediator 会丢掉它"可被政策直接干预"的杠杆属性。
- **vs Domain Adaptation 类公平 (Madras et al., 2018; Creager et al., 2021)**: 它们把跨情境异质性当作 distribution shift 去 normalize；本文反对这种"把情境视为噪声"的范式，主张情境正是要审计的信号本身。
- **vs Obermeyer et al. (2019) 黑人医疗算法**: 后者实证揭示了"用 cost 代替 need 会带来种族偏差"，是 SDoH 进入 ML 公平讨论的代表作；本文在这条线上更进一步，提供概念定义 + 理论模型 + 通用 pillar，把单点案例升级为方法论。
- **vs Kasirzadeh (2022)** 等结构性不公哲学讨论: 这些工作主要在哲学层呼吁；本文给出三判据 + 闭式定理 + 半合成实验，是哲学呼吁向 ML 工程实践的第一座桥梁。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在 ML 公平这条已经被刷得很满的赛道上，把"审计对象"本身重新定义并形式化，立场层级的创新。
- 实验充分度: ⭐⭐⭐⭐ 理论闭式 + Census 真实数据 + 真实医疗数据 + 半合成模拟四块齐全；扣一星是没和现有 fairness 算法做端到端对比。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链条 (I)–(V) 极清晰，table 1 / 三判据 / 配额定理彼此呼应，alternative views 一节正面回应了最强反驳。
- 价值: ⭐⭐⭐⭐⭐ 提供了可直接照抄的概念框架（三判据）+ 可计算的工具（Social Determinant Parity / 配额不等式）+ 可落地的三条 pillar，对 ICML 社区方向引导意义大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)
- [\[ICML 2026\] Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/medical_imaging/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](../../CVPR2026/medical_imaging/benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->
