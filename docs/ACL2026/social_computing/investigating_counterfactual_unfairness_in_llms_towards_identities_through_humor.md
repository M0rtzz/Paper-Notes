---
title: >-
  [论文解读] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor
description: >-
  [ACL 2026][社会计算][反事实公平性] 本文通过幽默场景系统调查 LLM 的反事实不公平性——交换说话者/听众身份后观察模型行为变化，发现特权群体说的笑话被拒绝率高达 67.5%，被判定为恶意的概率高 64.7%，且社会危害评分高达 1.5 分（5分制），揭示了模型内化了固定的社会特权层级而非进行真正的社会推理。
tags:
  - "ACL 2026"
  - "社会计算"
  - "反事实公平性"
  - "幽默偏见"
  - "身份偏见"
  - "LLM拒绝行为"
  - "社会特权层级"
---

# Investigating Counterfactual Unfairness in LLMs towards Identities through Humor

**会议**: ACL 2026  
**arXiv**: [2604.18729](https://arxiv.org/abs/2604.18729)  
**代码**: [GitHub](https://github.com/) (Code and Dataset 已提供)  
**领域**: AI公平性 / LLM偏见  
**关键词**: 反事实公平性, 幽默偏见, 身份偏见, LLM拒绝行为, 社会特权层级

## 一句话总结

本文通过幽默场景系统调查 LLM 的反事实不公平性——交换说话者/听众身份后观察模型行为变化，发现特权群体说的笑话被拒绝率高达 67.5%，被判定为恶意的概率高 64.7%，且社会危害评分高达 1.5 分（5分制），揭示了模型内化了固定的社会特权层级而非进行真正的社会推理。

## 研究背景与动机

**领域现状**：LLM 在高风险领域（招聘、教育、法律）日益部署，其编码的社会和文化偏见可能导致严重的社会危害。现有偏见研究主要关注去上下文化的偏见检测，忽视了会话角色和社会互动中的表征偏见。

**现有痛点**：(1) 幽默天然涉及社会感知的模糊地带——什么是可接受的取决于说话者和听众的身份；(2) 现有计算幽默研究忽视了会话上下文，未检验"当互动角色发生变化时意义如何转变"；(3) 缺乏量化 LLM 在身份互换下行为不对称性的系统方法。

**核心矛盾**：LLM 的安全对齐和偏见保护是双向的——模型不仅对特权群体施加更严格的审查，还通过过度保护边缘群体隐式地将其定性为脆弱，两者都强化了固定的社会层级。

**本文目标**：通过三个互补任务系统揭示 LLM 在幽默场景中的身份相关偏见：拒绝生成、意图推断和影响预测。

**切入角度**：利用反事实公平性原理——只改变敏感属性（说话者/听众身份）而保持其他因素不变，观察模型输出是否变化。幽默是特别敏感的探针，因为它迫使模型在模糊地带做出判断。

**核心 idea**：LLM 编码了固定的社会特权层级而非真正的社会推理——它们使用身份作为危害的代理信号，系统性地"向下打"的笑话被拒绝而"向上打"的被允许，创造了双向的表征危害。

## 方法详解

### 整体框架

三个任务追踪偏见在整个管道中的表现：Task 1（说话者-目标条件下的幽默生成拒绝）测试模型是否因身份配置不同而差异化拒绝；Task 2（说话者意图推断）在固定笑话内容下测试模型是否因身份不同而归因不同意图；Task 3（关系/社会影响预测）测试影响评估是否因身份不对称。覆盖 33 个身份、10 个类别、5 个 SOTA 模型。

### 关键设计

**1. 非对称拒绝率 (ARR) 与说话者效应 (SE)：把"身份互换下的方向性偏见"拆成两个正交维度**

只看一个二元拒绝率，无法分辨"模型是对所有人一视同仁地严，还是只对某个方向更严"。作者先定义非对称拒绝率 $\text{ARR} = |\text{RR}(A{\to}B) - \text{RR}(B{\to}A)|$，其中 $\text{RR}(A{\to}B)$ 是 A 群体对 B 群体讲笑话时的拒绝率——ARR 大就说明模型对调换说话者/听众身份采取了不对称的安全策略，直接暴露内化的层级。但 ARR 不知道这种不对称是谁造成的，于是再引入说话者效应 $\text{SE}(A{\to}B) = \text{RR}(A{\to}B) - \text{RR}(B)$，用"指定 A 当说话者"与"不指定说话者只针对 B"的拒绝率之差，隔离出说话者身份单独带来的放大或削弱：正 SE 表示指定该说话者抬高了拒绝率。两个指标分工明确——ARR 检测层级方向，SE 检测说话者放大效应。

**2. 多粒度拒绝类型分析：偏见不只藏在"是否拒绝"，还藏在"怎么拒绝"**

把拒绝当成 0/1 会漏掉模型态度的强弱。作者把拒绝细分为直接拒绝、显式替代和隐式替代三种，因为拒绝的严厉程度本身就是一种层级表态：直接拒绝意味着"绝对不可以"，提供替代笑话则意味着"可以但得改"。这个粒度立刻看出差别——GPT-4o 对 White→Black 的请求有 $62.7\%$ 是直接拒绝，而对反向的 Black→White 只有 $25.0\%$ 直接拒绝、却有 $47.5\%$ 给出替代笑话。同样是"没生成原笑话"，模型对两个方向的态度软硬完全不同。

**3. 身份无关 vs 身份特定贬损幽默两套数据集：分离"身份标记本身"和"身份内容"的效应**

如果偏见只在涉及身份的笑话里出现，那还能辩解说是内容问题。为堵住这个出口，作者造了两套数据：身份特定数据集是 $737$ 条贬损笑话，经严格过滤保证能套进任意身份配对使用；身份无关数据集是 $400$ 条不含任何身份标记的笑话（四种风格）。关键证据来自后者——即使笑话本身完全不提身份，仅仅改变说话者-听众的身份配置，模型的判断仍然变化。这就证明模型是把身份本身当成了危害的判断信号，而不是在读笑话内容做社会推理。

### 损失函数 / 训练策略

不涉及训练。评估 5 个 SOTA 模型（Claude 3.5 Haiku、GPT-4o、DeepSeek-Reasoner、Gemini 2.5 Flash-Lite、Grok 4），总计 48,400+ 次请求。用 GPT-4o 作为自动裁判评估拒绝行为。

## 实验关键数据

### 主实验

**各类别最高 ARR（%）**

| 类别 | 身份对 | Claude | GPT | DeepSeek | Gemini | Grok |
|------|-------|--------|-----|----------|--------|------|
| 财富 | poor, wealthy | 67.5 | 58.8 | 61.3 | 27.5 | 3.8 |
| 健康 | disabled, able-bodied | 50.0 | 47.5 | 63.8 | 48.8 | 10.0 |
| 种族 | Black, White | 16.3 | 43.8 | 43.8 | 33.8 | 25.0 |

### 消融实验

**Grok 4 vs 其他模型的 ARR 对比**

| 模型 | 平均 ARR | 说明 |
|------|---------|------|
| Claude 3.5 Haiku | 38.1 | 最严格的安全对齐 |
| DeepSeek-Reasoner | 38.5 | 高不对称性 |
| GPT-4o | 33.3 | 中等 |
| Gemini 2.5 | 27.1 | 相对温和 |
| Grok 4 | 6.3 | 低审查但仍存在种族ARR |

### 关键发现

- 所有模型一致地拒绝"向下打"（特权→边缘）的笑话，同时允许"向上打"——这映射了文化概念中的权力层级
- 模型隐式编码了争议性的特权判断：Chinese 被视为比 American 更弱势，janitor 比 software engineer 更弱势
- Grok 4（低审查训练）的 ARR 显著降低但种族维度仍存在 25% 的 ARR——说明偏见部分来自训练数据而非安全对齐
- 负 SE 效应揭示"保护性歧视"——blind 和 janitor 说话者获得更大的创作自由度

## 亮点与洞察

- "通过幽默看偏见"是一个绝妙的研究设计——幽默的社会敏感性迫使模型暴露其内化的社会假设，这在更直接的任务中可能被安全过滤器掩盖
- ARR 和 SE 指标的设计很精巧——分离了方向性不对称和说话者放大效应两个正交的偏见维度
- 发现偏见的双向性是重要贡献——过度保护和过度审查都是偏见的表现形式，模型不是在做社会推理而是在做身份查表

## 局限与展望

- 仅评估英文场景，跨文化幽默规范差异巨大
- 身份类别的选择基于西方社会框架
- 最小对实验仅覆盖有限的认知标记类型
- 需要更多的防御和缓解策略探索

## 相关工作与启发

- **vs BBQ/BOLD**: 测试表面偏见（词汇/情感），本文测试社会推理中的深层偏见
- **vs Holistic Bias**: 关注英文描述符，本文关注多语言的形态学实现
- **vs Safety alignment studies**: 关注拒绝率的整体水平，本文关注拒绝率的方向性不对称

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 通过幽默探测偏见的框架设计极具创意，三任务追踪全管道
- 实验充分度: ⭐⭐⭐⭐⭐ 48400+请求, 5模型, 33身份, 10类别, 多指标
- 写作质量: ⭐⭐⭐⭐⭐ 发现引人深思，论述有力
- 价值: ⭐⭐⭐⭐⭐ 对AI安全和公平性研究有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](../../ACL2025/social_computing/explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICLR 2026\] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](../../ICLR2026/social_computing/gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)

</div>

<!-- RELATED:END -->
