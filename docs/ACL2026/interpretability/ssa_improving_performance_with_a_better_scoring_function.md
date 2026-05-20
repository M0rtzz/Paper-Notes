---
title: >-
  [论文解读] SSA: Improving Performance With a Better Scoring Function
description: >-
  [ACL 2026][可解释性][Softmax饱和] 这篇论文指出 Softmax attention 在分布偏移下会因大幅值 token 产生近似 hardmax 的注意力塌缩，并提出 Scaled Signed Averaging 作为可训练的替代评分函数…
tags:
  - "ACL 2026"
  - "可解释性"
  - "Softmax饱和"
  - "注意力评分函数"
  - "SSA"
  - "分布偏移"
  - "ICL泛化"
---

# SSA: Improving Performance With a Better Scoring Function

**会议**: ACL 2026  
**arXiv**: [2508.14685](https://arxiv.org/abs/2508.14685)  
**代码**: https://github.com/omyokun/SSA/  
**领域**: 可解释性 / 注意力机制 / In-Context Learning  
**关键词**: Softmax饱和、注意力评分函数、SSA、分布偏移、ICL泛化

## 一句话总结
这篇论文指出 Softmax attention 在分布偏移下会因大幅值 token 产生近似 hardmax 的注意力塌缩，并提出 Scaled Signed Averaging 作为可训练的替代评分函数，在合成 ICL 任务、114M decoder-only 语言模型和 BabyBERTa encoder 探针上都比 Softmax 有更好的泛化表现。

## 研究背景与动机
**领域现状**：Transformer 的注意力机制默认使用 Softmax 把 query-key 分数归一化为权重。Softmax 的成功使它几乎成为标准配置，但越来越多 ICL 研究发现，模型在训练分布附近表现良好，一遇到简单分布偏移就会失效。

**现有痛点**：很多 ICL 失败分析停留在数据规模、预训练语料或任务分布层面，很难确认到底是模型没有学到规则，还是架构本身让模型在特定输入下无法聚合上下文。作者构造了从零训练的小模型实验，试图排除预训练语料和 prompt 工程的干扰。

**核心矛盾**：ICL 需要模型整合多个上下文示例，但 Softmax 在 logits 差距较大时会指数级集中到最大项。一个幅值异常大的 token 即使与任务无关，也可能吸走几乎全部注意力。

**本文目标**：证明 Softmax saturation 是 ICL 分布外泛化失败的一个架构性来源，并提出一个能延缓注意力塌缩、保留更多上下文质量的替代评分函数。

**切入角度**：论文用两个透明合成任务定位问题：量词判断任务要求模型看完整序列，线性函数任务要求模型从上下文样例中推断 $f(x)=ax+b$。两者都能清楚暴露“一个 deviant value 破坏整体推理”的现象。

**核心 idea**：用 SSA 将指数增长换成可训练的多项式型 signed scaling，使注意力在大幅值输入下不那么快退化为 hardmax，同时仍保留对重要 token 的选择能力。

## 方法详解
论文先证明问题，再提出替代函数。作者不是直接说 Softmax 不好，而是逐步排除原因：模型能识别单个数字正负，attention-only 模型能学 ICL，FF-only 模型学不会，失败在 full transformer 和 attention-only 模型中都出现，因此注意力机制是关键位置；进一步看 attention map，极端值 token 会吸走权重。

### 整体框架
实验流程分三层。第一层是合成 ICL：量词 every/some 和线性函数预测，在训练分布 $N(0,1)$ 与测试分布偏移之间观察错误。第二层是 decoder-only 语言模型：在 FineWeb 10B tokens 上从零训练 114M 模型，对比 Softmax 与 SSA。第三层是 encoder-only BabyBERTa：在 AO-CHILDES 上训练并测试语法探针。

### 关键设计
1. **Softmax 饱和的机制诊断**:

    - 功能：解释为什么一个 deviant token 能破坏需要多 token 聚合的 ICL 任务。
    - 核心思路：若最大 logit 与其他 logit 的差距为 $\Delta$，Softmax 中非最大项权重会按 $e^{-\Delta}$ 衰减。当输入数值经过线性 embedding 后保留幅值顺序，极端数值自然产生大 embedding norm，进而让注意力接近 hardmax。
    - 设计动机：ICL 的许多规则依赖多个上下文例子共同决定答案，hardmax 式注意力会把“相关性”和“幅值大”混为一谈。

2. **Scaled Signed Averaging 评分函数**:

    - 功能：提供一个可替代 Softmax 的注意力归一化方式，延缓塌缩并保留上下文质量。
    - 核心思路：对每个 logit 使用 $(1+b|x|)^{sgn(x)n}$ 变换后再归一化，其中 $b>0$、$n\geq1$ 可训练。正值以多项式方式增长，负值向 0 衰减；当 $b=1/m,n=m$ 且 $m\to\infty$ 时可近似指数函数。
    - 设计动机：Softmax 在全局放大或单个 token 过强时会指数级塌缩，SSA 在同样情况下只按多项式速度集中，给模型更多机会保留次强但仍相关的上下文 token。

3. **跨架构验证设计**:

    - 功能：确认 SSA 的收益不是只在玩具任务里成立。
    - 核心思路：作者把 SSA 放入 decoder-only Transformer 和 encoder-only BabyBERTa 中，与相同训练设置下的 Softmax 比较；还测试 temperature Softmax、Sparsemax、Entmax、CosFormer、SA-Softmax 等替代函数。
    - 设计动机：如果 SSA 只在合成数字任务上有效，它的架构价值有限；跨真实 NLP benchmark 和语法探针的收益说明它可能是通用 attention scoring 改动。

### 损失函数 / 训练策略
合成 ICL 中，线性函数任务用平方误差，量词任务用交叉熵，模型训练 500,000 steps，batch size 为 64。decoder-only 实验训练 114M Nemotron-style 模型，12 层、24 头、hidden size 768，在 FineWeb 10B tokens 上训练 22k steps。BabyBERTa 实验在 AO-CHILDES 上从零训练，并比较 SSA 固定指数 $n=1.5$ 和 $n=2$ 的版本。

## 实验关键数据

### 主实验

| Benchmark | 指标 | Softmax | SSA | 提升观察 |
|--------|------|------|------|------|
| arc_challenge | acc_norm | 0.2398 | 0.2713 | 科学常识题提升 |
| arc_easy | acc_norm | 0.2934 | 0.5387 | 最大幅度之一 |
| boolq | acc | 0.3783 | 0.5618 | 二分类理解明显改善 |
| cb | acc / f1 | 0.1429 / 0.1310 | 0.4643 / 0.2663 | 小数据 NLI 上提升显著 |
| copa | acc | 0.5900 | 0.6400 | 因果选择提升 |
| hellaswag | acc_norm | 0.2550 | 0.3283 | 常识续写提升 |
| record | f1 / em | 0.1983 / 0.1932 | 0.2482 / 0.2427 | 阅读理解指标提升 |
| winogrande | acc | 0.4972 | 0.5178 | 代词消歧小幅提升 |

### 消融实验

| 分析项 | Softmax | SSA | 说明 |
|------|------|------|------|
| FineWeb perplexity ↓ | 21.86 | 19.73 | 训练分布文本困惑度更低 |
| Wikipedia perplexity ↓ | 24.58 | 22.07 | 分布外文本也更好 |
| BabyBERTa subject-verb across PP | 56.00 | 65.95 (SSA-2) | 长距离一致性改善 |
| BabyBERTa swapped arguments | 83.30 | 92.00 (SSA-1.5) | 论元结构更敏感 |
| BabyBERTa binding principle A | 78.25 | 87.90 (SSA-1.5) | 绑定关系探针提升 |
| BabyBERTa quantifier superlative | 71.20 | 83.95 (SSA-1.5) | 量词相关语法提升 |

### 关键发现
- 在合成 ICL 中，Softmax 模型在训练分布内表现良好，但一旦出现大幅值 deviant input，attention 会集中到单个 token，every/some 和线性函数预测都明显退化。
- SSA 不只是“加温度”的 Softmax。作者测试了温度缩放、Sparsemax、Entmax、混合 scoring heads、linear attention、CosFormer 和 SA-Softmax，这些替代函数都没有在任务上稳定超过 Softmax。
- SSA 在真实语言建模中也有效：114M decoder-only 模型只训练 22k steps，已经在多个 zero-shot benchmark 和困惑度上系统性优于 Softmax。

## 亮点与洞察
- 论文把一个常见但容易被忽略的问题讲清楚了：注意力权重中的“大”并不总等于“相关”。Softmax 的指数放大让这个混淆在分布偏移下变得非常严重。
- SSA 的设计很简洁，只给每个 head 多了可学习的缩放形态，却能在理论上从指数塌缩改成多项式塌缩，这个 inductive bias 很干净。
- 合成任务、decoder-only LM、encoder-only BabyBERTa 三层实验让论文比普通“替换激活函数”工作更有说服力，因为它既解释失败，也验证迁移。

## 局限与展望
- decoder-only 实验只扩展到 114M 参数，训练 10B tokens、22k steps；能否在 7B 或更大模型、长训练和现代训练配方下保持优势仍未验证。
- SSA 缓解但没有完全解决强分布偏移，特别是输入分布和函数分布同时大幅偏移时仍会失败。
- 论文指出更根本的问题是 attention 结构会混淆 token 表示幅值与任务相关性，SSA 是局部修正而不是完整答案。
- 新评分函数可能影响已有高效 attention kernel 和推理部署，需要后续工程评估其速度、数值稳定性和硬件友好性。

## 相关工作与启发
- **vs Temperature Softmax**: 温度只能整体平滑分布，不能改变指数塌缩的本质；SSA 在极端放大时仍保留多项式上下文质量。
- **vs Sparsemax / Entmax**: 这些方法控制稀疏性，但在本文 ICL 分布偏移任务上没有稳定收益；SSA 针对的是大幅值 token 造成的饱和。
- **vs CosFormer / SA-Softmax**: 这些替代 attention 形式也尝试改变归一化或核函数，但实验中没有解决作者定位的 failure mode。
- **启发**: 对可解释性研究而言，架构诊断最好从可控任务做起；对模型设计而言，attention scoring function 仍是一个值得系统搜索的空间，不必默认 Softmax 最优。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 从 Softmax saturation 解释 ICL 泛化失败，并提出理论上有明确差异的评分函数，思路清晰。
- 实验充分度: ⭐⭐⭐⭐☆ 合成任务、真实 LM、encoder 探针和多替代函数比较都覆盖；大模型验证不足。
- 写作质量: ⭐⭐⭐⭐☆ 问题定位、数学解释和实验链条很顺，少量公式较长但总体可读。
- 价值: ⭐⭐⭐⭐☆ 对注意力机制改造和 ICL 泛化研究很有启发，实际大规模部署价值还需后续验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](../../NeurIPS2025/interpretability/empowering_decision_trees_via_shape_function_branching.md)
- [\[NeurIPS 2025\] Monte Carlo Expected Threat (MOCET) Scoring](../../NeurIPS2025/interpretability/monte_carlo_expected_threat_mocet_scoring.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/interpretability/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[AAAI 2026\] SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](../../AAAI2026/interpretability/som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)

</div>

<!-- RELATED:END -->
