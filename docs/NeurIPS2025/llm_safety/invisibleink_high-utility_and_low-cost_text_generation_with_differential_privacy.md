---
title: >-
  [论文解读] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy
description: >-
  [NeurIPS 2025][LLM安全][differential privacy] 提出 InvisibleInk 框架，通过差分裁剪（DClip）隔离敏感信息和 Top-$k^+$ 截断采样两项创新，将差分隐私长文本生成的计算成本降低 8 倍以上…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "differential privacy"
  - "text generation"
  - "exponential mechanism"
  - "LLM decoding"
  - "privacy-preserving inference"
---

# InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy

**会议**: NeurIPS 2025  
**arXiv**: [2507.02974](https://arxiv.org/abs/2507.02974)  
**代码**: [cerai-iitm/invisibleink](https://github.com/cerai-iitm/invisibleink)  
**领域**: AI安全  
**关键词**: differential privacy, text generation, exponential mechanism, LLM decoding, privacy-preserving inference

## 一句话总结
提出 InvisibleInk 框架，通过差分裁剪（DClip）隔离敏感信息和 Top-$k^+$ 截断采样两项创新，将差分隐私长文本生成的计算成本降低 8 倍以上，首次实现不到非隐私生成 4-8 倍开销的高质量隐私文本生成。

## 研究背景与动机

**领域现状**：LLM 在 RAG、推理时扩展等范式中展现出强大的长文本生成能力，但生成过程中不可避免地需要处理隐私敏感数据（如医疗记录、法律文书、用户对话），存在通过输出泄露隐私的风险。

**现有痛点**：已有的差分隐私（DP）文本生成方法存在严重的实用性问题。SOTA 方法 Amin et al. [2024] 将 LLM 的 next-token 采样解释为指数机制，但需要 100 倍以上的非隐私生成计算开销才能产出非退化文本，小 batch size 下根本无法工作。

**核心矛盾**：DP 文本生成面临隐私-效用-计算的三角 tradeoff。传统裁剪（clip）方法对整个 logit 向量裁剪，但 logit 中 95%+ 的信息是语言模型的公共先验（语法、常识等），裁剪这些非敏感信息白白浪费了隐私预算。同时，DP 噪声使得从全词表采样产出低质量退化文本。

**本文要解决**：(i) 如何降低每个 token 的隐私成本从而减少计算量？ (ii) 如何在 DP 约束下使用截断解码提升文本质量？

**切入角度**：作者关键观察——私有 logit $\phi_i$ 与公共 logit $\phi_{\text{pub}}$（不含敏感参考的 logit）之差的分布范围比原始 logit 小约 10 倍。这意味着敏感信息只贡献了 logit 的一小部分，可以用更小的裁剪阈值。

**核心 idea**：只裁剪 logit 中相对于公共模型的增量（敏感部分），结合从公共 logit 派生的 top-$k$ 超集采样，大幅降低隐私成本和提升文本质量。

## 方法详解

### 整体框架
给定查询 $\boldsymbol{q}$ 和 $B$ 个敏感参考文本 $\boldsymbol{R} = \{\boldsymbol{r}_1, \ldots, \boldsymbol{r}_B\}$，InvisibleInk 逐 token 生成隐私文本。每步获取 $B$ 个私有 logit $\phi_i = \phi(\cdot|\boldsymbol{q}, \boldsymbol{r}_i, \boldsymbol{x}_{<t})$ 和一个公共 logit $\phi_{\text{pub}} = \phi(\cdot|\boldsymbol{q}, \boldsymbol{x}_{<t})$，经 DClip 裁剪聚合后，在 Top-$k^+$ 词汇子集上用指数机制采样下一个 token。

### 关键设计

1. **DClip（差分裁剪）**:

    - 功能：只裁剪 logit 中的敏感增量部分而非全部 logit
    - 核心思路：$\text{DClip}_C(\phi_i, \phi_{\text{pub}}) := \phi_{\text{pub}} + \text{clip}_C(\phi_i - \phi_{\text{pub}})$，将每个私有 logit 与公共 logit 的差值裁剪到 $[-C, C]$，聚合后的敏感度为 $C/B$（replace-by-null 邻接下）
    - 设计动机：实验观察 $\phi_i - \phi_{\text{pub}}$ 的值域比 $\phi_i$ 小约 10 倍，95% 的词汇在 $C \approx 1$ 时不会被裁剪。传统方法需要 $C \approx 10$ 才能避免严重失真，而 DClip 用 $C \approx 1$ 即可，直接导致 8 倍计算节省（因为温度 $\tau \propto C/B$，$C$ 更小则需要更少的 $B$）

2. **Top-$k^+$ 采样**:

    - 功能：在公共 logit 确定的词汇集上做截断采样，无额外隐私成本
    - 核心思路：$V_k^+ = \{y \in V: \phi_{\text{pub}}(y) \geq \ell - 2C/B\}$，其中 $\ell$ 是公共 logit 的 top-$k$ 阈值。这是所有私有 top-$k$ 词汇的超集，但仅比 $V_k(\phi_{\text{pub}})$ 多约 10 个词汇
    - 设计动机：从全词表采样（$|V| \sim 10^5$）会产生退化文本；直接用公共 top-$k$ 会漏掉在敏感数据中常见但公共模型中罕见的领域特定词汇；Top-$k^+$ 兼顾两者——既截断了低概率噪声词汇，又保留了私有数据的有用信号

3. **非自适应隐私核算**:

    - 功能：提供封闭形式的隐私预算公式
    - 核心思路：对 $T$ 个 token 自适应组合，总隐私预算 $\rho_{\text{seq}} = TC^2/(2B^2\tau^2)$（zCDP），给定 $\rho_{\text{seq}}$ 和 $T$，可直接推导裁剪阈值 $C = B\tau\sqrt{2\rho_{\text{seq}}/T}$
    - 设计动机：之前方法的自适应 DP 保证只能通过网格搜索校准超参数，网格搜索本身增加隐私成本且被过去工作忽略；InvisibleInk 提供用户友好的非自适应保证，开箱即用

### 优雅退化特性
当隐私/计算预算极小时，$C$ 趋近 0，$\bar{\phi}_i \approx \phi_{\text{pub}}$，系统平滑退化为公共模型生成。之前方法在预算不足时要么完全无法生成（Amin et al.），要么只能输出极短文本（AdaPMixED）。

## 实验关键数据

### 主实验：隐私-效用-计算 tradeoff（MIMIC 数据集，TinyLLaMA 1.1B）

| 方法 | Batch Size $B$ | MAUVE(%) (ε=10) | MAUVE(%) (ε=1) | 备注 |
|------|------------|----------------|---------------|------|
| InvisibleInk (k=100) | 3 | 68.4 | 68.3 | 极低计算即可工作 |
| InvisibleInk (k=100) | 15 | 73.7 | 67.9 | |
| InvisibleInk (k=100) | 31 | **75.0** | **69.2** | SOTA |
| Amin et al. | 63 | 68.9 | INF | $B≤31$ 完全无法生成 |
| Amin et al. | 127 | 70.1 | INF | 需 8 倍以上计算 |
| AdaPMixED | 31 | 58.9 | 56.8 | 表现最差 |
| 非隐私生成 (ε=0) | - | 68.56 | - | 基准线 |

### 消融实验：DClip 和 Top-$k^+$ 的贡献（MIMIC, ε=10, B=7）

| 配置 | MAUVE (TinyLLaMA) | MAUVE (LLaMA3.2-1B) | 说明 |
|------|------------------|--------------------|----|
| InvisibleInk (k=100) | 72% | 76% | 完整模型 |
| InvisibleInk (k=\|V\|，无 Top-$k^+$) | 65% | 62% | 去掉截断采样掉 7-14% |
| Amin et al. (同等 B=7) | ~56% | - | 传统裁剪远不如 DClip |

### 下游任务（Yelp 分类，ε=10）

| 方法 | Batch Size | 50类 Accuracy | Top-5 Acc. | 类别 Acc. | Score L₁ |
|------|-----------|--------------|-----------|---------|---------|
| InvisibleInk | 7 | **32.98%** | **72.16%** | **64.90%** | **0.652** |
| Amin et al. | 127 | 29.44% | 64.56% | 60.18% | 0.748 |
| AdaPMixED | 31 | 7.44% | 34.72% | 56.82% | 1.858 |

### 关键发现
- DClip 贡献了 8 倍的计算效率提升：$\phi_i - \phi_{\text{pub}}$ 的值域比 $\phi_i$ 小约 10 倍，允许使用更小的 $C$
- Top-$k^+$ 在大词表模型上优势更大：LLaMA3.2 (128K 词表) 的提升比 TinyLLaMA (32K) 更显著（14% vs 7%）
- 在严格隐私预算 $\epsilon = 1$ 下，Amin et al. 在 $B \leq 127$ 时完全无法生成，而 InvisibleInk 在 $B=3$ 就能产出 MAUVE 68.3% 的文本
- 温度 $\tau \in [1.0, 1.1]$ 和 $k \approx 100$ 在各设置下接近最优，减少了超参数调优需求
- InvisibleInk 生成的医疗文本包含更多医学命名实体，表明更好地保留了领域信息

## 亮点与洞察
- **DClip 的核心洞察极为精巧**：logit 中大部分信息是语言先验（语法、常识），不含隐私信息，不应为其支付隐私成本。这个观察将敏感度降低了近一个数量级。
- **Top-$k^+$ 的设计巧妙平衡了隐私和效用**：直接对私有 top-$k$ 做截断会泄露隐私（token 跳出 top-$k$ 导致概率从正到零），而基于公共 logit 构建的超集完全不消耗隐私预算，且仅多出约 10 个候选词——开销几乎可以忽略。
- **优雅退化特性有实用价值**：预算不足时自然退化为公共模型，而非输出乱码，这在工程部署中非常重要。
- **提供 pip 可安装包 invink**，可直接调用生成 DP 文本，极大降低使用门槛。

## 局限与展望
- **信任模型限制**：需要可信的中心聚合器，拥有原始敏感数据和模型权重的白盒访问权限
- **仅保护推理阶段**：不防护训练/微调阶段的隐私泄露，也不覆盖数据污染或后门攻击
- **仅支持样本级 DP**：用户级隐私保护（同一用户的多个文档）尚未解决
- **8 倍开销仍有优化空间**：对于超大模型和极严格隐私预算，8 倍的计算开销仍显著，隐私预算对 token 数 $T$ 的线性依赖有可能通过正则化改善
- 实验主要使用 1B 级模型，8B 级只做了扩展验证，更大模型的表现未知

## 相关工作与启发
- **vs Amin et al. [2024]（EMNLP Findings）**：同样基于指数机制，但对整个 logit 裁剪导致敏感度过大，$B \geq 50$ 才能工作；InvisibleInk 用 DClip 将所需 $B$ 降至个位数
- **vs AdaPMixED [Flemings et al.]**：自适应 DP budget 在长文本生成时迅速耗尽（$B=8$ 时仅生成约 10 个 token），且数据依赖的隐私保证难以预先校准
- **vs AugPE [API-access]**：API 访问限制使隐私化天然更难，InvisibleInk 在所有设置下全面超越；但 AugPE 在预训练数据充分覆盖的数据集（如 Yelp）上有竞争力
- **vs DP 微调方法**：微调大模型的 DP 开销极高，InvisibleInk 的推理时隐私化更加经济实用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ DClip 的"只裁剪敏感增量"思想简洁而强大，Top-$k^+$ 的超集构造优雅解决了截断采样与 DP 的矛盾
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个隐私敏感领域数据集，多种模型规模，全面的消融和下游评估
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，动机讲解清晰，图示直观（Fig.3 的 logit 分布对比一目了然）
- 价值: ⭐⭐⭐⭐⭐ 将 DP 文本生成从 100 倍开销降到 4-8 倍，首次使其在实际规模下可行，开源代码进一步提升了影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] ModHiFi: Identifying High Fidelity Predictive Components for Model Modification](modhifi_identifying_high_fidelity_predictive_components_for_model_modification.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)

</div>

<!-- RELATED:END -->
