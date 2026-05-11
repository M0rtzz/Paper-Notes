---
title: >-
  [论文解读] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity
description: >-
  [ICLR 2026][数据过滤] 提出基于 token 先验（词频统计）的文本数据过滤方法，利用文档内 token 先验的均值和标准差作为 PPL 的近似替代，在 20 个下游基准上取得最高平均性能，同时比 PPL 过滤快 1000 倍以上。
tags:
  - ICLR 2026
  - 数据过滤
  - 多语言翻译
  - 困惑度
  - 词频先验
  - 数据质量
---

# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity

**会议**: ICLR 2026  
**arXiv**: [2509.18577](https://arxiv.org/abs/2509.18577)  
**代码**: [GitHub](https://github.com/ybseo-ac/prior_filter)  
**领域**: 多语言翻译  
**关键词**: 数据过滤, 预训练, 困惑度, 词频先验, 数据质量

## 一句话总结

提出基于 token 先验（词频统计）的文本数据过滤方法，利用文档内 token 先验的均值和标准差作为 PPL 的近似替代，在 20 个下游基准上取得最高平均性能，同时比 PPL 过滤快 1000 倍以上。

## 研究背景与动机

### 预训练数据质量的重要性

大语言模型依赖海量 web 数据预训练，但 web 数据噪声极多。两大挑战：(1) 数据量巨大需要高效筛选以节省计算资源；(2) 噪声数据会损害模型性能。

### PPL 过滤的局限

基于困惑度 (PPL) 的过滤目前是 SOTA 方法，但有两个固有缺陷：

- **时间成本**：需先训练参考模型（137M），再对整个语料推理 PPL。对 6B token 语料需要 216 GPU 小时
- **可靠性问题**：模型对 OOD 样本（如噪声数据）的 PPL 估计不准，小模型尤其容易对重复或模式化噪声给出低 PPL（误认为高质量）

### 语言学启发

论文灵感来自 8 世纪语言学家 Al-Kindi 的密码分析方法：**分析词频可以揭示语言结构**。

两个关键语言学洞察：
1. **词频是词角色的一维表示**：高频词 = 功能词（"the", "is"），低频词 = 内容词（"president", "algorithm"）
2. **规范句子具有稳定的词汇密度**：功能词和内容词的比例在不同文档中保持相对稳定

## 方法详解

### 整体框架

核心思想：PPL 可通过贝叶斯分解为 likelihood + prior 两项，用 token 先验（词频统计）近似 prior 项，跳过需要模型推理的 likelihood 项。

### Token 先验估计

给定语料 $D$ 和词表 $V$，token $x$ 的先验概率通过词频估计：

$$p_{\text{prior}}(x) = \frac{f_D(x)}{\sum_{x' \in V} f_D(x')}$$

其中 $f_D(x)$ 是 token $x$ 在语料中的出现次数。

### 关键设计：双指标过滤

定义每个文档 $\texttt{d}$ 的两个统计量：

**先验均值** $\mu_{\texttt{d}}$：

$$\mu_{\texttt{d}} = \mathbb{E}_{x_i \in \texttt{d}} [\log p_{\text{prior}}(x_i)]$$

反映文档中 token 的组成——高/低先验 token 的平衡程度。

**先验标准差** $\sigma_{\texttt{d}}$：

$$\sigma_{\texttt{d}} = \text{std}_{x_i \in \texttt{d}} [p_{\text{prior}}(x_i)]$$

反映文档中 token 先验的分布结构——多样性/均匀性。

### 异常值检测

以语料级中位数为参考中心值 $M_\mu = \text{median}(\mu_{\texttt{d}})$、$M_\sigma = \text{median}(\sigma_{\texttt{d}})$，用偏离距离衡量异常程度：

$$\delta_\mu(\texttt{d}) = |\mu_{\texttt{d}} - M_\mu|, \quad \delta_\sigma(\texttt{d}) = |\sigma_{\texttt{d}} - M_\sigma|$$

丢弃 $\delta$ 最大的样本，约束 $|F_\mu| = |F_\sigma|$，直到剩余子集达到目标大小。

### PPL 近似的理论联系

$$\log \text{PPL}(\texttt{d}) \propto \underbrace{\sum_i \log p_\theta(x_{<i}|x_i)}_{\pi_{\text{likelihood}}} + \underbrace{\sum_i \log p_\theta(x_i)}_{\pi_{\text{prior}}}$$

- $\mu_{\texttt{d}}$ 精确等价于 $\pi_{\text{prior}}$ 项
- $\sigma_{\texttt{d}}$ 近似捕捉 $\pi_{\text{likelihood}}$ 反映的 token 间关系规律性
- 两者结合可作为 PPL 的合理代理

### 先验的独特优势

先验不仅近似 PPL，某些方面甚至更优：
- 小模型难以准确学习 likelihood，但词频统计简单稳定
- 模型对 OOD 噪声数据的 likelihood 估计不可靠，但先验不受此限
- PPL 会将重复/模式化噪声误判为高质量文本

## 实验关键数据

### 主实验：Dolma 语料上的下游任务性能

GPT-2 架构，1.5B 和 137M 模型，训练 40K 步（约 6B tokens），20 个下游基准。

| 方法 | 类型 | 时间 | 平均 | 世界知识 | 常识推理 | 语言理解 | 符号推理 | 阅读理解 |
|------|------|------|------|----------|----------|----------|----------|----------|
| No-filter | 规则 | - | 5.78 | 5.52 | 0.44 | 6.14 | 13.22 | 3.59 |
| FastText | 分类器 | 3.6h | 7.09 | 6.71 | 6.11 | 6.89 | 11.93 | 3.82 |
| DSIR | n-gram | 4h | 7.56 | 7.03 | 6.84 | 7.31 | 12.67 | 3.97 |
| PPL-based | 模型 | **216 GPU h** | 8.22 | 9.98 | 11.91 | 7.34 | 7.91 | 3.96 |
| **Prior-based** | 统计 | **0.25h** | **9.20** | 9.53 | 11.27 | **10.31** | 11.13 | 3.79 |

**关键结论**：先验过滤以 PPL 的 0.1% 时间成本取得了比 PPL 更高的平均性能（9.20 vs 8.22）。

### 符号语言实验：Pile-github

| 方法 | 时间 | 平均 | CS | Dyck | 运算 | 初等数学 | GSM | SVAMP |
|------|------|------|-----|------|------|----------|-----|-------|
| No-filter | - | 9.51 | 35.75 | 12.30 | 5.71 | 1.15 | 0.15 | 2.00 |
| PPL-based | 224 GPU h | 11.21 | 37.42 | 20.60 | 7.14 | 2.09 | 0.00 | 0.00 |
| **Prior-based** | 0.26h | **12.03** | 38.86 | 21.30 | **9.04** | 1.17 | 0.15 | 1.67 |

先验方法在代码/数学等符号语言上同样优于 PPL 过滤。

### 消融实验

**大规模一致性验证**（3B Qwen2.5-3B 和 1.5B 模型，12B tokens 训练）：先验过滤持续优于 PPL 过滤。

**子采样效率**：仅用语料 1% 的子集计算 token 先验，过滤结果与全量几乎一致（耗时从 30 分钟降至约 70 秒）。

**PPL 重叠分析**：当过滤比例 $e=0.10$ 时，$F_\mu$ 与 $F_{\text{ppl}}$ 的重叠率接近 50%，证实先验过滤确实近似 PPL 过滤。

### 关键发现

1. **PPL 在符号推理上最差**：PPL 会过滤掉小但有意义的代码/数学片段
2. **$\mu_{\texttt{d}}$ 异常值**多为极端高/低先验 token 的文档（换行符堆积、非英语文本）
3. **$\sigma_{\texttt{d}}$ 异常值**多为无结构的名词列表——有内容词但缺乏句法
4. **多语言自适应**：当中文数据占英文语料 <1% 时被自动过滤为噪声，>20% 时被识别为可学习语言

## 亮点与洞察

1. **极简思想的胜利**：仅用词频统计就超越了需要模型训练和推理的 PPL 方法
2. **语言学基础扎实**：从 Al-Kindi 的密码分析到词汇密度理论，每一步都有语言学支撑
3. **速度优势悬殊**：0.25 小时 vs 216 GPU 小时（提速 ~1000×），且持续增长的 web 数据加剧差距
4. **自适应多语言处理**：无需人工指定参考数据集，自动根据语言占比判断过滤/保留
5. **双指标互补**：$\mu_{\texttt{d}}$ 捕捉 token 组成，$\sigma_{\texttt{d}}$ 捕捉分布结构，覆盖不同类型噪声

## 局限性

1. 方法基于语言学特性，不适用于非文本模态（图像、音频等）
2. 先验过滤是 PPL 的近似，在捕捉"表面规范但语义无意义"的噪声方面弱于 PPL
3. 实验主体使用 GPT-2 架构，更现代架构（Llama 等）的验证有限
4. 对于极度偏重某种特定数据类型的训练目标（如纯数学），可能需要手动调整

## 相关工作与启发

- **Ankner et al. 2024 (PPL 过滤)**：本文直接对标的主要基线，证明先验比 PPL 更好更快
- **DSIR (Xie et al. 2023)**：需要手动指定参考数据集，先验过滤自动完成
- **FastText 分类器**：需要人工标注的参考数据，先验过滤完全无监督
- **启发**：数据过滤不需要复杂的模型推理，回归统计学基础可能是更好的选择

## 评分

- **新颖性**: ⭐⭐⭐⭐☆ — 思路极其简洁优雅，从语言学基础出发
- **理论深度**: ⭐⭐⭐⭐ — PPL 近似的贝叶斯分解分析透彻
- **实验充分度**: ⭐⭐⭐⭐ — 20 个基准 + 符号语言 + 大规模验证 + 多语言分析
- **实用价值**: ⭐⭐⭐⭐⭐ — 1000× 加速且性能更好，直接可用于工业数据管线
- **总评**: ⭐⭐⭐⭐☆ — 简单有效的方法论贡献，对预训练数据筛选有重大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[ICLR 2026\] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)
- [\[CVPR 2026\] SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](../../CVPR2026/multilingual_mt/sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)
- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/multilingual_mt/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)

</div>

<!-- RELATED:END -->
