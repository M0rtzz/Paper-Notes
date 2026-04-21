---
title: >-
  [论文解读] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph
description: >-
   提出 GraphFilter 方法，将 SFT 数据集建模为句子-n-gram 的二部图，通过乘法优先级函数同时优化数据质量和多样性，在 3 个模型 6 个基准上全面超越 9 种基线方法。
tags:

---

# The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph

> **arXiv**: [2410.12458](https://arxiv.org/abs/2410.12458)
> **会议**: ICML 2025
> **领域**: LLM/NLP
> **作者**: Minghao Wu, Thuy-Trang Vu, Lizhen Qu, Gholamreza Haffari (Monash University)

## 一句话总结

提出 GraphFilter 方法，将 SFT 数据集建模为句子-n-gram 的二部图，通过乘法优先级函数同时优化数据质量和多样性，在 3 个模型 6 个基准上全面超越 9 种基线方法。

## 研究背景与动机

大语言模型（LLM）在监督微调（SFT）阶段的性能高度依赖训练数据的质量和多样性。然而，现有数据选择方法往往顾此失彼：

- **质量导向**方法（如 AlpaGasus、Deita、SuperFilter）侧重挑选高质量数据，但可能忽视语言模式的多样性，导致过拟合
- **多样性导向**方法（如 K-means 聚类、InsTag 话题标注）关注覆盖广度，但可能混入低质量数据
- 两者的失衡都会导致微调后的模型表现不佳

核心挑战：如何在数据选择中**同时最大化质量和多样性**？

## 方法详解

### 问题形式化

给定 SFT 数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，目标是选择大小为 $k$ 的子集 $\mathcal{S}_\pi$，使微调后的模型 $f_\theta$ 在下游任务上性能最优：

$$\pi^* = \arg\max_\pi \mathcal{R}(f_\theta; \mathcal{D}_{\text{tst}}), \quad \text{s.t. } |\mathcal{S}_\pi| = k$$

### 二部图建模

将数据集建模为二部图 $\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})$：

- **句子节点** $\mathcal{U} = \{u_i\}_{i=1}^N$：每条训练样本的指令部分
- **n-gram 节点** $\mathcal{V} = \{v_j\}_{j=1}^M$：包括 unigram (n=1)、bigram (n=2)、trigram (n=3)
- **边** $\mathcal{E} \subseteq \mathcal{U} \times \mathcal{V}$：句子包含某 n-gram 则连边

### 迭代选择算法（GraphFilter）

本质上是 **集合覆盖问题的贪心求解**：

1. 从空集 $\mathcal{S} = \emptyset$ 开始
2. 每轮选择优先级最高的句子 $u^* = \arg\max_{u \in \mathcal{U}} \phi(u)$
3. 将 $u^*$ 加入 $\mathcal{S}$，从图中移除与 $u^*$ 关联的所有 n-gram 节点及其连边
4. 重新计算剩余句子的优先级
5. 重复直到 $|\mathcal{S}| = k$

使用**最大堆**优化选择过程，将每轮复杂度从 $O(N)$ 降至 $O(\log N)$。

### 优先级函数

乘法组合质量和多样性：

$$\phi(u) = \text{Quality}(u) \times \text{Diversity}(u)$$

**质量指标** — SuperFilter (IFD)：

$$\text{Quality}(u) = \frac{\text{ppl}(y|x)}{\text{ppl}(y)}$$

其中 $\text{ppl}(y|x)$ 是给定指令后回复的困惑度，$\text{ppl}(y)$ 是回复单独的困惑度。比值越大表示指令对回复的信息增益越大。

**多样性指标** — TF-IDF 累积：

$$\text{Diversity}(u) = \sum_{v \in \mathcal{V}_u} \text{TF-IDF}(v)$$

$$\text{TF-IDF}(v) = \text{TF}(v) \times \log\frac{N}{d_v}$$

注意：$\mathcal{V}_u$ 随图的更新动态变化，已被覆盖的 n-gram 不再贡献分数。

### 关键设计：仅对指令应用 GraphFilter

GraphFilter 只作用于 SFT 数据的指令部分（instruction），不处理回复部分。这是基于"指令多样性比回复多样性更重要"的观察。

## 实验

### 主实验结果

使用 Magpie 300K 数据集，选择 10K 子集进行微调。在 4 类标准化基准（MMLU、ARC、HellaSwag、GSM8K）和 2 个 LLM-as-Judge 基准（AlpacaEval-2.0、MT-Bench）上评估。

**GraphFilter vs. 最强基线的提升（$\mu_{\text{all}}$ 综合分）**：

| 模型 | GraphFilter | 最强基线 | 提升 |
|------|------------|---------|------|
| Gemma-2-2b | 35.06 | 34.36 (SuperFilter) | +0.70 |
| Mistral-7B-v0.3 | 39.66 | 38.40 (SuperFilter) | +1.26 |
| Llama-3-8B | — | — | +3.38 |

GraphFilter 在**所有三个模型**上全面超越全部 9 种基线方法。

**计算效率**：GraphFilter 无需 GPU 即可完成数据选择，仅需 CPU 运算，显著优于需要 LLM 推理的方法（如 AlpaGasus 需调用 ChatGPT、ArmoRM 需 GPU 打分）。

### 消融分析

1. **质量 vs. 多样性**：
    - 仅用质量（SuperFilter）：表现不佳，缺乏覆盖
    - 仅用多样性（n-gram 度数）：比纯质量好，证明多样性的重要性
    - 乘法组合：最优，验证了两者互补

2. **指令 vs. 回复的多样性**：
    - 对指令应用 GraphFilter：最优
    - 对回复应用：效果下降
    - 验证了指令多样性的主导作用

3. **子集规模的影响**：
    - 小子集（1K-5K）：质量更重要
    - 大子集（10K-50K）：多样性更重要
    - GraphFilter 在各规模上都保持优势

4. **所选子集特征分析**：
    - GraphFilter 选出的子集 n-gram 覆盖率最高
    - 选出的指令长度更短但更多样化

## 亮点

- **理论支撑**：将数据选择形式化为集合覆盖问题，贪心算法有 $H(r)$ 近似因子保证
- **极致高效**：无需 GPU，仅 CPU 即可处理，远胜需要 LLM 推理的方法
- **设计简洁但有效**：乘法优先级函数的直觉清晰——选既好又独特的样本
- **全面实验**：3 个模型 × 6 个基准 × 9 种基线，结论可靠
- **实用性强**：可直接用于任何 SFT 数据选择场景

## 局限性

- 仅在 Magpie 数据集上验证，未覆盖其他合成或人工 SFT 数据集
- 质量指标固定为 SuperFilter，其他质量度量（如奖励模型分数）的兼容性未探索
- 仅使用 n-gram 作为多样性代理，未考虑语义级别的多样性
- 集合覆盖的贪心算法假设 n-gram 独立，忽略了 n-gram 间的语义关联
- 评估主要集中在英语任务，多语言场景未验证

## 评分

⭐⭐⭐⭐（4/5）

方法简洁高效、实验充分，将数据选择问题优雅地形式化为集合覆盖问题。尽管技术上并不复杂，但实际效果显著且实用性强，是数据选择领域的扎实贡献。

<!-- RELATED:START -->

## 相关论文

- [Soft Quality-Diversity Optimization](../../ICLR2026/llm_evaluation/soft_quality-diversity_optimization.md)
- [Hyperband-based Bayesian Optimization for Black-box Prompt Selection](hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)
- [IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [Unlocking Post-hoc Dataset Inference with Synthetic Data](unlocking_post-hoc_dataset_inference_with_synthetic_data.md)
- [How Much Can We Forget about Data Contamination?](how_much_can_we_forget_about_data_contamination.md)

<!-- RELATED:END -->
