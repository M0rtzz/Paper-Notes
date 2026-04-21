---
title: >-
  [论文解读] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity
description: >-
  [ICLR 2026][数据过滤] 提出基于 token 词频先验（term frequency）的文本数据过滤方法，通过计算文档中 token 先验的均值和标准差来检测异常文档，实现了比 PPL 过滤快 1000× 以上且下游性能更优的数据清洗效果。
tags:
  - ICLR 2026
  - 数据过滤
  - 预训练数据质量
  - 词频先验
  - 困惑度替代
  - 高效数据选择
---

# Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity

**会议**: ICLR 2026  
**arXiv**: [2509.18577](https://arxiv.org/abs/2509.18577)  
**代码**: https://github.com/ybseo-ac/prior_filter  
**领域**: LLM效率  
**关键词**: 数据过滤, 预训练数据质量, 词频先验, 困惑度替代, 高效数据选择

## 一句话总结
提出基于 token 词频先验（term frequency）的文本数据过滤方法，通过计算文档中 token 先验的均值和标准差来检测异常文档，实现了比 PPL 过滤快 1000× 以上且下游性能更优的数据清洗效果。

## 研究背景与动机
LLM 预训练依赖海量网页语料，但网页数据噪声严重，需要精心过滤。现有数据过滤方法的演进路线：规则方法 → 分类器方法（FastText）→ n-gram 方法（DSIR）→ **PPL 过滤**（当前 SOTA）。

PPL 过滤的做法是：先训练一个小型参考模型（137M），然后对全部语料推理得到 PPL 分数，丢弃 PPL 偏离中位数最远的样本。虽然效果最好，但存在两大痛点：

**时间成本高**：6B token 语料需要 216 GPU 小时（训练参考模型 + 推理），面对不断增长的网页数据极不经济

**可靠性差**：小模型对噪声/OOD 数据的 PPL 估计不准，可能把重复或模式化噪声误判为有效文本

核心洞察：PPL 可贝叶斯分解为 $\log \text{PPL} \propto \pi_{\text{likelihood}} + \pi_{\text{prior}}$，其中先验项 $p(x_i)$ 不依赖上下文，可以用简单的词频统计替代模型推理。结合语言学观察——（1）词频是词角色的一维表征（高频→功能词，低频→内容词），（2）格式良好的句子有稳定的词汇密度——就可以用先验的统计特征检测异常文档。

## 方法详解

### 整体框架
1. 从语料库估计 token 先验 $p_{\text{prior}}(x) = \frac{f_D(x)}{\sum_{x' \in V} f_D(x')}$
2. 对每篇文档计算两个指标：先验均值 $\mu_d$ 和先验标准差 $\sigma_d$
3. 按距中位数的偏离程度过滤异常文档

### 关键设计

1. **Token 先验的三区分析**: 对词频排序后呈现三个明显聚类——高频区（功能词如 "the", "is"）、中频区（内容词如 "Phone", "tackles"）、低频区（噪声 token 如乱码和非主语言字符）。这验证了词频作为词角色一维表征的语言学假说。

2. **双指标过滤机制**: 

    - **先验均值** $\mu_d = \mathbb{E}_{x_i \in d}[\log p_{\text{prior}}(x_i)]$：反映文档中 token 的整体组成，偏高说明全是功能词（如空行），偏低说明全是噪声字符
    - **先验标准差** $\sigma_d = \text{std}_{x_i \in d}[p_{\text{prior}}(x_i)]$：反映 token 分布的均匀性，偏低说明 token 种类单一（如名词列表没有句子结构）
    - 两者捕捉不同类型的异常：$\mu_d$ 对应 PPL 中的 $\pi_{\text{prior}}$ 项，$\sigma_d$ 近似 $\pi_{\text{likelihood}}$ 的结构正则性

3. **多语言自适应**: 当少量中文混入英文语料时（<1%），中文几乎全部被判为异常；当中文比例>20%时，先验过滤自动将其视为有效语言。这一特性无需人工指定参考数据集（不像 DSIR）。

4. **子采样加速**: 即便只用 1% 的语料估计词频，得到的异常集合与全量几乎一致，仅需约 70 秒。

### 过滤流程
计算 $\delta_\mu(d) = |\mu_d - M_\mu|$ 和 $\delta_\sigma(d) = |\sigma_d - M_\sigma|$（$M$ 为语料中位数），在 $|F_\mu| = |F_\sigma|$ 约束下丢弃偏离最大的文档，保留 50% 数据量。

## 实验关键数据

### 主实验（Dolma 语料 20 个下游基准）

| 方法 | 类型 | 时间 | Avg | World Knowledge | Commonsense | Language | Symbolic | Reading |
|---|---|---|---|---|---|---|---|---|
| No-filter | 规则 | - | 5.78 | 5.52 | 0.44 | 6.14 | 13.22 | 3.59 |
| FastText | 分类器 | 3.6 hrs | 7.09 | 6.71 | 6.11 | 6.89 | 11.93 | 3.82 |
| DSIR | n-gram | 4 hrs | 7.56 | 7.03 | 6.84 | 7.31 | 12.67 | 3.97 |
| PPL-based | 模型 | **216 GPU hrs** | 8.22 | 9.98 | 11.91 | 7.34 | 7.91 | 3.96 |
| **Prior-based (Ours)** | 统计 | **0.25 hrs** | **9.20** | 9.53 | 11.27 | 10.31 | 11.13 | 3.79 |

1.5B 模型结果：Prior-based 以 0.25 小时达到最高平均分 9.20，比 PPL-based（8.22, 216 GPU 小时）更快更强。

### 消融实验（符号语言 Pile-github）

| 方法 | 时间 | Avg | CS Algo | Dyck | Operators | Math QA | GSM8k | SVAMP |
|---|---|---|---|---|---|---|---|---|
| No-filter | - | 9.51 | 35.75 | 12.30 | 5.71 | 1.15 | 0.15 | 2.00 |
| PPL-based | 224 GPU hrs | 11.21 | 37.42 | 20.60 | 7.14 | 2.09 | 0.00 | 0.00 |
| **Prior-based** | **0.26 hrs** | **12.03** | 38.86 | 21.30 | 9.04 | 1.17 | 0.15 | 1.67 |

在代码/数学等符号语言上也有效，验证了语言学先验跨语言类型的泛化性。

### 关键发现
- PPL 与先验过滤的异常集重叠率约 50%（在 e=0.10 过滤比例下），证实先验是 PPL 的有效近似
- PPL 在符号问题解决类任务上反而表现最差，因为小模型对代码/数学数据的 likelihood 估计不稳定
- 先验过滤在更大规模实验（3B 模型, 12B token）中持续优于 PPL
- 仅需 1% 子采样即可获得近乎一致的过滤结果，总时间约 1 分钟

## 亮点与洞察
- **极简即极强**：用最朴素的词频统计替代昂贵的模型推理，效果反而更好，揭示了 PPL 中 likelihood 项的不稳定是累赘
- 方法零模型依赖、零 GPU 需求，仅需 CPU 即可在分钟级完成万亿 token 级语料的过滤决策
- 可增量更新：新数据到来时只需更新词频表，无需重新训练参考模型
- 语言学视角切入（Al-Kindi 频率分析 → 词汇密度）非常优雅，理论动机扎实
- 多语言自适应特性极具实用价值，在无监督场景下自动处理多语混合数据
- 方法论非常清晰：从 PPL 的贝叶斯分解出发，逻辑链完整

## 局限与展望
- 方法依赖语言学属性，**不适用于图像等非文本模态**（论文自述）
- 对极小众但有价值的数据（如稀有术语论文）可能误删，需手动增补对应子集
- 512 token 块大小固定，对极短/极长文档的适配性未讨论
- 只用 GPT-2 tokenizer 验证，其他 tokenizer 的鲁棒性还需更多实验

## 相关工作与启发
- 与 PPL-based（Ankner et al., 2024）的关系：先验过滤是 PPL 的近似，但绕过了不稳定的 likelihood 项
- 与 DSIR 的对比：DSIR 需要人工指定参考数据集，先验过滤完全自适应
- 可用于 continual pretraining 的在线数据筛选，因为词频估计可增量更新
- 思路可启发其他领域：用简单统计特征替代昂贵的模型推理做数据质量评估
- 提示了一个更广泛的开放问题：数据质量估计的"gold standard"应该是什么？PPL 可能并非最佳答案

## 评分
- 新颖性: ⭐⭐⭐⭐ 词频分析不新但贝叶斯分解视角和双指标设计有洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ 20个基准+符号语言+多语言+大规模验证+效率对比非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 语言学动机→方法推导→实验验证逻辑链紧凑优雅
- 价值: ⭐⭐⭐⭐⭐ 1000×加速+更好性能，对LLM预训练数据管线有直接工程价值

<!-- RELATED:START -->

## 相关论文

- [Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)
- [ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)
- [STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/multilingual_mt/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)

<!-- RELATED:END -->
