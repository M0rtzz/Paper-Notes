---
title: >-
  [论文解读] Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation
description: >-
  [CVPR 2026][神经元解释] 提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。
tags:
  - CVPR 2026
  - 神经元解释
  - 概念验证
  - 可解释AI
  - 神经元功能分析
  - 闭环验证
---

# Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation

**会议**: CVPR 2026  
**arXiv**: [2603.24953](https://arxiv.org/abs/2603.24953)  
**代码**: 无  
**领域**: 可解释性 / XAI  
**关键词**: 神经元解释, 概念验证, 可解释AI, 神经元功能分析, 闭环验证

## 一句话总结

提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。

## 研究背景与动机

1. **领域现状**：神经网络可解释性研究中，理解单个神经元的功能（即它编码什么概念）是一个核心问题。现有方法如 Network Dissection、CLIP-Dissect、FALCON、DnD 等通过自然语言描述神经元概念，取得了一定进展。

2. **现有痛点**：这些方法都基于一个共同假设——每个神经元都有明确定义的功能并为决策提供区分性特征。但研究表明，网络中存在冗余神经元，它们并不贡献于决策。为这些神经元生成描述会导致误解，让人错误理解网络的决策机制。

3. **核心矛盾**：现有方法本质上是"观察→假设"的过程，从探测数据集上的激活分布推断神经元功能，但由于数据覆盖有限，这些假设可能存在数据集偏差，无法准确反映神经元的真实功能。缺少验证环节。

4. **本文目标**：(1) 如何过滤掉不提供区分性特征的神经元；(2) 如何验证生成的概念是否真正匹配神经元功能。

5. **切入角度**：借鉴神经科学的"观察→假设→验证"科学方法论，认为深度网络的可解释性研究也应遵循同样的闭环逻辑。

6. **核心 idea**：通过激活分布筛选有效神经元、聚类生成概念假设、再用文生图生成验证图像来闭环验证概念-神经元匹配度。

## 方法详解

### 整体框架

SIEVE 框架包含三个阶段：输入一个预训练分类网络和探测数据集，(1) **Select**：根据激活分布筛选出激活模式一致的高质量样本；(2) **Hypothesize**：对筛选后的样本聚类，利用视觉语言模型为每个聚类生成概念假设；(3) **Verify**：基于假设概念用 Stable Diffusion 生成图像，测量生成图像是否能高激活对应神经元，以此验证概念的准确性。

### 关键设计

1. **高激活样本筛选（Select）**:

    - 功能：筛选出激活模式一致、反映神经元明确功能的样本
    - 核心思路：计算每个神经元在探测数据集上的激活分布，用第99百分位数与中位数的比值来量化神经元的响应区分度。如果比值超过阈值 $\beta$（默认为10），则认为该神经元编码了明确的功能特征。对于通过筛选的神经元，取 top-20 高激活样本。高区分度神经元（如 Neuron 507）在特定刺激上有一致的高响应，而低区分度神经元（如 Neuron 144）响应分散。
    - 设计动机：过滤掉冗余/无区分性神经元，避免为它们生成误导性概念描述

2. **概念假设生成（Hypothesize）**:

    - 功能：为每个神经元的高激活样本生成自然语言概念描述
    - 核心思路：先根据神经元激活图裁剪出高激活区域的 patch，提取特征后用凝聚聚类分组（聚类数由 Silhouette 分数自动确定），捕获单个神经元可能对应的多种功能模式。对每个聚类，用 CLIP 在预定义概念集中匹配 top-K（K=2）概念作为该聚类的功能假设：$h_{i,j} = \arg\text{top-}K(\{g(t_q, C_{i,j}) \mid t_q \in \mathcal{T}\}, K)$
    - 设计动机：通过聚类发现单个神经元的多种功能模式，比单一描述更准确

3. **概念验证（Verify）**:

    - 功能：通过构造性干预验证假设概念与神经元功能的一致性
    - 核心思路：用假设概念作为文本 prompt，通过 Stable Diffusion 生成独立于探测数据集的验证图像集。将这些图像输入目标网络，计算激活率（Activation Rate）：$AR_i = \frac{1}{|\mathcal{D}_{gen}^{(i,j)}|} \sum_{x_{gen}} \mathbb{1}\{a_i^l(x_{gen}) > T_i\}$，其中 $T_i$ 为 Top 1% 激活阈值。AR 低的假设被丢弃，保留高 AR 的概念作为最终解释。
    - 设计动机：区别于传统的破坏性干预（如神经元消融），采用构造性方法——主动生成与假设概念一致的刺激来观察神经元反应，类似科学实验中的对照验证

### 损失函数 / 训练策略

本方法不涉及训练，是一个后分析框架。关键超参：激活阈值 $\beta=10$，聚类数自动确定，每个聚类取 top-2 概念，验证时使用 mean AR 作为过滤阈值。

## 实验关键数据

### 主实验

在 ImageNet 预训练的 ResNet-50 上，使用 Common Words (3k) 概念集：

| 方法 | CLIP cos | mpnet cos | mean AR (%) |
|------|----------|-----------|-------------|
| Network Dissect | 0.7073 | 0.3256 | 45.01 |
| CLIP-Dissect | 0.7868 | 0.4462 | 57.91 |
| WWW | 0.7713 | 0.4463 | 50.23 |
| DnD | 0.7595 | 0.4371 | 51.46 |
| **SIEVE (本文)** | **0.7914** | **0.4547** | **86.29** |

ViT-B/16 上类似趋势：SIEVE 的 mean AR 达 85.24%，远超 CLIP-Dissect 的 57.70%。

### 消融实验

| 配置 | CLIP cos | mpnet cos | mean AR (%) |
|------|----------|-----------|-------------|
| Baseline (无任何模块) | 0.6738 | 0.2306 | 45.57 |
| + Select + Cluster | 0.7624 | 0.4301 | 77.90 |
| + Select + Verify | 0.7821 | 0.4423 | 81.52 |
| + Select + Cluster (无Verify) | 0.7656 | 0.4189 | 72.87 |
| Full model | 0.7914 | 0.4547 | 86.29 |

### 关键发现

- **Verify 模块贡献最大**：去掉 Verify 后 mean AR 从 86.29% 降至 72.87%，证明验证环节对确保概念-神经元匹配的关键性
- **阈值 β 鲁棒**：β 在 4-12 范围内变化对最终指标影响极小（mean AR 波动 <1%）
- **域迁移场景下验证仍有效**：在遥感数据（EuroSAT）上存在域偏移时，SIEVE 仍达 75.45% mean AR，而 CLIP-Dissect 仅 43.16%
- **SIEVE 能提供更细粒度的描述**：如 ViT-B/16 的 Neuron 37 被描述为"Short Dense Coat"，而 baseline 只给出粗略的"Dog"

## 亮点与洞察

- **科学方法论的引入**：将神经科学的"观察→假设→验证"范式引入 DNN 可解释性，是一个非常优雅的跨领域类比。验证这一步补齐了现有方法的关键短板。
- **构造性验证**：不同于传统的消融实验（破坏性），通过文生图主动构造符合假设的刺激，这种正向验证更直接、更有说服力。
- **冗余神经元过滤**：第一个显式处理"不是所有神经元都有意义"这个问题的工作，避免了对冗余神经元的误解释。

## 局限与展望

- **文生图模型的域偏移**：验证阶段依赖 Stable Diffusion 生成图像，当目标网络训练在特殊领域（如遥感）时，生成图像与真实数据差异大，可能影响验证准确度
- **概念集的限制**：仍依赖预定义概念集（如 Broden、Common Words），无法发现概念集之外的新概念
- **计算开销**：需要为每个神经元的每个概念生成多张验证图像，规模化到整个网络时计算量显著
- **仅关注倒数第二层**：未扩展到浅层神经元的解释

## 相关工作与启发

- **vs CLIP-Dissect**: CLIP-Dissect 直接用 CLIP 匹配概念与激活样本，缺少验证环节；SIEVE 在匹配之后增加闭环验证，mean AR 提升约 30%
- **vs FALCON/WWW**: 这些方法改进了概念描述质量但仍假设所有神经元都有意义，SIEVE 通过 Select 阶段过滤冗余神经元
- **vs DnD**: DnD 用 LLM 生成更高质量的自然语言描述，但同样缺少验证，mean AR 仅 51.46%

## 评分

- 新颖性: ⭐⭐⭐⭐ 将科学方法论的闭环验证引入神经元解释领域，思路清晰且有说服力
- 实验充分度: ⭐⭐⭐⭐ 覆盖了 ResNet-18/50、ViT-B/16 多个模型和多个数据集，消融全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，科学方法论的类比引入自然
- 价值: ⭐⭐⭐⭐ 提出的 mean AR 指标可作为通用评估标准，验证范式对后续可解释性工作有参考意义

<!-- RELATED:START -->

## 相关论文

- [Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [Elastic Weight Consolidation Done Right for Continual Learning](elastic_weight_consolidation_done_right_for_continual_learning.md)
- [SineProject: Machine Unlearning for Stable Vision–Language Alignment](sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)
- [Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering](association_and_consolidation_evolutionary_memory-enhanced_incremental_multi-vie.md)

<!-- RELATED:END -->
