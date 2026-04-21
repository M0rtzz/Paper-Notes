---
title: >-
  [论文解读] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models
description: >-
  [CVPR 2025][多模态][噪声标签学习] 本文发现在 CLIP 提示学习中简单替换 MAE 损失就能显著提升对噪声标签的鲁棒性，并通过特征学习理论证明了这一现象，进而提出 NLPrompt 方法——结合基于最优传输的数据净化（PromptOT）将数据分为干净/噪声子集后分别用 CE 和 MAE 损失训练，在多种噪声设置下大幅超越现有方法。
tags:
  - CVPR 2025
  - 多模态
  - 噪声标签学习
  - 提示学习
  - CLIP
  - 最优传输
  - 鲁棒性
---

# NLPrompt: Noise-Label Prompt Learning for Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.01256](https://arxiv.org/abs/2412.01256)  
**代码**: https://github.com/qunovo/NLPrompt  
**领域**: 多模态VLM  
**关键词**: 噪声标签学习, 提示学习, CLIP, 最优传输, 鲁棒性

## 一句话总结

本文发现在 CLIP 提示学习中简单替换 MAE 损失就能显著提升对噪声标签的鲁棒性，并通过特征学习理论证明了这一现象，进而提出 NLPrompt 方法——结合基于最优传输的数据净化（PromptOT）将数据分为干净/噪声子集后分别用 CE 和 MAE 损失训练，在多种噪声设置下大幅超越现有方法。

## 研究背景与动机

**领域现状**：基于 CLIP 的提示学习（如 CoOp、CoCoOp）已成为微调视觉-语言模型的主流轻量化方案，通过学习连续的文本提示向量来适配下游任务，参数量仅数千个。

**现有痛点**：现实标注数据中不可避免地存在噪声标签（标注错误），而提示学习在使用交叉熵（CE）损失时依然会过拟合噪声标签。已有工作（如 Wu et al.）证明提示学习比 adapter 等微调方式更鲁棒，但在高噪声率下仍然性能大幅下降。JoAPR 等后续工作用高斯混合模型区分干净/噪声数据再校正标签，但没有充分利用提示学习的特殊优势。

**核心矛盾**：传统噪声标签学习中 MAE 损失虽然理论上鲁棒，但在常规训练范式下收敛慢、性能差，几乎无人使用。然而在提示学习场景中，情况可能完全不同——这是一个被忽视的研究角度。

**本文目标** (1) 解释为什么 MAE 在提示学习中能同时保持鲁棒性和高准确率；(2) 如何进一步提升噪声条件下的提示学习性能。

**切入角度**：作者实验发现了一个有趣现象——在提示学习中使用 MAE 损失（PromptMAE）不仅鲁棒性远超 CE 损失，而且收敛速度和最终准确率也不逊色。这违背了 MAE 在传统训练中的表现规律。

**核心 idea**：利用 MAE 损失在提示学习中的独特鲁棒性优势，配合基于视觉-语言模型文本特征的最优传输数据净化，实现噪声标签下的鲁棒提示学习。

## 方法详解

### 整体框架

NLPrompt 包含两个核心组件：(1) PromptMAE——直接用 MAE 替代 CE 作为提示学习的训练损失；(2) PromptOT——利用 CLIP 文本编码器的文本特征作为类原型，通过最优传输算法将数据集划分为干净子集和噪声子集，对干净子集用 CE 损失、噪声子集用 MAE 损失训练。输入是带噪声标签的图像数据集和 CLIP 模型，输出是学到的鲁棒文本提示。

### 关键设计

1. **PromptMAE——MAE 损失的提示学习**:

    - 功能：直接在提示学习中用 MAE 替代 CE 损失，提升噪声鲁棒性
    - 核心思路：MAE 损失定义为 $\ell_{\text{MAE}} = \sum_{c=1}^{C} |y_{i,c} - s_{i,c}|$，其中 $s_{i,c}$ 是 softmax 后的相似度。作者通过特征学习理论证明，在提示学习中 MAE 能抑制噪声样本对任务相关特征系数的负面影响。具体来说，可学习提示可以分解为任务相关特征 $\mu$ 和任务无关特征 $\xi_l$ 的线性组合，MAE 损失下噪声样本导致的任务相关系数 $\beta$ 衰减速度远小于 CE 损失。理论上以高概率 $1-d^{-1}$ 证明了 PromptMAE 的测试损失低于 PromptCE。
    - 设计动机：MAE 在传统深度学习中因收敛慢而不实用，但提示学习只更新极少参数（数千），且预训练特征空间已对齐，使得 MAE 的鲁棒优势得以发挥而收敛劣势被消除

2. **PromptOT——基于最优传输的数据净化**:

    - 功能：将数据集划分为干净子集和噪声子集
    - 核心思路：传统 OT 方法用随机初始化的原型构建传输矩阵，而 PromptOT 利用 CLIP 文本编码器生成的文本特征 $\mathbf{T} \in \mathbb{R}^{C \times d}$ 作为类原型，计算与图像特征 $\mathbf{I} \in \mathbb{R}^{N \times d}$ 的相似度矩阵，取负对数作为代价矩阵，通过 Sinkhorn 算法求解带熵正则化的 OT 问题，得到传输矩阵 $\mathbf{Q}^*$。对每列取 argmax 得到伪标签 $\hat{y}_i$，若 $\hat{y}_i = \tilde{y}_i$ 则为干净样本，否则为噪声样本。
    - 设计动机：视觉-语言模型的特征空间已预训练对齐，文本特征天然是高质量的类原型，比随机初始化更准确。同时 OT 框架考虑全局分布约束，比逐样本预测更鲁棒

3. **CE + MAE 混合训练策略**:

    - 功能：综合 CE 在干净数据上的优势和 MAE 在噪声数据上的鲁棒性
    - 核心思路：总损失为 $\ell_{\text{NLPrompt}} = \sum_{i \in \mathcal{D}_{\text{clean}}} -\mathbf{y}_i^\top \log \mathbf{s}_i + \sum_{j \in \mathcal{D}_{\text{noisy}}} \|\mathbf{y}_j - \mathbf{s}_j\|_1$，对干净子集用 CE 能获得更快收敛和更高准确率，对噪声子集用 MAE 防止过拟合错误标签
    - 设计动机：单一使用 MAE 虽然鲁棒但在低噪声场景下不如 CE；单一使用 CE 在高噪声下崩溃。分而治之可以兼得两者优势

### 损失函数 / 训练策略

混合损失：干净集用 CE，噪声集用 MAE。PromptOT 在训练开始前一次性执行数据划分（无需迭代更新），整体训练流程极为简洁。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Sym-25% | Sym-50% | Sym-75% | Asym-25% | Asym-50% |
|--------|------|---------|---------|---------|----------|----------|
| Flowers102 | CoOp | 83.50 | 70.10 | 37.17 | 74.70 | 42.60 |
| Flowers102 | GCE | 88.33 | 84.07 | 70.37 | 86.37 | 69.93 |
| Flowers102 | JoAPR | 81.23 | 70.23 | 66.93 | 79.63 | 73.83 |
| Flowers102 | **NLPrompt** | **92.57** | **89.90** | **76.80** | **93.40** | **81.10** |
| OxfordPets | **NLPrompt** | **86.00** | **84.87** | **70.77** | **84.97** | **77.53** |
| StanfordCars | **NLPrompt** | **68.80** | **65.63** | **58.30** | **67.53** | **59.03** |

### 消融实验

| 配置 | Flowers102 (Sym-50%) | Caltech101 (Sym-50%) |
|------|---------------------|---------------------|
| CoOp (CE only) | 70.10 | 70.90 |
| PromptMAE (MAE only) | ~85+ | ~87+ |
| PromptOT + CE | ~86 | ~88 |
| NLPrompt (完整) | **89.90** | **90.70** |

### 关键发现

- PromptMAE 单独使用就已大幅超越 CoOp（在 Flowers102 Sym-50% 下从 70.1% 提升到 ~85%），验证了 MAE 在提示学习中的独特优势
- 在极端噪声率（75%对称噪声）下，NLPrompt 仍能保持相当竞争力，如 Flowers102 上 76.80%，而 CoOp 仅 37.17%
- 非对称噪声对所有方法冲击更大，但 NLPrompt 的优势更加明显
- NLPrompt 在所有 11 个数据集、12 种噪声设置下几乎全面领先，鲁棒性极强

## 亮点与洞察

- **MAE 在提示学习中"复活"**：MAE 在传统训练中因收敛慢而被弃用，但在提示学习中因参数少+预训练对齐而表现出色。这揭示了一个重要规律——损失函数的效果与优化场景强相关，不能简单迁移传统结论
- **文本特征作为 OT 原型**：利用 CLIP 已对齐的文本特征替代随机原型来构建传输矩阵，巧妙地将视觉-语言预训练的优势引入数据净化过程。这个思路可迁移到任何需要类原型的场景
- **理论+实验双重验证**：用特征学习理论严格证明了 MAE 在提示学习中的鲁棒性保证，再用大量实验印证，论证非常完整

## 局限与展望

- PromptOT 的数据划分是一次性的（训练前执行），没有在训练过程中动态更新，可能在边界样本上不够精确
- 理论分析基于二分类和线性特征假设，与实际多类别非线性场景有差距
- 仅在 CLIP-based 模型上验证，是否能推广到其他视觉-语言模型（如 BLIP-2、LLaVA）有待探索
- 没有讨论噪声标签的来源和类型对方法效果的影响（如实例依赖型噪声 vs 均匀噪声）

## 相关工作与启发

- **vs CoOp/CoCoOp**: 标准提示学习方法，使用 CE 损失，在噪声标签下性能急剧下降；NLPrompt 通过替换损失函数和数据净化解决了这一问题
- **vs JoAPR**: JoAPR 用高斯混合模型区分噪声数据并通过 mixup 校正标签，设计复杂且效果一般；NLPrompt 更简洁且性能更优
- **vs GCE (Generalized Cross-Entropy)**: GCE 是传统噪声标签学习中的鲁棒损失，在提示学习中也有一定鲁棒性，但 NLPrompt 通过差异化处理干净/噪声数据进一步拉开差距
- 这篇工作启发我们思考：预训练模型的微调方式是否应该重新审视传统被弃用的技术？

## 评分

- 新颖性: ⭐⭐⭐⭐ 发现 MAE 在提示学习中的独特优势是有洞察力的观察，但方法本身较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 11 数据集 × 12 噪声设置，覆盖极其全面
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验布局合理
- 价值: ⭐⭐⭐⭐ 对提示学习在现实场景（噪声标签）下的应用有重要实践指导意义

<!-- RELATED:START -->

## 相关论文

- [Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noiseaware_fewshot_learning_through_bidirectional.md)
- [VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)

<!-- RELATED:END -->
