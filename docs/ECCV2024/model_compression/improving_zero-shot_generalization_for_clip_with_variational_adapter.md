---
title: >-
  [论文解读] Improving Zero-Shot Generalization for CLIP with Variational Adapter
description: >-
  [ECCV 2024][模型压缩][CLIP] 提出 Prompt-based Variational Adapter (PVA)，通过变分适配器将 base 和 novel 类别样本在隐空间中分离，采用分治策略分别处理，结合残差连接增强 novel 类别的迁移能力，在广义零样本学习和跨数据集迁移学习基准上达到 SOTA。
tags:
  - ECCV 2024
  - 模型压缩
  - CLIP
  - 零样本泛化
  - 变分适配器
  - 分治策略
  - 视觉语言模型
---

# Improving Zero-Shot Generalization for CLIP with Variational Adapter

**会议**: ECCV 2024  
**arXiv**: N/A  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: CLIP, 零样本泛化, 变分适配器, 分治策略, 视觉语言模型

## 一句话总结

提出 Prompt-based Variational Adapter (PVA)，通过变分适配器将 base 和 novel 类别样本在隐空间中分离，采用分治策略分别处理，结合残差连接增强 novel 类别的迁移能力，在广义零样本学习和跨数据集迁移学习基准上达到 SOTA。

## 研究背景与动机

**领域现状**：预训练视觉-语言模型（VLMs）如 CLIP 展示了出色的零样本泛化能力。当前主流做法是通过 prompt tuning 或 adapter 对 CLIP 进行下游任务微调，在 base 类别上获得较好的专业性能，同时保持对 novel（未见类别）的泛化能力。

**现有痛点**：大多数现有微调方法在追求 base 类别高性能的同时，不可避免地导致 novel 类别的"特征混淆"（feature confusion）。具体来说，微调后的模型将 novel 类别的样本与某些 base 类别混淆，因为在微调过程中模型的特征空间被扭曲，使得 novel 类别的语义边界变得模糊。这种 base-novel 的性能权衡（trade-off）一直是零样本泛化的核心难题。

**核心矛盾**：微调使模型在 base 类别上更专业化，但这种专业化以牺牲 novel 类别的通用性为代价。根本原因在于微调在统一的特征空间中同时处理 base 和 novel 的预测，导致模型将 novel 样本的特征错误地拉向 base 类别的决策区域。

**本文目标**（1）如何识别并分离混淆的 base 和 novel 样本？（2）如何在分离后分别处理两组样本以消除预测偏差？（3）如何在保持 base 类别性能的同时增强 novel 类别的迁移能力？

**切入角度**：作者观察到 base 和 novel 样本在潜在表示空间中的分布差异可以被建模——变分推断可以学习一个结构化的隐空间，其中两类样本的分布可以被显式区分。如果能将混淆样本分离开来，就可以将一个困难的混合任务拆解为两个独立且更简单的子任务。

**核心 idea**：用变分适配器学习结构化潜在空间，基于隐特征的相似度将 novel 样本从混淆空间中分离出来，然后分治处理。

## 方法详解

### 整体框架

PVA 在冻结的 CLIP 模型上添加两个轻量级的变分适配器——一个用于视觉模态，一个用于文本模态。每个适配器包含可学习的文本 tokens 和变分推断模块。训练阶段在 base 类别数据上训练适配器，学习将输入特征映射到共享的结构化隐空间。推理阶段，利用隐空间中的特征相似度度量将输入样本分为"类似 base"和"类似 novel"两组，然后分别用不同的预测策略处理。

### 关键设计

1. **双模态变分适配器（Variational Adapters）**:

    - 功能：将视觉和文本特征对齐到共享的结构化隐空间
    - 核心思路：每个适配器由一个编码器网络和可学习的文本 prompt tokens 组成。编码器将输入特征（CLIP 的视觉或文本特征）映射到隐空间的均值 $\mu$ 和方差 $\sigma^2$，然后通过重参数化技巧采样得到隐表示 $z = \mu + \sigma \cdot \epsilon$。两个模态的隐表示在同一空间中对齐。可学习的 prompt tokens 嵌入到 CLIP 的文本编码器中，为适配器提供任务相关的语义引导。
    - 设计动机：变分推断允许模型学习隐空间中的概率分布而非确定性映射。这种概率性建模天然提供了不确定性估计——novel 类别样本在隐空间中的分布与 base 类别不同，可以被检测和分离。共享隐空间确保了跨模态的对齐。

2. **Base-Novel 样本分离机制**:

    - 功能：在推理时将输入样本分为 base-like 和 novel-like 两组
    - 核心思路：训练完成后，计算每个测试样本的隐表示与所有 base 类别隐表示原型的相似度。如果相似度高于阈值，则该样本被判定为 base-like；否则为 novel-like。相似度度量使用隐空间中的余弦距离或 KL 散度。分离后，base-like 样本用微调后的分类器预测，novel-like 样本用保持泛化能力的分类器预测。
    - 设计动机：这种"先分后治"的策略将一个困难的混合分类问题拆解为两个相对简单的子问题。对于 base-like 样本可以充分利用微调的专业性，对于 novel-like 样本则可以保留 CLIP 原始的泛化能力。

3. **残差连接的特征增强**:

    - 功能：提升 novel 类别的迁移表现
    - 核心思路：对于 novel-like 样本，将适配器输出的隐特征与 CLIP 原始的全局特征通过残差连接进行融合：$f_{out} = f_{adapter} + \lambda \cdot f_{CLIP}$。$\lambda$ 控制残差连接的强度。这使得 novel 类别的预测既包含适配器学到的结构化信息，也保留了 CLIP 预训练的强大零样本表示。
    - 设计动机：完全依赖适配器可能丢失 CLIP 预训练学到的通用语义信息；完全依赖 CLIP 原始特征则浪费了适配器学到的下游任务知识。残差连接在两者之间取得平衡。

### 损失函数 / 训练策略

训练损失包含：（1）交叉熵分类损失 $L_{CE}$，用于 base 类别分类学习；（2）KL 散度正则化 $L_{KL}$，约束隐空间分布接近先验（标准正态分布）；（3）跨模态对齐损失，确保视觉和文本隐表示的一致性。只在 base 类别数据上训练，novel 类别数据在训练中完全不可见。

## 实验关键数据

### 主实验

| 基准数据集 | 指标 | PVA | CoCoOp | MaPLe | 提升 |
|-----------|------|-----|--------|-------|------|
| ImageNet (Base) | Acc | 77.5 | 75.98 | 76.66 | +0.84 |
| ImageNet (Novel) | Acc | 71.8 | 70.43 | 70.54 | +1.26 |
| ImageNet (HM) | Acc | 74.5 | 73.10 | 73.47 | +1.03 |
| 11数据集平均 (Base) | Acc | 80.2 | 79.74 | 80.18 | +0.02 |
| 11数据集平均 (Novel) | Acc | 74.6 | 71.69 | 72.46 | +2.14 |
| 11数据集平均 (HM) | Acc | 77.3 | 75.83 | 76.18 | +1.12 |

### 消融实验

| 配置 | Base Acc | Novel Acc | HM | 说明 |
|------|----------|-----------|-----|------|
| Full PVA | 77.5 | 71.8 | 74.5 | 完整模型 |
| w/o 变分推断 | 77.2 | 69.5 | 73.2 | 改为确定性映射，Novel 明显下降 |
| w/o 分离机制 | 77.8 | 68.9 | 73.1 | 统一处理所有样本，Novel 大幅下降 |
| w/o 残差连接 | 76.8 | 70.1 | 73.3 | Novel 小幅下降 |

### 关键发现

- 分离机制是最关键的设计。去掉分离机制后 Novel Acc 下降约 2.9%，说明"分治"策略对解决 base-novel 偏差问题至关重要。
- 变分推断比确定性映射更优，说明概率性隐空间对检测 novel 样本有明显帮助。
- PVA 在 novel 类别上的提升比 base 类别更显著（+2.14 vs +0.02），表明方法确实解决了 novel 类别的特征混淆问题。
- 在跨数据集迁移（从 ImageNet 迁移到其他 10 个数据集）中也表现优异，说明方法的泛化性。

## 亮点与洞察

- **"分治"思想在零样本学习中的创新应用**：将混淆的 base 和 novel 样本显式分离再分别处理，这个思路简单但有效，比端到端地试图平衡 base-novel 更直接。这种策略可以迁移到其他需要处理已知-未知类别共存的场景，如开放集识别、持续学习。
- **变分推断作为 novel 检测器**：巧妙利用变分推断的概率建模能力来检测 novel 样本——训练时只见过 base 类别，novel 样本在隐空间中自然呈现不同的分布特征。这比设定固定阈值或使用额外的 OOD 检测模块更优雅。
- **与 prompt tuning 的互补性**：PVA 的核心不是学更好的 prompt，而是学更好的推理策略（分离 + 分治），这与现有 prompt tuning 方法形成互补。

## 局限与展望

- 分离阈值的选择可能需要验证集调优，在完全零样本的场景中可能不太实际。
- 方法假设 novel 类别样本与 base 类别在隐空间中有足够的分布差异，对于高度相似的 base-novel 类别对可能失效。
- 两个变分适配器增加了模型参数和训练复杂度，虽然适配器本身较轻量，但变分推断的采样过程可能影响推理速度。
- 评估仅限于分类任务，未在检测、分割等下游任务上验证。

## 相关工作与启发

- **vs CoCoOp**: CoCoOp 通过条件化可学习 prompt 来增强泛化性，但仍在统一空间中处理 base 和 novel。PVA 的显式分离策略更直接地解决了偏差问题。
- **vs PLOT**: PLOT 使用最优传输对齐文本和视觉 prompt。PVA 使用变分推断做对齐，额外获得了概率性分离能力。
- **vs Tip-Adapter**: Tip-Adapter 构建了缓存模型作为适配器。PVA 的变分适配器学习了更结构化的表示空间。

## 评分

- 新颖性: ⭐⭐⭐⭐ 分治策略解决 base-novel 偏差的思路新颖，变分适配器的设计有创意
- 实验充分度: ⭐⭐⭐⭐ 覆盖了广义零样本和跨数据集迁移两个设定
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法动机充分
- 价值: ⭐⭐⭐⭐ 对 CLIP 微调中的 base-novel trade-off 问题提供了实用解决方案

<!-- RELATED:START -->

## 相关论文

- [Logits DeConfusion with CLIP for Few-Shot Learning](../../CVPR2025/model_compression/logits_deconfusion_with_clip_for_few-shot_learning.md)
- [Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](../../NeurIPS2025/model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)
- [Boomerang Distillation Enables Zero-Shot Model Size Interpolation](../../ICLR2026/model_compression/boomerang_distillation_enables_zero-shot_model_size_interpolation.md)
- [Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)
- [Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](../../ICLR2026/model_compression/scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)

<!-- RELATED:END -->
