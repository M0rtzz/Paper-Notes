---
title: >-
  [论文解读] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory
description: >-
  [ICCV 2025][模型压缩][parameter-efficient fine-tuning] 本文发现预训练 ViT 权重矩阵的行/列向量具有近似正交性，而 LoRA/Adapter 的投影矩阵不具备此性质；提出 AOFT 策略，用单个可学习向量生成近似正交的下/上投影矩阵，使其与骨干网络性质对齐，从而降低泛化误差上界，在 FGVC 和 VTAB-1k 上用更少参数达到竞争性能。
tags:
  - ICCV 2025
  - 模型压缩
  - parameter-efficient fine-tuning
  - 近似正交
  - LoRA
  - Adapter
  - Transformer
---

# Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory

**会议**: ICCV 2025  
**arXiv**: [2507.13260](https://arxiv.org/abs/2507.13260)  
**代码**: [Google Drive](https://drive.google.com/file/d/1rg3JYfkmeLGDbRWXspO22wxVspbtnthV/view?usp=drive_link)  
**领域**: 模型压缩  
**关键词**: parameter-efficient fine-tuning, 近似正交, LoRA, Adapter, Vision Transformer

## 一句话总结
本文发现预训练 ViT 权重矩阵的行/列向量具有近似正交性，而 LoRA/Adapter 的投影矩阵不具备此性质；提出 AOFT 策略，用单个可学习向量生成近似正交的下/上投影矩阵，使其与骨干网络性质对齐，从而降低泛化误差上界，在 FGVC 和 VTAB-1k 上用更少参数达到竞争性能。

## 研究背景与动机
参数高效微调（PEFT）已成为适配大规模预训练 ViT 到下游任务的主流范式。LoRA 和 Adapter 等方法通过学习低秩的下投影-上投影矩阵来近似权重增量，仅需更新少量参数。

作者通过仔细分析预训练 ViT 的权重矩阵 $\mathbf{W}_q, \mathbf{W}_v$ 等，观察到一个重要且此前未被充分利用的现象：

**预训练骨干矩阵的行/列向量之间呈现近似正交性**——角度分布集中在 90° 附近

**LoRA/Adapter 训练出的下/上投影矩阵不具备此性质**——角度分布分散，远非正交

正交性在数学上意味着向量间的独立性，从泛化理论的角度看，正交的权重矩阵具有更小的 L2 范数，进而降低了 Rademacher 复杂度给出的泛化误差上界。

核心问题：如果让投影矩阵也具备近似正交性，能否提升微调后模型的泛化能力？AOFT 给出了肯定回答。

## 方法详解

### 整体框架
AOFT 是一种通用的投影矩阵替代策略，可以插入到 LoRA、Adapter、VPT 等现有 PEFT 框架中。核心思路是用单个可学习向量 $\vec{q} \in \mathbb{R}^N$ 生成一个近似正交矩阵 $\mathbf{Q} \in \mathbb{R}^{N \times N}$，然后从中取前 $d$ 列作为下/上投影矩阵。

### 关键设计
1. **近似正交矩阵生成**

    - 功能：用一个向量 $\vec{q} = (q_0, q_1, \cdots, q_N)^\top$ 构造正交矩阵 $\mathbf{Q}$
    - 核心思路：$\mathbf{Q}$ 的构造基于 Householder 变换的推广形式。矩阵 $\mathbf{Q}$ 的第 $(i,j)$ 元素为：
        - 第一行：$q_0, -q_1, -q_2, \cdots, -q_N$
        - 其余：对角元素 $1 - \frac{q_i q_i}{1+q_0}$，非对角元素 $-\frac{q_j q_i}{1+q_0}$
    - 当满足归一化约束 $\sum_{i=1}^N |q_i|^2 = 1$ 时，$\mathbf{Q}$ 严格正交
    - **关键放松**：不严格施加此归一化，使列向量保持"近似"正交，增强模型灵活性
    - 操作定义：$\text{AO}(\vec{q}) = \mathbf{Q}[:, 0:d]$，取前 $d$ 列

2. **AOFT 与不同 PEFT 方法的结合**

    - **LoRA + AOFT**：$\mathbf{X}_{FT}^{(l-1)} = \mathbf{X}^{(l-1)}(\mathbf{W}^{(l)} + \text{AO}(\vec{q}_{down}) \cdot \text{AO}(\vec{q}_{up})^\top)$
    - **Adapter + AOFT**：分别在 MHA 和 FFN 后添加 $\text{AO}(\vec{q}_{down}^{MHA}) \cdot \text{AO}(\vec{q}_{up}^{MHA})^\top$
    - **VPT + AOFT**：用近似正交矩阵替代 prompt tokens
    - 设计动机：由于 AOFT 不随 bottleneck 维度增加引入更多参数（仅需一个 $N$ 维向量），可以灵活调整 bottleneck 大小

3. **AOFT* 变体：可学习缩放**

    - 功能：引入可学习缩放向量 $\vec{\lambda}$ 进一步增强灵活性
    - 实现：$(\mathbf{W}_{down} \odot \vec{\lambda}^\top) \mathbf{W}_{up}$
    - 提供对每个秩分量的独立缩放控制

### 泛化误差理论分析
通过 Rademacher 复杂度分析泛化误差上界：

$$\mathbb{E}\left[\frac{1}{m} \sup_{\|\mathbf{W}\| \leq \gamma} \left\| \sum_{i=1}^m \xi_i \mathbf{W} \vec{x}_i \right\|\right] \leq \frac{\gamma}{m} \mathbb{E}\left[\left\| \sum_{i=1}^m \xi_i \vec{x}_i \right\|\right]$$

其中 $\gamma$ 是权重矩阵的 L2 范数。AOFT 的投影矩阵 L2 范数显著小于 LoRA/Adapter，因此泛化误差上界更低。

## 实验关键数据

### 主实验
FGVC 基准（5 个数据集，ViT-B/16）：

| 方法 | CUB-200 | NABirds | Flowers | Dogs | Cars | 均值 | 参数(M) |
|------|---------|---------|---------|------|------|------|---------|
| Full fine-tuning | 87.3 | 82.7 | 98.8 | 89.4 | 84.5 | 88.5 | 85.98 |
| Adapter | 87.1 | 84.3 | 98.5 | 89.8 | 68.6 | 85.7 | 0.41 |
| **Adapter+AOFT*** | **89.0** | **84.5** | **99.5** | **92.0** | **85.2** | **90.1** | **0.20** |
| LoRA | 88.3 | 85.6 | 99.2 | 91.0 | 83.2 | 89.5 | 0.44 |
| **LoRA+AOFT** | **88.8** | 84.2 | **99.4** | **92.0** | **85.1** | **89.9** | **0.22** |
| VPT-Deep | 88.5 | 84.2 | 99.0 | 90.2 | 83.6 | 89.1 | 0.85 |
| **VPT-Deep+AOFT** | **88.7** | 82.8 | **99.5** | **91.5** | 84.1 | **89.5** | **0.15** |

### 消融实验
VTAB-1k 基准（19 个数据集，3 组，ViT-B/16 部分结果）：

| 方法 | Natural均值 | Specialized均值 | Structured均值 | 总均值 | 参数(M) |
|------|------------|----------------|---------------|--------|---------|
| Full fine-tuning | 75.9 | 83.4 | 47.6 | 65.6 | 85.80 |
| Adapter | 79.0 | 84.1 | 58.5 | 71.4 | 0.16 |
| Adapter+AOFT | 79.3 | 84.2 | 60.6 | 72.5 | 0.06 |
| Adapter+AOFT* | 81.4 | 83.9 | 59.4 | 72.7 | 0.06 |
| LoRA | 79.5 | 84.9 | - | - | - |
| VPT-Deep+AOFT | 80.3 | 84.7 | 55.4 | 70.7 | 0.05 |

### 关键发现
- **参数效率显著提升**：Adapter+AOFT 使用 0.20M 参数超过原始 Adapter 的 0.41M，精度更高（90.1 vs 85.7）
- **泛化能力验证**：AOFT 施加后投影矩阵的列向量角度分布集中于 90° 附近，与预训练骨干一致
- **L2 范数显著降低**：AOFT 的投影矩阵 L2 范数远小于原始 LoRA/Adapter，理论预测的泛化优势得到实证支持
- **灵活的 bottleneck 调整**：AOFT 不增加参数量（仅需一个向量），可为不同任务自适应设置 bottleneck 维度
- **跨框架通用性**：LoRA、Adapter、VPT 三种 PEFT 框架均获益

## 亮点与洞察
- **从观察到理论再到方法的完整链条**：发现正交性现象 → Rademacher 复杂度理论分析 → AOFT 方法设计 → 实验验证，研究逻辑极为清晰
- **"一个向量生成一个矩阵"**：这个极简设计大幅减少了参数量，同时保持了足够的表达力
- **对 PEFT 方法的通用增强**：AOFT 作为 plug-and-play 模块可提升多种 PEFT 方法
- **不强制严格正交**：放松归一化约束让模型在正交性和灵活性之间取得平衡，这种设计选择体现了工程智慧

## 局限与展望
- 正交矩阵的构造依赖特定的数学形式（Householder 变换推广），是否存在更优的构造方式值得探索
- LoRA+AOFT 在某些数据集（如 NABirds）上性能略低于原始 LoRA，说明近似正交约束可能在部分场景下过于严格
- 仅在图像分类任务上验证，未扩展到检测、分割等密集预测任务
- 与 OFT、BOFT 等同样使用正交变换的方法相比，理论优势的差异未充分分析

## 相关工作与启发
- OFT 通过正交变换保持预训练语义，AOFT 则从泛化误差角度引入正交性，视角不同
- 本文的发现（预训练权重矩阵的近似正交性）可能对理解大模型训练动态有理论价值
- 单向量生成矩阵的思想可能启发其他需要结构化参数的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 近似正交性的观察和单向量生成正交矩阵的思路有独到之处
- 实验充分度: ⭐⭐⭐⭐ FGVC + VTAB-1k 24 个数据集，3 种 PEFT 框架，但缺少密集预测实验
- 写作质量: ⭐⭐⭐ 理论分析部分稍显冗长，符号标注可更简洁
- 价值: ⭐⭐⭐⭐ 提供了 PEFT 方法的通用增强策略，理论与实践价值兼具

<!-- RELATED:START -->

## 相关论文

- [EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/model_compression/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)
- [BHViT: Binarized Hybrid Vision Transformer](../../CVPR2025/model_compression/bhvit_binarized_hybrid_vision_transformer.md)

<!-- RELATED:END -->
