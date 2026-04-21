---
title: >-
  [论文解读] Scaling Diffusion Transformers Efficiently via μP
description: >-
  [NeurIPS 2025][图像生成][Transformer] 将 Maximal Update Parametrization (μP) 从标准 Transformer 推广到扩散 Transformer（DiT、PixArt-α、MMDiT 等），证明其超参数可从小模型稳定迁移到大模型，显著降低大规模扩散模型的调参成本。
tags:
  - NeurIPS 2025
  - 图像生成
  - Transformer
  - μP
  - 超参数迁移
  - 模型缩放
  - 高效训练
---

# Scaling Diffusion Transformers Efficiently via μP

**会议**: NeurIPS 2025  
**arXiv**: [2505.15270](https://arxiv.org/abs/2505.15270)  
**代码**: [有](https://github.com/ML-GSAI/Scaling-Diffusion-Transformers-muP)  
**领域**: Image Generation  
**关键词**: 扩散 Transformer, μP, 超参数迁移, 模型缩放, 高效训练

## 一句话总结

将 Maximal Update Parametrization (μP) 从标准 Transformer 推广到扩散 Transformer（DiT、PixArt-α、MMDiT 等），证明其超参数可从小模型稳定迁移到大模型，显著降低大规模扩散模型的调参成本。

## 研究背景与动机

扩散 Transformer 已成为视觉生成模型的基础架构，广泛应用于图像和视频生成。然而随着模型规模增长到数十亿参数，超参数（HP）调优变得极其昂贵，往往阻碍模型发挥全部潜力。

μP 之前已被提出用于标准 Transformer（如 LLM），能使小模型搜索到的最优超参数直接迁移到大模型，大幅降低调参成本。但扩散 Transformer 与标准 Transformer 存在根本差异：

**架构差异**：扩散 Transformer 包含额外组件（adaLN、cross-attention 等）来整合文本和时间步信息

**训练范式差异**：基于迭代去噪的生成框架，不同于自回归范式

因此现有 μP 理论能否直接应用于扩散 Transformer 是一个未解之谜。本文系统性地解决了这一问题。

## 方法详解

### 整体框架

本文的方法论分三步：(1) 理论证明扩散 Transformer 的 μP 形式与标准 Transformer 一致；(2) 验证 DiT-μP 的超参数迁移性；(3) 在大规模文生图任务上验证 μTransfer 的效率。

### 关键设计

1. **μP 的理论推广（Theorem 3.1）**：利用 Tensor Programs 技术，严格证明主流扩散 Transformer（U-ViT、DiT、PixArt-α、MMDiT）的前向传播可以在 Ne⊗or⊤ Program 框架下表示。关键在于证明扩散 Transformer 特有的模块（如 adaLN 块）也能用该框架的三个算子重写。这意味着 μP 的 abc-参数化规则直接适用：

    - 输入权重：$a_W=0, b_W=0, c_W=0$
    - 隐藏权重：$a_W=0, b_W=1/2, c_W=1$（学习率需要 $\eta \cdot n_{base}/n$ 缩放）
    - 输出权重：$a_W=1, b_W=0, c_W=0$（零初始化）

2. **宽度缩放策略**：固定注意力头维度，通过增加头数来缩放宽度。理论上（Bordelon et al., 2024），增大头维度会导致多头注意力退化为单头注意力，丧失注意力模式多样性。实践中也与 LLM 主流做法一致。

3. **μTransfer 算法**：代理模型（小宽度、小 batch、短训练）搜索最优基础超参数，然后直接迁移到目标大模型。代理和目标模型使用相同的 $n_{base}$ 进行 μP 参数化，从而共享相同的最优基础超参数。

### 训练策略

- DiT 实验：固定头维度 72，base width 288（4 头），AdamW 优化器，无学习率调度和权重衰减
- PixArt-α：代理模型 0.04B（4 头），目标模型 0.61B（16 头），代理训练 5 epoch
- MMDiT-18B：代理模型 0.18B（宽度 512），对 4 个基础超参数（学习率、梯度裁剪、REPA 损失权重、warm-up 步数）进行 80 次随机搜索

## 实验关键数据

### 主实验

| 模型 | 方法 | 关键指标 | 调参成本 |
|------|------|----------|----------|
| DiT-XL-2 | 原始 | FID 收敛@7M steps | 基准 |
| DiT-XL-2-μP | μTransfer | FID 收敛@2.4M steps | **2.9× 加速** |
| PixArt-α (30ep) | 原始 | GenEval 0.15, FID(MJHQ) 42.71 | 基准 |
| PixArt-α-μP (30ep) | μTransfer | GenEval 0.26, FID(MJHQ) 29.96 | **5.5% FLOPs** |
| MMDiT-18B | 人工调参 | GenEval 0.8154, 对齐 0.703 | 5× 预训练成本 |
| MMDiT-μP-18B | μTransfer | GenEval 0.8218, 对齐 0.715 | **3% 人工调参成本** |

### 消融实验

| 配置 | 关键发现 |
|------|----------|
| 学习率跨宽度迁移 | 最优基础学习率 $2^{-10}$ 在宽度 144-1152 间稳定迁移 |
| 学习率跨 batch 迁移 | 256-1024 batch size 共享最优学习率 |
| 学习率跨训练步数迁移 | 150K-400K steps 共享最优学习率 |
| MMDiT 超参数搜索 | 学习率影响最大；最优梯度裁剪为 1（非传统的 0.1）；warm-up 影响可忽略 |

### 关键发现

- PixArt-α-μP 在长训练后不易过拟合，而原始 PixArt-α 在 20 epoch 后性能下降，表明 μP 可能增强泛化能力
- μP 倾向于较大学习率（接近最大稳定值），与"大学习率带来有益梯度噪声"的理论一致
- MMDiT 单 epoch 训练中最优学习率不再接近最大稳定值，与多 epoch 训练行为不同

## 亮点与洞察

- **理论严谨性强**：不是经验性地将 μP 应用到扩散 Transformer，而是从 Tensor Programs 框架出发给出严格证明
- **实用价值巨大**：MMDiT-18B 调参成本仅为人工调参的 3%，节省的计算资源极为可观
- **意外发现**：PixArt-α 和 DiT 的最优基础学习率相同（$2^{-10}$），暗示最优基础 HP 可能具有跨数据集/架构的迁移性

## 局限与展望

- 未确定最优代理任务规模（代理模型多大、训练多少数据足够）
- 仅验证了固定头维度、缩放头数的方案，未探索缩放头维度
- 可扩展到线性 Transformer、MoE 等更高效架构
- 可应用于 Muon 优化器、warmup-stable-decay 学习率调度等更先进的训练策略

## 相关工作与启发

本文建立了 μP 在视觉生成领域的理论和实践基础。对于大规模扩散模型训练（如 Sora 级别的视频模型），μP 提供了一种科学的超参数搜索策略：先在小代理上搜索，再零样本迁移到大模型，避免了大模型上代价高昂的反复试错。

## 评分

- 新颖性: ⭐⭐⭐⭐ （μP 本身已有，贡献在于理论推广+大规模验证）
- 实验充分度: ⭐⭐⭐⭐⭐ （从 DiT 到 MMDiT-18B，逐步验证）
- 写作质量: ⭐⭐⭐⭐⭐ （条理清晰，理论与实验配合紧密）
- 价值: ⭐⭐⭐⭐⭐ （对工业界大模型训练有直接指导意义）

<!-- RELATED:START -->

## 相关论文

- [Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations](unleashing_diffusion_transformers_for_visual_correspondence_by_modulating_massiv.md)
- [Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [Scaling Can Lead to Compositional Generalization](scaling_can_lead_to_compositional_generalization.md)
- [OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers](omnisync_towards_universal_lip_synchronization_via_diffusion.md)
- [Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance](../../ICLR2026/image_generation/routing_matters_in_moe_scaling_diffusion_transformers_with_explicit_routing_guid.md)

<!-- RELATED:END -->
