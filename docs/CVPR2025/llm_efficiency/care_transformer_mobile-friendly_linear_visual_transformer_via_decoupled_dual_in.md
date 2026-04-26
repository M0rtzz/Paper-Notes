---
title: >-
  [论文解读] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction
description: >-
  [CVPR 2025][LLM效率][线性注意力] 本文提出CARE（deCoupled duAl-interactive lineaR attEntion）机制，通过非对称特征解耦策略将局部归纳偏置和长程依赖的学习过程分而治之，配合动态记忆单元和双交互模块充分利用跨特征互补性，在ImageNet-1K上以0.7/1.9 GMACs达到78.4/82.1% top-1精度，在移动端实现极低延迟。
tags:
  - CVPR 2025
  - LLM效率
  - 线性注意力
  - 非对称解耦
  - 双交互模块
  - 移动端部署
  - 动态记忆单元
---

# CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction

**会议**: CVPR 2025  
**arXiv**: [2411.16170](https://arxiv.org/abs/2411.16170)  
**代码**: https://github.com/zhouyuan888888/CARE-Transformer  
**领域**: LLM效率 / 高效视觉模型  
**关键词**: 线性注意力, 非对称解耦, 双交互模块, 移动端部署, 动态记忆单元

## 一句话总结

本文提出CARE（deCoupled duAl-interactive lineaR attEntion）机制，通过非对称特征解耦策略将局部归纳偏置和长程依赖的学习过程分而治之，配合动态记忆单元和双交互模块充分利用跨特征互补性，在ImageNet-1K上以0.7/1.9 GMACs达到78.4/82.1% top-1精度，在移动端实现极低延迟。

## 研究背景与动机

**领域现状**：设计高效的线性复杂度视觉Transformer是当前的研究热点，主要有两条路线：(1) 通过局部注意力限制感受野来降低复杂度，但牺牲了全局建模能力；(2) 使用线性注意力通过核技巧去除softmax并改变计算顺序，在保持全局感受野的同时实现线性复杂度。

**现有痛点**：线性注意力虽然理论复杂度为线性，但存在高熵问题——由于不显式计算token间的关系，模型难以抑制不相关token的影响。最近MLLA工作提出用"堆叠式"局部增强来缓解，即交替使用深度卷积和线性注意力。但堆叠方式存在效率瓶颈（输入必须经过所有操作）和灵活性不足（局部和全局信息的融合方式被固定）两个问题。

**核心矛盾**：线性注意力需要局部归纳偏置来弥补其高熵缺陷，但堆叠整合方式在效率和灵活性上都不理想。核心问题是：能否同时提升线性视觉Transformer的效率和精度？

**本文目标** (1) 找到比堆叠更好的方式来整合局部归纳偏置和长程全局信息；(2) 在提高精度的同时降低计算开销，使模型适合移动端部署。

**切入角度**：作者将堆叠式整合分解为两个独立步骤——"学习"和"融合"。学习步骤通过通道维度的特征解耦来独立、并行地学习局部和全局信息；融合步骤通过专门设计的交互模块来充分利用两种信息的互补性。非对称解耦（$d_1 < d_2$）可以进一步降低线性注意力对通道维度的二次开销。

**核心 idea**：通过非对称通道解耦将局部-全局信息学习分而治之，再用双交互模块和动态记忆充分融合，实现效率和精度的同步提升。

## 方法详解

### 整体框架

CARE Transformer采用标准的四阶段金字塔结构，每个阶段由若干CARE block组成。每个CARE block流程：输入特征 → 通道维度非对称切分 → 少量通道用线性注意力学全局依赖 / 多量通道用深度卷积学局部偏置 → Inter1融合局部和全局特征 → Inter2融合当前特征和动态记忆 → 输出特征+更新记忆。Stem层用4×4卷积，stride=4。

### 关键设计

1. **非对称特征解耦策略（Asymmetrical Feature Decoupling）**:

    - 功能：将局部归纳偏置和长程依赖的学习过程在通道维度上分离，减少冗余计算
    - 核心思路：将输入特征 $\mathbf{X} \in \mathbb{R}^{hw \times d}$ 在通道维度切分为 $\bar{\mathbf{X}} \in \mathbb{R}^{hw \times d_1}$（送入线性注意力）和 $\tilde{\mathbf{X}} \in \mathbb{R}^{hw \times d_2}$（送入深度卷积），其中 $d_1 + d_2 = d$ 且 $d_1 = \frac{1}{2}d_2$。关键洞察：线性注意力的复杂度为 $O(hw \cdot d_1^2)$，非对称设置可进一步削减二次开销。论文给出了严格的数学证明：当 $\Delta = d_2 - d_1 > 0$ 时，$\Omega(\Delta) < \Omega(0)$
    - 设计动机：堆叠方式要求所有通道经过全部操作，而解耦后每部分只经过对应操作。分配更多通道给计算便宜的卷积、更少通道给开销更大的注意力。局部偏置也采用解耦——分别用3×3和7×7深度卷积处理，实现多尺度感受野

2. **动态记忆单元（Dynamic Memory Unit）**:

    - 功能：沿网络管线保持和传递关键信息，支持跨层特征交互
    - 核心思路：每个阶段维护一个记忆张量 $\mathbf{Z} \in \mathbb{R}^{hw \times d'}$（$d' = d_1$）。阶段初始化：$\mathbf{Z}_0^s = \text{CONV}_{2\times2}(\mathbf{X}_{-1}^{s-1} \oplus \mathbf{Z}_{-1}^{s-1}, \text{stride}=2)$，通过拼接上一阶段的特征和记忆经卷积下采样得到。后续block中记忆通过Inter2持续更新，使浅层低级特征可以被"回放"并与深层高级特征交互
    - 设计动机：不同层的特征互补（浅层有细节纹理，深层有高级语义），通过记忆单元显式保存和传递跨层信息

3. **双交互模块（Dual Interaction Module）**:

    - 功能：充分利用局部-全局特征和跨层特征之间的互补性
    - 核心思路：两个交互块，结构相同 $\text{Inter}_*(\mathbf{x}, \mathbf{y}) = \text{CONV}_{1\times1,3\times3,1\times1}(\text{Norm}(\mathbf{x} \oplus \mathbf{y}))$，expansion=4。Inter1融合局部和全局特征，Inter2进一步融合Inter1输出和上一个block的记忆：$\mathbf{X}_t^s, \mathbf{Z}_t^s = \text{Inter2}(\text{Inter1}(\bar{\mathbf{X}}_t^s, \tilde{\mathbf{X}}_t^s), \mathbf{Z}_{t-1}^s)$
    - 设计动机：解耦后的局部和全局特征需要专门的融合机制来充分交换信息。双交互设计让模型既能融合同层的局部-全局信息，又能利用历史层信息

### 损失函数 / 训练策略

- ImageNet-1K上训练300 epochs，AdamW优化器，lr=$2 \times 10^{-3}$
- 12张RTX 4090 GPU，batch size 128/GPU
- 数据增强：RandomResizedCrop、RandAugment、Mixup、CutMix、Random Erasing
- 正则化：Label Smoothing、Stochastic Depth、Weight Decay
- 三个变体：CARE-S0 (0.7 GMACs, 78.4%)、CARE-S1 (1.2 GMACs, 80.1%)、CARE-S2 (1.9 GMACs, 82.1%)
- 前两个阶段不用线性注意力，用1×11和11×1深度卷积代替（浅层噪声多）

## 实验关键数据

### 主实验

ImageNet-1K分类：

| 方法 | 类型 | GMACs | iPhone13延迟 | Top-1 Acc |
|------|------|-------|-------------|-----------|
| MobileViG-T | GNN+CONV | ~0.7 | - | 75.7% |
| FastViT-T8 | SA+CONV | ~0.7 | - | 76.7% |
| **CARE-S0** | LA+CONV | **0.7** | **1.1ms** | **78.4%** |
| EdgeViT-S | SA+CONV | ~1.9 | - | 81.0% |
| MobileOne-S3 | CONV | ~1.9 | - | 78.1% |
| **CARE-S2** | LA+CONV | **1.9** | **2.0ms** | **82.1%** |
| MLLA-T | LA+CONV | ~4.2 | 5.0ms | 83.5% |

### 消融实验

非对称解耦策略（CARE-S2）：

| 设置 | GMACs | iPhone13 | Top-1 |
|------|-------|----------|-------|
| w/ Sta (堆叠式) | 更高 | 2.6ms | 81.4% |
| w/o Local (无局部偏置) | 更高 | 更慢 | <82% |
| **w/ Asym (非对称解耦)** | **最低** | **2.0ms** | **82.1%** |

双交互模块（CARE-S0）：

| Inter1 | Inter2 | Memory | GMACs | Top-1 |
|--------|--------|--------|-------|-------|
| ✗ | ✓ | ✓ | 0.5 | 76.2% |
| ✓ | ✗ | ✓ | 0.3 | 70.9% |
| ✓ | ✓ | ✗ | 0.5 | 76.5% |
| **✓** | **✓** | **✓** | **0.7** | **78.4%** |

### 关键发现

- 非对称解耦同时优于堆叠式和对称解耦——效率更高且精度不降
- 双交互模块两个部分都至关重要：去除Inter1掉2.2%，去除Inter2掉7.5%
- 动态记忆单元贡献1.9%精度提升
- CARE-S2仅用MLLA-T一半GMACs，在iPhone13上快2.5倍，精度差距仅1.4%

## 亮点与洞察

- **分而治之的哲学**：将"学习"和"融合"解耦为两个独立步骤，比堆叠式更灵活高效
- **非对称设计有理论支撑**：严格数学推导证明了非对称设置的复杂度优势
- **动态记忆的跨层信息传递**：比ResNet跳连接更灵活、比DenseNet特征拼接更高效
- **移动端部署实测**：在iPhone13和iPad Pro上实测延迟，更有实际参考价值
- **与MLLA的关系清晰**：作为MLLA的改进，分析堆叠方式不足来引出动机

## 局限与展望

- 未使用NAS搜索最优架构配置
- 受GPU限制未验证大模型效果
- 非对称比例 $d_1 = \frac{1}{2}d_2$ 是固定的，可探索自适应比例
- 可以与知识蒸馏结合进一步压缩

## 相关工作与启发

- **MLLA**：直接前身，提出堆叠式局部增强的线性注意力
- **FLatten Transformer**：聚焦线性注意力
- **SLAB**：在FLatten基础上引入重参数化BatchNorm加速
- 启发：特征解耦+专门融合的范式可推广到其他混合注意力架构

## 评分

- 新颖性: ⭐⭐⭐⭐（非对称解耦+双交互+动态记忆的组合设计新颖，有理论支撑）
- 实验充分度: ⭐⭐⭐⭐⭐（ImageNet、COCO、ADE20K三大基准+移动端延迟+详细消融）
- 写作质量: ⭐⭐⭐⭐（逻辑清晰、公式严谨）
- 价值: ⭐⭐⭐⭐（对移动端高效视觉模型有实际推动意义）

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Associative Transformer](associative_transformer.md)
- [\[CVPR 2025\] Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)
- [\[CVPR 2025\] Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[CVPR 2025\] Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks](efficient_data_driven_mixture-of-expert_extraction_from_trained_networks.md)

<!-- RELATED:END -->
