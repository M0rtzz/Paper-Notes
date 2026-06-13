---
title: >-
  [论文解读] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction
description: >-
  [CVPR 2025][线性注意力] 本文提出CARE Transformer，通过非对称特征解耦将局部归纳偏置和长距离依赖的学习分离，并设计动态记忆单元和双交互模块充分利用特征互补性，实现了移动端友好的线性复杂度视觉Transformer，在ImageNet上以仅0.7 GMACs达到78.4% top-1精度。
tags:
  - "CVPR 2025"
  - "线性注意力"
  - "Transformer"
  - "移动端部署"
  - "特征解耦"
  - "双交互"
---

# CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction

**会议**: CVPR 2025  
**arXiv**: [2411.16170](https://arxiv.org/abs/2411.16170)  
**代码**: [https://github.com/zhouyuan888888/CARE-Transformer](https://github.com/zhouyuan888888/CARE-Transformer)  
**领域**: 其他  
**关键词**: 线性注意力, 轻量化Transformer, 移动端部署, 特征解耦, 双交互

## 一句话总结
本文提出CARE Transformer，通过非对称特征解耦将局部归纳偏置和长距离依赖的学习分离，并设计动态记忆单元和双交互模块充分利用特征互补性，实现了移动端友好的线性复杂度视觉Transformer，在ImageNet上以仅0.7 GMACs达到78.4% top-1精度。

## 研究背景与动机
1. **领域现状**：高效视觉Transformer设计主要有两条路线——局部注意力（限制感受野）和线性注意力（降低复杂度），但现有线性注意力模型要么效率提升有限要么精度下降明显，难以部署到移动端。
2. **现有痛点**：线性注意力固有的高熵特性使其难以抑制无关token的影响。MLLA通过堆叠局部增强来缓解，但堆叠方式将局部和全局信息的学习与融合耦合在一起，灵活性和效率都受限。
3. **核心矛盾**：堆叠式学习要求输入经过所有卷积和线性注意力操作，是计算瓶颈；同时耦合设计阻碍了更有效的特征融合模块的设计。
4. **本文目标**：设计一种既高效又高精度的线性注意力机制，适合移动端部署。
5. **切入角度**：作者提出关键命题——局部增强过程可显式分为学习和交互两步，分离这两步可以分别优化。
6. **核心idea**：非对称解耦 + 双交互 = 分而治之地学习局部/全局信息，然后充分融合利用互补性。

## 方法详解

### 整体框架
输入图像 → 4×4卷积stem → 四阶段层级结构（每阶段含多个CARE块）→ 每个CARE块中：特征通道分割为两部分 → 小通道部分（$d_1$）经线性注意力捕获全局依赖 → 大通道部分（$d_2$）经深度卷积学习局部偏置 → 双交互模块融合 → 输出。

### 关键设计

1. **非对称特征解耦（Asymmetrical Feature Decoupling）**:

    - 功能：将输入特征在通道维度分为两部分，分别学习局部和全局信息。
    - 核心思路：将 $\mathbf{X} \in \mathbb{R}^{hw \times d}$ 分割为 $\bar{\mathbf{X}} \in \mathbb{R}^{hw \times d_1}$（送入线性注意力）和 $\tilde{\mathbf{X}} \in \mathbb{R}^{hw \times d_2}$（送入深度卷积），关键是设 $d_1 < d_2$（非对称）。由于线性注意力的复杂度对通道维度是二次的 $O(hwd^2)$，非对称设置使复杂度降为 $O(hw d_1^2)$，其中 $d_1 = d/3$。
    - 设计动机：线性注意力的计算瓶颈在于通道维度的二次开销，用更少通道学全局、更多通道学局部，效率和信息都得到保证。

2. **动态记忆单元（Dynamic Memory Unit）**:

    - 功能：沿网络流水线保持关键信息，实现跨层特征交互。
    - 核心思路：每个阶段的第一个CARE块将上一阶段最后的特征和记忆拼接后通过2×2卷积（stride=2）降采样构建初始记忆 $\mathbf{Z}_0^s = \text{CONV}_{2\times2}(\mathbf{X}_{-1}^{s-1} \oplus \mathbf{Z}_{-1}^{s-1})$。后续块中记忆通过双交互模块动态更新。
    - 设计动机：不同层的特征各有优势且互补，记忆单元使早期层的信息能在后续层被重放和利用。

3. **双交互模块（Dual Interaction Module）**:

    - 功能：实现两种层面的特征交互——局部与全局特征间、不同层特征间。
    - 核心思路：包含两个交互块。Inter₁融合局部偏置和长距离依赖特征 $\text{Inter}_1(\bar{\mathbf{X}}, \tilde{\mathbf{X}})$；Inter₂进一步与记忆单元交互 $\text{Inter}_2(\cdot, \mathbf{Z})$。每个交互块实现为：拼接 → 归一化 → 1×1卷积（通道交互+扩展4倍）→ 3×3深度卷积（空间交互）→ 1×1卷积（映射回原空间）。
    - 设计动机：解耦学习后需要有效的融合机制来利用互补性，双交互同时考虑了同层局部-全局和跨层的信息交流。

### 损失函数 / 训练策略
标准ImageNet分类训练。CARE-S0/S1/S2三个尺寸，块配置分别为⟨2,4,8,4⟩、⟨3,6,10,6⟩、⟨3,6,10,6⟩，通道维度递增。局部偏置学习器使用3×3和7×7双尺度深度卷积，也采用解耦方式处理。前两阶段不使用线性注意力，改用1×11和11×1大核深度卷积。线性注意力的计算复杂度对通道维度为二次$O(hwd_1^2)$，非对称设置$d_1=d/3$将计算量降至对称设置的约1/3。

## 实验关键数据

### 主实验

| 模型 | GMACs | 参数量 | iPhone13延迟 | Top-1 Acc (%) |
|------|-------|--------|-------------|---------------|
| MobileNetV2-1.0 | 0.3 | 3.5M | 1.0ms | 71.8 |
| EMO-2M | 0.4 | 2.3M | 2.0ms | 75.1 |
| **CARE-S0** | **0.7** | **3.5M** | **1.1ms** | **78.4** |
| EfficientFormerV2-S1 | 0.7 | 6.2M | 1.6ms | 79.0 |
| **CARE-S1** | **1.0** | **6.2M** | **1.5ms** | **80.6** |
| **CARE-S2** | **1.9** | **12.7M** | **2.0ms** | **82.1** |

### 消融实验

| 配置 | GMACs | Top-1 (%) | 说明 |
|------|-------|-----------|------|
| Full CARE-S1 | 1.0 | 80.6 | 完整模型 |
| 对称解耦 ($d_1=d_2$) | 1.2 | 80.3 | 非对称更优且更高效 |
| w/o 记忆单元 | 1.0 | 79.8 | 记忆单元贡献+0.8% |
| w/o 双交互 | 0.9 | 79.2 | 交互模块贡献+1.4% |
| 堆叠式 (MLLA) | 1.3 | 80.1 | CARE更高效且更准确 |

### 关键发现
- 非对称解耦在理论和实验上都被证明比对称解耦更高效：$\Omega(\Delta_1) < \Omega(0)$。
- CARE在COCO检测和ADE20K分割上也表现出色：CARE-S1在COCO目标检测AP$^b$达40.7，实例分割AP$^m$达37.5，ADE20K语义分割mIoU达41.9，均优于同量级EfficientFormer和MobileViG。
- 在iPhone13上实际延迟优于同GMACs的其他方法，说明设计对移动端硬件友好。
- 双交互模块是精度提升的主要来源（+1.4%），解耦策略则是效率提升的关键。
- CARE-S2在仅1.9 GMACs下达到82.1% Top-1，与Swin-T（4.5 GMACs）、ConvNeXt-T（4.5 GMACs）持平，计算量不到其一半。

## 亮点与洞察
- **非对称解耦的理论证明**：通过数学推导证明了非对称设置的效率优势，从理论和实验双重角度验证了设计选择。
- **分而治之的设计哲学**：将"学习"和"融合"两步解耦，各自可独立优化，大幅提升了设计灵活性。
- **可迁移到其他线性注意力模型**：非对称解耦和双交互的思想可应用于任何需要平衡局部和全局信息的高效模型。
- **设计细节**：前两阶段不使用线性注意力，改用1×11和11×1大核深度卷积（也采用解耦方式）。局部偏置学习器使用3×3和7×7双尺度深度卷积。$d_1=d/3$的非对称比例在效率和信息保留间取得最优平衡。
- **全面的下游验证**：COCO检测AP$^b$达40.7（CARE-S1），ADE20K语义分割mIoU达41.9，与GMACs高2-5倍的Swin-T和ConvNeXt-T性能持平。

## 局限与展望
- 目前仅在分类、检测、分割三个基础视觉任务上验证，未涉及生成任务。
- 前两阶段不使用线性注意力而用大核卷积，存在一定的工程妥协。
- 动态记忆单元的维度和更新策略选择需要更多理论指导。
- 线性注意力的高熵特性虽通过解耦缓解，但在极长序列上可能仍有局限。
- 非对称比例 $d_1=d/3$ 的选择基于经验，不同任务可能需要不同比例。
- 动态记忆单元通过2×2卷积stride=2降采样构建初始记忆，跨阶段传递关键信息。记忆在后续CARE块中通过双交互模块动态更新，实现了跨层特征复用。
- RTX 4090上CARE-S1延迟仅1.5ms，与EfficientFormerV2-S1的1.6ms相当但精度高+1.6%。

## 相关工作与启发
- **vs MLLA**: MLLA堆叠卷积和线性注意力，CARE解耦学习再交互融合，效率更高精度也更好。
- **vs FLatten/SLAB**: 这些方法改进线性注意力本身的聚焦能力，CARE从架构层面解决局部-全局平衡问题。
- **vs EfficientFormerV2**: EfficientFormerV2在相同GMACs下精度接近但延迟更高（1.6ms vs CARE-S1 1.5ms），CARE更适合实际移动端部署。
- **vs MobileNetV2**: MobileNetV2在0.3 GMACs下仅71.8%，CARE-S0在0.7 GMACs下78.4%，效率-精度权衡更优。
- **vs Swin-T/ConvNeXt-T**: 这些模型在4.5 GMACs下达82.1%，CARE-S2在1.9 GMACs（不到一半计算量）下持平。

## 评分

### 实现细节
CARE-S0/S1/S2块配置⟨2,4,8,4⟩/⟨3,6,10,6⟩/⟨3,6,10,6⟩。非对称比例$d_1=d/3$，$d_2=2d/3$。
- 新颖性: ⭐⭐⭐⭐ 非对称解耦+双交互组合新颖，理论推导扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 三个任务+移动端延迟+详细消融
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，图表直观
- 价值: ⭐⭐⭐⭐ 对移动端视觉Transformer部署有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Vision Transformer 微调中的非光滑分量优势](../../ICML2026/others/vision_transformer_finetuning_benefits_from_non-smooth_components.md)
- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[ICCV 2025\] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer](../../ICCV2025/others/layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)
- [\[NeurIPS 2025\] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](../../NeurIPS2025/others/4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)
- [\[ACL 2025\] Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data](../../ACL2025/others/capacity_matters_a_proof-of-concept_for_transformer_memorization_on_real-world_d.md)

</div>

<!-- RELATED:END -->
