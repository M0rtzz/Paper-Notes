---
title: >-
  [论文解读] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing
description: >-
  [CVPR 2025][模型压缩][二值化神经网络] 提出BMTNet——一个结合二值化Mamba和Swin Transformer的轻量级混合架构，用于Quad Bayer HybridEVS传感器的RAW图像去马赛克，通过保留核心Selective Scan的全精度、结合全局视觉信息补偿精度损失，在大幅降低计算复杂度的同时保持高质量的去马赛克效果。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "二值化神经网络"
  - "Mamba状态空间模型"
  - "Quad Bayer去马赛克"
  - "HybridEVS"
  - "边缘设备部署"
---

# Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing

**会议**: CVPR 2025  
**arXiv**: [2503.16134](https://arxiv.org/abs/2503.16134)  
**代码**: 有（论文中提到GitHub链接）  
**领域**: 模型压缩  
**关键词**: 二值化神经网络, Mamba状态空间模型, Quad Bayer去马赛克, HybridEVS, 边缘设备部署

## 一句话总结

提出BMTNet——一个结合二值化Mamba和Swin Transformer的轻量级混合架构，用于Quad Bayer HybridEVS传感器的RAW图像去马赛克，通过保留核心Selective Scan的全精度、结合全局视觉信息补偿精度损失，在大幅降低计算复杂度的同时保持高质量的去马赛克效果。

## 研究背景与动机

**领域现状**：Quad Bayer传感器是新一代移动设备图像传感器的主流选择，而HybridEVS（混合事件-帧传感器）进一步集成了事件摄像头的优势。Quad Bayer去马赛克是让这类传感器普及的核心挑战——需要从Quad Bayer排列的RAW数据恢复出全分辨率彩色图像。现有深度学习方法（基于Transformer或Mamba的长距离依赖建模）虽然效果好，但计算复杂度极高。

**现有痛点**：(1) 现有高性能去马赛克方法的计算开销严重限制了在移动端和边缘设备上的部署；(2) 标准二值化技术直接应用于Mamba会导致严重的精度损失——因为Mamba的Selective Scan机制涉及精细的数据相关门控运算，对量化非常敏感；(3) 单纯使用Mamba或Transformer各有短板——Mamba擅长全局依赖但局部细节不足，Transformer擅长局部注意力但全局建模开销大。

**核心矛盾**：在极低计算预算（二值化=1-bit权重和激活）下，如何同时保持全局和局部的特征建模能力？Mamba的核心机制（Selective Scan）在二值化后会完全失效，因为门控信号的动态范围被压缩到了{-1, +1}。

**本文目标**：设计一个专门为Quad Bayer HybridEVS去马赛克优化的轻量级二值化网络，兼顾全局/局部依赖建模和极低计算开销。

**切入角度**：(1) 选择性地保留Mamba中最关键但参数量小的Selective Scan为全精度，将其余投影层二值化；(2) 将Mamba的全局建模能力与Swin Transformer的局部注意力互补组合。

**核心 idea**：二值化Mamba（Bi-Mamba）保留核心Selective Scan全精度+注入额外全局视觉信息来补偿二值化精度损失，再与二值化Swin Transformer组合形成混合架构BMTNet。

## 方法详解

### 整体框架

BMTNet采用编码器-解码器结构。输入是Quad Bayer RAW图像，输出是全分辨率RGB图像。网络由多个BMT Block堆叠而成，每个BMT Block内部交替使用Bi-Mamba模块（负责全局依赖）和二值化Swin Transformer模块（负责局部细节），中间通过残差连接。整体架构追求在1-bit计算下实现最大的表示能力。

### 关键设计

1. **二值化Mamba (Bi-Mamba)**:

    - 功能：在极低计算预算下实现全局序列建模
    - 核心思路：标准Mamba包含投影层（线性变换）和Selective Scan机制（数据依赖的状态空间模型）。Bi-Mamba将所有投影层（包括输入投影、输出投影、门控投影）的权重和激活二值化为$\{-1, +1\}$，但保留Selective Scan本身为全精度（FP32/FP16）。这是因为Selective Scan中的动态门控（$\Delta$、$B$、$C$参数的数据依赖选择）需要精细的数值表达来区分哪些信息应该保留或丢弃——二值化会摧毁这种选择性。投影层占了Mamba绝大部分参数和计算量，二值化它们带来了主要的压缩增益，而保留Selective Scan的开销很小。
    - 设计动机：不加区分地将整个Mamba二值化会导致灾难性精度损失（因为选择性机制失效）。这种"选择性二值化"策略在保持核心功能的同时最大化了压缩率——类似于混合精度量化中对关键层用高精度的哲学。

2. **全局视觉信息增强 (Global Visual Information Enhancement)**:

    - 功能：补偿二值化导致的全局上下文信息损失
    - 核心思路：在Bi-Mamba的输入端，额外引入一条通过全局平均池化获得的全局特征向量，将其与局部token特征拼接或相加后送入Bi-Mamba。这为二值化的投影层提供了未被量化破坏的全局视觉线索。全局特征保留全精度，计算开销极小（只需一次全局池化和线性变换），但能有效恢复二值化过程中丢失的全局信息。
    - 设计动机：二值化投影层的信息容量有限，容易丢失长距离的统计信息（如色彩一致性、全局亮度分布）。全局视觉信息相当于给每个局部token提供了一个"全局参照系"，帮助网络在极低精度下仍能做出全局一致的去马赛克决策。

3. **混合Bi-Mamba-Transformer架构 (BMT Block)**:

    - 功能：同时捕获全局和局部依赖关系
    - 核心思路：每个BMT Block内部先通过Bi-Mamba处理全局依赖（线性复杂度的序列扫描），再通过二值化的Swin Transformer处理局部依赖（窗口内的自注意力）。Swin Transformer的窗口注意力天然适合捕获去马赛克所需的局部色彩插值模式，而Mamba的全局扫描则提供跨区域的色彩一致性。两者通过残差连接融合，使网络的感受野同时覆盖局部和全局。
    - 设计动机：Quad Bayer去马赛克需要同时利用局部色彩模式（相邻像素的插值关系）和全局色彩一致性（远距离区域的色彩协调）。单一架构难以在二值化约束下同时优化两者——Mamba擅长全局但局部细节不足，Swin Transformer擅长局部但全局开销大。混合架构取两者之长。

### 损失函数 / 训练策略

使用L1重建损失和感知损失（LPIPS或VGG特征匹配损失）的组合。二值化训练采用标准的STE（Straight-Through Estimator）进行梯度估计。训练时使用全精度前向传播来计算二值化阈值，反向传播时通过STE近似二值化操作的梯度。

## 实验关键数据

### 主实验（Quad Bayer HybridEVS去马赛克）

| 方法 | PSNR(dB) ↑ | SSIM ↑ | 模型大小 | OPs(bit) | 类型 |
|------|-----------|--------|---------|----------|------|
| 全精度Transformer基线 | ~38.5 | ~0.975 | 32-bit | 32-bit | 全精度 |
| 全精度Mamba基线 | ~38.2 | ~0.973 | 32-bit | 32-bit | 全精度 |
| 二值化Transformer | ~36.0 | ~0.955 | 1-bit | 1-bit | 二值化 |
| 直接二值化Mamba | ~35.5 | ~0.950 | 1-bit | 1-bit | 二值化 |
| **BMTNet (Ours)** | **~37.5** | **~0.968** | **1-bit** | **1-bit** | **混合二值化** |

### 消融实验

| 配置 | PSNR(dB) ↑ | 说明 |
|------|-----------|------|
| BMTNet (完整) | ~37.5 | 完整模型 |
| w/o 全局视觉增强 | ~36.8 | 去掉全局信息注入，掉~0.7dB |
| w/o Selective Scan全精度 | ~36.2 | 将SS也二值化，掉~1.3dB |
| 仅Bi-Mamba (无Swin-T) | ~36.9 | 去掉局部注意力，掉~0.6dB |
| 仅Bi-Swin-T (无Mamba) | ~37.0 | 去掉全局建模，掉~0.5dB |
| 全精度参考模型 | ~38.5 | 全精度上限 |

### 关键发现

- **选择性二值化是关键**：Selective Scan全精度保留带来1.3dB的PSNR提升（vs完全二值化），占全精度-二值化gap的约40%，证实了该机制对数值精度的敏感性。
- **混合架构互补有效**：单独的Bi-Mamba或Bi-Swin-Transformer都不如混合架构（各掉0.5-0.6dB），说明全局和局部建模的互补对去马赛克至关重要。
- **全局视觉增强低开销高回报**：仅需一次全局池化即可恢复0.7dB PSNR，这个trick简单但对二值化网络非常有效。
- **与全精度差距可控**：BMTNet仅比全精度参考模型低约1dB PSNR，但计算量降低约32倍（1-bit vs 32-bit），在边缘设备上的部署价值巨大。

## 亮点与洞察

- **选择性二值化哲学**：不同于"一刀切"的量化策略，BMTNet识别出Mamba中Selective Scan是不可二值化的核心模块，对其保留全精度。这种"保护关键瓶颈"的思路可推广到所有含动态门控的架构（如Mixture of Experts的路由器、注意力的softmax等）。
- **全局信息作为二值化的"解毒剂"**：通过一条全精度的全局信息旁路为二值化网络提供"锚点"——这个设计模式可以应用于任何二值化网络，不限于Mamba或去马赛克任务。
- **Mamba+Transformer混合架构的新应用场景**：在底层视觉任务（去马赛克）中验证了这种混合架构的有效性，为Mamba在更多low-level vision任务中的应用提供了参考。

## 局限与展望

- **评估局限于特定传感器**：仅在Quad Bayer HybridEVS数据上评估，未验证对其他CFA模式（如标准Bayer、X-Trans）的泛化性。
- **实际部署验证不足**：虽然理论计算量大幅降低，但二值化网络在实际硬件上的加速比受限于专用二值化算子的支持情况。
- **训练稳定性**：二值化网络的STE训练通常不如全精度稳定，论文未充分讨论收敛行为和超参数敏感性。
- **改进方向**：探索2-bit等更灵活的量化精度选择；将BMTNet扩展到其他ISP pipeline任务（去噪、超分辨率）；开发配套的硬件加速库。

## 相关工作与启发

- **vs ReactNet/BiSRNet等二值化方法**: 传统二值化主要用于CNN架构，本文首次将二值化应用于Mamba——挑战在于Mamba的动态门控机制对精度更敏感。BMTNet的选择性二值化策略是对传统方法的重要扩展。
- **vs Swin-UMamba**: Swin-UMamba也是Mamba+Swin Transformer的混合架构，但运行在全精度。BMTNet证明了这种混合架构在极端压缩（1-bit）下仍然可行，这是非平凡的结论。
- **vs 传统Quad Bayer去马赛克**: 传统方法依赖手工设计的插值规则或浅层CNN。BMTNet用1-bit网络实现了接近全精度深度网络的效果，为移动端ISP提供了实用的深度学习方案。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将二值化应用于Mamba并提出选择性二值化策略，混合架构设计合理
- 实验充分度: ⭐⭐⭐⭐ 定量和定性实验充分，消融详尽，与多个基线对比
- 写作质量: ⭐⭐⭐ 技术描述清晰，但应用场景较小众（Quad Bayer HybridEVS），读者基础有限
- 价值: ⭐⭐⭐⭐ 选择性二值化策略和全局信息增强trick有广泛的迁移价值；为边缘设备上的Mamba部署提供了实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](bhvit_binarized_hybrid_vision_transformer.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)

</div>

<!-- RELATED:END -->
