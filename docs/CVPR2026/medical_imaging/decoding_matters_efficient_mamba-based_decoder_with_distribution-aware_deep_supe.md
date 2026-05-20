---
title: >-
  [论文解读] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation
description: >-
  [CVPR 2026][医学图像][医学图像分割] 提出 Deco-Mamba，一种以解码器为中心的 Transformer-CNN-Mamba 混合架构，通过 Co-Attention Gate、视觉状态空间模块（VSSM）和可变形卷积增强解码过程，同时引入基于窗口化 KL 散度的分布感知深度监督策略…
tags:
  - "CVPR 2026"
  - "医学图像"
  - "医学图像分割"
  - "Mamba"
  - "解码器设计"
  - "深度监督"
  - "KL散度"
---

# Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.12547](https://arxiv.org/abs/2603.12547)  
**代码**: 待发布（接收后公开）  
**领域**: 医学图像  
**关键词**: 医学图像分割, Mamba, 解码器设计, 深度监督, KL散度

## 一句话总结

提出 Deco-Mamba，一种以解码器为中心的 Transformer-CNN-Mamba 混合架构，通过 Co-Attention Gate、视觉状态空间模块（VSSM）和可变形卷积增强解码过程，同时引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA。

## 研究背景与动机

现有医学图像分割方法（U-Net、TransUNet、Mamba-UNet 等）的一个共性问题是**过度关注编码器设计而忽视解码器**：

- CNN 编码器（U-Net 系列）：局部感受野限制长程依赖建模。
- Transformer 编码器（TransUNet、Swin-UNet）：自注意力 $O(n^2)$ 复杂度，高分辨率不可扩展。
- Mamba 编码器（U-Mamba、Swin-UMamba）：线性复杂度，但多数方法只在编码器引入 Mamba，解码器仍然简单。

核心矛盾：**强大的编码器提取了丰富的语义表示，但如果解码器设计不足，就无法在上采样过程中准确恢复物体边界和上下文结构**。现有方法要么用级联解码器导致参数暴增（如 Cascaded-MERIT, 148M 参数），要么解码器过于轻量丢失细节。

另一个问题：传统深度监督在低分辨率中间层需要 resize 到全分辨率再和 GT 计算损失，这个过程本身就损失了结构信息。

Deco-Mamba 的切入：(1) 将 Mamba 引入解码器而非编码器；(2) 设计分布感知的深度监督，直接在各解码层的原始分辨率计算 KL 散度。

## 方法详解

### 整体框架

U-Net 型结构：编码器使用 CNN 分支（7×7 卷积）+ PVT Transformer（4 阶段），解码器使用 6 个阶段，每阶段包含 Co-Attention Gate → VSSMB → Deformable Residual Block。

### 关键设计

1. **Co-Attention Gate (CAG，共注意力门控)**：传统 Attention Gate 只用解码器特征作为门控信号高亮编码器特征。CAG 让编码器和解码器特征**互为门控信号**，得到两路注意力输出后拼接，再用通道注意力（CA）精炼：$D_i' = CA[AG(x=X_i, g=D_{i+1}), AG(x=D_{i+1}, g=X_i)]$。设计动机：解码器特征同样需要空间显著性筛选，且通道维度的关系也应被建模。消融显示 CAG 优于 AG、LGAG 和 CBAM。

2. **Vision State Space Mamba Block (VSSMB，视觉状态空间模块)**：在解码器中引入 SSM（Mamba），通过选择性扫描在水平、垂直及其逆方向传播上下文信息，以线性复杂度建模长程依赖。瓶颈层用 2 个 VSSMB，中间层各 1 个，最后一层不用（全分辨率下卷积更合适）。设计动机：解码器在逐层上采样过程中需要保持全局语义一致性，SSM 比自注意力资源效率更高。

3. **Deformable Residual Block (DRB，可变形残差块)**：在每个 VSSMB 之后放置 DRB，包含标准 3×3 卷积和可变形卷积。可变形卷积预测逐像素偏移和调制掩码，使采样位置自适应几何变化。设计动机：VSSMB 擅长全局上下文但可能平滑局部细节，DRB 通过空间自适应恢复边界精度。

4. **Multi-Scale Distribution-Aware (MSDA) Deep Supervision**：传统深度监督将中间预测 resize 到 GT 分辨率再算 Dice/CE 损失，resize 操作丢失结构信息。MSDA 方法：对各解码层原始分辨率输出，用 distribution head 映射到类别数维度，GT 通过局部窗口平均得到同分辨率的类别分布 $\tilde{P}^{(s)}$，然后计算 KL 散度：$\mathcal{L}_{\text{KL}}^{(s)} = \sum_{b,h,w}\sum_c \tilde{P}_{b,c,h,w}^{(s)} \log\frac{\tilde{P}_{b,c,h,w}^{(s)}}{Q_{b,c,h,w}^{(s)}}$。还引入边界加权 $W_{h,w}^{(s)} = (1 - \max_n \tilde{P}_{h,w,n}^{(s)})^\alpha$ 来强调类别交界处。

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dice}} + \sum_{s=1}^S \lambda_s \mathcal{L}_{\text{dist}}^{(s)}$$

Dice 损失保证最终预测的空间重叠，MSDA 的 KL 散度损失在各解码阶段提供分布一致性监督。AdamW + 余弦学习率，224×224 输入，A5000 GPU。

## 实验关键数据

### 主实验

**Synapse（8 类腹部多器官 CT）**

| 方法 | DSC↑ | HD95↓ | 参数(M) | FLOPs(G) |
|------|------|-------|---------|----------|
| Cascaded-MERIT | 83.59 | 15.99 | 147.86 | 33.31 |
| PAG-TransYnet | 83.43 | 15.82 | 144.22 | 33.65 |
| **Deco-Mamba-V1** | **85.07** | **14.72** | 46.93 | 17.24 |
| Deco-Mamba-V0 | 83.16 | 15.89 | **9.67** | **9.73** |

**跨数据集泛化（7 个基准）**

| 数据集 | Deco-Mamba-V1 | 次优方法 | 提升 |
|--------|---------------|----------|------|
| Synapse | **85.07** | 83.59 (Cascaded-MERIT) | +1.48 |
| BTCV(13类) | **78.45** | 75.87 (PAG-TransYnet) | +2.58 |
| ACDC | **92.35** | 92.12 (PVT-EMCAD-B2) | +0.23 |
| ISIC17 | **86.01** | 85.67 (Cascaded-MERIT) | +0.34 |
| GlaS | **96.91** | 96.91 (Cascaded-MERIT) | 持平 |
| MoNuSeg | **85.14** | 83.41 (Deco-Mamba-V0) | +1.73 |

### 消融实验

| 配置 | DSC↑ | HD95↓ | 说明 |
|------|------|-------|------|
| w/o CNN 编码器分支 | 84.07 | 18.92 | 丢失高分辨率空间细节 |
| w/o VSSMB | 83.51 | 15.96 | 长程依赖建模缺失 |
| 用 AG 替换 CAG | 82.98 | 15.69 | 单向注意力不够 |
| 用标准卷积替换可变形卷积 | 84.53 | 16.18 | 边界自适应性下降 |
| 只用 Dice (无 MSDA) | 83.84 | 14.94 | 缺少多尺度分布约束 |
| Dice + 传统深度监督 | 84.24 | 15.89 | resize 反而增加 HD95 |
| **Deco-Mamba (full)** | **85.07** | **14.72** | — |

### 关键发现

- 以解码器为中心的设计确实有效：用同样的 PVT-B0 backbone，Deco-Mamba 比 Swin-UNet 高 5.58% DSC。
- Deco-Mamba-V0（9.67M 参数）性能超过大多数 100M+ 的方法，验证了"解码器比编码器更重要"的论点。
- MSDA 深度监督优于传统深度监督和边界损失，因为避免了 resize 导致的信息损失。

## 亮点与洞察

- "解码器为中心"的设计哲学值得关注：不追求更大的预训练编码器，而是在解码端精心设计。
- MSDA 的窗口化 KL 散度是一个优雅的解决方案：不需要 resize GT，而是对 GT 做局部窗口统计来匹配低分辨率预测。
- Mamba 在解码器中的应用比在编码器中更有效，因为解码器需要在上采样过程中保持全局一致性。

## 局限与展望

- 仅支持 2D 分割，3D 医学图像（如 CT/MRI 体数据）的扩展未被探讨。
- 7 个数据集虽多但都是常用基准，没有在更新或更难的数据集上验证。
- Window size 和 $\lambda_s$ 的选择对 MSDA 性能的敏感性未详细分析。
- 代码尚未公开。

## 相关工作与启发

- 与 EMCAD（EMCAD-B2）的对比：EMCAD 也注重解码器但用轻量卷积块+传统深度监督，Deco-Mamba 用 Mamba+分布感知监督更进一步。
- 与 Swin-UMamba 的对比：后者在编码器引入 Mamba，本文在解码器引入，两者互补的思路可以结合。
- MSDA 的窗口化分布思路可以推广到其他密集预测任务的深度监督中。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 解码器Mamba+分布感知深度监督两个创新点搭配合理
- **实验充分度**: ⭐⭐⭐⭐⭐ 7 个数据集，完整消融，backbone 对比，效率分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，模块解释充分
- **价值**: ⭐⭐⭐⭐ 以解码器为中心的思路对社区有启发，MSDA 可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)

</div>

<!-- RELATED:END -->
