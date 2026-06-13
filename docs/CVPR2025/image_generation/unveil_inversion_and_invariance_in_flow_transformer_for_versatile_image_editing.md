---
title: >-
  [论文解读] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing
description: >-
  [CVPR 2025][图像生成][流匹配反演] 针对基于 Flow Transformer (MM-DiT) 的无训练图像编辑，提出两阶段流反演方法（固定点迭代+速度补偿）和基于自适应层归一化（AdaLN）的不变性控制机制，统一支持刚性和非刚性编辑。 - 基于 Flow Transformer 的 T2I 模型（如 Sta…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "流匹配反演"
  - "图像编辑"
  - "MM-DiT"
  - "不变性控制"
  - "AdaLN"
---

# Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing

**会议**: CVPR 2025  
**arXiv**: [2411.15843](https://arxiv.org/abs/2411.15843)  
**代码**: 无  
**领域**: 图像生成/编辑  
**关键词**: 流匹配反演, 图像编辑, MM-DiT, 不变性控制, AdaLN

## 一句话总结

针对基于 Flow Transformer (MM-DiT) 的无训练图像编辑，提出两阶段流反演方法（固定点迭代+速度补偿）和基于自适应层归一化（AdaLN）的不变性控制机制，统一支持刚性和非刚性编辑。

## 研究背景与动机

- 基于 Flow Transformer 的 T2I 模型（如 Stable Diffusion 3.5）具有更强的生成先验和更好的文图对齐能力，但其编辑潜力尚未被充分挖掘
- 现有扩散模型中成熟的 DDIM 反演方法在 Rectified Flow 的 Euler 采样器上表现不佳，近似误差更大，直接将其迁移到流模型效果差
- 实验发现：即使在更强的 SD3.5 上运行 Euler 反演，其重建质量也远低于在较弱 SD1.5 上运行 DDIM 反演
- U-Net 架构中基于注意力操作的不变性控制方法（交叉注意力注入、KV注入等）难以在 MM-DiT 中直接复用，因为 MM-DiT 没有独立的交叉注意力模块
- 现有注意力注入方法在刚性编辑（如替换物体）和非刚性编辑（如改变姿态、数量）之间难以兼顾
- 需要专门为流模型和 Transformer 架构设计的反演与不变性控制方案

## 方法详解

### 整体框架

整体框架分为反演和编辑两个阶段。反演阶段使用两阶段策略将输入图像投射到流模型的噪声域：第一阶段通过固定点迭代优化速度估计以获得接近真实生成过程的反演轨迹，第二阶段在编辑时计算速度补偿以精确恢复原始图像。编辑阶段引入基于 AdaLN 的不变性控制，通过替换未编辑文本 token 对应的特征来保持非目标区域不变，同时不阻碍目标区域的编辑效果。

### 关键设计

**1. 两阶段流反演（Two-stage Flow Inversion）**

- **功能**：将真实图像精确投射到流模型的噪声域，同时保持编辑友好性
- **核心思路**：第一阶段利用固定点迭代改善速度场估计。标准 Euler 反演用 $v_\theta(\mathbf{x}_{t+1}, t)$ 近似 $v_\theta(\mathbf{x}_t, t)$，误差大。固定点迭代从 $\mathbf{x}_t^0 = \mathbf{x}_{t+1}$ 出发，迭代 $\mathbf{x}_t^{i+1} = \mathbf{x}_{t+1} + (\sigma_t - \sigma_{t+1})v_\theta(\mathbf{x}_t^i, t)$ 并对结果取平均，利用速度场理论上为常量 $\mathbf{x}_1 - \mathbf{x}_0$ 的性质获得更稳定的估计。第二阶段在编辑时计算补偿 $\epsilon_t = \mathbf{x}_{t+1} - \hat{\mathbf{x}}_{t+1}$ 加到速度上
- **设计动机**：理想的反演轨迹应贴近真实生成过程的轨迹，这样既能精确恢复原图，又能保留文图对齐的先验分布从而便于编辑。避免过拟合单张图像导致丧失编辑能力

**2. 基于 AdaLN 的不变性控制**

- **功能**：在编辑过程中保持非目标内容不变，同时支持刚性和非刚性编辑
- **核心思路**：在 MM-DiT 中，自适应层归一化（AdaLN）中的文本特征与图像语义（姿态、数量、物体类型等）紧密对应。定义 Map 函数，将目标文本特征 $M^a$ 中未变化 token 的特征替换为原始文本特征 $M^b$ 中对应 token 的特征，保持编辑 token 的特征不变。该操作从初始时刻执行到时刻 $S$
- **设计动机**：MM-DiT 将文本和图像特征在自注意力中联合处理，无法像 U-Net 那样通过交叉注意力控制。而 AdaLN 直接对应文本变化，能精确区分编辑/非编辑语义，避免注意力注入对非刚性编辑的阻碍

**3. 固定点迭代中的速度平均策略**

- **功能**：提升反演轨迹中每一步的速度估计质量
- **核心思路**：Rectified flow 的理想速度 $v_\theta = \mathbf{x}_1 - \mathbf{x}_0$ 是常量，因此固定点迭代序列 $\{\mathbf{x}_t^i\}_{i=1}^I$ 的平均值比单次迭代结果更接近真实值，因为迭代可能不单调收敛而是围绕真实不动点震荡
- **设计动机**：实验可视化显示未平均的 latent 在不同迭代次数时偏差较大，平均后更接近原始图像

### 损失函数 / 训练策略

- 本方法为无训练（tuning-free）方法，不涉及额外训练损失
- 基于 Stable Diffusion 3.5 实现，使用 Euler 采样器
- 反演步数30步，CFG 分别设为1（反演）和2（编辑）
- 固定点迭代次数设为3
- 不变性控制时刻 $S$ 对所有编辑类型统一设定，不需要逐类型调参

## 实验关键数据

### 主实验

在 PIE benchmark（700张自然/人工图像）上的定量对比：

| 方法 | Structure Dist↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|----------------|-------|--------|------|-------|------------|-------------|
| P2P | **13.44** | 27.03 | 60.67 | 35.86 | 84.11 | 24.75 | 21.86 |
| PnP | 24.29 | 22.46 | 106.06 | 80.45 | 79.68 | **25.41** | **22.62** |
| MasaCtrl | 24.70 | 22.64 | 88.79 | 81.09 | 80.76 | 24.38 | 21.35 |
| InfEdit | 24.70 | 26.31 | 87.94 | 75.19 | 81.33 | 23.67 | 21.86 |
| RFinv | 32.62 | 22.03 | 159.62 | 96.01 | 73.26 | 24.89 | 21.89 |
| **Ours** | 18.17 | **26.62** | **80.55** | **40.24** | **91.50** | 25.74 | 22.27 |

### 消融实验

固定点迭代次数消融（150张随机子集）：

| 迭代次数 | Structure Dist↓ | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP Whole↑ |
|---------|----------------|-------|--------|-------|------------|
| Iter 0 (plain Euler) | 48.49 | 20.78 | 199.93 | 80.59 | 24.14 |
| Iter 1 | 15.29 | 26.98 | 67.80 | 93.89 | 24.31 |
| Iter 3 | **14.92** | **27.51** | **66.48** | **94.40** | **24.48** |

### 关键发现

1. 朴素 Euler 反演（Iter 0）的结构距离高达48.49，而仅1次固定点迭代即可降至15.29，证实了近似误差问题的严重性
2. 本方法在背景保持（SSIM=91.50）和编辑能力（CLIP Edited=22.27）之间取得最佳平衡
3. AdaLN 替换时刻 $S$ 在较大范围内不敏感，且不阻碍非刚性编辑，优于注意力注入方法
4. 自注意力注入超过20%时步就会阻碍"坐→站"等非刚性编辑

## 亮点与洞察

- 系统揭示了 Euler 反演与 DDIM 反演的数学联系，证明 Euler 反演虽形式相似但更易受近似误差影响
- 利用 Rectified Flow 速度场的常量性质设计固定点迭代+平均策略，理论动机清晰
- AdaLN 不变性控制机制优雅地解决了刚性/非刚性编辑的统一问题，无需逐类型调参
- 首次在 MM-DiT 架构上实现高质量的无训练图像编辑

## 局限与展望

- 当输入图像超出模型域分布时，反演偏差大，AdaLN 控制失效
- 固定点迭代增加了推理时的计算开销（每步需多次 Transformer 前向传播）
- 编辑效果受限于基础模型的生成先验，无法实现模型不支持的编辑类型
- 未来可探索与指令微调方法结合，以进一步扩展编辑能力边界

## 相关工作与启发

- **DDIM Inversion + Null-text Optimization**: 扩散模型中经典的反演-编辑框架，本文揭示其在流模型上的失效并提出针对性解决方案
- **Prompt-to-Prompt (P2P)**: 通过交叉注意力替换实现编辑，但在 MM-DiT 中无交叉注意力可用
- **MasaCtrl / InfEdit**: 针对非刚性编辑的注意力操作方法，但与刚性编辑存在 trade-off
- 启发：在新一代 Transformer 架构的生成模型中，探索除注意力外的其他调控点（如归一化层）可能是更有效的编辑策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 对流模型反演的系统分析有深度，AdaLN 控制思路新颖
- **实验充分度**: ⭐⭐⭐⭐ — PIE benchmark 的定量评估全面，消融分析充分
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，问题分析系统
- **价值**: ⭐⭐⭐⭐ — 为流模型的图像编辑奠定了重要基础，方法简洁实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](../../ICML2025/image_generation/taming_rectified_flow_for_inversion_and_editing.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)

</div>

<!-- RELATED:END -->
