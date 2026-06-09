---
title: >-
  [论文解读] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration
description: >-
  [CVPR 2025][图像恢复][Mamba] 提出 MaIR，核心创新是嵌套 S 形扫描策略（NSS）通过条带划分保持局部性 + S 形路径保持连续性，以及序列洗牌注意力（SSA）通过通道级注意力智能聚合不同扫描方向的序列，在超分、去噪、去模糊、去雾 4 大任务 14 个数据集上达到 SOTA。
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "Mamba"
  - "state space model"
  - "scanning strategy"
  - "NSS"
  - "SSA"
  - "超分辨率"
  - "去噪"
---

# MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2412.20066](https://arxiv.org/abs/2412.20066)  
**代码**: [GitHub](https://github.com/XLearning-SCU/2025-CVPR-MaIR)  
**领域**: 图像复原  
**关键词**: Mamba, state space model, image restoration, scanning strategy, NSS, SSA, super-resolution, denoising

## 一句话总结

提出 MaIR，核心创新是嵌套 S 形扫描策略（NSS）通过条带划分保持局部性 + S 形路径保持连续性，以及序列洗牌注意力（SSA）通过通道级注意力智能聚合不同扫描方向的序列，在超分、去噪、去模糊、去雾 4 大任务 14 个数据集上达到 SOTA。

## 研究背景与动机

**领域现状**: CNN 和 Transformer 在图像恢复中表现优异，但受限于局部感受野（CNN 的小卷积核、Transformer 的窗口注意力），难以高效捕获长距离依赖。Mamba 以线性复杂度建模长序列引起关注。

**现有痛点**:
1. Mamba 处理 2D 图像需将其展平为 1D 序列，但现有扫描策略破坏图像的局部性和连续性
2. VMamba/ViM 的 Z 形扫描同时破坏局部性和连续性
3. Zigma 的 S 形扫描保持连续性但忽略局部性
4. LocalMamba 的窗口扫描保持局部性但窗口内外的 Z 形路径破坏连续性
5. 现有方法用逐像素求和聚合不同方向序列，忽略了序列间的差异

**核心矛盾**: 如何在展平 2D→1D 时同时保持局部关系和空间连续性？如何有效聚合不同展开方式产生的异质序列？

**本文切入角度**: 设计同时满足局部性+连续性的零开销扫描策略 + 通道级注意力序列聚合。

## 方法详解

### 整体框架

标准三阶段架构：浅层特征提取 → 深层特征提取（多个 Residual Mamba Group） → 重建。核心模块 MaIRM 由 NSS + SSO + SSA 三步构成：先用 NSS 展平 2D 为 4 个 1D 序列，再用 SSO 处理每个序列的长距离依赖，最后用 SSA 智能聚合 4 个序列。

### 关键设计

**1. 嵌套 S 形扫描策略（Nested S-shaped Scanning, NSS）**
- **功能**: 将特征图分为多个不重叠条带（stripe），在条带内和条带间都采用 S 形路径扫描
- **核心思路**:
    - 条带划分保持**局部性**：每个条带内的像素空间相邻
    - S 形路径保持**连续性**：序列中相邻 token 在原图中也空间相邻（不像 Z 形会跳行）
    - Shift-stripe 机制：相邻模块交替偏移条带起点（类似 Swin Transformer 的 shift window），确保条带边界区域的局部性和连续性
    - 4 个扫描方向：左上→右下、右下→左上、右上→左下、左下→右上
- **设计动机**: 综合 LocalMamba（局部性）和 Zigma（连续性）的优点，且完全零额外计算开销

**2. 序列洗牌注意力（Sequence Shuffle Attention, SSA）**
- **功能**: 替代简单的像素级求和，通过通道级注意力权重聚合 4 个方向的序列
- **核心思路**:
    - 对 4 个方向的序列做空间平均池化 → 拼接为 $\tilde{X} \in \mathbb{R}^L$（$L = K \times D$）
    - Sequence shuffle：重排为 $[x^1_1, x^2_1, x^3_1, x^4_1, x^1_2, ...]$，使同一通道的不同方向特征相邻
    - 分组卷积（group=4）生成注意力权重 → sequence unshuffle 恢复原序
    - 加权求和：$Y = \sum_{i=1}^{4} W^i * X^i$
- **设计动机**: 不同扫描方向携带互补信息，通道级注意力能自适应选择每个通道最有价值的方向信息

**3. Shift-Stripe 机制**
- **功能**: 相邻两个 MaIRM 模块交替使用原始条带和偏移条带
- **核心思路**: 第一个模块条带宽度 $w_s$，第二个模块首尾条带宽度 $w_s/2$、中间条带宽度 $w_s$，使前一模块的条带边界在当前模块被完整条带覆盖
- **设计动机**: 解决条带边界处局部性和连续性断裂的问题

### 损失函数 / 训练策略

- 超分辨率：$L_1$ loss $\mathcal{L} = \|y - y'\|_1$
- 去噪/去模糊/去雾：Charbonnier loss $\mathcal{L} = \sqrt{\|y - y'\|^2 + \epsilon^2}$，$\epsilon = 10^{-3}$
- 重建：超分用 pixel-shuffle 上采样 + 3×3 conv；去噪等用 3×3 conv + 残差连接

## 实验关键数据

### 主实验 — 经典超分辨率（×2）

| 方法 | Set5 PSNR | Set14 PSNR | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|---|
| SwinIR | 38.42 | 34.46 | 33.81 | 39.92 |
| SRFormer | 38.51 | 34.44 | 34.09 | 40.07 |
| MambaIR | 38.57 | 34.67 | 34.15 | 40.28 |
| **MaIR** | **38.56** | **34.75** | **34.19** | **40.30** |
| **MaIR+** | **38.62** | **34.82** | **34.38** | **40.48** |

### 主实验 — 经典超分辨率（×4）

| 方法 | Set5 PSNR | Urban100 PSNR | Manga109 PSNR |
|---|---|---|---|
| SwinIR | 32.92 | 27.45 | 32.03 |
| SRFormer | 32.93 | 27.68 | 32.21 |
| MambaIR | 33.03 | 27.68 | 32.32 |
| **MaIR+** | **33.15** | **27.94** | **32.64** |

### 消融实验

论文补充材料中提供了 NSS 与其他扫描策略的对比，以及 SSA 与简单求和的对比，结论：
- NSS 相比 Z-shaped、S-shaped、Window-based 扫描在所有数据集上均有提升
- SSA 相比像素级求和在 Urban100 ×2 上提升约 0.1-0.2 dB

### 关键发现

1. **同时保持局部性和连续性**是 Mamba 图像恢复的关键，缺一不可
2. **SSA 通道级注意力聚合**优于简单求和，不同通道对不同方向的依赖确实不同
3. **Shift-stripe** 类似 shift window 的效果，有效解决边界问题
4. MaIR 在 4 个任务 14 个数据集上超越 40+ 基线，包括 CNN、Transformer 和 Mamba 方法
5. MaIR 的 NSS 和 SSA 都是零/低额外计算开销的设计

## 亮点与洞察

- NSS 是一个极其优雅的设计：用条带（而非窗口）保持局部性，避免了窗口内 Z 形路径破坏连续性的问题
- "局部性"和"连续性"的显式分解对 Mamba 扫描策略设计具有指导意义
- SSA 的 shuffle-groupconv-unshuffle 流程简洁高效，灵感来自 ShuffleNet 的通道洗牌
- 覆盖 4 个任务 14 个数据集的全面评估增强了结论的可信度

## 局限与展望

- 条带宽度 $w_s$ 为固定超参，可能对不同分辨率/任务不是最优
- 仍然需要 4 个方向扫描（虽然 SSA 提供了更好的聚合方式），计算量是单方向的 4 倍
- 对高分辨率图像的效率分析缺乏详细数据
- 未与最新的 attention-SSM 混合方法（如 MambaIRv2）对比

## 相关工作与启发

- LocalMamba 提出窗口扫描保持局部性但牺牲了连续性；Zigma 提出 S 形保持连续性但忽略局部性——本文是两者的统一
- Shift window（Swin Transformer）的思想被迁移到条带（shift-stripe）
- ShuffleNet 的通道洗牌操作被创造性地用于序列聚合
- 启发：图像 → 序列的展平方式本身就是 Mamba 视觉模型设计的核心问题

## 评分

⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MambaIRv2: Attentive State Space Restoration](mambairv2_attentive_state_space_restoration.md)
- [\[CVPR 2025\] Detail-Preserving Latent Diffusion for Stable Shadow Removal](detail-preserving_latent_diffusion_for_stable_shadow_removal.md)
- [\[AAAI 2026\] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](../../AAAI2026/image_restoration/depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)
- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)

</div>

<!-- RELATED:END -->
