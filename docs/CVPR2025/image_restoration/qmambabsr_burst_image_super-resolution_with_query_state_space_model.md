---
title: >-
  [论文解读] QMambaBSR: Burst Image Super-Resolution with Query State Space Model
description: >-
  [CVPR 2025][图像恢复][连拍超分辨率] 提出 QMambaBSR，通过 Query State Space Model（QSSM）实现帧间查询和帧内扫描的联合子像素提取与噪声抑制，结合自适应上采样模块，在合成和真实连拍超分辨率任务上达到 SOTA。
tags:
  - CVPR 2025
  - 图像恢复
  - 连拍超分辨率
  - 状态空间模型
  - 子像素提取
  - 自适应上采样
  - 多帧去噪
---

# QMambaBSR: Burst Image Super-Resolution with Query State Space Model

**会议**: CVPR 2025  
**arXiv**: [2408.08665](https://arxiv.org/abs/2408.08665)  
**代码**: 无  
**领域**: 图像超分辨率 / 多帧融合  
**关键词**: 连拍超分辨率, 状态空间模型, 子像素提取, 自适应上采样, 多帧去噪

## 一句话总结

提出 QMambaBSR，通过 Query State Space Model（QSSM）实现帧间查询和帧内扫描的联合子像素提取与噪声抑制，结合自适应上采样模块，在合成和真实连拍超分辨率任务上达到 SOTA。

## 研究背景与动机

连拍超分辨率（BurstSR）旨在从手持拍摄的多帧低分辨率图像中融合亚像素信息以重建高分辨率图像，是克服智能手机传感器和镜头限制的重要技术。该领域面临两个核心挑战：

1. **子像素提取与噪声区分困难**：连拍 RAW 图像中同时包含有用的子像素信息和高频随机噪声。现有方法（如加权融合、逐帧交叉注意力）采用逐帧处理方式，无法有效利用"子像素在多帧中具有一致空间分布、而噪声随机出现"这一关键特性，导致提取不准确
2. **静态上采样无法适应场景变化**：现有 SOTA 方法（如 Burstormer、BIPNet）使用固定的插值、转置卷积或 PixelShuffle 进行上采样，无法感知不同场景中子像素的空间分布特点，导致细节过度平滑

本文的核心观察是：有效子像素在所有帧的对应位置具有一致的强度，而噪声仅随机出现在某些帧上。因此，同时考虑整个连拍序列进行融合可以更可靠地提取一致的子像素并抑制噪声异值。

## 方法详解

### 整体框架

QMambaBSR 的 pipeline 包含三个阶段：(1) 对齐阶段——使用现有对齐模块将当前帧与基准帧对齐；(2) 融合阶段——通过 QSSM 模块进行帧间查询和帧内扫描的联合子像素提取，并使用多尺度融合模块（MSFM）整合不同尺度的子像素信息；(3) 上采样阶段——通过自适应上采样模块（AdaUp）根据场景特征动态调整上采样核，重建高质量高分辨率图像。

### 关键设计

1. **Query State Space Model (QSSM)**:
    - 功能：同时从所有当前帧中提取与基准帧匹配的子像素信息，并抑制随机噪声
    - 核心思路：修改 SSM 中的控制矩阵 $B$ 和离散化步长 $\Delta$，使其由基准帧生成而非输入帧。具体地，首先将基准帧与所有当前帧在通道维度拼接并通过 MLP 融合（初步去噪），然后基准帧通过线性层生成 $\Delta_{base}$ 和 $B_{base}$，作为门控信号控制当前帧特征对状态的影响。所有当前帧特征在通道维度合并后通过线性层统一处理，使基准帧能一次性查询所有当前帧。QSSM 结合四个扫描方向和通道注意力
    - 设计动机：传统交叉注意力仅能逐帧、逐位置查询（$O(N^2)$ 复杂度），而 QSSM 通过 SSM 的递推结构同时实现帧间查询和帧内信息交互，复杂度更低。基准帧位置 $t$ 不仅查询自身位置的当前帧信息，还通过遗忘门/输入门引导邻近位置的查询，形成渐进衰减的感受野

2. **多尺度融合模块（MSFM）**:
    - 功能：融合不同尺度的子像素信息，增强细节重建能力
    - 核心思路：三分支并行设计——3×3 卷积处理局部子像素特征，SSM（水平+垂直扫描）处理轴向全局特征，通道 Transformer 增强全局感知能力。三个分支的输出通过可学习权重加权求和
    - 设计动机：SSM 的 A 矩阵衰减特性限制了远距离感知，Transformer 弥补这一不足；局部卷积捕获细粒度纹理；三者互补覆盖不同尺度

3. **自适应上采样模块（AdaUp）**:
    - 功能：根据当前场景的子像素空间分布动态调整上采样核
    - 核心思路：首先通过自适应池化感知输入特征的通道级子像素分布 $L$，再通过 1×1 卷积获得输出通道分布 $L_1$。将两个分布序列通过广播逐元素乘积应用到转置卷积核 $W$ 上：$W_f = (W \odot L) \odot L_1$，最后用调制后的核进行转置卷积上采样
    - 设计动机：静态上采样核无法适应不同连拍场景中子像素的不同空间排列，AdaUp 使核具有场景感知能力，从而更好地利用子像素信息重建细节

### 损失函数 / 训练策略

训练设置：在 Synthetic BurstSR 数据集上从头训练 300 epochs，AdamW 优化器（$\beta_1$=0.9, $\beta_2$=0.999），余弦退火学习率从 $3 \times 10^{-4}$ 降至 $10^{-6}$，训练 patch 大小 48×48，batch size 8，burst size 14，8 块 V100 GPU。Real BurstSR 微调 60 epochs（lr=$10^{-6}$, patch=56×56）。RealBSR-RAW/RGB 从头训练 100 epochs（patch=80×80）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | QMambaBSR | Burstormer (前SOTA) | 提升 |
|--------|------|-----------|---------------------|------|
| Synthetic BurstSR (×4) | PSNR | **43.12** | 42.83 | +0.29 dB |
| Synthetic BurstSR (×4) | SSIM | **0.97** | 0.97 | - |
| RealBSR-RAW (×4) | PSNR | **27.558** | 27.290 | +0.268 dB |
| RealBSR-RAW (×4) | SSIM | **0.820** | 0.816 | +0.004 |
| RealBSR-RAW (×4) | L-PSNR | **32.791** | 32.533 | +0.258 dB |
| RealBSR-RGB (×4) | PSNR | **31.401** | 31.197 | +0.204 dB |
| RealBSR-RGB (×4) | SSIM | **0.908** | 0.907 | +0.001 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | 说明 |
|------|-------|-------|------|
| Baseline (无模块) | 39.81 | 0.93 | 基础网络 |
| +MSFM | 41.15 | 0.94 | +1.34 dB |
| +MSFM+QSSM | 41.87 | 0.96 | +0.72 dB |
| +MSFM+QSSM+AdaUp | **42.13** | **0.96** | +0.26 dB |

| 融合方法对比 | PSNR↑ | 说明 |
|-------------|-------|------|
| Concat | 39.85 | 简单拼接 |
| PBFF (BIPNet) | 40.57 | 通道融合 |
| NRFE (Burstormer) | 41.72 | 邻域交互 |
| **QSSM+MSFM (Ours)** | **42.13** | +0.41 dB vs NRFE |

### 关键发现

- MSFM 贡献最大（+1.34 dB），表明多尺度子像素融合是关键
- QSSM 在 MSFM 基础上进一步提升 0.72 dB，验证了帧间联合查询的有效性
- AdaUp 相比 PixelShuffle 提升 0.16 dB，适应性上采样有明确增益
- MSFM 内部消融表明三分支（Conv+SSM+Transformer）的组合效果最佳（+0.56 dB vs 纯 Conv）
- 用户研究（20 名志愿者）中平均得分 8.56/10，显著高于 Burstormer 和 BIPNet

## 亮点与洞察

- **QSSM 的查询机制设计精巧**：通过修改 SSM 的 $B$ 和 $\Delta$ 参数来源（从基准帧生成），将 SSM 从自回归模型转变为跨序列查询模型，是 SSM 在多帧任务中的创新应用
- **子像素一致性 vs 噪声随机性**的利用非常自然：联合多帧查询天然地利用了这一先验
- **MSFM 的三分支设计**平衡了局部（CNN）、轴向（SSM）和全局（Transformer）的感受野
- **AdaUp 简洁有效**：仅通过通道级分布调制即可实现场景自适应上采样

## 局限与展望

- 当前方法主要关注融合和上采样阶段，对齐阶段仍使用现有方法，存在进一步优化空间
- 8 块 V100 训练 300 epochs 的计算成本较高
- 仅验证了 ×4 超分辨率，对其他放大倍数的适用性未充分验证
- 作者计划将 SSM 应用于对齐阶段，以及拓展到连拍去噪、HDR 等其他多帧恢复任务

## 相关工作与启发

- **vs Burstormer**: Burstormer 使用交叉注意力逐帧提取子像素，QSSM 能同时查询所有帧并实现帧内信息交互，更高效
- **vs BIPNet**: BIPNet 通过通道 shuffle 促进帧间信息流动，但缺乏显式的子像素提取机制
- **vs RBSR**: RBSR 使用 RNN 逐帧融合，未能区分子像素与噪声的不同特性
- **vs MambaIR**: MambaIR 首次将 SSM 用于图像恢复，但仅处理单帧；QMambaBSR 将 SSM 扩展到多帧查询场景

## 评分

- 新颖性: ⭐⭐⭐⭐ QSSM 将 SSM 创新性地改造为跨帧查询模型，AdaUp 的通道级核调制也是新颖设计
- 实验充分度: ⭐⭐⭐⭐⭐ 四个 benchmark（合成+真实）全面验证，消融实验详尽（模块级+组件级+用户研究）
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，公式展开详细，与交叉注意力的对比分析很有说服力
- 价值: ⭐⭐⭐⭐ 在连拍超分这一重要问题上取得全面 SOTA，对 SSM 在多帧任务中的应用有启发

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] MambaIRv2: Attentive State Space Restoration](mambairv2_attentive_state_space_restoration.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)

<!-- RELATED:END -->
