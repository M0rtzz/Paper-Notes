---
title: >-
  [论文解读] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][运动去模糊] 提出 DiET-GS 双阶段框架，通过事件双积分（EDI）先验和预训练扩散模型联合约束 3DGS 优化，从模糊多视角图像和事件流中重建清晰的 3D 表示，实现精确色彩和精细细节的高质量新视角合成。
tags:
  - CVPR 2025
  - 3D视觉
  - 运动去模糊
  - 3D高斯泼溅
  - 事件相机
  - 扩散先验
  - 新视角合成
---

# DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.24210](https://arxiv.org/abs/2503.24210)  
**代码**: [https://diet-gs.github.io](https://diet-gs.github.io)  
**领域**: 3D视觉  
**关键词**: 运动去模糊、3D高斯泼溅、事件相机、扩散先验、新视角合成

## 一句话总结

提出 DiET-GS 双阶段框架，通过事件双积分（EDI）先验和预训练扩散模型联合约束 3DGS 优化，从模糊多视角图像和事件流中重建清晰的 3D 表示，实现精确色彩和精细细节的高质量新视角合成。

## 研究背景与动机

从模糊多视角图像重建清晰 3D 表示是计算机视觉的长期挑战。事件相机因其高动态范围和微秒级时间分辨率，为运动去模糊提供了独特优势。然而现有方法存在两个关键问题：(1) 仅依赖模糊图像恢复颜色会导致不准确的色彩还原；(2) 事件流虽能提供无模糊细节但容易引入伪影（由未知阈值 $\Theta$ 和事件噪声累积导致）。本文的核心思路是：**"模型驱动"的 EDI 先验提供精确的物理约束，"数据驱动"的扩散先验提供自然图像分布的正则化，两者互补可达到最优视觉质量。**

## 方法详解

### 整体框架

DiET-GS 包含两个阶段：Stage 1 (DiET-GS) 通过事件流 EDI 约束和扩散 RSD 损失联合优化去模糊 3DGS；Stage 2 (DiET-GS++) 冻结 Stage 1 参数，引入额外可学习高斯特征 $\mathbf{f_g}$，最大化扩散先验的效果以增强边缘细节。

### 关键设计

1. **多重 EDI 约束体系**:
    - 功能：从事件流中提取精确的颜色和细节监督
    - 核心思路：三层 EDI 约束协同工作：
        - $\mathcal{L}_{\text{edi\_gray}}$：在亮度域通过可学习 CRF 函数恢复精细细节
        - $\mathcal{L}_{\text{edi\_color}}$：在 RGB 空间逐通道去模糊恢复精确色彩
        - $\mathcal{L}_{\text{edi\_simul}}$：用模拟模糊图像替代真实模糊图像构建 EDI 约束，确保目标函数间的循环一致性
    - 设计动机：$\mathcal{L}_{\text{edi\_gray}}$ 引入可学习 CRF 弥补 RGB 值与像素强度之间的差异，比直接将每个 RGB 通道当作亮度更贴近真实世界；两种约束互补补偿（灰度 → 细节但色彩偏差，RGB → 准确色彩但过平滑）

2. **Renoised Score Distillation (RSD) 扩散先验**:
    - 功能：利用预训练扩散模型的自然图像先验正则化渲染结果
    - 核心思路：将渲染的模糊图像编码到潜空间 $\mathbf{z}_0 = \mathcal{E}(\hat{\mathbf{C}}^B)$，在时间步 $t$ 和 $t-1$ 加噪得到 $\mathbf{z}_t, \mathbf{z}_{t-1}$，用扩散 UNet 预测去噪结果 $\hat{\mathbf{z}}_{t-1}$，优化 $\|\mathbf{z}_{t-1} - \hat{\mathbf{z}}_{t-1}\|$
    - 设计动机：事件流约束虽精确但易产生不自然伪影，扩散先验提供自然性约束；用模糊 GT 作条件替代不可用的清晰图像

3. **Stage 2 可学习潜空间残差（DiET-GS++）**:
    - 功能：最大化扩散先验效果，进一步增强边缘细节
    - 核心思路：为每个 3D 高斯附加零初始化特征 $\mathbf{f_g} \in \mathbb{R}^D$，通过 3DGS 渲染得到 2D 特征图 $\mathbf{f}_{2D}$，与编码的渲染图像 $\mathbf{z}_0$ 相加得到精化潜变量 $\mathbf{z}'_0 = \mathbf{z}_0 + \mathbf{f}_{2D}$，仅用 RSD 损失优化 $\mathbf{f_g}$
    - 设计动机：Stage 1 中事件约束和 RSD 达到平衡会削弱扩散效果；Stage 2 冻结原参数仅训练残差，避免破坏 Stage 1 学到的事件先验，同时利用 3DGS 渲染能力直接在新视角获取潜空间残差（比 DiSR-NeRF 更简洁）

### 损失函数 / 训练策略

- **Stage 1**: $\mathcal{L}_{s1} = \lambda_{\text{blur}} \mathcal{L}_{\text{blur}} + \lambda_{\text{ev}} \mathcal{L}_{\text{ev}} + \lambda_{\text{edi}} \mathcal{L}_{\text{edi}} + \lambda_{\text{rsd}} \mathcal{L}_{\text{rsd}}$
    - $\lambda_{\text{blur}} = \lambda_{\text{edi}} = \lambda_{\text{rsd}} = 1.0$，$\lambda_{\text{ev}} = 0.1$
    - 训练 100K 迭代
- **Stage 2**: 仅 RSD 损失，训练 2K 迭代（≤20 分钟），线性递减时间调度
- 初始化使用 EDI 恢复的清晰图像进行 SfM

## 实验关键数据

### 主实验

| 数据集 | 指标 | DiET-GS | DiET-GS++ | 之前SOTA (Ev-DeblurNeRF) | 提升 |
|--------|------|---------|-----------|------------------------|------|
| EvDeblur-blender | PSNR↑ | **26.69** | 26.23 | 24.76 | +1.93dB |
| EvDeblur-blender | LPIPS↓ | 0.1064 | **0.1052** | 0.1788 | -41% |
| EvDeblur-blender | MUSIQ↑ | 57.67 | **59.91** | 42.38 | +41% |
| EvDeblur-CDAVIS | PSNR↑ | **34.22** | 33.16 | 32.30 | +1.92dB |
| EvDeblur-CDAVIS | LPIPS↓ | **0.0496** | 0.0502 | 0.0571 | -13% |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | 说明 |
|------|-------|-------|--------|--------|------|
| $\mathcal{L}_{\text{blur}}$ only | 29.73 | 0.7797 | 0.2160 | 24.77 | 仅模糊重建 |
| + $\mathcal{L}_{\text{ev}}$ | 32.74 | 0.8460 | 0.1173 | 39.69 | 事件约束 +3.01dB |
| + edi_gray + edi_color | 34.92 | 0.9033 | 0.0624 | 43.79 | EDI 双约束互补 |
| + edi_simul | 35.04 | 0.9068 | 0.0587 | 45.04 | 循环一致性正则 |
| + RSD (S1) | 34.89 | 0.9049 | 0.0600 | 45.37 | S1扩散略降PSNR |
| + RSD (S2, DiET-GS++) | 33.86 | 0.8846 | 0.0634 | **51.71** | 无参考质量飞升 |

### 关键发现

- EDI gray 和 color 约束互补：gray 恢复纹理细节但有色偏，color 恢复精确颜色但过平滑
- Stage 2 (DiET-GS++) 在 PSNR/SSIM 上略降但在无参考质量指标（MUSIQ +6.34, CLIP-IQA +0.037）上显著提升
- 循环一致性正则化 $\mathcal{L}_{\text{edi\_simul}}$ 在视觉上有效抑制了局部伪影

## 亮点与洞察

- **多层次 EDI 约束的设计思路精巧**：亮度域恢复细节 + RGB 域恢复色彩 + 模拟域正则化，三管齐下形成完整约束体系
- **两阶段训练策略的必要性**：巧妙解决了事件约束和扩散先验联合优化时会相互削弱的问题
- 利用 3DGS 的显式渲染能力直接获取可学习潜空间残差，比 DiSR-NeRF 的方案更简洁且无需额外同步步骤

## 局限与展望

- DiET-GS++ 由生成模型驱动，可能产生与 GT 不完全一致的细节（PSNR 下降）
- 事件阈值 $\Theta$ 的设定仍需手动调节（合成 0.2，真实 0.25）
- 依赖 SfM 初始化，虽然通过 EDI 预处理缓解了模糊问题但仍有局限
- Stage 2 仅训练 2K 迭代，更长训练或更大模型可能进一步提升

## 相关工作与启发

- 与 Ev-DeblurNeRF 的核心区别：在亮度域引入可学习 CRF 的 EDI 约束，比直接将 RGB 视为亮度更合理
- 与 DiSR-NeRF 的核心区别：利用 3DGS 直接渲染潜空间残差，更简洁高效
- 启发：事件相机 + 扩散模型的组合范式可推广到其他退化问题（如低光、HDR）

## 评分

- 新颖性: ⭐⭐⭐⭐ EDI 多约束体系和两阶段策略设计精巧，但各组件独立来看不算全新
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集评估充分，消融详细，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导详细，但部分符号使用略显冗余
- 价值: ⭐⭐⭐⭐ 在事件辅助去模糊 3D 重建这一小众领域推进了 SOTA，实用性受限于事件相机普及度
