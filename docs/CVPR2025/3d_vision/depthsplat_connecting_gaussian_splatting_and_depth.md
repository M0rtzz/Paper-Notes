---
title: >-
  [论文解读] DepthSplat: Connecting Gaussian Splatting and Depth
description: >-
  [CVPR 2025][3D视觉][高斯泼溅] 将高斯泼溅（3DGS）和深度估计两个通常独立研究的任务统一起来：利用预训练单目深度特征增强多视角深度模型以改善 3DGS 重建质量，同时用 3DGS 的光度渲染损失作为无监督预训练目标来学习强大的深度模型，双任务在多个数据集上均达到 SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 高斯泼溅
  - 深度估计
  - 多视角重建
  - 前馈3D重建
  - 单目深度先验
---

# DepthSplat: Connecting Gaussian Splatting and Depth

**会议**: CVPR 2025  
**arXiv**: [2410.13862](https://arxiv.org/abs/2410.13862)  
**代码**: [https://github.com/cvg/depthsplat](https://github.com/cvg/depthsplat)  
**领域**: 3D视觉  
**关键词**: 高斯泼溅、深度估计、多视角重建、前馈3D重建、单目深度先验

## 一句话总结

将高斯泼溅（3DGS）和深度估计两个通常独立研究的任务统一起来：利用预训练单目深度特征增强多视角深度模型以改善 3DGS 重建质量，同时用 3DGS 的光度渲染损失作为无监督预训练目标来学习强大的深度模型，双任务在多个数据集上均达到 SOTA。

## 研究背景与动机

前馈 3DGS 方法（如 MVSplat）依赖多视角特征匹配来定位 3D 高斯中心，但在遮挡、无纹理区域和反射表面等场景下效果不佳。另一方面，单目深度模型（如 Depth Anything V2）虽然在多样场景下预测鲁棒，但缺乏跨视角尺度一致性，限制了其在 3D 重建中的应用。本文的核心洞察是：**多视角深度的几何一致性和单目深度的鲁棒先验之间是互补的**，将两者融合可以同时提升两个任务。

## 方法详解

### 整体框架

DepthSplat 采用共享架构连接深度估计和 3DGS。模型包含两个分支：多视角特征匹配分支（构建代价体积）和单目深度特征提取分支（来自 Depth Anything V2）。两分支的输出拼接后通过 2D U-Net 回归深度，再反投影到 3D 作为高斯中心，附加轻量头预测其他高斯参数。

### 关键设计

1. **多视角特征匹配分支（代价体积构建）**:
    - 功能：建模多视角几何一致性信息
    - 核心思路：使用轻量 ResNet 提取特征，经多视角 Swin Transformer 交叉注意力交换信息，再通过 plane-sweep stereo 构建代价体积 $\bm{C}_i \in \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times D}$，其中 $D$ 为深度候选数
    - 设计动机：代价体积编码了多视角光度一致性，是多视角深度估计的核心，但无法处理无纹理和反射区域

2. **单目深度特征融合**:
    - 功能：为困难区域提供鲁棒的深度先验
    - 核心思路：直接使用 Depth Anything V2 的 ViT 特征（非显式深度图），双线性插值到与代价体积相同分辨率后在通道维度简单拼接，送入 U-Net 进行深度回归
    - 设计动机：相比复杂融合策略（如注意力融合、显式尺度对齐），简单拼接效果出奇好（Tab. 3），且避免了误差传播。关键在于使用特征而非深度预测值，让网络自适应学习融合方式

3. **分层匹配（Hierarchical Matching）与无监督预训练**:
    - 功能：提升分辨率和深度精度；利用 3DGS 作为无监督目标预训练深度模型
    - 核心思路：采用 2-scale 分层架构（1/8 + 1/4 分辨率），粗到细逐步精化深度；3DGS 渲染损失仅需光度监督即可端到端训练整个深度模型，实现在大规模多视角数据集上的无监督预训练
    - 设计动机：分层匹配提升效率和精度；无监督预训练突破了深度标注的瓶颈，预训练后微调显著优于从零训练

### 损失函数 / 训练策略

- **深度估计**：$L_{\text{depth}} = \alpha |D_{\text{pred}} - D_{\text{gt}}| + \beta (|\partial_x D| + |\partial_y D|)$，$\alpha=\beta=20$（逆深度空间）
- **3DGS 渲染**：$L_{\text{gs}} = \text{MSE} + 0.05 \cdot \text{LPIPS}$
- 单目 ViT backbone 学习率 $2\times10^{-6}$（低学习率保留预训练知识），其他层 $2\times10^{-4}$

## 实验关键数据

### 主实验

| 数据集 | 指标 | DepthSplat | 之前SOTA | 提升 |
|--------|------|-----------|----------|------|
| ScanNet (2-view depth) | Abs Rel | 3.8 | 4.7 (NeuralRecon) | -19% |
| RealEstate10K (2-view GS) | PSNR | 27.47 | 26.39 (MVSplat) | +1.08dB |
| DL3DV (2-view GS) | PSNR | 21.28 | 19.28 (MVSplat) | +2.00dB |

### 消融实验

| 配置 | Abs Rel↓ | PSNR↑ | 说明 |
|------|---------|-------|------|
| Full (ViT-S, 1-scale) | 8.46 | 26.84 | 基线 |
| w/o 单目特征 | 12.25 | 26.04 | 深度下降严重，证明单目先验关键 |
| w/o 代价体积 | 11.34 | 23.24 | PSNR 暴降，多视角一致性不可或缺 |
| ViT-L, 2-scale | **5.57** | **27.47** | 更大 backbone + 分层匹配最优 |
| 无预训练 + 微调 | 10.86 | - | 从零训练 |
| GS 预训练 + 微调 | **10.20** | - | 无监督预训练提升深度 |

### 关键发现

- 去掉单目特征对深度的伤害（+3.79 Abs Rel）大于去掉代价体积（+2.88），但代价体积对 3DGS 更关键（PSNR差 3.6dB）
- 无监督 GS 预训练 → 微调策略在 TartanAir、ScanNet、KITTI 上均优于随机初始化训练
- 支持 12 个输入视角（512×960 分辨率）在 0.6 秒内完成前馈重建

## 亮点与洞察

- **简洁有效的融合设计**：简单拼接预训练单目特征与代价体积的效果超越各种复杂融合方案，体现了"less is more"的设计哲学
- **双向互利**：深度 → 3DGS 和 3DGS → 深度 的双向提升形成良性循环
- 4 GPU 训练 2 天即可达到 SOTA，远低于 GS-LRM 等方法（64 A100 训 2 天）

## 局限与展望

- 无监督预训练的效果在"容易"数据集（如 ScanNet）上提升有限，主要在困难场景显著
- 当前仅使用 top-2 邻近视角做交叉注意力和代价体积，更多视角的高效利用值得探索
- 单目 backbone 冻结大部分参数，联合训练可能进一步提升

## 相关工作与启发

- 与 MVSplat、pixelSplat 竞争，通过单目先验融合实现了显著超越
- 与 TranSplat 的关键区别在于早期融合 vs 晚期精炼，避免了误差传播
- 无监督深度预训练思路可推广到更多几何学习任务（法线估计、光流等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 将深度和 3DGS 双向连接的思路清晰有力，但各组件较为直接
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多任务、多消融，实验设计全面严谨
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰、图表出色，方法动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 提供了一个强大且高效的统一框架，代码开源，可复现性强

<!-- RELATED:START -->

## 相关论文

- [DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)
- [Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries](blurry-edges_photon-limited_depth_estimation_from_defocused_boundaries.md)

<!-- RELATED:END -->
