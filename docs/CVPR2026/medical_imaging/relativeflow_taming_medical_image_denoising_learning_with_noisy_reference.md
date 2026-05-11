---
title: >-
  [论文解读] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference
description: >-
  [CVPR 2026][医学图像][去噪] 提出 RelativeFlow，基于 flow matching 的框架，通过将绝对噪声到干净映射分解为相对更噪到噪声映射，结合一致传输约束和基于模拟的速度场，从异质噪声参考中学习统一的去噪流，突破参考偏差限制。
tags:
  - CVPR 2026
  - 医学图像
  - 去噪
  - flow matching
  - noisy reference
---

# RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference

**会议**: CVPR 2026  
**arXiv**: [2604.15459](https://arxiv.org/abs/2604.15459)  
**代码**: [github.com/Deliver0/RelativeFlow](https://github.com/Deliver0/RelativeFlow)  
**领域**: 医学影像  
**关键词**: medical image denoising, flow matching, noisy reference, CT denoising, MR denoising

## 一句话总结

提出 RelativeFlow，基于 flow matching 的框架，通过将绝对噪声到干净映射分解为相对更噪到噪声映射，结合一致传输约束和基于模拟的速度场，从异质噪声参考中学习统一的去噪流，突破参考偏差限制。

## 研究背景与动机

医学图像去噪 (MID) 缺乏绝对干净的图像用于监督——只有来自不同采集协议和扫描仪配置的相对高质量参考，其质量在类别间异质变化。现有三种范式的局限：(1) SimSDL 将噪声参考当干净目标，导致次优收敛或参考偏差学习；(2) SSL 的独立噪声假设在医学成像中难以满足；(3) SimSGL 同样天真地将噪声参考当生成目标。核心问题：如何从异质质量的噪声参考中学习统一的高质量去噪映射。

## 方法详解

### 整体框架

RelativeFlow 将绝对去噪流分解为多个相对去噪流的组合。对每个噪声参考 $x_t$，用退化算子生成更噪样本 $x_{t-\Delta t}$，构建局部相对去噪步。两个关键组件保证相对流的一致性和可学习性。

### 关键设计

1. **一致传输 (CoT) 位移映射**: 通过指数时间空间的线性插值定义概率路径 $p_t = \lambda p_{t_i} + (1-\lambda)p_{t_j}$，权重 $\lambda = \frac{e^{-t}-e^{-t_j}}{e^{-t_i}-e^{-t_j}}$。这保证嵌套插值的传递性——从任意质量 $t_i$ 到 $t_j$ 的相对流都是绝对流 $\psi_{0 \to +\infty}$ 的组成部分，且可依次组合为绝对流。数学证明了两个性质：相对流是绝对流的分量，以及连续相对流可逐步组合为绝对流。

2. **基于模拟的速度场 (SVF)**: 利用模态特定的退化算子 $D_{\Delta t}$（CT 的泊松-高斯噪声、MR 的 Rician 噪声）从噪声参考模拟更噪样本，构建速度场训练目标 $u = \frac{x_t - D_{\Delta t}(x_t)}{e^{\Delta t} - 1}$。训练损失为神经网络预测与目标速度的 L2 距离。

3. **渐进式步长课程**: 训练过程中逐步扩大退化步长范围 $[\Delta t_{min}, \Delta t_{max}]$（每轮乘以/除以系数 $\alpha$），使模型逐渐学习从小退化到大退化的速度场，覆盖完整质量谱。采样时用多步欧拉积分。

### 损失函数 / 训练策略

训练损失 $\mathcal{L}_{RF} = \mathbb{E}_{x_t, \Delta t}\left[\left\|\mathcal{N}_\theta(D_{\Delta t}(x_t), \Delta t) - \frac{x_t - D_{\Delta t}(x_t)}{e^{\Delta t} - 1}\right\|_2^2\right]$。渐进式课程学习策略。

## 实验关键数据

### 主实验

在 CT 和 MR 去噪任务上与 10 种方法（NID + MID）对比：

| 任务 | 指标 | 最优先前方法 | RelativeFlow |
|------|------|------------|-------------|
| CT 去噪 | PSNR/SSIM | 次优 | **最优** |
| MR 去噪 | PSNR/SSIM | 次优 | **最优** |

在 PSNR、SSIM、RMSE、LPIPS 四个指标上全面最优。

### 消融实验

- CoT 一致性约束是突破参考偏差的核心
- SVF 的模态特定退化建模优于通用噪声模型
- 渐进式步长课程优于固定步长训练

### 关键发现

- RelativeFlow 将不同质量水平的图像统一去噪到一致高质量
- CoT 的数学性质（分量性和组合性）为统一去噪提供了理论保证
- 模态特定退化算子使框架可扩展到不同医学成像模态

## 亮点与洞察

- 首次将噪声参考问题形式化并通过 flow matching 系统解决
- CoT 的数学分析（分量性+组合性证明）理论深度出色
- 跨 CT 和 MR 两种模态的验证展示了框架的通用性

## 局限与展望

- 退化算子的设计需要领域知识（泊松-高斯 vs Rician）
- 采样时的多步积分比判别式方法慢
- 理论上的 $t \to +\infty$ 极限在实践中的收敛质量需验证

## 相关工作与启发

- 相对流组合的思路对其他缺乏干净标签的学习场景有启发
- CoT 的数学框架可推广到其他渐进式质量提升任务
- 模态特定退化建模与通用 flow matching 的结合范式值得推广

## 评分

8/10 — 理论深度和实际效果兼备，是医学图像去噪领域的重要方法论贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graphbased_learning_of_inter_and_intraview_de.md)

</div>

<!-- RELATED:END -->
