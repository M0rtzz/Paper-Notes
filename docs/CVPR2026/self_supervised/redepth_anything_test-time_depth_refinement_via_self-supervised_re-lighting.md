---
title: >-
  [论文解读] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting
description: >-
  [CVPR 2026][自监督学习][单目深度估计] 提出 Re-Depth Anything，通过在推理时对预测深度图进行重光照增强并利用 2D 扩散模型的 SDS 损失进行自监督优化，在无标签的情况下精细化 Depth Anything V2/3 的深度预测。
tags:
  - CVPR 2026
  - 自监督学习
  - 单目深度估计
  - 测试时优化
  - Score Distillation Sampling
  - 重光照
  - Depth Anything
---

# Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting

**会议**: CVPR 2026  
**arXiv**: [2512.17908](https://arxiv.org/abs/2512.17908)  
**作者**: Ananta R. Bhattarai, Helge Rhodin (Bielefeld University)  
**代码**: [GitHub](https://github.com/anantarb/Re-Depth-Anything)  
**领域**: 3D视觉  
**关键词**: 单目深度估计, 测试时优化, Score Distillation Sampling, 重光照, Depth Anything

## 一句话总结

提出 Re-Depth Anything，通过在推理时对预测深度图进行重光照增强并利用 2D 扩散模型的 SDS 损失进行自监督优化，在无标签的情况下精细化 Depth Anything V2/3 的深度预测。

## 背景与动机

Depth Anything V2 (DA-V2) 等基础模型虽然性能优异，但对与训练分布差距较大的真实图像仍存在误差（如光照偏差导致微结构丢失、平坦区域出现伪噪声）。现有测试时适应方法要么需要多帧时序信息，要么依赖特定外部先验（3D 网格/稀疏点）。而大规模 2D 扩散模型学到了丰富的物理世界先验，尚未被充分利用于深度估计的测试时精细化。

## 核心问题

如何利用 2D 扩散模型先验，在**单张图像**上进行**无标签**的深度图测试时精细化？

## 方法详解

### 核心思路：重光照 vs 光度重建

关键创新在于**不尝试**重建输入图像的真实光照/材质（这是极度病态的逆渲染问题），而是**增强**输入图像——随机重光照深度图并叠加到原图上，利用扩散模型判断增强后的图像是否"看起来合理"。

### 4.1 深度-光照渲染

从视差图 $\hat{D}_{\text{disp}}$ 计算法线 $\mathbf{N}$，然后用 Blinn-Phong 模型重光照：

$$\hat{\mathbf{I}} = \tau\left(\beta_1 \max(\mathbf{N} \cdot \mathbf{l}, 0) \odot \tau^{-1}(\mathbf{I}) + \beta_2 \max(\mathbf{N} \cdot \mathbf{h}, 0)^\alpha\right)$$

- $\mathbf{l}$：随机采样光照方向
- $\beta_1, \beta_2$：漫反射/镜面反射强度
- $\tau(\cdot) = (\cdot)^{1/\gamma}$：色调映射（$\gamma=2.2$）
- 输入图像 $\mathbf{I}$ 直接作为漫反射 albedo 的近似

**处理相对深度**：DA-V2 输出归一化视差，法线对全局尺度不变，仅需优化偏移参数 $b = ms$（实际固定 $b=0.1$ 即可）。

### 4.2 增强目标

每步随机采样 $\mathbf{l}$, $\beta_1$, $\beta_2$, $\alpha$，用 SDS 损失衡量增强图像的合理性：

$$\mathcal{L} = \mathcal{L}_{\text{SDS}}(\hat{\mathbf{I}}, c) + \frac{\lambda_1}{hw} \sum_{i,j} \|\Delta \hat{D}_{\text{disp}}^{i,j}\|^2$$

- SDS 损失：扩散模型（Stable Diffusion v1.5）评估重光照图像的自然度
- 平滑正则：抑制深度噪声
- 文本条件 $c$：由 BLIP-2 从输入图像自动生成描述

### 4.3 优化方案

直接优化深度张量或全模型微调都会失败。**关键设计**：仅优化中间嵌入 $\mathbf{W}$ + DPT 解码器权重 $\theta$，冻结 ViT 编码器：

$$\mathbf{W}^*, \theta^* = \arg\min_{\mathbf{W}, \theta} \mathcal{L}(\hat{\mathbf{I}}, c, \hat{D}_{\text{disp}})$$

- 保留了 ViT 编码器学到的强几何先验
- 通过解码器权重允许结构性调整
- 通过嵌入优化允许逐图像定制

**集成预测**：SDS 损失随机性大，运行 $N=10$ 次独立优化后取均值，以 3 次运行即可获得大部分收益。

### 实现细节

- 1000 次 AdamW 迭代，嵌入学习率 $10^{-3}$，DPT 权重学习率 $2 \times 10^{-6}$
- 缩放正交投影（默认），正则化 $\lambda_1 = 1.0$
- 单次运行约 80 秒（RTX 5000）

## 实验关键数据

| 数据集 | 方法 | AbsRel ↓ | RMSE ↓ | SI log ↓ | SqRel ↓ |
|--------|------|---------|--------|---------|---------|
| KITTI | DA-V2 | 0.305 | 7.01 | 33.6 | 2.49 |
| | **Ours + DA-V2** | **0.283** | **6.71** | **30.7** | **2.20** |
| | 相对提升 | 7.10% | 4.29% | 8.51% | 11.4% |
| ETH3D | DA-V2 | 0.113 | 0.955 | 15.1 | 0.391 |
| | **Ours + DA-V2** | **0.104** | **0.875** | **14.1** | **0.347** |
| | 相对提升 | 8.30% | 8.39% | 6.22% | 11.1% |

| 数据集 | 方法 | AbsRel ↓ | SqRel ↓ | Normal MSE ↓ |
|--------|------|---------|---------|-------------|
| CO3D | DA3 | 0.00251 | 0.000317 | 0.000479 |
| | **Ours + DA3** | **0.00238** | **0.000294** | **0.000409** |
| | 相对提升 | 4.83% | 7.39% | **14.65%** |

*在 DA3 上也取得一致提升，法线误差改善高达 14.7%*

## 亮点

- **重光照而非光度重建**：巧妙回避了逆渲染的病态性，将扩散先验用于"增强合理性检查"
- **不需要任何标签、多视图或额外数据**：纯粹的单图自监督
- **适用于不同 backbone**：在 DA-V2 和 DA3 上都有效
- **细节增强显著**：增强了球面纹理、阳台栏杆等细节，去除平坦面伪噪声
- **理论优雅**：将 DreamFusion 的 SfS+SDS 范式从 text-to-3D 迁移到深度精细化

## 局限与展望

- 偶尔出现幻觉边缘（如卡车上的贴纸被误认为几何特征）
- 天空区域可能被错误延伸几何
- 暗区域（如树木阴影中）可能过度平滑
- 10 次运行集成耗时约 13 分钟，对实时应用不友好
- 使用 Stable Diffusion v1.5 作为先验，可能已非最优选择

## 与相关工作的对比

- vs **DreamFusion/RealFusion**：这些方法做完整的 3D 重建+光度重建；本文只做重光照增强，避免了光度一致性的严苛要求
- vs **经典 SfS**：SfS 假设 albedo 恒定/光照已知，失败案例多；本文用扩散先验取代手工假设
- vs **Marigold**：Marigold 用扩散模型直接做深度估计；本文用扩散模型做测试时优化，与前馈方法互补
- vs **多帧 TTA**：本文的单图方案不需要时序信息，应用范围更广

## 启发与关联

- 重光照增强 + SDS 的范式可推广到法线估计、表面重建等几何任务
- "不重建而增强"是处理病态逆问题的重要策略转变
- 冻结编码器 + 优化嵌入+解码器的方案在其他基础模型适应中也可借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 重光照 SDS 的自监督深度精细化是全新的
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集 + DA-V2/DA3 双 backbone + 详细消融
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰，方法优雅，分析透彻
- 价值: ⭐⭐⭐⭐ — 为基础模型的测试时精细化开辟了新路径

<!-- RELATED:START -->

## 相关论文

- [AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [Test-Time Canonicalization by Foundation Models for Robust Perception](../../ICML2025/self_supervised/test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](../../ICML2025/self_supervised/update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [Test-Time Training Provably Improves Transformers as In-Context Learners](../../ICML2025/self_supervised/test-time_training_provably_improves_transformers_as_in-context_learners.md)
- [TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)

<!-- RELATED:END -->
