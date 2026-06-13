---
title: >-
  [论文解读] SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation
description: >-
  [CVPR 2025][3D视觉][单目深度估计] 提出 SharpDepth，通过扩散蒸馏将生成式深度模型（如 Lotus）的精细边缘细节知识注入判别式度量深度模型（如 UniDepth）的预测中，利用噪声感知门控和无标注训练实现度量精度与边缘锐利度的最佳平衡。 - 零样本单目度量深度估计存在两类方法的互补缺陷：(1) 判…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "单目深度估计"
  - "度量深度"
  - "扩散蒸馏"
  - "边缘锐化"
  - "零样本泛化"
---

# SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation

**会议**: CVPR 2025  
**arXiv**: [2411.18229](https://arxiv.org/abs/2411.18229)  
**代码**: 无  
**领域**: 3D视觉 / 深度估计  
**关键词**: 单目深度估计, 度量深度, 扩散蒸馏, 边缘锐化, 零样本泛化

## 一句话总结

提出 SharpDepth，通过扩散蒸馏将生成式深度模型（如 Lotus）的精细边缘细节知识注入判别式度量深度模型（如 UniDepth）的预测中，利用噪声感知门控和无标注训练实现度量精度与边缘锐利度的最佳平衡。

## 研究背景与动机

- 零样本单目度量深度估计存在两类方法的互补缺陷：(1) **判别式方法**（UniDepth、Metric3D）在真实数据稀疏标注上训练，度量精度高但深度图过于平滑、缺乏边缘细节；(2) **生成式方法**（Marigold、Lotus）利用扩散模型先验产生锐利边缘，但只能预测仿射不变深度且存在合成-真实域间差距
- 现有深度精修方法（BetterDepth、PatchRefiner）依赖合成数据集训练，存在域间差距且在真实场景泛化性有限
- 核心问题：能否在不使用真实标注的情况下，将生成式模型的锐利细节蒸馏到度量深度模型中？
- SharpDepth 只需约 90,000 张无标注真实图像训练，是现有方法训练数据量的 1/100-1/150

## 方法详解

### 整体框架

给定输入图像，分别用预训练度量深度模型 $f_D$（UniDepth）和扩散深度模型 $f_G$（Lotus）预测度量深度 $d$ 和仿射不变深度 $\tilde{d}$。计算两者归一化后的差异图 $e$，高差异区域表示需要锐化的区域。SharpDepth 模型 $\mathbf{G}_\theta$ 基于 Lotus 架构初始化，输入为噪声感知门控处理后的深度潜变量和图像潜变量。

### 关键设计

**设计一：噪声感知门控（Noise-aware Gating）**

- **功能**：引导扩散模型精确聚焦于需要锐化的不确定区域
- **核心思路**：计算度量深度和仿射不变深度的差异图 $e$，对深度潜变量 $z_d$ 进行选择性加噪：$z'_d = \hat{e} \odot \epsilon + (1-\hat{e}) \odot z_d$。高差异区域被大量噪声覆盖（需模型重建），低差异区域保留原始信息（无需修改）
- **设计动机**：不同于传统扩散模型对所有像素均匀加噪，选择性加噪让模型能区分可靠区域和不确定区域，将计算资源集中在最需要改进的位置。训练过程中用 EMA 模型替代 $f_G$ 实现迭代精修

**设计二：SDS 深度先验蒸馏**

- **功能**：将预训练扩散深度模型的精细边缘知识注入锐化器
- **核心思路**：修改标准 SDS 损失为 $x_0$-prediction 形式匹配 Lotus：$\nabla_\theta \mathcal{L}_{SDS} \triangleq \mathbb{E}_{t,\epsilon}[w^t(\hat{z} - f_G(\hat{z}^t; z_i, t))]$。无需反向传播通过扩散模型 U-Net，将大型扩散深度先验蒸馏到锐化器中
- **设计动机**：直接使用扩散模型推理太慢且只提供相对深度；SDS 蒸馏可在训练阶段高效提取其图像先验中的高频细节知识

**设计三：噪声感知重建损失**

- **功能**：确保锐化后的深度保持度量精度，防止向仿射不变深度漂移
- **核心思路**：$\mathcal{L}_{recons} = \|e \odot (\hat{d} - d)\|$，使用差异图 $e$ 加权，高差异区域梯度更大，低差异区域几乎无梯度。作为正则化保持输出接近原始度量深度
- **设计动机**：纯 SDS 蒸馏会让网络继承扩散模型的缺陷（如度量不准），重建损失作为锚点防止度量精度退化

### 损失函数

$$\mathcal{L}_{total} = \lambda_{SDS} \cdot \mathcal{L}_{SDS} + \lambda_{recons} \cdot \mathcal{L}_{recons}$$

其中 $\lambda_{SDS} = 1.0$，$\lambda_{recons} = 0.3$。两个损失都通过不同机制聚焦于高差异区域。

## 实验关键数据

### 度量深度精度（零样本）

| 方法 | KITTI δ₁↑ | NYUv2 δ₁↑ | ETH3D δ₁↑ | nuScenes δ₁↑ |
|------|-----------|-----------|-----------|-------------|
| UniDepth | 0.98 | 0.98 | 0.25 | 0.84 |
| Metric3Dv2 | — | — | — | — |
| Lotus (GT对齐) | 0.88 | 0.97 | 0.96 | 0.51 |
| **SharpDepth** | **接近UniDepth** | **接近UniDepth** | **显著提升** | **改善** |

### 深度边缘质量（DBE/PDBE）

| 方法 | Sintel ↓ | UnrealStereo ↓ | Spring ↓ | iBims ↓ |
|------|----------|----------------|----------|---------|
| UniDepth | 高 | 高 | 高 | 高 |
| Lotus | 低 | 低 | 低 | 低 |
| **SharpDepth** | **接近Lotus** | **接近Lotus** | **接近Lotus** | **接近Lotus** |

### 关键发现

- SharpDepth 在精度-锐利度权衡曲线上达到最佳平衡点（Fig. 2）
- 仅用 ~90K 无标注真实图像训练，训练数据量约为判别式方法的 1%
- 噪声感知门控比均匀加噪在所有指标上显著更优
- 使用 EMA 自迭代精修比固定使用 Lotus 效果更好
- 点云重建质量显著优于 UniDepth（如刺猬刺、键盘按键等精细结构）

## 亮点与洞察

1. **无标注深度训练的巧妙设计**：通过两个预训练模型的预测差异作为伪监督信号，完全避免了真实标注的需求
2. **噪声感知门控的深刻洞察**：将传统扩散模型的均匀加噪改为区域自适应加噪，为扩散模型精修任务提供了新范式
3. **SDS + 重建损失的互补设计**：一个追求锐利度，一个锚定精度，优雅解决了两个目标之间的矛盾

## 局限与展望

- 依赖两个预训练深度模型（UniDepth + Lotus），增加了系统复杂度
- 差异图的质量直接依赖两个基础模型的互补性，如果两个模型犯同样的错误则无法检测
- 推理时需要运行 UniDepth + Lotus + SharpDepth 三个模型，延迟较高
- 未来可探索更轻量级的锐化器架构和端到端训练方案

## 相关工作与启发

- **UniDepth** [Piccinelli et al.] 和 **Metric3Dv2** [Hu et al.] 是度量深度的代表方法
- **Marigold** [Ke et al.] 和 **Lotus** [He et al.] 是扩散深度估计的代表
- **SDS** [Poole et al., DreamFusion] 的蒸馏技术从 3D 生成领域被迁移到深度精修
- 本文为"利用生成模型增强判别模型"提供了有价值的范例

## 评分

⭐⭐⭐⭐ — 方法动机清晰、设计优雅，无标注训练方案具有实际价值。噪声感知门控是核心创新点，实验在精度和边缘质量两个维度上均有说服力的验证。但推理时需三个模型增加了部署复杂度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[CVPR 2025\] Seurat: From Moving Points to Depth](seurat_from_moving_points_to_depth.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
