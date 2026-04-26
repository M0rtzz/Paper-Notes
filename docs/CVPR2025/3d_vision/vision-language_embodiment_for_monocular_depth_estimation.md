---
title: >-
  [论文解读] Vision-Language Embodiment for Monocular Depth Estimation
description: >-
  [CVPR 2025][3D视觉][单目深度估计] 提出一种具身深度估计框架，将相机模型的物理特性具身化到深度学习系统中，计算Embodied Scene Depth作为几何先验，同时利用视觉-语言互补（深度文本描述 + 文本VAE + 条件采样器），融合RGB图像特征和物理深度先验进行单目深度估计。
tags:
  - CVPR 2025
  - 3D视觉
  - 单目深度估计
  - 相机模型具身化
  - 视觉语言融合
  - 场景深度先验
  - 变分自编码器
---

# Vision-Language Embodiment for Monocular Depth Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.16535](https://arxiv.org/abs/2503.16535)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 单目深度估计, 相机模型具身化, 视觉语言融合, 场景深度先验, 变分自编码器

## 一句话总结

提出一种具身深度估计框架，将相机模型的物理特性具身化到深度学习系统中，计算Embodied Scene Depth作为几何先验，同时利用视觉-语言互补（深度文本描述 + 文本VAE + 条件采样器），融合RGB图像特征和物理深度先验进行单目深度估计。

## 研究背景与动机

- 单目深度估计是计算机视觉核心问题，但从3D到2D的映射本质上是不适定的
- 现有深度估计模型主要依赖图像间关系进行有监督训练，忽略了相机本身提供的固有信息
- 几何先验（法线约束、平面约束）虽能减少不确定性，但整体影响有限
- 现有的CLIP-based深度估计方法（DepthCLIP等）使用固定的离散深度描述，表达能力不足
- 图像和文本作为两种本质上模糊的模态，各有互补优势：图像提供直接3D观测，文本提供物体尺度先验
- 对于道路等平坦区域，相机模型参数可以直接计算绝对深度，精度极高（>99%像素误差<10%）
- 缺乏将相机物理模型、环境语义和语言先验统一整合的深度估计框架
- 稀疏LiDAR真值限制了可用的深度监督信号密度

## 方法详解

### 整体框架

系统包含三个具身化层次：**相机具身化**——利用相机内外参和道路分割计算Embodied Scene Depth；**语言具身化**——用ExpansionNet-v2生成图像描述，基于分割和Embodied Depth生成深度文本描述，合并为文本VAE的输入；**视觉-语言融合**——RGB编码器和深度编码器通过交叉注意力融合特征，条件采样器从文本VAE的潜在分布中采样，共享权重的深度解码器输出最终深度。训练采用交替优化策略。

### 关键设计

**设计一：Embodied Scene Depth具身场景深度**
- **功能**：利用相机物理参数实时计算稠密的场景深度先验
- **核心思路**：在平面假设下利用相机内外参矩阵 $A = [K|0][R,T;0,1]$，已知相机高度 $h$ 求解每个像素的深度 $z_c$。先通过语义分割识别道路区域获得Embodied Road Depth（精度99%+），扩展到地面、垂直表面，最后用Telea Inpainting填补空白得到完整Embodied Scene Depth
- **设计动机**：道路满足平面条件，可直接用解析方法计算绝对深度，比如LiDAR般准确但更稠密；逐步扩展到全场景虽降低精度但提供有价值的几何约束

**设计二：深度引导的文本变分自编码器**
- **功能**：利用语言描述建模可能3D场景布局的概率分布
- **核心思路**：为每个物体 $O_i$ 生成深度文本描述 $T_i$（含深度值 $d_i$ 和排序 $r_i$），与图像caption合并后通过CLIP文本编码器+MLP估计潜在分布的均值 $\hat{\mu}$ 和标准差 $\hat{\sigma}$，用重参数化技巧 $\hat{z} = \hat{\mu} + \epsilon \cdot \hat{\sigma}$ 采样，经深度解码器生成深度图
- **设计动机**：文本提供的物体尺度和空间布局先验可约束深度估计的解空间，变分框架自然建模了场景的多样可能性

**设计三：具身驱动的条件采样器**
- **功能**：从文本VAE的潜在分布中按图像条件采样出与特定图像对应的深度
- **核心思路**：Transformer blocks将RGB和Embodied Depth的融合特征（通过交叉注意力 $F_f^d = \text{softmax}(\frac{Q_r K_d}{d_k})V_d$ 融合）编码为 $h \times w$ 个局部样本 $\tilde{\epsilon}$，替代标准高斯噪声 $\epsilon$，生成 $\tilde{z} = \hat{\mu} + \tilde{\epsilon} \cdot \hat{\sigma}$，经共享权重的深度解码器输出深度
- **设计动机**：文本语言图谱只能描述可能的3D布局分布，需要图像信息来确定与当前场景最匹配的潜在向量

### 损失函数

交替训练：(1) 冻结条件采样器，训练文本VAE和深度解码器，使用 $\mathcal{L}_{KL}(\mu, \sigma) + \mathcal{L}_{SiLog}$；(2) 冻结文本VAE，训练条件采样器和深度解码器，使用SiLog损失。KL散度将潜在分布正则化向标准高斯，SiLog损失增强尺度不变性。

## 实验关键数据

### 主实验：KITTI深度估计

| 方法 | AbsRel↓ | SqRel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|---------|--------|-------|----------------|
| BTS | 0.061 | 0.261 | 2.834 | 0.954 |
| Adabins | 0.058 | 0.190 | 2.360 | 0.964 |
| iDisc | 0.053 | 0.175 | 2.216 | 0.971 |
| ECoDepth | 0.054 | 0.171 | 2.173 | 0.970 |
| **Ours** | **0.050** | **0.159** | **2.054** | **0.974** |

### 消融实验：Embodied Depth精度（KITTI数据集）

| 深度类型 | ±5%误差范围 | ±10%误差范围 |
|---------|-----------|------------|
| Embodied Road Depth | 80.24% | 99.33% |
| Embodied Ground Depth | 60.30% | 74.89% |
| Embodied Scene Depth | 38.88% | 52.45% |

### 关键发现
- Embodied Road Depth精度极高（99.33%像素误差<10%），可在道路区域替代LiDAR
- 不同语义分割模型对Embodied Depth影响极小（与GT分割差距<1%），鲁棒性强
- 完整框架在KITTI上AbsRel达0.050，超越ECoDepth和iDisc等SOTA
- Embodied Scene Depth虽然在非平面区域精度下降，但提供了有价值的稠密几何先验
- 交替训练策略（文本VAE和条件采样器）对整体性能至关重要

## 亮点与洞察

1. **相机物理模型的直接利用**：不是学习深度先验而是解析计算，在平面区域精度接近传感器级别
2. **三层次具身化整合**：相机+环境+语言的统一框架，各层次互补
3. **深度文本描述的创新**：将物体深度值和排序转化为自然语言，利用CLIP文本编码器的语义能力
4. **对分割模型低依赖**：不同分割模型的精度差异对最终深度估计影响极小

## 局限与展望

- 主要在KITTI和DDAD等驾驶场景验证，对室内等无明显地面平面的场景适用性有限
- Embodied Scene Depth依赖平面假设，对坡道、台阶等非平面地面精度下降
- 文本描述依赖自动caption生成工具的质量
- 需要已知相机内外参，限制了零样本通用性
- 未来可扩展到更多场景类型并结合更强的VLM

## 相关工作与启发

- 与DepthCLIP等使用固定文本模板不同，本文动态生成包含深度信息的文本描述
- 与WordDepth的文本VAE结构类似，但增加了Embodied Scene Depth的融合
- 相机模型解析计算深度的思路可作为任何深度估计方法的可选先验输入

## 评分

⭐⭐⭐⭐ — 相机模型具身化的思路在驾驶场景下实用价值高，语言-视觉融合框架设计合理；但应用场景受限于有平面假设的环境。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)
- [\[CVPR 2025\] DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](relative_pose_estimation_through_affine_corrections_of_monocular_depth_priors.md)

<!-- RELATED:END -->
