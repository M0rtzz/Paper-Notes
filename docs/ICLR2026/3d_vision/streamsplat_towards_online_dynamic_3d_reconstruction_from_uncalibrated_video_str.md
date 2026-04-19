---
title: >-
  [论文解读] StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams
description: >-
  [ICLR 2026][3D视觉][动态3D重建] StreamSplat 提出了一个完全前馈的在线动态3D重建框架，通过概率位置采样、双向形变场和自适应高斯融合三大创新，能从未标定视频流中即时生成动态3DGS表示，速度比优化方法快1200倍。
tags:
  - ICLR 2026
  - 3D视觉
  - 动态3D重建
  - 3D高斯溅射
  - 在线重建
  - 前馈模型
  - 视频流
---

# StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams

**会议**: ICLR 2026  
**arXiv**: [2506.08862](https://arxiv.org/abs/2506.08862)  
**代码**: [https://streamsplat3d.github.io/](https://streamsplat3d.github.io/)  
**领域**: 3D视觉  
**关键词**: 动态3D重建, 3D高斯溅射, 在线重建, 前馈模型, 视频流

## 一句话总结

StreamSplat 提出了一个完全前馈的在线动态3D重建框架，通过概率位置采样、双向形变场和自适应高斯融合三大创新，能从未标定视频流中即时生成动态3DGS表示，速度比优化方法快1200倍。

## 研究背景与动机

实时动态3D重建（4D重建）在机器人、AR/VR和自动驾驶等领域至关重要。然而现有方法存在根本性限制：

**离线依赖**：主流动态3DGS方法（如4DGS、DGMarbles）需要访问完整视频序列，并经历数小时的逐场景迭代优化，包括相机标定→静态高斯优化→形变场学习→时序融合的多步流水线

**实时性差**：即使最新方法仍需30分钟-24小时处理一个场景，无法部署于实时应用

**标定要求**：几乎所有方法都需要预先标定的相机参数

**前馈方法局限**：已有的前馈3DGS方法（pixelSplat、NoPoSplat、StreamGS）仅支持静态场景，动态变体仍需标定和全序列访问

作者提出了核心研究问题：**能否在完全在线的条件下，用未标定视频流达到离线方法的质量和功能？**

## 方法详解

### 整体框架

StreamSplat 维持一个正则高斯集合 $\tilde{\mathcal{G}}(t)$，对每一帧输入执行：编码→预测双向形变→自适应融合→渲染的流水线。采用两阶段训练：先训练静态编码器，再冻结编码器训练动态解码器。

### 关键设计

1. **概率位置采样 (Probabilistic Position Sampling)**

   针对3DGS对位置初始化敏感且前馈模型容易陷入局部最优的问题，本文预测每个3D偏移的截断正态分布而非直接回归：

   $$\boldsymbol{o} \sim \mathcal{N}_{[-1,1]}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$$

   最终3D位置通过像素对齐预测得到：$\boldsymbol{\mu}_i = (u_i + o_{i,0},\; v_i + o_{i,1},\; g(o_{i,2}))$，其中 $g(z) = 2/(1+z)$ 为深度映射。该策略在训练初期促进空间探索，后期稳定收敛到最优位置。消融实验显示，概率采样比确定性预测提升6.36dB PSNR。

2. **双向形变场 (Bidirectional Deformation Field)**

   传统方法对每帧实例化新高斯并迭代优化，难以适配前馈模型。本文联合建模前后向运动：前向场将上一帧高斯 $\mathcal{G}_{t-1}$ 变形到当前时间 $t$，后向场将当前帧高斯 $\mathcal{G}_t$ 变形回 $t-1$。这种对称设计：
   - 提供稳健的跨帧关联
   - 自然处理出现/消失的高斯
   - 简化端到端训练的预测和监督

3. **自适应高斯融合 (Adaptive Gaussian Fusion)**

   通过时间依赖的不透明度调制实现软匹配融合，每个高斯在两个连续帧间持续存在：

   $$\alpha(t) = \alpha \cdot \frac{\sigma(-\gamma_0(|t - t_0| - \gamma_1))}{\sigma(\gamma_0 \cdot \gamma_1)}$$

   其中 $t_0$ 为高斯创建帧，$\gamma_0$ 控制过渡速率，$\gamma_1$ 控制淡出窗口。该机制隐式融合前后向高斯：重建损失诱导软匹配，传播持久高斯同时处理出现/消失的高斯，无需硬分配或迭代融合即可维持时序一致性。

### 损失函数 / 训练策略

**阶段1 - 静态编码器**：
$$\mathcal{L}_{\text{static}} = \mathcal{L}_{\text{recon}}(\hat{I}_t, I_t) + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}(\hat{D}_t, D_t)$$
其中深度损失采用尺度-偏移不变形式，并引入自适应衰减因子 $\hat{\lambda}_{\text{depth}}$ 降低噪声伪深度的影响。

**阶段2 - 动态解码器**（冻结编码器）：
$$\mathcal{L}_{\text{dynamic}} = \mathbb{E}_t[\mathcal{L}_{\text{recon}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}]$$
新增运动前景区域的辅助重建损失，使用DAVIS/YouTube-VOS的分割掩码监督。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (StreamSplat) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| DAVIS Key Frame | PSNR↑ | 37.83 | 42.33 (MonST3R) | 竞争性 |
| DAVIS Key Frame | LPIPS↓ | 0.016 | 0.012 (MonST3R) | 接近 |
| DAVIS Middle-4 | PSNR↑ | **23.66** | 21.33 (DGMarbles) | +2.33 |
| DAVIS Middle-4 | LPIPS↓ | **0.193** | 0.313 (DGMarbles) | -0.12 |
| RE10K Average | PSNR↑ | **29.51** | 23.73 (DGMarbles) | +5.78 |
| 8帧插值 | PSNR↑ | **22.10** | 21.09 (AMT) | +1.01 |

### 消融实验

| 配置 | PSNR (Key)↑ | PSNR (Mid)↑ | 说明 |
|------|---------|---------|------|
| w/o 概率采样 | 31.47 | - | 确定性预测，降6.36dB |
| w/o 深度监督 | 36.68 | - | 空间结构失真 |
| w/o 双向形变 | - | 18.89 | 像素对齐结构丢失 |
| Full (Ours) | **37.83** | **23.66** | 完整模型 |

### 关键发现

- StreamSplat 是唯一支持近实时动态3D重建的方法，每帧0.049秒，比优化方法快1200×
- 在关键帧重建上与MonST3R竞争，但后者需要后优化且仅限关键帧
- 在中间帧重建上超过所有基线，包括2D视频插值方法
- 支持任意长度视频流的在线重建

## 亮点与洞察

1. **在线处理范式突破**：首次在未标定视频流上实现前馈式在线动态3D重建，颠覆了传统离线多阶段流水线
2. **概率位置采样**：简洁有效地解决了前馈3DGS的局部最优问题，提升巨大（+6.36dB）
3. **自适应不透明度融合**：通过时间依赖的不透明度实现软匹配，巧妙避免了传统方法的硬分配和迭代融合
4. **正则空间设计**：采用正交正则空间绕过逐场景相机标定，相机运动被吸收到高斯动力学中

## 局限与展望

- 关键帧重建质量略低于MonST3R（点云表示），但后者不支持在线处理
- 输入分辨率限制在512×288，高分辨率场景可能损失细节
- 仅在短-中等长度视频上评估，超长序列的误差累积需要更多验证
- 正交投影假设可能在强透视效果场景中受限

## 相关工作与启发

- NoPoSplat (Ye et al., 2024) 和 StreamGS (Li et al., 2025) 分别解决无姿态和在线问题，但均限于静态场景
- 双向形变的思路可推广到其他时序建模任务（视频生成、自动驾驶预测）
- 自适应高斯融合的生命周期管理思想来源于 Zhao et al. (2024)

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次实现未标定视频流的在线前馈动态3D重建，三个技术创新协同设计
- 实验充分度: ⭐⭐⭐⭐ 覆盖动态/静态多个基准，消融详尽，但缺少更长视频的评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，方法阐述逻辑性强，图表精美
- 价值: ⭐⭐⭐⭐⭐ 1200×加速具有重要实用价值，开启在线动态重建新范式

<!-- RELATED:START -->

## 相关论文

- [OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/3d_vision/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)
- [OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](../../ICCV2025/3d_vision/dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)

<!-- RELATED:END -->
