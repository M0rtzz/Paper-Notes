---
title: >-
  [论文解读] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments
description: >-
  [CVPR 2025][3D视觉][SLAM] 本文提出 WildGS-SLAM，一个基于 3D Gaussian Splatting 的单目 RGB SLAM 系统，通过 DINOv2 特征驱动的不确定性预测来引导追踪和建图中的动态物体去除，在动态环境下的追踪精度（ATE RMSE 0.46cm）和无伪影的新视角合成质量上大幅超越现有方法。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "SLAM"
  - "3D Gaussian Splatting"
  - "动态环境"
  - "不确定性估计"
  - "单目视觉"
---

# WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments

**会议**: CVPR 2025  
**arXiv**: [2504.03886](https://arxiv.org/abs/2504.03886)  
**代码**: [https://wildgs-slam.github.io](https://wildgs-slam.github.io)  
**领域**: 3D视觉  
**关键词**: SLAM, 3D Gaussian Splatting, 动态环境, 不确定性估计, 单目视觉

## 一句话总结

本文提出 WildGS-SLAM，一个基于 3D Gaussian Splatting 的单目 RGB SLAM 系统，通过 DINOv2 特征驱动的不确定性预测来引导追踪和建图中的动态物体去除，在动态环境下的追踪精度（ATE RMSE 0.46cm）和无伪影的新视角合成质量上大幅超越现有方法。

## 研究背景与动机

**领域现状**：传统 SLAM 系统（ORB-SLAM、DROID-SLAM）依赖静态场景假设，在动态环境中特征匹配和光度一致性被移动物体破坏，导致追踪漂移。近年来基于 3DGS 的 SLAM（MonoGS、Splat-SLAM）在静态场景下取得优异结果，但在动态环境中性能急剧下降。

**现有痛点**：(1) 现有动态 SLAM 方法（DG-SLAM、DDN-SLAM）依赖语义分割或物体检测来识别动态区域，需要预定义可移动物体类别，对未知类型的动态物体无法泛化；(2) NeRF On-the-go 和 WildGaussians 虽然展示了纯几何方法去除动态干扰的能力，但它们只处理稀疏视角重建且需要已知相机位姿作为输入；(3) 现有单目 SLAM 都不支持在不依赖先验类别信息的情况下处理动态环境。

**核心矛盾**：在单目 SLAM 的顺序输入设置下，如何在不依赖语义先验的情况下，以纯几何方式同时实现动态物体的鲁棒去除和高精度的相机追踪。

**本文目标**：设计一个纯几何方法的单目 RGB SLAM 系统，能在高度动态环境中同时实现精确的相机追踪、高保真静态场景重建和无伪影的新视角合成。

**切入角度**：受 NeRF On-the-go 启发，利用预训练的 3D-aware DINOv2 特征作为图像表示，训练一个浅层 MLP 在线预测逐像素不确定性图，将不确定性同时注入追踪（DBA）和建图（渲染损失）流程中。

**核心 idea**：用一个在线学习的不确定性 MLP 作为统一接口，将 DINOv2 的语义理解能力转化为追踪和建图中的动态物体感知能力。

## 方法详解

### 整体框架

WildGS-SLAM 以 RGB 图像流为输入，主要包含三个模块：(1) 不确定性预测模块——用 DINOv2 提取特征，浅层 MLP 解码为逐像素不确定性图；(2) 追踪模块——基于 DROID-SLAM 的光流追踪，将不确定性集成到 DBA 层并融合单目度量深度；(3) 建图模块——维护并优化 3DGS 地图，渲染损失按不确定性加权。不确定性 MLP 和 3DGS 地图独立优化，梯度流互不干扰。

### 关键设计

1. **不确定性预测模块**:

    - 功能：为每个输入帧预测逐像素不确定性图，标识动态区域
    - 核心思路：使用经过 3D 感知微调的 DINOv2 提取图像特征 $F_i = \mathcal{F}(I_i)$，浅层 MLP $\mathcal{P}$ 将其解码为不确定性图 $\beta_i = \mathcal{P}(F_i)$。MLP 在线训练，使用组合损失 $\mathcal{L}_{\text{uncer}} = (\mathcal{L}_{\text{SSIM}}' + \lambda_1 \mathcal{L}_{\text{depth}})/\beta_i^2 + \lambda_2 \mathcal{L}_{\text{reg\_V}} + \lambda_3 \mathcal{L}_{\text{reg\_U}}$，其中本文新增了深度不确定性损失项 $\mathcal{L}_{\text{depth}} = |\hat{D}_i - \tilde{D}_i|_1$（渲染深度与 Metric3D v2 预测深度的 L1 差异）
    - 设计动机：纯几何不确定性估计无需预定义动态物体类别，DINOv2 的 3D 感知能力提供了强先验；新增的深度损失项有效提升了区分动态干扰物的能力

2. **不确定性感知追踪**:

    - 功能：在 DBA 优化中降低动态区域像素的影响权重
    - 核心思路：将不确定性 $\beta_i$ 融入 DBA 目标函数的 Mahalanobis 距离权重中 $\|\tilde{p}_{ij} - \Pi_c(\omega_j^{-1}\omega_i \Pi_c^{-1}(p_i, d_i))\|_{\Sigma_{ij}/\beta_i^2}^2$，使动态区域像素对位姿优化贡献极小。同时引入度量深度正则化项 $\lambda_4 \sum_i \|M_i(d_i - 1/\tilde{D}_i)\|^2$ 在不确定性 MLP 尚未收敛的早期阶段稳定位姿估计
    - 设计动机：不确定性 MLP 在在线学习初期不够可靠，度量深度正则化提供了一个稳定的"安全网"，特别是在动态物体占主导的极端场景下

3. **不确定性感知建图**:

    - 功能：在 3DGS 优化中抑制动态区域的渲染损失贡献
    - 核心思路：渲染损失按不确定性加权 $\mathcal{L}_{\text{render}} = (\lambda_5 \mathcal{L}_{\text{color}} + \lambda_6 \mathcal{L}_{\text{depth}})/\beta^2 + \lambda_7 \mathcal{L}_{\text{iso}}$，动态区域的高不确定性自动降低其在地图优化中的权重。关键设计是 $\mathcal{P}$ 和 $\mathcal{G}$ 的梯度流完全分离——不确定性损失不影响 Gaussian 地图，渲染损失不影响不确定性 MLP
    - 设计动机：如果不分离优化，不确定性 MLP 可能会通过降低所有区域的不确定性来"欺骗"损失函数，导致动态物体无法被正确识别

### 损失函数 / 训练策略

不确定性 MLP 使用改良 SSIM 损失、深度不确定性损失和两个正则化项（特征相似性方差最小化 $\mathcal{L}_{\text{reg\_V}}$、防止 $\beta$ 无限大的 $\log \beta$ 正则化）联合训练。3DGS 地图使用 L1 + SSIM 颜色损失、L1 深度损失和各向同性正则化损失训练。两者在每个迭代中并行优化但梯度隔离。追踪使用 DROID-SLAM 框架的 ConvGRU 迭代更新光流，并集成回环检测和在线全局 BA。

## 实验关键数据

### 主实验（Wild-SLAM MoCap 数据集追踪性能，ATE RMSE↓ [cm]）

| 方法 | 输入 | ANYmal1 | Ball | Crowd | Person | Table2 | 平均 |
|------|------|---------|------|-------|--------|--------|------|
| DROID-SLAM | Mono | 0.6 | 1.2 | 2.3 | 0.6 | 95.6 | 16.17 |
| Splat-SLAM | Mono | 0.4 | 0.3 | 0.7 | 0.8 | 73.6 | 8.71 |
| MegaSaM | Mono | 0.6 | 0.6 | 1.0 | 3.2 | 9.4 | 2.40 |
| DynaSLAM (N+G) | RGB-D | 1.6 | 0.5 | 1.7 | 0.5 | 34.8 | 7.84 |
| **WildGS-SLAM** | **Mono** | **0.2** | **0.2** | **0.3** | **0.8** | **1.3** | **0.46** |

### 消融实验（新视角合成质量）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Splat-SLAM | 17.23 | 0.699 | 0.346 |
| **WildGS-SLAM** | **20.59** | **0.783** | **0.209** |

### 关键发现

- WildGS-SLAM 追踪精度（0.46cm）比最强单目基线 MegaSaM（2.40cm）提升 5.2 倍，甚至大幅超越使用 RGB-D 输入的 DynaSLAM（7.84cm）
- 在 Table2 场景（大面积遮挡的桌面操作）中，所有基线追踪误差极大（Splat-SLAM 73.6cm），WildGS-SLAM 仍保持 1.3cm，展示了极端动态场景下的鲁棒性
- 新视角合成 PSNR 提升 3.36dB，LPIPS 降低 0.137，所有场景都实现了无伪影渲染
- 与需要语义分割先验的 DynaSLAM (RGB) 相比，WildGS-SLAM 以纯几何方法达到更低误差（0.46 vs 5.19cm）

## 亮点与洞察

- **不确定性 MLP 与 3DGS 地图的独立优化**是一个关键的设计洞察——如果联合优化，系统会找到"作弊"的捷径。这种"对抗性"训练思想（类似 NeRF-W 中的瞬态分支）在很多联合优化场景中都值得借鉴
- **度量深度正则化作为"启动安全网"**的设计很实用——在不确定性 MLP 尚未收敛的关键初始阶段提供稳定性，随后不确定性逐渐接管。这种"冷启动"策略在很多在线学习系统中都有价值
- 纯几何方案的最大优势是**零类别依赖**——不需要知道动态物体是人、车还是动物，这在开放世界场景中极其重要

## 局限与展望

- 作者收集了新的 Wild-SLAM 数据集用于评估，但未在更多公开基准（如 TUM-RGBD 动态序列）上验证
- 依赖 Metric3D v2 提供度量深度和 DINOv2 提供特征，这两个预训练模型的推理开销可能影响实时性
- 浅层 MLP 在线训练可能在快速运动或突发大量新动态物体时响应不够迅速
- 未探讨如何利用估计的不确定性恢复动态物体的轨迹或进行动态场景理解

## 相关工作与启发

- **vs NeRF On-the-go / WildGaussians**: 这些方法也用不确定性去除动态干扰，但只处理稀疏视角（需要已知位姿），WildGS-SLAM 将这一思路扩展到顺序 SLAM 设置，是更具挑战性的场景
- **vs DG-SLAM / DDN-SLAM**: 这些动态 SLAM 依赖语义分割或物体检测确定动态区域，泛化能力受限于预定义类别；WildGS-SLAM 的纯几何方案对任意类型的动态物体都有效
- **vs Splat-SLAM**: 当前最优的单目 GS-SLAM，但假设静态场景；WildGS-SLAM 在动态场景下追踪精度提升 18.9 倍

## 评分

- 新颖性: ⭐⭐⭐⭐ 将不确定性估计与 GS-SLAM 的追踪和建图统一集成是重要贡献，但核心思想源自 NeRF On-the-go
- 实验充分度: ⭐⭐⭐⭐ 自建数据集设计合理（室内外、多种动态物体），定量和定性结果充分，但缺少公开基准对比
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述非常清晰，系统概述图信息量大，公式推导严谨
- 价值: ⭐⭐⭐⭐⭐ 首个在单目 RGB 下实现纯几何动态处理的 GS-SLAM，追踪精度提升一个数量级

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [\[CVPR 2025\] MNE-SLAM: Multi-Agent Neural SLAM for Mobile Robots](mne-slam_multi-agent_neural_slam_for_mobile_robots.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](../../ECCV2024/3d_vision/sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)

</div>

<!-- RELATED:END -->
