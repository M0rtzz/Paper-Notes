---
title: >-
  [论文解读] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
description: >-
  [CVPR 2026][自动驾驶][3D Gaussian Splatting] LR-SGS 提出利用 LiDAR 反射率引导的结构感知 Salient Gaussian 表示，通过将 LiDAR 强度校准为光照不变的反射率通道附加到每个 Gaussian、从几何与反射率特征点初始化结构化 Salient…
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "3D Gaussian Splatting"
  - "LiDAR反射率"
  - "自动驾驶场景重建"
  - "新视角合成"
  - "多模态融合"
---

# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2603.12647](https://arxiv.org/abs/2603.12647)  
**代码**: 待确认  
**领域**: 自动驾驶  
**关键词**: 3D Gaussian Splatting, LiDAR反射率, 自动驾驶场景重建, 新视角合成, 多模态融合

## 一句话总结

LR-SGS 提出利用 LiDAR 反射率引导的结构感知 Salient Gaussian 表示，通过将 LiDAR 强度校准为光照不变的反射率通道附加到每个 Gaussian、从几何与反射率特征点初始化结构化 Salient Gaussian、以及 RGB-反射率跨模态梯度一致性约束，在 Waymo 数据集的复杂光照场景中以更少 Gaussian 数量和更短训练时间超越 OmniRe 达 1.18 dB PSNR。

## 研究背景与动机

**领域现状**：3DGS 在自驾场景重建和新视角合成中展示了快速高保真渲染能力。现有方法如 StreetGS、OmniRe 等已构建动态场景图来处理时序动态物体。

**现有痛点**：纯相机方法在复杂光照（夜间、逆光）和大幅自车运动下易出现纹理不一致和优化不稳定。现有 LiDAR-增强方法（如 PVG、OmniRe）仅用点云做 Gaussian 初始化或深度监督，未充分挖掘 LiDAR 点云中的反射率信息和几何结构信息。

**核心矛盾**：RGB 信号对光照、曝光等外部因素敏感，无法在弱纹理区域和材质边界处提供可靠约束；LiDAR 虽提供精确深度且对光照不敏感，但其反射强度（intensity）中蕴含的材质属性（reflectance）和几何结构特征尚未被有效利用。

**本文目标**：(1) 如何将 LiDAR 的光照不变材质信息融入 Gaussian 表示？(2) 如何用结构感知的 Gaussian 更精确地建模边缘和平面？(3) 如何跨模态对齐 RGB 与反射率的边界？

**切入角度**：LiDAR 原始返回信号包含 intensity，经距离 $R$ 和入射角 $\alpha$ 校正后可得到近似光照不变的 reflectance $\rho$。场景中的边缘、平面等关键结构可以从点云的平滑度和反射率梯度中提取——以此初始化一种参数更少但能精确表征结构的 Salient Gaussian。

**核心 idea**：用 LiDAR 校正反射率作为光照不变通道附加到 Gaussian，从几何和反射率特征点初始化结构感知 Salient Gaussian，并通过 RGB-反射率梯度对齐来增强材质边界一致性。

## 方法详解

### 整体框架

LR-SGS 想解决的是纯相机 3DGS 在夜间、逆光等复杂光照下重建不稳定的问题，办法是把 LiDAR 里一直被忽视的反射率和结构信息接进 Gaussian 表示。整条流水线这样转：先把每帧 LiDAR 的原始强度校正成光照不变的反射率，并从点云里挑出边缘、平面、材质突变这三类"结构关键点"；用这些特征点初始化一批参数更省、专门贴合边缘和平面的 Salient Gaussian，再用常规 SfM 点补上其余的 Non-Salient Gaussian。两类 Gaussian 一起放进背景/动态物体/天空的场景图，渲染出彩色图、深度图和反射率图后，用 Color Loss、LiDAR Loss 和跨模态 Joint Loss 联合优化每个 Gaussian 的位置、不透明度、尺度、旋转、外观和反射率属性。训练过程中还会根据每个 Gaussian 实际呈现的形状，在 Salient 与 Non-Salient 之间双向切换。

### 关键设计

**1. LiDAR 强度校准：把强度还原成光照不变的材质反射率。**

LiDAR 原始 intensity 同时受距离和入射角影响——同一块材质，离得远、打得斜，回波就弱，所以它不能直接当材质属性用。本文按雷达方程 $I = \eta_{all}\frac{\rho\cos\alpha}{R^2}$ 反解出反射率 $\rho$：用每个点的距离 $R$ 和入射角 $\alpha$ 把这两个干扰因子除掉。入射角不是现成的，要先在点 $\mathbf{p}$ 的邻域里取 $\mathbf{p}_1,\mathbf{p}_2$ 构出局部法向量 $\mathbf{n}$，再算 $\cos\alpha = \frac{\mathbf{p}^T\mathbf{n}}{\|\mathbf{p}\|}$。校正后的反射率归一化到 $[0,1]$，投影到相机平面就得到一张稀疏反射率图 $F_{gt}$。这样得到的 $\rho$ 近似只跟表面材质有关、不随光照变化，于是它能在 RGB 信号最不可靠的弱纹理区和夜间场景里，提供一个稳定的跨帧约束信号。

**2. Salient Gaussian 结构感知表示：用一个主方向 + 一个共享尺度替代三个独立尺度。**

标准 3DGS 每个 Gaussian 有三个独立尺度参数，但真实场景里大量结构其实是道路边线、车身轮廓这类边缘，或者地面、墙面这类平面——这些东西只需要"沿某个方向拉长"或"沿某个方向压扁"就能表征，三个自由度是浪费。Salient Gaussian 因此只保留一个主导方向 $d_{spec}$ 和对应的主导尺度 $\sigma_\|$，剩下两轴共用一个 $\sigma_\perp$。协方差矩阵相应退化成两类：

$$\Sigma_{edge} = \mathbf{R}\operatorname{diag}(\sigma_\|^2, \sigma_\perp^2, \sigma_\perp^2)\mathbf{R}^T, \qquad \Sigma_{plane} = \mathbf{R}\operatorname{diag}(\sigma_\perp^2, \sigma_\perp^2, \sigma_\|^2)\mathbf{R}^T$$

Edge 型沿主方向拉成细长条贴合边缘，Planar 型沿主方向压扁贴合平面。少了一个尺度参数，却因为形状先验和真实结构对得更准，反而建模得更干净——这是后面效率优势（更少 Gaussian、更快收敛）的根源。

**3. LiDAR 特征点初始化：只在结构关键处搭脚手架，而不是撒满全部点。**

有了两种 Salient Gaussian，还得知道往哪儿放。本文不像 PVG/OmniRe 那样拿全部 LiDAR 点做初始化，而是先从点云里筛出三类特征点。几何上算每个点的平滑度 $c_j = \frac{1}{|K|\cdot\|\mathbf{p}_j\|}\|\sum_{\mathbf{p}_i\in\mathcal{P}_j}(\mathbf{p}_i - \mathbf{p}_j)\|$，按阈值把曲率大的判为 geometric edge points、曲率小的判为 geometric planar points；材质上算反射率梯度 $G_j$（沿同一扫描环取左右邻域均值之差），梯度大的判为 reflectance edge points。三类点分别初始化成 Edge 或 Planar Salient Gaussian。因为这些点正好落在物体轮廓、道路边界、材质变化这些"信息密集"的位置，等于给训练先搭好了一副结构脚手架，让优化从一开始就贴着真实结构走、收敛更快。

**4. 改进密度控制与 Salient Transform：让分裂顺着形状走，并允许两类 Gaussian 互相转化。**

结构会在训练中变化，初始化时的判断未必一直成立，所以密度控制也得跟着改。分裂时不再各向同性地复制：Edge Salient Gaussian 沿主导方向分裂、Planar 型在正交平面内分裂，保证子 Gaussian 继承父代的结构取向。更关键的是 Salient Transform——用每个 Gaussian 的奇异值定义 linearity $L(g) = (s_1 - s_2)/s_1$ 和 planarity $P(g) = (s_2 - s_3)/s_1$ 来衡量它现在到底有多"像边/像面"。一个 Non-Salient Gaussian 若连续两次满足 $\max\{L, P\} > \tau_{max}=0.5$，就升级成 Salient；反过来 Salient Gaussian 若连续两次 $\max\{L, P\} < \tau_{min}=0.1$，就降级回 Non-Salient。这套双向"晋升/降级"机制有两个用处：在 LiDAR 没扫到的区域，Non-Salient Gaussian 也能在优化中长出明确方向性后自动升级，把结构建模扩展到全场景；同时把那些已经失去方向性的冗余 Salient Gaussian 及时摘掉，避免硬分类带来的误判固化。

**5. RGB-反射率跨模态一致性 (Joint Loss)：对齐梯度而不是对齐像素值。**

反射率图和 RGB 图都能看出材质边界，理论上应该互相印证，但直接逐像素对齐它们没有物理意义——一个是反射率、一个是亮度，量纲都不一样。Joint Loss 改成对齐梯度：先把渲染 RGB 转灰度 $C^g$，和渲染反射率 $F$ 各自做高斯平滑、算 Scharr 梯度，再从方向和幅度两头约束。方向一致性 $\mathcal{L}_{dir} = 1 - \hat{\nabla}F\cdot\hat{\nabla}C^g$ 让两张图的梯度指向同一边界；幅度一致性 $\mathcal{L}_{val} = \|g_F/F - g_{C^g}/C^g\|_1$ 比较的是各自归一化后的梯度强度，用除以自身值的方式抹掉跨模态的绝对尺度差。这样既能借反射率把模糊的材质边界锐化清楚，又不会因为两种信号尺度不同而引入偏差，最终在车牌、灯组这类高频区域明显减少模糊伪影。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{lidar} + \mathcal{L}_{joint}$，其中：

- $\mathcal{L}_{rgb} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D\text{-}SSIM}$（光度一致性）
- $\mathcal{L}_{lidar} = \lambda_{depth}\mathcal{L}_{depth} + \lambda_{fle}\mathcal{L}_{fle} + \lambda'_{fle}\mathcal{L}'_{fle}$（深度 + 反射率 + 反射率梯度）
- $\mathcal{L}_{joint} = \lambda_{dir}\mathcal{L}_{dir} + \lambda_{val}\mathcal{L}_{val}$（跨模态梯度对齐）

权重设置：$\lambda_c = \lambda_{val} = 0.2$，$\lambda_{depth} = \lambda_{fle} = \lambda_{dir} = 0.1$，$\lambda'_{fle} = 0.05$。训练 30k 迭代，Salient transform 阈值 $\tau_{max}=0.5$，$\tau_{min}=0.1$。

## 实验关键数据

### 主实验

Waymo Open Dataset 上四类场景的新视角合成结果（PSNR/SSIM/LPIPS）：

| 场景类型 | 指标 | LR-SGS (Ours) | OmniRe | StreetGS | 提升 |
|---------|------|:---:|:---:|:---:|:---:|
| Dense Traffic | PSNR↑ | **28.89** | 28.44 | 27.01 | +0.45 |
| Dense Traffic | PSNR*↑ | **24.02** | 23.72 | 21.73 | +0.30 |
| High-Speed | PSNR↑ | **28.77** | 28.12 | 28.06 | +0.65 |
| Complex Lighting | PSNR↑ | **30.51** | 29.33 | 29.16 | **+1.18** |
| Static | PSNR↑ | **28.73** | 28.23 | 28.15 | +0.50 |

在所有场景类别的 PSNR/SSIM/LPIPS 上全面领先。Complex Lighting 场景提升最为显著，达到 +1.18 dB PSNR。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| Full model (Ours) | **29.22** | **0.850** | **0.139** |
| w/o Salient Gaussian | 28.74 | 0.830 | 0.152 |
| w/o LiDAR Feature Init | 28.94 | 0.839 | 0.144 |
| w/o Reflectance | 28.87 | 0.831 | 0.147 |
| w/o Joint Loss | 28.96 | 0.835 | 0.144 |

### 效率分析

| 方法 | PSNR↑ | Gaussian 数量↓ | 训练时间↓ | FPS↑ |
|------|:---:|:---:|:---:|:---:|
| StreetGS | 28.20 | 2,929,851 | 64m30s | 33.61 |
| OmniRe | 28.62 | 2,744,275 | 67m11s | 30.55 |
| **LR-SGS** | **28.95** | **2,510,883** | **59m25s** | **36.95** |

### 关键发现

1. **Salient Gaussian 贡献最大**：去掉后 PSNR 降 0.48 dB、LPIPS 增 0.013，说明结构感知表示对质量和效率都至关重要
2. **反射率通道在复杂光照下效果显著**：夜间场景中 LiDAR 反射率提供了 RGB 无法给出的稳定约束，有效抑制光照引起的伪影
3. **效率优势明显**：比 OmniRe 少约 23 万个 Gaussian、训练快 8 分钟、FPS 高 6.4，原因是 Salient Gaussian 的两参数减少了冗余并加速收敛
4. **Salient Transform 扩展覆盖**：即使 LiDAR 未覆盖区域，Non-Salient Gaussian 也可通过 transform 升级为 Salient，确保全场景结构建模
5. **Joint Loss 既提升 RGB 质量又改善反射率渲染**：车辆车牌、灯组等高频区域的重建清晰度显著改善

## 亮点与洞察

1. **LiDAR intensity → reflectance 的简单但有效校正**：仅用距离和入射角做物理校正即可获得光照不变的材质表征，无需复杂的逆渲染或材质估计网络。这个思路可迁移到任何使用 LiDAR 的重建任务中
2. **结构感知 Gaussian 的两参数化**：用"主导方向 + 共享非主导尺度"替代三个独立尺度，实现了"更少参数 + 更好结构建模"的双赢，打破了精度与效率的常见 trade-off
3. **跨模态梯度对齐而非像素对齐**：直接对齐 RGB 和反射率的像素值缺乏物理意义（它们量纲不同），而对齐归一化梯度的方向和幅度巧妙绕过了尺度差异问题，聚焦于边界一致性
4. **Salient Transform 双向适应机制**：类似于"晋升/降级"机制，让 Salient Gaussian 随训练过程自然确立和移除，避免了硬性分类带来的信息丢失

## 局限与展望

1. **仅在 Waymo 数据集验证**：未在 nuScenes、KITTI 等其他驾驶数据集上验证泛化性，不同 LiDAR 型号的 intensity 特性差异可能影响反射率校正效果
2. **反射率校正依赖简化物理模型**：实际中材质的 BRDF 远比 Lambert 反射复杂，校正精度在镜面反射、半透明材质等情况下可能不足
3. **动态物体建模依赖外部 mask**：物体掩码来自预训练模型，该环节的精度直接限制动态重建质量
4. **Salient Transform 阈值为手工设定**：$\tau_{max}=0.5$ 和 $\tau_{min}=0.1$ 可能不适用于所有场景，自适应阈值策略值得探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[CVPR 2026\] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection](look_before_you_fuse_2d-guided_cross-modal_alignment_for_robust_3d_detection.md)

</div>

<!-- RELATED:END -->
