---
title: >-
  [论文解读] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
description: >-
  [CVPR 2026][自动驾驶][3D Gaussian Splatting] LR-SGS 提出利用 LiDAR 反射率引导的结构感知 Salient Gaussian 表示，通过将 LiDAR 强度校准为光照不变的反射率通道附加到每个 Gaussian、从几何与反射率特征点初始化结构化 Salient Gaussian、以及 RGB-反射率跨模态梯度一致性约束，在 Waymo 数据集的复杂光照场景中以更少 Gaussian 数量和更短训练时间超越 OmniRe 达 1.18 dB PSNR。
tags:
  - CVPR 2026
  - 自动驾驶
  - 3D Gaussian Splatting
  - LiDAR反射率
  - 自动驾驶场景重建
  - 新视角合成
  - 多模态融合
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

输入为 RGB 图像序列和 LiDAR 点云序列。方法构建一个 3DGS 场景图（背景、动态物体、天空节点），其中初始场景 Gaussian 由两部分组成：从 LiDAR 特征点初始化的 Salient Gaussian 和从 SfM 点初始化的 Non-Salient Gaussian。通过渲染得到彩色图、深度图和反射率图，使用 Color Loss + LiDAR Loss + Joint Loss 联合优化所有 Gaussian 的位置、不透明度、尺度、旋转、外观和反射率属性。

### 关键设计

#### 1. LiDAR 强度校准 (LiDAR Intensity Calibration)

- **功能**：将 LiDAR 原始 intensity 校正为光照不变的反射率 $\rho$
- **核心思路**：根据雷达方程 $I = \eta_{all} \frac{\rho \cos\alpha}{R^2}$，用距离 $R$ 和入射角 $\alpha$ 对 intensity 进行归一化校正。入射角通过点 $\mathbf{p}$ 与其邻域点 $\mathbf{p}_1, \mathbf{p}_2$ 构建局部法向量 $\mathbf{n}$ 来计算：$\cos\alpha = \frac{\mathbf{p}^T \cdot \mathbf{n}}{\|\mathbf{p}\|}$。校正后的反射率归一化到 $[0,1]$ 并投影到相机平面得到稀疏反射率图 $F_{gt}$
- **设计动机**：原始 intensity 受距离和角度影响，不能直接作为材质属性。校正后的 reflectance 近似反映物体表面材质特性，不随光照变化，可作为稳定的跨帧约束信号

#### 2. Salient Gaussian 结构感知表示

- **功能**：设计一种参数高效的结构化 Gaussian，分为 Edge 型（沿边缘拉长）和 Planar 型（沿平面压扁）
- **核心思路**：每个 Salient Gaussian 具有一个主导方向 $d_{spec}$ 和对应的主导尺度 $\sigma_\|$，其余两轴共享一个尺度 $\sigma_\perp$。协方差矩阵简化为：Edge 型 $\Sigma_{edge} = \mathbf{R} \operatorname{diag}(\sigma_\|^2, \sigma_\perp^2, \sigma_\perp^2) \mathbf{R}^T$，Planar 型 $\Sigma_{plane} = \mathbf{R} \operatorname{diag}(\sigma_\perp^2, \sigma_\perp^2, \sigma_\|^2) \mathbf{R}^T$
- **设计动机**：普通 3DGS 每个 Gaussian 有 3 个独立尺度参数，而环境中大量边缘和平面结构只需沿特定方向拉伸/压扁即可表征。减少参数的同时更准确地匹配环境特征

#### 3. LiDAR 特征点初始化

- **功能**：从 LiDAR 点云中提取三类特征点来初始化 Salient Gaussian
- **核心思路**：(a) 计算每个点的平滑度 $c_j = \frac{1}{|K| \cdot \|\mathbf{p}_j\|} \|\sum_{\mathbf{p}_i \in \mathcal{P}_j}(\mathbf{p}_i - \mathbf{p}_j)\|$，按阈值划分为 geometric edge points 和 geometric planar points；(b) 计算反射率梯度 $G_j$（沿同一环的左右邻域均值差），提取 reflectance edge points。三类特征点分别初始化 Edge/Planar Salient Gaussian
- **设计动机**：与直接用全部 LiDAR 点初始化不同，特征点集中在结构关键位置（物体轮廓、道路边界、材质变化处），为训练提供稳定的结构脚手架，加速收敛

#### 4. 改进密度控制与 Salient Transform

- **功能**：适配 Salient Gaussian 的分裂策略，并实现 Salient/Non-Salient 状态自适应切换
- **核心思路**：分裂时，Edge Salient Gaussian 沿主导方向分裂，Planar 型在正交平面内分裂。定义 linearity $L(g) = (s_1 - s_2)/s_1$ 和 planarity $P(g) = (s_2 - s_3)/s_1$。Non-Salient Gaussian 若连续两次 $\max\{L, P\} > \tau_{max}=0.5$ 则升级为 Salient；Salient Gaussian 若连续两次 $\max\{L, P\} < \tau_{min}=0.1$ 则降级为 Non-Salient
- **设计动机**：训练过程中场景结构会动态变化，Salient Gaussian 需要能在 LiDAR 未覆盖区域自然生长，也需要排除不再具有明确方向性的冗余 Salient Gaussian

#### 5. RGB-反射率跨模态一致性 (Joint Loss)

- **功能**：对齐渲染反射率与灰度 RGB 图像的梯度方向和幅度
- **核心思路**：先将渲染 RGB 转为灰度 $C^g$，与渲染反射率 $F$ 分别做高斯平滑和 Scharr 梯度。Joint Loss 由两部分组成：方向一致性 $\mathcal{L}_{dir} = 1 - \hat{\nabla}F \cdot \hat{\nabla}C^g$（归一化梯度向量点积）；幅度一致性 $\mathcal{L}_{val} = \|g_F / F - g_{C^g} / C^g\|_1$（归一化幅度 L1 差异，消除跨模态尺度差异）
- **设计动机**：材质边界在 RGB 和反射率中都应表现为显著梯度，但两者的绝对尺度不同。通过对齐归一化梯度方向和幅度，可以在不引入尺度偏差的情况下锐化材质边界，减少模糊伪影

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

## 相关论文

- [F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi_modal_learning_in_3d_human_pose_estimation.md)
- [x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)

<!-- RELATED:END -->
