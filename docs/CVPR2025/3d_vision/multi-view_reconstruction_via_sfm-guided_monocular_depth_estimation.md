---
title: >-
  [论文解读] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
description: >-
  [3D视觉] 提出 Murre，将 SfM 稀疏点云作为条件注入扩散模型单目深度估计，生成多视角一致的度量深度图后进行 TSDF 融合，在仅用少量合成数据微调后即可在室内、街景、航拍等多种真实场景中超越 SOTA MVS 和神经隐式重建方法。
tags:
  - 3D视觉
---

# Multi-view Reconstruction via SfM-guided Monocular Depth Estimation

## 一句话总结

提出 Murre，将 SfM 稀疏点云作为条件注入扩散模型单目深度估计，生成多视角一致的度量深度图后进行 TSDF 融合，在仅用少量合成数据微调后即可在室内、街景、航拍等多种真实场景中超越 SOTA MVS 和神经隐式重建方法。

## 研究背景与动机

传统多视图3D重建方法面临三大瓶颈：

1. **高内存消耗**：基于学习的 MVS 方法（如 MVSNet、IGEV-MVS）需在 3D 空间中聚合特征构建代价体(cost volume)，GPU 内存开销巨大，限制重建分辨率
2. **稀疏视角失败**：当输入视角稀疏时，大量区域无法在多视角间匹配，基于匹配的方法失效
3. **泛化能力有限**：学习型 MVS 方法需要高质量 3D GT 数据训练，而此类数据稀缺，导致跨场景泛化差

单目深度估计虽可避免多视角匹配，但面临尺度歧义（scale ambiguity）和多视角不一致问题：
- 仿射不变方法（如 Marigold、Depth Anything）缺乏全局度量信息
- 度量深度方法（如 Metric3D）对训练数据域过拟合

核心洞察：SfM 点云是多视角信息的"浓缩形式"，它捕获了场景的全局结构和准确尺度，可以作为条件引导扩散模型产生既有度量尺度、又多视角一致的深度图。

## 方法详解

### 整体框架

Murre 的重建流程：
1. **SfM 稀疏重建**：对输入图像运行 Detector-free SfM 获取稀疏点云和相机位姿
2. **SfM 引导的深度估计**：将稀疏点云投影到各视角，稠密化后作为条件输入扩散模型
3. **深度对齐**：用 RANSAC 线性回归将预测深度与 SfM 深度对齐
4. **TSDF 融合**：将对齐后的度量深度图融合为最终 3D 几何

### 关键设计

#### 1. SfM 先验注入扩散模型

**稀疏深度稠密化**：
- 将 SfM 点云投影到每个视角得到稀疏深度图
- 用 k 近邻（KNN, k=3）插值稠密化：对每个无值像素，找 k 个最近有值像素，用距离倒数加权平均
- 同时计算距离图（distance map）：每个像素到最近有值像素的欧氏距离

**归一化策略**：
- 过滤顶部和底部 2% 的 SfM 深度值（去除离群点）
- 将范围扩展 20%（最小乘 0.8，最大乘 1.2）以覆盖完整深度范围
- GT 深度使用相同范围归一化

**条件输入**：将 RGB 图像、稠密化深度图通过编码器映射到 latent space，距离图直接下采样到 latent 分辨率，全部拼接后与噪声一起送入 UNet。

#### 2. 基于 Stable Diffusion v2 的条件扩散

- 以 Stable Diffusion V2 为初始化，冻结 VAE，仅微调 UNet
- 深度图复制三通道后用 VAE encoder 映射到 latent space
- 训练损失为标准的噪声预测 MSE：$\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$
- 推理时使用 5 次集成 + 像素级中位数提升鲁棒性
- 支持 LCM 蒸馏实现单步推理

#### 3. RANSAC 深度对齐

预测深度与 SfM 深度之间存在轻微尺度偏差，通过 RANSAC 线性回归对齐：
- 迭代随机采样子集估计尺度(scale)和偏移(shift)参数
- 选择内点最多的变换作为最终对齐

### 损失函数

$$\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$$

标准的扩散模型噪声预测损失，其中 $\epsilon$ 是添加到 GT 深度 latent code 的噪声。

## 实验关键数据

### 主实验表

**DTU 数据集（Chamfer Distance, mm, 3-view）**：

| 方法 | 训练数据量 | 平均 CD↓ |
|------|-----------|----------|
| COLMAP | - | 2.56 |
| MonoSDF | - | 1.86 |
| Marigold | 74K | 5.46 |
| Depth-Anything | 1.5M | 3.09 |
| Metric3D | 8M | 5.01 |
| MVSNet | 27.1K | 2.38 |
| DUSt3R | 8.5M | 2.81 |
| **Murre** | **86.4K** | **1.42** |

Murre 以最少的训练数据量（86.4K）取得了最低的 Chamfer Distance（1.42mm），比第二名 MonoSDF（1.86mm）低 24%。

### 消融实验

**速度-精度权衡（Replica）**：

| 设置（步数/集成/对齐） | 推理时间(s)↓ | F-Score↑ |
|------------------------|-------------|----------|
| 10步/5次集成/RANSAC | 12.166 | 0.853 |
| 10步/1次集成/RANSAC | 2.969 | 0.850 |
| 1步(LCM)/1次集成/RANSAC | 0.840 | 0.828 |
| 1步(LCM)/1次集成/无对齐 | 0.829 | 0.780 |

LCM 蒸馏后单步推理仅需 0.84 秒/视图，F-Score 仍达 0.828。

**SfM 方法选择**：

| SfM 方法 | F-Score↑ |
|----------|----------|
| COLMAP | 0.645 |
| DF-SfM (LoFTR) | 0.853 |
| DF-SfM (DKM) | 0.842 |

**深度条件消融**：

| k (KNN) | 距离图 | F-Score↑ |
|---------|--------|----------|
| 0 | ✗ | 0.543 |
| 3 | ✗ | 0.753 |
| 3 | ✓ | **0.853** |

### 关键发现

1. **仅用 86.4K 合成数据微调即可超越使用百万级数据训练的方法**：利用了 Stable Diffusion 的强大先验
2. **SfM 稀疏点云是极其有效的多视角信息载体**：将多视角匹配信息"压缩"为条件信号
3. **距离图至关重要**：F-Score 从 0.753 提升到 0.853，帮助网络区分原始 SfM 像素和 KNN 插值像素
4. **KNN 稠密化不可或缺**：直接使用稀疏深度图效果极差（0.543 vs 0.853），因为稀疏信号不适合 VAE encoder
5. **跨域泛化优异**：在室内（ScanNet、Replica）、街景（Waymo）、航拍（UrbanScene3D）等多种场景均表现出色

## 亮点与洞察

1. **将 SfM 和扩散模型巧妙结合**：SfM 提供 what（全局结构和度量尺度），扩散模型提供 how much（密集深度填充），两者优势互补
2. **绕过了多视角匹配的瓶颈**：通过将多视角信息"压缩"为 SfM 点云条件，将问题转化为条件单目深度估计，自然解决了内存消耗和稀疏视角问题
3. **稀疏条件的稠密化设计**：KNN 插值 + 距离图的组合简洁高效——距离图作为"置信度指示器"告诉模型哪些点是可靠的 SfM 观测、哪些是插值
4. **极强的数据效率**：仅在 Hypersim + 3D Ken Burns 两个合成数据集上训练，却能泛化到各种真实场景
5. **灵活的速度-质量权衡**：支持从 12 秒（高质量）到 0.8 秒（快速）的多种推理模式

## 局限性

1. **依赖 SfM 成功**：极端情况下（仅两张图、极小重叠）SfM 无法获得相机位姿和稀疏点云
2. **仅支持静态场景**：无法处理动态元素（如行人、车辆）
3. **SfM 点云质量影响上界**：在弱纹理区域 COLMAP 的稀疏点噪声大、分布稀疏，导致深度估计质量下降
4. **额外的 SfM 运行时间**：虽然深度估计很快，但 SfM 本身可能耗时较长
5. **归一化策略的假设**：扩展 20% 的深度范围是启发式的，极端场景可能不够

## 相关工作与启发

- **Marigold** [Ke et al., 2024]：基于扩散模型的仿射不变单目深度估计，Murre 的直接前身——Murre 在其基础上注入 SfM 引导以获得度量深度
- **Depth Anything** [Yang et al., 2024]：大规模训练的单目深度模型，但缺乏度量尺度
- **NeuralRecon** [Sun et al., 2021]：直接在世界坐标系构建 TSDF 体积的学习方法，但泛化能力差
- **DF-SfM** [He et al., 2023]：基于无检测器匹配的 SfM，适合处理弱纹理场景
- 启发：2D 基础模型（SD）的先验 + 3D 几何约束（SfM）的组合，是一种通用的"先验+约束"范式，可能推广到其他 3D 视觉任务

## 评分

⭐⭐⭐⭐⭐ (9/10)

- 创新性：⭐⭐⭐⭐⭐ — SfM + 扩散模型深度估计的组合思路新颖优雅，KNN+距离图的条件设计简洁有效
- 实用性：⭐⭐⭐⭐⭐ — 在多种真实场景中表现优异，支持灵活的速度-质量权衡
- 实验充分度：⭐⭐⭐⭐⭐ — 覆盖 5 种场景类型、多种基线方法、详细消融实验
- 写作清晰度：⭐⭐⭐⭐⭐ — 方法动机清晰、流程图优秀、消融设计精心

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[CVPR 2025\] Multi-View Pose-Agnostic Change Localization with Zero Labels](multi-view_pose-agnostic_change_localization_with_zero_labels.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](../../ICCV2025/3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](../../ICCV2025/3d_vision/hort_monocular_hand-held_objects_reconstruction_with_transformers.md)

<!-- RELATED:END -->
