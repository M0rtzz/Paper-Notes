---
title: >-
  [论文解读] GAS: Generative Avatar Synthesis from a Single Image
description: >-
  [ICCV 2025][3D视觉][人体Avatar生成] 提出GAS框架，通过将泛化NeRF重建的密集外观线索与视频扩散模型结合，统一新视角和新姿态合成为视频生成任务，配合模态切换器解耦两种任务，实现从单张图像生成视角一致和时序连贯的人体Avatar。 人体Avatar生成在游戏、电影、体育和远程呈现等领域应用广泛…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "人体Avatar生成"
  - "单图像"
  - "视频扩散"
  - "NeRF"
  - "多视角一致性"
---

# GAS: Generative Avatar Synthesis from a Single Image

**会议**: ICCV 2025  
**arXiv**: [2502.06957](https://arxiv.org/abs/2502.06957)  
**代码**: [项目页面](https://humansensinglab.github.io/GAS/)  
**领域**: 3D视觉  
**关键词**: 人体Avatar生成, 单图像, 视频扩散, NeRF, 多视角一致性

## 一句话总结

提出GAS框架，通过将泛化NeRF重建的密集外观线索与视频扩散模型结合，统一新视角和新姿态合成为视频生成任务，配合模态切换器解耦两种任务，实现从单张图像生成视角一致和时序连贯的人体Avatar。

## 研究背景与动机

人体Avatar生成在游戏、电影、体育和远程呈现等领域应用广泛，但现有技术通常需要昂贵的采集系统。

**现有方法的两大阵营及其局限**：

**回归式泛化方法**（GHNeRF, GPS-Gaussian等）：
   - 结合3D人体先验（SMPL）支持稀疏甚至单视角输入
   - 但回归性质导致**一对多映射取均值**，输出模糊
   - 仅限刚体变形，无法模拟衣物动态

**生成式扩散方法**（Animate Anyone, Champ, Human4DiT）：
   - 条件化在稀疏人体模板（深度/法线图）上生成高质量结果
   - 但稀疏条件信号与真实外观之间存在**间隙**，导致多视角闪烁和时序不一致

**GAS的核心洞察**：将NeRF重建的**密集外观线索**作为扩散模型的条件——比稀疏法线图提供更丰富的结构信息，弥合条件信号与真实外观的间隙。

## 方法详解

### 整体框架

两阶段训练：
1. **训练泛化人体NeRF**：在多视角数据集上训练，支持从单图像渲染任意视角/姿态
2. **训练视频扩散模型**：以NeRF渲染 + SMPL法线图为条件，学习分布偏移

### 第一阶段：泛化人体NeRF

基于单视角泛化NeRF，通过逆LBS将目标空间点变换到SMPL规范空间查询特征：

$$\boldsymbol{\sigma}(\boldsymbol{x}), \boldsymbol{c}(\boldsymbol{x}) = \mathcal{F}(\boldsymbol{x}, \boldsymbol{p}, \gamma_d(\boldsymbol{d}))$$

NeRF渲染虽然模糊（均值回归问题），但提供了密集且3D一致的外观条件。

### 第二阶段：视频扩散模型

基于Stable Video Diffusion (SVD)，三种条件输入融合：

1. **NeRF渲染** $C_{\text{nerf}}$：VAE编码 + 小CNN → 密集外观线索
2. **SMPL法线图** $C_{\text{smpl}}$：2D卷积提取 → 几何结构线索
3. **参考图像** $C_{\text{vae}}$：VAE编码 → 外观保持

$C_{\text{nerf}}$ 和 $C_{\text{smpl}}$ 通过逐元素相加融合后注入UNet第一卷积层。

### 统一的视角-姿态合成

**核心创新**：将新视角合成和新姿态合成统一为视频生成任务：
- 新视角：固定姿态，变化相机轨迹 $\{P_1, ..., P_T\}$
- 新姿态：固定相机，变化SMPL姿态 $\{\boldsymbol{\theta}_1, ..., \boldsymbol{\theta}_T\}$

两种任务共享模型参数，互联网动态视频用于姿态合成训练自然迁移到视角合成，提升泛化能力。

### 模态切换器

直接联合训练会导致动态运动破坏视角一致性。引入one-hot切换器 $\boldsymbol{s}$ 与时间嵌入拼接后注入UNet：

$$\mathcal{L}_{\mathcal{U}_\theta} = \mathbb{E}[\|\epsilon - \mathcal{U}_\theta(Z_t, t, \boldsymbol{h}_{\text{clip}}, C_{\text{vae}}, C_{\text{nerf}}, C_{\text{smpl}}, \boldsymbol{s})\|]$$

切换器让网络区分：视角合成→优先视角一致性，姿态合成→优先逼真变形。

### 训练细节

- NeRF：在MVHumanNet上训练
- 扩散模型：基于SVD 1.1初始化，8张A100训练3天，150k迭代
- 推理CFG：视角合成用三角形CFG（正面1→背面2→正面1），姿态合成固定CFG=2

## 实验

### 新视角合成（主实验）

| 方法 | THuman PSNR↑ | 2K2K PSNR↑ | THuman LPIPS↓ | 2K2K FVD↓ |
|:---|:---:|:---:|:---:|:---:|
| Animate Anyone | 22.48 | 18.48 | 0.061 | 1422.1 |
| Champ | 20.96 | 22.14 | 0.074 | 480.3 |
| Animate Anyone* | 25.20 | 26.22 | 0.046 | 286.4 |
| Champ* | 23.89 | 25.66 | 0.054 | 279.3 |
| **GAS (Ours)** | **26.77** | **28.82** | **0.041** | **191.3** |

*标记表示在本文3D扫描数据集上微调后的版本。GAS在所有指标上均大幅领先。

### 新姿态合成

| 方法 | TikTok PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|:---|:---:|:---:|:---:|:---:|
| Animate Anyone | 17.21 | 0.762 | 0.225 | 1274.1 |
| Champ | 18.48 | 0.806 | 0.182 | 585.0 |
| Champ* | 18.57 | 0.797 | 0.187 | 893.7 |
| **GAS (Ours)** | **19.11** | **0.833** | **0.176** | **362.0** |

### 消融实验

| 消融内容 | 关键发现 |
|:---|:---|
| 无NeRF条件 | 仅用SMPL法线图条件导致外观不一致（多视角闪烁） |
| 无切换器 | 联合训练损害视角一致性（动态运动干扰） |
| 仅3D扫描训练 | 泛化能力不足，野外数据上质量差 |
| +互联网视频 | 通过参数共享提升视角合成泛化（域内外均提升） |

### 关键发现

1. **NeRF条件的关键作用**：密集外观线索是多视角一致性的基石，稀疏法线图不够
2. **参数共享的跨任务迁移**：姿态合成训练的多样性自然提升视角合成质量
3. **切换器的必要性**：不同任务有不同的"一致性"要求，需要显式解耦
4. **三角形CFG**：视角合成的CFG从正面到背面线性增加有效平衡保真和生成

## 亮点与洞察

1. **密集条件替代稀疏条件**：用NeRF渲染代替法线图作为扩散条件是高度实用的简洁想法
2. **双任务统一**：同一模型同时做视角和姿态合成，参数效率高且互相增强
3. **实用的训练数据策略**：3D扫描（少量精确）+ 互联网视频（大量多样）的混合训练

## 局限性

1. 依赖NeRF的SMPL拟合精度，不准确的SMPL会导致NeRF渲染伪影并传播到扩散模型
2. 当前仅支持20帧视频生成，长序列需要滑动窗口
3. 极端遮挡区域（如完全背面）NeRF渲染质量受限

## 相关工作

- **泛化人体NeRF**：GHNeRF, GPS-Gaussian, EVA3D
- **生成式人体动画**：Animate Anyone, Champ, Human4DiT
- **视频扩散模型**：SVD, Sora

## 评分

- 新颖性：⭐⭐⭐⭐ — 密集NeRF条件+统一视角/姿态的视频生成框架
- 技术深度：⭐⭐⭐⭐ — 模态切换器、CFG策略设计有针对性
- 实验完整性：⭐⭐⭐⭐ — 多数据集(3D扫描+多视角视频+TikTok)，充分对比和消融
- 实用价值：⭐⭐⭐⭐ — 单图像输入，支持视角和姿态控制，应用前景广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](../../CVPR2025/3d_vision/anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[ICCV 2025\] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior](driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)

</div>

<!-- RELATED:END -->
