---
title: >-
  [论文解读] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction
description: >-
  [CVPR 2025][3D视觉][新视角合成] 本文提出深度引导的束采样策略(GDB)，将相邻光线打包成束并通过球面-锥体采样来联合处理，同时根据深度置信度自适应分配采样点数量，应用于ENeRF和MVSGaussian后在DTU数据集上实现PSNR提升1.27dB且FPS提升47%。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "新视角合成"
  - "泛化NeRF"
  - "束采样"
  - "深度引导"
  - "高效渲染"
---

# Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2505.19793](https://arxiv.org/abs/2505.19793)  
**代码**: [https://github.com/KLMAV-CUC/GDB-NeRF](https://github.com/KLMAV-CUC/GDB-NeRF)  
**领域**: 3D视觉  
**关键词**: 新视角合成, 泛化NeRF, 束采样, 深度引导, 高效渲染

## 一句话总结

本文提出深度引导的束采样策略(GDB)，将相邻光线打包成束并通过球面-锥体采样来联合处理，同时根据深度置信度自适应分配采样点数量，应用于ENeRF和MVSGaussian后在DTU数据集上实现PSNR提升1.27dB且FPS提升47%。

## 研究背景与动机

**领域现状**：泛化NeRF方法可以在不需要逐场景优化的前提下从多视角图像合成新视角，代表性方法如ENeRF、MVSGaussian等已经实现了不错的质量。然而，渲染高分辨率图像时仍需逐像素密集采样所有光线，计算开销依然很大。

**现有痛点**：现有泛化NeRF方法虽然通过深度引导减少了每条光线上的采样点数（如ENeRF仅采样2个点/光线），但仍然是每个像素独立发射一条光线，没有利用相邻像素之间的空间相关性。这意味着对于平滑区域中的大量像素，采样实际上是冗余的。

**核心矛盾**：Plenoptic sampling理论指出，自然场景通常是分段平滑的，高频信息只集中在边缘和深度不连续处。然而现有方法对所有像素使用相同的采样密度，在平滑区域浪费了大量计算，在复杂区域可能又采样不足。

**本文目标**：设计一种利用场景空间局部性的采样策略，在平滑区域减少采样、复杂区域增加采样，从而大幅提升渲染效率而不牺牲质量。

**切入角度**：从plenoptic sampling理论出发，观察到可以将相邻光线分组为"束"，用一个锥体代替多条独立光线进行联合采样。同时利用深度置信度作为场景复杂度的代理指标，自适应调整采样密度。

**核心 idea**：用束（bundle）替代单条光线进行采样，通过球面编码获取联合表示和逐光线细节表示，再结合深度引导的自适应采样，同时减少光线数量和每束的采样点数，实现效率和质量的双重提升。

## 方法详解

### 整体框架

给定多视角源图像及其相机参数，首先通过多尺度特征提取和代价体积构建来估计深度范围。然后将目标视角的像素按 $K \times K$ 分组为若干束，每个束模型化为一个锥体。在锥体内使用内切球进行采样，结合深度预测结果自适应分配采样点数。对每个球面样本提取联合束表示和逐光线表示，经体积渲染聚合后通过神经渲染器解码输出最终图像。

### 关键设计

1. **球面-锥体束采样 (Sphere-based Cone Sampling)**:

    - 功能：将 $K \times K$ 相邻光线打包为一个束，用锥体模型统一采样
    - 核心思路：将目标视角图像划分为 $H/K \times W/K$ 个束，每个束对应 $K \times K$ 个像素。从相机投影中心出发，沿所有光线方向的平均方向发射一个锥体，该锥体在图像平面的截面是以 $r_{tar} = K \cdot r_p$ 为半径的圆盘。在锥体内采样若干内切球 $\mathcal{S}(\dot{x}, \dot{r})$，球心为对应光线交点的质心、半径由锥体几何关系确定。这样采样点数从 $O(HWN)$ 降低到约 $O(HWN/K^2)$
    - 设计动机：相邻像素通常对应相似的场景内容，独立采样是冗余的。锥体采样一次覆盖多个像素，极大减少了总采样数

2. **多视角图像球面编码 (Multi-view Image-based Sphere Encoding)**:

    - 功能：为每个球面样本提取"联合束表示"（低频）和"逐光线表示"（高频）
    - 核心思路：联合束表示利用**mipmap层次结构**——将源视角特征图构建为mipmap金字塔，每个球投影到源视角后根据其覆盖面积确定合适的mipmap层级 $l = \log_2(r_{src}/r_p)$，通过三线性插值提取预滤波特征。逐光线表示则将球内 $K \times K$ 条光线对应的3D点投影到源视角提取pixel-aligned颜色，保留高频细节。两种表示拼接形成完整的采样特征
    - 设计动机：束采样不可避免会丢失高频信息，mipmap提供了适配采样尺度的低频特征，逐光线颜色补偿了高频细节，实现了效率与细节的平衡

3. **深度引导自适应采样 (Depth-Guided Adaptive Sampling)**:

    - 功能：根据深度置信度动态调整每个束的采样点数量
    - 核心思路：利用深度估计模块预测每个束的深度值和置信区间 $R$。采样点数通过 $N_{\mathcal{C}} = \max(\lceil 2R/\delta_s \rceil, N_{max})$ 计算，其中 $\delta_s$ 是最小采样间距。深度范围窄（平滑区域、深度确信）的束只需1-2个采样点，深度范围宽（边缘、遮挡区域）的束分配更多采样点。这与plenoptic sampling理论一致：每个样本需要覆盖窄的视差范围
    - 设计动机：不同于ENeRF等方法对所有光线使用固定采样数，自适应采样将计算资源重新分配到真正需要的地方，实现了50%以上的FPS提升

### 损失函数 / 训练策略

采用与MVSGaussian相同的损失函数。训练分两阶段：先用均匀采样（每个束固定 $N_{max}$ 个球）预训练100个epoch，确保模型稳定初始化；然后切换到深度引导自适应采样阶段继续训练。设置 $N_{max}=6$，$\delta_s$ 为场景深度范围的1/64。

## 实验关键数据

### 主实验 (DTU数据集, 3-view, 512×640)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 采样数/光线 | FPS↑ |
|------|-------|-------|--------|-----------|------|
| ENeRF | 27.61 | 0.957 | 0.089 | 2 | 19.5 |
| MVSGaussian | 28.21 | 0.963 | 0.076 | 1 | 21.5 |
| MuRF | 28.76 | 0.961 | 0.077 | 80 | 0.934 |
| **ENeRF+Ours (2×2)** | **28.86** | **0.964** | **0.073** | 0.42 | **28.6** |
| ENeRF+Ours (4×4) | 28.21 | 0.957 | 0.088 | 0.10 | 43.6 |
| MVSGaussian+Ours | 28.40 | 0.962 | 0.076 | 1 | 23.4 |

### 消融实验 (DTU, 2×2 bundle)

| 配置 | PSNR | FPS | 说明 |
|------|------|-----|------|
| Full model (2×2) | 28.86 | 28.6 | 完整模型 |
| w/o Sphere sampling | 27.66 | 29.2 | 换成球心点采样，PSNR降1.2dB |
| w/o Adaptive sampling | 28.85 | 17.0 | 固定采样，FPS降40% |
| w/o Ray-specific repr. | 28.47 | 29.4 | 丢失高频，PSNR降0.39dB |
| w/o Joint bundle repr. | 27.83 | 33.7 | 仅用逐光线，PSNR降1.03dB |

### 关键发现

- **束采样带来质量和速度的双赢**：ENeRF+Ours (2×2)相比原始ENeRF不仅提速47%（19.5→28.6 FPS），PSNR还提升了1.27dB，这归功于mipmap预滤波特征带来的抗锯齿效果
- **4×4束在速度上更具优势**：FPS达到43.6（是原ENeRF的2.2倍），PSNR仅轻微下降，适合对实时性要求高的场景
- **自适应采样贡献最大的速度提升**：去掉自适应采样后FPS从28.6降到17.0，这说明根据深度置信度减少平滑区域采样是效率提升的关键
- **联合束表示比逐光线表示更重要**：去掉联合束表示后PSNR降1.03dB，而去掉逐光线表示仅降0.39dB（2×2束下），说明预滤波低频特征对整体质量贡献更大
- **跨数据集泛化良好**：在Real Forward-facing和NeRF Synthetic上也取得了与SOTA相当或更优的结果

## 亮点与洞察

- **Plenoptic sampling理论指导网络设计**：这是少有的将经典光场采样理论(Chai 2000)引入深度学习框架的工作，理论分析为采样策略设计提供了明确指导，比纯数据驱动的方法更具可解释性
- **双表示策略的巧妙设计**：联合束表示+逐光线表示的组合策略很像图像处理中低频基础+高频残差的思想，可以迁移到任何需要平衡效率和细节的渲染任务中
- **方法的通用性**：所提策略不依赖特定网络架构，可以即插即用地加速ENeRF和MVSGaussian等不同骨干网络

## 局限与展望

- **深度估计精度是瓶颈**：自适应采样依赖于深度预测的置信度，如果深度估计错误会导致采样不足或过度
- **MVSGaussian+Ours提升有限**：由于MVSGaussian本身只采样1个点/光线，束采样的优势不如在ENeRF上显著
- **预训练阶段增加训练时间**：需要额外100个epoch的均匀采样预训练，增加了总训练成本
- 改进方向：可以探索更灵活的束划分策略（如非均匀大小的束），在复杂区域使用小束而在平滑区域使用大束

## 相关工作与启发

- **vs ENeRF**: ENeRF用深度引导减少每条光线采样数到2个，但仍逐像素发射光线。本文在其基础上进一步将光线打包成束，同时减少光线数和采样数
- **vs Mip-NeRF**: Mip-NeRF提出了锥体采样替代光线采样来实现抗锯齿，但每个像素仍独立一个锥。本文将多个像素的锥合并为一个更大的锥
- **vs MVSGaussian**: MVSGaussian结合MVS和3DGS实现实时泛化渲染，但单点采样限制了进一步提升。本文的策略也能加速3DGS流水线

## 评分

- 新颖性: ⭐⭐⭐⭐ 束采样策略结合plenoptic theory是新颖的思路，双表示策略设计精巧
- 实验充分度: ⭐⭐⭐⭐ 涵盖多个数据集、多种基线对比和完整消融实验
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验分析详尽
- 价值: ⭐⭐⭐⭐ 方法通用性强，对泛化NeRF加速有实际参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](probesdf_light_field_probes_for_neural_surface_reconstruction.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](../../ECCV2024/3d_vision/dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)

</div>

<!-- RELATED:END -->
