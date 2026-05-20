---
title: >-
  [论文解读] Efficient Depth-Guided Urban View Synthesis (EDUS)
description: >-
  [ECCV 2024][3D视觉][城市街景合成] 提出EDUS方法，利用噪声几何先验（单目/双目深度）引导可泛化NeRF，通过前景3D CNN + 背景/天空图像渲染的三部分分解，实现稀疏街景视角下的快速前馈推理和高效逐场景微调。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "城市街景合成"
  - "可泛化NeRF"
  - "稀疏视角"
  - "深度引导"
  - "自动驾驶"
---

# Efficient Depth-Guided Urban View Synthesis (EDUS)

**会议**: ECCV 2024  
**arXiv**: [2407.12395](https://arxiv.org/abs/2407.12395)  
**代码**: [https://xdimlab.github.io/EDUS/](https://xdimlab.github.io/EDUS/)  
**领域**: 3D视觉  
**关键词**: 城市街景合成, 可泛化NeRF, 稀疏视角, 深度引导, 自动驾驶

## 一句话总结

提出EDUS方法，利用噪声几何先验（单目/双目深度）引导可泛化NeRF，通过前景3D CNN + 背景/天空图像渲染的三部分分解，实现稀疏街景视角下的快速前馈推理和高效逐场景微调。

## 研究背景与动机

**领域现状**: 基于NeRF的街景新视角合成方法（Urban Radiance Fields、Block-NeRF等）取得进展，但依赖密集训练图像和大量计算资源。

**现有痛点**:
   - 自动驾驶中车辆高速行驶，大部分内容仅被2-3个视角捕获，视角间重叠严重不足
   - 前向运动导致视差角小，重建不确定性增大
   - 可泛化NeRF方法（IBRNet、MVSNeRF）依赖特征匹配恢复几何，在稀疏纹理不足的街景中表现差
   - 这些方法选取最近参考图像做特征匹配，容易过拟合特定相机位置配置，泛化到不同稀疏度时性能骤降

**核心矛盾**: 如何构建一个高效且可泛化的城市视角合成方法，能鲁棒应对不同稀疏度？

**本文目标** 在稀疏街景视角下实现快速前馈推理和高效微调的新视角合成。

**切入角度**: 用几何先验（深度估计）替代特征匹配来获取几何信息，直接在3D空间操作，避免对参考图像位姿的依赖。

**核心 idea**: 将深度估计先验融入全局3D体积表示，通过SPADE 3D CNN在世界坐标系中直接处理，使方法对参考图像位姿配置不敏感。

## 方法详解

### 整体框架

EDUS将无界街景分解为三个组件：前景（近距离体积内）、背景（远距离物体）和天空。每个组件有独立的可泛化模块。核心是深度引导的前景场：深度估计→点云累积→3D SPADE CNN提取体积特征→结合2D图像特征→解码颜色和密度。训练在多场景上进行，推理时可前馈或快速微调。

### 关键设计

1. **深度引导的可泛化前景场 (Depth-Guided Generalizable Foreground Fields)**:

    - **功能**: 利用深度估计构建3D点云，通过3D CNN提取体积特征来表示近距离前景区域。
    - **核心思路**:
        - **点云累积**: 对N张输入图像用深度估计器预测深度图 $\{D_i\}$，反投影到3D世界坐标系形成点云 $\mathcal{P} \in \mathbb{R}^{N_p \times 3}$：$\mathbf{x} = d\mathbf{R}_i\mathbf{K}^{-1}\mathbf{u} + \mathbf{t}_i$。使用深度一致性检查过滤噪声（阈值 $\sigma = 0.2m$）。
        - **SPADE 3D CNN**: 将点云离散化为体积 $\mathbf{P} \in \mathbb{R}^{H \times W \times D \times 3}$，通过SPADE CNN提取特征体 $\mathbf{F} = f_\theta^{3D}(\mathbf{P})$。SPADE CNN包含3个SPADE残差块和上采样层，多分辨率调制保持外观信息。
        - **2D特征检索**: 选取K=3个最近参考视角，将3D点投影到参考帧获取颜色特征 $\mathbf{f}_{fg}^{2D} \in \mathbb{R}^{3K}$。
        - **解码**: 密度仅由3D特征决定 $\sigma_{fg} = g_\theta(\mathbf{f}_{fg}^{3D})$，颜色由3D+2D特征联合预测 $\mathbf{c}_{fg} = h_\theta(\mathbf{f}_{fg}^{3D}, \mathbf{f}_{fg}^{2D}, \gamma(\mathbf{x}), \mathbf{d})$。
    - **设计动机**: 几何先验不受参考图像位姿影响，使方法对稀疏度变化鲁棒。3D CNN独立处理全局体积，避免局部cost volume对位姿配置的过拟合。SPADE CNN比传统U-Net更好地保持外观信息。

2. **可泛化背景场 (Generalizable Background Fields)**:

    - **功能**: 用基于图像的渲染处理前景体积外的远距离物体。
    - **核心思路**: 远距离物体在图像中占比小，相对深度变化小，图像渲染即可忠实重建：
    $\sigma_{bg}, \mathbf{c}_{bg} = h_\theta^{bg}(\mathbf{f}_{bg}^{2D}, \gamma(\mathbf{x}), \mathbf{d})$
    - **设计动机**: 远距离深度估计不可靠，且透视投影使背景外观变化小，图像渲染足够。

3. **可泛化天空场 (Generalizable Sky Fields)**:

    - **功能**: 将天空建模为依赖视角的环境贴图。
    - **核心思路**: 天空无物理碰撞，帧间外观变化极小：
    $\mathbf{c}_{sky} = h_\theta^{sky}(\mathbf{f}_{sky}^{2D}, \mathbf{d})$
    - **设计动机**: 天空是无限远区域，不需要位置信息。

4. **场景分解与组合渲染 (Scene Decomposition)**:

    - **功能**: 将前景、背景的体渲染结果与天空颜色组合。
    - **核心思路**: 沿光线采样点按位置分配到前景或背景模块，累积得到颜色和alpha:
    $\mathbf{C} = \mathbf{C}^{(fg+bg)} + (1 - \alpha^{(fg+bg)})\mathbf{c}_{sky}$
   使用预训练分割模型提供天空掩码监督。

### 损失函数 / 训练策略

- **训练损失**: $\mathcal{L}_{training} = \mathcal{L}_{rgb} + \lambda_1\mathcal{L}_{lidar} + \lambda_2\mathcal{L}_{sky} + \lambda_3\mathcal{L}_{entropy}$
- **微调损失**: $\mathcal{L}_{fine-tuning} = \mathcal{L}_{rgb} + \lambda_2\mathcal{L}_{sky} + \lambda_3\mathcal{L}_{entropy}$（无LiDAR）
- **LiDAR损失**: 改进的line-of-sight loss，使用指数衰减的bound width $\epsilon$（从0.5m衰减到0.1m）：
    - $\mathcal{L}_{empty}$: 抑制近端空间权重
    - $\mathcal{L}_{near}$: 鼓励表面附近密度集中
    - $\mathcal{L}_{dist}$: 抑制远端空间权重
- **熵正则化**: 惩罚半透明重建，鼓励不透明渲染
- **训练技巧**: 随机掩码输入体积（类似MAE增强补全能力），分层采样，逐帧外观嵌入
- **训练细节**: Adam优化器，学习率 $5 \times 10^{-3}$，$\lambda_1=0.1, \lambda_2=1, \lambda_3=0.002$，RTX 4090训练500k步约2天

## 实验关键数据

### 主实验 — 可泛化方法对比

在KITTI-360训练（80个场景），5个验证场景+5个Waymo场景测试。

| 方法 | 设置 | KITTI-360 drop50% PSNR↑ | drop80% PSNR↑ | Waymo drop50% PSNR↑ |
|------|------|------------------------|---------------|---------------------|
| IBRNet | 前馈 | 19.99 | 15.96 | 21.28 |
| MVSNeRF | 前馈 | 17.73 | 16.50 | 19.58 |
| MuRF | 前馈 | 22.19 | 18.69 | 23.12 |
| **EDUS** | **前馈** | **21.93** | **19.63** | **23.16** |
| MuRF | 微调 | 23.71 | 19.70 | 28.30 |
| **EDUS** | **微调** | **24.43** | **20.91** | **28.45** |

### 与逐场景优化方法对比

| 方法 | drop50% PSNR | drop80% PSNR | drop90% PSNR | 耗时 |
|------|-------------|-------------|-------------|------|
| MixNeRF | 21.50 | 18.89 | 17.89 | ~51min |
| SparseNeRF | 21.34 | 19.18 | 17.94 | ~35min |
| 3DGS | **24.37** | 19.80 | 17.46 | ~29min |
| **EDUS (微调)** | 24.43 | **20.91** | **19.16** | **~5min** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SPADE CNN vs U-Net | SPADE更好 | U-Net在新场景产生模糊伪影 |
| 仅3D特征 | 缺乏高频细节 | 点云离散化限制分辨率 |
| 仅2D特征 | 几何差 | 特征匹配在稀疏场景下不可靠 |
| 随机掩码 | 增强补全能力 | 类似MAE的训练策略 |

### 关键发现

- EDUS在高稀疏度（drop 80%/90%）下优势更明显，因为几何先验不依赖参考图像位姿
- 全局体积方法比MuRF的局部体积方法收敛更快（5分钟 vs 50分钟）
- 跨数据集泛化性强：KITTI-360训练的模型在Waymo上表现良好
- 内存效率高：全分辨率推理仅需6GB，MuRF需要16.2GB

## 亮点与洞察

- **几何先验替代特征匹配**: 核心洞察是深度预测虽然有噪声，但不依赖参考图像位姿，比基于特征匹配的方法更鲁棒
- **全局vs局部体积**: 全局体积只需更新一次即可适应新场景，微调时仅更新特征体积而非整个网络，极大加速收敛
- **分而治之**: 前景/背景/天空的三分策略让每个模块使用最适合的表示（3D几何 vs 2D图像渲染）
- **微调效率**: 5分钟微调即可达到SOTA，比其他方法快5-10倍

## 局限与展望

- 前景体积范围固定（±12.6m × [-3, 9.8m] × [-20, 31.2m]），可能不适应所有场景布局
- 3D CNN仍有平滑偏置，高频细节依赖2D特征补充
- 训练时使用LiDAR监督，但测试时假设RGB-only设置
- 体素分辨率0.2m限制了细粒度重建

## 相关工作与启发

- **vs IBRNet/MVSNeRF**: 这些方法基于特征匹配恢复几何，EDUS用几何先验直接在3D空间操作
- **vs MuRF**: MuRF在目标视角构建cost volume，泛化性好但微调慢。EDUS全局体积微调极快
- **vs PointNeRF**: PointNeRF也使用点云但直接作为辐射场，EDUS用3D CNN细化噪声点云
- **vs 3DGS**: 3DGS在稠密视角下优秀但在高稀疏度下性能骤降，EDUS保持鲁棒
- **启发**: 几何先验+可泛化架构的范式可扩展到室内场景或动态场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 几何先验替代特征匹配的思路简洁有效，场景分解设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多稀疏度、多基线对比，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 5分钟微调达到SOTA对自动驾驶应用有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](../../CVPR2025/3d_vision/depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[ECCV 2024\] MegaScenes: Scene-Level View Synthesis at Scale](megascenes_scene-level_view_synthesis_at_scale.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[ECCV 2024\] Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
