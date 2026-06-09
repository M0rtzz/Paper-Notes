---
title: >-
  [论文解读] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting
description: >-
  [CVPR 2025][自动驾驶][BEV感知] GaussianLSS 在经典的 Lift-Splat-Shoot（LSS）框架上引入深度不确定性建模，通过计算深度分布的方差并将其转换为 3D 高斯表示，再利用 Gaussian Splatting 高效光栅化生成具有不确定性感知的 BEV 特征…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "BEV感知"
  - "深度不确定性"
  - "Gaussian Splatting"
  - "Lift-Splat-Shoot"
  - "语义分割"
---

# Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2504.01957](https://arxiv.org/abs/2504.01957)  
**代码**: [https://hcis-lab.github.io/GaussianLSS/](https://hcis-lab.github.io/GaussianLSS/)  
**领域**: 自动驾驶  
**关键词**: BEV感知, 深度不确定性, Gaussian Splatting, Lift-Splat-Shoot, 语义分割

## 一句话总结

GaussianLSS 在经典的 Lift-Splat-Shoot（LSS）框架上引入深度不确定性建模，通过计算深度分布的方差并将其转换为 3D 高斯表示，再利用 Gaussian Splatting 高效光栅化生成具有不确定性感知的 BEV 特征，在 nuScenes 上达到了 unprojection 方法的 SOTA，同时比 projection 方法快 2.5 倍、省 70% 显存。

## 研究背景与动机

**领域现状**：BEV 感知是自动驾驶中的核心任务，为 3D 检测、语义分割、运动预测和规划提供统一的空间表示。现有方法分为两大范式：（1）2D unprojection 方法（如 LSS、FIERY）——估计深度后将 2D 特征"提升"到 3D 空间；（2）3D projection 方法（如 BEVFormer、SimpleBEV、PointBEV）——将预定义的 3D 查询投影到图像平面上采样特征，无需显式深度估计。

**现有痛点**：（1）3D projection 方法精度最高但计算量大（3D 网格采样成本高），难以实时部署；（2）传统 LSS 虽然效率高，但严重依赖准确的深度估计——深度估计本身是一个病态问题（ill-posed），深度误差会直接传播到 BEV 表示；（3）现有 LSS 变体虽然使用 softmax 概率分布做"软"深度分配，但缺乏对深度不确定性的显式建模——softmax 在相邻深度 bin 上可能产生差异巨大的概率，导致 BEV 特征不稳定。

**核心矛盾**：unprojection 方法效率高但精度受限于深度估计质量；projection 方法精度高但太慢。需要找到一个既高效又能容忍深度误差的方案。

**本文目标**：（1）在 LSS 框架中引入深度不确定性建模，降低对精确深度估计的依赖；（2）利用 Gaussian Splatting 实现高效的 BEV 特征聚合。

**切入角度**：作者观察到深度分布的方差本身就编码了深度估计的不确定性信息——方差大意味着深度估计不确定，此时应该让特征在更大的空间范围内"扩散"来覆盖可能的目标位置。这恰好与高斯分布的"展开"特性吻合。

**核心 idea**：计算每个像素的深度分布的均值和方差，将其转化为 3D 高斯分布（均值=3D位置，协方差=空间不确定性范围），然后用 Gaussian Splatting 渲染到 BEV 平面上，实现不确定性感知的 BEV 特征聚合。

## 方法详解

### 整体框架

输入多视角图像 → Backbone 提取特征 → CNN 预测 splat 特征 $F_i$、不透明度 $\alpha_i$ 和深度分布 $P_i$ → 深度不确定性变换（均值 $\mu$、方差 $\sigma^2$ → 3D 高斯）→ 多尺度 Gaussian Splatting 渲染到 BEV 平面 → 融合多尺度 BEV 特征 → 分割头输出预测。

### 关键设计

1. **深度不确定性建模**:

    - 功能：从深度概率分布中显式提取不确定性信息
    - 核心思路：在 LSS 的离散深度分布 $P$ 上计算深度均值 $\mu = \sum_{i} P_i(p) d_i$ 和方差 $\sigma^2 = \sum_{i} P_i(p)(d_i - \mu)^2$，然后定义一个容差范围 $\hat{\mathbf{D}} = [\mu - k\sigma, \mu + k\sigma]$。这个范围将"点估计"变成了"带不确定性的区间估计"——当模型对深度不确定时（$\sigma$ 大），特征会在更大的深度范围内扩散。$k$ 是误差容差系数，经验设为 0.5。
    - 设计动机：传统 LSS 中 softmax 深度分布看似是概率化的，但实际上只是做了加权求和，没有利用分布的"展开程度"这一关键信息。方差直接度量了深度估计的可信度。

2. **3D 不确定性变换与高斯表示**:

    - 功能：将 1D 深度不确定性转化为 3D 空间的高斯分布
    - 核心思路：利用相机内参 $I$ 和外参 $E$ 将每个深度 bin 对应的像素-深度点 $(u,v,d_i)$ 反投影到 3D 空间 $p_i^{3d} = E^{-1}(d_i \cdot I^{-1}[u,v,1]^T)$，然后计算 3D 均值 $\mu_{3d} = \sum_i P_i(p) p_i^{3d}$ 和协方差矩阵 $\Sigma = \sum_i P_i(p)(p_i^{3d} - \mu_{3d})(p_i^{3d} - \mu_{3d})^T$。由此得到 3D 高斯 $\mathcal{N}(\mu_{3d}, \Sigma)$，自然表征了空间位置和不确定性。
    - 设计动机：深度不确定性在相机坐标系中是 1D 的，但映射到世界坐标后会沿射线方向展开为 3D 椭球。高斯分布是描述这种空间不确定性的最自然的数学工具。

3. **多尺度 BEV 特征渲染**:

    - 功能：利用 Gaussian Splatting 高效渲染 BEV 特征，并通过多尺度缓解深度均值不一致问题
    - 核心思路：将 3D 高斯（含均值、协方差、特征、不透明度）投影到 BEV 平面，用 alpha-blending 渲染：$\mathbf{F}_{BEV}(\mathbf{x}) = \sum_i F_i \alpha_i \exp(-\frac{1}{2}(\mathbf{x}-\mu_i)^\top\Sigma_i^{-1}(\mathbf{x}-\mu_i))$。为解决相邻像素深度均值跳变导致的 BEV 特征畸变，在多个分辨率（50×50、100×100、200×200）上分别渲染 BEV 特征，然后上采样融合。
    - 设计动机：Gaussian Splatting 的光栅化操作极其高效（基于 tile-based 渲染），且天然支持空间展开（通过协方差矩阵），完美适配不确定性感知的特征聚合。多尺度渲染则借鉴了特征金字塔的思路。

### 损失函数 / 训练策略

使用三个损失函数：分割的 focal loss（$\lambda_1=1$）、centerness 的 L1 loss（$\lambda_2=2$）和 offset 的 L2 loss（$\lambda_3=0.1$）。优化器 AdamW，学习率 $3 \times 10^{-4}$，余弦退火，总 batch size 8，2×RTX 4090，训练 50 epochs。Backbone 为 EfficientNet-B4。

## 实验关键数据

### 主实验

nuScenes Vehicle BEV 语义分割（IoU，224×480 分辨率，无 visibility filtering）：

| 方法 | 类型 | Backbone | IoU↑ |
|------|------|----------|------|
| BEVFormer | 3D projection | RN-50 | 35.8 |
| SimpleBEV | 3D projection | RN-50 | 36.9 |
| PointBEV | 3D projection | EN-b4 | **38.7** |
| FIERY static | 2D unprojection | EN-b4 | 35.8 |
| CVT | 2D unprojection | EN-b4 | 31.4 |
| GaussianLSS | 2D unprojection | EN-b4 | **38.3** |

效率对比：

| 方法 | FPS↑ | 显存 (GiB)↓ | IoU |
|------|------|------------|-----|
| PointBEV | 32.0 | 1.26 | 38.7 |
| CVT | 107.6 | 0.35 | 31.4 |
| GaussianLSS | **80.2** | **0.33** | 38.3 |

### 消融实验

误差容差系数 $k$ 的影响：

| k值 | Vehicle IoU | 说明 |
|-----|-------------|------|
| 0.25 | ~37.0 | 太小，不确定性覆盖不足 |
| 0.50 | **38.3** | 最优 |
| 1.00 | ~38.0 | 仍在合理范围 |
| 2.00 | ~35.0 | 太大，特征过度扩散 |
| 直接预测 extent | 37.0 | 不用不确定性，差 1.3% |

不透明度学习效果：

| Epoch | 保留高斯比例 (α>0.01) | Vehicle IoU |
|-------|---------------------|-------------|
| 初始 | ~100% | 低 |
| 收敛后 | ~20% | 最优 |

### 关键发现

- GaussianLSS 在 unprojection 方法中达到 SOTA（38.3 IoU），仅比最强 projection 方法 PointBEV 低 0.4%，但速度快 2.5 倍、显存省 74%
- 直接预测固定 extent 比学习不确定性差 1.3%，证明不确定性建模优于确定性位置预测
- $k$ 在 0.5-1.25 范围内性能稳定，但过大时特征过度扩散导致精度下降
- 远距离物体（>30m）上 GaussianLSS 表现优于 PointBEV——不确定性建模在深度歧义性大的远距离场景尤为重要
- 训练收敛后 80% 的高斯点不透明度低于 0.01，模型自动学会了只在语义相关区域聚焦

## 亮点与洞察

- **不确定性 ≈ 目标范围**：深度方差不仅反映了估计的不确定性，还隐式编码了物体的空间范围（大物体的深度分布更"散"）。这是一个优雅的双重解释。
- **Gaussian Splatting 的新用法**：将 3DGS 从渲染任务迁移到 BEV 感知中做特征聚合，是一个很有创意的应用。GS 的高效光栅化天然适合需要空间展开的场景。
- **不透明度的自适应剪枝**：模型自动学会用不透明度过滤 80% 的冗余点，实现了无需后处理的自适应稀疏化。

## 局限与展望

- 仅在 nuScenes 上验证，未在 Waymo、Argoverse 等数据集上测试
- 目前仅处理单帧感知，未利用时序信息——加入时序后不确定性可以在时间维度上传播和更新
- 物体形状预测（IoU shape quality）略逊于 projection 方法
- 多尺度渲染引入了额外计算开销，可能对极端实时要求的场景不够理想
- 未来可以扩展到 3D 检测、地图分割等更多 BEV 任务

## 相关工作与启发

- **vs LSS (Philion&Fidler)**：LSS 开创了 lift-splat 范式，但仅做了 softmax 深度加权，缺乏不确定性意识。GaussianLSS 在同一范式上引入方差建模和 GS 渲染，是对 LSS 原理性的升级
- **vs PointBEV**：PointBEV 通过粗到细的 3D 网格策略（projection 方法）达到高精度但速度较慢。GaussianLSS 用 unprojection + GS 实现了接近的精度和更快的速度
- **vs BEVFormer**：BEVFormer 用 3D 查询做 cross-attention，计算量大。GaussianLSS 用高效的 GS 光栅化取代了 attention 操作

## 评分

- 新颖性: ⭐⭐⭐⭐ 将深度不确定性+GS渲染引入BEV感知，思路清晰优雅
- 实验充分度: ⭐⭐⭐⭐ nuScenes多个任务验证，消融全面，有效率分析和远距离分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对实际部署友好（快+省显存），是unprojection方法的实质性推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](../../ICCV2025/autonomous_driving/6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)

</div>

<!-- RELATED:END -->
