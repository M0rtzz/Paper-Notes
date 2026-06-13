---
title: >-
  [论文解读] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction
description: >-
  [ICCV 2025][3D视觉][3DGS] 提出一套面向城市级场景的高效鲁棒3DGS重建框架——通过可见性分区策略、可控LOD生成、细粒度外观变换模块及多种正则化技术，实现了在外观差异大、含瞬态物体的城市数据上高质量重建与实时渲染。 核心问题 城市级场景重建在自动驾驶、城市规划和数字孪生等领域至关重要…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3DGS"
  - "城市场景重建"
  - "LOD策略"
  - "外观变换"
  - "分区训练"
  - "实时渲染"
---

# Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2507.23006](https://arxiv.org/abs/2507.23006)  
**代码**: [https://yzslab.github.io/REUrbanGS](https://yzslab.github.io/REUrbanGS)  
**领域**: 3D视觉  
**关键词**: 3DGS, 城市场景重建, LOD策略, 外观变换, 分区训练, 实时渲染

## 一句话总结
提出一套面向城市级场景的高效鲁棒3DGS重建框架——通过可见性分区策略、可控LOD生成、细粒度外观变换模块及多种正则化技术，实现了在外观差异大、含瞬态物体的城市数据上高质量重建与实时渲染。

## 研究背景与动机

### 核心问题
城市级场景重建在自动驾驶、城市规划和数字孪生等领域至关重要。3DGS以其显式表示和实时渲染能力成为主流选择，但将其扩展至城市尺度面临三大挑战：

### 挑战分析

**可扩展性瓶颈**：场景越大Gaussian数量越多。重建MipNeRF360 Bicycle场景需600万+Gaussians，24GB GPU超过1100万即OOM。城市场景问题更严峻。

**外观不一致**：城市数据采集跨时间（季节/天气/光照变化），同一物体在不同图像中外观显著差异。3DGS倾向于为每个视角的外观差异创建冗余Gaussians，产生浮动伪影。

**瞬态物体干扰**：行人、车辆等瞬态物体不可避免，进一步产生伪影。

### 现有方法不足
- **VastGaussian**：用CNN做颜色变换处理外观差异，但图像级变换不稳定且不灵活；未解决实时渲染问题
- **CityGaussian/Hierarchical-3DGS**：训练时不控制资源，依赖后处理压缩+大量微调，大场景高压缩比下质量损失严重
- **Taming3DGS**：控制密度化策略但限于小场景
- **Grendel-GS**：多GPU方案，硬件随场景线性增长不实际

## 方法详解

### 整体框架
在原始3DGS基础上，从预处理到训练到渲染全链路增强：
1. **场景分区+可见性选图**（预处理效率）
2. **分区内优先密度化**（训练效率）
3. **可控LOD生成+动态选择**（渲染效率）
4. **外观变换+正则化**（重建质量）

### 关键设计

#### 1. 场景分区与基于可见性的图像选择
水平分区后，对分区外图像计算**基于点的可见性**：
- SfM生成3D点云和2D特征点关联
- 将3D点投影到未选图像平面，计算凸包面积 $V_i$
- 提取分区内特征点计算凸包面积 $V_{ij}$
- 可见性 = $V_{ij}/V_i$，仅高可见性图像参与训练
- 特征点天然具有遮挡感知，避免冗余图像选择

**分区再平衡**：中心分区图像多于边缘分区。图像过少的分区与最小邻居合并，过多的细分，迭代至均匀。

#### 2. 分区内优先密度化
分区外区域无需过度资源分配，但简单提高阈值会导致分区内Gaussian向外扩散补偿。提出距离相关阈值：
$$\tau_i = \hat{\tau}_{min} \left(\frac{\min(d_i, \hat{d}_{max})}{\hat{d}_{max}} \cdot (\eta - 1) + 1\right)$$
其中 $d_i$ 是第$i$个Gaussian到分区的距离。分区内阈值 $\hat{\tau}_{min}$，远处阈值线性增至 $\hat{\tau}_{max} = \hat{\tau}_{min} \cdot \eta$。仅当均值梯度 $\bar{\Delta}_{G_i} > \tau_i$ 时才密度化。

#### 3. 可控LOD生成（自底向上）
原始3DGS缺乏资源约束。扩展Taming3DGS的可控密度化策略，定义多级LOD参数：
- 预算 $B_1 < B_2 < \cdots < B_l$
- 密度化间隔 $T_1 > T_2 > \cdots > T_l$  
- 图像下采样因子 $D_1 < D_2 < \cdots < D_l = 1$

低级别用低预算、长间隔、低分辨率训练，不关注高频细节。每级完成后保存checkpoint，切换参数训练下一级——完全端到端，无需后处理压缩。

**渲染时动态选择**：按分区-相机距离选择LOD级别，近处高级远处低级，不可见分区剔除。配合StopThePop的tile-based culling进一步加速。

#### 4. 外观变换模块（细粒度）
为图像和每个3D Gaussian分别分配embedding $\ell^{(\mathcal{I})}$ 和 $\ell^{(\mathcal{G})}$，通过轻量MLP预测逐Gaussian的颜色偏移 $\Delta c$ 和不透明度偏移 $\Delta o$：

**相似性正则化**（邻近Gaussians应有相似外观变换）：
$$\mathcal{L}_{sim} = \frac{1}{|M|\binom{k}{2}} \sum_{i \in M} \sum_{j,l \in knn_{i;k}} w_{i,j} \left(1 - \frac{\ell_i^{(\mathcal{G})} \cdot \ell_j^{(\mathcal{G})}}{\|\ell_i^{(\mathcal{G})}\| \|\ell_j^{(\mathcal{G})}\|}\right)$$
其中 $w_{i,j} = \exp(-\lambda_w \|\mu_i - \mu_j\|)$ 为距离衰减因子。

**不透明度偏移正则化**（多数外观变换不涉及透明度变化）：
$$\mathcal{L}_{\Delta o} = \frac{1}{N} \sum_{i=1}^N \Delta o_i$$

#### 5. 尺度正则化（抑制异常Gaussian）
**最大值约束**（防止Gaussian增长至不合理大小）：
$$\mathcal{L}_{ms} = \frac{\sum_i \mathbb{1}\{S_i > s_{max}\} \cdot S_i}{\sum_i \mathbb{1}\{S_i > s_{max}\} + \delta}$$

**比例约束**（防止高度各向异性形状）：
$$r_i = \frac{\max(S_i)}{\text{median}(S_i)}, \quad \mathcal{L}_r = \frac{\sum_i \mathbb{1}\{r_i > r_{max}\} \cdot r_i}{\sum_i \mathbb{1}\{r_i > r_{max}\} + \delta}$$

#### 6. 深度正则化 + 抗锯齿
用Depth Anything V2预测伪深度，用SfM点云对齐至真实尺度，交替使用硬/软深度正则化。抗锯齿来自Mip-Splatting，细节增强用AbsGS。

### 总损失
$$\mathcal{L}' = \mathcal{L}_{3DGS} + 0.2\mathcal{L}_{sim} + 0.05\mathcal{L}_{\Delta o} + \lambda_d \mathcal{L}_d + 0.05(\mathcal{L}_{ms} + \mathcal{L}_r)$$
$\lambda_d$ 从0.5指数衰减至0.01。

## 实验

### 主实验：三个城市级场景

| 方法 | Rubble SSIM/PSNR | JNU-ZH SSIM/PSNR | BigCity SSIM/PSNR |
|------|------|------|------|
| Switch-NeRF | 0.544/23.05 | 0.574/21.96 | 0.469/20.39 |
| CityGaussian (no LOD) | 0.813/25.77 | 0.776/22.57 | 0.825/24.57 |
| 3DGS | 0.796/25.72 | 0.763/22.02 | 0.830/24.52 |
| **Ours (no LOD)** | **0.826/27.29** | **0.822/25.85** | **0.847/26.62** |
| CityGaussian (LOD) | 0.785/24.90 | 0.770/22.33 | 0.712/22.24 |
| Hierarchical-3DGS (LOD) | 0.741/23.38 | 0.760/21.12 | 0.775/23.17 |
| **Ours (LOD)** | **0.814/27.03** | **0.816/25.71** | **0.838/26.41** |

关键发现：LOD模式下质量仍超越其他方法的non-LOD结果，FPS显著提升至63-100 FPS。

### LOD预算消融

| 预算 B (×100) | SSIM | PSNR | #G(M) | FPS |
|------|------|------|------|------|
| (1024,2048,4096) | 0.771 | 26.13 | 1.61 | 126.9 |
| (4096,8192,16384) | 0.814 | 27.03 | 3.60 | 99.7 |
| (8192,16384,32768) | 0.816 | 27.11 | 3.80 | 96.4 |

预算超过一定阈值后质量不再提升，场景的内在复杂度决定了所需Gaussian数量上限。

### 组件消融

| 配置 | Rubble SSIM/PSNR | JNU-ZH SSIM/PSNR |
|------|------|------|
| w/o 可见性选图 | 0.803/26.95 | 0.809/25.14 |
| w/o 外观模块 | 0.771/25.17 | 0.780/22.57 |
| **完整方法** | **0.826/27.29** | **0.822/25.85** |

外观模块对跨时间采集的JNU-ZH数据集提升最为显著（PSNR +3.28dB）。

## 亮点与洞察
1. **全链路系统设计**：从分区→密度化→LOD→外观→正则化的完整技术栈，各组件协同解决城市场景重建
2. **自底向上LOD**优于先训后压缩：低级别用低频信息训练，高级别在低级上增量优化，质量更高且避免压缩损失
3. **Gaussian级外观变换**比图像级更精细灵活：重建完成后可通过改变image embedding实现外观编辑，且不影响渲染速度
4. **可见性选图**：特征点天然遮挡感知，比基于距离或视锥的选图策略更合理

## 局限性
1. 各组件超参数较多（LOD级数、预算、分区阈值等），不同场景可能需要调参
2. 依赖SfM预处理和Depth Anything V2预测，pipeline较长
3. 瞬态物体移除依赖开放词汇检测器，可能遗漏非典型动态物体
4. 评估仅限航拍场景，街景场景（如自动驾驶视角）未验证

## 相关工作
- **大场景重建**: Block-NeRF（分治NeRF）→ VastGaussian（分治3DGS）→ CityGaussian/Hierarchical-3DGS（LOD-3DGS）
- **外观处理**: NeRF-W（图像embedding）→ SWAG → 本文（Gaussian级embedding + 双流MLP）
- **资源控制**: Taming3DGS（可控密度化）→ 本文（LOD扩展）

## 评分
- 新颖性：⭐⭐⭐⭐ — 系统性集成而非单点突破，但LOD策略和外观模块设计有创新
- 技术深度：⭐⭐⭐⭐ — 各组件设计合理且有理论支撑
- 实验充分度：⭐⭐⭐⭐ — 多场景、有消融，但缺少与最新方法（Grendel-GS等）对比
- 实用价值：⭐⭐⭐⭐⭐ — 直接解决城市级重建+实时渲染的工程痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](../../CVPR2025/3d_vision/guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)

</div>

<!-- RELATED:END -->
