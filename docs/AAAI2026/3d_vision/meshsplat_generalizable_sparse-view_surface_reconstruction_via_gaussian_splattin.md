---
title: >-
  [论文解读] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting
description: >-
  [AAAI 2026][3D视觉][稀疏视角重建] 提出MeshSplat，首个基于2DGS的可泛化稀疏视角表面重建框架，通过加权Chamfer Distance损失正则化深度预测和基于不确定性的法线预测网络对齐2DGS朝向，从新视角合成任务中以自监督方式学习几何先验，在稀疏视角网格重建和跨数据集泛化上均达到SOTA。
tags:
  - AAAI 2026
  - 3D视觉
  - 稀疏视角重建
  - 表面重建
  - 2D高斯溅射
  - 前馈网络
  - 跨场景泛化
---

# MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2508.17811](https://arxiv.org/abs/2508.17811)  
**代码**: [https://hanzhichang.github.io/meshsplat_web/](https://hanzhichang.github.io/meshsplat_web/)  
**领域**: 3D视觉  
**关键词**: 稀疏视角重建, 表面重建, 2D高斯溅射, 前馈网络, 跨场景泛化

## 一句话总结

提出MeshSplat，首个基于2DGS的可泛化稀疏视角表面重建框架，通过加权Chamfer Distance损失正则化深度预测和基于不确定性的法线预测网络对齐2DGS朝向，从新视角合成任务中以自监督方式学习几何先验，在稀疏视角网格重建和跨数据集泛化上均达到SOTA。

## 研究背景与动机

3D场景表面重建是3D视觉的基础任务，在AR/VR和具身AI等应用中至关重要。基于NeRF/3DGS的逐场景优化方法在稀疏视角下表现不佳——稀疏视角仅提供有限的多视图几何约束，不足以支持高质量的逐场景几何优化。

**现有方法的局限**：

**NeuS based方法**（如SparseNeuS）：通过几何体素估计隐式SDF场提取网格。缺点是隐式表示效率低，渲染速度慢，局限于物体级场景。

**3DGS前馈方法**（如pixelSplat、MVSplat）：在新视角合成上效果好，但由于3DGS椭球体形状在不同视角下会产生不同的截面平面，导致**表面不一致**，无法有效提取网格。

**核心洞察**：2DGS（2D高斯溅射）是NVS与表面重建之间的天然桥梁。2DGS在不同视角下具有一致的截面平面，天然更适合表示薄表面，可以同时做新视角合成和网格提取。但将2DGS集成到前馈框架中并非简单任务——2DGS对位置和朝向估计**更加敏感**：

- **位置敏感性**：2DGS的薄特性使得深度图预测误差直接导致明显的位置偏移（3DGS由于椭球体积可以容忍更大误差）
- **朝向敏感性**：2DGS的朝向直接决定场景表面法线，朝向预测错误直接导致扭曲的场景表面

## 方法详解

### 整体框架

输入两张图像及投影矩阵，MeshSplat流程如下：
1. CNN + Multi-View Transformer提取特征图
2. Plane Sweeping构建每视图代价体积
3. Weighted Chamfer Distance Loss约束代价体积
4. Gaussian Prediction Network（含深度精化网络 + 法线预测网络）生成像素对齐的2DGS
5. 2DGS渲染新视角用于监督 + 提取场景网格

形式化描述：$\{I_i, \Pi_i\}_{i=1}^{2} \rightarrow \{\mu_j, s_j, r_j, \alpha_j, c_j\}_{j=1}^{2 \times H \times W}$

### 关键设计

#### 1. **代价体积构建与深度预测**

沿用MVSplat的框架，但引入平面扫描构建代价体积。对输入视图 $i$，将深度范围分为 $D=128$ 个深度候选，将另一视图特征图按当前深度候选进行变形（warp），计算点积得到代价体积：

$$V_i^{d_k} = \frac{F_i \cdot F_{j \to i}^{d_k}}{\sqrt{C}}$$

对代价体积沿深度维做Softmax得到深度概率，加权求和得到粗深度图：

$$D_i^{\text{coarse}} = \sum_k W_i^k d_k$$

#### 2. **加权Chamfer Distance损失（WCD Loss）**

理想情况下，相邻视图预测的高斯位置应有大量重叠。普通Chamfer Distance对所有点分配等权重，但由于遮挡和视图差异，不对应像素的chamfer距离很远，统一约束会产生不合理的约束。

**解决方案**：从代价体积中提取每个像素的匹配置信度图：

$$M_i = \max_{d_k} \text{Softmax}_D(V_i)$$

WCD Loss只在高置信度区域施加强约束：

$$\mathcal{L}_{\text{WCD}} = \frac{1}{2}\left(\frac{1}{N_1}\sum_{i=1}^{N_1} M_1(i)\min_j ||p_1^i - p_2^j|| + \frac{1}{N_2}\sum_{i=1}^{N_2} M_2(i)\min_j ||p_2^i - p_1^j||\right)$$

置信度图能清楚标示出无纹理区域和非重叠区域（低置信度），避免对这些区域的错误约束。

#### 3. **基于不确定性的法线预测网络**

2DGS的朝向直接决定场景表面法线。设计轻量级CNN $\phi_{\text{rot}}$ 预测2DGS的旋转四元数 $q$ 和不确定性 $\kappa$：

$$\{q, \kappa\} = \phi_{\text{rot}}(V_i || F_i || I_i), \quad n = R(q) \cdot [0, 0, 1]^T$$

使用Angular von Mises-Fisher分布的负对数似然（NLL）损失进行监督：

$$\mathcal{L}_{\text{AngMF}}(n_i, \hat{n}_i, \kappa_i) = -\log(\kappa_i^2 + 1) + \log(1 + \exp(-\kappa_i\pi)) + \kappa_i \cos^{-1} n_i^T \hat{n}_i$$

以预训练Omnidata模型的输出作为伪ground truth法线监督。采用基于 $\kappa$ 的不确定性引导采样：取 $\kappa$ 最低的top 70%像素 + 随机30%像素进行损失计算。

### 损失函数 / 训练策略

总训练损失：

$$\mathcal{L} = w_1\mathcal{L}_{\text{pho}} + w_2\mathcal{L}_{\text{WCD}} + w_3\mathcal{L}_{\text{normal}}$$

其中 $\mathcal{L}_{\text{pho}} = w_{11}\text{MSE}(I, \hat{I}) + w_{12}\text{LPIPS}(I, \hat{I})$

权重设置：$w_1=1.0$, $w_2=5.0\times10^{-3}$, $w_3=5.0\times10^{-3}$, $w_{11}=1.0$, $w_{12}=0.1$

训练策略：
- Re10K：裁剪到256×256，训练200k步，batch size 12
- Scannet：裁剪到512×384，训练75k步，batch size 4
- Adam优化器，最大学习率 $2\times10^{-4}$
- 单卡NVIDIA A800

## 实验关键数据

### 主实验

Re10K和Scannet数据集表面重建：

| 方法 | Re10K CD↓ | Re10K F1↑ | Scannet CD↓ | Scannet F1↑ |
|------|----------|----------|------------|------------|
| **MeshSplat** | **0.3566** | **0.3758** | **0.2606** | **0.3824** |
| MVSplat | 0.4015 | 0.3100 | 0.3748 | 0.2095 |
| pixelSplat | 1.4423 | 0.0944 | 0.3285 | 0.2948 |
| MVSNeRF | 0.6139 | 0.1407 | 0.5761 | 0.1514 |
| SparseNeuS | 6.0473 | 0.0020 | 7.1860 | 0.0107 |

跨数据集零样本迁移（仅用Re10K训练）：

| 方法 | Re10K→Scannet F1↑ | Re10K→Replica F1↑ |
|------|-------------------|-------------------|
| **MeshSplat** | **0.2956** | **0.0809** |
| MVSplat | 0.1418 | 0.0564 |
| SparseNeuS | 0.0006 | 0.0003 |

深度和法线预测质量：

| 方法 | Depth AbsRel↓ | Normal Mean↓ | Normal <30°↑ |
|------|-------------|-------------|-------------|
| **MeshSplat** | **0.0910** | **33.84** | **0.6026** |
| MVSplat | 0.1692 | 57.16 | 0.1357 |

### 消融实验

Scannet数据集消融：

| # | 配置 | CD↓ | 说明 |
|---|------|-----|------|
| 1 | 3DGS (MVSplat基线) | 0.3748 | 基线 |
| 2 | 2DGS | 0.2948 | 2DGS更适合表面重建 |
| 3 | 2DGS + WCD Loss | 0.2769 | 跨视图深度一致性提升 |
| 4 | 2DGS + NPN | 0.2642 | 法线预测网络贡献最大 |
| 5 | 2DGS + WCD + NPN | **0.2606** | 两者互补 |

模型效率：

| 方法 | 渲染时间(s) | 参数量(M) |
|------|-----------|----------|
| MeshSplat | 0.102 | 13.3 |
| MVSplat | 0.072 | 12.0 |
| SparseNeuS | 7.048 | 0.843 |

### 关键发现

- **2DGS vs 3DGS**：仅替换为2DGS即可将CD从0.3748降到0.2948，验证了2DGS作为NVS与表面重建桥梁的有效性
- **法线预测网络贡献最大**（CD: 0.2948→0.2642），说明2DGS朝向对网格质量的关键影响
- WCD Loss有效解决了非重叠区域的错误约束问题，置信度图能准确反映无纹理区域和非重叠区域
- 仅增加1.3M参数和30ms渲染时间，模型开销极小
- 跨数据集泛化：Re10K训练→Scannet/Replica零样本迁移，F1均显著优于基线
- $\kappa$ 图中高不确定性区域通常对应物体边界，与直觉一致

## 亮点与洞察

1. **2DGS作为桥梁的洞察**：将NVS的训练数据丰富性转化为表面重建的几何先验，巧妙避免了昂贵的3D ground truth标注
2. **WCD Loss设计精巧**：从代价体积中自然导出置信度图，无需额外模块
3. **不确定性引导采样**：在法线损失中基于 $\kappa$ 采样，让网络聚焦于不确定区域学习，提高训练效率
4. **自监督几何学习**：整个框架不需要3D ground truth，仅通过NVS监督学习几何

## 局限与展望

- 弱纹理区域可能预测不连续的深度图（虽然RGB渲染可靠）
- 无法重建输入视图未观察到的区域
- 仅使用两张输入图像，更多视图可能进一步提升
- 未探索生成式方法来补全未见区域
- Re10K数据集没有ground truth网格，需要用COLMAP重建稠密点云作为近似GT

## 相关工作与启发

- MVSplat是最直接的基线（同样的前馈框架，但用3DGS）
- 2DGS（Huang et al.）在逐场景优化设置下证明了其表面重建优势，本文首次将其推广到可泛化设置
- DUSt3R/MASt3R虽然能预测3D点图，但不支持新视角合成和表面重建
- 启发：2DGS在其他前馈3D任务（如全景重建、对象级重建）中也有潜力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将2DGS应用于可泛化稀疏视角表面重建
- 实验充分度: ⭐⭐⭐⭐ — 多数据集评估+跨数据集泛化+深度/法线评估+消融
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，2DGS vs 3DGS对比直观
- 价值: ⭐⭐⭐⭐⭐ — 开辟了2DGS前馈重建的新方向，实用价值高

<!-- RELATED:START -->

## 相关论文

- [SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction](surface-based_visibility-guided_uncertainty_for_continuous_active_3d_neural_reco.md)

<!-- RELATED:END -->
