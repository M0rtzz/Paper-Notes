---
title: >-
  [论文解读] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
description: >-
  [ECCV 2024][3D视觉][3D高斯溅射] 发现同时训练两个 3DGS 辐射场时它们在高斯位置和渲染结果上的差异（disagreement）与重建质量负相关，据此提出 CoR-GS 通过协同剪枝和伪视角协同正则化来抑制不准确重建，在稀疏视角下实现 SOTA 新视角合成。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D高斯溅射
  - 稀疏视角
  - 新视角合成
  - 协同正则化
  - 点云剪枝
---

# CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization

**会议**: ECCV 2024  
**arXiv**: [2405.12110](https://arxiv.org/abs/2405.12110)  
**代码**: 有 (https://jiaw-z.github.io/CoR-GS)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 稀疏视角, 新视角合成, 协同正则化, 点云剪枝

## 一句话总结

发现同时训练两个 3DGS 辐射场时它们在高斯位置和渲染结果上的差异（disagreement）与重建质量负相关，据此提出 CoR-GS 通过协同剪枝和伪视角协同正则化来抑制不准确重建，在稀疏视角下实现 SOTA 新视角合成。

## 研究背景与动机

3D Gaussian Splatting（3DGS）通过一组 3D 高斯表示场景实现了高质量实时新视角合成，但在**稀疏训练视角**下容易过拟合，导致新视角渲染质量下降。

现有改进方法（如 FSGS）主要依赖预训练深度估计器的外部深度先验作为正则化，但外部深度监督可能引入额外噪声。本文提出一个全新的**协同正则化**视角：

核心发现：训练两个 3DGS 辐射场表示同一场景时，由于密度控制（densification）实现中的随机性，两者在以下方面表现出差异：
- **点 disagreement**：高斯位置不同
- **渲染 disagreement**：渲染像素不同

关键洞察：**这两种 disagreement 与重建准确度呈负相关** — 差异越大的区域重建越不准确。因此可以无需 ground truth 就识别出不准确的重建区域。

## 方法详解

### 整体框架

CoR-GS 同时训练两个 3DGS 辐射场 $\Theta^1$ 和 $\Theta^2$，在训练过程中进行协同正则化：

1. **Co-pruning**：基于点 disagreement 识别并剪枝位于不准确位置的高斯
2. **Pseudo-view Co-regularization**：基于渲染 disagreement 在伪视角上抑制不一致的渲染结果

训练完成后只保留一个辐射场用于推理。

### 关键设计

#### 1. 点 Disagreement 度量

将两个辐射场的高斯中心视为两个点云，用以下指标度量差异：
- **Fitness**：在最大距离 $\tau{=}5$ 下计算重叠区域
- **RMSE**：计算对应点的平均距离

实验观察：两种 disagreement 在 densification 期间显著增长，因为密度控制创建新高斯时对场景几何是盲目的。

#### 2. Disagreement 与重建质量的负相关性

通过逐步遮掩掉 disagreement 最高的区域，发现剩余区域的重建质量（PSNR、SSIM）持续提升。这证实了 disagreement 可以作为重建质量的无监督代理指标。

#### 3. Co-Pruning

基于点 disagreement 进行协同剪枝：

- 用 KNN 对两个辐射场建立点匹配关系：$f(\theta_i^1) = \text{KNN}(\theta_i^1, \Theta^2)$
- 设最大容许距离 $\tau{=}5$ 计算非匹配掩码 $M$
- 在另一个辐射场中没有近邻匹配的高斯视为**位于不准确位置的异常值**，被剪枝
- 每 5 次优化/密度控制交替后执行一次 co-pruning

效果：减少了远离重建场景的分散高斯，使表示更紧凑。

#### 4. 伪视角协同正则化（Pseudo-view Co-regularization）

通过在未见视角上抑制渲染 disagreement 来提升泛化：

- 从两个最近的训练视角插值采样伪视角：$P' = (t + \epsilon, q)$
- 在伪视角上分别渲染两个辐射场的图像 $I'^1, I'^2$
- 计算两者的颜色差异作为正则化项：

$$\mathcal{R}_{pcolor} = (1-\lambda)\mathcal{L}_1(I'^1, I'^2) + \lambda\mathcal{L}_{D\text{-}SSIM}(I'^1, I'^2)$$

- 最终训练损失：$\mathcal{L} = \mathcal{L}_{color} + \lambda_p \mathcal{R}_{pcolor}$，其中 $\lambda_p{=}1.0$

### 损失函数 / 训练策略

**训练视角损失**（标准 3DGS）：

$$\mathcal{L}_{color} = (1-\lambda)\mathcal{L}_1(I^1, I^*) + \lambda\mathcal{L}_{D\text{-}SSIM}(I^1, I^*)$$

$\lambda{=}0.2$，与原始 3DGS 一致。

**总损失**：训练视角 GT 监督 + 伪视角协同正则化。

**初始化**：使用稀疏视角的立体融合点云（同 FSGS），而非 COLMAP 稀疏点。

**训练设置**：LLFF/DTU/Blender 训 10K 步，Mip-NeRF360 训 30K 步。

## 实验关键数据

### 主实验（LLFF 数据集，3/6/9 视角）

| 方法 | 3-view PSNR↑ | 3-view SSIM↑ | 3-view LPIPS↓ | 6-view PSNR↑ | 9-view PSNR↑ |
|---|---|---|---|---|---|
| FreeNeRF | 19.63 | 0.612 | 0.308 | 23.73 | 25.13 |
| 3DGS | 19.22 | 0.649 | 0.229 | 23.80 | 25.44 |
| FSGS | 20.43 | 0.682 | 0.248 | 24.09 | 25.31 |
| **CoR-GS** | **20.45** | **0.712** | **0.196** | **24.49** | **26.06** |

### DTU 数据集（3/6/9 视角）

| 方法 | 3-view PSNR↑ | 3-view SSIM↑ | 3-view LPIPS↓ |
|---|---|---|---|
| FreeNeRF | 19.92 | 0.787 | 0.182 |
| 3DGS | 17.65 | 0.816 | 0.146 |
| FSGS | - | - | - |
| **CoR-GS** | **19.21** | **0.853** | **0.119** |

### 效率比较（LLFF 3-view，RTX 3090 Ti）

| 方法 | 高斯数量 | FPS | PSNR↑ | 训练时间 |
|---|---|---|---|---|
| FreeNeRF | - | 0.09 | 19.63 | 2.3h |
| 3DGS | 1.16×10⁵ | 318 | 19.22 | 2.5min |
| **CoR-GS** | **7.85×10⁴** | **349** | **20.45** | 6min |

### 消融实验

| Co-Pruning | Pseudo-view Co-reg | LLFF PSNR↑ | LLFF SSIM↑ | LLFF LPIPS↓ | DTU PSNR↑ |
|---|---|---|---|---|---|
| ✗ | ✗ | 19.22 | 0.649 | 0.229 | 17.65 |
| ✓ | ✗ | 19.62 | 0.673 | 0.217 | 18.59 |
| ✗ | ✓ | 20.26 | 0.706 | 0.198 | 18.56 |
| **✓** | **✓** | **20.45** | **0.712** | **0.196** | **19.21** |

### 关键发现

1. 两种正则化互补：co-pruning 减少远离场景的异常高斯，pseudo-view co-reg 修正位置合理但渲染不准确的高斯
2. CoR-GS 将高斯数量减少 33%（11.6万→7.85万），推理反而更快（349 FPS vs 318 FPS）
3. 在 9 视角时 FSGS 相比 vanilla 3DGS 反而退步，因为深度先验引入了噪声；CoR-GS 在所有视角数下一致提升
4. 在 360° 场景（Mip-NeRF360）中同样有效，12 视角 PSNR 从 18.52（3DGS）提至 19.52

## 亮点与洞察

1. **全新正则化视角**：利用两个模型训练时的随机差异作为无监督质量指标，优雅且无需外部先验
2. **对"随机性"的深入分析**：密度控制的随机采样是稀疏视角 3DGS 错误几何的源头，这一观察非常有价值
3. **紧凑表示**：不仅提升质量，还减少了 1/3 的高斯数量，推理更快
4. **通用性强**：LLFF（前向场景）、Mip-NeRF360（360° 场景）、DTU（物体）、Blender（合成）全部有效
5. 类似机器学习中 co-training / mutual teaching 的思想，但首次应用于 3DGS

## 局限与展望

1. 训练两个辐射场将训练时间翻倍（2.5min → 6min），对实时应用有影响
2. 伪视角采样仅在两个最近训练视角之间插值，可能无法覆盖所有重要未见区域
3. Co-pruning 的距离阈值 $\tau$ 是固定的，可能不适用于所有场景尺度
4. 未与深度先验方法结合，两者可能互补

## 相关工作与启发

- **FSGS**：依赖外部深度先验，深度噪声会负面影响几何；CoR-GS 无需外部监督
- **Co-training 思想**（Blum & Mitchell 1998）：两个学习器互相纠正，本文将其引入 3DGS
- **半监督学习中的 prediction agreement**：用两个网络预测的一致性来伪标注或过滤噪声，本文类推到 3D 重建
- 启发：**模型训练的随机性本身可以是有用的信号**

## 评分

| 维度 | 分数 (1-10) |
|---|---|
| 创新性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 9 |
| 写作质量 | 8 |
| 实用价值 | 8 |
| **总分** | **8.0** |

<!-- RELATED:START -->

## 相关论文

- [Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis](thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)
- [GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)

<!-- RELATED:END -->
