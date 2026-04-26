---
title: >-
  [论文解读] Discretized Gaussian Representation for Tomographic Reconstruction
description: >-
  [3D视觉] 提出离散化高斯表示（DGR）用于 CT 重建，通过离散化高斯函数直接端到端重建 3D 体素，并设计高度并行化的快速体积重建技术，在稀疏视角和有限角度 CT 场景中以零训练数据超越深度学习和实例重建方法。
tags:
  - 3D视觉
---

# Discretized Gaussian Representation for Tomographic Reconstruction

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2411.04844](https://arxiv.org/abs/2411.04844)
- **代码**: [wskingdom/DGR](https://github.com/wskingdom/DGR)
- **领域**: 3D Vision / CT Reconstruction
- **关键词**: CT Reconstruction, 3D Gaussian, Discretized Representation, Fast Volume Reconstruction, Sparse-View CT, Limited-Angle CT

## 一句话总结

提出离散化高斯表示（DGR）用于 CT 重建，通过离散化高斯函数直接端到端重建 3D 体素，并设计高度并行化的快速体积重建技术，在稀疏视角和有限角度 CT 场景中以零训练数据超越深度学习和实例重建方法。

## 研究背景与动机

- **CT 重建挑战**：电离辐射限制投影数量、急诊需快速重建、需适配不同 CT 配置（锥束/扇束、稀疏视角/有限角度）
- **深度学习重建（DLR）的不足**：需大规模训练数据、泛化差（锥束训练模型在扇束上退化）、扩散模型计算开销大
- **实例重建方法的问题**：
    - NeRF 方法（NAF、SAX-NeRF）：需数小时重建，不适合临床部署
    - 3DGS 方法（R2-Gaussian、X-Gaussian）：存在**积分偏差**（density inconsistency），且 3DGS 的面向视角渲染与体素化重建之间存在根本不匹配
- **核心动机**：重新思考 CT 重建框架的三个设计原则——离散化表示、高效重建、统一全局优化

## 方法详解

### 整体框架

DGR 将 3D 体积表示为一组可学习的离散化高斯函数，通过快速体积重建将高斯贡献聚合到体素网格，然后在投影域全局优化。

### 1. 离散化高斯表示

**连续高斯**：

$$G(p, \mu, \Sigma) = e^{-\frac{1}{2}(p-\mu)^\top \Sigma^{-1}(p-\mu)}$$

使用**各向同性高斯**（CT 组织衰减具有各向同性特征），体素强度为所有高斯贡献之和：

$$V(p) = \sum_{i=1}^{n} G(p, \mu_i, \Sigma_i) \cdot I_i$$

**高斯效应限制**：将每个高斯影响范围限制在 $w_0 \times h_0 \times c_0$ 的局部长方体 $B_0$ 内，大幅降低计算量。

**离散化与对齐**：连续均值 $\mu$ 不能直接用于离散体素索引（不可微）。定义残差 $\Delta\mu = \mu - \lfloor\mu\rfloor$，将局部框坐标调整为：

$$\lfloor B \rfloor_{n,w_0,h_0,c_0,d} = B_{w_0,h_0,c_0,d} - \Delta\mu_{n,1,1,1,d}$$

保证梯度可通过聚合操作传播，维持整个流水线的可微性。

### 2. 快速体积重建（Fast Volume Reconstruction）

**并行计算**：利用 Einstein 求和约定并行计算 Mahalanobis 距离：

$$D^2_{n,w_0,h_0,c_0} = \sum_d \lfloor B \rfloor_{n,w_0,h_0,c_0,d} C^{-1}_{n,d,d} \lfloor B \rfloor_{n,w_0,h_0,c_0,d}$$

**分解加速**：将 $D^2$ 分解为四个更小的 Einstein 求和：

$$D^2 = (B^\top C^{-1} B) - (B^\top C^{-1} \Delta\mu) - (\Delta\mu^\top C^{-1} B) + (\Delta\mu^\top C^{-1} \Delta\mu)$$

其中 $B$ 仅为 $\lfloor B \rfloor$ 的 $\frac{1}{n}$ 大小，计算效率大幅提升。最终贡献：

$$\Gamma = e^{-\frac{1}{2}D^2} \cdot I$$

然后聚合到体素：$V_{x,y,z} \leftarrow V_{x,y,z} + \Gamma_{i,x,y,z}$

**即插即用**：FVR 模块可无缝集成到其他 3DGS 方法中（如 X-Gaussian）。

### 3. 全局优化

不同于 3DGS 的逐视角/逐块优化，DGR 对所有高斯联合优化，在 <1K 迭代内收敛（vs 3DGS 的 30K）。

投影变换：$\hat{P} = \mathcal{T}(V)$，损失函数：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_1(\hat{P}, P) + \lambda_2 \mathcal{L}_{SSIM}(\hat{P}, P) + \lambda_3 \mathcal{L}_{TV}(V)$$

$\lambda_1=0.6, \lambda_2=0.2, \lambda_3=1.0$

结合自适应密度控制（克隆/分裂/剪枝高斯），进一步提升重建质量。

## 实验关键数据

### 主实验：锥束稀疏视角 CT（FIPS 数据集，真实世界）

| 方法 | 75-view PSNR↑ | 75-view SSIM↑ | Time↓ | 50-view PSNR↑ | 25-view PSNR↑ |
|------|---------------|---------------|-------|---------------|---------------|
| NAF | 38.58 | 0.848 | 51m | 36.44 | 32.92 |
| SAX-NeRF | 34.93 | 0.854 | 13h | 34.89 | 33.49 |
| X-Gaussian* | 38.27 | 0.894 | 10m | 37.80 | 35.12 |
| R2-Gaussian (30k) | 39.40 | 0.875 | 14m | 38.24 | 34.83 |
| **DGR (300 iter)** | 39.91 | 0.937 | **3m36s** | 38.66 | 35.16 |
| **DGR (1000 iter)** | **41.28** | **0.952** | 13m | **39.27** | 34.58 |

- DGR 300 迭代（3 分钟）即超越 R2-Gaussian 30K 迭代（14 分钟）
- 75-view PSNR 高出 R2-Gaussian 1.88 dB

### 消融实验

**局部框大小消融（60-view AAPM-Mayo）**：

| Box-Size | Time (min) | V-RAM (GiB) | PSNR / SSIM |
|----------|-----------|-------------|-------------|
| 13×13×13 | 7.92 | 10.70 | 35.99 / 0.960 |
| 15×15×15 | 11.07 | 13.29 | 38.90 / 0.973 |
| **17×17×17** | **16.58** | **16.87** | **40.25 / 0.985** |
| 19×19×19 | 26.99 | 21.32 | 40.98 / 0.987 |

17×17×17 为最优平衡点。

**快速体积重建效果**：

| 方法 | VRAM (GiB) | Time/iter (s) |
|------|------------|---------------|
| Direct Reconstruction | 16662.50 (估) | / |
| FVR w/o Decomposition | 16.87 | 1.05 |
| **FVR w/ Decomposition** | **16.87** | **0.09** |

分解技术将重建时间从 1.05s 降至 0.09s（11.7×加速），显存不增。

### 扇束稀疏视角 CT（AAPM-Mayo LDCT）

| 方法 | 额外数据 | 180-view PSNR | 120-view PSNR | 60-view PSNR |
|------|----------|---------------|---------------|--------------|
| FBPConvNet | 4839 | 42.23 | 39.45 | 35.63 |
| SWORD | 4839 | 45.08 | 42.49 | 38.49 |
| **DGR** | **0** | **46.13** | **44.64** | **40.25** |

DGR 在零训练数据条件下超越所有需要大规模训练的 DLR 方法。

### 有限角度 CT（90° 重建）

| 方法 | Axial PSNR↑ | Coronal PSNR↑ | Sagittal PSNR↑ |
|------|-------------|---------------|----------------|
| DiffusionMBIR | 34.92 | 32.48 | 28.82 |
| **DGR** | **38.22** | **39.32** | **38.35** |

DGR 在三个方向上均大幅领先（+3.3/+6.8/+9.5 dB），且无需先验知识。

## 亮点与洞察

1. **表示设计精妙**：直接用离散化高斯建模体素网格，避免了 3DGS 的积分偏差和视角偏向问题
2. **零数据超越 DLR**：DGR 不需要任何训练数据，却在多个场景超越需要数千张训练图像的深度学习方法
3. **极致效率**：256³ 体积、15万+高斯，每迭代仅 0.09 秒；300 迭代约 3 分钟即可获得高质量重建
4. **强泛化性**：统一框架无需修改即适配锥束/扇束、稀疏视角/有限角度等多种 CT 配置
5. **FVR 即插即用**：可直接集成到 X-Gaussian 等现有方法，赋予其体积重建能力

## 局限性

- **各向同性假设**：仅使用各向同性高斯，对具有明显方向性结构的组织可能不够精确
- **局部框大小权衡**：更大框提升质量但显存和时间指数增长
- **临床验证不足**：未在真实临床场景验证诊断准确性
- **2D 切片扩展不充分**：主要展示 3D 体积重建，对 2D 切片级细节的分析有限

## 相关工作与启发

- R2-Gaussian 识别了 3DGS 的积分偏差并提出修正，但引入额外开销；DGR 从根本上避免了这一问题
- NAF、SAX-NeRF 等 NeRF 方法提供连续表示但推理慢，DGR 的离散化表示是关键加速因素
- SWORD 等扩散方法虽强但依赖大规模数据和高计算量
- 启发：离散化高斯的思路可推广到 MRI 重建、PET 成像等其他医学成像领域

## 评分 ⭐⭐⭐⭐⭐

方法设计从表示、重建到优化三个层面全面创新，理论推导严谨（Einstein 求和分解），实验极其充分（三个数据集、四种 CT 配置），且在零数据条件下全面超越需要大量训练数据的方法。FVR 的即插即用特性和 3 分钟重建的实用价值使其具有很高的临床应用潜力。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] LayerLock: Non-collapsing Representation Learning with Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)

<!-- RELATED:END -->
