---
title: >-
  [论文解读] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction
description: >-
  [AAAI 2026][3D视觉][CT重建] 提出 TG-Field，一种面向极端稀疏视角 CT 重建的几何感知高斯形变框架，通过多分辨率哈希编码器建模空间几何先验、时空注意力模块和运动流网络处理动态 CT，在静态和动态 CT 重建中均实现了 SOTA 性能。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "CT重建"
  - "3D高斯溅射"
  - "稀疏视角"
  - "动态CT"
  - "形变场"
---

# TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2602.11705](https://arxiv.org/abs/2602.11705)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: CT重建, 3D高斯溅射, 稀疏视角, 动态CT, 形变场

## 一句话总结

提出 TG-Field，一种面向极端稀疏视角 CT 重建的几何感知高斯形变框架，通过多分辨率哈希编码器建模空间几何先验、时空注意力模块和运动流网络处理动态 CT，在静态和动态 CT 重建中均实现了 SOTA 性能。

## 研究背景与动机

### 问题定义

锥束计算机断层扫描（CBCT）广泛用于医学、生物学和工业领域。高质量 3D 重建通常需要数百个 X 射线投影，但密集角度采样会带来辐射暴露风险。稀疏视角 CBCT 重建旨在减少投影数量的同时保持重建保真度。

### 已有方法的不足

**传统方法**：
- FDK（Feldkamp 滤波反投影）：在稀疏视角下质量急剧下降
- SART（迭代重建）：计算成本高，对超参数敏感

**NeRF 基方法**（NAF, SAX-NeRF 等）：
- 自监督，无需配对数据，但大多针对静态重建
- 需要密集射线采样，计算代价高
- STNF4D 尝试动态 CT 但收敛慢且重建质量不佳

**3DGS 基方法**的两大关键挑战：

**极端稀疏视角下的鲁棒性不足**：缺乏显式几何正则化，单高斯优化无法维持几何一致性，导致严重伪影

**动态 CT 重建困难**：无法建模呼吸运动等非刚体形变，时间一致性难以保证

### 核心动机

现有 3DGS CT 方法（如 R²-Gaussian、X-Gaussian）在中等稀疏条件下表现尚可，但极端稀疏时性能骤降。关键原因是：每个高斯原语独立优化，缺乏空间几何上下文的约束。作者提出引入**几何感知的形变场**，通过哈希编码器捕捉局部空间先验，约束高斯原语之间的空间相关性，从而在极端稀疏条件下也能保持结构连贯性。

## 方法详解

### 整体框架

TG-Field 的流程为：
1. 通过**迭代式初始化**生成高质量初始点云
2. 使用**多分辨率哈希编码器**捕捉空间几何特征
3. 通过**多头形变解码器**预测高斯原语的属性偏移
4. 对于动态 CT，加入**时空注意力模块**和**运动流网络**
5. 使用**语义一致性正则化**增强跨视角一致性

最终，形变后的高斯原语渲染为 X 射线投影并体素化为 CT 体积。

### 关键设计

#### 1. **迭代式初始化策略**：高质量几何先验的点云初始化

**功能**：两阶段迭代初始化——先用 CGLS（共轭梯度最小二乘法）获得粗略体积重建，再用 ASD-POCS（自适应最速下降-凸集投影）施加 TV 约束进行精化。

**核心思路**：不同于现有方法的均匀立方体采样（缺乏几何信息）或 FDK 初始化（稀疏条件下质量差），迭代式方法能从稀疏投影中提取更多几何信息：
- 第一阶段：CGLS 迭代逼近稀疏投影约束下的体积解
- 第二阶段：ASD-POCS 强制 TV 约束，减少噪声并保留结构边缘

**设计动机**：高质量初始化对 3DGS 收敛至关重要。在 5 视角这种极端稀疏条件下，均匀采样的点云几乎不包含有意义的结构信息，导致优化困难。

#### 2. **几何感知的溅射场**：用哈希编码器建模空间相关性

**功能**：使用多分辨率哈希网格编码器捕捉每个高斯原语的空间上下文，然后通过多头解码器预测属性偏移。

**核心思路**：对于位置 $\boldsymbol{\mu}_i$ 的高斯原语，通过哈希编码得到多尺度特征：

$$h_\phi(\boldsymbol{\mu}_i) = \text{concat}_{s \in S}[f_s(\boldsymbol{\mu}_i)] \in \mathbb{R}^{|S| \cdot C}$$

多头解码器分别预测位置、旋转、缩放和密度的偏移：

$$G'_i = (\boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i, R_i + \Delta R_i, S_i + \Delta S_i, \rho_i + \Delta\rho_i)$$

**设计动机**：哈希编码器天然地将空间上近邻的高斯原语映射到相似的特征空间，从而强制它们之间保持几何一致性。这在极端稀疏视角下尤为重要——当观测信息严重不足时，空间先验约束可以弥补缺失的几何信息。

#### 3. **时空注意力模块（STAB）**：解决 4D CT 中的哈希碰撞和时间漂移

**功能**：对联合编码的时空哈希特征施加注意力机制，消除时空歧义。

**核心思路**：对于每个高斯原语 $i$，堆叠时间窗口内的嵌入：

$$\mathbf{H}_i = [h_\phi(\boldsymbol{\mu}_i, t_1), \ldots, h_\phi(\boldsymbol{\mu}_i, t_T)]^\top$$

然后应用缩放点积注意力：

$$\text{Attn}(\mathbf{H}_i) = \text{softmax}\left(\frac{QK^\top}{\sqrt{C}}\right)V$$

**设计动机**：联合哈希空间和时间坐标会导致哈希碰撞——当相同或近似的空间位置在不同时间反复出现时，哈希桶产生歧义嵌入。STAB 通过聚合时间上下文，消除碰撞桶中的歧义，产生更稳定的动态形变。

#### 4. **运动流网络**：建模细粒度呼吸运动

**功能**：使用 ResFields MLP 预测精细位移场，在形变场输出的基础上进一步修正高斯中心位置。

$$\hat{\boldsymbol{\mu}}_i(t) = \boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t) + \text{Flow}(\boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t), t)$$

**设计动机**：初始形变场可能遗漏微妙的局部解剖形变（如肺部呼吸运动中的局部组织滑移），运动流网络作为残差修正模块捕捉这些细粒度运动。

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{TV}\mathcal{L}_{TV} + \lambda_{sem}\mathcal{L}_{sem}$

- L1 损失 + D-SSIM：监督渲染 X 射线投影
- 3D TV 正则化：同质性先验
- 语义一致性正则化 $\mathcal{L}_{sem}$：使用预训练 DINO-ViT 提取视觉特征，强制跨视角语义一致性

训练分两阶段：先预训练 R²-Gaussian 5000 迭代（warm-up），再加入形变场精化。

## 实验关键数据

### 主实验

**静态 CT 重建（合成 + 真实数据集）**：

| 方法 | 合成5视角 PSNR/SSIM | 合成10视角 PSNR/SSIM | 合成20视角 PSNR/SSIM | 真实10视角 PSNR/SSIM |
|------|-------------------|-------------------|-------------------|-------------------|
| FDK | 11.83/0.112 | 15.21/0.186 | 18.48/0.293 | 17.57/0.225 |
| SART | 22.10/0.683 | 24.32/0.768 | 27.24/0.845 | 28.72/0.846 |
| SAX-NeRF | 24.05/0.740 | 27.55/0.801 | 31.93/0.875 | 32.26/0.835 |
| R²-Gaussian | 23.81/0.735 | 28.15/0.833 | 32.25/0.923 | 32.73/0.859 |
| **Ours** | **24.54/0.779** | **28.95/0.849** | **32.92/0.936** | **33.59/0.872** |

**动态 CT 重建**：

| 方法 | XCAT PSNR/SSIM | TCIA PSNR/SSIM | SPARE PSNR/SSIM | 平均 PSNR/SSIM |
|------|---------------|---------------|----------------|---------------|
| Hex-plane | 21.79/0.866 | 23.91/0.835 | 26.43/0.856 | 24.04/0.852 |
| K-plane | 20.57/0.847 | 24.59/0.855 | 26.59/0.876 | 23.92/0.859 |
| STNF4D | 25.73/0.928 | 29.37/0.919 | 28.75/0.887 | 27.95/0.911 |
| 4DGS | 33.95/0.955 | 34.44/0.948 | 30.01/0.898 | 32.80/0.933 |
| **Ours** | **35.51/0.969** | **35.41/0.955** | **30.41/0.905** | **33.78/0.943** |

### 消融实验

| 设置 | 组件 | PSNR↑ | SSIM↑ | 说明 |
|------|------|-------|-------|------|
| 静态 | HE only | 28.71 | 0.841 | 仅哈希编码器 |
| 静态 | HE + SR | 28.95 | 0.849 | +语义正则化，提升0.24dB |
| 动态 | HE + STAB | 34.89 | 0.945 | +时空注意力 |
| 动态 | HE + STAB + MF | 35.23 | 0.952 | +运动流网络，提升0.34dB |
| 动态 | 全部 (HE+STAB+MF+SR) | **35.41** | **0.955** | 全组件 |

### 关键发现

1. **极端稀疏视角优势显著**：5 视角下比 R²-Gaussian 高 0.73dB（合成），0.65dB（真实），说明几何先验约束在信息极度匮乏时尤为关键
2. **动态 CT 全面领先**：平均 PSNR 比 4DGS 高 0.98dB，XCAT 上高 1.56dB
3. **初始化策略影响显著**：迭代式初始化在 2-8 视角设置下均优于 FDK 和均匀采样
4. **各组件贡献递增**：HE → +STAB → +MF → +SR 逐步提升性能
5. **运动流网络主要改善运动敏感区域**：如肺部呼吸运动导致的局部形变

## 亮点与洞察

1. **几何先验的关键作用**：通过哈希编码器将空间相关性注入到高斯优化中，是本文最核心的贡献。这解决了独立优化每个高斯原语时缺乏全局一致性的问题
2. **迭代式初始化标新立异**：巧妙结合经典迭代重建方法（CGLS + ASD-POCS）为 3DGS 提供高质量起点
3. **VFM 用于 CT 正则化**：使用预训练视觉基础模型（DINO-ViT）的语义特征进行跨视角一致性约束，是将自然图像领域的 foundation model 迁移到医学成像的有益尝试
4. **统一的静态/动态框架**：同一框架通过加入时间维度即可扩展到 4D CT

## 局限与展望

1. **计算开销未详细报告**：哈希编码器和注意力模块增加的训练/推理时间未量化对比
2. **语义正则化的合理性**：使用自然图像预训练的 DINO-ViT 对 X 射线图像的作用可能有限，领域差异可能削弱效果
3. **仅针对 CBCT**：未在平行束 CT 或其他成像模态中验证
4. **SPARE 数据集上优势较小**：PSNR 仅比 4DGS 高 0.4dB，说明在临床真实数据上的提升空间有限

## 相关工作与启发

- R²-Gaussian 提出的可微体素化为 3DGS 直接重建 CT 体积奠定了基础
- 4DGaussians 的形变场思路被本文继承，但加入了几何感知编码器使其更鲁棒
- 语义一致性正则化思路可推广到其他稀疏重建任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将几何先验引入 3DGS CT 重建的思路有价值，但各组件（哈希编码器、注意力、运动流）均非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 静态/动态 × 合成/真实 × 多视角设置的全面评估，消融完整
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，但公式密集
- **价值**: ⭐⭐⭐⭐ — 在医学成像中的潜在临床应用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](../../CVPR2025/3d_vision/geometry_field_splatting_with_gaussian_surfels.md)

</div>

<!-- RELATED:END -->
