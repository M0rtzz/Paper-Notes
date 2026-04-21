---
title: >-
  [论文解读] Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists
description: >-
  [CVPR2026][3D视觉][3D Gaussian Splatting] 通过定期重置高斯尺度（Scale Reset）和对 alpha blending 权重施加熵约束（Entropy Constraint），缩短每个像素的高斯列表长度，实现 3DGS 训练 **5-12 倍加速**，同时保持可比的渲染质量。
tags:
  - CVPR2026
  - 3D视觉
  - 3D Gaussian Splatting
  - 训练加速
  - Scale Reset
  - 熵约束
  - 高斯列表缩短
---

# Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists

**会议**: CVPR2026  
**arXiv**: [2603.09277](https://arxiv.org/abs/2603.09277)  
**代码**: [MachinePerceptionLab/ShorterSplatting](https://github.com/MachinePerceptionLab/ShorterSplatting)  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, 训练加速, Scale Reset, 熵约束, 高斯列表缩短

## 一句话总结

通过定期重置高斯尺度（Scale Reset）和对 alpha blending 权重施加熵约束（Entropy Constraint），缩短每个像素的高斯列表长度，实现 3DGS 训练 **5-12 倍加速**，同时保持可比的渲染质量。

## 背景与动机

3D Gaussian Splatting（3DGS）相较 NeRF 在渲染效率和质量上有显著优势，但其训练过程仍然较慢，限制了时间敏感应用。现有加速方法主要从以下角度入手：

- **减少训练迭代**：通过更好的初始化或密集化策略将迭代次数从 30K 降至 5K-8K
- **优化器与 CUDA 实现**：用二阶优化器加速收敛、优化 CUDA kernel
- **训练策略**：冻结已收敛高斯、剪枝冗余高斯、渐进分辨率训练

然而这些方法要么依赖减少高斯总数（不适合大规模复杂场景），要么仅提供边际加速（如更精确的覆盖估计仅加速约 10%）。本文提出了一个全新视角：**缩短每个像素的高斯列表长度**，从而减少每次 splatting 涉及的高斯数量，直接降低前向渲染和反向梯度计算的成本。

## 方法详解

### 整体框架

本文的核心思想是让每个高斯将其影响集中在局部图像区域，而非分散到大量像素上。具体包含三个模块：

1. **Scale Reset**：定期缩小所有高斯的尺度，减少单个高斯的像素覆盖范围
2. **Entropy Constraint**：对 alpha blending 权重施加熵正则化，锐化权重分布
3. **Resolution Scheduler**：渐进式分辨率调度，从低分辨率逐步训练到全分辨率

### 关键设计一：Scale Reset（尺度重置）

较大的高斯覆盖更多像素，导致更长的高斯列表。直觉上应鼓励更小的尺度，但简单地添加体积惩罚项到损失函数中存在调参困难——权重过大会导致高斯过小，过小则无效。

Scale Reset 采用更直接的策略：每隔固定周期，将所有高斯的尺度乘以缩放因子 $\zeta < 1$：

$$s_i \leftarrow \zeta \cdot s_i, \quad \forall i$$

默认参数为每 20 个 epoch 执行一次，$\zeta = 0.2$（即缩小到原来的 20%）。

**为什么 Scale Reset 优于体积正则化？**

- **即时生效**：Scale Reset 在执行瞬间就缩短了所有像素的高斯列表，后续所有迭代立即受益；而体积正则化需要通过梯度逐步优化，生效缓慢
- **自适应恢复**：重置后高斯有足够的迭代时间调整其他属性（颜色、不透明度、位置等）以逼近辐射场，保证渲染质量
- **副效应有益**：Scale Reset 会鼓励更高的不透明度，使高斯更加紧凑有效

### 关键设计二：Entropy Constraint（熵约束）

在 alpha blending 中，像素 $p$ 处第 $i$ 个高斯的权重为：

$$w_i = T_i \cdot \alpha_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

对于长度为 $N$ 的高斯列表，这些权重加上背景贡献构成有效概率分布：

$$\sum_{i=1}^{N+1} w_i = 1, \quad w_{N+1} = T_{N+1}$$

这个归一化性质是关键——它使得熵计算无需额外的归一化步骤，避免了维护全局统计信息并在反向传播中访问它们的开销。

像素 $j$ 处的熵定义为：

$$H_j = -\sum_{i=1}^{N+1} w_{i,j} \log w_{i,j}$$

熵损失为所有 $M$ 个像素的平均：

$$\mathcal{L}_E = \frac{1}{M} \sum_{j=1}^{M} H_j$$

**熵最小化的效果**：驱动主导权重变大、次要权重变小，使每个高斯更集中于其主导像素区域，减少对邻近像素的影响，从而缩短高斯列表。

### 梯度计算

总损失对高斯属性 $x_i$ 的梯度为：

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}_{\text{base}}}{\partial x_i} + \frac{\gamma}{M} \sum_{j=1}^{M} \frac{\partial H_j}{\partial \alpha_{i,j}} \frac{\partial \alpha_{i,j}}{\partial x_i}$$

其中熵对 $\alpha_{i,j}$ 的梯度通过引入中间变量 $R_{i,j}$ 高效计算：

$$R_{i,j} = \sum_{k=i}^{N+1} (\log w_{k,j} + 1) w_{k,j}$$

$$\frac{\partial H_j}{\partial \alpha_{i,j}} = (-\log w_{i,j} - 1) T_{i,j} + \frac{R_{i+1,j}}{1-\alpha_{i,j}}$$

论文设计了 $O(N)$ 复杂度的后向扫描算法：从 $i=N$ 到 $1$ 逐步累积 $R_j$，避免重复计算。

### 损失函数

总体损失函数结合基础重建损失和熵正则化：

$$\mathcal{L} = \mathcal{L}_{\text{base}} + \gamma \mathcal{L}_E$$

$$\mathcal{L}_{\text{base}} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

其中 $\gamma = 0.015$ 为熵损失权重，$\lambda$ 为 D-SSIM 权重（沿用 3DGS 默认值）。

### Resolution Scheduler（渐进分辨率调度）

采用 DashGaussian 的粗到细分辨率策略，从低分辨率（下采样因子 $r > 1$）逐步训练到全分辨率（$r=1$）。但过大的 $r$ 反而会降低效率——低分辨率下每个 tile 覆盖的场景区域更大，导致过多高斯重叠。经验阈值为每个 tile 的高斯数不超过 150，$r_{\max}$ 上限为 4。

在与分辨率调度结合时，采用阶段自适应策略：早期低分辨率阶段使用较弱的正则化以保留场景结构，后期全分辨率阶段增大 $\zeta$ 和 $\gamma$。

## 实验

### 主实验结果

在 Mip-NeRF 360、Deep Blending 和 Tanks & Temples 三个基准上评估，GPU 为 RTX 5090 D。

| 方法 | 迭代 | 高斯数 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练时间(s)↓ |
|------|------|--------|-------|-------|--------|-------------|
| 3DGS | 30K | 3.3M | 27.55 | 0.819 | 0.209 | 919.51 |
| AdR-Gaussian | 30K | 1.3M | 26.92 | 0.792 | 0.257 | 504.86 |
| Taming-3DGS | 30K | 3.3M | 27.85 | 0.823 | 0.208 | 402.54 |
| EDGS | 5K | 3.5M | 26.46 | 0.817 | 0.205 | 318.06 |
| Mini-Splatting2 | 18K | 3.6M | 27.56 | 0.827 | 0.184 | 220.22 |
| DashGaussian | 30K | 3.3M | 27.84 | 0.824 | 0.203 | 218.85 |
| LiteGS | 30K | 3.3M | 27.75 | 0.822 | 0.208 | 191.17 |
| **本文** | **30K** | **3.3M** | **27.28** | **0.810** | **0.224** | **99.58** |

*以上为 Mip-NeRF 360 数据集结果*

**加速倍数**：相较 3DGS 实现 **9.2×**（Mip-NeRF 360）、**11.9×**（Deep Blending）、**5.3×**（Tanks & Temples）加速；相较 LiteGS 基线近 **50%** 加速。

### 消融实验

| 模块组合 | PSNR↑ | SSIM↑ | LPIPS↓ | 时间(s)↓ |
|---------|-------|-------|--------|---------|
| L (LiteGS) | 27.75 | 0.822 | 0.208 | 191.17 |
| L+R (Scale Reset) | 27.33 | 0.815 | 0.212 | 147.33 |
| L+E (Entropy) | 27.35 | 0.815 | 0.218 | 162.53 |
| L+R+E | 27.14 | 0.812 | 0.215 | 141.28 |
| L+D (DashGaussian) | 27.85 | 0.822 | 0.213 | 134.99 |
| L+D+R | 27.52 | 0.815 | 0.219 | 108.62 |
| L+D+E | 27.38 | 0.813 | 0.223 | 112.13 |
| L+D+R+E (完整方法) | 27.28 | 0.810 | 0.224 | 99.58 |

### 关键发现

1. **Scale Reset vs 体积正则化**：Scale Reset 在质量和速度上均优于体积正则化（PSNR 27.28 vs 27.17，时间 99.58s vs 107.91s）
2. **Entropy vs Opacity 约束**：熵约束优于 opacity 正则化，因为熵作用于依赖多个属性（不透明度、尺度、位置、旋转）的混合权重，而 opacity 约束仅限制单一属性过于保守
3. **两模块互补**：Scale Reset 提供即时几何正则化，Entropy Constraint 在优化过程中持续调整贡献分布，联合使用效果最佳
4. **Tile 大小无关**：3DGS 使用 16×16 tile vs LiteGS 使用 8×8 tile，质量和训练时间相近，因为更小的 tile 产生更短的列表但更多的 tile
5. **质量折中适度**：Mip-NeRF 360 上 PSNR 损失仅 0.27dB（27.28 vs 27.55），但训练时间从 919s 降至 100s

## 亮点

- **新颖视角**：不通过减少高斯总数来加速，而是缩短每个像素的高斯列表长度，适用于大规模复杂场景
- **方法简洁高效**：Scale Reset 仅需一行代码实现（逐元素乘法），Entropy Constraint 有 $O(N)$ 的高效梯度算法
- **无需数据先验**：不依赖预训练模型或几何基础模型，纯训练策略层面的改进
- **与现有方法正交**：可与 LiteGS、DashGaussian 等 CUDA 优化和分辨率调度方法叠加使用

## 局限与展望

- **质量有一定损失**：PSNR 下降约 0.3-0.5dB，在对质量要求极高的场景中可能不可接受
- **参数敏感性**：$\zeta$ 和 $\gamma$ 的选择需要在速度和质量间权衡，不同场景可能需要不同配置
- **依赖 LiteGS 骨干**：在受限设置（少迭代/少高斯）下性能下降源于 LiteGS 骨干的局限
- **未探索与高阶优化器的结合**：如 Levenberg-Marquardt 等二阶优化器可能与本文方法进一步互补
- **仅评估静态场景**：未验证在动态场景重建等任务上的适用性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从高斯列表长度角度加速训练是新颖视角，Scale Reset 虽简单但有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个标准数据集、详尽消融、与多种替代方案对比、timing breakdown 分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，公式推导完整
- 价值: ⭐⭐⭐⭐ — 实现 9× 加速具有实际意义，方法简洁易于集成到现有 3DGS 流程

<!-- RELATED:START -->

## 相关论文

- [FastGS: Training 3D Gaussian Splatting in 100 Seconds](fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data](../../CVPR2025/3d_vision/megasynth_scaling_up_3d_scene_reconstruction_with_synthesized_data.md)
- [MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [PhyGaP: Physically-Grounded Gaussians with Polarization Cues](phygap_physically-grounded_gaussians_with_polarization_cues.md)
- [AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](../../ICCV2025/3d_vision/aaagaussians_antialiased_and_artifactfree_3d_gaussian_render.md)

<!-- RELATED:END -->
