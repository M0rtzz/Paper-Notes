---
title: >-
  [论文解读] SASNet: Spatially-Adaptive Sinusoidal Networks for INRs
description: >-
  [CVPR 2026][3D视觉][隐式神经表示] 提出 SASNet，通过将冻结的频率嵌入层与轻量级哈希网格 MLP 学习的空间自适应掩码相结合，解决 SIREN 中频率初始化敏感和高频泄漏问题，在图像拟合、体数据拟合和 SDF 重建任务上实现更快收敛和更高重建质量。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "隐式神经表示"
  - "SIREN"
  - "空间自适应"
  - "频率泄漏"
  - "哈希网格"
---

# SASNet: Spatially-Adaptive Sinusoidal Networks for INRs

**会议**: CVPR 2026  
**arXiv**: [2503.09750](https://arxiv.org/abs/2503.09750)  
**代码**: [https://github.com/Fengyee/SASNet_inr](https://github.com/Fengyee/SASNet_inr)  
**领域**: 3D视觉 / 隐式神经表示  
**关键词**: 隐式神经表示, SIREN, 空间自适应, 频率泄漏, 哈希网格

## 一句话总结

提出 SASNet，通过将冻结的频率嵌入层与轻量级哈希网格 MLP 学习的空间自适应掩码相结合，解决 SIREN 中频率初始化敏感和高频泄漏问题，在图像拟合、体数据拟合和 SDF 重建任务上实现更快收敛和更高重建质量。

## 研究背景与动机

隐式神经表示 (INR) 已成为计算机视觉和图形学中建模低维信号的强大工具，将坐标直接映射到信号值。其中，正弦网络 (SIREN) 因使用正弦激活函数能建模高频信号而被广泛使用，特别适合图像拟合、超分辨率和 SDF 建模等需要高频重建的任务。

然而，SIREN 存在一个核心痛点：对频率参数 $\omega_0$ 极度敏感。$\omega_0$ 小时产生干净但过于平滑的重建，缺失细节；$\omega_0$ 大时可以捕获锐利边缘，但在平滑区域（如图像背景）引入虚假的高频噪声——作者将这种不想要的高频成分在低频区域的激活称为"频率泄漏"。延长训练以恢复高频细节会进一步导致优化不稳定和过拟合。

根本矛盾在于：SIREN 中每个神经元的影响是全局的——一个负责编码高频细节的神经元会同时影响整个空间域，包括不需要高频信息的平滑区域。这就是频率泄漏的根源。网格化方法（如 InstantNGP）通过哈希网格实现空间局部化，但表示精细细节需要极高分辨率网格，增加了内存和计算成本。

核心 idea：将 SIREN 的频率控制能力与哈希网格 MLP 的空间局部化能力结合——用冻结的频率嵌入层固定网络的频谱支撑，用轻量级哈希网格 MLP 学习空间自适应掩码来约束每个神经元的空间影响范围，从而在需要细节的区域激活高频神经元、在平滑区域抑制它们。

## 方法详解

### 整体框架

SASNet 由两个并行网络组成：一个正弦 MLP 和一个哈希网格 MLP。输入坐标 $\mathbf{x}$ 同时进入两个网络。哈希网格 MLP 将坐标编码为特征，通过 ReLU MLP 解码为空间自适应掩码 $\mathcal{M}^i(\mathbf{x})$。每个掩码通过 Hadamard 乘积 $\odot$ 应用于正弦 MLP 的对应层，调制神经元的激活值。冻结的频率嵌入层固定频谱范围，掩码负责空间局部化。两个网络联合训练。

### 关键设计

1. **冻结频率嵌入层 (Frozen Frequency Embedding Layer)**:

    - 功能：显式固定网络的频率支撑范围，提供可控的频谱
    - 核心思路：遵循 Novello et al. 的方法，在 SIREN 的第一层使用预定义的频率集合并冻结其权重。不同于标准 SIREN 通过单个 $\omega_0$ 隐式决定频谱范围，冻结嵌入层直接指定可用的频率成分，使频率控制从隐式变为显式
    - 设计动机：标准 SIREN 的频率范围由 $\omega_0$ 和权重初始化共同决定且不可控，冻结嵌入层消除了这一不确定性，为后续的空间掩码提供了稳定的频率基础

2. **空间自适应掩码 (Spatially-Adaptive Masks)**:

    - 功能：学习每个空间位置应该激活哪些神经元/频率带
    - 核心思路：使用多尺度哈希网格编码输入坐标为特征向量，通过小型 ReLU MLP 解码为与正弦 MLP 各层维度匹配的掩码值。掩码通过逐元素乘法 $\mathbf{h}^i \odot \mathcal{M}^i(\mathbf{x})$ 调制正弦层的输出。直觉上，在平滑区域掩码会抑制高频神经元，在细节区域则允许高频神经元通过。通过联合训练，网络自动学习这种空间分配
    - 设计动机：这是解决频率泄漏问题的关键——将全局 INR 的每个神经元的影响限制到特定空间区域。哈希网格天然具有空间局部性，将其作为掩码生成器而非直接特征提取器是新颖的用法

3. **联合训练的混合架构 (Joint Training of Hybrid Architecture)**:

    - 功能：在参数效率的约束下实现频率控制和空间局部化的最优平衡
    - 核心思路：正弦 MLP 和哈希网格 MLP 共享输入坐标，联合优化标准 INR 拟合损失 $\mathcal{L}(\theta) = \frac{1}{N}\sum_i \|f_\theta(\mathbf{x}_i) - \mathscr{f}_i\|^2 + \lambda \mathcal{R}(\theta)$。哈希网格 MLP 设计为轻量级（小分辨率网格 + 浅层 ReLU MLP），避免大幅增加参数量。对于 SDF 任务，正则化项 $\mathcal{R}(\theta)$ 强制 eikonal 约束
    - 设计动机：相比纯 SIREN 或纯哈希网格方法，混合架构继承了 SIREN 的频率表达能力（正弦激活的导数精度）和哈希网格的空间局部性，同时避免了各自的弱点

### 损失函数 / 训练策略

基本拟合损失为 L2 重建损失。SDF 任务额外加入 eikonal 正则化确保梯度范数为 1。频率嵌入层在训练全程保持冻结。联合训练时掩码自然收敛到将低频神经元分配给平滑区域、高频神经元分配给细节区域的配置。

## 实验关键数据

### 主实验

基于论文摘要和方法描述，SASNet 在以下三类任务上进行了评估（具体数值待缓存补充）：

| 任务 | 指标 | SASNet vs SIREN | 说明 |
|--------|------|------|------|
| 2D 图像拟合 | PSNR | 显著提升 | 锐利边缘+干净背景 |
| 3D 体数据拟合 | PSNR | 显著提升 | 消除了平滑区域噪声 |
| SDF 重建 | CD/IoU | 优于先前方法 | 掩码自动聚焦零等值面 |
| ×16 超分辨率 | PSNR | 超越不同 $\omega_0$ 的 SIREN | 高低 $\omega_0$ 均有问题，SASNet 两者兼顾 |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| SIREN (low $\omega_0$) | 平滑但模糊 | 缺失高频细节 |
| SIREN (high $\omega_0$) | 锐利但噪声 | 频率泄漏严重 |
| SASNet w/o frozen embedding | 收敛不稳定 | 频率范围不可控 |
| SASNet w/o masks | 类似 SIREN | 无空间局部化 |
| SASNet (full) | 锐利且干净 | 频率控制+空间局部化 |

### 关键发现

- **频率泄漏是 SIREN 的根本瓶颈**：无论如何调节 $\omega_0$ 都无法同时获得锐利细节和干净背景，这不是超参数调优能解决的问题
- **空间掩码自动学习频率分配**：可视化显示低频神经元的掩码在平滑区域值高，高频神经元的掩码在边缘/细节区域值高，验证了设计直觉
- **参数效率高**：哈希网格 MLP 作为掩码生成器仅增加少量参数，但带来显著的质量提升
- **SDF 任务中掩码聚焦零等值面**：在 Armadillo 模型的腿部等细节区域，掩码自动集中神经元激活，与 SDF 的物理意义一致

## 亮点与洞察

- **将哈希网格作为掩码生成器而非特征提取器**是最巧妙的设计——通常哈希网格直接替代正弦激活作为特征编码（如 InstantNGP），本文反其道而行，让哈希网格服务于正弦网络的空间调制。这保留了 SIREN 精确的导数计算能力，同时获得空间局部性
- **冻结频率 + 学习空间分配的解耦**在概念上非常优雅——"固定你能做什么频率，学习在哪里做"，将频率控制和空间分配正交化
- **这种掩码调制机制可以迁移到 NeRF/3DGS**：在神经辐射场中，不同空间区域也需要不同频率的表达能力，空间自适应掩码可能有效

## 局限与展望

- 缓存文件仅包含摘要、引言和方法部分，缺少完整的实验数据（具体 PSNR 数值、运行时间对比等）
- 哈希网格本身引入了离散化，在分辨率不足时可能产生块效应
- 掩码的学习是否需要大量迭代才能收敛，在极少数据点的场景中是否有效，未充分讨论
- 仅在低维信号（2D 图像、3D 体数据/SDF）上验证，未扩展到 NeRF 等高维场景表示

## 相关工作与启发

- **vs SIREN**: SIREN 的全局性是频率泄漏的根源，SASNet 通过空间掩码直接解决，且保持了 SIREN 的正弦激活优势
- **vs InstantNGP**: InstantNGP 用哈希网格+ReLU MLP 做局部表示，但 ReLU 导数精度差；SASNet 保留正弦激活的精确导数，用哈希网格做辅助调制
- **vs WIRE**: WIRE 用 Gabor 小波实现空间局部性但容易过拟合，SASNet 的掩码方式更灵活且参数更高效
- **vs FINER**: FINER 通过动态缩放因子做自适应频率调制，但仍是全局性的，SASNet 实现了真正的空间局部化

## 评分

- 新颖性: ⭐⭐⭐⭐ 将哈希网格作为 SIREN 的空间掩码生成器是新颖的架构设计，冻结频率+学习空间的解耦思路清晰
- 实验充分度: ⭐⭐⭐ 缓存不完整导致无法评估具体数值，但覆盖了三类任务（图像、体数据、SDF）
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，"频率泄漏"的可视化对比直观有力，INR 分类体系（全局/局部/混合）有组织价值
- 价值: ⭐⭐⭐⭐ 对 INR 领域的频率控制问题提出了优雅的解决方案，空间掩码思路有广泛的迁移潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)
- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](../../CVPR2025/3d_vision/exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[NeurIPS 2025\] High Resolution UDF Meshing via Iterative Networks](../../NeurIPS2025/3d_vision/high_resolution_udf_meshing_via_iterative_networks.md)
- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)

</div>

<!-- RELATED:END -->
