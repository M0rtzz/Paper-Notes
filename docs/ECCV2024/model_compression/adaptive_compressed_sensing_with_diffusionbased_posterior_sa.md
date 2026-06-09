---
title: >-
  [论文解读] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling
description: >-
  [ECCV 2024][模型压缩][自适应压缩感知] 本文提出 AdaSense，利用预训练扩散模型的零样本后验采样能力来量化重建不确定性，从而自适应地选择最优测量矩阵，在人脸图像、MRI 和 CT 等多个领域实现了无需额外训练的自适应压缩感知，性能超越非自适应方法甚至基于 PCA 的最优非自适应方案。
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "自适应压缩感知"
  - "扩散模型"
  - "后验采样"
  - "医学成像"
  - "主动测量获取"
---

# Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling

**会议**: ECCV 2024  
**arXiv**: [2407.08256](https://arxiv.org/abs/2407.08256)  
**代码**: [https://github.com/noamelata/AdaSense](https://github.com/noamelata/AdaSense)  
**领域**: Model Compression / Compressed Sensing  
**关键词**: 自适应压缩感知, 扩散模型, 后验采样, 医学成像, 主动测量获取

## 一句话总结
本文提出 AdaSense，利用预训练扩散模型的零样本后验采样能力来量化重建不确定性，从而自适应地选择最优测量矩阵，在人脸图像、MRI 和 CT 等多个领域实现了无需额外训练的自适应压缩感知，性能超越非自适应方法甚至基于 PCA 的最优非自适应方案。

## 研究背景与动机
压缩感知（CS）通过选取少量关键测量值实现快速图像采集，是医学成像等测量代价高昂场景的重要技术。自适应 CS 更进一步，根据已有测量信息动态选择后续测量方向，理论上可以显著减少所需测量总数。然而，现有自适应 CS 方法存在几个核心痛点：(1) 大多数方法需要针对特定任务和退化模式进行复杂的训练流程，泛化性差；(2) 许多方法受限于特定的退化类型（如只能处理像素子采样）；(3) 在医学影像等数据敏感领域，往往难以获得充足的训练数据。本文的核心矛盾在于：如何在不进行额外训练的前提下，实现自适应的测量选择？切入角度是利用预训练扩散模型的零样本后验采样能力来估计重建的不确定性，进而贪心地选择最能减少不确定性的测量方向。核心 idea：通过后验采样近似后验协方差矩阵的主成分，将其作为下一步测量方向。

## 方法详解

### 整体框架
AdaSense 是一个迭代式的自适应测量获取框架。在每一步中：(1) 利用扩散模型的零样本后验采样器生成 $s$ 个后验样本；(2) 对这些样本进行 PCA，提取后验协方差矩阵的前 $r$ 个主成分作为新的测量方向；(3) 用新的测量矩阵获取真实测量值并追加到已有测量中。重复 $N$ 步后，最终用所有测量值进行图像重建。

### 关键设计
1. **基于后验协方差的测量选择**:
    - 功能：在每一步自适应选择最优的测量方向
    - 核心思路：将测量选择问题转化为 PCA 问题——在给定已有测量 $\mathbf{y}_{0:nr}$ 的条件下，最优的新测量方向是后验协方差矩阵 $\mathrm{Cov}[\mathbf{x}|\mathbf{y}_{0:nr}]$ 的前 $r$ 个特征向量，因为它们对应最大不确定性方向
    - 设计动机：线性最小 MSE 重建器的误差等于后验协方差沿未测量方向的投影，选择最大方差方向可以最大化信息增益

2. **约束测量场景下的简化优化**:
    - 功能：处理 MRI（k 空间行）和 CT（投影角度）等物理约束下的测量选择
    - 核心思路：当测量矩阵受限于可行集 $\mathcal{H}$ 时，将重建矩阵固定为伪逆 $\tilde{\bm{H}}^{\dagger}$，将目标函数简化为 $\argmax_{\tilde{\bm{H}} \in \mathcal{H}} \mathbb{E}[(\mathbf{x} - \mathbb{E}[\mathbf{x}|\mathbf{y}])^{\top} \tilde{\bm{H}}^{\dagger} \tilde{\bm{H}} (\mathbf{x} - \mathbb{E}[\mathbf{x}|\mathbf{y}]) | \mathbf{y}]$，可通过后验样本近似并穷举搜索
    - 设计动机：直接优化完整目标函数需要协方差矩阵秩至少为 $r$，在 MRI/CT 中 $r$ 通常很大，生成足够多的后验样本不切实际

3. **后验采样加速**:
    - 功能：避免每步重复计算 SVD，提升效率
    - 核心思路：利用一致性后验采样器的性质——已选测量方向上方差为零，因此新测量方向必然与旧方向正交。这使得测量矩阵的 SVD 分解平凡化为 $(U, \Sigma, V^{\top}) = (I, I, \bm{H})$，无需额外计算
    - 设计动机：在运行时动态构建测量矩阵导致需要频繁计算 SVD，这是主要的计算瓶颈

### 损失函数 / 训练策略
AdaSense 本身不需要训练——它是一个推理时的测量选择算法。核心依赖于预训练扩散模型和零样本后验采样器（如 DDRM）。最终重建可以使用：(1) 单次后验采样、(2) 多次后验采样取平均（近似 MMSE 估计）、(3) 针对特定模态的专用重建网络。

## 实验关键数据

### 主实验
**人脸图像重建**（CelebA-HQ 256，192 维测量）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DeepFace↓ |
|------|-------|-------|--------|-----------|
| Block Downsampling | 20.50 | 0.6128 | 0.3035 | 0.6134 |
| Bicubic Downsampling | 20.86 | 0.6193 | 0.2964 | 0.5865 |
| PCA (非自适应最优) | 24.60 | 0.7190 | 0.2307 | 0.3010 |
| **AdaSense** | **26.20** | **0.7515** | **0.1950** | **0.2674** |

**MRI 主动子采样**（FastMRI 单线圈膝盖，垂直子采样）：

| 方法 | PSNR(R10)↑ | SSIM(R10)↑ |
|------|-----------|-----------|
| Random R10 | 24.56 | 0.4786 |
| Equi-spaced R10 | 24.73 | 0.4767 |
| **AdaSense R10** | **27.01** | **0.5229** |

### 消融实验

| 配置 (自适应步数 N) | PSNR↑ | LPIPS↓ | 说明 |
|------|---------|------|------|
| N=1 (非自适应) | ~24.5 | ~0.24 | 无自适应，等价于一次性 PCA |
| N=4 | ~25.5 | ~0.21 | 中等自适应 |
| N=8 | ~26.2 | ~0.20 | 高度自适应，性能最佳 |

与专用 MRI 主动获取方法对比（R8-30L）：

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Low-to-High (非自适应) | 28.32 | 0.5948 |
| SS-DDQN (RL 训练) | 28.99 | 0.6129 |
| Greedy Oracle (上界) | 29.22 | 0.6264 |
| **AdaSense** | 28.89 | 0.6108 |

### 关键发现
- AdaSense 在无需任何额外训练的情况下，人脸图像重建中超越了使用真实训练数据的非自适应 PCA 方法 1.6 dB PSNR
- 自适应性越强（N 越大），重建质量单调提升，验证了自适应测量选择的有效性
- 在 MRI 任务中，AdaSense 与需要复杂强化学习训练的专用方法（SS-DDQN）性能相当，且接近理论上界（Greedy Oracle）
- CT 稀疏视角重建中也展现了潜力，证明了框架的跨模态通用性

## 亮点与洞察
- **零训练的通用性**：只需替换领域扩散模型即可迁移到新模态，无需针对退化模式重新训练，这在医学影像等数据稀缺领域尤为重要
- **理论优雅**：将自适应 CS 问题优雅地归结为条件后验的 PCA 问题，数学推导清晰
- **实践简洁**：算法流程简单（采样→PCA→测量→重复），易于理解和实现
- 超越 PCA 非自适应最优方案的事实表明，适应性（根据已有测量调整后续策略）确实带来了实质性收益

## 局限与展望
- 计算开销大：每步需要生成多个后验样本，扩散采样本身就很耗时
- 性能上限受后验采样器质量制约，采样器的不准确性会传播到测量选择中
- 目前仅限于线性测量，非线性测量场景（如相位检索）尚未探索
- 贪心策略可能导致次优解，未考虑当前选择对未来步骤的影响
- 在临床应用前需要更多安全验证

## 相关工作与启发
- 与基于 RL 的 MRI 主动获取方法（SS-DDQN, DS-DDQN）相比，AdaSense 用零样本灵活性换取了部分性能
- 与盲逆问题（blind inverse problem）方法有相似之处，都利用扩散模型适应退化矩阵，但目标不同——本文是选择最优测量而非恢复未知退化
- 启发：随着扩散后验采样器的快速发展（更快、更准确），AdaSense 的实用性有望大幅提升
- 可延伸到 burst photography、三维重建中的 best-next-view 选择等主动感知任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 将扩散后验采样与自适应 CS 优雅结合，但 PCA 选择测量方向的思路不算全新
- 实验充分度: ⭐⭐⭐⭐ 涵盖人脸、MRI、CT 三个领域，自适应性消融实验设计巧妙，但 MRI/CT 实验规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、数学推导严谨、图表直观，论文结构非常好
- 价值: ⭐⭐⭐⭐ 提供了一个统一的零训练自适应 CS 框架，对医学成像有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[ICML 2025\] Diffusion Sampling Correction via Approximately 10 Parameters](../../ICML2025/model_compression/diffusion_sampling_correction_via_approximately_10_parameters.md)
- [\[ICML 2026\] Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation](../../ICML2026/model_compression/global_convergence_of_adaptive_sensing_for_principal_eigenvector_estimation.md)

</div>

<!-- RELATED:END -->
