---
title: >-
  [论文解读] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing
description: >-
  [ECCV 2024][模型压缩][傅里叶压缩感知] 本文提出"自适应选择采样-重建对"($\mathcal{H}_{1.5}$)框架，利用超分辨率空间生成模型量化高频贝叶斯不确定性，为每个输入数据选择最佳的采样掩码-重建网络对，在理论和实验上同时优于非自适应联合优化方法（$\mathcal{H}_1$）和自适应采样方法（$\mathcal{H}_2$），在人脸图像和多线圈 MRI 重建中取得显著 SSIM 提升。
tags:
  - ECCV 2024
  - 模型压缩
  - 傅里叶压缩感知
  - 自适应采样
  - 采样-重建选择
  - 贝叶斯不确定性
  - MRI重建
---

# Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing

**会议**: ECCV 2024  
**arXiv**: [2409.11738](https://arxiv.org/abs/2409.11738)  
**代码**: https://smhongok.github.io/ada-sel.html (有项目页)  
**领域**: Model Compression / Compressed Sensing  
**关键词**: 傅里叶压缩感知, 自适应采样, 采样-重建选择, 贝叶斯不确定性, MRI重建

## 一句话总结
本文提出"自适应选择采样-重建对"($\mathcal{H}_{1.5}$)框架，利用超分辨率空间生成模型量化高频贝叶斯不确定性，为每个输入数据选择最佳的采样掩码-重建网络对，在理论和实验上同时优于非自适应联合优化方法（$\mathcal{H}_1$）和自适应采样方法（$\mathcal{H}_2$），在人脸图像和多线圈 MRI 重建中取得显著 SSIM 提升。

## 研究背景与动机
傅里叶压缩感知（Fourier CS）在 MRI、雷达等电磁成像中具有重要应用，核心挑战在于如何选择最优的采样模式以最少的测量实现最佳重建。现有方法分为两类：(1) 联合优化采样-重建（$\mathcal{H}_1$）——为全数据集优化一个固定的采样掩码和重建网络，但不能自适应到每个样本，且需要在离散空间进行困难的反向传播优化；(2) 自适应采样（$\mathcal{H}_2$）——为每个输入生成最优掩码，但面临掩码生成器优化困难和单一重建网络的 Pareto 次优问题（一个网络不可能同时对所有自适应生成的不同掩码都最优）。核心矛盾是：如何既实现数据自适应性，又确保重建网络的 Pareto 最优性？切入角度是不生成无限多的自适应掩码，而是预定义有限组（$J$ 个）采样-重建对，为每个输入选择最匹配的一对。核心 idea：用超分辨率生成模型量化输入的高频不确定性分布，通过聚类将不确定性模式分为 $J$ 类，每类对应一个专用的掩码-重建器。

## 方法详解

### 整体框架
$\mathcal{H}_{1.5}$ 框架分为训练和推理两阶段。训练阶段：(1) 对所有训练数据用超分辨率生成模型采样，计算归一化的高频方差（不确定性）；(2) 对不确定性向量做 k-means++ 聚类得到 $J$ 个质心；(3) 根据每个质心使用拒绝采样生成对应的采样掩码 $M_j$；(4) 为每个 $M_j$ 独立训练专用重建网络 $\theta_j$。推理阶段：(1) 先用低频掩码 $M_0$ 采集输入的低频分量；(2) 用超分辨率模型量化高频不确定性；(3) 找到最近的质心，选择对应的 $(M_{j^*}, \theta_{j^*})$ 对；(4) 额外采集 $M_{j^*}$ 指定的频率分量，用 $\theta_{j^*}$ 重建。

### 关键设计
1. **高频贝叶斯不确定性量化（掩码选择器 $e_\psi$）**:
    - 功能：为每个输入量化其高频部分的不确定性分布，用于自适应选择采样模式
    - 核心思路：利用条件归一化流（conditional normalizing flow）超分辨率模型从低频重建的图像生成多个高分辨率样本 $\{f_\psi^{-1}(z_s; M_0 k)\}$，计算 k 空间中的样本方差 $v(M_0 k)$，归一化后 $u = v / \|v\|_2$ 作为不确定性方向描述符
    - 设计动机：样本方差是 MSE 的无偏估计，在低频已知的条件下高频方差直接反映了重建误差的空间分布；不同图像的高频细节差异（如水平条纹 vs 垂直发丝）导致不同的不确定性方向，需要不同的采样策略

2. **采样-重建对构建（$(M_j, \theta_j)_{j=1}^J$）**:
    - 功能：为每个聚类创建专用的掩码和重建网络
    - 核心思路：根据聚类质心 $c_j$ 使用拒绝采样（而非简单排序）生成掩码 $M_j$，引入随机性以优化 SSIM；然后为每个 $M_j$ 独立训练重建网络 $\theta_j$，确保 Pareto 最优
    - 设计动机：排序方法虽然最大化 PSNR（Proposition 1），但引入随机性对 SSIM 更有利；独立训练避免了一个网络服务多个掩码的 Pareto 次优问题

3. **理论保证（Theorem 3.1 & 3.2）**:
    - 功能：证明 $\mathcal{H}_{1.5}$ 的理论优势
    - 核心思路：Theorem 3.1 证明 $\mathcal{H}_1 \subseteq \mathcal{H}_{1.5}$（自适应选择包含非自适应作为特例），因此 $\inf_{h \in \mathcal{H}_{1.5}} \mathcal{L}[h] \leq \inf_{h \in \mathcal{H}_1} \mathcal{L}[h]$；Theorem 3.2 证明在 $|\pi_\phi(M_0 \mathcal{K})| \leq J$ 条件下 $\mathcal{H}_2 \subseteq \mathcal{H}_{1.5}$（每个自适应采样的掩码都能找到对应子模型）
    - 设计动机：理论上说明 $\mathcal{H}_{1.5}$ 综合了 $\mathcal{H}_1$ 的自适应性和 $\mathcal{H}_2$ 的 Pareto 最优性

### 损失函数 / 训练策略
每个子重建网络 $\theta_j$ 独立训练，损失函数为 $l(I, \hat{I}) = 1 - \mathrm{SSIM}(I, \hat{I})$。训练使用全数据集而非仅对应聚类的子集，确保每个 $\theta_j$ 在其固定掩码 $M_j$ 下达到最优。超分辨率生成模型使用已有的条件归一化流方法预训练。

## 实验关键数据

### 主实验
**CelebA 人脸图像 2D 傅里叶 CS**（SSIM↑）：

| 方法 | 类别 | 8× | 16× |
|------|------|------|------|
| VD | $\mathcal{H}_1$ | 0.9073 | 0.8734 |
| LOUPE | $\mathcal{H}_1$ | 0.8742 | 0.8673 |
| Policy | $\mathcal{H}_2$ | 0.8501 | 0.8394 |
| **Ours** | $\mathcal{H}_{1.5}$ | **0.9405** | **0.8952** |

**fastMRI 多线圈脑部 1D 线采样**（SSIM↑）：

| 方法 | 类别 | 4× | 8× |
|------|------|------|------|
| VD | $\mathcal{H}_1$ | 0.9603 | 0.9367 |
| LOUPE | $\mathcal{H}_1$ | 0.9541 | 0.9218 |
| Policy | $\mathcal{H}_2$ | 0.9569 | 0.9240 |
| **Ours** | $\mathcal{H}_{1.5}$ | **0.9624** | **0.9407** |

### 消融实验
**分段数 $J$ 的影响**（CS-MRI 2D 8×，SSIM 差值相对 J=1）：

| J 值 | 平均 SSIM 增量 | 最差 5% SSIM 增量 | 说明 |
|------|---------------|-------------------|------|
| J=1 | 0 (baseline) | 0 (baseline) | 非自适应 |
| J=2 | +0.006 | +0.008 | 最大平均增益 |
| J=3 | +0.007 | +0.012 | 平均收益放缓，但 outlier 处理更好 |
| J=4 | +0.007 | +0.014 | 平均平台，outlier 继续改善 |

**拒绝采样 vs 排序**（CS-MRI 1D 8×, J=3）：

| 掩码生成方式 | SSIM |
|------------|------|
| kmeans-Sorted | 0.9167 |
| **Rejection sampling (ours)** | **0.9407** |

### 关键发现
- $\mathcal{H}_{1.5}$ 在所有 6 种实验设置（2 数据集 × 3 加速率/采样模式）中一致超越 $\mathcal{H}_1$ 和 $\mathcal{H}_2$
- CelebA 8× 下，SSIM 提升约 0.04（0.9073→0.9405），这在图像重建中是非常显著的改善
- 在临床级别的多线圈 MRI 设置中，SSIM 约 0.004 的提升在 MRI 重建领域被认为是有意义的
- 超分辨率生成模型的样本方差有效量化了高频不确定性（Sorted-Self PSNR 37.15 vs Sorted-Another 36.36 vs VD 33.33）
- 拒绝采样比排序生成掩码高出 0.02+ SSIM，验证了引入随机性对 SSIM 优化的重要性

## 亮点与洞察
- **概念优雅**：$\mathcal{H}_{1.5}$ 的命名体现了介于固定采样（$\mathcal{H}_1$）和完全自适应（$\mathcal{H}_2$）之间的折中智慧——有限数量的预定义方案 + 数据自适应选择
- **实际可部署**：与 $\mathcal{H}_2$ 相比，避免了离散空间的困难优化，且推理时只需一次快速的不确定性计算和最近邻查找
- **Pareto 最优性洞察**：深刻指出了自适应采样中被忽视的问题——一个重建网络不可能同时对所有自适应的掩码都最优，这与多退化恢复领域的发现（盲去噪不如知噪去噪）一致
- **理论美感**：两个定理简洁有力，证明仅需数行

## 局限与展望
- 训练开销随 $J$ 线性增长，需要训练 $J$ 个独立的重建网络
- 仅探索了 1D 线采样和 2D 点采样，未涉及更复杂的非笛卡尔采样（如螺旋、径向）
- 超分辨率模型的质量直接影响不确定性量化的准确性
- $J$ 的最优选择需要启发式确定，增加 $J$ 收益递减但 outlier 鲁棒性持续提升之间的权衡缺乏自动化方法
- 聚类在训练集上进行，域迁移场景下可能需要重新聚类

## 相关工作与启发
- 与 LOUPE 等联合优化方法相比，本文通过"有限选择"巧妙避开了离散优化困难
- 与基于 cGAN 的自适应采样方法类似地利用不确定性，但本文进一步解决了 Pareto 次优问题
- 类似于分段回归/混合专家（MoE）思想——将数据空间划分为子区域，每个区域使用专家模型
- 启发：这种"预定义方案库 + 自适应选择"的范式可以推广到其他需要自适应退化处理的任务，如自适应量化、自适应压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ $\mathcal{H}_{1.5}$ 概念新颖，将 MoE 思想引入采样-重建联合优化，理论证明简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖人脸和多线圈 MRI 两个领域、多种加速率和采样模式，消融实验系统全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，理论与实践结合紧密，$\mathcal{H}_1/\mathcal{H}_2/\mathcal{H}_{1.5}$ 的对比框架直观
- 价值: ⭐⭐⭐⭐ 在临床 MRI 场景具有直接实用价值，方法具有良好的可扩展性

<!-- RELATED:START -->

## 相关论文

- [Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](../../NeurIPS2025/model_compression/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)

<!-- RELATED:END -->
