---
title: >-
  [论文解读] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models
description: >-
  [ICML2025][图像生成][扩散模型加速] 提出 Morse 双采样框架，通过快速 Dot 模型学习残差反馈来补偿 Dash（原扩散模型）跳步采样的信息损失，实现 1.78×–3.31× 的无损加速。
tags:
  - ICML2025
  - 图像生成
  - 扩散模型加速
  - 双采样框架
  - 残差反馈
  - 跳步采样
  - 无损加速
---

# Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models

**会议**: ICML2025  
**arXiv**: [2506.18251](https://arxiv.org/abs/2506.18251)  
**代码**: [deep-optimization/Morse](https://github.com/deep-optimization/Morse)  
**作者**: Chao Li, Jiawei Fan, Anbang Yao
**领域**: image_generation  
**关键词**: 扩散模型加速, 双采样框架, 残差反馈, 跳步采样, 无损加速

## 一句话总结

提出 Morse 双采样框架，通过快速 Dot 模型学习残差反馈来补偿 Dash（原扩散模型）跳步采样的信息损失，实现 1.78×–3.31× 的无损加速。

## 研究背景与动机

扩散模型（DM）在图像生成、文生图等任务上效果出色，但采样过程需要数百步迭代 function evaluation，推理开销极大。现有加速路线分两类：

**改进采样器**：DDIM、SDE/ODE solver（DPM-Solver）等，通过更好的步长调度减少采样步数，但步数过少时质量下降明显；

**知识蒸馏**：Progressive Distillation、Consistency Distillation 等，训练学生模型用更少步数匹配教师输出，但通常伴随性能损失。

本文提出的核心问题：**给定任意预训练扩散模型和任意采样器，能否在不损失生成质量的前提下，在广泛的步数预算（几步到上百步）下获得一致的加速？**

关键观察：主流 DM 普遍支持**跳步采样（Jump Sampling, JS）**——只访问时间步的一个子序列。JS 使采样更快，但跳过的步会造成信息损失，步长越大质量退化越严重。如果能高效弥补 JS 的信息损失，就能同时获得速度和质量。

## 方法详解

### 整体框架：Dash + Dot 双模型

Morse 将单模型的迭代生成重构为两个模型的交互：

- **Dash**：就是待加速的预训练 DM，在 JS 模式下运行（跳步采样，减少 Dash 调用次数）；
- **Dot**：比 Dash 快 $N$ 倍（$N \approx 5\text{–}10$）的轻量模型，负责在相邻 Dash 采样点之间插入额外步，通过残差反馈补偿 JS 的信息损失。

两者按时间交替执行，Dash 负责"定锚"，Dot 负责"填充"。

### 核心公式

给定采样步序列 $t_n > \cdots > t_0$，令 $S$ 为 Dash 负责的步集合，则各步的噪声估计为：

$$
\mathbf{z}_{t_i} = \begin{cases}
\theta(\mathbf{x}_{t_i}, t_i) & t_i \in S \\
\mathbf{z}_{t_s} + \eta(\mathbf{x}_{t_s}, \mathbf{x}_{t_i}, \mathbf{z}_{t_s}, t_s, t_i) & t_i \notin S
\end{cases}
$$

- $\theta$：Dash 模型（预训练 DM），独立估计噪声；
- $\eta$：Dot 模型，基于当前 Dash 轨迹点的观测（输入样本 $\mathbf{x}_{t_s}$、输出样本 $\mathbf{x}_{t_i}$、噪声估计 $\mathbf{z}_{t_s}$、时间步 $t_s, t_i$）生成**残差反馈**；
- Dot 不独立估计噪声，而是在 Dash 的估计 $\mathbf{z}_{t_s}$ 上加残差修正，使结果逼近 Dash 不跳步时的估计。

### 加速分析

标准流程 $n$ 步需 $n$ LSD（Latency per Step of Dash）。使用 Morse 后，在相同 $n$ LSD 预算内，可运行 $(n-k)$ 步 Dash + $Nk$ 步 Dot，共 $(n - k + Nk)$ 步。理论加速上界：

$$\text{Speedup} = \frac{n - k + Nk}{n}$$

当 $N = 5, k = n/2$ 时，理论加速 $3\times$。

### 权重共享与 Dot 构建

Dot 模型并非从零训练，而是基于 Dash 的权重共享策略构建：

1. 在 Dash 模型顶部和底部各加 $m$ 个轻量下采样/上采样块（通常 $m = 2$），输入分辨率降低 $4^m = 16$ 倍；
2. 共享 Dash 的预训练层权重（冻结），仅训练新增块和 LoRA 模块；
3. Dot 继承 Dash 大部分知识，训练极其高效。

### 训练目标

Dot 的训练目标是让残差反馈输出 $\mathbf{z}_{t_s} + \eta(\cdot)$ 逼近 Dash 在不跳步情况下对 $t_i$ 的噪声估计 $\theta(\mathbf{x}_{t_i}, t_i)$，即标准的监督回归损失。

## 训练与实验设置

- **Dot 训练**：沿用对应 Dash 模型的官方训练配置，但 batch size 和迭代次数大幅减少；
- **额外块数** $m = 2$，使 Dot 比 Dash 快 5–10 倍；
- **硬件**：8× NVIDIA RTX 3090；
- **评估指标**：FID（主要）、CLIP score（文生图任务）；
- **加速度量**：定义 LSD（Latency per Step of Dash）为时间单位，在相同 FID 下比较有/无 Morse 所需的 LSD，计算平均加速比；
- **Stable Diffusion 实验**：Dash 为 SD v1.4（860M 参数），Dot 仅 97.84M 参数，用 2M 训练样本（仅 Dash 训练数据的 0.1%），190 A100-hours（仅 Dash 的 0.1%）。

## 主要结果

### 不同采样器加速（CIFAR-10）

| 采样器 | 平均加速 |
|--------|---------|
| DDPM | 2.01× |
| DDIM | 2.94× |
| DPM-Solver (Discrete) | 显著加速 |
| SDE | 一致加速 |
| DPM-Solver (SDE) | 一致加速 |

Morse 对离散和连续时间方法均有效，甚至能加速已经利用轨迹信息的 SOTA 采样器 DPM-Solver。

### 不同基准测试（DDIM 采样器）

在 CIFAR-10 (32²)、ImageNet (64²)、CelebA (64²)、CelebA-HQ (256²)、LSUN-Church (256²) 上均获得约 2× 加速，CelebA 上部分 LSD 下超过 4×。

### 文生图（Stable Diffusion）

| 方法 | FID@10 LSD | FID@50 LSD |
|------|-----------|-----------|
| SD (best scale) | 10.65 | 8.22 |
| SD + Morse (best scale) | 8.60 | 8.15 |

平均加速 2.29×，FID-CLIP 曲线全面优于基线。

### 跨模型汇总

在 9 个基线扩散模型、6 个图像生成任务上，平均无损加速 **1.78×–3.31×**。

### 加速 LCM-SDXL（已蒸馏模型）

Morse 可叠加在 Consistency Distillation 之上，进一步加速已经加速过的 LCM-SDXL，证明与蒸馏方法互补。

## 消融实验要点

- **轨迹信息**是 Dot 模型成功的关键——去掉轨迹信息输入后性能显著下降；
- **权重共享**确保 Dot 继承 Dash 的生成知识，同时大幅降低训练代价；
- **LoRA 微调**比全量微调更高效且效果相当。

## 局限与展望

1. **需要额外训练 Dot**：虽然训练代价是 Dash 的 ~0.1%，但对每个新 DM 仍需单独训练一个 Dot，通用性受限；
2. **加速上界受 $N$ 约束**：Dot 的加速倍数 $N$ 取决于分辨率下采样幅度，过高会损失精度；
3. **未覆盖视频/3D 等模态**：实验集中在图像生成，未验证在视频生成或 3D 生成上的效果；
4. **GPU 依赖**：论文指出不同 GPU 上 $N$ 值可能变化，实际部署时加速比可能浮动；
5. **评估指标有限**：主要用 FID/CLIP score，未涉及人类偏好评估或更细粒度的质量分析。

## 可复现性要点

- 代码和模型已开源：[GitHub](https://github.com/deep-optimization/Morse)
- Dot 模型训练成本极低（190 A100-hours for SD 规模），普通实验室可复现
- 框架与采样器解耦，兼容 DDPM/DDIM/DPM-Solver/SDE 等主流采样器
- 需要注意 $m$ 和 $N$ 的选择对特定架构的适配

## 相关工作与启发

- **与 DDIM/DPM-Solver 的关系**：Morse 可叠加在任意快速采样器之上，互不冲突；
- **与蒸馏方法的关系**：蒸馏是"训练一个更快的学生"，Morse 是"保留教师 + 加一个轻量辅助"，两者互补可叠加使用；
- **残差学习思想**：类似 ResNet 的残差连接理念，Dot 只需学习"差值"而非完整噪声估计，降低学习难度；
- **对后续工作的启发**：该双模型交互范式可推广到视频扩散、一致性模型等场景。

## 个人点评

本文的核心贡献是将扩散模型加速从"改采样器"或"蒸馏"的二分法中跳出，提出了第三条路径——双模型协作。Dot 作为轻量残差补偿器的设计直觉清晰：既然跳步损失信息，就用一个快模型来补。权重共享 + LoRA 的工程设计也很务实，使 Dot 训练成本仅为 Dash 的千分之一。实验覆盖 9 个基线模型、5 种采样器、6 个数据集，说服力较强。主要遗憾是方法仍需为每个新模型训练专属 Dot，如能设计一个通用 Dot 或零样本迁移方案将更有实际价值。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 双采样框架思路新颖，从 JS 信息补偿角度切入
- 实验充分度: ⭐⭐⭐⭐⭐ — 多模型、多采样器、多数据集的广泛验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，Dash/Dot 的类比有趣
- 价值: ⭐⭐⭐⭐ — 实用性强，与现有方法互补

<!-- RELATED:START -->

## 相关论文

- [Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](../../NeurIPS2025/image_generation/breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)
- [SADA: Stability-guided Adaptive Diffusion Acceleration](sada_stability-guided_adaptive_diffusion_acceleration.md)
- [Importance Sampling for Nonlinear Models](importance_sampling_for_nonlinear_models.md)
- [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)
- [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)

<!-- RELATED:END -->
