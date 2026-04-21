---
title: >-
  [论文解读] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation
description: >-
  [ICLR 2026][特征缓存] 提出 PreciseCache——精确检测并跳过视频生成中真正冗余计算的即插即用加速框架，由 LFCache（步级，基于低频差异 LFD 度量）和 BlockCache（块级，基于输入输出差异度量）组成，在 Wan2.1-14B 等主流模型上实现平均 2.6× 加速且无明显质量损失。
tags:
  - ICLR 2026
  - 特征缓存
  - 视频扩散
  - 低频差异
  - 步级缓存
  - 块级缓存
---

# PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation

**会议**: ICLR 2026  
**arXiv**: [2603.00976](https://arxiv.org/abs/2603.00976)  
**代码**: 无  
**领域**: 视频生成/推理加速  
**关键词**: 特征缓存, 视频扩散, 低频差异, 步级缓存, 块级缓存

## 一句话总结

提出 PreciseCache——精确检测并跳过视频生成中真正冗余计算的即插即用加速框架，由 LFCache（步级，基于低频差异 LFD 度量）和 BlockCache（块级，基于输入输出差异度量）组成，在 Wan2.1-14B 等主流模型上实现平均 2.6× 加速且无明显质量损失。

## 背景与动机

**领域现状**：视频扩散模型（如 Sora、HunyuanVideo、CogVideoX、Wan2.1）的生成质量不断提升，但推理极其缓慢——Wan2.1-14B 在 4 张 A800 上生成单个 720P 视频需要约 907 秒。特征缓存（feature caching）是目前免训练加速的主流方法，通过复用前序去噪步的缓存特征来跳过部分步骤的网络推理。

**现有痛点**：

- **均匀缓存策略（如 PAB）**：每隔 $n$ 步缓存一次，忽略了不同去噪步对最终生成质量的不同贡献——高噪声步建立视频的结构和内容信息（不可跳过），低噪声步精修高频细节（可安全跳过）
- **现有自适应缓存方法**：需要复杂的额外拟合或大量超参调优，且缓存决策标准仍不够精确
- **直接使用相邻步预测差异作为缓存指标**（如 TeaCache）：该指标与最终生成质量的关联性不强，导致次优的缓存策略

**核心矛盾**：如何设计一个运行时自适应的缓存判据，能够精确区分"真正冗余的计算"和"对生成质量至关重要的计算"，从而在最大化加速的同时保持视频质量？

**本文方案**：提出低频差异（Low-Frequency Difference, LFD）作为步级冗余的精确度量指标——基于关键洞察：扩散过程在高噪声阶段建模低频结构信息（重要），在低噪声阶段精修高频细节（可缓存）。LFD 与缓存对最终质量的影响高度一致。

## 方法详解

### 整体框架

PreciseCache 由两个互补的缓存机制组成：

1. **LFCache（步级缓存）**：在每个去噪步，计算当前步与上一缓存步的低频差异（LFD）作为缓存决策指标。LFD 小于阈值则跳过该步（复用缓存），否则执行完整推理
2. **BlockCache（块级缓存）**：在 LFCache 未跳过的步骤内，进一步分析每个 Transformer 块的冗余程度。仅保留对输入特征有显著修改的关键块（pivotal blocks），跳过冗余块（non-pivotal blocks）

两者级联使用，LFCache 消除步级冗余，BlockCache 在保留步内消除块级冗余，实现双层加速。

### 关键设计 1：低频差异（LFD）度量及其高效计算

**LFD 的定义**：将网络预测 $\bm{F}_i$ 通过快速傅里叶变换（FFT）分解为低频分量 $\bm{F}_i^{LF}$ 和高频分量 $\bm{F}_i^{HF}$，定义相邻步间的低频差异：

$$\Delta_i^{LF} = \| \bm{F}_i^{LF} - \bm{F}_{i+1}^{LF} \|_2$$

低频区域定义为以最小空间维度 $\frac{1}{5}$ 为半径的圆形 mask。论文通过实验验证了 $\Delta_i^{LF}$ 与"在该步复用缓存对最终视频质量的影响"高度一致：高噪声步 LFD 大（结构变化显著，不可缓存），低噪声步 LFD 小（仅高频细节变化，可安全缓存）。

**高效估计**：直接计算 LFD 需要当前步的完整推理（违背加速目的）。关键观察：LFD 对 latent 分辨率不敏感。因此先将 latent 降采样后进行快速 "trial" 推理来估计 LFD：

$$\widetilde{\bm{Z}}_i = \text{Downsample}(\bm{Z}_i), \quad \widetilde{\bm{F}}_i = \epsilon_\theta(\widetilde{\bm{Z}}_i, t_i)$$

降采样比例为时间维度 2×、空间维度 4×4，使 trial 推理的额外开销可忽略。

**累积误差策略**：使用累积 LFD $\sum_{i=a}^{b} \widetilde{\Delta}_i^{LF}$ 作为最终指标。超过阈值 $\delta$ 时执行完整推理，否则复用缓存。阈值通过相对因子设定：$\delta = \widetilde{\Delta}_{max}^{LF} \times \alpha$，其中 $\alpha = 0.5$（Base 配置）或 $0.7$（Turbo 配置）。

### 关键设计 2：BlockCache（块级缓存）

对于 LFCache 未跳过的时间步，通过分析 DiT 内部各 Transformer 块的冗余程度进一步加速。计算每个块的输入输出差异：

$$\bm{D}_{k_i}^j = \bm{F}_{k_i}^j - \bm{F}_{k_i}^{j-1}$$

选取差异最大的前 $c\%$ 块为关键块（pivotal blocks），其余为非关键块。在后续 $L$ 个非跳过步中，非关键块直接用缓存的差异 $\bm{D}_{k_i}^j$ 估计输出：

$$\bm{F}_{k_{i-l}}^j = \begin{cases} \mathcal{B}^j(\bm{F}_{k_{i-l}}^{j-1}, t_{k_{i-l}}), & j \in \mathcal{I}_i \text{（关键块）} \\ \bm{F}_{k_{i-l}}^{j-1} + \bm{D}_{k_i}^j, & j \notin \mathcal{I}_i \text{（非关键块）} \end{cases}$$

Flash 配置中缓存率设为 40%（即 60% 的块被跳过），$L = 3$。

### 关键设计 3：即插即用和跨架构适配

PreciseCache 不修改基础模型的任何参数或结构：
- LFD 仅依赖 FFT（标准运算）和降采样
- BlockCache 仅需访问各块的输入输出（标准 hook）
- 唯一超参为相对因子 $\alpha$ 和块缓存率 $c\%$，跨模型几乎不需调整

## 实验结果

### 主实验：4 种主流模型上的效率与质量对比（4 A800 GPU）

| 方法 | 模型 | MACs (P) ↓ | 加速比 ↑ | VBench ↑ | LPIPS ↓ | PSNR ↑ |
|:-----|:-----|:----------|:---------|:---------|:--------|:-------|
| 基线 | Wan2.1-14B | 329.2 | 1× | 83.62% | - | - |
| PAB | Wan2.1-14B | 233.5 | 1.38× | 82.91% | 0.1853 | 26.18 |
| TeaCache | Wan2.1-14B | 166.3 | 1.94× | 83.24% | 0.1012 | 27.22 |
| FasterCache | Wan2.1-14B | 183.9 | 1.73× | 83.47% | 0.0741 | 28.45 |
| **Ours-base** | Wan2.1-14B | 204.5 | **1.59×** | **83.56%** | **0.0451** | **29.12** |
| **Ours-turbo** | Wan2.1-14B | 151.0 | **2.15×** | **83.52%** | **0.0633** | **28.98** |
| **Ours-flash** | Wan2.1-14B | 122.4 | **2.63×** | **83.43%** | 0.0812 | 28.76 |
| 基线 | HunyuanVideo | 14.92 | 1× | 80.66% | - | - |
| TeaCache | HunyuanVideo | 8.93 | 1.64× | 80.51% | 0.0911 | 28.15 |
| **Ours-turbo** | HunyuanVideo | 7.49 | **1.95×** | **80.49%** | 0.0884 | 29.06 |
| **Ours-flash** | HunyuanVideo | 6.04 | **2.44×** | 80.02% | 0.0902 | 28.64 |

核心结论：PreciseCache-flash 在 Wan2.1-14B 上达到 2.63× 加速，VBench 仅下降 0.19%（83.62% → 83.43%），而 PAB 在 1.38× 加速时 VBench 已下降 0.71%。在 LPIPS/PSNR 指标上，PreciseCache-base 始终最优（LPIPS 0.0451 vs. 竞品最佳 0.0741）。

### 消融实验：降采样率与 GPU 数量影响

| 降采样因子 (T×H×W) | 延迟 (s) | VBench ↑ | LPIPS ↓ |
|:-------------------|:---------|:---------|:--------|
| 基线（无缓存） | 907 (1×) | 83.62% | - |
| 1×2×2 | 918 (0.98×) | 83.57% | 0.0797 |
| 1×4×4 | 525 (1.73×) | 83.49% | 0.0801 |
| **2×4×4（默认）** | **416 (2.18×)** | **83.52%** | **0.0793** |
| 1×8×8 | 401 (2.26×) | 83.18% | 0.1946 |
| 4×4×4 | 403 (2.25×) | 83.02% | 0.1875 |

2×4×4 为最佳平衡点：过小的降采样率无法有效加速（1×2×2 仅 0.98×），过大则 LFD 估计不准导致质量下降（4×4×4 VBench 降至 83.02%）。

| GPU 数量 | Wan2.1 基线 | + PreciseCache | 加速比 |
|:---------|:-----------|:--------------|:-------|
| 1 | 3326s | 1330s | 2.50× |
| 2 | 1732s | 753s | 2.30× |
| 4 | 907s | 416s | 2.18× |
| 8 | 459s | 229s | 2.00× |

PreciseCache 在不同 GPU 数量下均有效，单 GPU 时加速比最高（2.50×），与 DSP 并行策略正交互补。

## 评价

**评分**: ⭐⭐⭐⭐

**优点**：

- LFD 指标的设计优雅且有物理直觉支撑——扩散过程的低频→高频生成顺序决定了步级冗余的分布
- 降采样 trial 推理的巧妙设计解决了"计算缓存指标本身需要推理"的鸡生蛋问题，实际开销可忽略
- 双层缓存架构（步级+块级）互补，加速效果叠加
- 实验覆盖 4 种主流视频生成模型、多种分辨率和 GPU 配置，验证了泛化性
- 即插即用，无需训练，超参极少且跨模型稳定

**不足**：

- LFD 的低频/高频划分比例（$\frac{1}{5}$ 半径）缺乏理论推导，依赖经验
- Flash 配置在部分指标（如 LPIPS）上有一定质量损失，说明激进加速存在边界
- 与蒸馏加速方法（如 consistency distillation）的对比缺失——两类方法可互补但未讨论
- BlockCache 的关键块选择是静态的（基于上一次完整推理），在去噪过程中块的重要性可能动态变化

<!-- RELATED:START -->

## 相关论文

- [DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](../../CVPR2026/video_generation/disca_accelerating_video_diffusion_transformers_wi.md)
- [MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](../../ECCV2024/video_generation/magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](../../NeurIPS2025/video_generation/lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)
- [Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](../../CVPR2025/video_generation/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)

<!-- RELATED:END -->
