---
title: >-
  [论文解读] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer
description: >-
  [AAAI 2026][Kolmogorov-Arnold Network] 深入分析 KAT（Kolmogorov-Arnold Transformer）训练慢 123 倍的根因，发现瓶颈并非 FLOPs 而是反向传播中**梯度累积的内存停顿**（atomic add 导致全局内存竞争），提出 FlashKAT 通过重构 GPU 核函数将训练加速 **86.5 倍**并降低近一个数量级的梯度舍入误差。
tags:
  - AAAI 2026
  - Kolmogorov-Arnold Network
  - KAN
  - Transformer
  - GPU 优化
  - 内存瓶颈
---

# FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer

**会议**: AAAI 2026  
**arXiv**: [2505.13813](https://arxiv.org/abs/2505.13813)  
**代码**: [github](https://github.com/OSU-STARLAB/FlashKAT)  
**领域**: 其他 (高效计算 / Transformer 优化)  
**关键词**: Kolmogorov-Arnold Network, KAN, Transformer, GPU 优化, 内存瓶颈

## 一句话总结

深入分析 KAT（Kolmogorov-Arnold Transformer）训练慢 123 倍的根因，发现瓶颈并非 FLOPs 而是反向传播中**梯度累积的内存停顿**（atomic add 导致全局内存竞争），提出 FlashKAT 通过重构 GPU 核函数将训练加速 **86.5 倍**并降低近一个数量级的梯度舍入误差。

## 研究背景与动机

### KAN 的兴起与困境

Kolmogorov-Arnold Network（KAN）作为 MLP 的替代方案而受到关注，其核心思想是在每条边上学习灵活的非线性函数（而非固定权重 + 固定激活），因此具有更强的表达力和可解释性。但 KAN 面临严重的实用性障碍：
- **计算每条边需要高达 204 FLOPs**（MLP 仅需 2）
- 基于 B-spline 的实现**不适合 GPU 优化**（需递归算法）
- **训练不稳定**

### KAT 的突破与残留问题

KAT 通过引入 Group-Rational KAN（GR-KAN）解决了三个问题：
1. 输入分组共享系数，大幅降低 FLOPs
2. 使用 safe Padé Activation Unit（PAU）代替 B-spline，消除递归
3. 方差保持初始化，提升训练稳定性

KAT 在视觉任务上取得了优于 ViT 的结果（如 KAT-B 82.3% vs ViT-B 79.1%）。然而，**尽管 FLOPs 已接近 ViT，KAT 训练仍然慢 123 倍**。这一巨大落差暗示 FLOPs 不是真正的性能瓶颈。

### 核心问题追踪

本文首次从**以内存为中心的视角**重新审视 KAT 性能，这是之前工作未曾探索的角度。

## 方法详解

### 整体框架

本文工作分两部分：

**Part 1：性能瓶颈诊断**（4 个 Insight）→ 定位到反向传播中梯度累积的 atomic add 导致的内存停顿

**Part 2：FlashKAT 方案**→ 重构 GPU 核函数的网格结构，将梯度在块内累积后再做 atomic add，将全局内存访问减少 $S_{block} \cdot d_g$ 倍

### 关键设计

#### 1. **性能瓶颈诊断（4 个洞察）**

**Insight 1：KAT 确实极慢**

在 H200 GPU 上的实测结果：
- KAT-T vs ViT-T：102 倍慢
- KAT-S vs ViT-S：123 倍慢
- KAT-B vs ViT-B：116 倍慢

**Insight 2：FLOPs 不是瓶颈**

人为将 GR-KAN 中的 FLOPs 增加到 8 倍（通过嵌套循环），发现**执行时间和周期数完全不变**。因为 GR-KAN 的额外 FLOPs 项 $(2m+2n+3) \times d_{in}$ 相对于 $2 \times d_{in} \times d_{out}$ 是可忽略的。

**Insight 3：反向传播占据绝对主导**

前向传播仅需 4.96ms，反向传播需要 1.03s——**207.7 倍**。因此优化重点应放在反向传播。

**Insight 4：内存停顿是罪魁祸首**

通过 Nvidia Nsight Compute 的 warp 状态统计分析：
- SM 吞吐率仅 1.97%、L1 仅 4.38%、HBM 仅 1.01%
- 每个 warp 在"Stall Long Scoreboard"（全局内存传输等待）上花费的时间是"Selected"（实际计算）的 **412 倍**
- 看似矛盾（内存受限却利用率低）的原因：**atomic add 造成了严重的内存竞争**，使 warp 调度器无法有效隐藏延迟

#### 2. **原始 KAT 的 atomic add 问题分析**

GR-KAN 的梯度 $\frac{\partial \mathcal{L}}{\partial a_{g,i}}$ 需要累积所有 batch、sequence、group 内元素的贡献（公式 10-11）。

KAT 的实现（Algorithm 1）中，每个线程独立计算一个元素的梯度贡献，然后**立即**对全局内存中的 $\mathbf{dA}$ 和 $\mathbf{dB}$ 做 atomic add。这导致：
- 每个元素产生 $3(m+n+1)$ 次全局内存访问（读系数 + 读累积值 + 写回）
- 多个线程/块同时写同一位置，造成极严重的竞争
- 总全局内存访问 = $3(m+n+2) \cdot B \cdot N \cdot d$

#### 3. **FlashKAT 核函数重构（Algorithm 2）**

核心改进：**将 1D 网格重构为 2D 网格，在块内累积梯度后再做一次 atomic add**。

2D 网格设计：
- 第一维：$T = \lceil B \cdot N / S_{block} \rceil$，处理 batch 和 sequence 维度
- 第二维：$n_g$（组数），每个块处理一个 group 的所有 $d_g$ 个元素

关键流程变化：
1. 每个块只加载**一次**对应 group 的系数 $\mathbf{A}_j, \mathbf{B}_j$（而非每个线程分别加载）
2. 在块内的共享内存/寄存器中累积所有 $S_{block} \times d_g$ 个贡献
3. 最后**每个块只做一次** atomic add（而非每个线程多次）

全局内存访问优化：
$$\text{原始} = 3(m+n+2) \cdot B \cdot N \cdot d$$
$$\text{FlashKAT} = 3\left(\frac{m+n+1}{S_{block} \cdot d_g} + 1\right) \cdot B \cdot N \cdot d$$

atomic add 和全局内存访问减少了 $\frac{1}{S_{block} \cdot d_g}$ 倍。以典型配置 $S_{block}=1024, d_g=96$ 计算，这是约 **98,000 倍**的减少。

### 损失函数 / 训练策略

- FlashKAT 的训练 curriculum 与原 KAT 完全一致（使用 DeiT 超参数）
- batch size 1024，AdamW 优化器
- Mimetic 初始化 attention 层
- GR-KAN 配置：8 组，6 个分子系数，4 个分母系数
- 第一层 GR-KAN 初始化为恒等函数，第二层为 Swish
- 数据增强：RandAugment、Mixup、CutMix、Random Erasing、Label Smoothing、Stochastic Depth
- 使用 Triton 实现自定义 GPU 核函数

## 实验关键数据

### 主实验

ImageNet-1K 训练吞吐量与精度对比：

| 模型 | 参数量 | Top-1 精度 | 训练吞吐量 (img/s) | 相对 KAT 加速 |
|------|--------|-----------|-------------------|-------------|
| ViT-T | 5.7M | 72.7% | 8954.97 | — |
| KAT-T | 5.7M | 74.6% | 87.73 | 1× |
| **FlashKAT-T** | 5.7M | **74.6%** | **6317.90** | **72.0×** |
| ViT-S | 22.1M | 78.8% | 5311.71 | — |
| KAT-S | 22.1M | 81.2% | 43.28 | 1× |
| **FlashKAT-S** | 22.1M | **81.4%** | **3741.91** | **86.5×** |
| ViT-B | 86.6M | 79.1% | 2457.15 | — |
| KAT-B | 86.6M | 82.3% | 21.24 | 1× |
| **FlashKAT-B** | 86.6M | **82.2%** | **1801.75** | **84.5×** |

反向传播核函数性能对比：

| 模型 | 周期数 | 时间 | SM 吞吐 | L1 吞吐 | L2 吞吐 | HBM 吞吐 |
|------|--------|------|---------|---------|---------|----------|
| KAT | 2.4T | 1.03s | 1.97% | 4.38% | 5.24% | 1.01% |
| **FlashKAT** | **16.9M** | **7.33ms** | **32.24%** | **34.14%** | **44.76%** | **92.05%** |

### 消融实验

梯度舍入误差对比：

| 模型 | 梯度 | 平均绝对误差 | 方差 |
|------|------|------------|------|
| KAT | $\mathbf{dA}$ | $8.84 \times 10^{-2}$ | $1.45 \times 10^{-2}$ |
| KAT | $\mathbf{dB}$ | $9.63 \times 10^{-2}$ | $8.11 \times 10^{-2}$ |
| **FlashKAT** | $\mathbf{dA}$ | $8.42 \times 10^{-4}$ | $1.35 \times 10^{-6}$ |
| **FlashKAT** | $\mathbf{dB}$ | $9.81 \times 10^{-4}$ | $1.11 \times 10^{-5}$ |

Warp 状态分析对比：

| 状态 | KAT (周期) | FlashKAT (周期) |
|------|-----------|----------------|
| Selected (计算) | ~2.4 | 计算占比增大 |
| Stall Long Scoreboard | 981.51 | **2.31** |

### 关键发现

1. **精度完全保持**：FlashKAT-T/S/B 与 KAT-T/S/B 精度相同（甚至 FlashKAT-S 微升 0.2%）
2. **极其显著的加速**：72.0-86.5 倍，将 KAT 与 ViT 的训练速度差距从 100+ 倍缩小到约 25%
3. **内存利用率从极低到合理**：HBM 吞吐从 1.01% → 92.05%，SM 吞吐从 1.97% → 32.24%
4. **梯度精度提升约 100 倍**：MAE 从 $10^{-2}$ 降至 $10^{-4}$，方差降低 3-4 个数量级
5. **Stall Long Scoreboard 减少 425 倍**：从 981.51 周期降至 2.31 周期
6. **加速是纯粹的系统优化**：不改变模型结构、不影响训练结果、不引入近似

## 亮点与洞察

1. **系统级分析的典范**：不是简单套用 profiler 数据，而是逐步排除假设（FLOPs？前向传播？），最终精准定位到 atomic add 的内存竞争
2. **反直觉的发现**："内存受限但内存利用率极低"——这源于 atomic add 的串行化效应，导致 warp 大量时间在等待而非传输/计算
3. **优雅的解决方案**：2D 网格重构 + 块内累积是经典的并行计算优化策略，但在 KAN/PAU 社区无人意识到需要这样做
4. **梯度精度提升是额外收益**：块内 reduction sum 比大量 atomic add 具有更好的数值稳定性
5. **对整个 learnable rational activation 社区的贡献**：FlashKAT 的优化直接适用于 PAU 及其变体

## 局限与展望

1. **仍比 ViT 慢约 25%**：FlashKAT-B 1801 img/s vs ViT-B 2457 img/s，差距主要来自 GR-KAN 前向传播的额外计算
2. **仅优化了反向传播**：前向传播虽然快很多，但如能进一步用类似策略优化也可获益
3. **仅验证视觉任务**：KAT 在 NLP 等其他领域的表现未验证
4. **需要自定义 Triton 核**：对最终用户的易用性有一定门槛
5. **未探索混合精度训练**：考虑到梯度精度的提升，FP16/BF16 混合精度下 FlashKAT 的优势可能更加显著

## 评分

- 新颖性: ⭐⭐⭐⭐ — 以内存为中心的分析视角新颖，解决方案虽基于经典策略但在该领域首创
- 实验充分度: ⭐⭐⭐⭐⭐ — Nsight Compute 深度 profiling、多尺度模型、精度/速度/误差全面评估
- 写作质量: ⭐⭐⭐⭐⭐ — 从问题发现到诊断到方案的叙事逻辑极其清晰，堪称系统优化论文的范例
- 价值: ⭐⭐⭐⭐⭐ — 从 123 倍慢到仅 25% 慢，直接使 KAT 成为可行的竞争方案

<!-- RELATED:START -->

## 相关论文

- [Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](../../ICLR2026/interpretability/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](../../ICLR2026/interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)

<!-- RELATED:END -->
