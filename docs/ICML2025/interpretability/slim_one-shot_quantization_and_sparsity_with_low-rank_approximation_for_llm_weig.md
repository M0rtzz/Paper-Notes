---
title: >-
  [论文解读] SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression
description: >-
  [ICML 2025][one-shot compression] 提出 SLiM，一种一次性压缩框架，将硬件友好的均匀量化、半结构化稀疏和基于显著性的低秩适配器无缝整合，在 4-bit + 2:4 稀疏条件下准确率提升最高 5.66%。
tags:
  - ICML 2025
  - one-shot compression
  - 量化
  - sparsity
  - low-rank adapter
  - SLiM-Quant
  - SLiM-LoRA
---

# SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

**会议**: ICML 2025  
**arXiv**: [2410.09615](https://arxiv.org/abs/2410.09615)  
**代码**: [GitHub](https://github.com/Mohammad-Mozaffari/slim)  
**领域**: Model Compression / Quantization / Pruning  
**关键词**: one-shot compression, quantization, sparsity, low-rank adapter, SLiM-Quant, SLiM-LoRA  

## 一句话总结

提出 SLiM，一种一次性压缩框架，将硬件友好的均匀量化、半结构化稀疏和基于显著性的低秩适配器无缝整合，在 4-bit + 2:4 稀疏条件下准确率提升最高 5.66%。

## 研究背景与动机

- 单独使用剪枝或量化可以有效降低推理成本，但**联合使用时误差累积**导致严重性能下降
- 现有一次性压缩方法在 4-bit + 结构化稀疏场景下难以恢复稠密模型精度
- 低秩适配器可以弥补压缩损失，但通常需要**昂贵的重训练**
- 现有低秩方法（如 L2QER）仅针对量化设计，与稀疏结合时效果差

## 方法详解

### 三步流水线

#### Step 1: SLiM-Quant（概率化均匀量化）

将量化问题从非凸优化重新形式化为概率框架：

$$\alpha^* = \arg\min_\alpha \int_{-\infty}^{\infty} f(x)|Q^{-1}(Q(x)) - x|^2 dx$$

分解为量化误差 + 裁剪误差：

$$E_Q(\alpha) = E_{quant}(\alpha) + E_{clip}(\alpha)$$

由于实际权重分布不符合任何标准分布（高斯、拉普拉斯等均被排除），采用**数值积分+多网格策略**求解最优 $\alpha$：先粗搜索 10 个均匀采样点，再在最优区域细化。

**激活感知扩展 (SLiM-Quant^O)**：定义通道显著性为 $|diag(\mathbf{x}) \times \mathcal{W}|$，对约 1% 的最显著通道放大权重、缩小激活。

#### Step 2: 稀疏化

在量化权重上应用 Wanda 进行半结构化（2:4）或非结构化稀疏。

#### Step 3: SLiM-LoRA（显著性低秩适配器）

**核心创新**：设计满足**可逆性**和**可加性**的显著性函数 $F(\mathcal{W}) = diag(\mathbf{x})\mathcal{W}$：

$$F(A+B) = F(A) + F(B) \quad \text{(可加性)}$$

利用这两个性质，直接通过 SVD 数学求解适配器值，无需迭代优化：

$$diag(\mathbf{x})\mathcal{L}, \mathcal{R} = -SVD(diag(\mathbf{x})(E_Q + E_S))$$

其中 $E_Q, E_S$ 分别为量化和稀疏误差。

### 适配器量化

对低秩适配器也进行 4-bit 量化（AbsMax 分组量化，组大小 128），减少 4× 开销。

### 可选后压缩微调

冻结稀疏量化权重，仅微调低秩适配器，使用 STE 实现量化感知微调。

## 实验结果

### 主实验：零样本精度 (4-bit + 2:4 稀疏)

| 方法 | LLaMA-2-7B | LLaMA-2-13B | OPT-13B |
|------|-----------|-------------|---------|
| Dense | 56.6 | 60.8 | 48.7 |
| Wanda + Group AbsMax | 40.6 | 49.6 | 37.7 |
| SparseGPT + OPTQ | 42.6 | 53.3 | 43.0 |
| JSQ | 44.3 | 53.7 | 42.0 |
| **SLiM** | **46.3** | **57.2** | **43.6** |
| **SLiM + PEFT** | **47.0** | **58.9** | - |

- LLaMA-2-7B 上平均提升 **5.66%**
- LLaMA-2-13B 上平均提升 **3.89%**
- SLiM 甚至在某些配置下**超越稠密模型** 0.6%

### 硬件加速

| GPU | 层级加速比 |
|-----|----------|
| RTX 3060 | 4.3× |
| A100 | 3.8× |
| 内存减少 | 0.23× |

### 消融：各组件贡献

| 配置 | PPL (WikiText2) |
|------|---------------|
| SLiM-Quant only | 6.89 |
| + Wanda | 8.12 |
| + SLiM-LoRA | **7.45** |
| + 适配器量化 | 7.51 |
| + PEFT | 7.32 |

## 亮点

- 显著性函数的可逆+可加设计非常优雅，使低秩适配器有封闭解
- 均匀量化 + 概率优化优化消除了分组量化的额外开销
- 三种压缩技术的无缝整合，每一步都有明确的数学动机
- 端到端一次性方案，不需要大规模重训练
- 在极端压缩条件下（4-bit + 2:4 稀疏）仍保持高精度

## 局限性

- SLiM-Quant 的数值积分依赖权重直方图，对异常分布可能不鲁棒
- SLiM-LoRA 的显著性函数选择有限（仅 $diag(\mathbf{x})\mathcal{W}$），可能非最优
- 仅在 LLaMA-2 和 OPT 上验证，缺少对 LLaMA-3、Mistral 等新模型的实验
- 适配器的秩选择缺乏自适应机制
- 可选 PEFT 步骤虽然增加精度但也增加了流水线复杂性

## 评分

⭐⭐⭐⭐ — 理论推导严谨、工程实现完整，三种压缩技术的联合优化在一次性压缩中达到新 SOTA。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization](../../ACL2026/interpretability/from_signal_degradation_to_computation_collapse_uncovering_the_two_failure_modes.md)
- [\[NeurIPS 2025\] Conditional Distribution Compression via the Kernel Conditional Mean Embedding](../../NeurIPS2025/interpretability/conditional_distribution_compression_via_the_kernel_conditional_mean_embedding.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[AAAI 2026\] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](../../AAAI2026/interpretability/fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)
- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](../../NeurIPS2025/interpretability/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)

<!-- RELATED:END -->
