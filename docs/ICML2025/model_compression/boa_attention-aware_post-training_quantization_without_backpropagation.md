---
title: >-
  [论文解读] BoA: Attention-aware Post-training Quantization without Backpropagation
description: >-
  [ICML 2025][模型压缩][训练后量化] 提出 BoA——首个在训练后量化中考虑跨层依赖性的无反向传播算法，通过构建注意力感知 Hessian 矩阵捕捉注意力模块内的层间交互，在低位宽（INT2）下显著超越现有 PTQ 方法。 领域现状：PTQ 是 LLM 部署的关键技术。GPTQ 等方法使用 Hessian 信息逐…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "训练后量化"
  - "注意力感知"
  - "Hessian优化"
  - "LLM量化"
  - "跨层依赖"
---

# BoA: Attention-aware Post-training Quantization without Backpropagation

**会议**: ICML 2025  
**arXiv**: [2406.13474](https://arxiv.org/abs/2406.13474)  
**代码**: [https://github.com/SamsungLabs/BoA](https://github.com/SamsungLabs/BoA)  
**领域**: 模型压缩  
**关键词**: 训练后量化, 注意力感知, Hessian优化, LLM量化, 跨层依赖

## 一句话总结
提出 BoA——首个在训练后量化中考虑跨层依赖性的无反向传播算法，通过构建注意力感知 Hessian 矩阵捕捉注意力模块内的层间交互，在低位宽（INT2）下显著超越现有 PTQ 方法。

## 研究背景与动机

**领域现状**：PTQ 是 LLM 部署的关键技术。GPTQ 等方法使用 Hessian 信息逐层优化量化权重，但假设层间独立。

**现有痛点**：逐层独立优化忽略了注意力模块中 Q/K/V/O 投影层之间的交互——一层的量化误差会通过注意力传播并影响其他层的最优量化。

**核心矛盾**：考虑跨层依赖需要更大的 Hessian 矩阵，计算和内存开销巨大。

**本文目标**：如何在无反向传播框架下高效考虑注意力层间的依赖性？

**切入角度**：将 Hessian 从逐层重建误差扩展到注意力模块级重建误差。

**核心 idea**：注意力感知 Hessian + Hessian 松弛 + 逐头同时量化，在精度和效率间取得平衡。

## 方法详解

### 整体框架
1. 构建注意力模块级的重建误差目标（而非逐层）
2. 推导注意力感知 Hessian 矩阵
3. 通过 Hessian 松弛和高效逆矩阵计算降低开销
4. 逐头同时量化 Q/K/V 投影

### 关键设计

1. **注意力感知 Hessian**:

    - 功能：将量化目标从 $\|\Delta W \cdot X\|_F^2$ 扩展为注意力模块整体输出重建误差
    - 核心思路：Hessian 包含 Q/K/V 层之间的交叉信息，捕捉如"Q 层量化误差如何通过注意力权重传播到输出"
    - 设计动机：注意力中的 softmax 使 Q/K/V 高度耦合，逐层独立量化次优

2. **Hessian 松弛与高效计算**:

    - 功能：通过块对角近似和 Cholesky 分解减少计算量
    - 核心思路：保留同一注意力头内的跨层交互，忽略不同头之间的交互
    - 设计动机：同一头内 Q/K/V 交互最强，头间交互相对较弱

3. **逐头同时量化**:

    - 功能：一次性量化一个注意力头的 Q/K/V 投影
    - 核心思路：利用注意力感知 Hessian 的块结构并行优化
    - 设计动机：比逐层串行量化更好地利用层间依赖信息

### 损失函数 / 训练策略
- 无反向传播，基于 Hessian 的 OBS 框架
- 与 SmoothQuant/QuaRot 等激活异常值抑制方法兼容
- 计算开销与 GPTQ 可比

## 实验关键数据

### 主实验
Llama-2-7B W2A16 量化 perplexity：

| 方法 | WikiText PPL ↓ | C4 PPL ↓ |
|------|---------------|---------|
| GPTQ | 107.8 | 89.2 |
| QuIP# | 12.7 | 14.8 |
| **BoA** | **10.2** | **12.1** |

### 消融实验

| 配置 | PPL | 说明 |
|------|-----|------|
| 逐层 Hessian (GPTQ) | 107.8 | 不考虑跨层 |
| 注意力感知 Hessian (完整) | 10.0 | 内存开销大 |
| 注意力感知 Hessian (松弛) | 10.2 | 精度几乎无损，内存可控 |
| BoA + QuaRot | 8.1 | 与旋转方法组合最优 |

### 关键发现
- 在极低位宽（W2）下优势最明显——GPTQ PPL 107.8 vs BoA 10.2
- 与 QuaRot/SmoothQuant 有良好协同（组合后 W2A16 达 8.1 PPL）
- W4A4 weight-activation 量化也达到 SOTA

## 亮点与洞察
- **注意力层间依赖**的建模是关键突破——在正常位宽（W4）下改进温和，但在极端压缩（W2）下差异巨大
- Hessian 松弛的块对角近似保留了最重要的头内交互，是优雅的工程决策
- 与预处理方法正交，可叠加使用

## 局限与展望
- 仅考虑注意力模块内的跨层依赖，FFN 部分仍逐层独立
- Hessian 计算仍需要校准数据的前向传播
- 在 W4 精度下改进幅度有限

## 相关工作与启发
- **vs GPTQ**: 逐层 Hessian，忽略跨层依赖
- **vs QuIP#**: 使用惰性码本但仍独立处理各层
- **vs any4**: any4 改进码本设计，BoA 改进量化优化策略，正交

## 评分
- 新颖性: ⭐⭐⭐⭐ 注意力感知 Hessian 是新颖的技术方向
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多位宽、与多种方法组合
- 写作质量: ⭐⭐⭐⭐ 技术严谨，推导清晰
- 价值: ⭐⭐⭐⭐ 在极低位宽量化上有重要突破

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](../../ICLR2026/model_compression/turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [\[ICML 2025\] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)
- [\[ICML 2025\] Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models](q-resafe_assessing_safety_risks_and_quantization-aware_safety_patching_for_quant.md)
- [\[ICML 2025\] WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training](wildchat-50m_a_deep_dive_into_the_role_of_synthetic_data_in_post-training.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](../../CVPR2025/model_compression/quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)

</div>

<!-- RELATED:END -->
