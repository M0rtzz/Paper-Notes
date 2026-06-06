---
title: >-
  [论文解读] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation
description: >-
  [ICLR 2026][模型压缩][量化] TurboBoA 提出了一种无需反向传播的 LLM 后训练量化方法，通过多 out-channel 联合量化、前层误差补偿和自适应网格选择三大创新，在保留 BoA 精度优势的同时实现了 3 倍以上加速。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "量化"
  - "注意力机制"
  - "backpropagation-free"
  - "LLM compression"
---

# TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation

**会议**: ICLR 2026  
**arXiv**: [2602.04929](https://arxiv.org/abs/2602.04929)  
**代码**: [GitHub](https://github.com/SamsungLabs/TurboBoA)  
**领域**: 模型压缩 / 量化 / LLM  
**关键词**: post-training quantization, attention-aware, backpropagation-free, low-bit quantization, LLM compression

## 一句话总结

TurboBoA 提出了一种无需反向传播的 LLM 后训练量化方法，通过多 out-channel 联合量化、前层误差补偿和自适应网格选择三大创新，在保留 BoA 精度优势的同时实现了 3 倍以上加速。

## 研究背景与动机

LLM 规模的快速增长使得后训练量化（PTQ）成为降低内存和计算成本的关键技术。基于 Hessian 引导误差补偿的无反向传播方法（如 GPTQ）因其高效性广受关注。

然而存在两类方法之间的权衡：
- **GPTQ**：假设层间独立，在低比特（如 INT2）下精度严重下降
- **BoA**：利用注意力模块内的跨层依赖改进 Hessian 近似，大幅提升精度，但需要逐 out-channel 顺序量化，效率远低于 GPTQ

核心问题：**能否在保持 BoA 精度的同时大幅提升效率？**

## 方法详解

### 整体框架

TurboBoA 引入三项关键创新：
1. **多 out-channel 联合量化**（Feature 1）——解决顺序瓶颈
2. **前层量化误差补偿**（Feature 2）——缓解误差累积
3. **自适应网格选择 + 坐标下降精炼**（Feature 3）——保持对齐

### 关键设计 1：多 out-channel 联合量化

BoA 逐个 out-channel 序列量化（如 128 次顺序操作），TurboBoA 同时量化 $N$ 个 out-channel：

将误差补偿问题形式化为：

$$\min_{\Delta\mathbf{W}} \|\mathbf{G}\Delta\mathbf{W}\mathbf{X}\|_F^2, \quad \text{s.t. } \mathbf{e}_i^T \Delta\mathbf{W} = \mathbf{Q}_{i,:} - \mathbf{W}_{i,:} \; (0 \leq i < N)$$

**Proposition 3.1**（闭合形式解）：

$$[\Delta\mathbf{W}]_{N:,:} = -[\mathbf{U}_{out}^T]_{N:,B}[\mathbf{U}_{out}^T]_{B,B}^{-1}(\mathbf{W}_{B,:} - \mathbf{Q}_{B,:})$$

其中 $B = \{0, \ldots, N-1\}$，$\mathbf{U}_{out} = \text{Chol}(\mathbf{H}_{out}^{-1})^T$。

当 $N=16$ 时，比 BoA 加速 3 倍以上（128→8 次顺序操作），精度损失可忽略。

### 关键设计 2：前层量化误差补偿

前层量化误差会传播到后续层，BoA 未考虑此问题。TurboBoA 显式建模输入偏差 $\Delta\mathbf{X} = \mathbf{X} - \tilde{\mathbf{X}}$：

$$\mathbf{G}\mathbf{Q}\mathbf{X} - \mathbf{G}\mathbf{W}\tilde{\mathbf{X}} = \mathbf{G}\Delta\mathbf{W}\mathbf{X} + \mathbf{G}\mathbf{W}\Delta\mathbf{X}$$

**Proposition 3.2**（含前层误差的闭合形式）：

$$[\Delta\mathbf{W}]_{N:,:} = -[\mathbf{U}_{out}^T]_{N:,B}[\mathbf{U}_{out}^T]_{B,B}^{-1}\left((\mathbf{W}_{B,:} - \mathbf{Q}_{B,:}) - \mathbf{W}_{B,:}\mathbf{R}\mathbf{H}_{in}^{-1}\right)$$

其中 $\mathbf{R} = \Delta\mathbf{X}\mathbf{X}^T$。相比 GPTAQ 的向量级优化，TurboBoA 处理了一般的稠密 $\mathbf{H}_{out}$。

### 关键设计 3：自适应网格 + 坐标下降精炼

- **自适应网格**：在每次量化前即时计算网格，确保与更新后的权重对齐
- **坐标下降精炼**：冻结整数权重 $\mathbf{W}_{int}$，仅优化 scale：

$$\min_{\mathbf{s}} \|\mathbf{G}(\text{diag}(\mathbf{s})\mathbf{W}_{int} - \mathbf{W})\mathbf{X} + \mathbf{G}\mathbf{W}\Delta\mathbf{X}\|_F^2$$

**Proposition 3.3**（CD 更新规则）：

$$s_j^* = s_j + \frac{[\mathbf{W}_{int}(\mathbf{H}_{in}(\mathbf{W}-\mathbf{Q})^T - \mathbf{R}^T\mathbf{W}^T)\mathbf{H}_{out}]_{j,j}}{[\mathbf{W}_{int}\mathbf{H}_{in}\mathbf{W}_{int}^T]_{j,j}[\mathbf{H}_{out}]_{j,j}}$$

### 损失函数

使用标准的 attention 重建误差，基于 Kronecker 结构 Hessian $\mathbf{H} = \mathbf{H}_{in} \otimes \mathbf{H}_{out}$。

## 实验

### 主实验：INT2 量化加速

| 方法 | N | Llama3-8B 时间 | Wiki2 PPL |
|------|---|-------------|-----------|
| BoA | 1 | 94.75 min | 15.20 |
| TurboBoA | 4 | 39.46 min | 15.27 |
| TurboBoA | 8 | 30.55 min | 15.30 |
| TurboBoA | **16** | **25.30 min** | **15.41** |
| TurboBoA | 32 | 22.95 min | 15.22 |

70B 模型：BoA 需 17 小时，TurboBoA (N=16) 仅需 5.6 小时，节省约 11 小时。

### 消融实验：三大特征

| 方法 | F2 | F3 | Llama3-8B Wiki2↓ | C4↓ |
|------|----|----|-----------------|-----|
| BoA | - | - | 15.20 | 36.95 |
| TurboBoA (F1 only) | ✗ | ✗ | 15.41 | — |
| TurboBoA (F1+F2) | ✓ | ✗ | 改善 | — |
| TurboBoA (全部) | ✓ | ✓ | 最佳 | 最佳 |

### SOTA 结果

结合 QuaRot 等异常值抑制技术后：
- **Weight-only 量化**：在 INT2 下全面超越 GPTQ、BoA 等方法
- **Weight-activation 量化**：同样达到 SOTA

### 关键发现

1. $N$ 增大到 64 精度退化仍可忽略，说明剩余 out-channel 提供了充足的误差补偿能力
2. 加速效果在 $N > 16$ 后收益递减，$N=16$ 是最优平衡点
3. 前层误差补偿和网格精炼各自贡献独立且互补

## 亮点

- 三个 Proposition 均提供了闭合形式解，理论优雅
- 3 倍以上加速的同时精度持平甚至提升
- 方法不依赖特定的 Hessian 形式，可直接适配更先进的 Hessian
- 70B 模型节省超 11 小时量化时间，实用价值显著

## 局限性

- 仅在 Llama 系列模型上验证，未测试其他架构（如 Mixtral、Qwen）
- $N$ 的选择虽然鲁棒，但缺乏理论上的误差界分析
- 稳定化系数 $\alpha$ 需要手动调参（从 {0.05, 0.125, 0.25} 中选择）
- 仅聚焦于注意力层的量化，FFN 层使用标准 GPTQ

## 相关工作

- **无反向传播量化**：GPTQ (Frantar et al., 2023)、BoA (Kim et al., 2025)、GPTAQ (Li et al., 2025)
- **变换方法**：SmoothQuant (Xiao et al., 2023)、QuaRot (Ashkboos et al., 2024)
- **早期 PTQ**：AdaRound (Nagel et al., 2020)、BRECQ (Li et al., 2021)

## 评分

- 新颖性：⭐⭐⭐⭐ — 联合量化的闭合形式解是核心创新
- 理论深度：⭐⭐⭐⭐⭐ — 三个 Proposition 完整严谨
- 实验充分性：⭐⭐⭐⭐ — 多规模模型，完善的消融
- 实用价值：⭐⭐⭐⭐⭐ — 直接解决 BoA 的效率瓶颈
- 写作质量：⭐⭐⭐⭐ — 符号体系清晰，数学推导详实

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](../../ICML2025/model_compression/boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)

</div>

<!-- RELATED:END -->
