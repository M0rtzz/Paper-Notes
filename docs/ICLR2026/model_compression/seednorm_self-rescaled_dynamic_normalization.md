---
title: >-
  [论文解读] SeeDNorm: Self-Rescaled Dynamic Normalization
description: >-
  [ICLR 2026][模型压缩][归一化层] 提出 SeeDNorm，一种自适应动态归一化层，通过将输入自身作为条件来动态调整缩放系数，从而在前向传播中保留输入范数信息，同时在反向传播中保持类似 RMSNorm 的自适应梯度调整能力，以极少额外参数在语言建模和视觉任务上全面超越 RMSNorm、LayerNorm 和 DyT。
tags:
  - ICLR 2026
  - 模型压缩
  - 归一化层
  - 动态缩放
  - RMSNorm
  - DyT
  - 大语言模型
---

# SeeDNorm: Self-Rescaled Dynamic Normalization

**会议**: ICLR 2026  
**arXiv**: [2510.22777](https://arxiv.org/abs/2510.22777)  
**代码**: 无  
**领域**: 模型压缩 / 归一化层  
**关键词**: 归一化层, 动态缩放, RMSNorm, DyT, 大语言模型

## 一句话总结

提出 SeeDNorm，一种自适应动态归一化层，通过将输入自身作为条件来动态调整缩放系数，从而在前向传播中保留输入范数信息，同时在反向传播中保持类似 RMSNorm 的自适应梯度调整能力，以极少额外参数在语言建模和视觉任务上全面超越 RMSNorm、LayerNorm 和 DyT。

## 研究背景与动机

归一化层（Normalization Layer）是现代深度神经网络的基本构建模块，在稳定训练和加速收敛方面起着关键作用。在 Transformer 架构中，RMSNorm 是目前最主流的归一化方法，它将向量约束到单位超球面上，然后通过可学习的缩放参数 γ 进行逐维缩放以恢复表达能力。

然而，RMSNorm 存在两个核心局限：

**前向传播中丢弃输入范数信息**：归一化操作本身会抹除输入的尺度信息，这限制了网络的表达能力，尤其在零样本泛化场景中更为突出

**静态缩放因子 γ 不够灵活**：γ 是与输入无关的固定参数，无法适应输入数据的广泛变化和分布偏移

近期的替代方案如 DyT（Dynamic Tanh）虽然在前向传播中保留了输入范数信息，但由于 tanh 的饱和特性导致梯度消失问题。作者通过理论分析证明（Proposition 6.1），在假设输入范数恒定的条件下，DyT 在梯度层面等价于 RMSNorm 的逐元素操作，这意味着 DyT 失去了 RMSNorm 根据输入范数动态调整梯度的能力。

这引出了一个根本性问题：**能否设计一种方法，同时兼具训练稳定性、优化效率和保留输入范数信息三大优势？**

## 方法详解

### 整体框架

SeeDNorm 的核心设计是在 RMSNorm 的基础上，将静态缩放因子替换为依赖于输入的动态缩放因子。给定输入 $\mathbf{x} \in \mathbb{R}^{N \times D}$，SeeDNorm 的公式为：

$$\text{SeeDNorm}(\mathbf{x}) = [\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha} + \boldsymbol{\gamma}] \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

其中 $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \epsilon}$，$\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma} \in \mathbb{R}^{1 \times D}$ 为可学习参数，$\sigma$ 为非线性激活函数（默认 tanh）。

### 关键设计

1. **自适应缩放矩阵（Self-Rescaling Matrix）**：通过 $\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha}$ 生成与输入相关的动态缩放项。输入 $\mathbf{x}$ 与参数 $\boldsymbol{\beta}$ 做矩阵乘法得到标量，经 tanh 激活约束到 $[-1, 1]$ 范围内，再与 $\boldsymbol{\alpha}$ 相乘生成逐维缩放矩阵。这使得缩放因子能根据当前输入动态调整，保留了输入范数信息。

2. **尺度不变性分析**：当输入被缩放 $k$ 倍时，由于 RMS 归一化的尺度不变性，SeeDNorm 中唯一变化的部分是自适应缩放矩阵中的 $\sigma(k\mathbf{x} \cdot \boldsymbol{\beta}^T)$。通过将 $\boldsymbol{\beta}$ 初始化为零，使 $\nabla_\mathbf{x} f$ 初始为零，SeeDNorm 在训练初期对输入尺度变化不敏感。

3. **梯度自适应调整**：在反向传播中，当输入 $k\mathbf{x}$ 异常大时，梯度主要由 $\frac{1}{\text{RMS}(k\mathbf{x})} = \frac{1}{k \cdot \text{RMS}(\mathbf{x})}$ 主导，梯度会按 $k$ 的倍数缩小。当输入异常小时，梯度会相应放大。这种自适应梯度调整机制确保了训练稳定性。

4. **Multi-Head SeeDNorm**：在高维空间中，$\mathbf{x} \cdot \boldsymbol{\beta}^T$ 点积的方差与维度 $D$ 成正比（Theorem 3.2），这会导致梯度方差过大。提出多头形式，将 $\mathbf{x}$ 和 $\boldsymbol{\beta}$ 分为 $n$ 个子向量分别计算点积后拼接，有效降低梯度方差。视觉任务中使用多头版本。

5. **AdaSeeDNorm**：针对 DiT 中 AdaLN 的特殊结构，设计了兼容类条件信息注入的变体：
$$\text{AdaSeeDNorm}(\mathbf{x}, c) = [(\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha} + 1) \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}](1 + \boldsymbol{\gamma}(c)) + \boldsymbol{\eta}(c)$$

### 损失函数 / 训练策略

- **参数初始化**：$\boldsymbol{\gamma}$ 初始化为 1（与 RMSNorm 一致），$\boldsymbol{\beta}$ 初始化为零（确保训练初期 $\boldsymbol{\alpha}$ 的梯度从小值开始），$\boldsymbol{\alpha}$ 初始化为 1（语言建模任务）
- **正则化**：对 $\boldsymbol{\alpha}$ 和 $\boldsymbol{\beta}$ 施加 weight decay 以控制数值稳定性，$\boldsymbol{\gamma}$ 保持与基线模型一致的正则化
- **视觉任务额外技巧**：在 ViT 分类中对动态系数 $\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha}$ 施加 dropout（与 drop path rate 相同），并将 $\boldsymbol{\alpha} \cdot \boldsymbol{\beta}^T$ 除以维度来降低方差

## 实验关键数据

### 主实验

**大语言模型（MoE 架构）**

| 模型 | 训练Token | c4_en Loss | PPL | ARC-C | ARC-E | HellaSwag | PIQA |
|------|----------|-----------|------|-------|-------|-----------|------|
| OLMoE-1.3B (RMSNorm) | 500B | 2.922 | 18.63 | 32.3 | 62.2 | 55.2 | 72.6 |
| OLMoE-1.3B-DyT | 500B | 2.968 | 19.45 | 30.4 | 61.9 | 53.2 | 70.6 |
| **OLMoE-1.3B-SeeDNorm** | 500B | **2.900** | **18.12** | **34.5** | **65.4** | **56.8** | **73.1** |
| OLMoE-7B (RMSNorm) | 1000B | 2.644 | 14.07 | 40.8 | 73.7 | 71.2 | 76.6 |
| **OLMoE-7B-SeeDNorm** | 1000B | **2.631** | **13.88** | **44.5** | **76.1** | **71.8** | **79.1** |

**大语言模型（Dense 架构）**

| 模型 | 训练Token | c4_en Loss | PPL | ARC-C | ARC-E |
|------|----------|-----------|------|-------|-------|
| OLMo2-1B (RMSNorm) | 500B | 2.884 | 17.88 | 35.6 | 68.7 |
| **OLMo2-1B-SeeDNorm** | 500B | **2.879** | **17.79** | **37.8** | **70.0** |

**计算机视觉任务**（ImageNet-1K 分类 Acc@1）

| 模型 | LayerNorm | DyT | SeeDNorm |
|------|-----------|-----|----------|
| ViT-B | 82.3 | 82.5 | **82.7** |
| ViT-L | 83.1 | 83.6 | **83.6** |
| ConvNeXT-B | 83.7 | 83.7 | **83.7** |
| ConvNeXT-L | 84.3 | 84.4 | **84.6** |
| ViT-B (MAE) | 83.2 | 83.2 | **83.5** |
| ViT-L (MAE) | 85.5 | 85.4 | **85.5** |

### 消融实验

| 配置 | c4 Loss | PPL | ARC-C | 说明 |
|------|---------|------|-------|------|
| SeeDNorm (默认，α←1) | 2.900 | 18.12 | 34.5 | 最佳配置 |
| α←0.1 | 2.912 | 18.39 | 31.2 | 初始值过小限制收敛 |
| α←10 | 3.154 | 23.42 | 27.8 | 初始值过大导致不稳定 |
| scalar α（标量替换向量） | 2.909 | 18.33 | 32.6 | 逐维调整优于统一缩放 |
| 逐元素乘法 x⊙β | 2.909 | 18.33 | 36.5 | 点积表达力更强 |
| 去掉 α | 2.907 | 18.29 | 32.1 | 失去逐维动态调整 |
| 去掉 β | 2.911 | 18.37 | 31.9 | 失去非线性形状控制 |
| 去掉 γ | 2.913 | 18.41 | 33.7 | 等价于直接替换 RMS 缩放 |

**多头消融**（ViT-B ImageNet 分类）

| 头数 | Acc@1 | 说明 |
|------|-------|------|
| 1 Head | 不收敛 | 梯度方差过大 |
| 8 Head | 82.5 | 可行但非最优 |
| 16 Head | **82.7** | 最佳 |
| 32 Head | 82.5 | 过多导致梯度多样性不足 |

### 关键发现

1. **MoE 架构放大 SeeDNorm 优势**：MoE 模型中的动态激活参数使 SeeDNorm 的收敛加速优势更明显，Dense 模型虽然训练 loss 提升较小，但零样本评估提升显著
2. **有界激活函数是必须条件**：使用 GeLU/Swish 等无界函数时模型无法收敛，tanh/sigmoid/hardtanh 均可收敛，其中 tanh 效果最佳
3. **DyT 在 LLM 预训练中失败**：用 DyT 替换 OLMoE-1.3B 的归一化层导致收敛缓慢且性能下降
4. **训练 token 越多优势越大**：随着训练 token 增加，SeeDNorm 相对于 baseline 的 loss 优势持续扩大

## 亮点与洞察

1. **理论深度出色**：通过 Proposition 6.1 证明 DyT 在常数输入范数假设下等价于 RMSNorm 的逐元素微分方程解，揭示了 DyT 的根本局限——无法根据输入范数动态调整梯度
2. **设计极简但有效**：仅增加两个 $D$ 维向量参数（$\boldsymbol{\alpha}$, $\boldsymbol{\beta}$），计算复杂度增加为 $O(D)$（远小于线性层的 $O(D^2)$），是真正的"即插即用"改进
3. **多头机制的方差分析**：从点积方差与维度正比的角度分析训练不稳定性，并通过分头计算有效解决，体现了理论指导设计的风格
4. **完整的梯度分析**：对所有参数（$\boldsymbol{\alpha}$, $\boldsymbol{\beta}$, $\boldsymbol{\gamma}$, $\mathbf{x}$）在极端条件下的梯度行为给出了详尽分析，不仅停留在实验层面

## 局限与展望

1. **PyTorch 原生实现效率有限**：SeeDNorm 的碎片化操作会增加内存访问次数，影响延迟和效率，需要融合 kernel 才能达到与 RMSNorm 接近的效率
2. **AdaLN 兼容性需要特殊处理**：不能直接替换 DiT 中的 AdaLN，需要设计专用的 AdaSeeDNorm 变体
3. **视觉任务需要额外超参数**：多头数量、dropout、维度缩放等在视觉任务中需要仔细调整，增加了使用门槛
4. **未在更大规模模型上验证**：最大实验为 OLMoE-7B（1B 激活参数），在 70B+ 参数规模上的表现未知
5. **KV Cache 兼容性未讨论**：在推理阶段 SeeDNorm 对 KV cache 的影响和额外开销值得关注

## 相关工作与启发

- **BatchNorm → LayerNorm → RMSNorm** 的演化路线：逐步简化归一化操作，但始终面临丢弃输入尺度信息的问题
- **DyT（Zhu et al., 2025b）**：用 dynamic tanh 替代归一化层，保留输入范数但牺牲梯度自适应能力
- **Frac-Connection（Zhu et al., 2025a）**：SeeDNorm 可与该方法结合使用，进一步提升性能
- 本文揭示的 RMSNorm 与 DyT 的梯度等价关系可能启发后续设计更好的归一化/激活替代方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 动态缩放因子的想法直观但有效，理论分析让设计不止于启发式
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖 LLM（MoE+Dense）、分类、生成、自监督，消融详尽
- 写作质量: ⭐⭐⭐⭐ — 理论推导完整，实验叙述清晰，附录充实
- 价值: ⭐⭐⭐⭐ — 实用性强的即插即用组件，但需要 kernel 融合才能广泛部署

<!-- RELATED:START -->

## 相关论文

- [Stronger Normalization-Free Transformers](../../CVPR2026/model_compression/stronger_normalization-free_transformers.md)
- [GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [Dependency Parsing is More Parameter-Efficient with Normalization](../../NeurIPS2025/model_compression/dependency_parsing_is_more_parameter-efficient_with_normalization.md)
- [SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer](../../CVPR2026/model_compression/soda_sensitivity-oriented_dynamic_acceleration_for_diffusion_transformer.md)

<!-- RELATED:END -->
