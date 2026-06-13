---
title: >-
  [论文解读] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution
description: >-
  [ACL 2025][模型压缩][circular convolution] 提出 C3A 方法用循环卷积算子替代 LoRA 的低秩矩阵分解实现参数高效微调，核心优势是矩阵秩与参数量解耦——可用少量参数实现高秩适配，同时通过 FFT 保持与 LoRA 相当的计算和内存效率，在多种微调任务上一致超越 LoRA 及其变体。
tags:
  - "ACL 2025"
  - "模型压缩"
  - "circular convolution"
  - "LoRA"
  - "PEFT"
  - "FFT"
  - "参数高效微调"
---

# C3A: Parameter-Efficient Fine-Tuning via Circular Convolution

**会议**: ACL 2025  
**arXiv**: [2407.19342](https://arxiv.org/abs/2407.19342)  
**代码**: [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft) (集成到 HuggingFace PEFT)  
**领域**: 模型压缩 / 参数高效微调  
**关键词**: circular convolution, LoRA, PEFT, FFT, 参数高效微调

## 一句话总结
提出 C3A 方法用循环卷积算子替代 LoRA 的低秩矩阵分解实现参数高效微调，核心优势是矩阵秩与参数量解耦——可用少量参数实现高秩适配，同时通过 FFT 保持与 LoRA 相当的计算和内存效率，在多种微调任务上一致超越 LoRA 及其变体。

## 研究背景与动机

### 领域现状
**领域现状**：大型基础模型（LFM）在 NLP、CV 等领域取得了前所未有的性能，但其巨大参数量带来的微调成本成为实际部署的障碍。参数高效微调（PEFT）技术，以 LoRA 为代表，通过低秩矩阵 $\Delta W = BA$（$B \in \mathbb{R}^{d_1 \times r}, A \in \mathbb{R}^{r \times d_2}$，$r \ll \min(d_1, d_2)$）近似权重变化，大幅降低可训练参数量。

### 现有痛点与挑战
**现有痛点**：(1) **LoRA 的内在低秩局限**——参数量 $r(d_1+d_2)$ 同时决定了 $\Delta W$ 的秩上界 $r$，秩受限于参数预算，Zeng & Lee (2023) 证明了这一限制对目标模型逼近能力的约束；(2) **高秩方法的效率问题**——VeRA 等变体通过固定随机矩阵实现高秩，但计算和内存开销远超 LoRA（$O(r_v(d_1+d_2))$，$r_v$ 可能超过 $\max(d_1, d_2)$）；(3) **现有方法无法同时兼顾高秩、低参数量和低计算/内存开销**三个目标。

**核心矛盾**：PEFT 中秩、参数量和效率三者之间的权衡——LoRA 牺牲秩换效率，VeRA 牺牲效率换秩，如何三者兼具？

### 研究目标与方案
**本文目标**：实现高秩适配而不牺牲时间和内存效率——解耦矩阵秩与参数量。

**切入角度**：循环卷积算子 $\Delta w \star x = \mathcal{C}(\Delta w)x$ 对应的循环矩阵 $\mathcal{C}(\Delta w)$ 的秩由多项式 GCD 决定（理论上界为 $d$），与参数量（仅 $d$ 个元素）完全无关；且循环矩阵可由 Fourier 基对角化，通过 FFT 实现 $O(d \log d)$ 高效计算。

**核心 idea**：用循环卷积替代矩阵乘法作为 PEFT 的加性线性操作——实现参数量与秩的解耦 + FFT 加速。

## 方法详解

### 整体框架
C3A 的适配权重计算替换 LoRA 的 $\Delta z = BAx$ 为 $\Delta z = \Delta w \star x$，其中 $\star$ 为循环卷积。循环卷积核 $\Delta w$ 为可训练参数，其对应的循环矩阵 $\mathcal{C}(\Delta w)$ 为实际的权重变化矩阵。通过 FFT 实现的前向传播和反向传播确保计算效率。对于非方阵权重矩阵，使用块循环卷积扩展。

### 关键设计

1. **循环卷积适配（Circular Convolution Adaptation）**：

    - 功能：实现秩与参数量解耦的高效权重适配
    - 核心思路：学习循环卷积核 $\Delta w \in \mathbb{R}^d$（仅 $d$ 个参数），其对应循环矩阵 $\mathcal{C}(\Delta w)$ 的秩为 $d - \text{Deg}(\gcd(f(x), x^d-1))$，理论上界为 $d$（满秩）。前向传播通过 FFT 实现：$\Delta w \star x = \text{FFT}(\text{FFT}(\Delta w) \circ \text{iFFT}(x))$；反向传播利用循环卷积交换性 $\mathcal{C}(\Delta w)x = \mathcal{C}(x)\Delta w$，梯度计算也是循环卷积可用 FFT 加速
    - 设计动机：循环矩阵是唯一同时具备高秩灵活性和 FFT 可对角化效率的结构化矩阵形式

2. **块循环卷积扩展（Block-Circular Convolution）**：

    - 功能：支持非方阵权重矩阵（如 LLaMA-8B 中的 $4096 \times 1024$）并提供灵活的参数量控制
    - 核心思路：将激活向量 $x$ 和输出 $\Delta z$ 分为大小为 $b$ 的块，分配 $d_1 d_2 / b^2$ 个独立循环卷积核密集连接各块对。$\Delta z_i = \sum_j \Delta w_{ij} \star x_j$，对应块循环矩阵 $\mathcal{C}_{\text{blk}}(\Delta w)$。总参数量为 $d_1 d_2 / b$，其中 $b$ 为 $d_1, d_2$ 的公约数
    - 设计动机：$b$ 类似 LoRA 的 $r$ 控制参数量，但关键区别是 $b$ 不约束秩——解耦了参数量和表达能力

3. **FFT 加速的高效实现**：

    - 功能：确保计算和内存效率与 LoRA 可比
    - 核心思路：GPU 上 cuFFT 后端自动并行化 FFT 操作（并行度 $p$），C3A 总时间复杂度为 $O((d_1+d_2)/p \cdot \log b + d_1 d_2/b)$，当 $b$ 取为 $\gcd(d_1, d_2)$ 时与 LoRA 的 $O(r(d_1+d_2))$ 相当；空间复杂度仅 $d_1 d_2/b$（可训练参数）+ $pb$（FFT 缓冲），无需 VeRA 的大型固定随机矩阵
    - 设计动机：实际中 FFT 的 $O(n \log n)$ 在 GPU 上有高度优化的实现，使得理论优势可转化为实际加速

### 额外特性：循环模式作为归纳偏置
循环矩阵的结构化模式为微调引入了隐式正则化。Dosovitskiy et al. (2020) 指出 dense 线性层缺乏归纳偏置导致 Transformer 在小数据集上训练困难。C3A 的循环模式在下游数据有限时可作为有效的归纳偏置提升泛化。

## 实验关键数据

### 主实验：LLaMA-8B 微调对比

| 方法 | 可训练参数量 | 附加内存 | 时间复杂度 | 性能 |
|------|------------|---------|-----------|------|
| LoRA (r=8) | $r(d_1+d_2)$ | 0 | $O(r(d_1+d_2))$ | 基线 |
| VeRA | $r_v+d_1$（少） | $r_v(d_1+d_2)$（大） | $O(r_v(d_1+d_2))$（慢） | 略好 |
| **C3A** | $d_1 d_2/b$ | $pb$（小） | $O((d_1+d_2)/p \log b)$ | **最优** |

### 多任务微调结果

| 任务 | LoRA | VeRA | DoRA | **C3A** |
|------|------|------|------|---------|
| 常识推理 | 基线 | +0.3 | +0.5 | **+1.2** |
| 数学推理 | 基线 | +0.1 | +0.4 | **+0.9** |
| 指令遵循 | 基线 | +0.2 | +0.6 | **+1.1** |

### 消融实验：秩解耦验证

| 配置 | 参数量 | 实际秩 | 性能 |
|------|--------|-------|------|
| LoRA r=8 | 8(d₁+d₂) | ≤8 | 基线 |
| LoRA r=64 | 64(d₁+d₂) | ≤64 | +1.5 |
| C3A b=d | d | 理论上界 d | **+1.8** |

### 关键发现
- C3A 在参数量可比甚至更少时一致超越 LoRA——得益于秩解耦
- VeRA 虽参数少但内存/计算开销大，实际部署成本高；C3A 兼顾三者
- 循环模式的归纳偏置在小数据微调中提供额外增益
- 已集成到 HuggingFace PEFT 库，说明方法的工程实用性

## 亮点与洞察
- **秩-参数解耦的核心贡献**：这是 PEFT 领域的概念性突破——证明了高秩适配不必须以大参数量为代价
- **FFT 使信号处理和深度学习交叉**：循环卷积在信号处理中成熟的高效计算直接移植到 PEFT 场景
- **块循环扩展的灵活性**：$b$ 作为超参数提供了与 $r$ 类似的调节能力但更灵活
- **HuggingFace PEFT 集成**：说明方法已通过实际工程验证，可直接在生产环境使用

## 局限与展望
- **循环矩阵的表达能力上界**：循环矩阵虽然秩灵活，但其结构化约束是否在某些任务中限制表达能力尚需研究
- **$b$ 选择依赖 $\gcd(d_1, d_2)$**：当 $d_1, d_2$ 互素时 $\gcd=1$，退化为全参数微调——需要调整架构维度
- **与 LoRA 组合的可能性**：循环卷积和低秩分解是否可以互补尚未探索
- **在视觉模型中的验证**：主要实验在 LLM 上，CV 领域的 ViT 微调效果待验证

## 相关工作与启发
- **vs LoRA (Hu et al., 2021)**：低秩分解的开创性工作，秩受限于 $r$——C3A 解耦了这一约束
- **vs VeRA (Kopiczko et al., 2023)**：固定随机矩阵实现高秩但计算/内存开销大——C3A 用 FFT 解决效率问题
- **vs DoRA (Liu et al., 2024)**：正交微调方向——与 C3A 的循环结构正交，可能互补
- **vs 循环矩阵在压缩中的应用 (Cheng et al., 2015)**：早期在 LeNet 上验证但未推广到 LFM；C3A 首次证明循环卷积在现代大模型微调中的可行性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 秩-参数解耦是 PEFT 领域的概念突破
- 实验充分度: ⭐⭐⭐⭐ 多模型多任务全面对比，集成到 PEFT 库
- 写作质量: ⭐⭐⭐⭐ 理论清晰，动机充分
- 价值: ⭐⭐⭐⭐⭐ 实际工程价值高，已被社区采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [\[ACL 2025\] Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)

</div>

<!-- RELATED:END -->
