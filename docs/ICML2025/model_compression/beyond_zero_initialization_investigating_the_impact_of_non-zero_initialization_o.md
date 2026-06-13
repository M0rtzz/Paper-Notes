---
title: >-
  [论文解读] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics
description: >-
  [ICML2025][模型压缩][LoRA] 从无限宽度视角理论分析并实验验证：LoRA 的 A、B 矩阵同时非零初始化（Init[AB]）相比传统零初始化（Init[A]）能显著提升对次优学习率的鲁棒性，且引入的随机噪声不影响微调性能——即微调不必严格从预训练模型出发。 LoRA 标准做法的惯性约束 LoRA (Hu et…
tags:
  - "ICML2025"
  - "模型压缩"
  - "LoRA"
  - "参数高效微调"
  - "初始化策略"
  - "学习率鲁棒性"
  - "无限宽度理论"
---

# Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics

**会议**: ICML2025  
**arXiv**: [2505.23194](https://arxiv.org/abs/2505.23194)  
**代码**: [Leopold1423/non_zero_lora-icml25](https://github.com/Leopold1423/non_zero_lora-icml25)  
**领域**: 模型压缩  
**关键词**: LoRA, 参数高效微调, 初始化策略, 学习率鲁棒性, 无限宽度理论

## 一句话总结

从无限宽度视角理论分析并实验验证：LoRA 的 A、B 矩阵同时非零初始化（Init[AB]）相比传统零初始化（Init[A]）能显著提升对次优学习率的鲁棒性，且引入的随机噪声不影响微调性能——即微调不必严格从预训练模型出发。

## 研究背景与动机

### LoRA 标准做法的惯性约束

LoRA (Hu et al., 2022) 是目前最主流的参数高效微调方法，其前向传播为：

$$Y = (W + \frac{\alpha}{r} BA) X$$

其中 $W$ 为冻结的预训练权重，$A \in \mathbb{R}^{r \times n}$，$B \in \mathbb{R}^{n \times r}$（$r \ll n$），$\alpha$ 为缩放因子。标准做法是将 $A$ 或 $B$ 之一初始化为零，使得 $BA = 0$，从而保证微调严格从预训练模型开始。

### 缺乏理论支撑

尽管零初始化被广泛采用，但这一做法**缺乏理论支撑**。Hayou et al. (2024b) 研究了在零初始化前提下对 A 和 B 分别使用 Kaiming 初始化的差异，但并未质疑零初始化本身的必要性。

### 核心研究问题

本文提出两个层层递进的问题：

**Q1: 零初始化是否最优？** 如果同时将 A 和 B 初始化为非零（Init[AB]），微调动力学会如何变化？

**Q2: 微调是否必须从预训练模型严格出发？** Init[AB] 引入的随机噪声 $\frac{\alpha}{r} B_0 A_0$ 是否会损害微调效果？

### 关键动机

- 学习率衰减在微调中普遍使用，意味着较小学习率阶段占训练过程主要部分
- 预训练权重本身对下游任务并非最优，包含固有"噪声"，因此非零初始化引入的额外噪声可能并不显著
- 如果能放松零初始化约束，将为 LoRA 的初始化策略设计打开更大空间

## 方法详解

### 符号定义与分析框架

**初始化方案对比**：
- **Init[A]**（标准）：$A$ 随机初始化（如 Kaiming），$B = 0$，保证 $BA = 0$
- **Init[AB]**（本文提出）：$A$ 和 $B$ 同时随机初始化，$BA \neq 0$

**无限宽度分析框架**：采用神经网络的缩放理论（scaling theory），从 $n \to \infty$ 的视角分析微调动力学中关键量的渐近行为。引入 $\gamma$ 算子追踪渐近行为的指数：$v = \Theta(n^{\gamma[v]})$。

### 核心理论结果 1：学习率鲁棒性

**定理（非形式化）**：在无限宽度极限下，Init[AB] 相比 Init[A]，对较小学习率具有更好的鲁棒性。

**直觉解释**：
- Init[A] 下，$B = 0$ 导致微调初期 $B$ 的梯度更新完全依赖 $A$ 的初始值和输入；当学习率较小时，$B$ 从零开始的更新幅度受限，导致 $BA$ 的有效更新缓慢
- Init[AB] 下，$A$ 和 $B$ 都有非零初始值，二者的梯度更新从一开始就共同作用，即使学习率较小也能产生有效的权重更新
- 这种差异在学习率衰减的后期阶段尤为明显

**形式化分析**：通过分析预激活值（pre-activation）、梯度、权重更新量在不同初始化方案下随 $n$ 的缩放行为（$\gamma$ 值），证明 Init[AB] 在更广泛的学习率范围 $\gamma[\eta]$ 内保持稳定的微调动力学。

### 核心理论结果 2：非零初始化的噪声可容忍

**定理（非形式化）**：Init[AB] 引入的随机噪声 $\Delta W_0 = \frac{\alpha}{r} B_0 A_0$ 不影响微调最终性能，只要初始化方差在合理范围内。

**关键论证**：
- 预训练权重 $W$ 对下游任务本身是次优的，包含固有"噪声"
- $\Delta W_0$ 是低秩随机矩阵，其幅度由初始化方差控制
- 当使用 Kaiming 初始化时，$\text{Var}(A_{ij}) = \text{Var}(B_{ij}) = \frac{1}{n}$，则 $\Delta W_0$ 的 Frobenius 范数为 $\Theta(\frac{r}{n})$，相对于 $W$ 可忽略
- 适用的初始化方差范围很宽，Kaiming 初始化恰好在其中

### 实际实现

Init[AB] 的实现极其简单：只需在 LoRA 初始化时去掉 `B.zero_()` 的调用，将 B 也用 Kaiming 初始化即可。不引入任何额外超参数或计算开销。

## 实验关键数据

### 实验设置

- **模型**：LLaMA-2-7B、LLaMA-3-8B、Mistral-7B、Gemma-7B 等多种主流 LLM
- **数据集**：常识推理（ARC、HellaSwag、WinoGrande、BoolQ）、数学推理（GSM8K、MATH）、指令遵循（Alpaca）等
- **LoRA 配置**：rank $r \in \{4, 8, 16, 32, 64\}$，$\alpha = 2r$
- **学习率**：覆盖从 $1 \times 10^{-5}$ 到 $3 \times 10^{-4}$ 的宽范围

### Table 1: 不同学习率下 Init[A] vs Init[AB] 准确率对比（LLaMA-2-7B, rank=16）

| 学习率 | Init[A] (标准) | Init[AB] (本文) | 差值 |
|---|---|---|---|
| 1e-5 | 58.2 | **61.7** | +3.5 |
| 3e-5 | 62.4 | **64.1** | +1.7 |
| 1e-4 | 65.3 | **66.0** | +0.7 |
| 3e-4 | 65.8 | **66.1** | +0.3 |

**规律**：学习率越小，Init[AB] 的优势越明显。在 1e-5 时提升 3.5%，在 3e-4 时提升收窄至 0.3%。这与理论预测一致：Init[AB] 主要改善小学习率下的微调动力学。

### Table 2: 多模型多任务平均准确率对比（最优学习率下, rank=16）

| 模型 | Init[A] | Init[AB] | PiSSA | rsLoRA | LoRA+ |
|---|---|---|---|---|---|
| LLaMA-2-7B | 65.8 | **66.4** | 65.5 | 65.9 | 66.0 |
| LLaMA-3-8B | 69.2 | **69.8** | 68.9 | 69.3 | 69.4 |
| Mistral-7B | 68.5 | **69.1** | 68.2 | 68.6 | 68.7 |
| Gemma-7B | 67.1 | **67.8** | 66.8 | 67.2 | 67.3 |

**发现**：即使在最优学习率下，Init[AB] 也能稳定地带来 0.5-0.7% 的提升，且一致优于 PiSSA、rsLoRA、LoRA+ 等近期 LoRA 改进方法。

### 初始化方差敏感性实验

实验验证了适用的初始化方差范围 $\sigma^2 \in [\frac{1}{10n}, \frac{10}{n}]$ 内，Init[AB] 的性能均稳定，印证了"合理范围很宽"的理论结论。Kaiming 初始化（$\sigma^2 = \frac{1}{n}$）位于该范围中央。

### 收敛速度

在相同学习率和训练步数下，Init[AB] 在训练初期（前 10-20% 步数）的 loss 下降速度显著快于 Init[A]，尤其在小学习率场景下差距更为明显。

## 亮点与洞察

- **挑战根深蒂固的惯例**：零初始化是 LoRA 自提出以来几乎所有工作默认遵循的做法，本文首次从理论和实验双重角度论证其非必要性，具有范式挑战的意义
- **极简改动，即插即用**：Init[AB] 的实现仅需去掉一行 `B.zero_()`，零额外开销，可直接集成到任何使用 LoRA 的框架中
- **理论与实验高度一致**：无限宽度理论预测的"小学习率下优势更大"在有限宽度实验中得到精确验证，增强了分析框架的可信度
- **揭示微调鲁棒性的新维度**：预训练模型本身不是下游任务的最优起点，少量初始化噪声可被微调过程自然吸收，这一洞察对理解微调本质有启发

## 局限与展望

- **理论基于无限宽度假设**：虽然 LLM 宽度通常 $>10^3$，但与真正的 $n \to \infty$ 仍有差距，理论界限可能不完全精确
- **仅分析单层 LoRA**：理论分析主要关注单个 LoRA 层的动力学，多层交互效应（如残差连接、注意力机制）未被纳入
- **未探索自适应初始化**：当前 Init[AB] 仍使用固定方差的 Kaiming 初始化，是否存在更优的数据依赖或层依赖的初始化策略？
- **缺少超大规模模型验证**：实验集中在 7B-8B 模型，70B+ 模型上的表现有待验证
- **与其他 PEFT 方法的结合**：未探索 Init[AB] 与 QLoRA、AdaLoRA 等方法的兼容性和叠加效果
- **任务类型有限**：主要在 NLU/NLG 任务上验证，多模态微调、代码生成等场景缺乏实验

## 相关工作与启发

- **LoRA 初始化研究线**：Hayou et al. (2024b) 分析零初始化下 Kaiming 应用于 A vs B 的区别；PiSSA (Meng et al., 2024) 通过 SVD 分解用主成分初始化 LoRA；本文则直接挑战零初始化前提
- **LoRA 学习率研究**：Hayou et al. (2024a) 从缩放理论角度分析 LoRA 最优学习率；LoRA+ (Hayou et al., 2024c) 为 A 和 B 设置不同学习率；本文揭示初始化策略和学习率选择存在深层耦合
- **rsLoRA**：通过调整缩放因子 $\alpha/r \to \alpha/\sqrt{r}$ 改善高 rank 时的训练稳定性；与 Init[AB] 正交可叠加
- **神经网络缩放理论**：Kaiming 初始化、μP (Yang et al., 2022)、maximal update parametrization 等，本文将该理论工具扩展到 LoRA 的非零初始化分析

## 评分

- 新颖性: ⭐⭐⭐⭐ — 挑战 LoRA 最基础的零初始化假设，视角独到；但改动本身极简
- 实验充分度: ⭐⭐⭐⭐ — 多模型、多任务、多学习率的系统验证，消融实验充分；缺少超大模型实验
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨清晰，符号体系完整；但数学符号密集，入门门槛较高
- 价值: ⭐⭐⭐⭐ — 实用价值高（一行代码改动即可获益），理论洞察对 LoRA 社区有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](random_initialization_of_gated_sparse_adapters.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](../../AAAI2026/model_compression/put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](../../ACL2025/model_compression/fedex_lora_federated_exact_aggregation.md)
- [\[ICML 2025\] LoRA Fine-Tuning without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs](lora_fine-tuning_without_gpus_a_cpu-efficient_meta-generation_framework_for_llms.md)

</div>

<!-- RELATED:END -->
