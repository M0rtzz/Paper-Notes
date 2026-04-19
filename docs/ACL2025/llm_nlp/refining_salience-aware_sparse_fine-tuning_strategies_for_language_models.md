---
title: >-
  [论文解读] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models
description: >-
  [ACL2025][LLM/NLP][参数高效微调] 首次系统评估了 8 种显著性度量在稀疏参数高效微调（SPEFT）中的效果，发现简单的梯度基方法配合静态掩码即可一致性地超越 LoRA，挑战了"PEFT 需要复杂设计"的常见认知。
tags:
  - ACL2025
  - LLM/NLP
  - 参数高效微调
  - 稀疏微调
  - 显著性度量
  - LoRA
  - PEFT
  - 稀疏掩码
---

# Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models

**会议**: ACL2025  
**arXiv**: [2412.13488](https://arxiv.org/abs/2412.13488)  
**代码**: [0-ml/speft](https://github.com/0-ml/speft)  
**领域**: llm_nlp  
**关键词**: 参数高效微调, 稀疏微调, 显著性度量, LoRA, PEFT, 稀疏掩码

## 一句话总结

首次系统评估了 8 种显著性度量在稀疏参数高效微调（SPEFT）中的效果，发现简单的梯度基方法配合静态掩码即可一致性地超越 LoRA，挑战了"PEFT 需要复杂设计"的常见认知。

## 研究背景与动机

**领域现状**：参数高效微调（PEFT）是大模型适配下游任务的主流范式，LoRA（低秩适配）是最流行的方法；稀疏微调（SPEFT）作为替代方案，通过只训练权重矩阵中的少量非零元素来实现参数高效。

**现有痛点**：SPEFT 领域存在多种显著性度量（梯度、Fisher 信息、SNIP 等）和掩码策略（静态 vs 动态），但缺乏统一的系统比较，各方法各自为政，没有公认的最佳实践。

**核心矛盾**：现有 SPEFT 方法（如 FishMASK 用 Fisher 信息、DiffPruning 用可学习掩码）倾向于使用更复杂的二阶度量或动态更新策略，但这些复杂性是否真正带来了性能提升，尚无定论。

**本文目标**：回答两个基本设计问题——(1) 哪种显著性度量最优？(2) 静态掩码是否已经足够，还是动态掩码更好？

**切入角度**：借鉴零成本 NAS（网络架构搜索）代理的研究框架，将 NAS 中的参数重要性度量统一引入 SPEFT 的掩码构建中进行系统评估。

**核心 idea**：简单的梯度基显著性度量 + 静态掩码是 SPEFT 的最佳设计，复杂的二阶度量和动态掩码不仅不能显著提升性能，反而增加了计算开销。

## 方法详解

### 整体框架

SPEFT 将每层权重重参数化为 $\theta = \theta_0 + \theta_{sp}$，其中 $\theta_0$ 冻结，$\theta_{sp}$ 是极稀疏可训练矩阵。框架分为三步：(1) 用显著性度量计算每个参数的重要性分数；(2) 选 top-ρ 构建二值掩码 τ；(3) 只训练掩码覆盖的非零位置。

### 关键设计1：8种显著性度量的统一评估

- **功能**：系统评估 6 种一阶度量（Magnitude、Gradient、SNIP、FORCE、Taylor-FO、SynFlow）和 2 种二阶度量（GRaSP、Fisher Information）
- **为什么**：这些度量分散在剪枝、NAS、SPEFT 等不同文献中，从未在统一条件下比较
- **怎么做**：
    - **Gradient**：$\partial \ell / \partial \theta$，损失对权重的梯度
    - **SNIP**：$|\partial \ell / \partial \theta \odot \theta|$，连接敏感性
    - **Fisher Info**：$(\partial \ell / \partial \theta)^2$，Fisher 信息对角近似
    - **GRaSP**：$-(H \cdot \partial \ell / \partial \theta) \odot \theta$，基于 Hessian 的梯度信号保持
    - 所有度量在相同可训练参数量下公平比较

### 关键设计2：全局 vs 局部稀疏掩码

- **功能**：比较全局（跨所有层统一排序取 top-ρ）和局部（每层独立取 top-ρ）两种掩码构建策略
- **为什么**：全局策略允许不同层有不同稀疏度，可能更灵活；局部策略保证每层都有参数被训练
- **怎么做**：全局——$\tau = \mathbf{1}[s \geq \text{top}_\rho(s)]$，所有参数统一排序；局部——$\tau^{(l)} = \mathbf{1}[s^{(l)} \geq \text{top}_\rho(s^{(l)})]$，逐层排序

### 关键设计3：静态 vs 动态掩码

- **功能**：比较训练前一次性确定掩码（静态）和训练中每隔 I 步重新计算掩码（动态）
- **为什么**：动态掩码理论上能适应训练过程中权重的变化，但增加开销且需重置优化器动量
- **怎么做**：动态掩码每 1000 步用 1024 个样本重新估计显著性分数并更新掩码；静态掩码在训练前计算一次后固定

### 训练策略

- **稀疏度**：固定 ρ 使可训练参数量与 LoRA 相同（如 OPT-350m 为 0.35%，BERT-base 为 0.27%）
- **显著性估计开销**：一阶度量仅需 64 步 × batch 16 = 1024 样本，占训练时间不到 1%
- **优化器**：标准 SGD/Adam，学习率网格搜索 5e-4 到 5e-5

## 实验关键数据

### 主实验：OPT-350m 上 GLUE 基准（0.35% 可训练参数）

| 方法 | MNLI | MRPC | QNLI | QQP | SST-2 | STS-B | 平均 | 最优数 |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 83.56 | 84.56 | **89.69** | **89.66** | 93.87 | 88.57 | 88.32 | 2 |
| PiSSA | 83.45 | 83.09 | 89.38 | 89.66 | 93.58 | 88.39 | 87.93 | 1 |
| **Gradient** | **83.86** | **84.80** | 89.68 | 89.51 | **93.93** | **88.95** | **88.45** | **3** |
| Fisher-Info | 35.45 | 84.31 | 88.12 | 86.34 | 87.16 | 88.61 | 78.33 | 0 |
| SynFlow | 77.45 | 77.94 | 83.19 | 88.03 | 92.32 | 79.18 | 83.02 | 0 |
| Magnitude | 79.34 | 71.57 | 86.45 | 87.68 | 91.98 | 45.04 | 77.01 | 0 |

### 消融实验：静态/动态 + 全局/局部掩码（OPT-350m）

| 掩码策略 | MNLI | MRPC | QNLI | QQP | SST-2 | STS-B | 平均 |
|----------|---:|---:|---:|---:|---:|---:|---:|
| Static-Global (SG) | 83.86 | 84.80 | 89.68 | 89.51 | 93.93 | 88.95 | 88.46 |
| Static-Local (SL) | 84.31 | 83.33 | 90.63 | 90.97 | 94.50 | 88.52 | 88.71 |
| Dynamic-Global (DG) | 78.03 | 85.29 | 89.22 | 84.24 | 91.51 | 88.54 | 86.14 |
| Dynamic-Local (DL) | 78.86 | 71.57 | 80.84 | 84.52 | 87.27 | 87.52 | 81.76 |

### 大模型实验：GSM8K（Gemma2-2b，0.97% 参数）

| 方法 | Flexible Extract | Strict Match | 平均 |
|------|---:|---:|---:|
| Pretrained | 24.56 | 17.66 | 21.11 |
| LoRA | 39.20 | 28.81 | 34.00 |
| **Gradient SPEFT** | **50.27** | **37.15** | **43.71** |
| GRaSP | 50.15 | 37.03 | 43.59 |

### 关键发现

1. **梯度是最可靠的显著性度量**：在所有模型（OPT-125m/350m/1.3b、BERT、RoBERTa、Gemma2-2b、Qwen2-7b、Llama3-8b）上，Gradient 一致性地表现最优或接近最优
2. **二阶度量不值得**：Fisher-Info 在 OPT-350m 上 MNLI 仅 35.45%（严重崩溃），GRaSP 不稳定且计算量翻倍
3. **静态掩码足够好**：动态掩码在大模型上反而性能下降（OPT-350m 平均 86.14 vs 88.46），且带来额外计算和优化器重置开销
4. **SPEFT 在复杂任务上优势放大**：GSM8K 上 Gradient SPEFT 超越 LoRA 22.6%（43.71 vs 34.00），表明稀疏适配在多步推理中特别有效
5. **全局和局部稀疏差异不大**：无一致性优劣，具体取决于模型和任务

## 亮点与洞察

1. **"简单即有效"的实验验证**：系统性地证明了简单梯度 + 静态掩码的 SPEFT 是强基线，挑战了对复杂方法的偏好
2. **NAS 代理到 SPEFT 的知识迁移**：首次将零成本 NAS 代理框架引入 SPEFT 评估，统一了分散的文献
3. **GSM8K 上的显著优势**：Gradient SPEFT 超越 LoRA 22.6% 是非常有说服力的结果，特别是在参数量完全相同的公平对比下
4. **硬件趋势的前瞻性**：指出 NVIDIA A100/H100/H200 等硬件原生支持稀疏计算，SPEFT 的实际加速潜力将随硬件发展而增大
5. **开源框架**：提供了统一的 SPEFT benchmark 框架（github.com/0-ml/speft），方便后续研究者复现和扩展

## 局限与展望

1. **显著性估计仍需前向/反向传播**：虽然开销小（<1% 训练时间），但掩码确定前需要有标注数据做梯度计算
2. **未与 LoRA 变体深入对比**：如 QLoRA、GaLore、DoRA 等最新低秩方法未纳入比较
3. **稀疏训练的硬件加速未实测**：论文指出了硬件稀疏计算趋势，但实际训练中仍用密集计算模拟稀疏
4. **动态掩码的探索深度有限**：仅尝试了固定间隔更新，未探索自适应更新频率或渐进式掩码策略
5. **MMLU 上提升有限**：Gemma2-2b 上 Gradient SPEFT (53.11) vs LoRA (53.07) 差距很小，大模型在知识密集型任务上优势不明显

## 相关工作与启发

### vs. LoRA / PiSSA（低秩适配方法）

LoRA 通过低秩矩阵 BA 约束适配空间，结构简单但限制了参数选择的灵活性（只能在行/列方向低秩）。SPEFT 允许在权重矩阵任意位置选择最重要的参数训练，提供了更细粒度的控制。在 GLUE 上两者接近，但在 GSM8K 上 SPEFT 显著胜出（43.71 vs 34.00），表明稀疏适配在需要精准调整特定参数的复杂任务上有优势。

### vs. FishMASK / Fish-DIP（Fisher 信息基方法）

FishMASK 使用 Fisher 信息构建静态掩码，Fish-DIP 进一步允许动态更新。本文的实验表明 Fisher 信息度量不如简单梯度可靠（OPT-350m 上 78.33 vs 88.45），且在某些任务上会严重崩溃。这说明二阶度量的计算代价并未换来一致的性能提升。

### vs. DiffPruning（可学习掩码）

DiffPruning 通过直通梯度估计器学习掩码，本质上是一种动态策略。本文发现静态掩码已经足够好，且避免了掩码学习的额外开销和优化器重置问题。

## 评分

- **新颖性**: ⭐⭐⭐ — 方法本身是已有技术的系统性比较，创新点在于统一评估框架和"简单即有效"的结论
- **实验充分度**: ⭐⭐⭐⭐⭐ — 8种度量 × 多个模型 × 多个基准 × 静态/动态/全局/局部消融，覆盖极为全面
- **写作质量**: ⭐⭐⭐⭐ — 问题提炼清晰，实验组织有序，NAS代理到SPEFT的联系阐述自然
- **价值**: ⭐⭐⭐⭐ — 为SPEFT领域确立了简单而强的基线，开源框架有长期价值，GSM8K上大幅超越LoRA是实用发现

<!-- RELATED:START -->

## 相关论文

- [Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)
- [A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)

<!-- RELATED:END -->
