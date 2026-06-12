---
title: >-
  [论文解读] CoLA: Collaborative Low-Rank Adaptation
description: >-
  [ACL 2025][LoRA] 提出 CoLA，一种灵活的 LoRA 架构，打破矩阵 A 和 B 之间的固定数量约束（#A=M, #B=N），并设计三种协作策略（全协作/随机协作/启发式协作），结合扩展的 PiSSA 初始化，在低样本场景下显著优于现有 PEFT 方法。
tags:
  - "ACL 2025"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "Low-Rank Adaptation"
  - "Multi-Task Learning"
  - "LLM Fine-Tuning"
---

# CoLA: Collaborative Low-Rank Adaptation

**会议**: ACL 2025  
**arXiv**: [2505.15471](https://arxiv.org/abs/2505.15471)  
**代码**: [https://github.com/zyy-2001/CoLA](https://github.com/zyy-2001/CoLA)  
**领域**: Others (参数高效微调)  
**关键词**: LoRA, parameter-efficient fine-tuning, Low-Rank Adaptation, Multi-Task Learning, LLM Fine-Tuning

## 一句话总结

提出 CoLA，一种灵活的 LoRA 架构，打破矩阵 A 和 B 之间的固定数量约束（#A=M, #B=N），并设计三种协作策略（全协作/随机协作/启发式协作），结合扩展的 PiSSA 初始化，在低样本场景下显著优于现有 PEFT 方法。

## 研究背景与动机

参数高效微调（PEFT）是在资源受限条件下微调大语言模型的关键技术。LoRA 凭借其简洁性和有效性成为最受欢迎的方法之一，但在多任务和低样本场景下面临挑战：

**单一 LoRA 的局限**：单个 LoRA 模块将不同任务的特征投影到同一低维空间，导致任务间干扰。

**MOE-LoRA 的问题**：虽然引入多专家（#A=#B=N）解耦多任务信息，但各专家独立运作，难以捕获领域知识的共性。

**非对称 LoRA 的缺陷**：HydraLoRA 等方法采用一对多结构（#A=1, #B=N），但单一矩阵 A 在样本稀缺时难以有效学习共性，且容易受噪声干扰。

**初始化问题**：现有 LoRA 变体统一使用高斯噪声+零初始化，可能导致训练早期梯度微小或随机，减慢收敛速度。

核心洞察：现有 LoRA 方法都受限于矩阵 A 和 B 之间的固定数量关系，且矩阵间的协作关系尚未被充分探索。作者观察到在 LoRA 中，A 倾向于学习数据的共性，B 则关注每个内在分量的独特性——这类似于人对面孔的记忆，记住轮廓（共性）但忽略细节（如鼻子宽度），而细节对精确识别至关重要。

## 方法详解

### 整体框架

CoLA 的核心创新在于两点：(1) 灵活的多矩阵架构（#A=M, #B=N，M 和 N 独立设置）；(2) 三种不同的矩阵协作策略。同时将 PiSSA 初始化方案扩展至 CoLA 架构。

### 关键设计

1. **灵活 LoRA 架构**: 不强制 A 和 B 之间的固定数量关系（#A=M, #B=N），使现有 LoRA 架构（vanilla LoRA：M=N=1；MOE-LoRA：M=N=K；HydraLoRA：M=1, N=K）都成为 CoLA 的特例。这种灵活性使模型能够根据数据特点和任务需求调整结构，核心动机是让模型有更大的自由度来分别学习共性和多样性知识。

2. **扩展 PiSSA 初始化**: 对预训练权重矩阵 W 进行 SVD 分解，将主奇异值和向量均匀分配给各 $A_i$ 和 $B_j$：$A_i = \frac{U_{[:,:r]} S_{[:r,:r]}^{1/2}}{M}$，$B_j = \frac{S_{[:r,:r]}^{1/2} V_{[:,:r]}^T}{N}$。这使每个矩阵初始时包含预训练权重的主要方向信息，在微调过程中各矩阵可向不同方向优化，增强泛化多样性。动机来源于 Eckart-Young-Mirsky 定理——初始 BA 包含 W 的最重要方向，有助于更快、更好地收敛。

3. **三种协作策略**:

    - **全协作 CoLA⊺**: $\Delta W = (B_1 + \cdots + B_N)(A_1 + \cdots + A_M)$，所有 A 和 B 充分交互共享知识，打破信息传递壁垒，但计算开销最高。
    - **随机协作 CoLA†**: 每个 A 与随机选择的 B 组合，类似 dropout 正则化的思想，不依赖特定组合使知识学习更鲁棒，计算开销最低。
    - **启发式协作 CoLA‡**: $\Delta W = B_1A_1 + \cdots + B_{M-1}A_{M-1} + (B_M + \cdots + B_N)A_M$（假设M<N），结合一对一和一对多关系的优势，兼顾通用和多样知识学习，计算开销适中。

### 训练策略

- 使用 LlamaFactory 框架训练
- 将生成评估统一转化为分类评估（模型输出仅一个大写字母），确保公平可复现
- 默认 LoRA rank=8，实验在随机种子 42-46 下重复 5 次取平均
- 对冲突数据集（GSM8K/BBH）使用 LLM 标准化为选择题格式

## 实验关键数据

### 主实验（单领域，Llama-3.1-8B）

| 方法 | #A\|#B | 参数量% | 通用 | 法律 | 医学 | 数学 | 金融 |
|------|--------|---------|------|------|------|------|------|
| LoRA (r=8) | 1\|1 | 0.26% | 50.36 | 25.98 | 42.66 | 51.02 | 40.38 |
| PiSSA | 1\|1 | 0.26% | 54.72 | 26.58 | 44.64 | 57.00 | 46.79 |
| HydraLoRA | 1\|3 | 0.58% | 45.86 | 26.26 | 40.61 | 47.31 | 38.87 |
| **CoLA** | 1\|3 | 0.53% | **58.04** | **36.25** | **56.11** | **57.71** | **52.45** |
| **CoLA⊺** | 2\|3 | 0.66% | **58.21** | **41.46** | **54.33** | **59.14** | **50.19** |

CoLA 和 CoLA⊺ 在所有领域均显著优于基线（p<0.01）。

### 多领域实验（Multi-tasking）

| 方法 | #A\|#B | Llama-3.2-3B | Llama-3.1-8B |
|------|--------|-------------|-------------|
| LoRA (r=64) | 1\|1 | 34.89 | 42.99 |
| MOELoRA | 8\|8 | 30.77 | 40.53 |
| HydraLoRA | 1\|14 | 29.64 | 39.08 |
| **CoLA** | 1\|14 | **36.87** | 42.87 |
| **CoLA⊺** | 4\|10 | 36.47 | **43.62** |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|----------|------|
| 有无 PiSSA 初始化 | PiSSA 对 CoLA 影响极显著 | 样本≤200时尤其明显，无 PiSSA 的 CoLA 性能骤降 |
| #A vs #B 数量关系 | #A < #B 效果最佳 | 增加 B 的收益大于增加 A |
| CoLA† vs CoLA†̂ | 随机 A 优于随机 B | 验证了 #A < #B 原则的普适性 |
| 三种策略能耗 | CoLA†<CoLA‡<CoLA⊺ | 分别对应低/中/高计算配置 |

### 关键发现

- **Observation 1**: CoLA 在单领域和多领域均有效。HydraLoRA 因高斯噪声初始化在低样本时容易过拟合。
- **Observation 2**: PiSSA 初始化对 CoLA 的影响比对 LoRA 更显著，尤其在样本量持续减少时。这归因于多矩阵 + PiSSA 使 A 和 B 都能学到预训练模型的基础指令模式。
- **Observation 3**: 矩阵 A 数量应少于 B——A 学习数据共性，B 关注各分量独特性，且高层特征获得更多权重。
- **Observation 4**: 三种策略能耗差异显著，适用于不同资源约束场景。实验总能耗不到 HydraLoRA 的 1/10。

## 亮点与洞察

- **统一框架视角**：将 vanilla LoRA、MOE-LoRA、HydraLoRA 等都纳入 CoLA 的统一框架，清晰地展示了不同结构的本质区别在于矩阵数量和协作方式。
- **A 与 B 的角色不对称性**：通过系统实验揭示了 A 学共性、B 学多样性的规律，为后续 LoRA 变体设计提供了重要的设计准则。
- **实用的能耗-性能权衡**：三种协作策略提供了灵活的计算预算选择，符合实际部署中不同资源约束的需求。

## 局限与展望

- 未在代码领域验证，因为代码题难以转化为选择题格式。
- A 和 B 之间的协作策略空间远未被充分探索——二部图结构的最大匹配等图论性质可能带来更优的策略。
- 仅在 Llama 系列模型上实验，缺少在其他架构（如 Qwen、Mistral）上的验证。
- 分类评估模式可能低估了生成任务中的性能差异。

## 相关工作与启发

- PiSSA 的 SVD 初始化思想在 CoLA 的多矩阵场景中放大了优势，说明好的初始化在更复杂结构中可能更加关键。
- A/B 矩阵的角色分工与深度学习中的层次抽象机制（LeCun, Hinton 等经典工作）一脉相承。
- 启发：LoRA 的调优不应只关注 rank 的大小，矩阵间的结构关系和协作方式同样重要，甚至更加关键。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 灵活架构 + 三种协作策略的设计空间探索有新意，将已有方法统一在同一框架下。
- **实验充分度**: ⭐⭐⭐⭐ — 6 个领域、2 种模型规模、4 个维度的 Observation 分析系统全面。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，4 个观察总结非常好，但公式符号较多增加了阅读难度。
- **价值**: ⭐⭐⭐⭐ — 对 LoRA 变体设计有重要指导意义，低样本场景下的优势具有实际应用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[ACL 2025\] Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)

</div>

<!-- RELATED:END -->
