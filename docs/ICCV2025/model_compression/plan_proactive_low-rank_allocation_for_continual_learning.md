---
title: >-
  [论文解读] PLAN: Proactive Low-Rank Allocation for Continual Learning
description: >-
  [ICCV 2025][模型压缩][持续学习] 提出 PLAN 框架，通过为每个任务前瞻性地分配正交低秩子空间并使用扰动策略最小化任务间干扰，在持续学习场景下实现了高效且无遗忘的大模型微调，在标准 CL 基准上建立了新的 SOTA。 领域现状：持续学习 (Continual Learning, CL) 要求模型顺序学习多个任…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "持续学习"
  - "低秩适配"
  - "LoRA"
  - "灾难性遗忘"
  - "子空间分配"
---

# PLAN: Proactive Low-Rank Allocation for Continual Learning

**会议**: ICCV 2025  
**arXiv**: [2510.21188](https://arxiv.org/abs/2510.21188)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 持续学习, 低秩适配, LoRA, 灾难性遗忘, 子空间分配

## 一句话总结

提出 PLAN 框架，通过为每个任务前瞻性地分配正交低秩子空间并使用扰动策略最小化任务间干扰，在持续学习场景下实现了高效且无遗忘的大模型微调，在标准 CL 基准上建立了新的 SOTA。

## 研究背景与动机

**领域现状**：持续学习 (Continual Learning, CL) 要求模型顺序学习多个任务而不遗忘已学知识。随着大规模预训练模型的普及，基于参数高效微调 (PEFT) 的持续学习方法成为主流，其中 LoRA（Low-Rank Adaptation）因只需训练少量参数而受到广泛关注。现有方法如 O-LoRA、InfLoRA 等尝试用正交约束在 LoRA 框架下解决遗忘问题。

**现有痛点**：现有的基于 LoRA 的持续学习方法存在两个核心问题：(1) **被动分配**——大多数方法在学习新任务时才决定如何使用参数空间，未考虑未来任务的需求，导致后续任务可用的"参数预算"越来越少；(2) **干扰累积**——即使使用正交约束，在共享权重矩阵上叠加多个低秩更新仍会产生隐式干扰，因为正交性约束只保证投影方向不同，不能保证在梯度更新时不互相影响。

**核心矛盾**：LoRA 的参数空间是有限的（受秩 $r$ 限制），在持续学习中需要在"为当前任务充分学习"和"为未来任务保留空间"之间取得平衡。现有方法要么贪婪地分配空间导致后续任务性能下降，要么过分保守导致每个任务都学不好。

**本文目标**：(1) 设计一种前瞻性的子空间分配策略，提前规划好每个任务使用的低秩方向；(2) 引入干扰感知的基向量选择机制，确保新任务的学习不会破坏已有知识。

**切入角度**：作者观察到，如果能提前为每个任务预分配一组正交基向量，并在实际训练时选择对已有参数干扰最小的基向量子集来使用，就能从根本上解决被动分配和干扰累积的问题。关键洞察是：不同基向量方向对已学参数的敏感度可以通过扰动分析来量化。

**核心 idea**：预先生成一组全局正交基向量，在学习新任务时通过扰动策略评估每个基向量对已有任务的干扰程度，主动选择干扰最小的基向量来构建新任务的低秩适配器。

## 方法详解

### 整体框架

PLAN 在预训练模型的每个线性层旁添加低秩适配器（与 LoRA 结构类似）。在训练开始前，为所有可能的任务预分配一个共享的正交基向量池。当新任务到来时，PLAN 首先从池中选出对已有任务干扰最小的基向量子集，然后使用这些基向量构建当前任务的低秩更新矩阵，最后在冻结基向量方向的条件下仅优化投影系数。推理时，不同任务的更新可以合并进原始权重而无需额外计算开销。

### 关键设计

1. **正交基向量池的前瞻性构建**:

    - 功能：为所有任务提供一个预定义的、互不干扰的参数子空间划分
    - 核心思路：对每个权重矩阵 $W \in \mathbb{R}^{d \times k}$，预生成一组正交基向量 $\{v_1, v_2, ..., v_R\}$，其中 $R$ 是总预算（需要覆盖所有任务），每个任务分配 $r$ 个基向量（$r = R / T$，$T$ 是预期任务数）。基向量通过 Gram-Schmidt 正交化或随机正交矩阵生成，确保 $v_i^T v_j = 0, \forall i \neq j$。任务 $t$ 的低秩更新为 $\Delta W_t = \sum_{i \in S_t} \alpha_i v_i u_i^T$，其中 $S_t$ 是分配给任务 $t$ 的基向量索引集
    - 设计动机：前瞻性分配避免了"先到先得"的贪婪策略，正交性保证了不同任务的更新在数学上互不干扰，为后续的干扰感知选择提供了基础

2. **扰动感知的基向量选择**:

    - 功能：在分配给当前任务的候选基向量中，选择对已有任务干扰最小的子集
    - 核心思路：对每个候选基向量 $v_i$，计算在其方向上施加微小扰动 $\epsilon v_i$ 后、已有任务损失函数的变化量作为"干扰度"指标：$\text{sensitivity}(v_i) = \|\nabla_{v_i} \mathcal{L}_{\text{prev}}\|$。具体实现时，通过在 replay buffer 或已有任务的少量样本上计算梯度来估计这个敏感度。选择敏感度最低的 $r$ 个基向量作为当前任务使用的子空间
    - 设计动机：即使基向量在数学上正交，它们在模型的损失景观中的实际影响仍然不同——某些方向恰好与已有任务的关键参数对齐，扰动它们会显著影响已学知识。通过显式量化这种敏感度，PLAN 能在不牺牲新任务学习能力的前提下最小化遗忘

3. **投影系数优化与合并推理**:

    - 功能：在选定的基向量方向上高效学习任务特定的缩放系数
    - 核心思路：确定了任务 $t$ 的基向量集 $S_t$ 后，冻结基向量方向，仅优化投影系数 $\{\alpha_i\}_{i \in S_t}$ 和对应的输入投影向量 $\{u_i\}_{i \in S_t}$，将每个任务的可训练参数量降至 $O(r \times (d + k))$。推理时所有任务的更新直接求和合并到原始权重中：$W' = W + \sum_t \Delta W_t$，不增加推理延迟
    - 设计动机：冻结基向量方向是保持正交性的关键——如果允许基向量在训练中更新，可能破坏已分配任务的正交保证。仅优化系数的策略既保证了理论上的无干扰，又保留了足够的学习灵活性

### 损失函数 / 训练策略

每个任务使用标准的任务损失 $\mathcal{L}_t$（如交叉熵）进行训练，无需额外的正则化项（因为正交性已由基向量结构保证）。训练流程为：(1) 对扰动敏感度进行一次前向+后向传播估计；(2) 选基向量；(3) 标准微调。不需要 replay buffer 进行训练（仅在敏感度估计时可能使用少量旧数据）。

## 实验关键数据

### 主实验

在 Class-Incremental Learning 和 Task-Incremental Learning 标准基准上的表现：

| 方法 | CIFAR-100 (10 tasks) Avg Acc | ImageNet-R (10 tasks) Avg Acc | CUB-200 (10 tasks) Avg Acc | 类型 |
|------|------|------|------|------|
| Sequential FT | 52.3 | 41.7 | 38.5 | 上界参考 |
| EWC | 68.4 | 55.2 | 52.1 | 正则化方法 |
| L2P | 83.6 | 72.4 | 68.9 | Prompt-based |
| DualPrompt | 85.1 | 73.8 | 71.2 | Prompt-based |
| O-LoRA | 86.3 | 75.6 | 72.8 | LoRA-based |
| InfLoRA | 87.5 | 76.9 | 74.3 | LoRA-based |
| **PLAN** | **89.7** | **79.4** | **77.1** | **本文** |

### 消融实验

| 配置 | CIFAR-100 Avg Acc (%) | 说明 |
|------|---------|------|
| Full PLAN | 89.7 | 完整模型 |
| w/o Proactive Allocation（随机分配） | 86.8 | 去掉前瞻性分配，随机选基向量 |
| w/o Perturbation Selection（按序分配） | 87.4 | 去掉扰动选择，按顺序分配基向量 |
| w/o Orthogonality（普通 LoRA） | 84.2 | 去掉正交约束，退化为标准 LoRA |
| Smaller rank (r=2) | 87.1 | 降低每个任务的秩 |
| Larger rank (r=8) | 89.9 | 增大每个任务的秩 |

### 关键发现

- 正交约束是最关键的设计，去掉后（退化为标准 LoRA）性能下降 5.5%，说明任务间参数干扰是遗忘的主因
- 扰动感知选择比随机分配提升 2.9%，比按序分配提升 2.3%，表明不同基向量方向对已有任务的干扰确实存在显著差异
- 秩 $r$ 的选择对性能有影响但不敏感——从 $r=2$ 到 $r=8$ 只提升了 2.8%，说明 PLAN 在每个方向上的学习效率很高
- 在长序列任务（20 tasks）中优势更为明显，因为前瞻性分配有效避免了后期任务的"参数饥荒"问题

## 亮点与洞察

- **"前瞻性分配 + 干扰感知选择"的两阶段策略非常优雅**：先通过正交基保证数学上的不干扰，再通过扰动分析保证实际训练中的不干扰，两层防护互补
- **扰动敏感度作为子空间选择标准的想法可广泛迁移**：这个思路不仅适用于持续学习，在多任务学习、模型合并、联邦学习等需要在共享参数空间中协调多个目标的场景中都有潜在应用
- **无需 replay buffer 或额外正则化**：相比传统的经验回放或 EWC 类方法，PLAN 通过结构化的参数隔离从根本上避免了遗忘，不需要存储旧数据或维护 Fisher 信息矩阵

## 局限与展望

- 需要预先知道（或估计）总任务数 $T$ 来分配基向量预算，在任务数未知的 open-ended 场景中需要动态扩展策略
- 当任务数很大（$T > R/r_{\min}$）时，每个任务能分到的秩可能不足以有效学习，存在容量瓶颈
- 扰动敏感度的估计需要访问旧任务的数据（至少少量样本），在严格的 data-free 持续学习设定下不直接适用
- 实验主要在图像分类任务上验证，在生成任务或更复杂的多模态持续学习场景中的效果有待验证
- 未来可以探索动态秩分配——让每个任务根据其复杂度自适应地获取不同数量的基向量

## 相关工作与启发

- **vs O-LoRA**: O-LoRA 也使用正交约束来隔离任务，但它是在训练过程中动态寻找正交方向，属于"被动分配"。PLAN 的优势在于前瞻性规划，确保每个任务都有均等且无干扰的参数预算
- **vs InfLoRA**: InfLoRA 通过无限宽度近似来构建任务特定的低秩空间，理论更优雅但计算上更昂贵。PLAN 用扰动分析替代了昂贵的理论推导，在实践中更高效且效果更好
- **vs L2P/DualPrompt**: 这些 prompt-based 方法通过在输入空间添加可学习的 prompt token 来适配新任务，而 PLAN 在参数空间做适配。参数空间的方法通常有更大的表达能力和更低的推理开销

## 评分

- 新颖性: ⭐⭐⭐⭐ "前瞻性分配 + 扰动选择"的组合思路新颖，但单个组件（正交 LoRA、扰动分析）都已有前人工作
- 实验充分度: ⭐⭐⭐⭐ 在多个标准基准上取得 SOTA，消融实验覆盖了所有关键设计，但缺少大规模/真实场景的验证
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，但部分数学推导可以更简洁
- 价值: ⭐⭐⭐⭐ 为 LoRA-based 持续学习提供了一个强 baseline，前瞻性分配的思路有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](../../ICML2026/model_compression/energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](../../CVPR2025/model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)

</div>

<!-- RELATED:END -->
