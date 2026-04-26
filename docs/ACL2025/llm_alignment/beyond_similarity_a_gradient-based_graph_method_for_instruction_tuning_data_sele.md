---
title: >-
  [论文解读] Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection
description: >-
  [ACL 2025][LLM对齐][指令调优数据选择] 本文提出 G2IS（Gradient-based Graph Instruction Selection），通过构建基于梯度的指令图来建模指令数据之间的联合分布和相互依赖关系，结合梯度遍历算法进行数据选择，仅用 1% 的数据即可超越全数据指令调优的效果。
tags:
  - ACL 2025
  - LLM对齐
  - 指令调优数据选择
  - 梯度图
  - 领域适配
  - 联合分布
  - 图遍历算法
---

# Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection

**会议**: ACL 2025  
**arXiv**: [2502.11062](https://arxiv.org/abs/2502.11062)  
**代码**: 无  
**领域**: LLM对齐/指令调优  
**关键词**: 指令调优数据选择、梯度图、领域适配、联合分布、图遍历算法

## 一句话总结
本文提出 G2IS（Gradient-based Graph Instruction Selection），通过构建基于梯度的指令图来建模指令数据之间的联合分布和相互依赖关系，结合梯度遍历算法进行数据选择，仅用 1% 的数据即可超越全数据指令调优的效果。

## 研究背景与动机

**领域现状**：指令调优是 LLM 适配特定领域的关键方法，但领域特定数据往往有限。数据选择方法从大规模通用数据集中挑选与目标领域最相关的样本来弥补这一不足。LESS 等方法基于梯度相似度选择与验证集最相似的训练数据。

**现有痛点**：现有方法主要关注训练样本与目标领域的独立相似性，忽略了指令数据之间的相互依赖关系。例如，两个分别教授"加法"和"进位"的指令独立看来与"多位数乘法"目标不够相似，但它们组合在一起就能提供关键的基础知识。

**核心矛盾**：指令数据形成联合分布，指令之间存在互补、冗余甚至冲突关系，而现有相似度方法将每个样本独立对待，导致选出的数据不是最优组合。

**本文目标**：设计一种能捕获指令间依赖关系的数据选择方法，从通用数据集中选择对目标任务贡献最大的指令子集。

**切入角度**：梯度天然编码了训练样本对模型参数更新的影响，梯度之间的关系可以反映指令间的互补和冲突关系。

**核心 idea**：用梯度表示构建指令图（节点=指令梯度，边=梯度余弦相似度），从验证集梯度中用 PCA 提取核心知识，然后通过梯度遍历算法在图上选择满足三个约束（无知识冲突、与核心知识一致、知识连贯性）的最优指令子集。

## 方法详解

### 整体框架
输入为大规模通用指令训练集和小规模目标领域验证集，输出为选定的训练子集。流程包括：（1）计算所有样本的梯度表示；（2）构建梯度图；（3）从验证集提取核心知识；（4）基于梯度遍历选择数据。最终用选定的子集对 LLM 进行指令调优。

### 关键设计

1. **梯度知识表示（Gradient-based Knowledge Representation）**:

    - 功能：用梯度向量表示每个指令样本的知识内容
    - 核心思路：对训练集使用动量调整梯度 $\nabla\Gamma(z, \theta_t)$，通过 warmup 方法初始化 Adam 优化器的动量状态，确保梯度反映实际的优化动态；对验证集使用标准 SGD 梯度避免动量干扰。利用 LoRA 层梯度+随机投影将维度降至 8192
    - 设计动机：标准梯度不考虑优化器状态，动量调整后的梯度更准确地反映样本对模型学习的贡献。LoRA+随机投影大幅降低计算成本

2. **梯度图构建与核心知识提取**:

    - 功能：建模指令间的依赖关系并提取目标任务的核心能力
    - 核心思路：以每个训练样本的梯度为节点 $N_z = \nabla\Gamma(z, \theta_t)$，边权重为梯度余弦相似度 $R_{ij} = \cos(\nabla\Gamma(z_i, \theta_t), \nabla\Gamma(z_j, \theta_t))$。正值表示知识对齐，负值表示冲突。对验证集梯度做 PCA，提取前 50% 的主成分作为核心知识 $K_\mathcal{V}$
    - 设计动机：PCA 过滤验证集的噪声，提取最关键的任务相关能力。图结构使得可以显式建模指令间的互补/冲突关系

3. **梯度遍历数据选择算法（Gradient Walk）**:

    - 功能：在梯度图上有约束地扩展指令子集
    - 核心思路：从与核心知识最相似的锚点出发，按三个原则迭代选择新节点：（1）**无知识冲突**——新样本与已选样本的梯度相似度非负；（2）**与核心知识一致**——加入新样本后集合与核心知识 $K_\mathcal{V}$ 的余弦相似度不低于阈值 $\delta=0.8$ 倍的当前值；（3）**知识连贯性**——新样本与最近添加样本的梯度最相似。形式化为 $z^* = \arg\max_{z \in \mathcal{Z}} \cos(\nabla\Gamma(z, \theta_t), \nabla\Gamma(s^*, \theta_t))$ 满足上述约束
    - 设计动机：三个约束确保选出的数据既互补又不冲突、既与目标对齐又保持内部连贯

### 损失函数 / 训练策略
数据选择后使用 LoRA 微调（rank=128, α=512），余弦学习率调度，3 个 epoch，在 A100 GPU 集群上训练。

## 实验关键数据

### 主实验（1% 数据 vs 全数据，Llama3.1-8B + Infinity-Instruct）

| 任务 | G2IS-1% | LESS-1% | BERT-1% | All(100%) |
|------|---------|---------|---------|-----------|
| BBH | **64.78** | 63.46 | 63.05 | 64.71 |
| GPQA | **31.57** | 29.94 | 30.55 | 29.74 |
| GSM8K | **62.02** | 60.65 | 56.79 | 58.30 |
| Math | **20.96** | 18.66 | 20.22 | 20.26 |
| MMLU | **63.42** | 63.15 | 61.87 | 62.75 |

### 消融实验（COT 数据集，Llama3.1-8B，1%）

| 配置 | BBH | GPQA | GSM8K | Math | MMLU | vs Full |
|------|-----|------|-------|------|------|---------|
| G2IS | **65.66** | **32.59** | **62.70** | **21.38** | **64.22** | 1.00 |
| w/o graph | 65.64 | 30.55 | 57.85 | 20.10 | 61.42 | 0.95 |
| w/o gradient | 64.57 | 32.53 | 58.91 | 20.24 | 63.94 | 0.97 |

### 关键发现
- G2IS 仅用 1% 数据在大多数任务上超越了全数据指令调优，特别是在 GSM8K 上 Gemma-7B 提升了 12.66%
- 图结构（梯度遍历）对性能的贡献大于梯度表示本身（w/o graph 衰减更多），说明建模指令间依赖是关键
- 在多任务优化中，LESS 出现性能退化而 G2IS 保持稳健，因为图约束能有效平衡多个目标
- 选择 5% 数据时有时不如 1%，再次验证了"less is more"——过多的数据引入噪声
- PCA 主成分比例对 MMLU（多领域知识，噪声多）影响大，对 GSM8K（数学推理，噪声少）影响小

## 亮点与洞察
- **从独立到联合的范式升级**：将数据选择从"逐个评分"升级为"组合优化"，图结构自然地建模了样本间的协同和冲突关系。这一思路可以迁移到任何需要数据子集选择的场景（如主动学习、课程学习）
- **1% 数据超越 100%**：在多个模型和数据集上一致地仅用 1% 数据超过全量训练，有力地证明了"精选优于堆量"的原则，对实际的 LLM 训练有很强的指导意义

## 局限与展望
- 仅使用 LoRA 层梯度而非全参数梯度，可能遗漏了部分信息
- 实验仅在 7B-8B 模型上进行，更大模型（13B/65B/175B）的效果未验证
- 梯度计算仍需要对每个样本做前向-反向传播，大规模数据集上的计算成本值得关注
- 梯度遍历的三个约束中阈值 δ=0.8 是固定的，自适应调整可能进一步提升效果

## 相关工作与启发
- **vs LESS (Xia et al., 2024)**: LESS 基于梯度相似度独立选择数据，G2IS 引入图结构建模联合分布，在所有设置下超过 LESS
- **vs Sentence-BERT 选择**: 基于语义相似度选择忽略了训练动态信息，G2IS 的梯度表示更能反映对模型学习的贡献
- **vs LIMA (Zhou et al., 2024)**: LIMA 证明了少量高质量数据足够好，G2IS 提供了系统化的数据选择方法论

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 梯度图+遍历算法的组合在数据选择领域是全新范式
- 实验充分度: ⭐⭐⭐⭐⭐ 3个模型、3个数据集、5个基准、多任务优化、充分消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 对指令调优的数据选择有重大实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](../../NeurIPS2025/llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] TableDreamer: Progressive and Weakness-Guided Data Synthesis from Scratch for Table Instruction Tuning](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)

<!-- RELATED:END -->
