---
title: >-
  [论文解读] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning
description: >-
  [ACL 2025][LoRA] 提出 MoRE (Mixture of Low-Rank Experts)，将 LoRA 中的不同秩视为不同专家，通过自适应秩选择器为每个任务动态选择最合适的秩，配合对比学习优化的任务嵌入和平衡数据采样策略，使用单个 LoRA 模块实现高效的多任务微调。
tags:
  - "ACL 2025"
  - "LoRA"
  - "多任务学习"
  - "混合专家"
  - "自适应秩选择"
  - "参数高效微调"
---

# MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning

**会议**: ACL 2025  
**arXiv**: [2505.22694](https://arxiv.org/abs/2505.22694)  
**代码**: [GitHub](https://github.com/NLPfreshman0/MoRE)  
**领域**: 其他  
**关键词**: LoRA, 多任务学习, 混合专家, 自适应秩选择, 参数高效微调

## 一句话总结

提出 MoRE (Mixture of Low-Rank Experts)，将 LoRA 中的不同秩视为不同专家，通过自适应秩选择器为每个任务动态选择最合适的秩，配合对比学习优化的任务嵌入和平衡数据采样策略，使用单个 LoRA 模块实现高效的多任务微调。

## 研究背景与动机

LoRA 作为最主流的参数高效微调方法，在单任务场景中表现优异，但在多任务场景中面临核心矛盾：

**不同任务需要不同秩**：作者通过实验发现（Table 1），T5-base 在 MRPC 上最优秩为 r=1，而 CoLA 上最优秩为 r=8，RTE 上为 r=4。固定秩无法适配所有任务。

**搜索最优秩代价高**：为每个任务独立搜索 + 训练多个 LoRA 在计算和存储上都不经济。

**现有方案的不足**：
   - DyLoRA / AdaLoRA / SoRA 等动态秩方法仅面向单任务，未考虑任务间的联系与区别
   - 并行 LoRA 方案（MultiLoRA、MixLoRA）增加了参数量，违背了 LoRA 减少参数的初衷
   - Prompt Tuning 方法需要两阶段训练，效率不高

核心问题：**如何在多任务场景中实现高效的 LLM 微调？** 关键在于用一个 LoRA 模块同时服务多个任务，并自动分配合适的秩。

## 方法详解

### 整体框架

MoRE 在 Transformer 的 attention 和 FFN 层引入改进的 LoRA 模块，核心流程：
1. 为每个任务分配一个可学习的任务嵌入
2. 通过自适应秩选择器，根据任务嵌入选择合适的秩（即"秩专家"）
3. 使用选定秩截断 LoRA 矩阵进行前向计算
4. 通过对比学习优化任务嵌入质量，通过平衡采样稳定训练

### 关键设计

1. **低秩专家 (Low-Rank Experts)**：

    - 功能：将 LoRA 中的每个秩 $r_i \in [1, r]$ 视为一个独立的"专家"
    - 核心思路：不同秩的专家通过 LoRA 矩阵的重叠部分共享信息——秩 $r_i$ 和 $r_j$ 的专家共享前 $\min(r_i, r_j)$ 行/列的参数
    - 设计动机：相似的任务应该选择相近的秩（共享更多参数），不同的任务选择差异大的秩（保持个性化）

2. **自适应秩选择器 (Adaptive Rank Selector)**：

    - 功能：根据任务嵌入 $\mathbf{e}_t$ 输出秩的概率分布 $\mathbf{p}_t = \text{softmax}(\mathbf{W}_g \mathbf{e}_t + \mathbf{b}_g)$
    - 选择最大概率对应的秩：$r_t = \arg\max \mathbf{p}_t$
    - STE (Straight-Through Estimator) 解决 argmax 不可微分问题
    - 线性缩放：$\frac{r_t}{|T|}$ 平衡不同秩专家的更新频率（低秩部分被更多任务共享，更新更频繁，需要降低学习率）

3. **任务嵌入与对比学习优化**：

    - 功能：为每个任务分配可学习的嵌入向量 $\mathbf{e}_t$
    - 核心思路：用对比学习确保同任务样本的表示与对应任务嵌入接近，与其他任务嵌入远离
    - 损失函数：$\mathcal{L}_{con} = \frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{e}_t)/\tau)}{\sum_{k=1}^{T}\exp(\text{sim}(\mathbf{h}_i, \mathbf{e}_k)/\tau)}$

4. **平衡数据采样 (Balanced Data Sampling)**：

    - 功能：为每个数据集分配采样权重 $\phi_t = \exp(|\mathcal{D}_t| / \sum_i |\mathcal{D}_i|)$
    - 设计动机：GLUE 中 MNLI 有 392K 样本而 RTE 仅 2.5K，不平衡采样会导致小数据集欠拟合

### 损失函数 / 训练策略

总损失函数：$\mathcal{L} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{con}$
- $\mathcal{L}_{gen}$：交叉熵生成损失
- $\mathcal{L}_{con}$：对比学习损失
- $\lambda = 0.1$，温度 $\tau = 0.05$
- 优化器：AdamW，学习率 $3 \times 10^{-4}$，线性衰减 + 500 步预热

## 实验关键数据

### 主实验—GLUE 基准（表格）

| 方法 | 参数/任务 | MNLI | SST-2 | MRPC | RTE | CoLA | AVG |
|------|-----------|------|-------|------|-----|------|-----|
| LoRA (r=8) | 0.39M | 85.8 | 93.2 | 89.9 | 76.3 | 62.8 | 85.1 |
| MultiLoRA | 1.56M | 85.9 | 94.5 | 88.2 | 80.6 | 66.9 | 86.0 |
| MOELoRA | 0.78M | 86.3 | 94.2 | 89.7 | 81.3 | 68.4 | 86.7 |
| **MoRE** | **0.78M** | **86.2** | **93.7** | **91.2** | **83.5** | **69.9** | **87.3** |

LLaMA2-7B 上：

| 方法 | 参数/任务 | AVG |
|------|-----------|-----|
| LLaMA2-LoRA | 2.5M | 87.8 |
| LLaMA2-MOELoRA | 5M | 87.3 |
| **LLaMA2-MoRE** | **5M** | **88.8** |

### 常识推理实验（表格）

| 方法 | BoolQ | PIQA | OBQA | ARC-E | ARC-C | AVG |
|------|-------|------|------|-------|-------|-----|
| LoRA | 80.9 | 77.7 | 79.0 | 83.7 | 76.9 | 79.6 |
| MixLoRA | 84.3 | 79.5 | 82.6 | 86.8 | 76.3 | 81.9 |
| **MoRE** | **87.2** | **82.3** | **83.0** | **86.7** | **74.2** | **82.7** |

### 消融实验（表格）

| 条件 | GLUE AVG |
|------|----------|
| MoRE (完整) | 87.3 |
| w/o 线性缩放 | 87.0 (-0.3) |
| w/o 任务嵌入 | 86.1 (-1.2) |
| w/o 对比学习优化 | 86.3 (-1.0) |
| w/o STE | 86.4 (-0.9) |
| w/ 子集专家 | 86.2 (-1.1) |
| w/ 随机采样 | 86.2 (-1.1) |

### 关键发现

1. MoRE 以与 MOELoRA 相同的参数量（0.78M/任务）取得更高的平均分（87.3 vs 86.7），证明"秩即专家"比"多 LoRA 并行"更高效
2. 任务嵌入和对比学习是最重要的组件——移除后分别下降 1.2 和 1.0
3. 可视化显示：相似任务（如 MRPC 和 STS-B）被分配相近的秩，不同任务（如 CoLA）被分配不同的秩
4. Few-shot 迁移实验中 MoRE 在 4-shot SciTail 上达到 83.8%，超过所有基线

## 亮点与洞察

1. **秩即专家**：这个概念简洁优雅——不需要多个独立的 LoRA 模块，仅通过截断同一个 LoRA 的不同行/列就实现了专家分化。这保证了参数共享的天然性
2. **推理零开销**：与传统 MoE 不同，MoRE 在推理时只需使用选定秩的子矩阵，没有额外的路由开销或并行 LoRA 的额外参数
3. **对比学习妙用**：在无监督信号的情况下用对比学习让任务嵌入自动学会区分任务特征，是一个很实用的技巧

## 局限与展望

1. **秩的上限受限**：最大秩 r 是预设的，如果某任务需要的秩超过 r，框架无法满足
2. **任务定义依赖**：需要明确知道输入属于哪个任务才能查找任务嵌入，不适用于任务边界模糊的场景
3. **评估规模有限**：主要在 T5-base 和 LLaMA2-7B 上实验，未在更大规模模型（13B+）上验证
4. **动态秩实际分布**：论文未充分分析不同层中秩选择的差异——是否不同层的秩分配模式不同？

## 相关工作与启发

- DyLoRA 动态训练所有秩但仅针对单任务，MoRE 将这个思路扩展到多任务并加入了任务感知的选择机制
- 与 MoE 的关系：MoRE 借鉴了 MoE 的思想但专家不是独立模块而是同一矩阵的不同截断，更参数高效
- 线性缩放策略 $r_t/|T|$ 与 LoRA 原始论文中的 $\alpha/r$ 缩放有异曲同工之妙

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "秩即专家"是一个自然但此前未被提出的视角，将 LoRA 内部结构与 MoE 统一的想法很有创意
- **实验充分度**: ⭐⭐⭐⭐ — GLUE、常识推理、Few-shot 迁移、消融实验全面，两个骨干模型验证
- **写作质量**: ⭐⭐⭐⭐ — 动机用 Table 1 直观展示不同任务需要不同秩，方法描述清晰
- **价值**: ⭐⭐⭐⭐ — 实用性强，推理零额外开销，代码已开源，可直接用于多任务微调场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](../../ECCV2024/others/dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ACL 2025\] Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)

</div>

<!-- RELATED:END -->
