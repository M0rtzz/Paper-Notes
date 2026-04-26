---
title: >-
  [论文解读] TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree
description: >-
  [ICML 2025][模型压缩][持续学习] 本文提出 TreeLoRA，通过构建基于梯度相似度的层级 K-D 树来组织历史任务的 LoRA 适配器，利用 Lower Confidence Bound (LCB) 多臂老虎机算法高效搜索最相关的任务分支实现知识共享，配合稀疏梯度更新，在 ViT 上实现 3.2× 加速、LLM 上 2.4× 加速，同时维持或超越 SOTA 性能。
tags:
  - ICML 2025
  - 模型压缩
  - 持续学习
  - LoRA
  - 层级树结构
  - 梯度相似度
  - 多臂老虎机
---

# TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree

**会议**: ICML 2025  
**arXiv**: [2506.10355](https://arxiv.org/abs/2506.10355)  
**代码**: https://github.com/ZinYY/TreeLoRA  
**领域**: 模型压缩 / 持续学习 / 参数高效微调  
**关键词**: 持续学习, LoRA, 层级树结构, 梯度相似度, 多臂老虎机

## 一句话总结

本文提出 TreeLoRA，通过构建基于梯度相似度的层级 K-D 树来组织历史任务的 LoRA 适配器，利用 Lower Confidence Bound (LCB) 多臂老虎机算法高效搜索最相关的任务分支实现知识共享，配合稀疏梯度更新，在 ViT 上实现 3.2× 加速、LLM 上 2.4× 加速，同时维持或超越 SOTA 性能。

## 研究背景与动机

**领域现状**：持续学习（CL）领域的三大范式——正则化方法（EWC）、回放方法（GEM）和架构方法（HiDeLoRA）——在大预训练模型（LPM）时代都面临效率问题。现有方法的计算复杂度通常随任务数线性增长。

**现有痛点**：大模型时代 CL 的核心挑战是效率。现有 CL 方法要么计算复杂度 $O(N)$ 随任务数增长（需要回顾所有历史任务的梯度），要么无法充分利用任务间的共享知识来加速适应。HiDeLoRA 等分层方法虽然性能好，但训练时间仍然较长。

**核心矛盾**：任务相似性的估计需要计算所有历史任务的梯度（$O(N)$ 复杂度），但高效 CL 又必须避免这种线性复杂度；直接为每个新任务独立训练 LoRA（SeqLoRA）则会导致灾难性遗忘。

**本文目标** 如何在不遍历所有历史任务的前提下，高效地找到与当前任务最相似的历史任务并复用其知识？

**切入角度**：作者观察到任务之间存在层级化的相似性结构——浅层特征跨任务共享、深层特征任务特定。将这种结构显式建模为 K-D 树，就可以用树搜索代替线性搜索，将复杂度从 $O(\sqrt{N})$ 降到 $O(\sqrt{\log N})$。

**核心 idea**：用层级梯度相似度树 + 多臂老虎机算法，将持续学习中任务相似性搜索的复杂度从线性降为对数级。

## 方法详解

### 整体框架

TreeLoRA 维护一棵动态增长的 K-D 树，根节点代表所有任务，叶节点对应单个任务的 LoRA 适配器。当新任务到达时，使用 LCB 老虎机算法沿树搜索找到梯度方向最相似的分支，加载对应的 LoRA 适配器作为初始化，然后通过稀疏梯度更新适应新任务。任务完成后，将新学到的适配器插入树中并更新结构。

### 关键设计

1. **梯度相似度树结构（Gradient-Similarity Tree）**:

    - 功能：将历史任务按梯度方向的相似性组织成层级树
    - 核心思路：用 L1 范数度量两个任务梯度方向的差异，阈值 $\delta$ 内的任务被归入同一组。树的浅层节点对应适配器捕获底层任务共享特征，深层节点捕获高层任务特定语义。阈值 $\delta$ 不需要手动调参，而是像 K-D 树一样在构建时自动用中位数切分：$\mathcal{N}_j = \max\{\mathcal{N} \subseteq [N]: \|\mathbf{g}_i - \mathbf{g}_{i'}\|_1 \leq \delta, \forall i,i' \in \mathcal{N}\}$
    - 设计动机：Transformer 模型本身就有层级结构（浅层通用、深层特定），树结构天然对齐了模型的层级特性，比扁平的任务列表更能捕获任务间的多粒度关系

2. **LCB 老虎机搜索（Lower Confidence Bound Bandit）**:

    - 功能：在不计算所有历史梯度的前提下，高效找到最相关的任务分支
    - 核心思路：将选择哪个历史任务来估计相似度建模为多臂老虎机问题。每个训练样本到来时，只选一个历史任务计算梯度差异。叶节点的 LCB 为 $\text{LCB}_k = \hat{\mu}_k - 2\sqrt{\frac{\log t}{n_k}}$，非叶节点则取子节点 LCB 的最优值减去 $\delta$。选择 LCB 最小的分支进行探索
    - 设计动机：将搜索复杂度从遍历所有 $N$ 个任务降为沿树搜索的 $O(1)$；理论上 regret 从 $O(\sqrt{N})$ 降到 $O(\sqrt{\log N})$

3. **稀疏梯度更新（Sparse Gradient Updates）**:

    - 功能：只更新与新任务最相关的参数子集
    - 核心思路：选定最相关分支后，用自适应正则项 $\ell_{\text{reg}}^t = \|\hat{\mathbf{g}}_n^t - \hat{\mathbf{g}}_k^t\|_1$ 来识别需要更新的参数。更新公式为 $\mathbf{w}_n^{t+1} = \mathbf{w}_n^t - \alpha \cdot \mathcal{S}[\nabla\ell] - \lambda \cdot \nabla\ell_{\text{reg}}^t$，其中 $\mathcal{S}$ 是低秩稀疏化函数（实际用 LoRA 适配器实现）
    - 设计动机：稀疏更新既降低了计算和存储开销（只需维护低秩适配器），又通过只修改相关参数来缓解灾难性遗忘

### 损失函数 / 训练策略

总损失 = 任务损失 + $\lambda$ × 梯度差异正则项。ViT 上树深度设为 5，LLM 上设为 64。每个任务结束后存储任务特定 LoRA 适配器，通过深度优先搜索将其插入最近分支。推理时使用最近学到的适配器。

## 实验关键数据

### 主实验（ViT）

| 数据集 | 方法 | 准确率 (%) ↑ | 遗忘率 BWT (%) ↓ | 训练时间 (s) ↓ |
|--------|------|-------------|-----------------|--------------|
| Split CIFAR-100 | HiDeLoRA | 88.46±0.04 | 4.33±0.41 | 692±7 |
| Split CIFAR-100 | TreeLoRA | **88.54±0.05** | 4.37±0.15 | **214±4** |
| Split ImageNet-R | HiDeLoRA | **72.28±0.15** | 4.16±0.05 | 796±9 |
| Split ImageNet-R | TreeLoRA | 71.94±0.47 | 4.06±0.40 | **260±5** |
| Split CUB-200 | HiDeLoRA | 73.48±1.35 | 5.38±0.21 | 194±3 |
| Split CUB-200 | TreeLoRA | **73.66±0.22** | **4.87±0.30** | **86±3** |

### 主实验（LLM, TRACE 数据集）

| 模型 | 方法 | OP (%) ↑ | BWT (%) ↓ | 训练时间 (s) ↓ |
|------|------|---------|----------|--------------|
| Mistral-7B | HiDeLoRA | 51.81±0.9 | 6.25±0.3 | 1288±28 |
| Mistral-7B | TreeLoRA | **54.77±1.1** | **3.77±0.4** | **510±20** |
| LLaMA-2-7B | HiDeLoRA | 41.60±0.8 | 7.12±0.4 | 1286±27 |
| LLaMA-2-7B | TreeLoRA | **43.52±1.0** | **3.46±0.4** | **485±18** |
| Gemma-2B | O-LoRA | **33.73±0.8** | 12.36±0.4 | 1302±29 |
| Gemma-2B | TreeLoRA | 33.41±0.9 | **8.50±0.5** | **510±15** |

### 消融实验

| 配置 | 准确率 (%) | BWT (%) | 训练时间 (s) | 说明 |
|------|-----------|---------|-------------|------|
| TreeLoRA (2 epochs) | 87.73±0.12 | 4.13±0.25 | 23 | 极少训练即有效 |
| TreeLoRA (10 epochs) | 88.23±0.16 | 4.29±0.11 | 108 | 中等训练 |
| TreeLoRA (20 epochs) | 88.54±0.05 | 4.37±0.15 | 214 | 完整训练 |

### 关键发现

- TreeLoRA 在 ViT 上实现 3.2× 加速（CUB-200: 194s→86s），LLM 上 2.4× 加速（Mistral: 1288s→510s），同时性能持平或更优
- 仅 2 个 epoch 训练即可达到 87.73% 准确率（vs 20 epoch 的 88.54%），说明任务共享知识的有效复用
- 在 LLM 场景下遗忘率大幅降低（Mistral: 6.25%→3.77%），说明树结构有效管理了任务间的知识冲突
- SeqLoRA（为每个任务独立训练 LoRA）灾难性遗忘严重（97.62% BWT on CIFAR-100），验证了知识共享的必要性

## 亮点与洞察

- **树结构对齐 Transformer 层级**：树的浅层 LoRA 捕获任务共享特征、深层捕获任务特定特征，与 Transformer 本身浅层通用/深层特定的特性天然对齐；这个设计思路可以迁移到其他涉及层级知识管理的场景
- **老虎机理论保证**：将任务搜索建模为 bandit 问题并在 smooth tree 假设下证明了 regret 从 $O(\sqrt{N})$ 到 $O(\sqrt{\log N})$ 的改进，是少有的在持续学习中给出理论保证的工作
- **自动阈值确定**：借鉴 K-D 树用中位数自动切分的思路，避免了梯度相似度阈值的超参数调优

## 局限与展望

- 树深度（ViT=5, LLM=64）作为超参数需要手动设定，且不能超过 Transformer 层数，在极深模型上可能需要更灵活的深度自适应机制
- 在 Gemma-2B 上 OP 略低于 O-LoRA（33.41 vs 33.73），说明对小模型的适用性可能有限
- 论文假设任务按序到达且任务边界已知（task-incremental setting），在更具挑战性的 class-incremental 或 domain-incremental 设置下需要进一步验证
- 当前仅用 LoRA 实现稀疏更新，是否可以推广到其他 PEFT 方法（如 AdaLoRA、Prefix-Tuning）值得探索

## 相关工作与启发

- **vs HiDeLoRA/HiDePrompt**: 同属层级分解方法，但 HiDe 系列的分解是预设的固定层级，TreeLoRA 的树是动态构建的，能自适应任务流的结构
- **vs O-LoRA**: O-LoRA 使用正交子空间分配来避免遗忘，TreeLoRA 通过梯度相似度聚类实现类似目标但额外获得了加速优势
- **vs GEM**: GEM 靠存储历史数据来约束梯度方向，计算开销大（CIFAR-100: 22456s vs TreeLoRA 214s）；TreeLoRA 不需要存储数据，仅维护轻量 LoRA 适配器

## 评分

- 新颖性: ⭐⭐⭐⭐ 树结构+老虎机的组合在持续学习中是新颖的，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 ViT 和 LLM、三个视觉数据集+NLP TRACE 数据集、三个不同 LLM
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，理论和实验逻辑链完整
- 价值: ⭐⭐⭐⭐ 在大模型持续微调这个实际需求强烈的方向上同时解决了效率和性能问题

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[CVPR 2026\] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](../../CVPR2026/model_compression/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [\[CVPR 2025\] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](../../CVPR2025/model_compression/lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)

<!-- RELATED:END -->
