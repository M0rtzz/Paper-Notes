---
title: >-
  [论文解读] ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning
description: >-
  [ICLR 2026][强化学习][Mixture-of-LoRAs] ReMix 发现现有 Mixture-of-LoRAs 模型存在严重的路由权重坍缩问题（即使激活 k>1 个 LoRA，有效 LoRA 数也迅速降到 1），提出用非可学习的常数路由权重确保所有激活 LoRA 平等贡献…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "Mixture-of-LoRAs"
  - "路由权重坍缩"
  - "强化学习路由"
  - "RLOO"
  - "参数高效微调"
---

# ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning

**会议**: ICLR 2026  
**arXiv**: [2603.10160](https://arxiv.org/abs/2603.10160)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: Mixture-of-LoRAs, 路由权重坍缩, 强化学习路由, RLOO, 参数高效微调

## 一句话总结

ReMix 发现现有 Mixture-of-LoRAs 模型存在严重的路由权重坍缩问题（即使激活 k>1 个 LoRA，有效 LoRA 数也迅速降到 1），提出用非可学习的常数路由权重确保所有激活 LoRA 平等贡献，并用 RLOO 强化学习梯度估计器训练路由器，显著优于 SOTA PEFT 方法。

## 研究背景与动机

**领域现状**：LoRA 是最流行的参数高效微调方法，Mixture-of-LoRAs 通过在每层维护多个 LoRA 并用路由器选择子集来扩展模型容量。现有方法（如 MixLoRA、HydraLoRA）使用可学习的路由权重，通过 softmax 计算每个 LoRA 的权重。

**现有痛点**：作者发现现有 Mixture-of-LoRAs 路由器存在一个严重的根本性缺陷——**路由权重坍缩**。即使设定激活 k>1 个 LoRA，在训练过程中，softmax 路由权重会迅速集中到单一 LoRA 上（有效支撑大小 ESS 降到 1），其他 LoRA 的权重趋近于 0。这意味着额外 k-1 个 LoRA 的计算完全被浪费。

**核心矛盾**：可学习路由权重允许端到端训练但天然倾向于不平衡——Theorem 1 证明在 Gaussian 初始化下，有效 LoRA 数以高概率极小（如 8 个 LoRA 中，84% 概率只有 ≤2 个有效）。而且这种不平衡在训练过程中还会加剧。

**本文目标**：(1) 理论和实验揭示路由权重坍缩问题；(2) 设计不会坍缩的路由器；(3) 解决非可学习权重带来的不可微问题。

**切入角度**：根本重新思考路由器设计——放弃可学习权重，改用常数权重确保所有激活 LoRA 平等贡献。由此产生的梯度不可计算问题，重新建模为强化学习问题来解决。

**核心 idea**：用常数路由权重 $\omega$ 消除坍缩（$ESS = k$），用 RLOO 梯度估计器训练路由器做 LoRA 选择，推理时用 top-k 选择（理论证明当路由器训练充分时 top-k 是最优策略）。

## 方法详解

### 整体框架

ReMix 要解决的是 Mixture-of-LoRAs 里"路由器越训越偏、最后只剩一个 LoRA 真正起作用"的坍缩问题。它的做法是把每层的路由器从"给 LoRA 算权重"改成"挑哪几个 LoRA"。具体来说，每层挂 $n$ 个 LoRA 和一个路由器，路由器先用 softmax 算出一个概率分布 $\mathbf{q}^{(l)} = \text{softmax}(\mathbf{P}^{(l)}\mathbf{x}^{(l)})$；训练时按这个分布**无替换地采样** $k$ 个 LoRA，每个被选中的都赋同一个常数权重 $\omega$，前向计算变成 $\mathbf{y}^{(l)} = \mathbf{W}^{(l)}\mathbf{x}^{(l)} + \omega \sum_{j=1}^{k} \mathbf{B}_{i_j}^{(l)}\mathbf{A}_{i_j}^{(l)}\mathbf{x}^{(l)}$。因为"选哪几个"是个离散决策、不可微，路由器的梯度改用 RLOO 估计；而推理时不再随机采样，直接挑概率最高的 top-k 个 LoRA。下面三个设计依次解决"如何不坍缩""如何训不可微的路由器""推理时怎么选"。

### 关键设计

**1. 非可学习常数路由权重：把"分配权重"问题换成"挑子集"问题**

坍缩的根源在于可学习的 softmax 权重天然倾向于把质量堆到单个 LoRA 上。ReMix 干脆不让权重可学：被激活的 LoRA 一律赋固定常数 $\omega$，没被激活的为 0。$\omega$ 可按 LoRA 型取 $2/(kr)$ 或按 rsLoRA 型取 $2/\sqrt{kr}$。这样一来有效支撑大小恒等于激活数，$ESS(\boldsymbol{\pi}^{(l)}) = k$，从机制上堵死了坍缩——既然所有被选中的 LoRA 权重相同，就不存在某一个被边缘化的可能。代价是问题性质变了：原来是"如何给 LoRA 分配权重"的连续优化，现在变成"如何挑出 $k$ 个 LoRA"的离散选择，而离散选择带来了下一个设计要解决的不可微难题。

**2. RLOO 强化学习梯度估计器：给不可微的离散选择一个无偏梯度**

既然路由器现在是在做离散的 LoRA 选择，常规反向传播算不出梯度，ReMix 把它重新建模成强化学习问题：SFT 损失 $\mathcal{L}(\mathfrak{I})$ 当作负奖励，路由分布 $\mathbf{q}^{(l)}$ 当作策略。训练时独立采样 $M$ 个 selection $\mathfrak{J}_1, \ldots, \mathfrak{J}_M$，每个 selection 是所有层 LoRA 选择的组合，然后用 RLOO 估计器算路由器梯度

$$\hat{\mathbf{G}}_{\mathbf{P}^{(l)}} = \frac{1}{M-1}\sum_{m}\left(\mathcal{L}(\mathfrak{I}_m) - \bar{\mathcal{L}}\right)\nabla_{\mathbf{P}^{(l)}}\log Q(\mathfrak{J}_m)$$

其中 $\bar{\mathcal{L}}$ 是所有采样损失的均值，充当 baseline。之所以不用更朴素的 REINFORCE，是因为后者方差太大；RLOO 的关键是用 leave-one-out 的方式取 baseline——直接拿其它采样的平均损失，既无偏又有效降方差，且不必额外训一个 value 网络。

**3. Top-k 推理选择与理论保证：训练靠随机探索，推理靠确定性选择**

随机采样在训练时是必要的，它提供探索；但推理时再随机就是徒增噪声。ReMix 推理时直接取概率最高的 top-k 个 LoRA，并由 Theorem 2 给出理论背书：只要路由器训得够好（最优子集被采中的概率 > 50%），top-k 就保证能恢复出最优子集。直觉很简单——若最优子集 $\mathcal{I}^*$ 整体被采中的概率最高，则 $\mathcal{I}^*$ 里每个 LoRA 的边际概率也最高，逐个挑概率最大的自然就把 $\mathcal{I}^*$ 凑齐了。于是 top-k 成了一个既确定又最优的推理策略。

### 损失函数 / 训练策略

两套参数分开更新：LoRA 自身参数走标准 SFT 梯度 $\nabla_{\mathbf{A},\mathbf{B}}\mathcal{L}(\mathfrak{I})$，路由器参数走上面的 RLOO 估计器。值得一提的是，ReMix 的训练计算量可以通过增大采样数 $M$ 来扩展——多采样几次就能让路由器梯度估得更准、效果更好，这是基线方法没有的旋钮（它们的训练计算量是固定的）。实验用 Llama 3 8B 作基础模型，基于 LLaMA-Factory 训练。

## 实验关键数据

### 主实验

| 方法 | GSM8K | HumanEval Pass@1 | ARC-c | 平均 | 参数量 |
|------|-------|------------------|-------|------|--------|
| LoRA | 59.21 | 26.83 | 83.05 | 56.36 | 0.112B |
| rsLoRA | 62.47 | 28.66 | 82.71 | 57.95 | 0.028B |
| MixLoRA | 61.87 | 28.05 | 82.37 | 57.43 | 0.101B |
| HydraLoRA | 62.47 | 20.12 | 82.71 | 55.10 | 0.084B |
| **ReMix** | **65.66** | **32.93** | **83.73** | **60.77** | **0.070B** |

ReMix 在三个基准上一致超越所有基线，平均提升 2.82 准确率。

### 消融实验

| 配置 | GSM8K 准确率 | 说明 |
|------|-------------|------|
| 完整 ReMix | 最高 | RLOO + top-k |
| 去除 RLOO | 显著下降 | 路由器训练不充分 |
| 去除 top-k（随机采样推理） | 下降 | 引入不必要随机性 |
| Rank-kr LoRA (k=4, r=8) | 59.21 | 单个高秩 LoRA |
| k 个 Rank-r LoRA (ReMix) | 64.22 | 证明激活了多样化子集 |
| 训练计算 M=2→32 | 56.03→58.83 | 持续改善 |

### 关键发现

- **路由权重坍缩**在 MixLoRA 中确实存在且迅速恶化——ESS 从初始 ~4 在 1000 步内降至 1，之后再也不回升
- ReMix (k=4, r=8) 显著优于 Rank-32 LoRA（64.22 vs 59.21），证明 ReMix 确实激活了不同的 LoRA 子集，而非始终选择同一子集
- ReMix 的训练计算量可扩展（$M$ 从 2 到 32 持续改善），这是基线方法所不具备的独特优势
- 10% 的额外训练时间换来 15.97% 的准确率相对提升，效率显著

## 亮点与洞察

- **路由权重坍缩的理论揭示**是本文最重要的贡献之一——Theorem 1 给出了 ESS 的概率上界，将这个普遍存在但被忽视的问题严格化。这个发现对所有使用 softmax 路由的 MoE 架构都有警示意义
- **用 RL 训练路由器**的思路非常优雅——常数权重使路由器"选择"LoRA 而非"加权"LoRA，这恰好是离散决策问题，天然适合 RL。RLOO 的引入同时解决了梯度估计和方差控制
- **可扩展训练计算**是一个独特优势——对于追求极致性能的场景，可以直接增大 $M$ 来提升效果，而不需要改变模型结构

## 局限与展望

- 额外的 $M$ 次前向传播增加了训练成本（虽然每步仅增加 ~10%）
- 理论分析基于 Gaussian 初始化，训练后期的坍缩机制可能更复杂
- 仅在 Llama 3 8B 上验证，更大模型和更多任务的泛化性有待确认
- 常数路由权重是否是唯一解？渐进的权重均衡（如引入 load balancing loss）可能也有效

## 相关工作与启发

- **vs MixLoRA (Li et al., 2024)**: MixLoRA 用标准可学习路由权重，本文证明其会坍缩；ReMix 用常数权重+RL 消除坍缩
- **vs MoE 中的 load balancing**: Switch Transformer 等用 auxiliary loss 平衡专家使用率，但那是跨样本的平衡；本文关注的是单个样本内不同 LoRA 的权重平衡
- **vs VB-LoRA (Li et al., 2024)**: VB-LoRA 用向量量化共享 LoRA 参数，参数效率更高但性能较差；ReMix 在参数效率和性能间取得更好平衡

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 路由权重坍缩的理论发现+RL路由的解决方案，两个贡献都很有洞见
- 实验充分度: ⭐⭐⭐⭐ 三个基准、详细消融、效率和扩展性分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机极强，理论和实验紧密配合
- 价值: ⭐⭐⭐⭐⭐ 对 MoE/MoLoRA 范式有根本性改进，即插即用的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Routing, Cascades, and User Choice for LLMs](routing_cascades_and_user_choice_for_llms.md)
- [\[ICLR 2026\] Trinity: An Evolved LLM Coordinator](trinity_an_evolved_llm_coordinator.md)
- [\[ICLR 2026\] References Improve LLM Alignment in Non-Verifiable Domains](references_improve_llm_alignment_in_non-verifiable_domains.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
