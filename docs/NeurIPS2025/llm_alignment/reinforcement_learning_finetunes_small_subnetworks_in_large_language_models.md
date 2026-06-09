---
title: >-
  [论文解读] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models
description: >-
  [NeurIPS 2025][LLM对齐][reinforcement-learning] RL 微调 LLM 时实际上只更新了 5%-30% 的参数（稀疏子网络），且该子网络在不同种子、数据和算法间具有高度一致性，仅微调子网络即可复现完整微调的模型性能甚至参数值。
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "reinforcement-learning"
  - "sparse subnetwork"
  - "parameter update sparsity"
  - "LLM finetuning"
  - "lottery ticket hypothesis"
---

# Reinforcement Learning Finetunes Small Subnetworks in Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.11711](https://arxiv.org/abs/2505.11711)  
**代码**: [GitHub](https://github.com/SagnikMukherjee/sparsity_in_rl)  
**领域**: LLM对齐  
**关键词**: reinforcement-learning, sparse subnetwork, parameter update sparsity, LLM finetuning, lottery ticket hypothesis

## 一句话总结

RL 微调 LLM 时实际上只更新了 5%-30% 的参数（稀疏子网络），且该子网络在不同种子、数据和算法间具有高度一致性，仅微调子网络即可复现完整微调的模型性能甚至参数值。

## 研究背景与动机

- 强化学习（RL）是 LLM 后训练的重要阶段，用于推理能力提升和人类价值对齐
- 主流观点认为 RL 需要大幅修改模型参数来实现目标行为，因此广泛采用全参数微调
- 然而，**RL 真的更新了所有参数吗？** 本文给出了否定的回答
- 这一现象不依赖任何显式稀疏正则化、架构约束或参数高效训练方法，而是自然涌现的
- 现有 LTH（Lottery Ticket Hypothesis）关注性能复现，本文更进一步发现可以复现几乎相同的参数值

## 方法详解

### 整体框架

本文不是提出新算法，而是系统性地分析 RL 微调中的参数更新稀疏性现象。核心流程：

1. **测量更新稀疏性**：对比 RL 微调前后的参数，定义 $\text{sparsity}(\theta^0, \theta^1) = 1 - \|\theta^1 - \theta^0\|_0 / n$
2. **提取子网络**：定义二值掩码 $m \in \{0,1\}^{|\theta|}$，其中 $m_i = 1$ 当 $(\theta_{\text{init}} - \theta_{\text{full}})_i \neq 0$
3. **子网络微调验证**：用掩码 $m \odot \nabla_\theta \mathcal{L}(\theta)$ 限制梯度更新，仅训练子网络
4. **跨条件一致性分析**：比较不同种子、数据、算法下的子网络重叠度

### 关键设计

**参数更新稀疏性的定义与度量**：

- 采用 bfloat16 精度，绝对差 ≤ $10^{-5}$ 视为相等（与 PyTorch 默认容差一致）
- 覆盖 7 种 RL 算法：PPO、GRPO、DPO、ORPO、KTO、SimPO、PRIME
- 覆盖 10 个不同家族的 LLM

**子网络重叠度量**：

$$o_1 = \frac{|\mathcal{I}_1 \cap \mathcal{I}_2|}{|\mathcal{I}_1|}, \quad o_2 = \frac{|\mathcal{I}_1 \cap \mathcal{I}_2|}{|\mathcal{I}_2|}$$

其中 $\mathcal{I}_1, \mathcal{I}_2$ 为两次训练中更新参数的索引集。

**核心猜想（Conjecture 1）**：在相同数据和超参数下，子网络微调得到的 $\theta_{\text{sub}} \approx \theta_{\text{full}}$，即不仅性能一致，参数值也几乎完全相同。

### 更新稀疏性的成因分析

论文系统排查了多个可能因素：

| 因素 | 影响 |
|------|------|
| 梯度裁剪 | 有限影响（有无裁剪稀疏性相近：69.8% vs 68.8%） |
| KL 正则化 | 有限影响（SimPO 去掉 KL 仍然稀疏） |
| SFT 前置 | 非必要（DeepSeek-R1-Zero 无 SFT 仍 86% 稀疏） |
| 训练步数 | 早期影响大，后期趋于收敛 |
| **分布内数据训练** | **主要驱动因素** |

核心发现：**在分布内数据上训练是稀疏性的主因**。在 policy 已分配高概率的序列上计算梯度，参数几乎不需要更新。

## 实验关键数据

### 主实验：RL 更新稀疏性

| 算法 | 模型 | 更新稀疏性 |
|------|------|-----------|
| DPO | Tulu-3-8B | 81.4% |
| DPO | Tulu-3-70B | 95.2% |
| GRPO | DeepSeek-Math-7B | 68.5% |
| GRPO | DeepSeek-R1-Zero | 86.0% |
| KTO | Eurus-7B | 96.0% |
| PPO | Math-Shepherd-Mistral-7B | 80.8% |
| SimPO | Llama-3-8B-SimPO | 86.5% |
| PRIME | Eurus-2-7B | 77.0% |

**所有 RL 微调模型的 68.5%-96.0% 参数保持不变**。对比 SFT 仅有 6%-15% 稀疏性。

### 子网络微调验证

| 任务 | $\theta_{\text{full}}$ | $\theta_{\text{sub}}$ | 差异 |
|------|----------------------|---------------------|------|
| AGIEval LSAT-AR (DPO) | 21.3 | 24.8 | +3.5 |
| AGIEval LSAT-LR (DPO) | 53.1 | 54.7 | +1.6 |
| MMLU Pro Math (DPO) | 50.8 | 51.6 | +0.8 |
| MATH500 Overall (PRIME) | 69.8 | 72.2 | +2.4 |
| MATH500 Lvl5 (PRIME) | 40.3 | 45.5 | +5.2 |

**子网络微调不仅恢复了完整模型的性能，在所有任务上还优于全参微调。** 在容差 $10^{-4}$ 下，$\theta_{\text{full}}$ 和 $\theta_{\text{sub}}$ 100% 相同。

### 消融实验：跨条件子网络重叠

| 变化因素 | 随机基线 | RL 子网络重叠 |
|----------|---------|-------------|
| 不同种子 | 36.7% | 60.5% |
| 不同数据 | 14.6%/36.7% | 26.7%/67.1% |
| 种子+数据+算法 | 23.0%/12.9% | 59.1%/33.2% |

### 关键发现

- 更新矩阵的秩几乎满秩（99.2%-99.8%），说明 RL 更新是"稀疏但满秩"的
- 更新不集中在特定层——几乎所有参数矩阵都有类似的稀疏更新（LayerNorm 除外）
- PRIME 中约 72% 参数从未被更新，8% 有互相抵消的梯度，20% 构成实际子网络
- 分布内 SFT（如 rejection sampling）也能产生稀疏更新（~90% 稀疏），分布外 DPO 则仅有 ~7% 稀疏

## 亮点与洞察

1. **超越 LTH 的发现**：不仅子网络性能可复现，参数值也几乎完全一致——这是比 Lottery Ticket Hypothesis 更强的结论
2. **稀疏但满秩**：与 LoRA 的低秩假设形成鲜明对比，RL 更新选择了一小部分参数但几乎跨越了参数矩阵的全部子空间
3. **分布内训练是关键**：统一解释了为什么 on-policy RL 和 SFT 后的 off-policy RL 都产生稀疏更新
4. **实践意义**：为高效 RL 训练方法提供理论基础——可以只训练子网络来节省计算量
5. **预训练模型中存在可迁移结构**：不同条件下子网络的高重叠度暗示模型本身存在"天然适合 RL 的子结构"

## 局限与展望

- 由 RL 计算成本限制，每次只变化一个因素，可能忽略因素间的交互效应
- 部分实验依赖公开 checkpoint 而非完全控制的训练
- 仅聚焦语言模型，未探索多模态和扩散模型
- 缺乏对早期子网络识别方法的研究——如何在训练初期就发现子网络？
- 未深入理论分析更新稀疏性的数学本质
- 部分反例（如 prolonged RL）存在但未充分探讨

## 相关工作与启发

- **对 LoRA 的挑战**：RL 更新是稀疏但满秩的，这与 LoRA 的低秩假设不同，暗示 LoRA 可能不是 RL 微调的最优策略
- **对高效训练的启发**：如果能在训练早期识别子网络，可以大幅减少 RL 训练的计算开销
- **对 SFT vs RL 的理解**：RL 保留预训练能力优于 SFT，可能正是因为更新了更少的参数
- **跨算法子网络复用**：可以用便宜的算法（如 DPO）识别子网络，用于更昂贵的算法（如 PPO）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统揭示 RL 微调的参数更新稀疏性，发现超越 LTH
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 7 种算法、10 个模型，系统消融各因素
- 写作质量: ⭐⭐⭐⭐ 结构清晰、论证有力，但部分证据依赖公开 checkpoint
- 价值: ⭐⭐⭐⭐⭐ 对 RL 微调本质的深刻洞察，具有重要的理论和实践意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ACL 2025\] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](../../ACL2025/llm_alignment/beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)

</div>

<!-- RELATED:END -->
