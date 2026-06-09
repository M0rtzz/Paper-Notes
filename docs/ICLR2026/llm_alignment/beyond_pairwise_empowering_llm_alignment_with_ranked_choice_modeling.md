---
title: >-
  [论文解读] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling
description: >-
  [ICLR 2026][LLM对齐][preference optimization] 提出 RCPO 框架，将 LLM 对齐从成对偏好扩展到排名选择（ranked choice）建模，通过 MLE 统一了效用模型（MNL）和排名模型（Mallows-RMJ）…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "preference optimization"
  - "ranked choice"
  - "DPO"
  - "Mallows model"
  - "multinomial logit"
  - "alignment"
---

# Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling

**会议**: ICLR 2026  
**arXiv**: [2510.23631](https://arxiv.org/abs/2510.23631)  
**代码**: 无  
**领域**: LLM对齐 / 偏好优化  
**关键词**: preference optimization, ranked choice, DPO, Mallows model, multinomial logit, alignment

## 一句话总结
提出 RCPO 框架，将 LLM 对齐从成对偏好扩展到排名选择（ranked choice）建模，通过 MLE 统一了效用模型（MNL）和排名模型（Mallows-RMJ），在 single-best 和 top-k 反馈格式下都优于 DPO 及其变体。

## 研究背景与动机
**领域现状**：DPO 及其变体（SimPO, R-DPO, AlphaPO 等）已成为 LLM 对齐的主流方法，但它们都基于成对偏好——即每个 prompt 只比较两个 response（preferred vs dispreferred）。

**现有痛点**：实际标注中，偏好反馈远比成对比较丰富——InstructGPT 收集 K 个 response 的排名后却将其拆分为 $\binom{K}{2}$ 对来训练；学术工作通常只保留最高和最低分的两个。这种"成对压缩"丢失了中间排名信息，可能歪曲原始偏好结构。

**核心矛盾**：标注者提供的是多路比较/排名，但训练算法只能消化成对数据——信息浪费和结构扭曲是相互耦合的问题。

**本文目标**：如何设计一个能直接利用 ranked choice（单选 best、top-k 排名）反馈的对齐框架？

**切入角度**：经济学/运筹学中的离散选择模型（discrete choice models）已有成熟理论来处理多选和排名数据。将 prompt 视为 context、response 视为 item、候选集视为 assortment，LLM 对齐可直接映射为选择模型的 MLE。

**核心 idea**：用选择模型理论统一 LLM 偏好优化，DPO 只是 Bradley-Terry 的特例，还有 MNL 和 Mallows 等更强的选择模型可以用。

## 方法详解

### 整体框架
RCPO 的出发点是把"偏好优化"看成一个离散选择问题：标注者面对一个候选集，从中挑出最好的（single-best）或排出前 k 名（top-k），这本质上就是经济学里的离散选择行为。于是论文把 prompt $x$ 当作 context、候选 response 当作 item、候选集 $S$ 当作 assortment，把整套对齐目标写成选择模型 $g$ 的极大似然：

$$\max_{\pi_\theta} \sum_i \log g\big(\mu_i^k, S_i, \{r_{\pi_\theta}(x_i, y)\}_{y \in S_i}\big),\quad r_{\pi_\theta}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

这里 $\mu_i^k$ 是标注的 top-k 排名，$r_{\pi_\theta}$ 是和 DPO 完全一致的隐式 reward。换掉选择模型 $g$ 就得到不同的对齐算法，DPO 只是其中最简单的一个分支。论文给出两类选择模型：基于基数效用的 MNL，和基于序关系的 Mallows-RMJ。

### 关键设计

**1. MNL（Multinomial Logit）分支：把 Bradley-Terry 从"2 选 1"推广到"多选 1 与 top-k"**

DPO 背后的 Bradley-Terry 模型一次只能比较两个 response，这正是成对压缩的根源。MNL 直接放开候选集大小：在 single-best（discrete）格式下，损失变成对整个候选集做归一化，$-\log\sigma\big(-\log\sum_{y_i \in S \setminus \{y_w\}} \exp(f_\theta(x, y_i, y_w))\big)$，相比 DPO 多出的那一层 logsumexp，就是同时把所有非 preferred response 都纳入对比、而不是只挑一个负样本。在 top-k 格式下，它进一步连乘 k 个阶段的 softmax——每个阶段从尚未选中的剩余候选里挑出下一名，逐级展开整条排名链。当 $|S|=2, k=1$ 时这套退化回 DPO，说明 DPO 确实是该框架的一个特例。

**2. Mallows-RMJ 分支：用纯序关系建模，绕开对精确 reward 数值的依赖**

MNL 仍然吃 reward 的基数大小，对 reward model 的噪声敏感。Mallows-RMJ 改成只看排名：某个 response 被选中的概率正比于 $\phi(x)^{d(y_i, S)}$，其中 $d(y_i, S)$ 是 $y_i$ 在候选集 $S$ 中的相对排名位置，dispersion 参数 $\phi(x)$ 越小、概率越向高名次集中。在 discrete 格式下，它的损失实质是去数有多少个 non-preferred 项的 reward 反超了 preferred 项；在 top-k 格式下则扩展为沿排名链的逐对比较，再补上"未入选项与第 k 名"之间的比较。由于全程只用到 ordinal 信息（谁排在谁前面），它天然对 reward 的数值抖动更鲁棒，这也是后面实验里 Mallows-RMJ 大幅领先的原因。

**3. Sigmoid 平滑：让 rank-based 目标可微、能跑 SGD**

Mallows-RMJ 的损失里含有指示函数 $\mathbb{I}\{\cdot\}$（判断某项 reward 是否超过另一项），它在 0/1 之间跳变、不可微。论文用 sigmoid 把这个硬判断近似成平滑过渡，从而让整个 rank-based 目标对梯度下降友好，可以直接接进现有训练流程。

### 训练策略
在 UltraFeedback 数据集上为每个 prompt 生成多个 response，用 Skywork-Reward-V2 reward model 打分后构建排名，再据此切出 pairwise / single-best / top-k 三种反馈格式来训练，对应不同的选择模型分支。

## 实验关键数据

### 主实验：Llama-3-8B-Instruct

| 方法 | AlpacaEval LC↑ | AlpacaEval WR↑ | Arena-Hard WR↑ | UltraFeedback WR↑ |
|------|---------------|----------------|----------------|-------------------|
| DPO | 41.24 | 40.24 | 32.6 | 62.36 |
| SimPO | 44.15 | 38.84 | 33.5 | 50.17 |
| DPO-AllPairs | 33.02 | 38.47 | 29.6 | 51.95 |
| **Mallows-RMJ-Pairwise** | **39.33** | **48.71** | - | - |
| **MNL-Top-k** | - | - | - | - |

### 多模型验证
RCPO 在 Llama-3-8B, Gemma-2-9B, Mistral-7B 上均一致优于或持平 DPO 和 SimPO。

### 消融实验
- DPO-AllPairs（将排名拆成所有成对）性能反而下降，证实了成对压缩的信息扭曲。
- Mallows-RMJ 在 pairwise 设置下就已超越 DPO，说明 rank-based 模型本身更适合偏好学习。
- Top-k 反馈进一步提升性能，验证了更丰富反馈格式的价值。

### 关键发现
- Mallows-RMJ 系列表现最佳，尤其在 AlpacaEval WR 上大幅领先（+8-10 pp），表明 rank-based 模型对 reward 噪声的鲁棒性是关键优势。
- 梯度分析揭示 Mallows-RMJ 会自适应加权：对 dispersion 低的 prompt 给予更大权重，对 reward 接近的对给予更大权重，实现"难样本挖掘"。
- MNL 的多路扩展（从 2 选 1 到 n 选 1）也带来提升，但不如 Mallows-RMJ 显著。

## 亮点与洞察
- **选择模型理论 → LLM 对齐的桥接**：将运筹学中成熟的离散选择理论引入 LLM 对齐，为设计新的对齐算法提供了系统化的理论框架。DPO/SimPO/R-DPO 等都可视为该框架的特例。
- **Rank-based vs Utility-based 的对比洞察**：Mallows-RMJ 仅用序关系建模，比 MNL（依赖精确 reward 数值）更鲁棒。这一发现对 RLHF 实践有启示——当 reward model 噪声大时，rank-based 方法可能更优。
- **信息效率**：直接用 top-k 排名训练比拆成 $\binom{K}{2}$ 对更高效且效果更好，这对偏好数据收集和标注策略有直接指导意义。

## 局限与展望
- 实验主要在 7-9B 模型上进行，缺少更大模型的验证。
- 排名反馈由 reward model 自动生成，未使用真实人类排名标注——reward model 的系统性偏差可能影响结论的外部有效性。
- Mallows-RMJ 的 dispersion 参数 $\phi(x)$ 用 entropy proxy 估计，准确性未充分验证。
- 论文聚焦 single-best 和 top-k，未探索其他排名模型（如 Plackett-Luce、Thurstone）。

## 相关工作与启发
- **vs DPO (Rafailov et al., 2023)**：DPO = Bradley-Terry + 成对数据，是 RCPO 的特例。RCPO 扩展了偏好格式（多选/排名）和选择模型（MNL/Mallows）两个维度。
- **vs SimPO (Meng et al., 2024)**：SimPO 用 length-normalized log-likelihood 作为 reward，仍限于成对比较。可直接嵌入 RCPO 框架。
- **vs Align Once (MLC)**：MLC 关注跨语言一致性，RCPO 关注偏好反馈的信息效率。两者互补。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将选择模型理论系统性引入 LLM 对齐是新颖的理论贡献
- 实验充分度: ⭐⭐⭐⭐ 3 个模型 × 多基线 × ID/OOD 评估，但仅限 7-9B 规模
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，框架呈现清晰，梯度分析有深度
- 价值: ⭐⭐⭐⭐ 为 LLM 对齐提供了更通用的框架，尤其 Mallows-RMJ 的实践价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[ICLR 2026\] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)
- [\[AAAI 2026\] On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](../../AAAI2026/llm_alignment/on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)
- [\[ACL 2025\] IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](../../ACL2025/llm_alignment/iopo_input_output_preference.md)

</div>

<!-- RELATED:END -->
