---
title: >-
  [论文解读] Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning
description: >-
  [ACL 2025][监督微调] 系统研究了在固定数据标注预算下，如何在监督微调（SFT）和偏好微调（PFT/DPO）两个阶段之间最优分配资源，发现低数据量时纯 SFT 最优，高预算时组合使用效果最佳，且仅将 <10% 预算分配给 SFT 就能解决 DPO 的冷启动问题并带来 15-20% 的数学推理提升。
tags:
  - "ACL 2025"
  - "监督微调"
  - "偏好优化"
  - "数据预算分配"
  - "冷启动问题"
  - "DPO"
---

# Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning

**会议**: ACL 2025  
**arXiv**: [2502.11284](https://arxiv.org/abs/2502.11284)  
**代码**: 无  
**领域**: 其他  
**关键词**: 监督微调、偏好优化、数据预算分配、冷启动问题、DPO

## 一句话总结

系统研究了在固定数据标注预算下，如何在监督微调（SFT）和偏好微调（PFT/DPO）两个阶段之间最优分配资源，发现低数据量时纯 SFT 最优，高预算时组合使用效果最佳，且仅将 <10% 预算分配给 SFT 就能解决 DPO 的冷启动问题并带来 15-20% 的数学推理提升。

## 研究背景与动机

- **领域现状**：大语言模型的后训练（post-training）通常遵循"SFT → PFT"两阶段管道——先通过监督微调让模型学会遵循指令，再通过偏好微调（如 DPO、RLHF）提升输出质量和对齐度。这已成为 LLM 开发的标准范式。
- **现有痛点**：SFT 数据（指令-回复对）和 PFT 数据（偏好对/排名数据）在结构、获取成本和标注难度上有根本差异。在实际开发中，标注预算是有限的——高质量数据标注是 LLM 开发总成本的重要组成部分。但对于如何在两个阶段之间分配这一有限预算，缺乏系统性的研究指导。
- **核心矛盾**：直觉上，SFT 和 PFT 都有各自的优势，但它们的边际收益曲线如何随数据量变化？在什么条件下优先投入 SFT 数据？在什么条件下偏好数据更有价值？这些问题没有清晰的经验回答。
- **本文目标**：通过大规模实验，提供关于 SFT 与 PFT 数据分配的实用指南，帮助研究者和从业者在预算约束下做出最优决策。
- **切入角度**：设计了涵盖四种多样化任务、多种模型规模和不同标注成本设置的系统实验，控制总数据预算不变，仅变化 SFT/PFT 分配比例，直接测量性能变化。
- **核心 idea**：数据预算分配的最优策略取决于数据规模——少量标注时 SFT 占主导，大量标注时偏好数据逐步增加；同时发现直接在基座模型上运行 DPO 存在严重的"冷启动问题"，而仅需极少量的 SFT 数据（<10%）就能有效解决。

## 方法详解

### 整体框架

本文是一项系统性的实证研究，而非提出新方法。核心实验设计为：给定固定数据预算 N，将其分配为 SFT 预算 $N_s$ 和 PFT 预算 $N_p$（$N_s + N_p = N$），遍历不同的分配比例（如 100:0、90:10、70:30、50:50、30:70、10:90、0:100），在多种任务和模型规模下测量最终性能。

### 关键设计

1. **多任务多规模实验设计（Multi-Task Multi-Scale Design）**：实验覆盖四种任务——数学推理（GSM8k）、一般指令遵循、摘要生成和对话生成。模型规模从小到大（如 1B、3B、7B 等），标注成本设置考虑了 SFT 数据和偏好数据的不同单价（偏好数据通常需要生成多个候选并进行人工比较，成本更高）。这种全面的实验矩阵确保了结论的普适性。
2. **冷启动问题分析（Cold Start Problem Analysis）**：发现直接在基座模型上运行 DPO（跳过 SFT 阶段）会导致严重的性能问题，特别是在数学推理任务上。通过分析发现这是由"分布偏移"（distribution shift）导致的——基座模型的输出分布与 DPO 训练数据中的 chosen/rejected 对的分布差异太大，DPO 的对比学习信号无法有效传播。
3. **最小 SFT 预算阈值（Minimum SFT Budget Threshold）**：实验表明仅将总预算的 <10% 分配给 SFT 就能有效解决冷启动问题。这一少量的 SFT 起到了"分布桥接"的作用——将模型的输出分布从基座模型的预训练分布拉近到指令跟随分布，使后续的 DPO 训练能够有效学习。

### 损失函数 / 训练策略

- **SFT 阶段**：标准的因果语言建模损失，在指令-回复对上训练：$\mathcal{L}_{SFT} = -\sum_t \log p_\theta(y_t | x, y_{<t})$
- **PFT 阶段**：DPO（Direct Preference Optimization）损失：$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$
- **参考模型**：DPO 的参考模型为 SFT 阶段训练后的模型（当 SFT 比例为 0 时则为基座模型本身）。
- **超参数**：各实验使用统一的学习率调度和正则化设置，DPO 的 $\beta$ 参数经过调优。

## 实验关键数据

### 主实验

不同数据预算规模下，各 SFT/PFT 分配比例的最优性能对比。

| 总预算 | 最优 SFT:PFT 比例 | 最优任务性能 | 纯 SFT 性能 | 纯 PFT 性能 | 说明 |
|--------|-------------------|------------|------------|------------|------|
| <500 | 100:0 | 基线 | 最优 | 很差 | 低数据量，纯 SFT 最优 |
| 500-1000 | 80:20~70:30 | +5-8% vs 纯SFT | 基线 | 较差 | 开始受益于少量偏好数据 |
| 1000-5000 | 50:50~30:70 | +10-15% vs 纯SFT | 基线 | 追赶中 | 偏好数据比例逐步增加 |
| >5000 | 20:80~10:90 | +15-25% vs 纯SFT | 饱和 | 接近最优 | 更多预算应投入偏好数据 |
| >5000 (无SFT) | 0:100 | 低于10:90 | - | 冷启动问题 | 完全跳过 SFT 有损失 |

### 消融实验

以 GSM8k 数学推理任务为例，展示冷启动问题的影响：

| 配置 | GSM8k 准确率 | 说明 |
|------|------------|------|
| 纯 DPO（0% SFT） | ~25% | 严重冷启动问题 |
| 5% SFT + 95% DPO | ~42% | 少量 SFT 显著改善 |
| 10% SFT + 90% DPO | ~45% | 接近最优 |
| 30% SFT + 70% DPO | ~43% | SFT 过多，偏好数据不足 |
| 100% SFT（0% DPO） | ~35% | 无偏好对齐 |
| 模型规模影响（7B vs 3B） | +8-12% | 大模型更能利用偏好数据 |

### 关键发现

- **低数据量时 SFT 占绝对优势**（<1000 标注样本）：在标注数据极度稀缺时，将所有预算投入 SFT 是最佳策略。偏好数据本身需要足够的样本量才能提供有效的对比信号。
- **高预算时偏好数据比例应逐步增加**：随着预算增长，SFT 的边际收益递减，而偏好微调的收益逐步显现。最优策略是分配更多（70-90%）预算给偏好数据。
- **冷启动问题是真实存在的**：在 GSM8k 等需要逐步推理的任务上，直接在基座模型上运行 DPO 性能暴跌。根因是基座模型不具备"逐步推理"的输出格式，导致 DPO 的参考分布严重偏离。
- **极少量 SFT 即可解决冷启动**：仅 <10% 的 SFT 预算（甚至几十个高质量样本）就能让模型学会基本的指令跟随格式，为后续 DPO 奠定基础，带来 15-20% 的数学推理改善。
- **任务敏感性**：冷启动问题在分析性任务（数学、逻辑）上最严重，在开放域对话任务上较轻微，这与任务对输出格式的依赖程度有关。

## 亮点与洞察

- **极具实践指导价值**：直接回答了"如何分配标注预算"这一工业界和学术界都频繁面临的实际问题，结论清晰可操作。
- **冷启动问题的深入分析**：不仅发现了问题，还追溯到分布偏移这一根因，并给出了极简的解决方案（微量 SFT）。
- **全面的实验设计**：多任务、多规模、多预算水平的交叉实验确保了结论的鲁棒性和普适性。
- **挑战了"SFT 可有可无"的流行观点**：一些研究声称可以直接在基座模型上进行偏好对齐，本文用实证数据表明至少在某些关键任务上，最小量的 SFT 仍然是必要的。

## 局限与展望

- 实验基于 DPO 作为偏好微调方法，其他方法（如 KTO、IPO、ORPO）的表现可能不同。
- 标注成本模型相对简化，未考虑数据质量的异质性（同样数量的标注，质量可能天差地别）。
- 未探索多轮迭代策略（如先 SFT→DPO→再 SFT→再 DPO）的效果。
- 基座模型的选择可能影响最优分配比例——经过大量预训练的强基座模型可能更早受益于偏好数据。
- 未来可以构建数据分配的预测模型，根据任务类型和模型规模自动推荐最优预算比例。

## 相关工作与启发

- **vs Zephyr/Tulu 等训练 recipe**：这些工作展示了 SFT+DPO 管道的有效性，但未系统研究两阶段的预算分配。本文填补了这一空白。
- **vs SPIN/Self-Play 方法**：这些方法尝试用合成数据减少对人工偏好标注的依赖，是从另一个角度减轻预算压力。
- **vs ORPO/SimPO 等统一方法**：这些方法尝试将 SFT 和偏好对齐合并为单一阶段，但本文的发现表明两阶段的分离设计仍有其价值。

## 评分

- 新颖性: ⭐⭐⭐ 研究问题重要但方法论上是系统实证研究而非新方法
- 实验充分度: ⭐⭐⭐⭐⭐ 四任务×多规模×多预算的全面实验矩阵令人印象深刻
- 写作质量: ⭐⭐⭐⭐ 结论表述清晰，实验图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 开发实践极具指导意义，冷启动问题的发现和解决方案直接可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](../../ICML2025/others/near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[ACL 2025\] Entailed Between the Lines: Incorporating Implication into NLI](entailed_between_the_lines_incorporating_implication_into_nli.md)
- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[ACL 2025\] Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding](well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)

</div>

<!-- RELATED:END -->
