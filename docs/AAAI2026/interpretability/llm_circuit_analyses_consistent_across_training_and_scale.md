---
title: >-
  [论文解读] LLM Circuit Analyses Are Consistent Across Training and Scale
description: >-
  [AAAI 2026][可解释性][机械可解释性] 本文首次系统追踪 decoder-only LLM 的内部电路（circuits）在 3000 亿 token 训练过程中和 70M–2.8B 参数规模间的演化，发现虽然具体注意力头会发生更替，但执行的算法保持稳定，且跨规模具有一致性，表明在小模型上做的电路分析可推广到更大模型和更长训练。
tags:
  - "AAAI 2026"
  - "可解释性"
  - "机械可解释性"
  - "电路分析"
  - "训练动态"
  - "模型规模"
  - "注意力头"
---

# LLM Circuit Analyses Are Consistent Across Training and Scale

**会议**: AAAI 2026  
**arXiv**: [2407.10827](https://arxiv.org/abs/2407.10827)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 机械可解释性, 电路分析, 训练动态, 模型规模, 注意力头

## 一句话总结
本文首次系统追踪 decoder-only LLM 的内部电路（circuits）在 3000 亿 token 训练过程中和 70M–2.8B 参数规模间的演化，发现虽然具体注意力头会发生更替，但执行的算法保持稳定，且跨规模具有一致性，表明在小模型上做的电路分析可推广到更大模型和更长训练。

## 研究背景与动机
1. **领域现状**：机械可解释性（Mechanistic Interpretability）近年来发展迅速，研究者通过发现模型内部的"电路"（即执行特定任务的计算子图）来逆向工程神经网络。已有工作在 IOI（间接宾语识别）、Greater-Than（年份比较）等任务上发现了具体的电路结构和关键组件（如 name-mover heads、induction heads）。
2. **现有痛点**：绝大多数电路分析工作只研究预训练结束时的模型快照，但实际部署的 LLM 通常会经历持续训练或微调。现有关于训练动态的可解释性研究集中在编码器模型或玩具模型上，与主流的 decoder-only 架构差异显著，研究结论的可迁移性存疑。
3. **核心矛盾**：如果电路分析的结论只适用于特定训练时刻的特定模型，那么该领域的研究价值将大打折扣——我们需要知道这些分析是否具有时间稳定性和规模可迁移性。
4. **本文要解决什么？**（1）电路中的功能组件在训练过程中何时涌现？是否跨规模一致？（2）当具体的注意力头发生更替时，底层算法是否改变？（3）电路的图级别属性（大小、组成）如何随训练和规模变化？
5. **切入角度**：作者利用 Pythia 模型族这一独特资源——它提供了从 70M 到 12B 参数的多个规模模型，且每个模型都有 154 个训练检查点，覆盖 3000 亿 token 的完整训练过程。这使得系统性的纵向追踪成为可能。
6. **核心idea一句话**：利用 Pythia 模型族跨 3000 亿 token 训练和 70M–2.8B 参数规模系统追踪电路演化，揭示"组件会变但算法不变"的稳定性规律。

## 方法详解

### 整体框架
研究框架分为三个层次的分析：（1）**行为评估 + 组件涌现**：追踪模型在四个任务上的表现何时出现，以及对应的功能组件何时涌现；（2）**算法稳定性分析**：当组件发生更替时，验证底层算法是否改变；（3）**图级别电路分析**：研究电路子图本身（节点集合、大小）如何随训练演化。输入是 Pythia 模型族的全部检查点，输出是关于电路跨时间和规模稳定性的系统性结论。

### 关键设计

1. **高效电路发现方法（EAP-IG）**:
    - 做什么：在每个检查点使用边归因剪裁与积分梯度（Edge Attribution Patching with Integrated Gradients）自动发现电路。
    - 核心思路：EAP-IG 通过梯度近似估计每条边被破坏时对损失的影响，为所有边打分后，贪心搜索能达到全模型性能 80% 以上的最小电路。使用二分搜索确定最优电路大小，搜索范围从 1 条边到模型总边数的 5%。
    - 设计动机：传统的 patching 方法（如逐边激活剪裁）需要的前向传播次数随模型大小增长，对 154 个检查点 × 多个模型规模的设置完全不可行。EAP-IG 在固定次数的前后向传播中完成，使大规模纵向研究成为可能。

2. **功能组件涌现追踪**:
    - 做什么：量化追踪四类关键注意力头组件——induction heads（归纳头）、successor heads（后继头）、copy suppression heads（复制抑制头）、name-mover heads（名称移动头）——在训练过程中的涌现和演化。
    - 核心思路：对每个检查点的电路中的注意力头，使用已建立的功能度量指标（如 copy score、CSPA score、induction score、succession score）打分，然后对电路内所有头的得分求和并跨检查点归一化，得到各组件行为强度的时间序列。
    - 设计动机：只有理解功能组件何时出现，才能解释模型任务能力的涌现时机，并验证跨规模的一致性。

3. **算法稳定性验证（Path Patching）**:
    - 做什么：对 IOI 电路进行深入的三阶段分析——逆向工程最终电路算法、开发量化指标、跨检查点验证算法稳定性。
    - 核心思路：将 IOI 算法分解为三个逻辑步骤：（Step 1）name-mover heads 和 copy suppression heads 直接影响 logit 差异；（Step 2）S-inhibition heads 通过 token 和位置信息引导 name-mover heads 关注正确的名字；（Step 3）induction heads 和 duplicate-token heads 向 S-inhibition heads 提供信息。对每个步骤构建 path patching 指标（目标组件的贡献占比），跨检查点追踪这些指标是否稳定。
    - 设计动机：组件更替不一定意味着算法改变。需要区分"实现细节的波动"和"本质算法的变化"，这对电路分析的可信度至关重要。

4. **图级别电路分析**:
    - 做什么：计算相邻检查点间电路节点集的 Jaccard 相似度（EWMA 平滑），分析电路大小与模型规模的关系。
    - 核心思路：EWMA-Jaccard 相似度 $\hat{x}_t = 0.5 \hat{x}_{t-1} + 0.5 x_t$，衡量电路组成在时间上的稳定性。
    - 关键发现：更大的模型倾向于形成更稳定的电路；电路大小与模型规模正相关（Pearson $r = 0.72$–$0.9$）。

### 研究的四个任务
- **IOI（间接宾语识别）**：输入 "When John and Mary went to the store, John gave a drink to"，模型应输出 Mary 而非 John。度量标准为两个名字的 logit 差值。
- **Gendered-Pronoun（性别代词预测）**：输入 "So Paul is such a good cook, isn't"，模型应偏好 "he" 而非 "she"。使用 logit 差值度量。
- **Greater-Than（年份比较）**：输入 "The war lasted from the year 1732 to the year 17"，模型应输出 ≥32 的年份。使用概率差度量。
- **SVA（主谓一致）**：输入 "The keys on the cabinet"，模型应预测 "are" 而非 "is"。使用概率差度量。

这些任务足够简单以适用于小模型，且已有前人的深入电路分析可供验证。

## 实验关键数据

### 主实验：组件涌现时间一致性

| 组件类型 | 任务 | 涌现时间（token数） | 跨规模一致性 |
|----------|------|---------------------|-------------|
| Induction Heads | IOI, Greater-Than | ~2×10⁹ | 所有规模在相似时间点涌现 |
| Successor Heads | Greater-Than | ~2-5×10⁹ | 跨规模一致，后期强度下降 |
| Name-Mover Heads | IOI | ~2-8×10⁹ | 跨规模一致，高强度 |
| Copy Suppression Heads | IOI | ~2-8×10⁹ | 涌现速度和强度因规模而异 |

### 算法稳定性验证

| 验证指标 | Pythia-160M | Pythia-410M | Pythia-1B | Pythia-2.8B |
|----------|-------------|-------------|-----------|-------------|
| Name-Mover + Copy Suppression 贡献占比 | >70% | >70% | >70% | >70% |
| S-Inhibition → Name-Mover 路径重要性 | >50% | >50% | >50% | >50% |
| Induction/Dup-Token → S-Inhibition 路径重要性 | >50% | 变化 | >50% | >50% |

### 关键发现
- **组件涌现的高度一致性**：所有规模的模型（70M 除外）在相似的 token 数处习得任务能力，且功能组件的涌现时间与任务学习曲线高度吻合，证实是这些组件驱动了能力涌现。
- **算法稳定但组件会更替**：以 Pythia-160M 为例，name-mover head (4,6) 在约 3×10¹⁰ token 处突然失去功能，但其他头接替了这一角色，整体算法指标保持稳定。这种"负载均衡"机制保证了模型行为的连续性。
- **学习速率存在上限**：意外发现更大的模型并不总是学习更快——在某些任务上，超过一定规模后学习速率不再提升，甚至略微下降（如 IOI 任务中 6.9B 和 12B 的学习曲线反而更接近 160M）。
- **电路大小正相关于模型规模**：更大的模型需要更多组件来完成相同任务（Pearson $r$ 最高达 0.9），说明角色在更多头之间分散而非集中。
- **更大模型的电路更稳定**：EWMA-Jaccard 相似度分析显示，Pythia-70m/160m 的电路波动较大，而更大模型的电路在训练过程中变化更为平缓，表明规模带来的稳定性优势。
- **电路逐渐趋近最终状态**：虽然中间检查点的电路与最终电路有明显差异（组件在不断更替），但整体趋势是逐步趋近最终电路结构，说明训练并非随机游走而是有方向性的。

## 亮点与洞察
- **"组件变而算法不变"的核心发现**：这是本文最关键的洞察——即使具体执行某功能的注意力头在训练中发生了更替，模型执行任务的整体算法保持不变。这为电路分析的可靠性和可迁移性提供了坚实的实证基础。这一发现可以类比为"公司员工换了但业务流程不变"。
- **大规模纵向实证设计**：覆盖 154 个检查点 × 多个模型规模 × 4 个任务的系统性实验设计，在机械可解释性领域前所未有。这种方法论本身可以作为未来研究的模板。
- **对小模型研究价值的验证**：如果小模型的电路分析确实可推广到大模型，那么可解释性研究可以大幅降低计算成本——这对整个领域有重大的实际意义。

## 局限性 / 可改进方向
- **任务过于简单**：四个研究任务（IOI、性别代词、年份比较、主谓一致）都是小模型就能解决的简单任务。对于更复杂的任务（如多步推理、代码生成），可能存在更多样的算法解决方案，稳定性结论不一定成立。
- **仅限 Pythia 模型族**：所有模型共享相同的架构和训练设置，无法区分结论是架构通用的还是 Pythia 特有的。对 Llama、GPT 等不同架构的验证是必要的。
- **未涉及 SAE 特征级分析**：作者自己也指出，当前分析是在注意力头级别进行的，但近年来基于 Sparse Autoencoder（SAE）的特征级分析可能揭示更细粒度的规律。
- **电路完整性难以保证**：虽然设置了 80% 忠实度阈值，但无法确保电路捕获了所有相关机制，尤其是 MLP 的贡献可能被低估。

## 相关工作与启发
- **vs Wang et al. (IOI Circuit)**：Wang et al. 在 GPT-2 Small 上通过手工 path patching 发现了完整的 IOI 电路算法。本文验证了 Pythia 模型中存在类似但不完全相同的 IOI 算法（如 copy suppression heads 在 Pythia 中是正贡献而非负贡献），并进一步证明该算法跨训练时间保持稳定。本文的自动化方法（EAP-IG）使大规模分析成为可能，弥补了手工分析无法扩展的不足。
- **vs Olsson et al. (Induction Heads)**：Olsson et al. 发现 induction heads 在约 2B–5B token 处跨规模一致涌现。本文复现了这一发现并将分析扩展到更多组件类型（successor heads、name-mover heads、copy suppression heads），证明"跨规模一致涌现"是一种更普遍的现象，而非 induction heads 独有的特例。
- **vs Prakash et al. (Fine-tuning & Circuits)**：Prakash et al. 研究了微调后电路的变化，但仅限于单一检查点的前后对比。本文将分析扩展到连续 3000 亿 token 的预训练过程，提供了更全面的纵向视角。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次对 decoder-only LLM 进行如此大规模的电路纵向追踪，发现了重要的稳定性规律
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多规模多检查点多任务的系统性实验设计堪称典范
- 写作质量: ⭐⭐⭐⭐ 结构清晰逻辑连贯，但部分指标定义和实验细节需要查阅附录
- 价值: ⭐⭐⭐⭐ 为机械可解释性领域的"可复现性/可迁移性"问题提供了重要实证，但任务简单限制了结论的普适性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](../../NeurIPS2025/interpretability/sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](../../ACL2026/interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](../../ACL2026/interpretability/mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)
- [\[ACL 2026\] HistLens: Mapping Idea Change across Concepts and Corpora](../../ACL2026/interpretability/histlens_mapping_idea_change_across_concepts_and_corpora.md)

</div>

<!-- RELATED:END -->
