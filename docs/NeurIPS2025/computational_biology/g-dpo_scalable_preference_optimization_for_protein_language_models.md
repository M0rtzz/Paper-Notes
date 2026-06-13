---
title: >-
  [论文解读] g-DPO: Scalable Preference Optimization for Protein Language Models
description: >-
  [NeurIPS 2025][计算生物][DPO] 针对蛋白质语言模型（PLM）应用 DPO 时偏好对数量随样本数二次增长导致训练不可扩展的问题，提出 g-DPO 框架：(1) 通过序列空间 union mask 聚类剪枝冗余偏好对，保留局部邻域中信息量更大的比较；(2) 利用共享 union mask 的分组似然摊销，一次前向传播同时计算组内所有序列的 log-likelihood。在三个蛋白质工程任务上，g-DPO 保持与标准 DPO 统计上不可区分的 in silico 和 in vitro 性能，同时实现 1.7-5.4× 的训练加速。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "DPO"
  - "蛋白质语言模型"
  - "偏好优化"
  - "可扩展性"
  - "突变景观"
---

# g-DPO: Scalable Preference Optimization for Protein Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.19474](https://arxiv.org/abs/2510.19474)  
**代码**: 有  
**领域**: LLM对齐 / 蛋白质工程  
**关键词**: DPO, 蛋白质语言模型, 偏好优化, 可扩展性, 突变景观

## 一句话总结
针对蛋白质语言模型（PLM）应用 DPO 时偏好对数量随样本数二次增长导致训练不可扩展的问题，提出 g-DPO 框架：(1) 通过序列空间 union mask 聚类剪枝冗余偏好对，保留局部邻域中信息量更大的比较；(2) 利用共享 union mask 的分组似然摊销，一次前向传播同时计算组内所有序列的 log-likelihood。在三个蛋白质工程任务上，g-DPO 保持与标准 DPO 统计上不可区分的 in silico 和 in vitro 性能，同时实现 1.7-5.4× 的训练加速。

## 研究背景与动机

**领域现状**：蛋白质语言模型（PLM，如 ESM-2）通过自监督学习在大规模序列数据上预训练，已经学到了蛋白质结构和功能的隐含信号。DPO（Direct Preference Optimization）作为一种不需要独立奖励模型的偏好优化方法，自然适用于蛋白质工程——实验数据天然定义了偏好关系（如更高的热稳定性更好、更低的毒性更好），可以直接构建偏好对进行 DPO 训练。

**现有痛点**：蛋白质 DPO 面临一个 NLP 中不存在的可扩展性瓶颈——NLP 中偏好结构是天然的（多个输出来自同一 prompt，标注者选最好的），但蛋白质数据集只有标量标签（如热稳定性数值），需要穷举所有序列对来构建偏好。$n$ 条序列可产生 $O(n^2)$ 个偏好对，对于中等规模数据集（几百条序列）就已经导致训练时间不可接受。现有应对策略包括：(a) 阈值划分（好/坏二分，但丢失序列间的精细排名信息）；(b) 随机采样偏好对（但可能丢弃有价值的训练信号）；(c) 排名采样（保留排序但不解决计算量问题）。

**核心矛盾**：偏好对数量 vs 训练效率——更多的偏好对提供更丰富的训练信号，但计算成本二次增长；随机削减偏好对虽然降低成本，但风险是丢弃在序列空间中真正有信息量的比较。关键洞察是：蛋白质突变的局部性意味着大量偏好对实际上是冗余的。

**本文目标** 让 DPO 可扩展到大规模蛋白质突变景观的偏好优化，同时不牺牲优化质量。具体分解为：(a) 如何识别和剪枝冗余偏好对？(b) 如何减少每个偏好对的计算成本？(c) 如何保证剪枝后的训练效果？

**切入角度**：蛋白质序列的独特结构为优化提供了天然把手——突变通常只改变少数位点，因此同一聚类内的序列共享大部分 token，可以用共享的 union mask 一次前向传播计算多个序列的似然。这在 NLP 中不成立（不同句子几乎完全不同），但在蛋白质中，突变的局部性使得这种似然摊销近似误差极小。

**核心 idea**：利用蛋白质突变的局部性，通过序列空间聚类剪枝冗余偏好对 + union mask 分组摊销似然计算，将 DPO 训练加速 5.4× 且无性能损失。

## 方法详解

### 整体框架
g-DPO 训练分三个阶段：(1) **Evo-tuning**——在野生型蛋白的进化相关序列上无监督微调 PLM（ESM-2-650M），提供进化上下文；(2) **Union mask 聚类**——对实验突变体数据集，基于共享突变位置的贪心凝聚聚类，将序列分组；(3) **分组 DPO 训练**——从每个聚类中均匀采样组（大小 $g$），一次前向传播计算组内所有序列的似然，评估组内所有偏好对的 DPO 损失。

### 关键设计

1. **Union Mask 聚类**：

    - 功能：将蛋白质序列按共享突变位置分组，使得同组序列仅在少数位置不同
    - 核心思路：定义 union mask $M(S) = \{i \in [L] : \exists j,k \text{ s.t. } s_i^{(j)} \neq s_i^{(k)}\}$ 为一组序列中所有差异位置的集合。使用贪心凝聚聚类，合并代价为 $\phi(C_i, C_j) = m(C_i \cup C_j) - m(C_i)$（即合并后 union mask 的增量），优先合并那些不增加太多差异位置的聚类。停止条件为 $\min_{i \neq j} \phi(C_i, C_j) > \tau L$，其中 $\tau$ 控制 union mask 占序列总长的最大比例
    - 设计动机：与 NLP 中不同 prompt 输出完全不同，蛋白质突变体之间只有少数位点差异。利用这种局部性，可以将"序列空间中的近邻"聚在一起——这些序列的偏好比较更有信息量（捕捉非加性突变效应），而距离远的序列间比较信号被大量突变差异淹没

2. **分组似然摊销（Group Sampling）**：

    - 功能：一次前向传播计算组内多个序列的 pseudo log-likelihood
    - 核心思路：对于组内 $g$ 个序列，构造共享的 union mask $D$——将所有差异位置 mask 掉后得到共同输入 $y_{\setminus D}$。一次前向传播得到所有位置的 logits，然后对每个序列分别读取差异位置的 log 概率：$\log p(y_w) \approx \sum_{i \in D} \log p({y_w}_i | y_{\setminus D})$。因为非差异位置的 logits 完全相同（token 一样），所以只需计算差异位置。$g=4$ 时，一次前向传播就得到 $\binom{4}{2}=6$ 个偏好对的信号，而标准 DPO 需要 6 次前向传播
    - 设计动机：这是一种 mean-field 近似——假设在已知共享 token 条件下，差异位置是独立的。当差异位置数远小于序列总长时，近似误差极小。这正是 union mask 聚类确保的——通过控制 $\tau$ 保证组内差异位置比例不超过阈值

3. **两阶段效率增益的协同**：

    - 功能：聚类减少偏好对数量 + 分组减少每个偏好对的成本
    - 核心思路：聚类通过设定 $\tau$ 阈值剪枝跨聚类的偏好对（这些对信息量低），将 $O(n^2)$ 的偏好对数量大幅降低；分组通过 union mask 共享将每个偏好对的前向传播成本从 2 次降为 $2/g$ 次。两者效果相乘
    - 设计动机：单独使用分组会因为 mutation span 过大导致近似崩溃（消融实验证实）；单独使用聚类只减少对数不减少单次成本。只有两者配合才能同时保证质量和效率

### 损失函数 / 训练策略
- 标准 DPO 损失：$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(y_w,y_l)} [\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})]$
- 基础模型：ESM-2-650M，先做 evo-tuning 再做 g-DPO
- 每个 epoch 确保所有序列至少出现在一个组中
- 单卡 NVIDIA A100 训练

## 实验关键数据

### 主实验（三个蛋白质工程任务）

| 数据集 | 功能目标 | 序列数 | 位置覆盖率 | DPO Spearman ρ | g-DPO Spearman ρ | 加速比 |
|--------|---------|--------|-----------|----------------|-------------------|--------|
| Anti-SARS-CoV-2 VHH | 热稳定性 | 462 | 47.1% | ~0.55 | ~0.55 | 1.7× |
| Trastuzumab scFv | 表达量 | 76 | 13.1% | ~0.50 | ~0.50 | - |
| DhaA 卤代烷脱卤酶 | 热稳定性 | 474 | 40.3% | ~0.52 | ~0.52 | 5.4× |

- KS 检验证实 DPO 和 g-DPO 生成序列的预测属性分布统计上不可区分（D statistic < 0.03）
- In vitro 验证（DhaA 热稳定性 + Trastuzumab 表达量）：两种方法生成的序列实验结果分布一致

### 消融实验

| 配置 | Spearman ρ | 训练对数 | 收敛速度 | 说明 |
|------|-----------|---------|---------|------|
| 标准 DPO (g=2) | 基线 | $O(n^2)$ | 基线 | 穷举所有偏好对 |
| 仅聚类 (τ=0.3, g=2) | ≈基线 | 大幅减少 | 加速 | 剪枝冗余对，性能不变 |
| 仅分组 (no cluster, g=4) | 下降 | 不变 | 加速但降质 | mutation span 过大导致似然近似崩溃 |
| 聚类+分组 (τ=0.3, g=4) | ≈基线 | 大幅减少 | **最大加速** | 最佳 trade-off |
| 过度聚类 (τ<0.3, g=2) | 下降 | 过度减少 | 快但降质 | 丢失有价值训练信号 |

### 关键发现
- **聚类阈值 τ≈0.3 是性能保持的临界点**：低于此值后性能显著下降，说明约 70% 的偏好对是冗余的可以安全剪枝
- **分组单独使用不可行**：无聚类时 g=4 直接降低性能，因为跨越大量突变位置的序列共享同一 union mask 会导致似然近似误差过大
- **加速随数据集增大而增大**：VHH L 数据集（1833 序列）的加速比最大，印证了冗余偏好对随 $n$ 增长的理论预期
- **In vitro 结果一致**：DhaA 和 Trastuzumab 的实验验证中，DPO 和 g-DPO 设计的序列在实际测定中无显著差异——加速不以实验效果为代价

## 亮点与洞察
- **利用蛋白质序列的结构先验加速训练**——核心洞察是蛋白质突变的局部性使得大量偏好对冗余，这在 NLP 中不成立。这种"数据结构 → 计算优化"的思路可以推广到任何具有局部结构的序列优化问题（如 DNA 序列设计、化学反应优化）。
- **Union mask 聚类的巧妙设计**——用变异位置的并集大小作为聚类距离度量，直接与似然近似的误差边界挂钩（差异越少，mean-field 近似越准确），使得聚类策略和似然计算策略在理论上耦合。
- **In vitro 验证闭环**——不仅做 in silico 指标（Spearman 相关、KS 检验），还在实际实验室验证了 DhaA 热稳定性和 Trastuzumab 表达量，这在蛋白质 ML 论文中是较高的验证标准。

## 局限与展望
- **实验规模有限**：三个数据集均为中等规模（76-474 序列），虽然理论上加速随数据集增大而增大，但未在大规模数据集（>1000 变体）上充分验证
- **仅适用于掩码语言模型**：当前方法依赖 pseudo log-likelihood（需要逐位置 mask），对自回归 PLM 不直接适用
- **聚类参数 τ 需要调参**：最优 τ 取决于具体数据集的突变景观结构，没有提供自动选择策略
- **未探索排名目标**：当前仍用 pairwise DPO 损失，组内有完整的排名信息但未利用——结合 listwise 排名损失可能进一步提升效率
- **单模态限制**：仅用蛋白质序列，未考虑结构或功能等多模态信息

## 相关工作与启发
- **vs 标准 DPO（Rafailov et al., NeurIPS 2023）**：标准 DPO 为 NLP 设计，假设偏好数据已由人类标注提供。g-DPO 解决的是蛋白质场景中从标量标签构建偏好对时的可扩展性问题——这是 DPO 在生物学应用中的特有挑战
- **vs CtrlProt（Liu et al., AAAI 2025）**：CtrlProt 用 rank-wise 偏好目标提升多目标优化的可控性，但没有解决计算效率问题。g-DPO 聚焦于效率，两者互补——可以将 g-DPO 的聚类策略与 CtrlProt 的多目标框架结合
- **vs Widatalla et al.**：他们发现在排名空间中随机采样偏好对效果基本一致（不同 gap level 差别不大），暗示了冗余性。g-DPO 从序列空间（而非排名空间）切入剪枝，更有实际意义
- 对蛋白质工程中使用 DPO/RLHF 的研究有直接指导——确认了"不是所有偏好对都有价值"，局部邻域内的比较更有信息量

## 评分
- 新颖性: ⭐⭐⭐⭐ 将蛋白质序列的局部突变结构与 DPO 计算优化巧妙结合，union mask 聚类设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 三任务 in silico + 两任务 in vitro 验证 + 充分消融，闭环验证
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，从动机到设计到实验层层递进
- 价值: ⭐⭐⭐⭐ 对蛋白质工程中 DPO 的实际应用有直接推动，加速训练使大规模突变景观优化成为可能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](../../ACL2025/computational_biology/kpo_protein_safety.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2025\] Steering Protein Language Models](../../ICML2025/computational_biology/steering_protein_language_models.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](../../AAAI2026/computational_biology/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)

</div>

<!-- RELATED:END -->
