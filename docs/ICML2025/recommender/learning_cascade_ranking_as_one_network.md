---
title: >-
  [论文解读] LCRON: Learning Cascade Ranking as One Network
description: >-
  [ICML 2025][推荐系统][级联排序] 提出LCRON，将多阶段级联排序系统作为统一网络进行端到端训练：通过可微排序技术构建的端到端代理损失$L_{e2e}$直接优化ground truth items通过整个级联的存活概率下界，辅以从下界紧致度推导出的各阶段辅助损失$L_{single}$驱动阶段间协同，在公开基准和工业广告系统的线上A/B测试中均取得显著提升（广告收入+4.10%，用户转化+1.60%）。
tags:
  - "ICML 2025"
  - "推荐系统"
  - "级联排序"
  - "端到端训练"
  - "可微排序"
  - "存活概率"
  - "代理损失"
  - "多阶段协同"
---

# LCRON: Learning Cascade Ranking as One Network

**会议**: ICML 2025  
**arXiv**: [2503.09492](https://arxiv.org/abs/2503.09492)  
**代码**: 无  
**领域**: 推荐系统 / 级联排序  
**关键词**: 级联排序, 端到端训练, 可微排序, 存活概率, 代理损失, 多阶段协同

## 一句话总结

提出LCRON，将多阶段级联排序系统作为统一网络进行端到端训练：通过可微排序技术构建的端到端代理损失$L_{e2e}$直接优化ground truth items通过整个级联的存活概率下界，辅以从下界紧致度推导出的各阶段辅助损失$L_{single}$驱动阶段间协同，在公开基准和工业广告系统的线上A/B测试中均取得显著提升（广告收入+4.10%，用户转化+1.60%）。

## 研究背景与动机

**领域现状**: 大规模推荐/广告系统普遍采用级联排序架构（Matching→Pre-ranking→Ranking→Mix-ranking），通过多阶段漏斗式过滤逐步筛选。每个阶段使用不同容量的模型，在资源效率和性能间取得平衡。系统的最终目标是使ground truth items（用户真正感兴趣的物品）通过所有阶段被最终选出。

**现有痛点**: 传统方法独立训练各阶段，存在两大核心问题：(1) **目标不对齐**——各阶段使用pointwise/pairwise损失分别优化，比真正的级联目标（协同选出所有相关item）更严格，导致在模型容量有限时效率低下；(2) **缺乏协同学习**——独立训练的阶段无法学到交互模式（如召回模型预先避开排序模型会高估的item），在线serving时的协同全靠运气。

**核心矛盾**: ICC只允许单向交互，RankFlow需要迭代训练（复杂且不稳定），FS-LTR虽用全阶段样本但未对齐全局目标，ARF只优化单阶段且假设下游模型最优。没有任何现有方法同时解决目标对齐和协同学习两大挑战。

**本文目标**: 设计能同时解决目标不对齐和缺乏协同学习的端到端训练范式，使所有阶段作为一个统一网络联合优化。

**切入角度**: 将级联排序的目标形式化为ground truth items通过所有阶段的存活概率最大化问题，利用可微排序技术将离散的top-k选择松弛为连续概率，推导下界后直接优化。

**核心 idea**: 用可微排序的软置换矩阵构建各阶段的top-k选择概率，将级联的联合存活概率分解为各阶段概率之积的下界$\hat{P}_{CS}^{q_2} = \prod_i P_{\mathcal{M}_i}^{q_i}$，直接优化该下界实现端到端对齐，并从下界紧致度推导出辅助损失促进阶段间一致性。

## 方法详解

### 整体框架

以两阶段级联（召回$\mathcal{M}_1$+排序$\mathcal{M}_2$）为例：构建全阶段训练样本（来自各阶段的降采样），计算各阶段的可微top-k选择概率$P_{\mathcal{M}_i}^{q_i}$，通过概率之积的下界构建$L_{e2e}$，再从下界紧致度推导各阶段辅助损失$L_{single}$。总损失通过UWL（Uncertainty-based Weighted Loss）自适应加权，减少超参数。

### 关键设计

1. **端到端代理损失$L_{e2e}$**
    - 功能：直接优化ground truth items通过整个级联的存活概率
    - 核心思路：级联存活概率$P_{CS}^{q_2} = \mathbb{E}_{\pi \sim P_\pi} \frac{P_{\mathcal{M}_2}^{q_2} \odot \pi}{\langle \pi, P_{\mathcal{M}_2}^{q_2} \rangle / \langle \mathbf{1}, P_{\mathcal{M}_2}^{q_2} \rangle}$计算复杂，但可以证明$\hat{P}_{CS}^{q_2} = \prod_i P_{\mathcal{M}_i}^{q_i} \leq P_{CS}^{q_2}$。其中$P_{\mathcal{M}_i}^{q_i} = \frac{\sum_{j=1}^{q_i} (\hat{\mathcal{P}}_{\mathcal{M}_i}^\downarrow)_{j,:}}{\oslash sp(\sum_t (\hat{\mathcal{P}}_{\mathcal{M}_i}^\downarrow)_{t,:})}$，$\hat{\mathcal{P}}$是可微排序产生的软置换矩阵。用$\hat{P}_{CS}^{q_2}$和标签$\mathbf{y}$的交叉熵作为$L_{e2e}$
    - 设计动机：相比各阶段独立的pointwise/pairwise损失，$L_{e2e}$允许模型在容量不足时优先保障关键排序而容忍次要错误；当某阶段给ground truth打低分时，不仅优化该阶段，还鼓励其他阶段补偿，实现双向协同

2. **辅助阶段损失$L_{single}$**
    - 功能：对各阶段提供额外监督，收紧下界的紧致度
    - 核心思路：下界紧致度与各阶段top-k选择的一致性相关——当$\langle \pi, P_{\mathcal{M}_2}^{q_2} \rangle / \langle \mathbf{1}, P_{\mathcal{M}_2}^{q_2} \rangle$越接近1（即$\mathcal{M}_1$选出的items在$\mathcal{M}_2$中得分也高），下界越紧。因此设计$L_{single}$优化每个阶段独立的Recall，迫使各阶段从完整候选集（而非上游过滤后子集）中识别ground truth
    - 设计动机：解决$L_{e2e}$在某阶段存活概率接近0时梯度消失的问题；继承ARF的$L_{Relax}$思路但改进了对软置换矩阵信息的利用

3. **基于可微排序的概率松弛**
    - 功能：将离散的top-k选择操作转化为可微操作
    - 核心思路：利用NeuralSort或SoftSort等可微排序方法生成软置换矩阵$\hat{\mathcal{P}} \in [0,1]^{N \times N}$，其中$(\hat{\mathcal{P}})_{j,k}$表示item $k$被排在位置$j$的软概率。温度$\tau \to 0$时收敛到硬置换。对行归一化保证概率有效性
    - 设计动机：可微排序是将学习目标从pointwise/pairwise提升到listwise recall优化的核心技术基础

### 损失函数 / 训练策略

总损失 $L = L_{e2e} + \sum_i L_{single}^{(i)}$，各项通过UWL (Kendall et al., 2018)形式自适应加权减少超参数。全阶段训练样本按FS-LTR策略构建：从各阶段（$\mathcal{Q}_0, \mathcal{Q}_1, \mathcal{Q}_2, CS_{gt}$）降采样，标签由阶段顺序和阶段内排名共同决定。端到端训练使所有阶段参数通过梯度同时更新。

## 实验关键数据

### 公开基准RecFlow（Streaming评估）

| 方法 | e2e-Recall@50 | e2e-Recall@100 | 训练方式 |
|------|-------------|---------------|---------|
| Independent | 基线 | 基线 | 各阶段独立 |
| ICC | 0.1316 | 0.2253 | 单向融合分数 |
| RankFlow | 0.1373 | 0.2320 | 迭代训练 |
| FS-LTR | 0.1392 | 0.2362 | 全阶段样本+LambdaRank |
| **LCRON** | **0.1429** | **0.2417** | **端到端统一网络** |

### 线上A/B测试（真实广告系统）

| 指标 | 对比FS-LTR改善 |
|------|-------------|
| 广告收入 | **+4.10%** |
| 用户转化数 | **+1.60%** |
| 端到端Recall | 显著提升 |

### 消融实验

| 配置 | e2e-Recall变化 |
|------|-------------|
| 仅$L_{e2e}$ | +2.3%（相比FS-LTR） |
| 仅$L_{single}$ | +1.5% |
| $L_{e2e}$ + $L_{single}$（LCRON） | **+4.2%** |
| 不同可微排序方法 | NeuralSort和SoftSort效果相当 |
| UWL vs 固定权重 | UWL更稳定 |

### 关键发现

- $L_{e2e}$和$L_{single}$是互补的：$L_{e2e}$提供全局优化方向，$L_{single}$在局部存活概率接近0时提供有效梯度
- 双向协同优于ICC的单向交互：LCRON允许任何阶段的改善都能通过梯度影响其他阶段
- 相比RankFlow的迭代训练，LCRON的一次性端到端训练更稳定高效
- 线上A/B测试的4.10%广告收入提升直接证明了工业价值
- Streaming评估（任何day t测试用day 0到t-1训练）的一致优势说明方法的时序泛化性好

## 亮点与洞察

- **存活概率下界的推导数学严谨优雅**：$P_{CS}^{q_2} \geq \prod_i P_{\mathcal{M}_i}^{q_i}$的证明简洁且直觉清晰（归一化因子$\leq 1$），为端到端损失提供了坚实理论基础
- **辅助损失从下界紧致度自然导出**：$L_{single}$不是人为设计的正则项，而是从收紧下界的动机中推导出的，理论和实践的一致性很好
- **线上A/B测试的巨大商业价值**：4.10%的广告收入提升在工业界是非常显著的改善，直接说明了方法的实际影响力
- **解决了级联排序训练的两大核心问题**：目标对齐和协同学习在同一框架下同时得到处理

## 局限与展望

- 端到端训练对内存和计算要求更高——可微排序生成$N \times N$软置换矩阵，$N$大时开销显著
- 本文以两阶段级联为例，虽声称可扩展到更多阶段，但未提供超过两阶段的实验
- 全阶段训练样本的降采样策略（各阶段样本数$n_i$）对性能有影响但未深入分析
- 可微排序的温度$\tau$选择未详细讨论——$\tau$过低可能导致梯度问题
- 未与最新的多目标优化方法（如帕累托优化）结合探讨

## 相关工作与启发

- **ICC (Gallagher et al., 2019)**: 最早尝试联合训练级联，但仅允许单向交互（排序→召回），样本空间有限
- **RankFlow (Qin et al., 2022)**: 迭代训练范式，上游决定下游训练样本+下游知识蒸馏到上游，显著优于ICC但不稳定
- **FS-LTR (Zheng et al., 2024)**: 全阶段样本+LambdaRank，是LCRON的最主要对比基线，LCRON在此基础上增加了e2e目标对齐
- **ARF (Wang et al., 2024)**: 提出基于可微排序的Recall代理损失，但仅优化单阶段且假设下游最优——LCRON扩展到全级联
- 启发：可微排序技术使得从pointwise/pairwise到listwise recall优化的飞跃成为可能，是级联排序下一代训练范式的核心组件

## 评分

- **新颖性**: ⭐⭐⭐⭐（存活概率下界+辅助损失的联合设计是核心贡献，理论推导扎实）
- **实验充分度**: ⭐⭐⭐⭐⭐（公开基准+工业部署+线上A/B测试，消融完整）
- **写作质量**: ⭐⭐⭐⭐（数学推导清晰，但符号较多需要仔细跟读）
- **价值**: ⭐⭐⭐⭐⭐（4.10%广告收入提升的工业验证价值极高，具有直接商业影响力）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ICML 2025\] Not All Explanations for Deep Learning Phenomena Are Equally Valuable](not_all_explanations_for_deep_learning_phenomena_are_equally_valuable.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](../../AAAI2026/recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [\[ICLR 2026\] RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](../../ICLR2026/recommender/rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese.md)

</div>

<!-- RELATED:END -->
