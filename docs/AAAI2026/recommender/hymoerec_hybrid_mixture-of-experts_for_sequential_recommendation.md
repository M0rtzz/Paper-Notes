---
title: >-
  [论文解读] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation
description: >-
  [AAAI 2026 (Student Abstract)][序列推荐] 本文提出 HyMoERec，一个结合共享专家和特化专家分支的混合专家架构，通过自适应专家融合机制替代传统序列推荐模型中的统一前馈网络，以捕捉用户行为模式的异质性和物品复杂度的多样性，在 MovieLens-1M 和 Beauty 数据集上一致超越 SOTA。
tags:
  - AAAI 2026 (Student Abstract)
  - 序列推荐
  - 混合专家
  - 用户行为建模
  - 前馈网络
  - 自适应融合
---

# HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation

**会议**: AAAI 2026 (Student Abstract)  
**arXiv**: [2511.06388](https://arxiv.org/abs/2511.06388)  
**代码**: 无  
**领域**: 推荐系统  
**关键词**: 序列推荐, 混合专家, 用户行为建模, 前馈网络, 自适应融合

## 一句话总结

本文提出 HyMoERec，一个结合共享专家和特化专家分支的混合专家架构，通过自适应专家融合机制替代传统序列推荐模型中的统一前馈网络，以捕捉用户行为模式的异质性和物品复杂度的多样性，在 MovieLens-1M 和 Beauty 数据集上一致超越 SOTA。

## 研究背景与动机

**领域现状**：序列推荐（sequential recommendation）旨在根据用户的历史交互序列预测其下一个感兴趣的物品。基于 Transformer 的方法（如 SASRec、BERT4Rec）在此任务上取得了显著进展，其核心组件包括自注意力层和逐位置前馈网络（Position-wise FFN）。

**现有痛点**：现有模型中的 FFN 层对所有用户交互和所有物品使用相同的参数，本质上是一种"一视同仁"的处理。但实际中：（1）用户行为模式高度异质——有些用户偏好明确且稳定，有些则多变且探索性强；（2）物品复杂度多样——热门物品与长尾物品的表征需求不同。统一的 FFN 无法同时适应这种多样性。

**核心矛盾**：模型容量有限 vs 用户/物品多样性丰富——单一 FFN 的表达能力不足以捕捉所有类型的行为模式和物品特征。

**本文目标**：设计一个能自适应处理不同类型用户行为和物品的序列推荐架构。

**切入角度**：用混合专家（MoE）替代统一 FFN，不同专家专门处理不同的行为模式，同时保留共享专家捕捉通用模式。

**核心 idea**：结合共享专家分支和特化专家分支的混合 MoE 架构，通过门控网络自适应选择和融合专家输出，为不同用户行为和物品提供定制化的特征变换。

## 方法详解

### 整体框架

HyMoERec 基于标准的 Transformer 序列推荐架构，将其中的逐位置 FFN 替换为混合 MoE 模块。输入为用户的物品交互序列，经过 embedding、自注意力层和 MoE 层后输出下一物品的预测分布。

### 关键设计

1. **混合专家架构（Hybrid MoE）**:

    - 功能：同时捕捉通用行为模式和特化行为模式。
    - 核心思路：MoE 层由两部分组成：（a）共享专家分支——所有输入都经过这些专家，学习跨用户、跨物品的通用特征变换；（b）特化专家分支——门控网络根据输入动态选择 top-k 个特化专家，每个特化专家学习处理特定类型的行为模式。最终输出是共享分支和选中的特化分支输出的加权融合。
    - 设计动机：纯共享的 FFN 缺乏适应性，纯 MoE 可能训练不稳定（专家可能坍缩或负载不均）。混合设计通过共享分支保证基础性能和训练稳定性，通过特化分支提供额外的适应能力。

2. **自适应专家融合机制**:

    - 功能：动态决定每个输入应该使用哪些专家以及如何融合输出。
    - 核心思路：门控网络将输入 embedding 映射到专家权重分布，使用 softmax 归一化后选择 top-k 专家。融合权重由门控网络输出决定，使得相似的行为模式被路由到相同的专家，不同的模式使用不同专家。
    - 设计动机：固定路由无法适应输入的多样性。自适应路由使得模型可以"按需"分配计算资源——简单输入可能只需共享专家，复杂输入则需要特化专家的额外处理。

3. **训练稳定化策略**:

    - 功能：防止 MoE 训练中的专家坍缩和负载不均问题。
    - 核心思路：引入负载均衡辅助损失，惩罚专家使用频率的方差，确保所有专家都被均匀使用。同时，共享专家分支提供了稳定的梯度流，防止整体训练崩溃。
    - 设计动机：MoE 的经典问题是"赢者通吃"——少数专家被过度使用而其他专家退化。均衡策略是 MoE 实用化的关键。

### 损失函数 / 训练策略

标准的序列推荐训练目标（如交叉熵或 BPR 损失）+ 负载均衡辅助损失。

## 实验关键数据

### 主实验

| 数据集 | 指标 | HyMoERec | SOTA基线 | 提升 |
|--------|------|----------|----------|------|
| MovieLens-1M | HR@10/NDCG@10 | 最佳 | SASRec等 | 一致超越 |
| Beauty | HR@10/NDCG@10 | 最佳 | SASRec等 | 一致超越 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| 混合MoE (Full) | 最佳 | 共享+特化专家 |
| 仅共享专家 | 接近标准FFN | 无特化能力 |
| 仅特化专家 | 不稳定 | 缺乏共享基础 |
| 标准FFN | 基线 | 无专家分支 |

### 关键发现

- 混合 MoE 相比标准 FFN 一致提升了推荐性能，验证了用户行为异质性处理的价值。
- 共享分支和特化分支都不可或缺——去掉任何一个都会导致性能下降或训练不稳定。
- 模型在长尾用户（交互少）和活跃用户（交互多）上都有效，说明 MoE 确实在适应不同行为模式。

## 亮点与洞察

- **将 MoE 引入序列推荐 FFN 层**的思路直接且有效，是对现有架构的自然增强。
- **混合设计（共享+特化）的平衡**是实用 MoE 系统的关键经验——纯 MoE 往往不如混合版本稳定。

## 局限与展望

- 作为 Student Abstract，实验规模和分析深度有限。
- 未分析不同专家实际学到了什么样的行为模式。
- 可以结合用户画像信息来辅助专家路由。

## 相关工作与启发

- **vs SASRec/BERT4Rec**: 这些方法使用统一 FFN，HyMoERec 用混合 MoE 替代以增强适应性。
- **vs Switch Transformer 的 MoE**: Switch 在 NLP 中验证了 MoE 的有效性，HyMoERec 将思路迁移到推荐领域。

## 评分

- 新颖性: ⭐⭐⭐ 混合MoE用于推荐较新但技术组件已有
- 实验充分度: ⭐⭐⭐ Student Abstract篇幅限制
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰
- 价值: ⭐⭐⭐ 对序列推荐有改进意义

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)
- [\[AAAI 2026\] Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents](bid_farewell_to_seesaw_towards_accurate_long-tail_session-based_recommendation_v.md)
- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](../../ACL2025/recommender/bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)

<!-- RELATED:END -->
