---
title: >-
  [论文解读] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation
description: >-
  [ICCV 2025][多模态VLM][稀疏优化] 提出稀疏优化（SO）框架，通过动态稀疏梯度选择和基于重要性的动量剪枝来替代低秩适配方法（如LoRA），在11个数据集上的少样本VLM适配任务中实现了SOTA，同时降低了内存开销。 视觉-语言模型（VLM）如CLIP通过大规模图文对预训练获得了强大的零样本能力…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "稀疏优化"
  - "少样本适配"
  - "CLIP"
  - "低秩分解"
  - "参数高效微调"
---

# Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation

**会议**: ICCV 2025  
**arXiv**: [2504.12436](https://arxiv.org/abs/2504.12436)  
**代码**: [https://github.com/nairouz/SO](https://github.com/nairouz/SO)  
**领域**: 多模态VLM  
**关键词**: 稀疏优化, 少样本适配, CLIP, 低秩分解, 参数高效微调

## 一句话总结
提出稀疏优化（SO）框架，通过动态稀疏梯度选择和基于重要性的动量剪枝来替代低秩适配方法（如LoRA），在11个数据集上的少样本VLM适配任务中实现了SOTA，同时降低了内存开销。

## 研究背景与动机
视觉-语言模型（VLM）如CLIP通过大规模图文对预训练获得了强大的零样本能力，但在少样本场景下适配到新任务仍面临严重的过拟合和灾难性遗忘问题。现有参数高效微调（PET）方法主要分为两条路线：

**适配器/提示学习方法**：通过引入额外可训练参数来适配模型，但容易过拟合且依赖大量超参搜索。

**低秩重参数化方法**（LoRA、DoRA等）：将权重更新分解为低秩矩阵来减少参数量，但存在根本性矛盾——**低秩约束限制了模型表达能力**，而**固定秩值需要在欠拟合和过拟合之间权衡**。

作者通过系统实验揭示了LoRA的核心弊端：
- 测试精度在训练初期上升后急剧下降（严重过拟合）
- 最优秩值在不同数据集间差异巨大，且无法通过验证集调优（少样本场景不存在验证集）
- 由于秩是离散值（最低为1），低秩方法缺乏精细控制少量参数的灵活性

核心idea：**受人脑稀疏激活机制启发，用高稀疏性替代低秩约束**——每次迭代仅动态更新极少量参数（如0.05%），既限制了局部学习能力防止过拟合，又通过支撑集动态变化保持整体模型表达力。

## 方法详解

### 整体框架
SO（Sparse Optimization）是一个基于Adam优化器改造的稀疏优化器。当稀疏率为0时等价于Adam。核心改造在于对梯度和动量同时施加稀疏化，遵循两个关键范式。

### 关键设计
1. **局部稀疏与全局稠密（Local Sparsity & Global Density）**:

    - 功能：每次迭代仅更新参数的一个极小子集（由密度比 $\kappa$ 控制），且该子集动态变化
    - 核心思路：每步保留 $M = \lfloor \kappa d \rfloor$ 个梯度值（$d$ 为总参数量），稀疏支撑集每 $T$ 次迭代刷新一次
    - 设计动机：静态稀疏（固定更新同一批参数）会导致模型仅依赖部分连接，性能下降3.4%（Table 4）。动态选择确保不同参数在训练过程中轮流获得更新机会，兼顾低局部学习能力和高全局表达力

2. **局部随机与全局重要性（Local Randomness & Global Importance）**:

    - 功能：梯度通过**随机选择**稀疏化，一阶动量通过**重要性排序**稀疏化
    - 核心思路：
        - 稀疏梯度：$\tilde{g}_t = \text{Random-}M(g_t)$，随机保留 $M$ 个梯度元素
        - 临时动量：$\mu_t = \beta_1 \tilde{\mu}_{t-1} + (1-\beta_1)\tilde{g}_t$（最多含 $2M$ 个值）
        - 稀疏动量：$\tilde{\mu}_t = \text{Top-}M(\mu_t)$，按幅值保留前 $M$ 个
        - 二阶动量对齐：$\tilde{\nu}_t = \nu_t[\mathcal{I}(\tilde{\mu}_t)]$，使用与一阶动量相同的索引
    - 设计动机：按重要性剪枝梯度会导致模型过度依赖局部高幅值更新，加速过拟合（Table 3中重要性梯度+重要性动量的平均精度仅66.6% vs SO的74.6%）；而动量聚合了整条路径上的长期信息，按重要性保留确保关键参数持续获得更新

3. **参数更新**:

    - 功能：标准Adam式更新，但仅作用于稀疏选中的参数
    - 核心公式：
        - 偏差校正：$\hat{\mu}_t = \tilde{\mu}_t / (1-\beta_1^t)$，$\hat{\nu}_t = \tilde{\nu}_t / (1-\beta_2^t)$
        - 更新规则：$\Theta_{t+1} = \Theta_t - \frac{\eta}{\sqrt{\hat{\nu}_t}+\epsilon}\hat{\mu}_t$

### 损失函数 / 训练策略
- 使用标准交叉熵损失优化类别原型
- 训练至收敛（loss < 0.01）或最大2000次迭代
- 密度比 $\kappa = 0.05\%$，支撑集刷新间隔 $T = 10$
- 所有超参在11个数据集上保持一致，无需逐数据集调优

## 实验关键数据

### 主实验

| 数据集 | 指标 | SO (本文) | ReLoRA (之前最佳) | 提升 |
|--------|------|-----------|-------------------|------|
| 11数据集平均 (1-shot) | Top-1 Acc | **73.8%** | 72.5% | +1.3% |
| 11数据集平均 (2-shot) | Top-1 Acc | **76.6%** | 75.3% | +1.3% |
| 11数据集平均 (4-shot) | Top-1 Acc | **78.9%** | 77.1% | +1.8% |
| Aircraft (1-shot) | Top-1 Acc | **31.5%** | 28.8% | +2.7% |
| EuroSAT (1-shot) | Top-1 Acc | **78.2%** | 73.8% | +4.4% |
| UCF101 (4-shot) | Top-1 Acc | **83.4%** | 80.0% | +3.4% |

### 消融实验

| 配置 | 1-shot平均 | 4-shot平均 | 说明 |
|------|-----------|-----------|------|
| Dense梯度 + Dense动量 (=Adam) | 4.4% | 13.6% | 全更新导致严重过拟合 |
| Sparse梯度 + Dense动量 | 73.6% | 77.2% | 梯度稀疏化是关键 |
| Sparse梯度 + Sparse动量 (SO) | **74.6%** | **80.4%** | 动量稀疏化进一步提升 |
| 重要性梯度 + 重要性动量 | 66.6% | — | 重要性梯度选择导致过拟合 |
| 随机梯度 + 随机动量 | 72.9% | — | 随机动量不如重要性动量 |
| 静态支撑集 | 71.2% | 77.0% | 固定支撑不如动态 |
| 动态支撑集 (SO) | **74.6%** | **80.4%** | 动态刷新效果显著 |
| 无动量 | 67.8% | 76.7% | 动量不可或缺 |

### 关键发现
- Adam在少样本场景下几乎完全失败（1-shot仅3.6%），证实全参数更新的灾难性过拟合
- LoRA的测试精度在训练过程中呈现剧烈震荡，最优秩和迭代次数高度数据集相关
- SO在0.05%密度比下仍能恢复到接近全参数训练的表达能力，因为不同参数在不同时间步被激活
- 随机梯度选择 + 重要性动量剪枝的组合是最优的，验证了"局部随机+全局重要"范式

## 亮点与洞察
1. **思路新颖**：跳出了低秩适配的主流范式，从稀疏性角度解决少样本过拟合，类比人脑的稀疏激活机制
2. **极简设计**：仅引入两个超参（密度比 $\kappa$ 和刷新间隔 $T$），且对超参不敏感，无需逐数据集调优
3. **理论直觉清晰**：将稀疏化分解为"梯度选什么"和"动量保留什么"两个独立问题，通过随机/重要性的正交设计实现去耦
4. **内存高效**：梯度和动量均仅维护 $M$ 个值（远小于全参数量），实际内存占用低于LoRA

## 局限与展望
1. 仅在CLIP (ViT-B/16)上验证，未在更大规模VLM或LLM上测试（作者在结论中也提到这一点）
2. 密度比 $\kappa$ 的最优值可能随模型规模变化，需要进一步研究自适应密度调节策略
3. 支撑集刷新间隔 $T$ 虽然对结果不太敏感，但理论上可以设计为自适应机制
4. 未探索与提示学习方法的结合——SO优化器是否可以用于优化prompt参数？

## 相关工作与启发
- **与LoRA的关系**：LoRA通过低秩分解减少参数，SO通过稀疏性减少每步更新的参数——两者都是约束更新空间的策略，但稀疏性具有动态变化的优势
- **与GaLore的关系**：GaLore在低秩子空间中投影梯度，SO直接对梯度随机采样——后者更简单且效果更好
- **启发**：该方法启示我们在资源受限场景下，**动态性比固定结构更重要**——与Lottery Ticket假说的"赢家票"思想形成互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 从稀疏性角度挑战低秩主流，两个范式设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 11个数据集、7个基线、详尽消融实验
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，范式总结到位，图表示意性强
- 价值: ⭐⭐⭐⭐ 为少样本微调提供了新范式，但大模型场景验证缺失

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](../../CVPR2025/multimodal_vlm/rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[ICML 2025\] Vision Graph Prompting via Semantic Low-Rank Decomposition](../../ICML2025/multimodal_vlm/vision_graph_prompting_via_semantic_low-rank_decomposition.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](../../CVPR2025/multimodal_vlm/improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
