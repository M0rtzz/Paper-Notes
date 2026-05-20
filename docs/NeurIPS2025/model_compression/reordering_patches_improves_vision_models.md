---
title: >-
  [论文解读] REOrdering Patches Improves Vision Models
description: >-
  [NeurIPS 2025][模型压缩][patch排列顺序] 揭示了视觉模型中 patch 排列顺序对长序列模型性能有显著影响，并提出 REOrder 框架通过信息论先验和强化学习自动发现最优 patch 排列，在 ImageNet-1K 上提升高达 3.01%，在 FMoW 上提升 13.35%。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "patch排列顺序"
  - "长序列模型"
  - "Mamba"
  - "强化学习"
  - "Plackett-Luce"
---

# REOrdering Patches Improves Vision Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.23751](https://arxiv.org/abs/2505.23751)  
**代码**: [项目页面](https://arxiv.org/abs/2505.23751)  
**领域**: 模型压缩  
**关键词**: patch排列顺序, 长序列模型, Mamba, 强化学习, Plackett-Luce

## 一句话总结

揭示了视觉模型中 patch 排列顺序对长序列模型性能有显著影响，并提出 REOrder 框架通过信息论先验和强化学习自动发现最优 patch 排列，在 ImageNet-1K 上提升高达 3.01%，在 FMoW 上提升 13.35%。

## 研究背景与动机

视觉 Transformer 和其他序列模型需要将 2D 图像展平为 1D patch 序列。传统做法是使用行优先（raster-scan）顺序，基于以下假设：完整自注意力是排列等变的（permutation-equivariant），因此 patch 顺序无关紧要。

然而，这个假设在现代长序列模型中**不成立**：

- **Transformer-XL** 引入了基于分段循环和相对位置编码的记忆机制，相对索引 $i-j$ 使其对排列敏感。
- **Longformer** 使用滑动窗口局部注意力 + 全局标记，固定的局部掩码在排列下不保持等变性。
- **Mamba** 采用内容依赖的状态空间更新（线性递归），逐 token 的因果扫描天然依赖输入顺序。

这些架构通过破坏完整注意力的排列等变性来换取 $O(n)$ 或 $O(nw)$ 的复杂度，但代价是输出对 patch 顺序敏感。作者实验发现，仅仅将行优先改为列优先或希尔伯特曲线就能带来显著精度变化。更关键的是，**没有单一固定排列在所有模型和数据集上最优**，因此需要一种数据驱动的方法来发现最优排列。

## 方法详解

### 整体框架

REOrder 是一个两阶段框架：

1. **信息论先验发现**：通过评估不同 patch 序列的可压缩性，筛选出可能有效的初始排列。
2. **强化学习优化**：用 Plackett-Luce 排列策略模型 + REINFORCE 算法在组合空间中搜索最优排列。

### 关键设计

1. **排列等变性的理论分析**: 作者严格证明了完整自注意力满足 $\text{Attn}(\mathbf{P}\mathbf{X}) = \mathbf{P}\,\text{Attn}(\mathbf{X})$，并逐一分析 Transformer-XL、Longformer、Mamba 为何违反此性质。这为"patch 顺序很重要"提供了理论基础。核心洞察是：*高效注意力近似在降低复杂度的同时引入了对输入排列的敏感性*。

2. **信息论先验（压缩率分析）**: 作者将图像 patch 离散化为 VQ-VAE token，然后用 unigram/bigram 分词后用 LZMA 压缩。发现行优先和希尔伯特曲线的压缩率高（局部冗余大），而列优先和螺旋排列压缩率低。压缩率与下游精度呈弱的模型依赖相关——低压缩率并不总是好的，但可作为初始化的弱先验。

3. **Plackett-Luce 排列策略**: 用参数向量 $\mathbf{z} \in \mathbb{R}^n$（$n$ 为 patch 数）参数化排列分布。采样通过 Gumbel-top-$k$ 技巧并行化：$\pi = \text{argsort}_{\text{desc}}(\mathbf{z} + \tau \mathbf{g})$，其中 $g_i \sim \text{Gumbel}(0,1)$。对 196 个 patch 只需 196 个参数，开销可忽略。对数概率有解析闭式：$\log P(\pi|\mathbf{z}) = \sum_{i=1}^n [z_{\pi_i} - \log\sum_{k=i}^n \exp(z_{\pi_k})]$，通过反向 cumulative logsumexp 高效计算。

### 损失函数 / 训练策略

采用三阶段课程学习：
- **前 $N$ 个 epoch**：用标准行优先顺序训练分类器，建立稳定的表征基础。
- **第 $N$ 到 $N+M$ 个 epoch**：激活 PL 策略，用 REINFORCE 训练。温度 $\tau$ 从 0 线性升到 0.2 再降回 0，鼓励探索后收敛。
- **$N+M$ 之后**：策略冻结为确定性最大似然排列 $\hat{\pi} = \text{argsort}(\mathbf{z})$，仅训练骨干网络。

奖励定义为交叉熵损失的负值 $r = -\mathcal{L}_{\text{CE}}$，使用动量基线 $b_{t+1} = \beta b_t + (1-\beta)r_t$（$\beta=0.99$）减小方差。总损失为 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{policy}}$。

## 实验关键数据

### 主实验 — 固定排列对比

| 模型 | 数据集 | 行优先 | 最优固定排列 | 精度变化范围 |
|------|--------|--------|-------------|-------------|
| ViT | IN-1K | 37.5% | ~37.5% | ±0%（等变） |
| T-XL | IN-1K | 基线 | 列优先: +1.92% | -6.43% ~ +1.92% |
| Longformer | IN-1K | 基线 | 列优先: +1.83% | -0.5% ~ +1.83% |
| Mamba | IN-1K | 基线 | — | 对非行/列排列普遍下降 |
| T-XL | FMoW | 基线 | 列优先最优 | 变化较小 |

### 主实验 — REOrder 提升

| 模型 | 数据集 | REOrder 最佳提升 | 提升来源排列 |
|------|--------|-----------------|-------------|
| Mamba | IN-1K | +3.01% (±0.23) | 希尔伯特曲线 |
| Mamba | FMoW | +13.35% (±0.21) | 对角线 |
| T-XL | IN-1K | +1.50% (±0.21) | 希尔伯特曲线 |
| T-XL | FMoW | +1.10% (±0.21) | 列优先 |
| Longformer | IN-1K/FMoW | ~0% | 无显著提升 |

### 消融实验

| 配置 | 关键观察 | 说明 |
|------|----------|------|
| 随机排列 vs REOrder | REOrder 显著优于随机 | 排除了训练动态偶然因素 |
| 100 vs 300 epoch | 提升持续存在 | 非训练不足导致的偶然结果 |
| 有/无信息论初始化 | 初始化有帮助但非必需 | 弱先验提供合理起点 |
| Longformer | 几乎无提升 | 其近似接近完整注意力，排列敏感性最低 |

### 关键发现

- Mamba 对排列最敏感，其固定的四方向因果扫描与非标准排列冲突，但 REOrder 能通过学习找到更好的排列。
- 不同数据集有不同的最优排列（ImageNet 偏好列优先，FMoW 偏好对角线），说明无通用最优排列。
- 策略训练过程中，重要 patch（如键盘键）倾向于被排到序列末尾，反映了数据集的中心偏差。

## 亮点与洞察

- **深刻的理论洞察**：将"patch 顺序无关"这一广泛接受的假设精确限定为仅适用于完整自注意力，对所有高效注意力近似均不成立。
- **极低开销**：PL 策略仅 196 个参数（对于 14×14 patch），训练时只需一次 argsort 操作。
- **FMoW 上 13.35% 的提升令人瞩目**：暗示卫星图像等非自然图像可能有更大的排列优化空间。

## 局限与展望

- Longformer 上无显著提升，说明方法主要适用于排列敏感性强的架构。
- 当前使用全局共享排列（每 batch 一个），未探索样本自适应排列的可能性。
- 仅在分类任务上验证，检测/分割等密集预测任务中的效果未知。
- Mamba 的四方向扫描方向固定，理想情况下扫描方向应与排列联合优化。

## 相关工作与启发

- 与 ARM（Mamba 视觉变体）选择小尺度行内聚类的做法相比，REOrder 在更大的排列空间中搜索，效果更好。
- 这项工作对任何将 2D 结构数据转为 1D 序列的场景（如点云、图结构）都有启示。
- 可与 token pruning/merging 等效率方法结合：先排列优化再剪枝。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统揭示并解决 patch 排列对长序列视觉模型的影响
- 实验充分度: ⭐⭐⭐⭐ 四种模型、两个数据集、六种排列、多种消融全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析严谨，行文流畅，图表设计精美
- 价值: ⭐⭐⭐⭐ 对长序列视觉模型的社区有即时价值，但需扩展到更多任务和架构

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](../../ICCV2025/model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](../../ECCV2024/model_compression/isomorphic_pruning_for_vision_models.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](../../ICML2026/model_compression/dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)

</div>

<!-- RELATED:END -->
