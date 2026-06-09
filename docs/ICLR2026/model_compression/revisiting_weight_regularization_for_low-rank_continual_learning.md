---
title: >-
  [论文解读] Revisiting Weight Regularization for Low-Rank Continual Learning
description: >-
  [模型压缩] 在低秩持续学习中重新引入弹性权重巩固（EWC），通过在全维空间估计 Fisher 信息矩阵来正则化共享 LoRA 模块，实现恒定存储开销下的有效遗忘缓解。
tags:
  - "模型压缩"
---

# Revisiting Weight Regularization for Low-Rank Continual Learning

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2602.17559](https://arxiv.org/abs/2602.17559)
- **代码**: [GitHub](https://github.com/yaoyz96/low-rank-cl)
- **领域**: 持续学习 / 模型压缩
- **关键词**: Continual Learning, EWC, LoRA, Fisher Information, Parameter-Efficient Learning

## 一句话总结

在低秩持续学习中重新引入弹性权重巩固（EWC），通过在全维空间估计 Fisher 信息矩阵来正则化共享 LoRA 模块，实现恒定存储开销下的有效遗忘缓解。

## 研究背景与动机

### 问题背景
随着大规模预训练模型（PTM）的兴起，持续学习的范式从从头训练转向持续适配 PTM。参数高效持续学习（PECL）成为主流，通常通过为每个任务分配独立的 LoRA 模块来缓解任务干扰。

### 现有方法的局限

**存储线性增长**：现有低秩 CL 方法（如 InfLoRA、SD-LoRA）为每个任务维护独立的 LoRA 分支，存储开销随任务数线性增长；

**权重正则化被忽视**：EWC 等经典正则化方法在 PTM 时代未被充分探索——直接对 PTM 应用 EWC 需要三倍模型大小的内存来存储旧模型副本和 Fisher 矩阵；

**朴素集成次优**：将 EWC 简单与 LoRA 结合（分别正则化 A 和 B 矩阵）忽略了两者的交互，导致信息失真。

### 核心洞察
通过低秩参数化，可以高效实现 EWC：在全维空间 $\Delta\mathbf{W} = \mathbf{AB}$ 上估计 Fisher 信息，既能准确捕获参数重要性，又保持存储恒定。

## 方法详解

### 整体框架

EWC-LoRA 抛弃"每个任务一组 LoRA"的扩张式做法，全程只维护一组共享的 LoRA 模块和一个对角 Fisher 矩阵。每个新任务在这组共享模块上更新，并用累积的 Fisher 信息把更新方向约束在不破坏旧知识的子空间里，任务训练结束后把低秩更新合并回基础权重，从而在恒定存储下完成持续学习。

### 关键设计

**1. 低秩参数化的问题形式化：把权重更新锁在低秩子空间，为正则化提供恒定大小的载体。** 对于任务 $\mathcal{T}_t$，方法不直接改动庞大的基础权重，而是把这一任务带来的改变限制在一个低秩增量上，即 $\mathbf{W}_t = \mathbf{W}_{t-1} + \Delta\mathbf{W} = \mathbf{W}_{t-1} + \mathbf{AB}$，其中 $\mathbf{A} \in \mathbb{R}^{d_O \times r}$、$\mathbf{B} \in \mathbb{R}^{r \times d_I}$，秩 $r \ll \min(d_I, d_O)$。这样每个任务只需训练 $\mathbf{A}$ 和 $\mathbf{B}$ 两个小矩阵，既继承了 LoRA 的参数高效性，又让后续要正则化的对象始终是同一组共享参数，避免了存储随任务数线性膨胀。

**2. 全维空间 Fisher 正则化：在 $\Delta\mathbf{W}$ 而非 A、B 上度量重要性，保住两者的交互信息。** 朴素地把 EWC 套到 LoRA 上，会分别给 $\mathbf{A}$ 和 $\mathbf{B}$ 各算一份 Fisher 再各自惩罚，但 $\mathbf{A}$、$\mathbf{B}$ 只有乘起来才有物理意义，拆开度量会丢掉两者的耦合、扭曲重要性估计。EWC-LoRA 因此把正则化作用在合成后的全维增量 $\Delta\mathbf{W}=\mathbf{AB}$ 上：

$$
\mathcal{L}_t'(\mathbf{A}, \mathbf{B}) = \mathcal{L}_t(\mathbf{A}, \mathbf{B}) + \frac{\lambda}{2} \text{vec}(\mathbf{AB})^\top \mathbf{F}_{t-1}^{\text{cum}} \text{vec}(\mathbf{AB})
$$

其中 $\mathbf{F}_{t-1}^{\text{cum}}$ 是到上一个任务为止累积的对角 Fisher 矩阵，$\lambda$ 控制正则化强度。论文从理论与实验两方面验证了"分别正则化低秩矩阵是次优的"：

| 策略 | $\bar{A}_{10}$ | 稳定性 | 可塑性 | 额外内存 |
|------|----------------|--------|--------|---------|
| 无 Fisher | 82.99 | 87.56 | **98.86** | 0 GB |
| 预计算 $\mathbf{F}_{\mathbf{W}}$ | 83.87 | 93.15 | 94.74 | 1 GB |
| 分别 $\mathbf{F}_{\mathbf{A}}, \mathbf{F}_{\mathbf{B}}$ | 86.41 | 94.23 | 96.47 | 4 GB |
| **全维 $\mathbf{F}_{\Delta\mathbf{W}}$ (本文)** | **87.91** | **94.45** | 97.99 | 6 GB |

分别度量 $\mathbf{F}_{\mathbf{A}}, \mathbf{F}_{\mathbf{B}}$ 比全维方案低约 1.5%，印证了交互信息不可丢弃。

**3. 借梯度等价免去全维存储的 Fisher 估计：准确度量参数重要性，又不付出三倍模型的代价。** 在任务 $\mathcal{T}_t$ 训练收敛后，方法按经典定义估计对角 Fisher：

$$
F_t^{i,i} = \mathbb{E}_{x \sim \mathcal{D}_t}\left[\mathbb{E}_{y \sim p_{\mathbf{W}_t^*}}\left[\left(\frac{\partial \log p_{\mathbf{W}}(y|x)}{\partial w_i}\bigg|_{\mathbf{W}=\mathbf{W}_t^*}\right)^2\right]\right]
$$

直接对 PTM 做 EWC 之所以代价高昂，是因为它要额外存旧模型副本和全维 Fisher，内存接近三倍模型大小。这里的关键观察是：既然只有低秩增量 $\Delta\mathbf{W}$ 可训练，对 $\mathbf{W}$ 求的梯度与对 $\Delta\mathbf{W}$ 求的梯度在可训练方向上完全等价，因此无需显式构造或存储全维更新就能拿到所需的重要性度量——这正是全维 Fisher 既准确又不爆内存的原因。

### 损失函数 / 训练策略

每个任务的训练与收尾遵循固定流程：先初始化共享 LoRA 分支（$\mathbf{A}$ 零初始化、$\mathbf{B}$ 均匀分布初始化），训练时冻结基础权重 $\mathbf{W}_{t-1}$ 只更新 $\mathbf{A}$、$\mathbf{B}$，并用累积 Fisher $\mathbf{F}_{t-1}^{\text{cum}}$ 对 $\Delta\mathbf{W}=\mathbf{AB}$ 施加上式正则化；任务结束后把更新合并回基础权重 $\mathbf{W}_t = \mathbf{W}_{t-1} + \mathbf{AB}$，估计本任务 Fisher $\mathbf{F}_t$ 并叠加到累积矩阵 $\mathbf{F}_t^{\text{cum}}$，随后即可丢弃任务数据与单任务 Fisher。整个过程始终只携带一组 LoRA 加一个对角 Fisher，存储不随任务数增长。

## 实验

### 主实验结果（视觉任务）

| 方法 | CIFAR-100 $\bar{A}_{10}$ | DomainNet $\bar{A}_{5}$ | ImageNet-R $\bar{A}_{10}$ | ImageNet-A $\bar{A}_{10}$ |
|------|--------------------------|------------------------|--------------------------|--------------------------|
| Finetune | 79.09 | 65.57 | 60.42 | 32.85 |
| L2P | 83.18 | 70.26 | 71.26 | 42.94 |
| CODA-Prompt | 86.31 | 70.58 | 74.05 | 45.36 |
| InfLoRA | 86.34 | 71.01 | 74.41 | 50.75 |
| SD-LoRA | 86.77 | — | — | — |
| **EWC-LoRA (本文)** | **87.91** | **72.13** | **75.20** | **52.48** |

### 消融实验：稳定性-可塑性权衡

| 正则化强度 $\lambda$ | 稳定性 (↑) | 可塑性 (↑) | $\bar{A}_{10}$ (↑) |
|---------------------|-----------|-----------|-------------------|
| 0（无正则化） | 87.56 | 98.86 | 82.99 |
| 10 | 92.13 | 98.12 | 86.42 |
| 100 | 94.45 | 97.99 | 87.91 |
| 1000 | 96.21 | 95.34 | 87.15 |

### 关键发现

1. **EWC-LoRA 平均提升 vanilla LoRA 8.92%**，在 CIFAR-100 上达到最优 87.91%；
2. **存储恒定**：与 InfLoRA 等需要线性增长 LoRA 分支的方法相比，EWC-LoRA 仅维护一组 LoRA + 一个对角 Fisher；
3. **全维 Fisher 关键**：分别正则化 A 和 B 导致 1.5% 精度损失，验证了交互信息的重要性；
4. **灵活的稳定性-可塑性权衡**：通过调节 $\lambda$ 可在整个 Pareto 前沿上自由移动，优于固定工作点的方法；
5. **语言任务同样有效**：在 T5-large 和 LLaMA-3.2-1B 上也验证了方法的通用性。

## 亮点

- 首次系统研究 EWC 在低秩 CL 中的应用，揭示朴素集成的理论缺陷
- 全维 Fisher 估计巧妙利用梯度等价性，无需显式存储全维更新
- 存储开销恒定，不随任务数增长，适合长序列任务场景
- 提供了完整的理论分析（附录 A 中的数学证明）

## 局限性

- Fisher 矩阵估计仍需在全维空间操作，对于超大模型（>10B 参数）内存开销仍然可观
- 对角 Fisher 假设忽略了参数间的相关性，可能低估某些参数的重要性
- 仅在 ViT-B/16 上做了视觉实验，更大骨干网络的效果未知
- 低秩约束 $r$ 的选择对性能有显著影响，但缺乏自动选择机制

## 相关工作

- **EWC**: Kirkpatrick et al. (2017) 提出通过 Fisher 信息矩阵惩罚重要参数的变化
- **LoRA**: Hu et al. (2022) 提出低秩适配方法用于高效微调
- **低秩 CL**: InfLoRA (Liang & Li, 2024)、SD-LoRA (Wu et al., 2025)、O-LoRA (Wang et al., 2023)
- **Prompt 式 CL**: L2P、DualPrompt、CODA-Prompt

## 评分

- 新颖性：⭐⭐⭐⭐ — 经典方法在新范式中的深刻重新审视
- 技术深度：⭐⭐⭐⭐⭐ — 理论证明 + 实验验证 + 系统性分析
- 实验充分度：⭐⭐⭐⭐ — 覆盖视觉与语言任务，多个基准
- 实用价值：⭐⭐⭐⭐ — 恒定存储、即插即用，部署友好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICML 2025\] Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
