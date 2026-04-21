---
title: >-
  [论文解读] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning
description: >-
  [ICCV 2025][机器人][多任务学习] 提出 Rep-MTL，一种基于表示空间任务显著性（task saliency）的多任务优化方法，通过熵正则化保留任务特定学习模式（TSR）和样本级跨任务对比对齐（CSA）来缓解负迁移并显式促进任务互补性，无需修改优化器或网络架构。
tags:
  - ICCV 2025
  - 机器人
  - 多任务学习
  - 任务显著性
  - 表示空间
  - 对比学习
  - 负迁移缓解
---

# Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning

**会议**: ICCV 2025  
**arXiv**: [2507.21049](https://arxiv.org/abs/2507.21049)  
**代码**: 无（项目页面可用）  
**领域**: 机器人  
**关键词**: 多任务学习, 任务显著性, 表示空间, 对比学习, 负迁移缓解

## 一句话总结

提出 Rep-MTL，一种基于表示空间任务显著性（task saliency）的多任务优化方法，通过熵正则化保留任务特定学习模式（TSR）和样本级跨任务对比对齐（CSA）来缓解负迁移并显式促进任务互补性，无需修改优化器或网络架构。

## 研究背景与动机

多任务学习（MTL）通过共享表示来提升效率和泛化性，但不同任务之间的冲突更新会导致负迁移。现有的多任务优化（MTO）方法主要分为两大流派：

**损失缩放方法**（如 UW、DWA、FAMO）：调整任务损失权重

**梯度操控方法**（如 PCGrad、CAGrad、Nash-MTL）：修改共享参数的梯度方向

然而这些方法存在两个根本性问题：

- **效果不稳定**：在需求更高的场景（demanding scenarios）中，许多 MTO 方法甚至无法带来一致的性能增益，大量方法的 $\Delta p_{task}$ 为负值
- **忽视互补性**：现有方法几乎完全聚焦于"解决冲突"，而忽略了任务之间的**互补性信息共享**这一同等重要的方面，将其完全留给了网络架构设计

**核心论点**：共享表示空间才是任务交互真正发生的地方，蕴含着丰富的信息和操作潜力。作者提出可以不修改优化器，而是通过在表示空间中调节任务显著性来同时实现两个目标：（1）保留任务特定模式以缓解负迁移；（2）显式促进跨任务互补性共享。

## 方法详解

### 整体框架

Rep-MTL 作为正则化项添加到标准 MTL 目标中，包含两个互补模块：

1. **TSR（Task-specific Saliency Regulation）**：基于熵的显著性正则化，保持任务特定学习模式的区分度
2. **CSA（Cross-task Saliency Alignment）**：基于对比学习的跨任务对齐，促进互补信息共享

总损失为 $\mathcal{L}_{Rep} = \sum_{t=1}^T \mathcal{L}_t(\theta_s, \theta_t) + \lambda_{tsr}\mathcal{L}_{tsr}(Z) + \lambda_{csa}\mathcal{L}_{csa}(Z)$

### 关键设计

1. **表示级任务显著性（Task Saliency）定义**:

    - 功能：量化不同任务在共享表示空间中的交互方式
    - 核心思路：任务 $\mathcal{T}_t$ 的显著性定义为损失对共享表示 $Z$ 的梯度 $\mathcal{S}_t = \nabla_Z \mathcal{L}_t(\theta_s, \theta_t) \in \mathbb{R}^{B \times C \times H' \times W'}$，衡量每个任务目标对表示变化的敏感程度
    - 设计动机：不同于参数梯度用于直接更新模型，表示级显著性作为动态指示器，可以识别和调节任务间的依赖关系，提供丰富的学习信号

2. **TSR：任务特定显著性正则化**:

    - 功能：通过熵惩罚鼓励每个空间位置保持明确的任务特定学习模式
    - 核心思路：首先对显著性进行通道聚合 $\hat{\mathcal{S}}_t = \frac{1}{|C|}\sum_c \mathcal{S}_{t,b,c,h,w}$，然后归一化为跨任务概率分布 $\mathcal{P}_{i,t} = \frac{|\hat{\mathcal{S}}_{i,t}|}{\sum_{k=1}^T |\hat{\mathcal{S}}_{i,k}|}$，最后最小化熵 $\mathcal{L}_{tsr} = \frac{1}{BH'W'}\sum_i(-\sum_t \mathcal{P}_{i,t}\log\mathcal{P}_{i,t})$
    - 设计动机：高熵分布意味着空间位置对所有任务同等重要（过度共享），低熵意味着该位置对特定任务更关键。惩罚高熵可以保留任务特定的学习模式，从而从源头缓解负迁移，而非事后修补梯度冲突

3. **CSA：跨任务显著性对比对齐**:

    - 功能：在样本维度利用对比学习显式促进跨任务互补性
    - 核心思路：计算显著性亲和矩阵 $\mathcal{M}_t = \mathcal{S}_t\mathcal{S}_t^\top \in \mathbb{R}^{B \times C \times C}$，对每个样本 $b$ 计算跨任务平均锚点 $\hat{\mathcal{A}_b} = \mathcal{A}_b\mathcal{A}_b^\top$。同一样本的不同任务亲和矩阵为正样本对，不同样本为负样本对，采用 InfoNCE 损失 $\mathcal{L}_{csa} = \frac{1}{B}\sum_b -\log\frac{\exp(\text{sim}(z_b^a, z_b^t)/\tau)}{\sum_{k \neq b}\exp(\text{sim}(z_b^a, z_k^a)/\tau)}$
    - 设计动机：MTO领域几乎未探索如何显式促进任务互补性，CSA 通过鼓励同一样本在不同任务下共享一致的特征交互模式来实现这一目标，同时通过批内负样本维持任务区分度

### 损失函数 / 训练策略

- 总损失 = 标准多任务损失 + $\lambda_{tsr} \cdot \mathcal{L}_{tsr}$ + $\lambda_{csa} \cdot \mathcal{L}_{csa}$
- 作为纯正则化方法，不修改优化器（甚至可以配合基本的等权重策略 EW 使用）
- 与现有 MTO 方法正交，可以叠加使用
- 梯度自然流过所有组件，隐式调节模型参数更新

## 实验关键数据

### 主实验

**NYUv2 数据集（3任务，DeepLabV3+）：**

| 方法 | Semseg mIoU↑ | Depth Abs.Err↓ | Normal Mean↓ | $\Delta p_{task}$↑ |
|------|-------------|---------------|-------------|-------------------|
| Single-Task | 53.50 | 0.3926 | 21.99 | 0.00 |
| EW | 53.93 | 0.3825 | 23.57 | -1.78 |
| GLS | 54.59 | 0.3785 | 22.71 | +0.30 |
| Nash-MTL | 53.41 | 0.3867 | 22.57 | -1.01 |
| DB-MTL | 53.92 | 0.3768 | 21.97 | +1.15 |
| **Rep-MTL (EW)** | **54.59** | **0.3750** | **21.91** | **+1.70** |

**Cityscapes 数据集（2任务）：**

| 方法 | Semseg mIoU↑ | Depth Abs.Err↓ | $\Delta p_{task}$↑ |
|------|-------------|---------------|-------------------|
| Single-Task | 69.06 | 0.01282 | 0.00 |
| EW | 68.93 | 0.01315 | -2.05 |
| Rep-MTL (EW) | 最优 | 最优 | 正值 |

### 消融实验

| 配置 | NYUv2 $\Delta p_{task}$↑ | 说明 |
|------|------------------------|------|
| EW (基线) | -1.78 | 标准等权重 |
| + TSR only | 提升 | 仅任务显著性正则化 |
| + CSA only | 提升 | 仅跨任务对比对齐 |
| + TSR + CSA (Rep-MTL) | **+1.70** | 两个模块互补协作 |

### 关键发现

1. **仅用 EW 就能超越大多数 MTO 方法**：Rep-MTL 配合基本的等权重策略，在 NYUv2 上达到 +1.70%，是所有方法中最高的
2. **效率优势**：比 Nash-MTL 快约 26%，比 FairGrad 快约 12%，因为不需要二阶梯度计算
3. **大多数 MTO 方法实际效果为负**：在 NYUv2 上多达 15+ 种方法的 $\Delta p_{task}$ 为负值，说明解决梯度冲突本身可能不是正确方向
4. Power Law 指数分析验证了 Rep-MTL 能同时改善任务特定学习和跨任务共享的质量

## 亮点与洞察

- **思想转变**：从"解决冲突"转向"保持有效训练 + 显式促进互补性"，这是一种范式转换
- **TSR 的熵抑制思路**：用熵来衡量空间位置的任务专属程度并作为正则化目标，简洁而有效
- **CSA 的对比学习设计**：利用同一样本在不同任务下的显著性一致性来促进互补性，是 MTO 研究中首次显式探索这一方向
- **实验的大量负面结果**：论文诚实地展示了15+种方法无法提升性能的事实，对领域有警示作用
- **正则化即 MTO**：不需要修改优化器即可实现有效的多任务优化，降低了使用门槛

## 局限与展望

1. 显著性计算需要对每个任务进行反向传播到表示层，任务数增加时计算开销线性增长
2. $\lambda_{tsr}$ 和 $\lambda_{csa}$ 两个超参数需要调节，虽然论文报告了对超参数不敏感
3. 亲和矩阵 $\mathcal{M}_t \in \mathbb{R}^{B \times C \times C}$ 在通道数 $C$ 很大时可能存在内存问题
4. CSA 的正负样本构建依赖于批大小，小 batch 可能影响对比学习效果
5. 目前限于 HPS（Hard Parameter Sharing）架构，是否适用于 soft sharing 场景未知

## 相关工作与启发

- 与 RotoGrad（旋转特征空间）和 SRDML（正则化任务相似性）相比，Rep-MTL 不仅关注冲突解决，还显式促进互补性共享
- TSR 的熵正则化思路可以推广到其他需要维持特征区分度的场景（如对比学习中的表示坍缩问题）
- CSA 的样本级对齐机制可能对多模态学习（如视觉-语言对齐）提供启发

## 评分

- 新颖性: ⭐⭐⭐⭐ 从表示空间切入MTO是新颖的视角，TSR和CSA的设计思路清晰独特
- 实验充分度: ⭐⭐⭐⭐⭐ 四个benchmark，大量对比方法，Power Law分析等深度验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰，方法推导流畅，实验分析深入
- 价值: ⭐⭐⭐⭐ 实用性强（即插即用正则化），且对MTO社区提出了重要的反思视角

<!-- RELATED:START -->

## 相关论文

- [Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](../../ACL2025/robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [Embodied Representation Alignment with Mirror Neurons](embodied_representation_alignment_with_mirror_neurons.md)

<!-- RELATED:END -->
