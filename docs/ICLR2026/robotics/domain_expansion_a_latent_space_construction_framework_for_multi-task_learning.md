---
title: >-
  [论文解读] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning
description: >-
  [ICLR 2026][机器人][多任务学习] 提出 Domain Expansion 框架，通过正交池化(Orthogonal Pooling)将潜在空间重构为互相正交的子空间，从结构上防止多目标训练中的梯度冲突与表征崩塌，实现可解释、可组合的概念代数。
tags:
  - "ICLR 2026"
  - "机器人"
  - "多任务学习"
  - "正交池化"
  - "潜在空间构造"
  - "表征崩塌"
  - "可组合表征"
---

# Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning

**会议**: ICLR 2026  
**arXiv**: [2601.20069](https://arxiv.org/abs/2601.20069)  
**代码**: 待确认  
**领域**: Representation Learning / Multi-Task Learning  
**关键词**: 多任务学习, 正交池化, 潜在空间构造, 表征崩塌, 可组合表征  

## 一句话总结
提出 Domain Expansion 框架，通过正交池化(Orthogonal Pooling)将潜在空间重构为互相正交的子空间，从结构上防止多目标训练中的梯度冲突与表征崩塌，实现可解释、可组合的概念代数。

## 研究背景与动机

**领域现状**：多任务学习（MTL）旨在用单一网络同时满足多个学习目标（如分类+回归），但竞争目标产生的冲突梯度会将共享表征拉向相反方向，导致表征退化。作者将此问题形式化为"潜在表征崩塌"（latent representation collapse）——特征空间被压缩到一个对所有目标都不优的折中小区域。

**现有痛点**：(a) 梯度级 MTL 方法（GradNorm、PCGrad、Nash-MTL、CAGrad、MGDA 等）本质上是"反应式"的——在冲突梯度已经产生之后再行调解，每步都要计算额外的梯度操作；(b) 这些方法不改变潜在空间本身的结构，学到的表征仍然纠缠、不可解释。一个典型案例：Objective Set 2 下，Nash-MTL 等基线在分类准确率很高但 V-score 接近 0——说明模型学到了"捷径"映射而非有意义的内部表征。

**核心矛盾**：如何设计一个表征空间，使得多个学习目标在学习过程中天然不干扰——而非在干扰发生后再调解？

**本文目标**：从表征空间的结构设计层面消除任务间干扰，构建一个内在支持多目标的"主动式"潜在空间。

**切入角度**：类比变形艺术（如圆柱上的图案从不同角度看到不同形状），一个高维潜在向量可以通过不同正交方向的投影同时编码多个独立概念。

**核心 idea**：用特征分解的正交基将潜在空间分割为互不干扰的概念子空间，梯度在子空间内流动、跨子空间为零。

## 方法详解

### 整体框架
Domain Expansion 是一个在训练过程中动态执行的三步框架，每个 epoch 更新一次正交基：

### 关键设计

1. **发现主轴（Find Principal Axes）**:

    - 对当前 epoch 的潜在特征计算经验均值 $\mu$ 和协方差矩阵 $\Sigma$
    - 对 $\Sigma$ 做特征分解得到正交特征向量基 $V = [v_0, v_1, \ldots, v_{D-1}]$
    - 各 epoch 间用 Hungarian 算法对齐特征向量，解决训练早期不稳定问题

2. **定义正交域（Define Orthogonal Domain）**:

    - 选取前 $M$ 个最大特征值对应的特征向量构成"域" $V_M$
    - 每个特征向量 $v_m$ 唯一分配给一个目标概念 $\mathcal{C}_m$（如方位角、类别、ID）
    - 投影算子：$\text{Proj}_m = v_m v_m^\top$

3. **正交池化（Orthogonal Pooling）**:

    - 将潜在特征 $f$ 投影到各正交子空间：$f^{\text{proj},m} = \text{Proj}_m(f - \mu)$
    - 各子空间的损失梯度天然解耦——概念 A 的学习无法影响概念 B 的子空间
    - 总损失 = 各子空间上独立损失的加权和：$\mathcal{L}_{\text{total}} = \sum_m w_m \cdot \mathcal{L}_m(\mathcal{F}_m^{\text{proj}}, \mathcal{C}_m)$

### 损失函数与训练策略
- 回归概念（方位角、仰角、旋转）用 Rank-N-Contrast (RNC) loss（温度 $\tau=2.0$，权重 1.0）
- 分类概念（类别、ID）用改进的 SupCon loss（L2 距离替代内积，权重 0.02）
- 两阶段训练：先训练编码器同时动态更新正交基 → 冻结编码器训练线性解码器

### 代数性质（概念代数）
- **概念正交性**：$\mathcal{F}_0^{\text{proj}} \perp \mathcal{F}_1^{\text{proj}} \perp \cdots$，修改一个概念不影响其他
- **概念组合算子**：$f_j = f_i \pm f_\Delta^{\text{proj},m}$，支持向量加减实现单一概念调整
- **重构性**：$f_i = \mu + \sum_m f_i^{\text{proj},m}$，可从各子空间分量完整重构

## 实验关键数据

### 主实验：ShapeNet（5 个目标：方位角/仰角/旋转+类别/ID）

| 方法 | Spearman(az/el/rot)↑ | V-score(cat/id)↑ | 组合相似度↑ |
|------|---------------------|-----------------|-----------|
| Baseline | 0.41/0.34/0.35 | 0.16/0.14 | 0.22 |
| FAMO | 0.49/0.41/0.42 | 0.00/0.00 | 0.28 |
| Nash-MTL | 0.38/0.41/0.42 | 0.00/0.00 | 0.28 |
| IMTL | 0.31/0.16/0.16 | 0.39/0.28 | 0.14 |
| **Ours** | **0.95/0.87/0.85** | **0.99/0.91** | **0.95** |

### 消融实验与关键发现

| 发现 | 证据 |
|------|------|
| 梯度方法学"捷径" | Objective Set 2 下 Nash-MTL 分类准确率高但 V-score=0→表征崩塌 |
| 正交池化有效解耦 | Ours 的 Spearman 从 0.41→0.95，V-score 从 0.16→0.99 |
| 概念组合可行 | 组合余弦相似度达 0.93-0.95，远超基线 0.14-0.28 |
| 跨数据集泛化 | MPIIGaze（注视估计）和 Rotated MNIST 上同样有效 |
| PCA 可视化 | 基线空间纠缠无序，Ours 各概念沿对应正交轴清晰排列 |

## 亮点与洞察
- **"主动式"vs"反应式"**是本文最核心的思想贡献——不是在优化过程中调解冲突，而是在空间结构上消除冲突的可能性。这类似于"预防胜于治疗"。
- **可解释性强**：每个正交轴直接对应一个语义概念，PCA 可视化清晰展示有组织的潜在空间——这在黑盒深度学习中非常稀有。
- **概念代数**：向量加减可实现概念级操控（如"给这个椅子换个姿态"），验证了潜在空间的组合推理能力。
- **极其轻量**：正交池化仅需 PCA + 矩阵投影，无额外可学习参数，几乎零额外计算成本。
- **变形艺术的类比**非常直观——一个圆柱体从正面看是矩形、从顶部看是圆形，正交投影让一个向量同时"存储"多个独立视角。

## 局限与展望
- **维度限制**：每个概念仅分配 1 维子空间，对于需要多维度才能刻画的复杂概念（如纹理、形状组合）可能不够
- **缺乏生成端**：可以表征 "chair + boat" 的概念组合但无法生成对应图像——与 VAE/GAN 的结合未探索
- **规模有限**：仅在 ShapeNet/MNIST/MPIIGaze 上验证，缺乏大规模真实多任务场景（如 NYUv2、Cityscapes）实验
- **特征向量不稳定性**：训练早期正交基剧烈变化，需要 Hungarian 对齐稳定化——在更大模型/数据上可能更不稳定
- **固定基数量**：$M$ 需预先指定，未探索自适应选择机制

## 相关工作与启发
- **vs GradNorm/PCGrad/IMTL**：它们在梯度空间操作（投影、重加权），本文在特征空间操作（投影到正交子空间），层次更高
- **vs Nash-MTL/FAMO**：更高级的多任务优化目标，但不改变表征结构——本文实验证明结构改变比优化改进更有效
- **vs β-VAE 等解纠缠方法**：β-VAE 通过 KL 散度惩罚实现松散解纠缠，本文通过硬正交约束实现严格解纠缠
- **vs 对比学习**：SupCon/RNC 被本文用作子空间内的损失函数，是方法的组件而非竞争者
- **启发**：正交投影的思路可以推广到多模态学习——不同模态（文本/图像/音频）的表征被限制在正交子空间中

## 评分
- 新颖性: ⭐⭐⭐⭐ 正交池化+概念代数是有创造力的设计
- 实验充分度: ⭐⭐⭐ 数据集规模偏小，缺乏大规模验证
- 写作质量: ⭐⭐⭐⭐ 数学定义严谨，变形艺术类比直观
- 价值: ⭐⭐⭐⭐ 为 MTL 的表征学习提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ICML 2025\] Efficient Robotic Policy Learning via Latent Space Backward Planning](../../ICML2025/robotics/efficient_robotic_policy_learning_via_latent_space_backward_planning.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](../../ICCV2025/robotics/rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](../../ICCV2025/robotics/beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)

</div>

<!-- RELATED:END -->
