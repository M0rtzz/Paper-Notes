---
title: >-
  [论文解读] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness
description: >-
  [NeurIPS 2025][AI安全][adversarial robustness] 通过在 CNN 中嵌入旋转等变（P4群）和尺度等变卷积层，提出 Parallel 和 Cascaded 两种对称性感知架构，无需对抗训练即可显著提升对抗鲁棒性，并从 CLEVER 框架出发理论证明等变架构能压缩假设空间、正则化梯度、收紧认证鲁棒性界。
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "adversarial robustness"
  - "equivariance"
  - "group-equivariant CNN"
  - "CLEVER bound"
  - "rotation equivariance"
  - "scale equivariance"
---

# Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness

**会议**: NeurIPS 2025  
**arXiv**: [2510.16171](https://arxiv.org/abs/2510.16171)  
**代码**: [ifratmitul/Role-of-Equivariance](https://github.com/ifratmitul/Role-of-Equivariance)  
**领域**: AI安全  
**关键词**: adversarial robustness, equivariance, group-equivariant CNN, CLEVER bound, rotation equivariance, scale equivariance

## 一句话总结

通过在 CNN 中嵌入旋转等变（P4群）和尺度等变卷积层，提出 Parallel 和 Cascaded 两种对称性感知架构，无需对抗训练即可显著提升对抗鲁棒性，并从 CLEVER 框架出发理论证明等变架构能压缩假设空间、正则化梯度、收紧认证鲁棒性界。

## 研究背景与动机

**领域现状**：对抗样本（adversarial examples）暴露了 DNN 对微小输入扰动的脆弱性，对抗训练（adversarial training）是当前主流防御策略，通过在训练中注入对抗样本引导模型学习更鲁棒的特征。

**现有痛点**：对抗训练存在三大问题——(a) 计算开销大（需持续生成对抗样本）；(b) 损害干净数据准确率；(c) 防御效果通常局限于训练时见过的攻击类型，泛化性差。

**核心矛盾**：对抗训练是"反应式"防御——通过修改数据来应对攻击，而非从模型架构层面"主动"编码鲁棒性。模型的脆弱性根源在于过拟合到非语义的虚假统计模式。

**本文目标** 能否通过架构先验（architectural priors）单独提升对抗鲁棒性？具体包括：(a) 等变架构能否理论保证更紧的认证鲁棒性界？(b) 等变卷积如何平滑梯度行为？(c) 不使用对抗训练时，等变 CNN 能否超越标准 CNN？

**切入角度**：标准 CNN 仅具备平移等变性，但对旋转和缩放不等变。对抗扰动往往违反数据的固有对称性，如果模型被约束沿群轨道一致响应，就能自然抑制偏离轨道的扰动。

**核心 idea**：将群等变卷积作为架构先验编码到 CNN 中，使模型的决策边界与数据的几何结构对齐，从而提供架构层面的对抗防御。

## 方法详解

### 整体框架

方法包含两部分：(1) **理论框架**——从假设空间复杂度、Jacobian 结构、CLEVER 认证界、梯度平滑四个角度分析等变性与对抗鲁棒性的关系；(2) **架构设计**——提出 Parallel（并行）和 Cascaded（级联）两种将标准/旋转等变/尺度等变卷积组合进 CNN 的方案。

### 关键设计

1. **等变函数与 Jacobian 不变性**:

    - 功能：建立等变网络的梯度/Lipschitz 常数在群轨道上的不变性
    - 核心思路：等变函数满足 $f(g \cdot x) = \rho(g) f(x)$，其 Jacobian 变换为 $J_f(g \cdot x) = \rho(g) J_f(x) Dg^{-1}$。当 $\rho(g)$ 和 $Dg^{-1}$ 为正交矩阵时，$\|J_f(g \cdot x)\|_2 = \|J_f(x)\|_2$（Lemma 1），即 Lipschitz 常数在轨道上不变
    - 设计动机：Lipschitz 常数控制着模型对输入扰动的最大响应幅度，不变性意味着鲁棒性保证在整个等价类上统一成立

2. **CLEVER 认证界的轨道不变性（Theorem 1）**:

    - 功能：证明等变网络在群轨道上有一致的认证鲁棒性半径
    - 核心思路：margin 函数 $g_{c,j}(x) = f_c(x) - f_j(x)$ 的梯度范数在群轨道上不变 $\|\nabla g_{c,j}(g \cdot x)\|_q = \|\nabla g_{c,j}(x)\|_q$，因此 CLEVER bound $\epsilon_{\min}^{(p)}(x) = \min_{j \neq c} g_{c,j}(x) / L_q^{(j)}$ 在轨道上保持一致
    - 设计动机：标准 CNN 的认证界仅在单点成立，等变网络将保证推广到整个等价类

3. **梯度方向性抑制（Theorem 2）**:

    - 功能：证明等变网络选择性地抑制沿对称方向的梯度变化
    - 核心思路：将扰动分解为轨道切向分量 $\delta_G$ 和正交分量 $\delta_\perp$，等变网络满足 $\|\nabla f(x + \delta_G) - \nabla f(x)\|_2 \to 0$（轨道方向梯度被抑制），而 $\|\nabla f(x + \delta_\perp) - \nabla f(x)\|_2$ 保持大值。轨道平均梯度 $\bar{\phi}_j(x) = \frac{1}{|G|}\sum_{g \in G} \nabla f_j(g \cdot x)$ 起到平滑作用
    - 设计动机：对抗扰动通常偏离数据流形（正交于群轨道），等变约束使模型对这些方向的梯度变化更大（即更"警觉"），同时对对称方向不敏感

4. **尺度等变的梯度平滑分析**:

    - 功能：分析不保范数的尺度变换如何贡献鲁棒性
    - 核心思路：尺度等变不满足正交性条件，无法直接获得认证界。但通过多尺度聚合 $h(x) = \sum_{s \in G_s} w_s \cdot \phi_s(x)$ 和尺度群卷积 $[\Phi f](x) = \bigoplus_{s \in G_s} \psi_s * f(T_s^{-1}x)$，梯度场被多尺度平均平滑，类似于频域低通滤波
    - 设计动机：尺度等变提供了与旋转等变互补的正则化机制：旋转保范数提供认证界，尺度不保范数但通过梯度平滑实际降低对抗脆弱性

5. **Parallel vs Cascaded 架构**:

    - **Parallel（并行）**：标准卷积分支 + P4 旋转等变分支 + 尺度等变分支，各分支独立处理后通过 concatenation 融合。保留互补特征空间
    - **Cascaded（级联）**：输入先经旋转等变层再送入标准卷积层，顺序处理。计算效率更高但特征多样性较低
    - 融合策略：concatenation 始终优于 weighted summation，因后者在对抗设置下学到的权重可能不鲁棒

### 损失函数 / 训练策略

使用标准分类交叉熵损失训练，**不使用任何对抗训练技巧**。等变层通过权重共享机制自动实现群等变约束，无需额外正则化。4 层和 10 层 CNN 架构均进行 5 次随机种子实验。

## 实验关键数据

### 主实验 — FGSM/PGD 对抗准确率（CIFAR-10, 10层, ε=0.01）

| 模型 | Clean Acc | FGSM Acc | PGD Acc |
|------|:---------:|:--------:|:-------:|
| Baseline CNN | ~73% | ~45% | ~35% |
| Cascaded GCNN | ~72% | ~50% | ~40% |
| Parallel GCNN | ~75% | ~60% | ~50% |
| **Parallel GCNN (R&S)** | ~74% | **~73%** | **~65%** |
| Weighted Parallel | ~73% | ~48% | ~38% |

Parallel GCNN (R&S) 在所有扰动强度下取得最优对抗准确率，10 层全等变模型达 73.01% FGSM 和 64.96% PGD 准确率。

### 消融实验 — CIFAR-10C 自然损坏（4层, ε₁）

| 损坏类型 | Baseline | Cascaded | Parallel | Parallel (R&S) | Weighted |
|---------|:--------:|:--------:|:--------:|:--------------:|:--------:|
| Brightness | 14.79 | 8.27 | **18.38** | 8.38 | 8.13 |
| Gaussian Noise | 5.38 | 1.82 | **14.68** | 1.84 | 1.78 |
| Frost | 9.32 | 3.15 | **11.90** | 3.44 | 3.07 |
| Fog | 3.64 | 1.31 | **6.97** | 1.39 | 1.11 |

### 关键发现
- **并行 > 级联**：Parallel 设计保留了互补特征空间，在对抗设置下始终优于 Cascaded
- **concatenation > weighted fusion**：可学习权重在对抗设置下不可靠，简单拼接更稳定
- **深度叠加效果**：10 层全等变模型比 4 层效果更好，说明等变性的正则化效果随深度累积
- **自然损坏 vs 对抗扰动**：Parallel GCNN（无尺度）在自然损坏上最优，但 Parallel GCNN (R&S) 在对抗攻击下最优，说明等变性对两类扰动的效果不同
- **所有等变模型均未使用对抗训练**，纯架构层面即超越标准 CNN 的鲁棒性

## 亮点与洞察
- **无需对抗训练的鲁棒性提升**：纯架构改进，零额外训练开销，为对抗鲁棒性提供了"免费午餐"
- **轨道一致性保证**：不仅在单点鲁棒，而是在整个群轨道（等价类）上统一保证，这是标准对抗训练无法提供的
- **Theorem 2 的方向性洞察**：等变网络对沿对称方向的扰动不敏感、对正交方向（典型对抗扰动方向）保持敏感，形成"方向性鲁棒"
- **可与对抗训练正交组合**：等变架构不排斥对抗训练和随机平滑，组合使用可能实现更强防御

## 局限与展望
- 仅在 CIFAR 级别小数据集上验证，未测试 ImageNet 等大规模数据集
- 仅考虑 $\ell_p$ 范数攻击，未涉及语义攻击、patch 攻击等复杂威胁
- 等变基于离散 P4 群（仅 4 个旋转角度），对连续旋转覆盖有限
- 尺度等变缺乏正式认证界，只能提供梯度平滑的定性分析
- 架构仅限 CNN，未探索 Vision Transformer 上的应用
- 多分支并行设计的参数量和推理时间增加缺少定量分析

## 相关工作与启发
- **vs PGD 对抗训练**: 后者需额外训练开销且损害干净准确率；本文零额外训练，干净准确率无明显下降
- **vs Randomized Smoothing**: 后者通过随机化实现认证防御但牺牲精度；本文通过架构约束实现 CLEVER 认证界
- **vs G-CNN (Cohen & Welling, 2016)**: G-CNN 关注分类性能和样本效率；本文重点研究等变性对对抗鲁棒性的影响
- **vs Harmonic Networks**: 后者实现连续旋转等变但未分析对抗鲁棒性；本文使用离散群但增加尺度等变分支并提供理论分析
- 可扩展方向：将等变约束引入 ViT 的 self-attention（如 equivariant attention）

## 评分
- 新颖性: ⭐⭐⭐⭐ 等变性与对抗鲁棒性的系统性桥接是有意义的新视角
- 实验充分度: ⭐⭐⭐ 理论分析充分但实验仅在 CIFAR 级数据集，缺大规模验证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，架构描述系统化
- 价值: ⭐⭐⭐⭐ 无需对抗训练即提升鲁棒性的方向有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](../../ICCV2025/ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [\[ICML 2025\] Quadratic Upper Bound for Boosting Robustness](../../ICML2025/ai_safety/quadratic_upper_bound_for_boosting_robustness.md)
- [\[CVPR 2025\] Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](../../CVPR2025/ai_safety/leveraging_perturbation_robustness_to_enhance_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
