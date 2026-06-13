---
title: >-
  [论文解读] Towards Source-Free Machine Unlearning
description: >-
  [CVPR 2025][AI安全][机器遗忘] 本文提出了一种无源机器遗忘（Source-Free Machine Unlearning）算法，在无法获取原始训练数据的条件下，通过近似估计保留数据的 Hessian 矩阵（仅使用待遗忘数据和训练好的模型），实现了对线性和混合线性分类器的高效遗忘，并提供了严格的理论上界保证。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "机器遗忘"
  - "数据隐私"
  - "无源遗忘"
  - "Hessian估计"
  - "理论保证"
---

# Towards Source-Free Machine Unlearning

**会议**: CVPR 2025  
**arXiv**: [2508.15127](https://arxiv.org/abs/2508.15127)  
**代码**: [https://github.com/UCR-Vision-and-Learning-Group/mixed-linear-forgetting](https://github.com/UCR-Vision-and-Learning-Group/mixed-linear-forgetting)  
**领域**: AI安全 / 机器遗忘  
**关键词**: 机器遗忘, 数据隐私, 无源遗忘, Hessian估计, 理论保证

## 一句话总结

本文提出了一种无源机器遗忘（Source-Free Machine Unlearning）算法，在无法获取原始训练数据的条件下，通过近似估计保留数据的 Hessian 矩阵（仅使用待遗忘数据和训练好的模型），实现了对线性和混合线性分类器的高效遗忘，并提供了严格的理论上界保证。

## 研究背景与动机

**领域现状**：随着 GDPR 等数据保护法规的实施，从训练好的模型中移除特定数据的影响（"被遗忘权"）已成为机器学习的核心需求。机器遗忘（Machine Unlearning）旨在高效修改模型参数以遗忘指定数据，同时保持对剩余数据的性能。现有方法分为精确遗忘（如 SISA 分片训练）和近似遗忘（如基于影响函数的参数更新）两大类。

**现有痛点**：几乎所有现有遗忘方法都假设可以获取完整或部分原始训练数据，但在实际场景中，由于存储成本和隐私问题，模型拥有者可能已经不再持有原始训练数据。现有的"零样本遗忘"方法存在严重限制：（1）Chundawat 等人的方法只能遗忘整个类别，无法遗忘特定实例；（2）Cha 等人的实例级遗忘方法在遗忘样本数增加时性能急剧下降，可扩展性差；（3）最关键的是，所有现有零样本遗忘方法都无法提供遗忘效果的理论保证。

**核心矛盾**：基于影响函数的遗忘（Newton 更新步）$w_{uf} = w^* + H_r^{-1}\nabla_f + \sigma^2\varepsilon$ 需要计算保留数据（remaining data）的 Hessian 矩阵 $H_r$，但在无源设置下，保留数据不可获取，Hessian 无法直接计算。

**本文目标**：在仅有待遗忘数据 $\mathcal{D}_f$ 和训练好的模型参数 $w^*$ 的条件下，近似估计保留数据的 Hessian $H_r$，从而实现高效且有理论保证的机器遗忘。

**切入角度**：利用损失函数在最优点附近的 Taylor 展开和一个关键假设——在小扰动下，保留数据和遗忘数据的损失差值相近（$|\delta\mathcal{L}_r(w_i) - \delta\mathcal{L}_f(w_i)| \leq \epsilon$）。

**核心 idea**：通过在最优点周围生成小扰动，利用遗忘数据的损失差值作为保留数据损失差值的代理，将 Hessian 估计问题转化为半定规划（SDP）优化问题。

## 方法详解

### 整体框架

给定训练好的最优分类器 $w^*$ 和待遗忘数据 $\mathcal{D}_f$：（1）在 $w^*$ 周围生成 $m$ 个小扰动 $w_i = w^* + (\delta w)_i$；（2）对每个扰动计算遗忘数据的损失差值 $\delta\mathcal{L}_f(w_i)$；（3）通过最小化目标函数 $\tilde{\Psi}(H_r)$ 求解 Hessian 估计 $\hat{H}_r$；（4）使用估计的 Hessian 执行 Newton 更新步完成遗忘。

### 关键设计

1. **Hessian 近似估计 (Retain Hessian Estimation)**:

    - 功能：在不接触保留数据的条件下，估计保留数据对应的 Hessian 矩阵
    - 核心思路：对凸损失函数进行 Taylor 展开，得到 $\delta\mathcal{L}_r \approx \nabla_r(w^*)^\top \delta w + \frac{1}{2}(\delta w)^\top H_r(w^*) \delta w$。由于训练收敛到最优点 $\nabla(w^*) = 0$，可知 $\nabla_r(w^*) = -\nabla_f(w^*)$（可从遗忘数据计算）。然后构造目标函数 $\tilde{\Psi}(H_r) = \frac{1}{m}\sum_{i=1}^m (\tilde{f}_i(H_r))^2$，其中用 $\delta\mathcal{L}_f(w_i)$ 替代不可计算的 $\delta\mathcal{L}_r(w_i)$。最终转化为一个 PSD 约束下的 SDP 优化问题
    - 设计动机：Newton 更新步是理论最优的遗忘方法，但需要保留数据的 Hessian。本方法利用小扰动下保留数据和遗忘数据损失差值的接近性（二者都被 $L\|\delta w\|$ 上界）实现了零样本估计

2. **理论保证框架**:

    - 功能：为估计的 Hessian 与真实 Hessian 的差距提供严格上界
    - 核心思路：Lemma 1 证明了 $\|\Delta H_r\|_F \leq \frac{2\epsilon\sqrt{d}}{2+d}$，其中 $\epsilon$ 是损失差值近似误差的上界，$d$ 是特征维度。注意这个上界随维度 $d$ 增大而减小——高维场景下 Hessian 估计反而更准确。Theorem 1 进一步给出遗忘后模型在保留数据上的梯度范数上界：$\|\nabla\mathcal{L}(w_{uf}, \mathcal{D}_r)\|_2 \leq \frac{4\gamma C^2 n_f^2(n-n_f)}{[\lambda(n-n_f) - \frac{2\epsilon}{2+d}]^2}$
    - 设计动机：缺乏理论保证是现有零样本遗忘方法的最大弱点，理论上界为遗忘的可靠性和隐私合规提供了信心

3. **混合线性遗忘扩展 (Mixed Linear Unlearning)**:

    - 功能：将线性分类器的遗忘方法扩展到深度神经网络
    - 核心思路：利用 Neural Tangent Kernel（NTK）理论，对网络最后几层进行一阶 Taylor 展开线性化：$\mathcal{L}(w) = \sum_{i=1}^n \|f_{w_c^*}(x_i) + \nabla_w f_{w_c^*}(x_i) \cdot w - y_i\|_2^2 + \frac{\lambda n}{2}\|w\|_2^2$。线性化后损失变为凸的，可以直接应用本文的 Hessian 估计和遗忘算法
    - 设计动机：直接处理深度网络的非凸损失极其困难，但通过 NTK 线性化可以将问题转化为凸优化，使理论保证依然成立

### 损失函数 / 训练策略

- 使用 $\ell_2$ 正则化的凸损失函数（二次损失用于实验）
- Hessian 估计通过 SDP 优化求解
- 扰动 $\delta w$ 的每个元素从标准高斯分布 $\mathcal{N}(0, 1)$ 采样
- 遗忘后添加高斯噪声 $\sigma^2\varepsilon$ 防止信息泄露

## 实验关键数据

### 主实验

CIFAR-10 线性分类器（遗忘 10% 训练数据）：

| 方法 | 测试准确率 | 保留数据准确率 | 遗忘数据准确率 | MIA |
|------|-----------|-------------|-------------|-----|
| Retrained (目标) | 72% | 74% | 72% | 50% |
| NegGrad | 51.9% | 53.2% | 51.2% | 48% |
| Random Labels | 20.6% | 21.6% | 21.4% | 47% |
| JiT | 52.1% | 53.1% | 51.1% | 49.1% |
| Adversarial | 51.5% | 52.7% | 51.0% | 50.0% |
| **Unlearned (-) (本文)** | **70%** | **71%** | **68%** | **51.4%** |

本文方法大幅超越所有无源基线，接近理想的重训练性能。

### 消融实验

遗忘数据比例的影响（CIFAR-10）：

| 遗忘比例 | 测试性能差距 | 保留数据差距 | MIA差距 |
|---------|-----------|-----------|--------|
| 5% | 0% | 0% | 0% |
| 10% | 2% | 3% | 1.4% |
| 15% | 14% | 15% | 5.8% |

扰动数量的影响：

| 扰动数量 | 测试性能差距 | MIA差距 |
|---------|-----------|--------|
| 250 | 15% | 6.2% |
| 500 | 2% | 1.4% |
| 1000 | 0% | 1% |

### 关键发现

- 遗忘比例为 5% 时，遗忘后模型与重训练模型完全一致（0% 性能差距）
- 增加扰动数量能显著改善 Hessian 估计质量，1000 个扰动时基本消除性能差距
- 增大正则化参数 $\lambda$ 可以收紧理论上界，实验也验证了这一点
- 混合线性网络上（ResNet-18 + CIFAR-100）同样有效，遗忘 10% 数据时性能差距仅 0.5-3.7%

## 亮点与洞察

- 首个为无源遗忘提供理论保证的工作，填补了该领域的重要空白
- 核心洞察非常巧妙：小扰动下保留数据和遗忘数据的损失差值接近，使得 Hessian 可以用 SDP 估计
- 理论上界随维度增大而收紧——这意味着在高维（深度学习）场景下方法反而更有效，颇具实践意义
- 将问题转化为 SDP 是一个优雅的数学处理
- MIA 分数接近 50% 说明遗忘确实消除了成员信息

## 局限与展望

- 当前直接适用于线性和混合线性分类器，对完全非线性的深度网络需要通过 NTK 线性化近似
- 遗忘比例超过 10% 时性能下降明显，可扩展性受 Hessian 估计精度限制
- SDP 求解的计算复杂度较高，大规模 Hessian 的存储和求逆是瓶颈
- 未来可结合 Hessian 对角化或 Hessian-向量积等近似方法提升实用性
- 对更复杂的遗忘场景（如生成模型中的概念擦除）的扩展值得探索

## 相关工作与启发

- Guo et al. (2020) 的 Newton 更新步是本文的理论基础，本文解决了其需要保留数据 Hessian 的限制
- 与 Chundawat et al. 的零样本遗忘（仅类级别）和 Cha et al. 的实例级遗忘相比，本文同时实现了实例级遗忘和理论保证
- Mixed Linear Unlearning (Golatkar et al., 2021) 为深度网络提供了可行的线性化路径
- 对数据隐私合规（GDPR）和模型治理有直接的实践意义

## 评分

- **新颖性**: 8/10 — 核心 Hessian 估计方法巧妙，理论贡献扎实
- **实验充分度**: 7/10 — 多数据集多消融验证，但缺少大规模深度学习场景
- **写作质量**: 8/10 — 理论推导清晰完整，实验设计合理
- **价值**: 8/10 — 无源遗忘的理论保证对隐私合规有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](../../NeurIPS2025/ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[ICML 2025\] A Certified Unlearning Approach without Access to Source Data](../../ICML2025/ai_safety/a_certified_unlearning_approach_without_access_to_source_data.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](../../NeurIPS2025/ai_safety/rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](../../NeurIPS2025/ai_safety/efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](../../NeurIPS2025/ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)

</div>

<!-- RELATED:END -->
