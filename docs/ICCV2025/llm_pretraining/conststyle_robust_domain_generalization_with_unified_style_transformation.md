---
title: >-
  [论文解读] ConstStyle: Robust Domain Generalization with Unified Style Transformation
description: >-
  [ICCV 2025][预训练][域泛化] 提出ConstStyle框架，通过构建一个理论驱动的"统一域"（Unified Domain），在训练时将所有样本风格对齐到该统一域，测试时将未见域样本部分投影到统一域，有效缩小域间差距并提升泛化性能。
tags:
  - "ICCV 2025"
  - "预训练"
  - "域泛化"
  - "风格变换"
  - "统一域"
  - "分布对齐"
  - "鲁棒性"
---

# ConstStyle: Robust Domain Generalization with Unified Style Transformation

**会议**: ICCV 2025  
**arXiv**: [2509.05975](https://arxiv.org/abs/2509.05975)  
**代码**: [https://github.com/nduongw/ConstStyle](https://github.com/nduongw/ConstStyle)  
**领域**: 域泛化  
**关键词**: 域泛化, 风格变换, 统一域, 分布对齐, 鲁棒性

## 一句话总结

提出ConstStyle框架，通过构建一个理论驱动的"统一域"（Unified Domain），在训练时将所有样本风格对齐到该统一域，测试时将未见域样本部分投影到统一域，有效缩小域间差距并提升泛化性能。

## 研究背景与动机

深度神经网络在测试数据分布与训练分布不同时性能会显著下降（域偏移问题）。现有域泛化（DG）方法主要有两类策略：（1）学习域不变特征；（2）数据增强以增强多样性。但这两类方法各有局限：

- 不变表示学习方法需要大量多样化的域才能有效提取不变特征
- 数据增强方法假设更多域训练会带来更好性能，但实验发现这并不总是成立——有时用更少但精选的域训练反而能产生更好的类别分离
- 现有方法通常只关注训练阶段，忽略了测试阶段（域差距最大时）的处理

关键洞察：训练和测试都应在一个共同的"统一域"中进行，才能有效缩小域间差距。

## 方法详解

### 整体框架

ConstStyle分为训练和测试两个阶段：训练阶段包含（i）确定统一域风格、（ii）将训练数据风格变换到统一域、（iii）用对齐后的样本训练模型。测试阶段将未见域样本部分对齐到统一域再进行推理。

### 关键设计

1. **统一域确定（Unified Domain Determination）**: 将每个样本的风格统计量定义为通道级均值和方差的拼接 $\epsilon_x = \text{concat}(\mu_x, \sigma_x)$，域风格建模为多元正态分布 $\mathcal{P}_S \sim \mathcal{N}(\epsilon_S, \Sigma_S)$。统一域的风格定义为所有已见域风格分布的Wasserstein重心（Barycenter），均值为 $\epsilon_B = \frac{1}{N}\sum_{k=1}^{N}\epsilon_{S_k}$，协方差通过迭代优化求解。理论分析表明（Theorem 1），在统一域上训练的模型与在原始域上训练的模型之间的经验损失差距受域间距离约束：$\sum_k(L^{S_k^T} - L^{S_k}) \leq \beta \times \sum_k(\mathcal{D}_\mu(\mathcal{T}, S_k) + \mathcal{D}_\sigma(\mathcal{T}, S_k))$。实际中不知道域标签时，用GMM聚类风格统计量来近似域划分。

2. **训练过程（两阶段训练）**: 第一阶段：用原始数据进行初始ERM训练，获得风格特征提取器$\theta_s$，提取所有样本的风格特征并确定初始统一域。第二阶段：每个epoch中，将训练样本的风格特征对齐到统一域风格——从统一域分布$\mathcal{N}(\epsilon_T, \Sigma_T)$中采样风格统计量$(\mu_s, \sigma_s)$，通过AdaIN式变换 $z_{x_i}^T = \sigma_s \times \frac{z_{x_i} - \mu_x}{\sigma_x} + \mu_s$ 实现风格对齐。统一域每$\gamma$个epoch更新一次。整个E轮训练产生$E \times D$种不同的风格变体，增强了模型对统一域风格的适应性。

3. **推理阶段的部分投影（Partial Alignment）**: 测试时不完全将未见域映射到统一域（可能丢失原始信息），而是用部分投影策略平衡域对齐和信息保留：$z_u^T = (\alpha \sigma_u + (1-\alpha)\sigma_T)\frac{z_u^o - \mu_u}{\sigma_u} + (\alpha \mu_u + (1-\alpha)\mu_T)$。超参数$\alpha \in (0,1)$控制保留原始特征的程度。Theorem 2给出性能保证：$L^{\mathcal{U}^T} - L^{\mathcal{S}^T} \leq \alpha \beta (\mathcal{D}_\mu(\mathcal{U},\mathcal{T}) + \mathcal{D}_\sigma(\mathcal{U},\mathcal{T})) + \epsilon\sqrt{2 \cdot Tr(I)}$，表明$\alpha$越小域间差距的影响越小，但可能增加信息损失。

### 损失函数 / 训练策略

使用标准交叉熵损失进行分类训练。关键训练策略：初始训练仅几个epoch（非完全收敛），以节省成本快速建立初始统一域；统一域周期性更新（每$\gamma$ epoch）以提高质量同时保持训练稳定性。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ConstStyle | 之前SOTA（CSU） | 提升 |
|--------|------|------------|-----------------|------|
| PACS（单未见域平均） | 准确率 | 86.77% | CSU: 85.33% | +1.44% |
| PACS（Sketch域） | 准确率 | 82.32% | CSU: 78.11% | +4.21% |
| Digits5（平均） | 准确率 | 86.88% | EDFMix: 86.14% | +0.74% |
| PACS In-domain | 准确率 | 96.50% | DSU: 96.30% | +0.20% |

在域差距最大的场景（如Sketch域），ConstStyle改进最为显著。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 多未见域（2个）平均 | 准确率提升+2.43% | 即使训练域减少仍保持优势 |
| Cartoon+Sketch未见域 | 准确率提升+5.91% | 大域差距时提升最显著 |
| 域差距最大时（Photo→Sketch） | 比CSU高达+15.03% | 验证统一域策略的核心优势 |
| α=0（完全投影） | 性能下降 | 过度对齐损失原始信息 |
| α=1（不投影） | 性能下降 | 未利用统一域对齐 |
| 少见域训练 | 提升高达19.82% | 对比次优方法在极端少域场景 |

### 关键发现

- 更多训练域不总是带来更好泛化：有时单域训练可以产生更清晰的类别边界
- 域差距越大，ConstStyle相对其他方法的优势越明显（最高达15%+）
- 部分投影策略（α=0.5-0.6）在信息保留和域对齐之间实现了最佳平衡
- 在见域上的性能也有提升（96.50% vs 95.94% ERM），说明统一域训练不损害见域性能

## 亮点与洞察

- 统一域概念是一个优雅的理论框架，通过Wasserstein重心提供了数学上有依据的统一域选择方法
- 同时处理训练和测试阶段的域偏移，不像大多数DG方法只关注训练
- 理论分析完整（Lemma 1 + Theorem 1/2），为方法设计提供了清晰的理论指导
- 部分投影策略简单但有效，可解释性强（α的物理含义清晰）

## 局限与展望

- 域风格仅用通道级均值和方差表征，可能无法捕获更高阶的域特征
- GMM聚类估计域数量存在超参数敏感性
- Barycenter的近似计算可能引入误差
- 在极端域差距下部分投影仍可能不足，可考虑自适应α

## 相关工作与启发

- 与MixStyle、DSU等风格增强方法不同，ConstStyle不是增加风格多样性，而是将所有域统一到一个点——这种"收敛"而非"发散"的思路在DG中较为独特
- 测试时域自适应的思路可与test-time training/adaptation方法结合
- 统一域框架可推广到其他域偏移问题（如目标检测、语义分割等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一域概念新颖，训练+测试联合处理域偏移的思路独特
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多场景（少域、多未见域、不同域差距）
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合紧密，结构清晰
- 价值: ⭐⭐⭐⭐ 方法简洁有效，理论保证完善，可广泛应用于DG场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics](../../ICML2025/llm_pretraining/on_the_clean_generalization_and_robust_overfitting_in_adversarial_training_from_.md)
- [\[CVPR 2025\] A Unified Framework for Heterogeneous Semi-supervised Learning](../../CVPR2025/llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](../../NeurIPS2025/llm_pretraining/superposition_yields_robust_neural_scaling.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](aceg_improving_generalization_of_scene_coordinate_regression.md)
- [\[ICML 2025\] Towards Robust Influence Functions with Flat Validation Minima](../../ICML2025/llm_pretraining/towards_robust_influence_functions_with_flat_validation_minima.md)

</div>

<!-- RELATED:END -->
