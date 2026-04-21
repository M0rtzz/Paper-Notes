---
title: >-
  [论文解读] Deep Learning is Not So Mysterious or Different
description: >-
  [ICML 2025][自监督学习][泛化理论] 本文是一篇 position paper，论证深度学习中被认为"神秘"的泛化现象（良性过拟合、双重下降、过参数化的成功）并非深度学习独有，也不神秘，可以通过长期存在的泛化框架（PAC-Bayes、可数假设界）形式化描述，并提出**软归纳偏置（soft inductive biases）**作为统一解释原则。
tags:
  - ICML 2025
  - 自监督学习
  - 泛化理论
  - 软归纳偏置
  - PAC-Bayes
  - 良性过拟合
  - 双重下降
---

# Deep Learning is Not So Mysterious or Different

**会议**: ICML 2025  
**arXiv**: [2503.02113](https://arxiv.org/abs/2503.02113)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: 泛化理论, 软归纳偏置, PAC-Bayes, 良性过拟合, 双重下降

## 一句话总结

本文是一篇 position paper，论证深度学习中被认为"神秘"的泛化现象（良性过拟合、双重下降、过参数化的成功）并非深度学习独有，也不神秘，可以通过长期存在的泛化框架（PAC-Bayes、可数假设界）形式化描述，并提出**软归纳偏置（soft inductive biases）**作为统一解释原则。

## 研究背景与动机

深度神经网络常被认为与其他模型有本质不同，其泛化行为似乎违反了传统直觉。最常被引用的例子包括：
- **良性过拟合（benign overfitting）**：模型完美拟合噪声数据却仍能泛化
- **双重下降（double descent）**：泛化误差随参数量增加先降-后升-再降
- **过参数化的成功**：参数远多于数据点的模型仍然泛化良好

自 Zhang et al. (2016) 的 "Understanding Deep Learning Requires Rethinking Generalization" 以来，深度学习社区广泛认为这些现象需要"重新思考泛化"，是深度学习独有的谜题。

然而本文作者 Andrew Gordon Wilson 提出了相反立场：这些现象**既非深度学习独有，也不神秘**。它们可以在简单的线性模型中复现，可以直观理解，并且早已有严格的泛化理论框架（PAC-Bayes 和可数假设界）能够形式化刻画。核心矛盾在于：学术界过度关注 VC 维和 Rademacher 复杂度等**无法**解释这些现象的框架，而忽视了**能够**解释它们的 PAC-Bayes 框架——后者已经存在数十年。

本文的切入角度是通过**软归纳偏置**（soft inductive biases）这一统一原则来解释所有这些现象：不要限制假设空间以避免过拟合，而是拥抱灵活的假设空间，同时对与数据一致的较简单解赋予软偏好。

## 方法详解

### 整体框架

本文并非提出新算法，而是一个理论立场论文（position paper），其论证结构为：

1. 引入**软归纳偏置**概念作为统一直觉
2. 介绍 PAC-Bayes 和可数假设界作为形式化工具
3. 依次分析良性过拟合、过参数化、双重下降，证明每个现象都可以用简单模型复现且用上述框架解释
4. 讨论深度学习**真正独特**的方面：表示学习、模式连通性、通用学习能力

### 关键设计

1. **软归纳偏置 vs 限制偏置**：传统观点认为归纳偏置是对假设空间的**限制**（restriction bias），如 CNN 的平移等变性约束。但作者指出，限制偏置不仅不必要，还可能不利。更好的方式是**软偏置**：拥抱灵活假设空间，但对某些解赋予偏好。典型例子是**带阶数依赖正则化的高阶多项式**：$\mathcal{L}(w) = -\log p(y|f(x,w)) + \sum_j \gamma^j w_j^2, \gamma > 1$。模型尽可能用低阶项拟合数据，仅在需要时使用高阶项。这既保证了灵活性，又提供了简洁性偏好，在所有数据量和复杂度下都表现良好。

2. **PAC-Bayes 和可数假设界**：核心泛化上界为 $R(h) \leq \hat{R}(h) + \Delta\sqrt{\frac{K(h|A)\log 2 + \log\frac{1}{\delta}}{2n}}$，其中 $K(h|A)$ 是假设 $h$ 关于架构 $A$ 的前缀无关 Kolmogorov 复杂度。可简化为：**期望风险 ≤ 经验风险 + 模型可压缩性**。关键洞察是：大模型如果能很好地拟合数据**且**可以被有效压缩，就能保证良好泛化。与 VC 维/Rademacher 复杂度不同，这些界不惩罚假设空间的**大小**，而关注假设的**可能性**。近年来已为具有数十亿参数的 LLM 提供了非空泛化保证。

3. **有效维度（Effective Dimensionality）**：定义为 $N_{\text{eff}}(A) = \sum_i \frac{\lambda_i}{\lambda_i + \alpha}$，衡量 Hessian 矩阵中"相对较大"特征值的数量，即损失景观中的陡峭方向数。低有效维度意味着更平坦的解（参数可被扰动而不显著增加损失），而平坦性与泛化之间存在机制性联系：平坦解更可压缩，有更好的 Occam 因子。

### 三大"神秘"现象的解释

**良性过拟合**：只需一个灵活假设空间 + 要求完美拟合数据的损失 + 简洁性偏好。带阶数依赖正则化的 150 阶多项式就能复现：合理拟合简单/复杂结构化数据，同时也能完美拟合纯噪声。高斯过程可以**精确**复现 Zhang et al. (2016) 在 CIFAR-10 上的实验结果。PAC-Bayes 界在结构化数据上非空，在噪声数据上空泛——完美通过了 Zhang et al. 提出的检验标准。

**过参数化**：参数计数是模型复杂度的糟糕代理。增加参数带来两个好处：(1) 增加灵活性使模型能更好拟合数据；(2) 增加压缩偏置（更大模型训练后反而有更少的有效参数）。直觉是：增大参数量使平坦解在整体假设空间中占据更大的**相对体积**，使训练更容易找到这些解。实验证据包括：全批梯度下降甚至随机猜测-检查都能找到泛化不错的解，说明随机优化器的隐式正则化并非泛化的必要条件。

**双重下降**：在欠参数化区间，增加灵活性捕获更多有用信息；在过渡区间，信息增加但出现过拟合；在插值区间（参数>数据点），所有模型完美拟合数据，但继续增加参数使可压缩平坦解的体积增大，有效维度下降，泛化改善。简单线性模型即可复现此现象。

## 实验关键数据

### 主实验

本文的"实验"以证实性示例为主，非传统的性能对比：

| 模型/设置 | 现象 | 关键观察 | 说明 |
|-----------|------|----------|------|
| 150阶多项式 + 阶数依赖正则化 | 良性过拟合 | 完美拟合噪声，合理拟合结构化数据 | 图1(a)-(c) |
| GP on CIFAR-10 | 良性过拟合 | 精确复现 Zhang et al. 结果 | 图1(d) |
| ResNet-20 on CIFAR-10 | 良性过拟合 | 边际似然随噪声标签增加而下降 | 图1(e) |
| ResNet-18（变宽度）on CIFAR-100 | 双重下降 | 有效维度精确跟踪第二次下降 | 图1(f) |
| 线性随机特征模型 | 双重下降 | 与 ResNet 展示相同的双重下降模式 | 图1(g) |

### 泛化界对比

| 泛化框架 | 解释良性过拟合 | 解释过参数化 | 解释双重下降 | 非空界 |
|----------|--------------|------------|-------------|--------|
| VC 维 / Rademacher 复杂度 | ✗ | ✗ | ✗ | 对大模型空泛 |
| PAC-Bayes / 可数假设界 | ✓ | ✓ | ✓ | 对数十亿参数模型非空 |
| Kolmogorov 复杂度上界 | ✓ | ✓ | ✓ | CIFAR-10: 16.6% 误差上界 |

### 关键发现

- PAC-Bayes 和可数假设界能为数百万甚至数十亿参数的模型提供非空泛化保证
- Lotfi et al. (2022a) 将数百万参数模型在 CIFAR-10 上的分类误差上界至 16.6%（95% 概率），这是相当不错的性能
- 更大的模型不仅更灵活，反而有更少的有效参数（Maddox et al., 2020）
- 视觉 Transformer 训练后甚至比 CNN 更具平移等变性（Gruver et al., 2023）
- 全批梯度下降和随机猜测-检查都能找到泛化良好的解（Geiping et al., 2021; Chiang et al., 2022）

## 亮点与洞察

- **极具教育价值**：用最简单的例子（高阶多项式、线性模型）复现所有"神秘"现象，让读者理解本质
- **软归纳偏置**的统一视角非常优雅：限制假设空间 → 在灵活空间中偏好简单解
- 明确指出了深度学习**真正独特**的方面：表示学习（自适应基函数）、模式连通性、通用学习能力（一个预训练模型跨模态泛化）
- "期望风险 ≤ 经验风险 + 可压缩性" 的公式化理解简洁有力
- Residual Pathway Priors 的讨论很有启发：将硬架构约束转为软归纳偏置

## 局限与展望

- 作为 position paper，没有提出新方法或新实验，主要是综合已有工作
- 对"为什么更大模型有更强压缩偏置"这一核心问题仅给出了直觉而非严格回答
- 对 Kolmogorov 复杂度不可计算的实际影响讨论不够深入
- 未充分讨论 NTK 和 mean-field 理论等其他理论流派的视角
- 对 grokking 和 scaling laws 只是简略提及，未做深入分析
- 深度学习的计算效率优势（为什么 SGD 在实践中is sufficient）与理论分析的联系未展开

## 相关工作与启发

- Zhang et al. (2016, 2021) "Understanding Deep Learning Requires Rethinking Generalization" 是本文的主要对话对象
- Lotfi et al. (2022a, 2024b) 的非空 PAC-Bayes 界工作是核心技术支撑
- Goldblum et al. (2024) 关于大模型偏好低 Kolmogorov 复杂度的发现很关键
- Residual Pathway Priors (Finzi et al., 2021) 提供了将硬约束转软约束的实用机制
- 对自监督学习的启发：自监督目标本质上也在构建软归纳偏置，引导模型学习可压缩的表示
- 对模型设计的实践启发：不必追求针对特定问题的硬约束架构，拥抱灵活架构 + 适当偏好可能是更好的通用策略

## 评分

- 新颖性: ⭐⭐⭐ — 主要综合已有洞察，核心观点并非首次提出
- 实验充分度: ⭐⭐⭐ — 示例充分但缺乏新实验
- 写作质量: ⭐⭐⭐⭐⭐ — 论述清晰，逻辑严密，图示精美，极具教育性
- 价值: ⭐⭐⭐⭐ — 对澄清泛化理论的常见误解有重要贡献，对社区建设有益

<!-- RELATED:START -->

## 相关论文

- [To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](../../ICCV2025/self_supervised/to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)
- [BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](../../CVPR2025/self_supervised/boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [Weak-SIGReg: Covariance Regularization for Stable Deep Learning](../../ICLR2026/self_supervised/weak-sigreg_covariance_regularization_for_stable_deep_learning.md)
- [ScaleLSD: Scalable Deep Line Segment Detection Streamlined](../../CVPR2025/self_supervised/scalelsd_scalable_deep_line_segment_detection_streamlined.md)
- [SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](../../CVPR2026/self_supervised/sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)

<!-- RELATED:END -->
