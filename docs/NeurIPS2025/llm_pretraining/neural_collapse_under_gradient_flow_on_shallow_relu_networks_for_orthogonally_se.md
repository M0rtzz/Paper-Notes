---
title: >-
  [论文解读] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data
description: >-
  [NeurIPS 2025][Neural Collapse] 首次证明在正交可分数据上，两层ReLU网络的梯度流（GF）在小初始化下可证收敛到Neural Collapse（NC）解，揭示了GF隐式偏置（早期神经元对齐+渐近最大间隔偏置）在促进NC出现中的关键作用。
tags:
  - NeurIPS 2025
  - Neural Collapse
  - 梯度流
  - ReLU网络
  - 隐式偏置
  - 正交可分数据
  - 最大间隔
---

# Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data

**会议**: NeurIPS 2025  
**arXiv**: [2510.21078](https://arxiv.org/abs/2510.21078)  
**作者**: Hancheng Min (上海交通大学), Zhihui Zhu (Ohio State University), René Vidal (University of Pennsylvania)  
**代码**: 未公开  
**领域**: others  
**关键词**: Neural Collapse, 梯度流, ReLU网络, 隐式偏置, 正交可分数据, 最大间隔  

## 一句话总结

首次证明在正交可分数据上，两层ReLU网络的梯度流（GF）在小初始化下可证收敛到Neural Collapse（NC）解，揭示了GF隐式偏置（早期神经元对齐+渐近最大间隔偏置）在促进NC出现中的关键作用。

## 研究背景与动机

### 问题背景
Neural Collapse（NC）是深度网络训练末期出现的一种引人注目的现象：最后一层的特征呈现出简单的几何结构——(1) 同类特征塌缩为单点；(2) 类均值最大分离（形成simplex ETF）；(3) 分类器权重与类均值对齐（自对偶性）。NC的理论研究对理解深度学习的成功机制至关重要。

### 已有工作的不足
- 现有理论主要基于**无约束特征模型（UFM）**，将最后一层特征视为自由优化变量，证明全局最优解展示NC。但UFM忽略了输入数据结构和非线性激活的影响
- 少数考虑收敛性的工作需要额外条件：接近全局最优的初始化、权重衰减正则化、或极大网络宽度
- **关键缺口**：训练神经网络"如何"导致NC这一动态过程几乎未被探索，NC与训练算法隐式偏置之间的关系不清楚
- 实际中NC甚至在没有显式正则化的网络中也被观察到，暗示隐式偏置起关键作用

### 核心动机
通过分析两层ReLU网络在正交可分数据上梯度流的完整动态过程，建立NC现象与GF隐式偏置之间的显式联系，从而：(1) 放松UFM的无约束假设；(2) 揭示数据结构和ReLU非线性对NC特征的影响；(3) 阐明隐式偏置如何促进NC的涌现。

## 方法详解

### 问题设定
- **数据假设（正交可分性）**：同类数据正相关（内积 $\geq \mu_s$），异类数据负相关（内积 $\leq -\mu_d$）
- **网络**：宽度为 $h$ 的两层ReLU网络 $f(\mathbf{x};\boldsymbol{\theta}) = \mathbf{V}\sigma(\mathbf{W}^\top \mathbf{x})$，无偏置
- **训练**：梯度流（GF），即无限小步长的梯度下降；$\epsilon$-小且平衡的初始化
- **损失**：二分类用指数/logistic损失，多分类用交叉熵损失

### 主定理（Theorem 1）
在正交可分数据和小初始化下，GF可证收敛到以下NC解：

**1. 类内方向塌缩**：同类数据的最后一层特征塌缩到一维子空间（而非UFM中的单点），即 $\phi_{\bar{\theta}}(\mathbf{x}_i) = \langle s_k \mathbf{u}_k, \mathbf{x}_i \rangle \cdot \bar{\phi}_k$，其中 $\mathbf{u}_k$ 是类 $k$ 的最大间隔方向，$s_k$ 由各类间隔比例决定。特征方向一致但长度可变。

**2. 正交类均值**：类均值特征两两正交且非负：$\langle \bar{\phi}_k, \bar{\phi}_{k'} \rangle = 0$ ($k \neq k'$)。由ReLU的非负性导致正交结构（而非UFM中的simplex ETF），但减去全局均值后可转化为simplex ETF。

**3. 投影自对偶性**：分类器权重与投影后的类均值对齐。二分类时 $\bar{\mathbf{V}} = s_+ \bar{\phi}_+^\top - s_- \bar{\phi}_-^\top$；多分类时 $\bar{\mathbf{V}}$ 形成（缩放的）simplex ETF。

### 证明框架：两阶段隐式偏置
本文的核心贡献在于将NC的出现分解为GF隐式偏置驱动的两个阶段：

**阶段一：对齐阶段（早期）——类间分离**

- 小初始化下，网络权重范数在 $\Theta(\log 1/\epsilon)$ 时间内保持较小
- 此阶段神经元权重的方向动态近似解耦，每个神经元被其对应类的数据均值方向吸引
- 正相关类神经元对齐同类数据，排斥异类数据，最终实现**类间特征正交分离**
- 关键结果：存在时间 $T^*$ 使得之后不同类的特征内积恒为零

**阶段二：渐近阶段（晚期）——类内塌缩**

- 类间分离后，损失函数解耦为各类独立的子问题
- 每个子问题等价于在正相关数据上训练两层线性网络
- 利用Ji & Telgarsky (2021) 的最大间隔偏置结果，证明权重方向收敛到秩-1矩阵
- 秩-1结构意味着最后一层特征塌缩到一维子空间，实现**类内方向塌缩和自对偶性**

### 多分类扩展的技术挑战
- 二分类中输出权重 $v_j$ 是标量（符号不变），多分类中 $\mathbf{v}_j$ 是 $K$ 维向量，方向动态非平凡
- 输入和输出权重方向动态在 $\mathbb{S}^{D-1} \times \mathbb{S}^{K-1}$ 上耦合，形成高度非线性的黎曼流
- 本文引入"半局部初始化"（Assumption 4）条件，构造吸引域的不变子集
- 将最大间隔结果从二分类扩展到多分类交叉熵损失——这是重要的技术贡献

## 实验关键数据

### 实验1：MNIST数字分类验证Theorem 1

在三类MNIST数字{0,1,2}上训练两层ReLU网络，验证NC理论预测。

| 指标 | 结果 | 说明 |
|------|------|------|
| 正交可分性 | 近似满足 | 中心化后归一化相关矩阵显示同类正相关、异类负相关 |
| 输出神经元权重 | 与伪标签方向对齐 | 可视化验证 |
| 输入神经元权重 | 与对应类平均数字高度相似 | 灰度图对比 |
| 原始数据PCA（前3主成分） | 相对近似误差约61% | 原始数据分布分散 |
| **最后层特征PCA（前3主成分）** | **相对近似误差约0.2%** | 特征近乎完美塌缩到低维，验证NC |
| 特征与分类器结构 | 正交类均值 + 投影自对偶性 | 与Theorem 1预测一致 |

### 实验2：归一化层对NC的影响（修改版ResNet18）

在MNIST和CIFAR10上训练修改版ResNet18（最终线性分类器替换为两层ReLU分类器），比较不同归一化层。每组实验重复5次，报告均值和标准差。训练50个epoch。

| 数据集 | 归一化方法 | NC1（类内塌缩，越低越好） | NC2（类均值分离，越低越好） | NC3（自对偶性，越低越好） |
|--------|-----------|--------------------------|---------------------------|-------------------------|
| MNIST | None/Identity | 中等水平 | 中等水平 | 中等水平 |
| MNIST | LayerNorm | 略优于None | 略优于None | 略优于None |
| MNIST | **RMSNorm** | **显著最优** | 接近最优 | 接近最优 |
| CIFAR10 | None/Identity | 中等水平 | 中等水平 | 中等水平 |
| CIFAR10 | LayerNorm | 略优于None | 略优于None | 略优于None |
| CIFAR10 | **RMSNorm** | **显著最优** | 接近最优 | 接近最优 |

关键发现：RMSNorm归一化层显著改善了类内方向塌缩（NC1指标），验证了Theorem 1中关于归一化可促进更显著NC的推论——因为RMSNorm将特征归一化到单位范数后，方向塌缩直接转化为单点塌缩。

## 亮点

- **首次动态证明NC**：区别于之前仅分析优化景观的静态方法，首次完整分析GF动态过程证明训练确实收敛到NC，建立了NC与隐式偏置的显式桥梁
- **两阶段机制清晰优雅**：早期对齐阶段（小初始化 → 神经元方向对齐 → 类间分离）+ 晚期最大间隔阶段（权重方向收敛 → 秩-1 → 类内塌缩），物理图像直观
- **ReLU导致的"修正NC"**：揭示ReLU非线性产生正交类均值（非simplex ETF）和方向塌缩（非单点塌缩），纠正了UFM的过度简化，更贴近实际
- **RMSNorm的理论支持**：理论预测并实验验证RMSNorm可促进更强的NC，为归一化层设计提供理论指导
- **多分类隐式偏置扩展**：将GF最大间隔偏置的分析从二分类扩展到多分类交叉熵损失，填补了隐式偏置理论的空白

## 局限与展望

- **正交可分性假设较强**：要求同类正相关、异类负相关，现实数据仅近似满足
- **多分类需更强条件**：Proposition 1需要额外的数据条件限制（涉及数据范数比和分离度），是分析工具的限制
- **半局部初始化假设（Assumption 4）**：要求每个神经元权重与某类的（类均值-伪标签）对良好对齐，比随机初始化更受限
- **仅限浅层ReLU网络**：两层网络表达能力有限，深层网络的NC动态分析仍是开放问题
- **梯度流 vs 梯度下降**：GF是连续时间极限，有限步长GD的收敛保证未涉及
- **无偏置网络**：虽附录中讨论了扩展到有偏置的情况，但主定理未直接覆盖
- **NC度量的实用性**：方向塌缩而非单点塌缩，在实际应用中对泛化的影响尚不清楚

## 与相关工作的对比

- **UFM系列 (Mixon et al. 2022, Zhu et al. 2021, Zhou et al. 2022等)**：分析无约束特征模型的全局最优解，本文推进到有约束的真实网络设定，揭示数据/ReLU对NC特征的影响
- **Tirer & Bruna (2022)**：UFM中引入ReLU约束，得到正交类均值结论与本文一致，但仍是静态景观分析
- **Jacot et al. (2025)**：宽网络+权重衰减下证明收敛到NC，本文不需要显式正则化或宽度过参数化
- **Phuong & Lampert (2021), Min et al. (2024)**：研究浅层ReLU网络的神经元对齐动态但不涉及NC，本文在其基础上建立了对齐与NC的联系
- **Ji & Telgarsky (2019, 2020)**：研究深度线性网络的最大间隔偏置，本文将其结果应用于类间分离后的子问题分析
- **Hong & Ling (2024)**：浅层网络+一般数据的NC，但未分析动态收敛

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次从训练动态角度证明NC，建立NC与隐式偏置的显式联系
- 实验充分度: ⭐⭐⭐ — MNIST验证清晰，ResNet实验有启发性，但规模和场景覆盖有限
- 写作质量: ⭐⭐⭐⭐⭐ — 二分类到多分类的递进式展开极为清晰，理论与直觉兼顾
- 价值: ⭐⭐⭐⭐ — 理论贡献突出，对理解NC机制有重要推动，但强假设限制了直接应用

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](../../ICML2025/llm_pretraining/bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)

<!-- RELATED:END -->
