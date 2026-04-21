---
title: >-
  [论文解读] VIKING: Deep Variational Inference with Stochastic Projections
description: >-
  [NeurIPS 2025][优化][变分推断] VIKING 提出了一种基于 Fisher-Rao 度量核空间与像空间分解的变分近似后验族，通过随机交替投影算法实现可扩展的全相关贝叶斯训练，在多个基准上超越了现有贝叶斯深度学习方法。
tags:
  - NeurIPS 2025
  - 优化
  - 变分推断
  - 贝叶斯深度学习
  - 过参数化
  - Fisher-Rao度量
  - 随机交替投影
---

# VIKING: Deep Variational Inference with Stochastic Projections

**会议**: NeurIPS 2025  
**arXiv**: [2510.23684](https://arxiv.org/abs/2510.23684)  
**代码**: [GitHub](https://github.com/fadel/viking)  
**领域**: 优化  
**关键词**: 变分推断, 贝叶斯深度学习, 过参数化, Fisher-Rao度量, 随机交替投影

## 一句话总结

VIKING 提出了一种基于 Fisher-Rao 度量核空间与像空间分解的变分近似后验族，通过随机交替投影算法实现可扩展的全相关贝叶斯训练，在多个基准上超越了现有贝叶斯深度学习方法。

## 研究背景与动机

贝叶斯深度学习的核心困难在于**过参数化**带来的多对一映射——不同参数配置可以描述完全相同的函数。例如 $f(x) = w_1 \mathrm{ReLU}(w_2 x)$ 可以重参数化为 $f(x) = w_1/\gamma \cdot \mathrm{ReLU}(\gamma w_2 x)$，参数维度与自由度之间存在巨大差距，且随模型规模增大而加剧。

传统的均匀场（mean-field）近似假设参数之间独立，无法反映过参数化带来的参数间的强相关结构，导致实际应用中常出现训练不稳定、预测质量差和校准性欠佳的问题。作者观察到 IVON（当前最先进的均匀场方法）学出的方差几乎对所有权重完全一致，暗示其近似后验极不准确。

核心动机是：如果贝叶斯深度学习要成功，近似后验**必须**反映过参数化的几何结构。Roy et al. (2024) 证明描述同一函数的参数构成权重空间中的连续连通集，可以通过 Fisher-Rao 度量的核空间来刻画。

## 方法详解

### 整体框架

VIKING（Variational Inference with Kernel- and Image-spaces of numerical Gauss-Newton matrices）将参数空间分解为 Fisher-Rao 度量的核空间（ker）和像空间（im），分别用一个标量方差控制，构建一个简单但全相关的近似后验。

### 关键设计

1. **变分族设计**: 近似后验 $q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \hat{\boldsymbol{\theta}}, \boldsymbol{\Sigma}_{\hat{\boldsymbol{\theta}}})$，其中协方差矩阵为：
   $$\boldsymbol{\Sigma}_{\hat{\boldsymbol{\theta}}} = \sigma_{\ker}^2 \mathbf{U}_{\hat{\boldsymbol{\theta}}} \mathbf{U}_{\hat{\boldsymbol{\theta}}}^\top + \sigma_{\mathrm{im}}^2 (\mathbb{I} - \mathbf{U}_{\hat{\boldsymbol{\theta}}} \mathbf{U}_{\hat{\boldsymbol{\theta}}}^\top)$$
   这里 $\mathbf{U}_{\hat{\boldsymbol{\theta}}}$ 是核空间的正交基。$\sigma_{\ker}^2$ 控制训练数据支撑外的不确定性，$\sigma_{\mathrm{im}}^2$ 控制训练数据上的不确定性。虽然仅有两个标量参数，但由于包含投影矩阵 $\mathbf{U}\mathbf{U}^\top$，实际上捕获了所有参数之间的全相关结构。

2. **ELBO 优化**: ELBO 包含重构项和 KL 项。KL 项可以闭式计算（因为先验和变分分布均为高斯分布），关键量只需核空间维度 $R$，可通过 Hutchinson 迹估计器估算。重构项通过从 $q(\boldsymbol{\theta})$ 采样并计算对数似然的蒙特卡洛估计。采样的核心难点在于向核空间的投影。

3. **随机交替投影算法**: 这是 VIKING 的核心计算创新。向 Fisher-Rao 核空间投影等价于求解约束最小二乘问题 $\boldsymbol{\epsilon}_{\ker} = \arg\min_{\mathbf{u}} \|\mathbf{u} - \boldsymbol{\epsilon}\|^2 \ \text{s.t.} \ \mathbf{J} \mathbf{u} = \mathbf{0}$，需要求解 $N \times N$ 线性系统，对整个数据集操作。原始的交替投影算法需要多次遍历整个数据集来投影一个向量，不适合小批量优化。作者提出随机扩展：
   $$\boldsymbol{\epsilon}^{(t)} = \mathbf{U}^{(t)} \mathbf{U}^{(t)\top} (\sqrt{\gamma} \boldsymbol{\epsilon}^{(t-1)} + \sqrt{1-\gamma} \boldsymbol{\eta}^{(t)})$$
   超参数 $\gamma \in [0,1]$ 控制历史信息保持程度。$\gamma=1$ 为无噪声的朴素方法，$\gamma=0$ 为仅依赖当前批次。中间值实现了滑动窗口效果，实验表明 $\gamma=0.5$ 附近效果最好。

### 训练策略

- 使用共轭梯度法（带完全再正交化）无矩阵地求解线性系统，避免显式构造 $D \times D$ 矩阵。
- 支持预训练热启动（warmup）：先用最大似然估计训练，再切换到 ELBO 优化，可显著加速收敛。
- 存在一个最佳切换点：如果最大似然完全收敛后再切换，ELBO 优化可能陷入局部最优。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 准确率↑ | NLL↓ | ECE↓ | MCE↓ |
|--------|------|---------|------|------|------|
| MNIST | MAP | 0.986 | 0.070 | 0.247 | 0.861 |
| MNIST | IVON | 0.989 | **0.043** | **0.077** | **0.651** |
| MNIST | **VIKING** | **0.991** | 0.055 | 0.096 | 0.690 |
| Fashion MNIST | MAP | 0.883 | 0.410 | 0.153 | 0.590 |
| Fashion MNIST | IVON | 0.897 | 0.335 | **0.073** | 0.683 |
| Fashion MNIST | **VIKING** | **0.900** | 0.332 | 0.075 | **0.611** |
| SVHN | MAP | 0.947 | 0.201 | 0.055 | 0.608 |
| SVHN | IVON | 0.943 | 0.302 | 0.082 | 0.492 |
| SVHN | **VIKING** | **0.960** | **0.177** | **0.028** | **0.308** |
| CIFAR-10 | MAP | 0.824 | 0.536 | 0.075 | 0.619 |
| CIFAR-10 | **VIKING** | 0.877 | 0.407 | **0.041** | **0.331** |
| Imagenette | **VIKING** | **0.887** | **0.403** | 0.077 | 0.612 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| $\gamma=0.0$（纯噪声） | 训练准确率一般，泛化适中 | 每步仅依赖当前批次投影 |
| $\gamma=0.5$（混合） | 最佳泛化性能 | 历史投影与新噪声平衡 |
| $\gamma=1.0$（无噪声） | 训练准确率最高但泛化差 | 投影样本每个epoch仅更新一次 |
| 后验调优 vs. 全训练 | 全训练更优 | 但后验调优也能产生合理结果 |
| 预训练热启动 | 显著加速 | 但完全收敛后切换可能有问题 |

### 关键发现

- 在 SVHN 和 CIFAR-10 上，VIKING 的校准指标（ECE、MCE）大幅优于所有基线，因为这些数据集上模型过参数化更严重。
- OOD 检测（MNIST→FMNIST/KMNIST/EMNIST）中，VIKING 的 AUROC 显著优于基线方法。
- 可扩展到 ResNet34（21.7M 参数）在 Imagenette 上的训练。

## 亮点与洞察

- **极简但有效的设计**: 仅用两个标量参数就能校准深度神经网络的不确定性，核心在于通过投影矩阵隐式捕获全相关结构。
- **理论与实践的桥梁**: 将过参数化的微分几何理论转化为实用的变分推断算法。
- **随机交替投影的创新**: 巧妙地将后验更新与 ELBO 优化融合在一起，使之兼容小批量训练。
- 玩具回归实验直观展示了 VIKING 的优越性——IVON 的后验样本无法反映边界处的不确定性。

## 局限与展望

- 核空间投影的计算成本较高，每个训练步需要多次共轭梯度迭代。
- 仅用两个标量可能过于简化——在大型模型上可能需要更灵活的核/像空间方差结构。
- CIFAR-10 上准确率仍落后于 Last Layer LA（0.877 vs 0.894），说明全相关后验并不总是最优选择。
- 论文主要在小到中等规模模型上验证，更大规模（如 BERT、GPT）上的可扩展性有待确认。

## 相关工作与启发

- 与 IVON（均匀场 ELBO 优化）直接对比，说明考虑过参数化结构至关重要。
- Miani et al. (2025) 的后验投影方法是 VIKING 的直接前身，VIKING 将其从事后近似扩展到完整训练。
- 为"贝叶斯深度学习是否有前途"这一争论提供了积极证据。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将过参数化几何结构融入变分推断，思路非常巧妙
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多基线比较，但缺少大模型实验
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 为贝叶斯深度学习提供了新的有效范式

<!-- RELATED:START -->

## 相关论文

- [Least Squares Variational Inference](least_squares_variational_inference.md)
- [Brain-like Variational Inference](brain-like_variational_inference.md)
- [NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

<!-- RELATED:END -->
