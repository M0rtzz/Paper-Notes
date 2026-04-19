---
title: >-
  [论文解读] Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg
description: >-
  [优化] 从 NTK 理论出发，证明 FedAvg 中数据异质性导致的模型发散上界为 $\mathcal{O}(n^{-1/2})$（$n$ 为网络宽度），在无穷宽极限下全局和局部模型均线性化，FedAvg 在相同迭代次数下等价于集中式梯度下降，泛化性能一致。
tags:
  - 优化
---

# Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg

- **会议**: ICML 2025
- **arXiv**: [2508.12576](https://arxiv.org/abs/2508.12576)
- **代码**: [kkhuge/ICML2025](https://github.com/kkhuge/ICML2025)
- **领域**: 联邦学习 / 过参数化理论 / NTK
- **关键词**: FedAvg, 数据异质性, 网络宽度, Neural Tangent Kernel, 过参数化, 模型发散, 联邦学习收敛

## 一句话总结

从 NTK 理论出发，证明 FedAvg 中数据异质性导致的模型发散上界为 $\mathcal{O}(n^{-1/2})$（$n$ 为网络宽度），在无穷宽极限下全局和局部模型均线性化，FedAvg 在相同迭代次数下等价于集中式梯度下降，泛化性能一致。

## 研究背景与动机

联邦学习（FL）允许去中心化客户端在不共享数据的情况下协作训练模型。核心挑战在于客户端数据的 **非独立同分布（non-IID）** 特性——由用户行为、地理差异和设备特定模式导致的分布偏移，引起局部优化方向分歧，降低全局模型的收敛性和泛化能力。

**现有工作的局限**：
1. 许多收敛分析依赖于**严格假设**：凸损失函数、有界梯度相似性、有界梯度等，实际中难以满足
2. FedProx、SCAFFOLD 等方法需要复杂的超参数调优或约束放松
3. 过参数化 FL 的研究（如 FL-NTK）大多局限于两层网络，模型容量受限
4. Song et al. (2023) 仅关注训练损失收敛，未分析泛化性能

**核心问题**：增加网络宽度能否本质地缓解 FL 中数据异质性的影响？

## 方法详解

### 整体框架

考虑标准 FL 设置：$M$ 个客户端、$L$ 层全连接网络、MSE 损失、FedAvg 聚合。每个客户端执行 $\tau$ 次本地 GD 迭代后上传参数聚合。

**模型发散度量**：量化数据异质性影响的指标为

$$\sum_{i=1}^{M} p_i \|\Delta \theta_i^{(t+1)\tau}\|_2 = \sum_{i=1}^{M} p_i \|\theta_i^{t\tau+\tau} - \theta^{(t+1)\tau}\|_2$$

IID 数据下此值趋近零，non-IID 数据下此值随异质性增大而增大。

### 关键理论结果一：模型发散界（Theorem 1）

**假设条件**（均为过参数化分析的标准假设）：
- 最小宽度 $n$ 足够大
- 解析 NTK 矩阵 $\Theta$ 满秩（最小特征值 $\lambda_m > 0$）
- 输入数据范数有界 $\|x\|_2 \leq 1$
- 激活函数 Lipschitz 连续

**核心不等式**：模型发散上界

$$\sum_{i=1}^{M} p_i \|\Delta \theta_i^{(t+1)\tau}\|_2 \leq \zeta \triangleq \frac{2\eta_0 \tau C R_0}{\sqrt{n}(1-q)}$$

其中 $q = 1 - \frac{\eta_0 \tau \lambda_m}{3|\mathcal{D}|} + \frac{\eta_0^2 \tau^2 C^4}{2} e^{\eta_0 \tau C^2}$。

**关键含义**：
- $\zeta = \mathcal{O}(n^{-1/2})$：增大宽度直接减小异质性影响
- $n \to \infty$ 时 $\zeta \to 0$：异质性影响完全消失，收敛恢复线性速率

**训练误差收敛**：

$$\|g(\theta^{t\tau})\|_2 \leq q^t R_0 + \frac{2\eta_0 \tau C C_1 R_0 \zeta (1-q^t)}{(1-q)^2}$$

$\zeta > 0$ 使收敛不再是纯线性的，但 $n \to \infty$ 时恢复线性收敛且训练误差趋零。

**NTK 稳定性**：全局和局部 NTK 变化幅度均为 $\mathcal{O}(n^{-1/2})$，宽度趋无穷时 NTK 保持常数——将集中式学习的 lazy training 现象推广到 FL。

### 关键理论结果二：线性化与等价性（Theorem 2 & 3）

**Theorem 2**：$n \to \infty$ 时，全局和局部模型均可用一阶 Taylor 展开的线性模型逼近：

$$\sup_{t \geq 0} \|f^{\text{lin}}(\theta^{t\tau}) - f(\theta^{t\tau})\|_2 = \mathcal{O}(n^{-1/2})$$

**Theorem 3**：无穷宽 FedAvg 的全局参数和输出有闭式解：

$$\theta^{t\tau} = -\frac{1}{n} J(\theta^0)(\Theta^0)^{-1}\left(I - e^{-\frac{\eta_0 t\tau}{|\mathcal{D}|}\Theta^0}\right)g(\theta^0) + \theta^0$$

**等价性结论**（式 42）：当集中式 GD 总迭代次数 $t' = t\tau$ 时：

$$\theta_{\text{cen}}^{t'} = \theta^{t\tau}, \quad f(x, \theta_{\text{cen}}^{t'}) = f(x, \theta^{t\tau})$$

即**无穷宽 FedAvg 与集中式 GD 产生完全相同的模型参数和输出**，泛化性能等价。

### 证明框架

使用**数学归纳法**证明 Theorem 1：
1. 证明全局和局部 Jacobian 的 Lipschitz 连续性
2. 将 GD 近似为梯度流（因学习率 $\eta = \eta_0/n$ 很小）
3. 利用积分中值定理建立本地模型参数变化的递推关系
4. 通过 Taylor 展开建立全局误差的递推不等式

## 实验关键数据

### 主实验：网络宽度 vs. non-IID 影响

| 模型族 | 宽度因子 $k$ | IID→non-IID 精度下降 |
|--------|-------------|---------------------|
| FNN1 | 1 | -17.4% |
| FNN2 | 2 | -9.5% |
| FNN4 | 4 | -6.3% |
| FNN16 | 16 | **-2.0%** |
| CNN1 | 1 | -44.9% |
| CNN2 | 2 | -26.7% |
| CNN8 | 8 | -5.1% |
| CNN32 | 32 | **-2.4%** |
| ResNet1 | 1 | -44.6% |
| ResNet4 | 4 | -18.7% |
| ResNet16 | 16 | **-14.8%** |

**关键结论**：网络越宽，non-IID 的影响越小。FNN32 和 CNN32 在 IID 和 non-IID 上的收敛曲线几乎重合。

### NTK 和参数稳定性验证

实验在 mini-MNIST 上用 GD+MSE 训练 FNN 验证：
- 网络越宽，全局和局部 NTK 变化越小
- 模型参数的更新幅度越小（lazy training 行为）
- FNN512 的全局/局部模型输出与对应线性模型几乎完全一致

### FedAvg vs. 集中式学习

在 mini-CIFAR-10 上，$\tau=2$ 和 $\tau=5$ 时 FedAvg 与集中式学习的训练/测试损失几乎完全重合，验证了 Theorem 3 的等价性。

## 亮点与洞察

1. **首次建立网络宽度与异质性影响的量化关系**：$\mathcal{O}(n^{-1/2})$ 的发散界是清晰、可操作的理论指导
2. **将 NTK 理论从集中式推广到 FL**：证明全局和局部 NTK 在宽网络中保持常数
3. **"不需要假设"的优雅性**：不依赖凸性、有界梯度等常见的限制性假设
4. **实践启示明确**：
    - 面对严重异质性数据时，**加宽网络**是一种简单有效的缓解策略
    - 无穷宽极限下，FL 客户端可以只传输模型输出而非参数，大幅降低通信开销
5. **跨架构一致性**：FNN、CNN、ResNet 均验证了理论预测

## 局限性

1. **理论依赖无穷宽假设**：实际网络宽度有限，理论结果是渐近性的，有限宽度下差距可能显著
2. **仅考虑 MSE 损失和 GD**：理论推导要求 MSE 损失和 GD，未覆盖实际中常用的交叉熵+SGD（虽然实验中用了 SGD+CE 验证趋势一致）
3. **学习率限制**：$\eta = \eta_0/n$ 在宽网络中极小，梯度流近似的精度取决于此
4. **实验规模有限**：仅在 MNIST 和 CIFAR-10 上验证，缺少大规模真实 FL 场景
5. **未考虑通信效率**：更宽的网络意味着更多参数需要传输，可能抵消宽度带来的收敛优势
6. **全参与假设**：假设所有客户端每轮都参与，未考虑部分参与场景

## 相关工作

- **FL 中数据异质性**：FedProx（Li et al., 2020）、SCAFFOLD（Karimireddy et al., 2020）、FedNova（Wang et al., 2020b）
- **过参数化 FL**：FL-NTK（Huang et al., 2021，限两层）、Song et al.（2023，线性收敛至零训练损失但未分析泛化）
- **NTK 理论**：Jacot et al.（2018）、Lee et al.（2019）宽网络线性化理论
- **通信高效 FL**：FedMA（Wang et al., 2020a）层级匹配聚合

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献突出——首次定量建立宽度与异质性的关系，证明无穷宽等价性优雅且有意义。但理论与实践之间存在较大鸿沟（无穷宽假设、MSE+GD 限制、学习率趋零），实验规模偏小。对理论社区有重要价值，但实践指导力（"把网络变宽"）相对有限。

<!-- RELATED:START -->

## 相关论文

- [The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)

<!-- RELATED:END -->
