---
title: >-
  [论文解读] Wasserstein Transfer Learning
description: >-
  [NeurIPS 2025][优化][迁移学习] 提出了首个针对Wasserstein空间中概率分布输出的迁移学习框架（WaTL），通过加权辅助估计、偏差校正和投影三步法，结合自适应信息源选择，从源域迁移知识以提升目标域分布回归的估计性能。
tags:
  - NeurIPS 2025
  - 优化
  - 迁移学习
  - Wasserstein空间
  - Fréchet回归
  - 最优传输
  - 分布数据分析
---

# Wasserstein Transfer Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.17404](https://arxiv.org/abs/2505.17404)  
**代码**: [GitHub](https://github.com/h7nian/WaTL)  
**领域**: 优化  
**关键词**: 迁移学习, Wasserstein空间, Fréchet回归, 最优传输, 分布数据分析

## 一句话总结

提出了首个针对Wasserstein空间中概率分布输出的迁移学习框架（WaTL），通过加权辅助估计、偏差校正和投影三步法，结合自适应信息源选择，从源域迁移知识以提升目标域分布回归的估计性能。

## 研究背景与动机

迁移学习在图像、文本等欧氏空间数据上已取得巨大成功，但现有方法几乎都假设数据位于欧氏空间，对于**概率分布作为输出**的回归模型束手无策。在死亡率分析、温度研究、体力活动监测等场景中，观测对象本身就是概率分布，这些分布自然地存在于 Wasserstein 空间中。

Wasserstein 空间是一个测地度量空间，**缺乏传统线性结构**——两个密度函数的和不再是有效密度。因此，标准迁移学习方法（如高维线性模型迁移、非参回归迁移）无法直接套用。此前虽有利用最优传输度量进行域自适应的工作，但没有人系统性地研究输出为概率分布时的迁移学习问题。

本文的核心动机是：当目标域数据稀少时，如何从多个源域借力，利用 Wasserstein 度量来衡量域间差异，实现概率分布空间中的迁移学习？

## 方法详解

### 整体框架

WaTL 方法建立在 **Fréchet 回归** 基础上，将目标回归函数定义为条件 Fréchet 均值：

$$m_G^{(0)}(x) = \arg\min_{\mu \in \mathcal{W}} E\{s_G^{(0)}(x) d_{\mathcal{W}}^2(\nu^{(0)}, \mu)\}$$

核心思路是三步法：加权聚合 → 偏差校正 → Wasserstein 投影。

### 关键设计

1. **加权辅助估计器（Step 1）**: 将目标域和所有源域的信息按样本量加权聚合。具体地，对每个域 $k$ 计算 $\hat{f}^{(k)}(x) = n_k^{-1} \sum_{i=1}^{n_k} s_{iG}^{(k)}(x) F_{\nu_i^{(k)}}^{-1}$，然后加权平均得到 $\hat{f}(x) = \frac{1}{n_0 + n_{\mathcal{A}}} \sum_{k=0}^{K} n_k \hat{f}^{(k)}(x)$。这一步利用了所有域的信息但可能引入偏差。

2. **偏差校正（Step 2）**: 利用目标域数据对聚合估计进行正则化校正。求解 $\hat{f}_0(x) = \arg\min_{g \in L^2(0,1)} \frac{1}{n_0}\sum_{i=1}^{n_0} s_{iG}^{(0)}(x) \|F_{\nu_i^{(0)}}^{-1} - g\|_2^2 + \lambda \|g - \hat{f}(x)\|_2$。正则化项 $\lambda$ 平衡了目标域精度与辅助域信息的贡献。

3. **Wasserstein 投影（Step 3）**: 将校正后的估计投影至 Wasserstein 空间，确保输出是合法的概率分布。$\hat{m}_G^{(0)}(x) = \arg\min_{\mu \in \mathcal{W}} \|F_\mu^{-1} - \hat{f}_0(x)\|_2$。投影的唯一性由 $\mathcal{W}$ 作为 $L^2(0,1)$ 闭凸子集的性质保证。

4. **自适应信息源选择（AWaTL）**: 当不知道哪些源域有用时，AWaTL 通过计算经验差异分数 $\hat{\psi}_k = \|\hat{f}^{(0)}(x) - \hat{f}^{(k)}(x)\|_2$ 来排序源域，选取差异最小的 $L$ 个源域作为信息集。$L$ 可通过交叉验证确定。

### 损失函数 / 训练策略

- 正则化参数 $\lambda$ 通过五折交叉验证在 $[0, 3]$ 范围内选取
- 理论最优正则化量级为 $\lambda \asymp n_0^{-1/2+\epsilon}$
- 核心理论（Theorem 2）给出收敛率：$d_{\mathcal{W}}^2(\hat{m}_G^{(0)}(x), m_G^{(0)}(x)) = O_p(n_0^{-1/2+\epsilon}(\psi + (n_0+n_\mathcal{A})^{-1/2}))$

## 实验关键数据

### 主实验（模拟实验）

| 目标样本量 $n_0$ | Only Target RMSPR | Only Source RMSPR | WaTL RMSPR ($\tau=100$) | WaTL RMSPR ($\tau=200$) |
|---|---|---|---|---|
| 200 | ~0.30 | ~0.25 | ~0.15 | ~0.12 |
| 400 | ~0.18 | ~0.25 | ~0.12 | ~0.10 |
| 800 | ~0.12 | ~0.25 | ~0.09 | ~0.08 |

### 真实数据实验（NHANES体力活动数据）

| 比较方法 | 女性 RMSPR | 男性 RMSPR |
|---|---|---|
| Only Target | 较高 | 较高 |
| WaTL | 显著更低 | 显著更低 |

| 配置 | 说明 |
|---|---|
| 信息源选择（AWaTL） | $\psi > 0.6$ 时，信息源1和2的选择率接近100% |
| 负迁移阈值 | $\psi_1 \geq 0.9$ 时会产生负迁移 |
| 源数据量影响 | 源样本从 $\tau=100$ 增到 $\tau=200$，WaTL性能持续提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| K=1, $\psi_1 < 0.9$ | WaTL优于Only Target | 源域足够相似时迁移有益 |
| K=1, $\psi_1 \geq 0.9$ | Only Target更优 | 源域过于不同时产生负迁移 |
| AWaTL, L=2 | 选择率→100% | 随着非信息源差异增大，信息源被准确识别 |

### 关键发现

- 目标样本小时WaTL优势最明显，RMSPR降低约50%
- AWaTL在源域差异 $\psi > 0.6$ 时达到100%正确识别
- 源数据量增大时WaTL收益持续增长，验证了理论收敛率

## 亮点与洞察

- **首次将迁移学习扩展到Wasserstein空间**，填补了分布数据迁移学习的理论空白
- 理论结果具有一般性：Theorem 1的收敛率可推广至网络、正定矩阵、树等其他度量空间输出
- AWaTL的自适应选择机制优雅地解决了负迁移问题
- 利用 $d_\mathcal{W}^2(\mu_1, \mu_2) = \int_0^1 (F_{\mu_1}^{-1}(u) - F_{\mu_2}^{-1}(u))^2 du$ 将Wasserstein度量转化为 $L^2$ 度量进行计算

## 局限与展望

- 目前仅处理**一维分布**，多维分布需要Sinkhorn或Sliced Wasserstein距离
- 理论分析假设完整观测分布，实际中通常只有有限采样
- 正则化参数选择依赖交叉验证，计算开销较大
- 未考虑源域样本量不均衡的最优加权策略

## 相关工作与启发

- 本文将Fréchet回归与迁移学习框架巧妙结合，为其他非欧氏空间的迁移学习提供了范式
- 证明技术基于经验过程理论，可为后续工作提供参考
- 对于多模态学习中分布特征的迁移具有启示意义

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性研究Wasserstein空间迁移学习，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ 模拟和真实数据结合，但真实数据实验偏少
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰
- 价值: ⭐⭐⭐⭐ 为分布数据分析开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [Statistical and Computational Guarantees of Kernel Max-Sliced Wasserstein Distances](../../ICML2025/optimization/statistical_and_computational_guarantees_of_kernel_max-sliced_wasserstein_distan.md)
- [Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [Learning from Interval Targets](learning_from_interval_targets.md)
- [Learning Parameterized Skills from Demonstrations](learning_parameterized_skills_from_demonstrations.md)
- [Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)

<!-- RELATED:END -->
