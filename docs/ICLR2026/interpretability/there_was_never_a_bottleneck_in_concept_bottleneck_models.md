---
title: >-
  [论文解读] There Was Never a Bottleneck in Concept Bottleneck Models
description: >-
  [ICLR 2026][概念瓶颈模型] 指出概念瓶颈模型（CBM）实际上并不存在真正的"瓶颈"——表征变量 $z_j$ 能预测概念 $c_j$ 不意味着它只编码 $c_j$ 的信息。提出 MCBM（Minimal Concept Bottleneck Model），通过信息瓶颈正则化约束每个 $z_j$ 仅保留对应概念的信息，实现真正的解耦表征和可靠的概念干预。
tags:
  - ICLR 2026
  - 概念瓶颈模型
  - 信息瓶颈
  - 信息泄漏
  - 可干预性
  - 表征学习
---

# There Was Never a Bottleneck in Concept Bottleneck Models

**会议**: ICLR 2026  
**arXiv**: [2506.04877](https://arxiv.org/abs/2506.04877)  
**代码**: 无（根据论文描述）  
**领域**: 可解释性 / 概念瓶颈模型  
**关键词**: 概念瓶颈模型, 信息瓶颈, 信息泄漏, 可干预性, 表征学习

## 一句话总结

指出概念瓶颈模型（CBM）实际上并不存在真正的"瓶颈"——表征变量 $z_j$ 能预测概念 $c_j$ 不意味着它只编码 $c_j$ 的信息。提出 MCBM（Minimal Concept Bottleneck Model），通过信息瓶颈正则化约束每个 $z_j$ 仅保留对应概念的信息，实现真正的解耦表征和可靠的概念干预。

## 研究背景与动机

- **CBM 的承诺**：通过让表征的每个分量 $z_j$ 预测一个可理解的概念 $c_j$，提供可解释性和可干预性
- **信息泄漏问题**：$z_j$ 预测 $c_j$ 不等于 $z_j$ 只编码 $c_j$。极端情况下，$z_j$ 可能编码整个输入 $\mathbf{x}$ 仍然满足 CBM 约束
- **两个后果**：
  1. **可解释性受损**：$z_j$ 不能用 $c_j$ 完全解释
  2. **干预无效**：修改 $z_j$ 不仅改变 $c_j$，还影响其中编码的其他信息
- **CBM 干预的理论缺陷**：CBM 中没有从 $c_j$ 到 $z_j$ 的有向路径，$p(z_j|c_j)$ 在图模型中未定义。现有的干预通过 sigmoid 逆函数的经验分位数近似，是 ad-hoc 的。

### CBM vs MCBM 核心区别

| | VM | CBM | MCBM |
|--|-----|-----|------|
| $z_j$ 编码全部 $c_j$？ | ✗ | ✓ | ✓ |
| $z_j$ 仅编码 $c_j$？ | ✗ | ✗ | **✓** |

## 方法详解

### 1. 数据生成过程

- 输入 $\mathbf{x}$ 由概念 $\mathbf{c}$ 和 nuisance $\mathbf{n}$ 决定：$p(\mathbf{x}, \mathbf{y}, \mathbf{c}, \mathbf{n}) = p(\mathbf{x}|\mathbf{c}, \mathbf{n}) p(\mathbf{y}|\mathbf{x}) p(\mathbf{c}, \mathbf{n})$
- Nuisance 分为与任务相关的 $\mathbf{n}_y$ 和无关的 $\mathbf{n}_{\bar{y}}$

### 2. 三个信息论目标

**目标 1**（VM/CBM/MCBM 共有）：最大化 $I(Z; Y)$——表征预测目标

$$\max_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})} \left[\mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x})} \left[\log q_\phi(\hat{\mathbf{y}}|\mathbf{z})\right]\right]$$

**目标 2**（CBM/MCBM）：最大化 $I(Z_j; C_j)$——$z_j$ 充分统计量

$$\max_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, c_j)} \left[\mathbb{E}_{p_\theta(z_j|\mathbf{x})} \left[\log q_\phi(\hat{c}_j|z_j)\right]\right]$$

**目标 3**（仅 MCBM）：最小化 $I(Z_j; X | C_j)$——信息瓶颈

$$\min_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, c_j)} \left[D_{KL}\left(p_\theta(z_j|\mathbf{x}) \| q_\phi(\hat{z}_j|c_j)\right)\right]$$

这确保 $z_j$ 在给定 $c_j$ 后不再包含关于 $\mathbf{x}$ 的额外信息（Markov 链 $X \leftrightarrow C_j \leftrightarrow Z_j$）。

### 3. MCBM 训练目标

$$\max_{\theta, \phi} \sum_{k=1}^N \sum_i \log q_\phi(\hat{\mathbf{y}}|f'_\theta(x^{(k)}, \epsilon^{(i)})) + \beta \sum_{j=1}^n \log q_\phi(\hat{c}_j|f'_{\theta,j}(\mathbf{x}^{(k)}, \epsilon^{(i)})) - \gamma \sum_{j=1}^n D_{KL}(p_\theta(z_j|\mathbf{x}^{(k)}) \| q_\phi(\hat{z}_j|c_j^{(k)}))$$

- 第一项：任务预测损失
- 第二项（$\beta$ 加权）：概念预测损失
- 第三项（$\gamma$ 加权）：**信息瓶颈正则化**（MCBM 独有）

### 4. 干预机制

在 MCBM 中，干预变得有理论基础：
$$p(z_j|c_j) = q_\phi(z_j|c_j)$$

因为优化目标 3 后 $z_j$ 仅编码 $c_j$ 的信息，所以修改 $z_j$ 严格对应于对 $c_j$ 的干预。

### 5. 表征头设计

- 二值概念：$g_\phi^z(c_j) = \lambda$ if $c_j=1$, else $-\lambda$
- 多类概念：$g_\phi^z(c_j) = \lambda \cdot \text{one\_hot}(c_j)$（原型学习）
- 连续概念：$g_\phi^z(c_j) = \lambda \cdot c_j$

编码器使用随机版本 $p_\theta(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; f_\theta(\mathbf{x}), \sigma_x^2 I)$，通过重参数化技巧训练。

## 实验结果

### 信息泄漏度量：URR（不确定性缩减比）

衡量 $z$ 中编码了多少超出概念集的 nuisance 信息（越低越好）。

#### 任务相关 nuisance 泄漏

| 方法 | MPI3D | Shapes3D | CIFAR-10 | CUB | AwA2 |
|------|-------|----------|----------|-----|------|
| Vanilla | 35.0 | 45.5 | 19.8 | 3.8 | 1.5 |
| CBM | 28.1 | 18.1 | 18.5 | 3.8 | 1.4 |
| CEM | 43.2 | 15.8 | **27.2** | 3.9 | 1.1 |
| ECBM | 25.2 | 47.1 | 18.1 | 4.5 | 1.1 |
| **MCBM (high γ)** | **0.0** | **0.0** | **17.6** | **2.4** | **0.7** |

#### 任务无关 nuisance 泄漏

| 方法 | MPI3D | Shapes3D |
|------|-------|----------|
| Vanilla | 11.3 | 42.7 |
| CBM | 7.4 | 20.6 |
| CEM | 15.5 | 40.9 |
| **MCBM (任意 γ)** | **0.0** | **0.0** |

### 关键发现

1. **CEM 和 ECBM 反而加剧泄漏**：在某些数据集上泄漏比 Vanilla 模型还多
2. **MCBM 彻底消除泄漏**：high γ 下所有数据集上 nuisance 信息降至 0
3. **ARCBM 和 HCBM 无系统优势**：相比标准 CBM 并未更好地控制泄漏
4. **代价**：MCBM 的任务准确率略有下降——因为排除了 $\mathbf{n}_y$ 中对任务有用的信息

## 亮点与洞察

1. **根本性的概念批判**：指出 CBM 名不副实——从未有过真正的"瓶颈"
2. **信息瓶颈的自然引入**：用 $I(Z_j; X | C_j) = 0$ 精确形式化"仅编码概念"
3. **CBM 干预的理论缺陷分析**（Section 5）：证明 CBM 的干预假设在概率论上不成立
4. **实用的 KL 散度正则化**：在高斯假设下退化为简单的 MSE 损失
5. **解耦表征的可视化**：MCBM 的表征空间中，同概念值的样本紧密聚类

## 局限性

- 任务准确率和概念纯度之间存在固有权衡：概念集不完整时，排除 nuisance 必然降低性能
- 需要概念标注——与所有 CBM 方法共享此限制
- 连续概念的处理依赖高斯假设
- 超参数 $\gamma$ 需要调节以平衡解耦程度和任务性能
- 尚未在更大规模模型或更复杂任务上验证

## 相关工作

- **CBM 变体**：CEM（概念嵌入）、HCBM（硬瓶颈）、ARCBM（自回归）、SCBM（随机）
- **信息泄漏分析**：Margeloiu et al. 2021、Parisini et al. 2025
- **信息瓶颈**：Tishby et al. 2000、Alemi et al. 2016（变分信息瓶颈）

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 对 CBM 领域的根本性重新审视
- **技术深度**: ⭐⭐⭐⭐ — 信息论形式化严谨，变分推导清晰
- **实验充分性**: ⭐⭐⭐⭐ — 5 个数据集，8+ 种方法对比，多角度分析
- **实用价值**: ⭐⭐⭐⭐ — 为真正可解释的概念模型提供原则性解决方案

<!-- RELATED:START -->

## 相关论文

- [Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)
- [Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](../../ECCV2024/interpretability/improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)

<!-- RELATED:END -->
