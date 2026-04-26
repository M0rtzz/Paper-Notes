---
title: >-
  [论文解读] Cooperation of Experts: Fusing Heterogeneous Information with Large Margin
description: >-
  [ICML2025][专家协作] 提出 Cooperation of Experts (CoE) 框架，将异构信息编码为多重网络，通过两级专家设计与大间隔置信张量优化实现专家**协作**（而非竞争），在节点分类任务上全面超越现有 MoE 和多重网络方法。
tags:
  - ICML2025
  - 专家协作
  - 异构多重网络
  - 大间隔优化
  - 互信息最大化
  - 图结构学习
---

# Cooperation of Experts: Fusing Heterogeneous Information with Large Margin

**会议**: ICML2025  
**arXiv**: [2505.20853](https://arxiv.org/abs/2505.20853)  
**代码**: [strangeAlan/CoE](https://github.com/strangeAlan/CoE)  
**领域**: 集成学习 / 图神经网络  
**关键词**: 专家协作, 异构多重网络, 大间隔优化, 互信息最大化, 图结构学习

## 一句话总结

提出 Cooperation of Experts (CoE) 框架，将异构信息编码为多重网络，通过两级专家设计与大间隔置信张量优化实现专家**协作**（而非竞争），在节点分类任务上全面超越现有 MoE 和多重网络方法。

## 研究背景与动机

现实数据通常是异构的：多模态数据（图像+文本）、社交网络中的多种关系（友谊、家庭、职业）等。现有方法面临的核心问题：

**单一预测器的局限**：传统方法在整个多重网络上训练单一预测器，忽略了不同关系层中节点模式的固有异质性。实验显示，独立训练于各层的分类器性能差异显著（如 ACM 和 Yelp 数据集）

**MoE 的竞争机制缺陷**：Mixture of Experts 通过门控机制仅激活部分专家，限制了对异构数据丰富信息的充分利用

**两个关键挑战**：(a) 如何设计框架有效提取和整合跨网络复杂信息？(b) 训练好的专家如何协作贡献最终预测？

## 方法详解

### 整体架构

CoE 框架包含四个核心步骤：异构信息编码 → 两级专家设计 → 专家协作策略 → 置信张量优化。

### 1. 异构信息编码为多重网络

将多类型信息编码为异构多重网络 $G = \{G_1, \ldots, G_V\}$，每层包含相同节点但不同类型的连接。采用图结构学习 (GSL) 策略优化网络结构，利用 Simple Graph Convolution (SGC) 作为网络学习器：

$$H_v = \sigma\left((\tilde{D}_v^{-1/2}\tilde{A}_v\tilde{D}_v^{-1/2})^r X^v \odot W_1^v\right) \odot W_2^v$$

然后通过 KNN 重构邻接矩阵，并进行非负性、对称性和归一化后处理。

### 2. 两级专家设计

- **低级专家**：在单个网络上学习特定的关系模式，最大化 $I(G_i'; Y)$
- **高级专家**：在融合网络上捕获跨网络的高阶依赖关系

网络融合通过最大化跨网络互信息 $I(G_i'; G_j')$ 实现。专家训练损失为：

$$\hat{\mathcal{L}}_E = \sum_{i=1}^{V}\mathcal{L}_{cls}(Z^i; Y) - \sum_{i=1}^{V}\sum_{j=i+1}^{V}I_{lb}(Z^i; Z^j) - \sum_{i=1}^{V}I_{lb}(Z^i; Z^{tot}) - \sum_{i=1}^{V}\sum_{j \neq i}I_{lb}(Z^i; Z^{ij})$$

其中互信息下界 $I_{lb}$ 通过对比学习方式估计。

### 3. 大间隔协作机制

定义置信张量 $\Theta \in \mathbb{R}^{c \times c \times k}$（$c$ 为类别数，$k$ 为专家数），$\Theta_{rst}$ 量化第 $t$ 个专家对样本属于第 $r$ 类（真实第 $s$ 类）预测的可信度。最终预测：

$$\hat{y}_i = \underset{j=1\ldots c}{\arg\max}\; \mathcal{S}(\Theta g_i)_j$$

核心创新——**大间隔损失**：最大化最高与次高预测之间的间隔：

$$\mathcal{M} = \sum_{i=1}^{N}\left[Y_i^\top(Y_i \odot \mathcal{S}(\Theta g_i)) - \frac{1}{\alpha}\log\sum_{j=1}^{c} e^{\alpha(\mathcal{S}(\Theta g_i) - Y_i \odot \mathcal{S}(\Theta g_i))_j}\right]$$

用 logsumexp 函数平滑逼近非凸非光滑的 $\max_2$ 操作。总损失 $\mathcal{L} = \mathcal{C} - \eta\mathcal{M} + \hat{\mathcal{L}}_E$。

### 4. 理论保证

- **部分凸性**：$\mathcal{L}(\Theta g_i)$ 关于 $\Theta g_i$ 是凸函数
- **Lipschitz 连续性**：$L \leq 2\sqrt{c}\,k(1 + \gamma + \frac{\gamma}{c}e^\alpha)$
- **收敛性**：梯度下降在步长 $\eta \leq 1/L$ 时收敛到临界点
- **泛化界**：$\mathbb{E}[\ell_{0\text{-}1}(f)] \leq \frac{1}{n}\sum_i \ell_\gamma(f; x_i, y_i) + \frac{2B_\Theta G_e\sqrt{k}}{\gamma\sqrt{n}} + 3\sqrt{\frac{\log(2/\delta)}{2n}}$

## 实验关键数据

### 节点分类（多重网络，5 数据集）

| 方法 | ACM | DBLP | Yelp | MAG | Amazon |
|------|-----|------|------|-----|--------|
| GCN | 89.04 | 80.70 | 74.03 | 74.60 | 93.12 |
| HAN | 91.30 | 81.28 | 52.04 | OOM | OOM |
| InfoMGF | 92.81 | 91.45 | 92.01 | 77.32 | 97.78 |
| GMoE | 90.29 | 91.18 | 91.92 | 77.27 | 97.78 |
| Mowst | 85.69 | 89.69 | 91.31 | 77.40 | 97.89 |
| **CoE** | **94.21** | **92.27** | **93.40** | **78.37** | **98.01** |

CoE 在所有 5 个数据集上均取得最优，且标准差最低（如 ACM ±0.14, Amazon ±0.09）。

### 多模态分类（4 数据集，无初始图结构）

| 方法 | ESP | Flickr | IAPR | NUS |
|------|-----|--------|------|-----|
| QMF | 80.14 | 69.24 | 69.08 | 65.42 |
| CPM-Nets | 80.09 | 69.49 | 67.33 | 65.34 |
| **CoE** | **81.11** | **70.24** | **71.04** | **66.80** |

### 消融实验

| 变体 | ACM | DBLP | Yelp |
|------|-----|------|------|
| RF（随机森林替代） | 93.39 | 91.48 | 91.61 |
| WRF（加权随机森林） | 93.64 | 91.97 | 93.05 |
| w/o 高级专家 | 91.25 | 90.71 | 68.27 |
| w/o GSL | 93.60 | 91.13 | 93.14 |
| **CoE（完整）** | **94.21** | **92.27** | **93.40** |

去掉高级专家影响最大（Yelp 从 93.40 降至 68.27），证明跨网络融合至关重要。

## 亮点与洞察

1. **从竞争到协作的范式转变**：首次在多重网络中提出专家协作（而非 MoE 的竞争），所有专家都参与决策，避免了门控机制导致的信息丢失
2. **置信张量设计精巧**：$\Theta \in \mathbb{R}^{c \times c \times k}$ 同时编码专家特长和类别关系，比简单加权更具表达力
3. **大间隔优化有理论支撑**：logsumexp 对 $\max_2$ 的光滑逼近使非凸优化可行，且有收敛和泛化保证
4. **鲁棒性突出**：在 ACM 数据集上，即使 90% 边被扰动，CoE 仍保持稳定性能
5. **通用性强**：同一框架同时处理多关系网络和多模态数据，无需结构性修改

## 局限与展望

1. **可扩展性隐患**：置信张量 $\Theta \in \mathbb{R}^{c \times c \times k}$ 随类别数和专家数增长，大规模场景可能面临内存问题（部分 baseline 在 MAG 上 OOM，CoE 虽未 OOM 但未讨论计算开销）
2. **仅限分类任务**：实验仅覆盖节点分类，未验证在链接预测、图分类等其他图任务上的表现
3. **KNN 建图依赖**：对无图结构的多模态数据，用 KNN 构建邻接矩阵，K 值选择可能较敏感
4. **专家数量有限**：由于两级设计，专家数量受网络层数约束，难以像大规模 MoE 那样灵活扩展
5. **超参数 $\alpha$ 设置**：虽实验显示对 $\alpha$ 不太敏感，但 logsumexp 中 $\alpha$ 过大可能导致数值不稳定

## 评分

- 新颖性: ⭐⭐⭐⭐ — 专家协作 + 大间隔置信张量是新颖组合，从竞争到协作的视角转换有意义
- 实验充分度: ⭐⭐⭐⭐ — 9 个数据集 + 消融 + 鲁棒性 + 超参数敏感性分析，较为全面
- 写作质量: ⭐⭐⭐⭐ — 理论分析严谨，框架阐述清晰，top-down 的表述方式易于理解
- 价值: ⭐⭐⭐⭐ — 对多重网络学习和专家机制都有启发，代码已开源，可复现性好

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems](../../NeurIPS2025/llm_evaluation/small_language_models_as_compiler_experts_auto-parallelization_for_heterogeneous.md)
- [\[ICML 2025\] Disentangling and Integrating Relational and Sensory Information in Transformer Architectures](disentangling_and_integrating_relational_and_sensory_information_in_transformer_.md)
- [\[NeurIPS 2025\] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](../../NeurIPS2025/llm_evaluation/let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[ICML 2025\] G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)

<!-- RELATED:END -->
