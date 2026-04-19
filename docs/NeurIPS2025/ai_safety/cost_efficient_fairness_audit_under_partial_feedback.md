---
title: >-
  [论文解读] Cost Efficient Fairness Audit Under Partial Feedback
description: >-
  [NeurIPS 2025][AI安全][fairness audit] 在部分反馈（partial feedback）设定下，提出了一套包含新颖成本模型的公平性审计框架，分别在黑盒与混合模型两种场景给出近最优审计算法，审计成本比自然基线降低约 50%。
tags:
  - NeurIPS 2025
  - AI安全
  - fairness audit
  - partial feedback
  - equalized odds
  - rejection sampling
  - exponential family mixture
---

# Cost Efficient Fairness Audit Under Partial Feedback

**会议**: NeurIPS 2025  
**arXiv**: [2510.03734](https://arxiv.org/abs/2510.03734)  
**代码**: 随论文提供（supplementary）  
**领域**: ai_safety  
**关键词**: fairness audit, partial feedback, equalized odds, rejection sampling, exponential family mixture  

## 一句话总结

在部分反馈（partial feedback）设定下，提出了一套包含新颖成本模型的公平性审计框架，分别在黑盒与混合模型两种场景给出近最优审计算法，审计成本比自然基线降低约 50%。

## 背景与动机

机器学习分类器在贷款审批、犯罪风险评估、大学录取等高风险决策中广泛应用，但可能放大训练数据中的偏见。**公平性审计（fairness audit）**旨在检验分类器是否满足特定的公平指标（如 equalized odds）。

现实中存在一个关键困难：**部分反馈（partial feedback）**——真实标签仅对被正向分类的个体可观测。例如银行贷款场景中，只有获批贷款的申请者才有还款记录，被拒绝者的还款能力完全未知。

已有工作要么研究部分反馈下的公平分类器学习，要么在完全标签下做公平审计，但**没有工作在部分反馈设定下研究成本敏感的公平性审计**。此外，获取缺失标签本身是有代价的——强行放贷给被拒者若发生违约则产生实际损失，这一成本不对称性此前未被建模。

## 核心问题

给定分类器 $f: \mathcal{X} \times \mathcal{A} \to \{0,1\}$，在部分反馈下以尽可能低的审计成本完成 $(\gamma, \varepsilon, \delta)$-fairness audit：

- **公平假设** $\mathsf{FAIR}$：equalized odds difference $\Delta \leq \gamma$
- **不公平假设** $\mathsf{UNFAIR}$：$\Delta > \gamma + \varepsilon$

**审计成本模型**：审计者观察在线样本，对被负向分类 ($f=0$) 的个体：

- 请求特征需支付 $c_{feat}$
- 请求标签且标签为负 ($Y=0$) 需额外支付 $c_{lab}$（如贷款违约损失）

实际中 $c_{lab} \gg c_{feat}$，该成本模型精准刻画了现实中获取负面结果标签的高昂代价。

## 方法详解

### 1. 黑盒模型——Baseline（自然基线）

对所有 $f=0$ 的个体请求标签，利用停止时间（stopping time）构造保证估计精度。审计成本上界：

$$\text{cost} \leq \widetilde{O}\left(\sum_{a \in \mathcal{A}} \sum_{y \in \{0,1\}} \frac{c_{feat} \cdot \mathbb{P}[f=0] + c_{lab} \cdot \mathbb{P}[f=0, Y=0]}{q_{y,a} \cdot \varepsilon^2}\right)$$

**缺点**：对所有个体无差别请求标签，浪费了 $A \neq a$ 群体的请求成本。

### 2. 黑盒模型——RS-Audit（拒绝采样审计）

**核心洞察**：估计 $\mathbb{P}[f=1|Y=y, A=a]$ 只需条件概率 $\mathbb{P}[Y=y|A=a]$，无需联合概率 $\mathbb{P}[Y=y, A=a]$。因此对于群体 $a$，可以**忽略 $A \neq a$ 的样本**（拒绝采样），仅对目标群体请求标签。

**算法流程**：

1. 对每个 $(y, a)$，先在历史正类数据库 $D$ 上通过 Past 子程序估计 $p_{y|a} = \mathbb{P}[f=1, Y=y | A=a]$
2. 在线数据中通过 Online 子程序，仅观察 $A=a$ 的个体，收集标签直到累计 $\tau$ 个 $Y=y$ 样本
3. 利用 Negative Binomial 分布的集中性质保证估计精度
4. 计算 $\hat{\Delta}$，与阈值 $\gamma + \varepsilon/2$ 比较判定公平与否

**审计成本**（Theorem 2）：

$$\text{cost} \leq \widetilde{O}\left(\sum_{a \in \mathcal{A}} \sum_{y \in \{0,1\}} \frac{c_{feat} \cdot \mathbb{P}[f=0|A=a] + c_{lab} \cdot \mathbb{P}[f=0, Y=0|A=a]}{q_{y|a} \cdot \varepsilon^2}\right)$$

相比 Baseline，分子从 $\mathbb{P}[f=0]$ 降为 $\mathbb{P}[f=0|A=a]$，分母从 $q_{y,a}$ 升为 $q_{y|a}$，双重改进。

### 3. 下界（Theorem 3）

通过构造两个 KL 散度接近但公平性不同的实例族，证明任何算法的期望成本至少为 $\widetilde{\Omega}$ 量级，与 RS-Audit 的上界在对数因子内匹配。**黑盒设定的刻画本质完整。**

### 4. 混合模型——Exp-Audit

当特征分布服从指数族混合模型 $\mathbb{P}[X|Y=y, A=a] = \mathcal{E}_{\theta^*_{y,a}}(X)$ 时，引入结构假设可大幅降低成本。

**算法三步走**：

1. **截断样本学习**：将历史正类数据视为被 $f$ 截断的样本，用 TruncEst 子程序恢复分布参数 $\hat{\theta}_{y,a}$
2. **构造 MAP oracle**：利用估计参数和粗略混合权重 $\tilde{q}_{y|a}$ 构建最大后验判别器 $M_a(x) = \arg\max_y \tilde{q}_{y|a} \mathcal{E}_{\hat{\theta}_{y,a}}(x)$
3. **代理标签估计混合权重**：仅请求特征（成本 $c_{feat}$），用 MAP oracle 给代理标签，避免请求真实标签（成本 $c_{lab}$）

**审计成本**（Theorem 4）：

$$\text{cost} \leq \widetilde{O}\left(\sum_{a \in \mathcal{A}} \left(\frac{c_{feat} \cdot \mathbb{P}[f=0|A=a]}{q_{m,a} \cdot \varepsilon^2} + \sum_{y \in \{0,1\}} \frac{c_{lab} \cdot \mathbb{P}[f=0, Y=0|A=a]}{q_{y|a}}\right)\right)$$

**关键改进**：$c_{lab}$ 项与 $\varepsilon$ 无关（是问题相关常数），仅 $c_{feat}$ 项依赖 $\varepsilon^{-2}$。由于 $c_{lab} \gg c_{feat}$，总成本显著降低。

### 假设条件

- **结构假设**（Assumption 1）：正类概率下界 $\alpha > 0$、指数族的对数凹性和光滑性
- **分离假设**（Assumption 2）：$\|\theta^*_{1,a} - \theta^*_{0,a}\| \geq \Omega(\sqrt{\log(1/\varepsilon)})$，远弱于聚类要求的 $\Omega(1/\varepsilon)$

## 实验关键数据

| 数据集 | 方法对比 | 成本降低 |
|--------|---------|---------|
| Adult Income | RS-Audit vs Baseline | 约 50% 成本节省 |
| Law School | RS-Audit vs Baseline | 类似幅度的改进 |
| Adult Income | Exp-Audit vs RS-Audit | 成本显著更低，且独立于 $\varepsilon$ |
| 合成数据 | Exp-Audit vs RS-Audit | Exp-Audit 成本曲线平坦，RS-Audit 随 $\varepsilon$ 增长 |

- 黑盒实验使用 Logistic Regression 分类器
- 混合模型实验使用神经网络倒数第二层 embedding，假设多元高斯分布
- Adult Income 实验中未强制分离条件，Exp-Audit 仍表现出显著优势，体现鲁棒性

## 亮点

1. **首个在部分反馈下研究成本敏感公平审计的工作**，填补了重要空白
2. **成本模型设计精巧**：区分 $c_{feat}$ 和 $c_{lab}$，准确刻画标签获取的不对称代价
3. **黑盒设定完整刻画**：RS-Audit 上界与下界在对数因子内匹配
4. **截断样本视角**：将部分反馈数据解释为截断样本是概念性贡献，连接了两个不同领域
5. **指数族 MAP oracle 泛化**：将球面高斯的 MAP oracle 结果推广到一般指数族，具有独立理论价值
6. **算法对分类器类型无关**：适用于任意分类器的审计

## 局限与展望

1. **混合模型假设较强**：要求已知指数族类型（$h, T, W$ 已知），以及结构和分离条件
2. **仅考虑二元群体标签和二元分类**：多分类和连续敏感属性的扩展未讨论
3. **成本模型为均匀成本**：实际中不同个体的违约成本可能不同（非均匀成本）
4. **未知分类器场景未覆盖**：当分类器本身未知（如来自某假设类）时的审计复杂度是开放问题
5. **实验规模有限**：仅在两个标准数据集上验证，缺少大规模高维场景
6. **分离条件的实际可验证性**：虽然算法可在线检测是否满足分离条件并切换，但切换的额外成本未分析

## 与相关工作的对比

| 方面 | 本文 | 已有工作 |
|-----|------|---------|
| 设定 | 部分反馈 + 成本模型 | 完全标签审计 / 部分反馈分类 |
| 目标 | 最小化审计成本 | 假设检验 / 公平分类器学习 |
| 与 Bechavod et al. 2019 | 审计（检验）问题 | 在线公平分类（学习）问题 |
| 与 Keswani et al. 2024 | 成本敏感审计 | 数据收集 + 公平分类 |
| 与 Chugg et al. 2023 | 部分反馈 | 完全标签下的序贯检验 |
| 与 Si et al. 2021 | 部分反馈 + 最优成本 | 最优传输框架，完全标签 |
| 截断样本技术 | 首次用于公平审计 | 仅用于分布参数估计 |

## 启发与关联

- **截断样本与部分反馈的联系**是有启发性的概念贡献，可推广到其他部分观测场景（如推荐系统中的选择偏差审计）
- 成本模型的思路可迁移到**主动学习（active learning）**中，设计标签获取成本不对称的查询策略
- MAP oracle 的指数族推广可用于**半监督公平分类**，在缺少标签时利用特征分布结构
- 公平-准确性权衡的分析（Remark 4）暗示审计与再训练可以统一为一个框架

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个部分反馈下成本敏感公平审计，截断样本视角新颖）
- 实验充分度: ⭐⭐⭐ （数据集规模偏小，但理论验证充分）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，问题动机阐述充分）
- 价值: ⭐⭐⭐⭐ （填补重要空白，理论贡献扎实）

<!-- RELATED:START -->

## 相关论文

- [Fairness under Competition](fairness_under_competition.md)
- [Efficient Fairness-Performance Pareto Front Computation](efficient_fairness-performance_pareto_front_computation.md)
- [Accelerating Spectral Clustering under Fairness Constraints](../../ICML2025/ai_safety/accelerating_spectral_clustering_under_fairness_constraints.md)
- [InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)

<!-- RELATED:END -->
