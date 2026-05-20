---
title: >-
  [论文解读] Improving Forecasts of Suicide Attempts for Patients with Little Data
description: >-
  [NeurIPS 2025 (TS4H Workshop)][Gaussian Process] 提出 Latent Similarity Gaussian Process (LSGP)，将患者嵌入连续隐空间以捕获异质性，使数据稀少的患者能从相似患者"借用"预测趋势，从而改进基于 EMA 数据的自杀未遂预测。
tags:
  - "NeurIPS 2025 (TS4H Workshop)"
  - "Gaussian Process"
  - "Ecological Momentary Assessment"
  - "suicide risk prediction"
  - "patient heterogeneity"
  - "latent variable model"
  - "time series for health"
---

# Improving Forecasts of Suicide Attempts for Patients with Little Data

**会议**: NeurIPS 2025 (TS4H Workshop)  
**arXiv**: [2511.18199](https://arxiv.org/abs/2511.18199)  
**作者**: Genesis Hang, Annie Chen, Hope Neveux, Matthew K. Nock, Yaniv Yacoby
**机构**: Wellesley College, Harvard University
**代码**: 未开源  
**领域**: 其他  
**关键词**: Gaussian Process, Ecological Momentary Assessment, suicide risk prediction, patient heterogeneity, latent variable model, time series for health

## 一句话总结

提出 Latent Similarity Gaussian Process (LSGP)，将患者嵌入连续隐空间以捕获异质性，使数据稀少的患者能从相似患者"借用"预测趋势，从而改进基于 EMA 数据的自杀未遂预测。

## 研究背景与动机

### 问题背景

Ecological Momentary Assessment (EMA) 通过智能手机每天多次采集患者的自杀意念、情绪状态（如绝望感、愤怒、自我厌恶等17种情感状态），为实时自杀风险预测提供了数据基础。然而，利用 ML 预测自杀未遂（suicide attempts）面临三大核心挑战：

**极低基率（low base-rate）**：即使在最大规模数据集（~623名患者）中，真正发生自杀相关事件（SRE）的患者也极为稀少，最终仅77名患者满足分析条件

**患者异质性（patient heterogeneity）**：不同患者走向自杀的路径截然不同——抑郁症、焦虑症、PTSD、边缘型人格障碍等不同诊断可能对应不同机制，加上个体生活经历的差异，使得"一个模型适用所有人"的假设不成立

**数据不均衡**：患者之间的数据量差异巨大，数据少的患者（bottom 30%）在几乎所有指标上都表现更差

### 核心发现与动机

作者通过实验揭示了一个关键矛盾：

- **Single model**（对所有患者统一训练）：受数据多的患者主导，对少数据患者泛化差
- **Idiographic model**（每个患者单独训练）：性能提升显著，但少数据患者严重过拟合
- **随机分组实验**：即使随机将患者分组训练，性能也会随分组数增加而提升，说明异质性极高；而基于人口统计学的分组表现反而更差

这一矛盾促使作者寻找一种**介于 single 和 idiographic 之间的解决方案**。

## 方法详解

### Latent Similarity Gaussian Process (LSGP)

LSGP 的核心思想是：将患者嵌入一个**连续隐空间**，空间中的距离反映患者预测趋势的相似性。数据充足的患者通过训练获得优质的隐空间位置，而数据稀少的患者可以从"邻居"智能借用趋势。

**模型架构由四步组成**：

1. **隐变量先验**：每个患者 $n$ 有隐变量 $z_n \sim \mathcal{N}(0, \mathbb{I}_{D_z})$，$D_z = 3$
2. **输入增广**：将患者的 EMA 响应 $x_i$（包含17维情感+3维自杀意念）与对应患者的隐变量 $z_{n_i}$ 拼接，得到增广输入 $\hat{x}_i$
3. **高斯过程**：在增广输入空间上定义 GP 先验 $F|\hat{X}; \theta \sim \mathcal{N}(0, K_\theta(\hat{X}, \hat{X}))$
4. **Bernoulli 似然**：$y_i | f_i \sim \text{Bernoulli}(\text{sigmoid}(f_i))$，预测未来一周是否发生 SRE

### 稀疏变分推断

由于非高斯似然和大规模观测（14763条来自77名患者），作者采用 **Sparse Variational GP** 框架：

- 使用 $M = 2000$ 个 inducing points
- 变分族包括：(1) inducing points 的全协方差高斯后验 $q(U; \phi)$；(2) 每个患者隐变量的均场高斯后验 $q(z_n; \phi)$
- 通过最大化 ELBO 进行随机变分推断（SVI），mini-batch 大小 $B = 150$，学习率 0.005，训练 15000 步

### 核函数设计

核函数可分解为输入空间和隐空间的乘积：$K_\theta(\hat{X}, \hat{X}') = K^x_\theta(X, X') \cdot K^z_\theta(Z, Z')$

其中 $K^z_\theta$ 使用 ARD 核，$K^x_\theta$ 使用**状态依赖线性核**：

$$k_\theta(\hat{x}, \hat{x}') = b_\theta(z) \cdot b_\theta(z') + v_\theta(z) \cdot v_\theta(z') \cdot (x - c_\theta(z))^\top \cdot (x' - c_\theta(z'))$$

其中 $b_\theta(\cdot)$、$v_\theta(\cdot)$、$c_\theta(\cdot)$ 均为神经网络，允许不同隐空间位置的患者拥有不同的先验。

### 患者相似性分析

利用核的乘积分解特性，可以构建**患者相似性图**：

- 对患者隐变量后验均值计算 $K^z_\theta$ 协方差矩阵
- 将其视为邻接矩阵，患者为节点，边权为协方差
- 使用 **modularity** $Q \in [-1, 1]$ 定量评估人口统计学分组与学习到的相似性的对齐程度

## 实验关键数据

### 表1：主要方法对比（测试集指标）

| 方法 | Avg. Log-Likelihood (Bottom/Mid/Top/All) | ROC-AUC | PPV | Sensitivity | Specificity |
|------|----------------------------------------|---------|-----|-------------|-------------|
| **Single KNN** | N/A | 0.70±0.01 | 0.61±0.02 | 0.22±0.02 | 0.97±0.00 |
| **Single RBF-GP** | -0.62/-0.49/-0.36/-0.43 | 0.74±0.00 | 0.66±0.04 | 0.15±0.01 | 0.98±0.00 |
| **Single LR** | -0.66/-0.53/-0.38/-0.45 | 0.68±0.01 | 0.60±0.06 | 0.05±0.01 | 0.99±0.00 |
| **Idiographic KNN** | N/A | 0.78±0.01 | 0.73±0.02 | 0.34±0.03 | 0.97±0.00 |
| **Idiographic RBF-GP** | -0.52/-0.42/-0.31/-0.37 | 0.84±0.00 | 0.73±0.01 | 0.37±0.01 | 0.97±0.00 |
| **Idiographic VB-LR** | -0.47/-0.38/-0.28/-0.33 | **0.87±0.00** | 0.73±0.01 | 0.42±0.02 | 0.96±0.00 |
| **SV-LSGP** | -0.50/-0.40/-0.29/-0.35 | 0.85±0.01 | 0.73±0.01 | 0.37±0.02 | 0.97±0.00 |

**关键结论**：SV-LSGP 在仅使用简单核设计的情况下，已接近最优基线 Idiographic VB-LR，且超越其他所有基线。

### 表2：患者相似性图 Modularity 分析

| 人口统计学分组 | Modularity $Q$ | 含义 |
|---------------|----------------|------|
| Gender | 0.12 | 组内/组间相似性接近平衡 |
| Sexual Orientation | 0.13 | 组内/组间相似性接近平衡 |
| Age | 0.08 | 最接近0，年龄几乎不解释相似性 |
| Highest Education | 0.15 | 略有组内偏好但仍接近0 |

**关键结论**：所有人口统计学属性的 modularity 均接近0，表明这些属性**不能解释**患者在自杀风险预测趋势上的相似性。最大的相似性往往出现在不同组间的患者之间。

## 亮点与洞察

1. **问题洞察深刻**：通过系统实验揭示了 single vs. idiographic 的根本矛盾，以及离散分组方法的脆弱性（随机分组优于人口统计学分组），为模型设计提供了坚实的实证基础
2. **方法自然优雅**：LSGP 以连续隐空间优雅地统一了 single 和 idiographic 两个极端——当 $D_z = 0$ 退化为 single GP，当隐空间维度足够高且每个患者隐变量独立时接近 idiographic
3. **可解释性分析**：基于学习到的隐空间构建患者相似性图，并通过 modularity 定量评估人口统计学属性的解释力，为临床理解提供了新视角
4. **数据规模与质量**：基于623名参与者、每天6次EMA调查、持续3个月的大规模纵向数据集，具有较高的临床可信度
5. **负结论的价值**：明确指出人口统计学属性不能解释患者异质性，这一"负结论"本身对临床实践有重要警示意义

## 局限性

1. **仅接近最优基线**：SV-LSGP 尚未超过 Idiographic VB-LR，作者承认核函数设计尚处于初步阶段，未来需要更深入的归纳偏置探索
2. **数据限制严格**：最终仅77名患者满足至少3个SRE+3个非SRE的纳入标准，大幅减少了可用数据，泛化性存疑
3. **冷启动问题**：模型要求新患者至少有一个已记录的SRE才能使用，这在实际部署中是强假设
4. **单一核设计**：当前仅尝试了一种核函数（状态依赖线性核×ARD核），未系统探索其他核组合
5. **因果性缺失**：模型本质上是相关性预测，无法解释"为什么"某些患者相似，限制了临床可操作性
6. **Workshop 论文**：作为初步成果（preliminary results），实验深度和消融研究有限

## 相关工作

- **EMA 自杀预测**：Kleiman et al. (2017, 2018) 开创了 EMA 自杀意念预测，但自杀未遂预测尚无可靠方法
- **患者异质性建模**：S-GIMME (Gates et al., 2017) 等方法使用离散分组，但本文表明连续隐空间更合适
- **GP 隐变量模型**：GP with Latent Covariate (Wang et al., 2012)、Covariate GPLVM (Martens et al., 2019) 是相关前驱工作，但未应用于此场景
- **Multi-Group GP**：Li et al. (2025) 的多组GP使用离散组标签，而LSGP将"组"连续化和隐变量化
- **元学习GP**：Saemundsson et al. (2018) 的 meta-learning GP 类似但包含控制信号
- **社区检测**：Modularity (Newman, 2004) 原用于网络社区发现，被创新性地应用于患者协方差矩阵分析

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 3.5 | LSGP 本身是已有模型的组合应用，但在自杀预测场景下的适配和实验分析有原创性 |
| 技术深度 | 3.5 | 变分推断框架成熟，核函数设计有亮点，但整体方法基于标准GP工具 |
| 实验充分度 | 3.0 | 基线对比充分，但缺乏消融实验，未系统探索核设计空间 |
| 写作质量 | 4.0 | 动机阐述清晰，从empirical findings自然过渡到方法设计，逻辑严密 |
| 临床意义 | 4.0 | 患者异质性分析和人口统计学"负结论"对临床有重要启示 |
| 综合评分 | 3.5 | 有价值的初步工作，将已有GP方法巧妙适配到关键临床场景，但需更多实验验证 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Little Human Data Goes A Long Way](../../ACL2025/others/a_little_human_data_goes_a_long_way.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[ICCV 2025\] I Am Big, You Are Little; I Am Right, You Are Wrong](../../ICCV2025/others/i_am_big_you_are_little_i_am_right_you_are_wrong.md)

</div>

<!-- RELATED:END -->
