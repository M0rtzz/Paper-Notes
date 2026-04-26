---
title: >-
  [论文解读] STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings
description: >-
  [ICLR 2026][人体理解][Functional Decomposition] STRIDE 将模型解释重新定义为 RKHS 中的正交函数分解问题，通过递归核中心化无需枚举 $2^d$ 个子集即可解析计算正交功能组件 $f_S(x_S)$，不仅能给出标量重要性分数还能揭示特征如何协同或冗余地影响预测，在表格数据上实现了比 TreeSHAP 快 3 倍且 $R^2=0.93$ 的性能。
tags:
  - ICLR 2026
  - 人体理解
  - Functional Decomposition
  - RKHS
  - Centered Kernels
  - Feature Interaction
  - Component Surgery
---

# STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings

**会议**: ICLR 2026  
**arXiv**: [2509.09070](https://arxiv.org/abs/2509.09070)  
**代码**: 无  
**领域**: 可解释 AI / 核方法  
**关键词**: Functional Decomposition, RKHS, Centered Kernels, Feature Interaction, Component Surgery

## 一句话总结

STRIDE 将模型解释重新定义为 RKHS 中的正交函数分解问题，通过递归核中心化无需枚举 $2^d$ 个子集即可解析计算正交功能组件 $f_S(x_S)$，不仅能给出标量重要性分数还能揭示特征如何协同或冗余地影响预测，在表格数据上实现了比 TreeSHAP 快 3 倍且 $R^2=0.93$ 的性能。

## 研究背景与动机

**领域现状**：可解释 AI（XAI）的主流方法（如 SHAP、LIME、IG）将每个特征的影响压缩为单个标量值 $\phi_i$。Shapley 值框架提供了公理化公平的归因分配，TreeSHAP 等优化算法在特定模型上实现了高效计算，成为实践中的标准工具。

**现有痛点**：标量归因有两个根本性缺陷。第一，**表达力不足**：将特征的复杂非线性效应压缩为一个数字，无法回答"特征如何影响预测"——只能回答"哪个特征重要"。特征间的协同（synergy）和冗余（redundancy）关系被完全遮蔽。第二，**计算代价高**：精确计算 Shapley 值需要遍历 $2^d$ 个特征子集，计算复杂度呈指数增长，实践中只能依赖近似方法。

**核心矛盾**：XAI 社区面临一个根本困境——要么用标量归因获得高效但粗糙的解释（告诉你"什么重要"），要么用函数分解获得详细但低效的解释（告诉你"如何影响"）。此前没有一个框架能同时提供两者且保持实用的计算效率。

**本文目标** (1) 如何在不枚举 $2^d$ 子集的情况下高效计算函数分解？(2) 如何在保持模型无关性的前提下恢复正交功能组件？(3) 如何量化证明交互效应的"功能必要性"而非仅仅观察到相关性？

**切入角度**：作者观察到 Hoeffding 的功能 ANOVA 分解和再生核希尔伯特空间（RKHS）理论可以结合——通过定义递归中心化的核函数，构造出与每个特征子集对应的正交子空间，然后通过解析投影恢复功能组件。关键洞察是中心化核的 Möbius 反演结构使得分解可以无需子集枚举。

**核心 idea**：用 RKHS 中的递归中心化核构造正交子空间，将模型解释从标量归因升级为无需子集枚举的正交函数分解。

## 方法详解

### 整体框架

STRIDE 的 pipeline：输入为任意黑盒模型 $f: \mathcal{X} \rightarrow \mathbb{R}$ 和数据集。(1) **核构造**：选择基核 $k_i$ 并归一化，构建乘积核 $K_S$；(2) **递归中心化**：对乘积核进行递归中心化得到 $K_S^{(c)}$，构造正交子空间 $\mathcal{H}_S$；(3) **解析投影**：将 $f$ 投影到各子空间得到正交组件 $f_S(x_S)$；(4) **聚合**（可选）：将功能组件聚合为兼容 Shapley 公理的标量归因 $\phi_i$。输出为完整的功能分解 $f = \sum_S f_S$ 和/或标量归因。

### 关键设计

1. **递归中心化核**:

    - 功能：构造与每个特征子集 $S$ 对应的中心化核 $K_S^{(c)}$，使不同子集的核函数在 $L^2(\mu)$ 中正交
    - 核心思路：定义递归关系 $K_S^{(c)} := K_S - \sum_{R \subsetneq S} K_R^{(c)}$，基例 $K_\emptyset^{(c)} := 1$。通过 Möbius 反演可得显式公式 $K_S^{(c)} = \sum_{R \subseteq S}(-1)^{|S|-|R|}K_R$。中心化核的核心性质是"部分零均值"——对 $S$ 中任意维度 $i$ 积分后为零：$\int_{T_i} K_S^{(c)}(x_S, t_S) d\mu_i(t_i) = 0$。这直接导致不同子集的中心化核正交（Theorem 1）
    - 设计动机：正交性保证了分解的唯一性和可解释性——每个组件 $f_S$ 精确捕获子集 $S$ 的"纯交互效应"，不与低阶或高阶效应混淆

2. **解析投影与 Shapley 兼容聚合**:

    - 功能：通过正交投影恢复功能组件，并可选地聚合为满足 Shapley 公理的标量归因
    - 核心思路：由正交投影定理，$f_S(x_S) = \langle f, K_S^{(c)}(\cdot_S, x_S) \rangle$，即 $f$ 在子空间 $\mathcal{H}_S$ 上的投影。实际计算中用经验样本平均近似积分，用低秩近似和正则化保证数值稳定性。标量归因定义为 $\phi_i(x) := \sum_{S \ni i} \frac{1}{|S|} f_S(x_S)$，满足效率性、对称性、哑变量和线性公理
    - 设计动机：既保留了 Shapley 值的公理化优势（用户熟悉），又提供了更丰富的功能组件信息——后者是前者的严格超集

3. **组件手术（Component Surgery）**:

    - 功能：量化单个交互组件对模型性能的直接影响，验证其"功能必要性"
    - 核心思路：先用 STRIDE 分解找出最关键的交互组件（如最高影响的 $f_{ij}$），然后从模型预测中"手术切除"该组件——即用 $f(x) - f_{ij}(x_{ij})$ 替代 $f(x)$ 来预测，测量性能下降。在 California Housing 数据集上，移除单个最重要的交互组件导致测试 $R^2$ 下降 $0.023 \pm 0.004$
    - 设计动机：从定性的"这个交互看起来重要"跨越到定量的"移除它会损失多少性能"，提供了因果级别的证据而非仅仅是统计观察

### 损失函数 / 训练策略

STRIDE 是 post-hoc 方法，不涉及模型训练。核超参数（如 RBF 带宽）使用中位数启发式设定。低秩近似的秩和正则化强度作为超参数。交互阶数通过特征依赖性评分筛选，仅建模最相关的特征对。

## 实验关键数据

### 主实验：与 TreeSHAP 基准对比

| 数据集 | 维度 $d$ | STRIDE 时间 (s) | TreeSHAP 时间 (s) | 加速比 | $R^2$ | Spearman $\rho$ |
|--------|---------|----------------|-------------------|--------|-------|-----------------|
| California Housing | 8 | 0.550 | 5.331 | **9.7×** | 0.932 | 0.955 |
| Credit Default | 23 | 1.609 | 11.679 | **7.3×** | 0.988 | 0.945 |
| Online Shoppers | 25 | 1.039 | 3.914 | **3.8×** | 0.965 | 0.898 |
| YearPredictionMSD | 90 | 72.613 | 168.976 | **2.3×** | 0.808 | 0.553 |
| Breast Cancer | 30 | 0.069 | 0.038 | 0.6× | 0.999 | 0.736 |

中位加速比约 3.0×，平均 $R^2 = 0.93$，大部分数据集上 Spearman 相关性超过 0.8。

### 组件手术与 XAI 框架对比

| 方法 | 目标模型 | 解释单元 | 高阶交互 | 计算效率 |
|------|---------|---------|---------|---------|
| LIME | 任意 | $\phi_i$ | ✗ | 高 |
| KernelSHAP | 任意 | $\phi_i$ | ✗ | 很高 |
| TreeSHAP | 树模型 | $\phi_i$ | ✓(部分) | 低 |
| IG / DeepLIFT | 神经网络 | $\phi_i$ | ✗ | 很低 |
| KAN | 神经网络 | 函数 | (隐式) | 中 |
| **STRIDE** | **任意*** | **$\phi_i$ + $f_S$** | **✓** | **低-中** |

组件手术结果：California Housing 上移除最重要的单个交互 → 测试 $R^2$ 下降 $0.0232 \pm 0.0035$，证明交互的功能必要性。

### 关键发现

- **功能组件揭示领域知识**：在 California Housing 上，STRIDE 自动发现 Latitude 和 Longitude 之间的冗余关系以及 Longitude 与 Population 之间的正协同（对应加州沿海高密度高房价区域），这些与领域知识完美一致
- **What-if 分析能力**：模拟增加 MedInc 时，模型对代理特征（Latitude, Longitude）的依赖大幅减少，说明模型正确学到了"位置是收入的代理变量"
- **高维场景可扩展**：在 $d=90$ 的 YearPredictionMSD 上仍保持 2.3× 加速，尽管 $R^2$ 下降到 0.808
- **小任务上 TreeSHAP 更快**：Breast Cancer ($d=30$) 上 STRIDE 慢于 TreeSHAP，说明对小规模简单任务优化空间有限

## 亮点与洞察

- **从"什么"到"如何"的范式提升**：STRIDE 不仅告诉你哪个特征重要，还能展示特征之间如何协同或冗余地工作。这种功能视角严格比标量归因更有表达力——标量归因可以从功能组件聚合得到，反之则不行
- **组件手术的创新性**：首次实现了对黑盒模型中单个高阶交互的定量因果验证。这个方法可以直接迁移到任何提供功能分解的 XAI 框架中，作为验证解释可靠性的标准工具
- **理论优雅性**：递归中心化核 → Möbius 反演 → 正交子空间 → 解析投影的数学链条非常干净，将复杂的组合问题优雅地转化为线性代数操作

## 局限与展望

- **仅验证表格数据**：当前实验局限于表格数据的随机森林模型，对深度学习模型（如 Transformer）和非表格数据（图像、文本）的适用性未验证
- **高阶交互的可扩展性**：虽然避免了 $2^d$ 枚举，但建模高阶交互的计算成本仍可能较高，当前仅建模部分特征对
- **核选择的影响**：分解结果依赖于核函数和带宽的选择，不同核可能导致不同的分解结果
- **弱信号场景退化**：在 YearPredictionMSD ($\rho \approx 0.55$) 上与 TreeSHAP 的排名一致性较低，说明在特征效应微弱时两种方法存在分歧

## 相关工作与启发

- **vs TreeSHAP**: TreeSHAP 是模型特定的（仅限树模型）且只给标量归因。STRIDE 模型无关且提供完整的功能组件，但目前仅在表格场景验证
- **vs RKHS-SHAP (Chau et al., 2022)**: 同样基于 RKHS 但仍专注于标量归因聚合，STRIDE 保留了功能组件这一更丰富的中间结果
- **vs KAN**: KAN 通过架构本身提供可解释性（样条激活），是模型内置的而非 post-hoc 的。STRIDE 作为 post-hoc 方法可以应用于任何已训练模型，灵活性更高
- **vs Functional ANOVA**: 经典 ANOVA 需要边际积分且通常是全局分析，STRIDE 提供了实例级的局部分解

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 RKHS 正交分解引入 XAI 的思路新颖，组件手术是有意义的创新
- 实验充分度: ⭐⭐⭐⭐ 10 个数据集、多种基准对比、消融分析完整
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，实验叙述流畅，动机-理论-实验的逻辑链完整
- 价值: ⭐⭐⭐⭐ 在表格 XAI 领域提供了有意义的工具升级，但应用范围待扩展

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICLR 2026\] DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)
- [\[NeurIPS 2025\] Decomposition of Small Transformer Models](../../NeurIPS2025/human_understanding/decomposition_of_small_transformer_models.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](../../ACL2026/human_understanding/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ICML 2025\] TabFlex: Scaling Tabular Learning to Millions with Linear Attention](../../ICML2025/human_understanding/tabflex_scaling_tabular_learning_to_millions_with_linear_attention.md)

<!-- RELATED:END -->
