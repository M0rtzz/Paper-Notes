---
title: >-
  [论文解读] Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization
description: >-
  [ECCV 2024][马尔可夫毯] 提出 CMBRL 框架，在隐空间中发现马尔可夫毯（Markov Blanket）特征——目标变量的最小充分统计量——代替现有方法中仅选择因果/反因果变量的做法，构建不变预测机制实现跨域泛化。
tags:
  - ECCV 2024
  - 马尔可夫毯
  - 因果表示学习
  - 域泛化
  - 不变预测
  - 隐变量
---

# Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization

**会议**: ECCV 2024  
**arXiv**: N/A  
**代码**: 无  
**领域**: 因果推断  
**关键词**: 马尔可夫毯, 因果表示学习, 域泛化, 不变预测, 隐变量

## 一句话总结

提出 CMBRL 框架，在隐空间中发现马尔可夫毯（Markov Blanket）特征——目标变量的最小充分统计量——代替现有方法中仅选择因果/反因果变量的做法，构建不变预测机制实现跨域泛化。

## 研究背景与动机

**领域现状**：域泛化（Domain Generalization, DG）的目标是在源域上训练的模型能在未见的目标域上保持性能。因果域泛化方法是当前的研究热点，其核心思想是识别生成输入数据的隐因果变量，并构建不变（invariant）的因果机制用于预测，从而在分布偏移下保持鲁棒性。

**现有痛点**：现有因果 DG 方法在选择哪些隐因果变量用于预测时缺乏共识。有的方法选择"因果变量"（causes of the target），有的选择"反因果变量"（effects of the target），但两种策略都有明显局限：只选因果变量可能遗漏了与目标高度相关但不具因果关系的判别性特征；只选反因果变量则可能引入虚假相关。更本质的问题是，"因果"或"反因果"的二分法过于粗糙，忽略了其他同样具有不变性和判别性的特征。

**核心矛盾**：现有方法在"选哪些变量做预测"上存在理论缺口——因果变量和反因果变量的划分不够精细，无法保证选出的变量集合是预测目标的最优特征子集。理想的特征子集应既最小（避免冗余）又充分（包含所有必要信息）。

**本文目标**（1）如何在隐因果空间中定义和发现一个理论上最优的预测特征子集？（2）如何确保这个特征子集在不同域之间保持不变性？（3）如何在端到端框架中同时实现因果表示学习和最优特征选择？

**切入角度**：作者引入了概率图模型中的经典概念——马尔可夫毯（Markov Blanket, MB）。在贝叶斯网络中，一个变量的马尔可夫毯是由其父节点、子节点和共父节点组成的最小集合，具有性质：给定 MB，目标变量与所有其他变量条件独立。这意味着 MB 是"拥有与目标最大互信息的最小变量集合"——精确地符合预测的最优特征需求。

**核心 idea**：在隐因果空间中发现目标变量的马尔可夫毯特征，用这些最小充分特征构建不变预测机制，实现比仅用因果/反因果变量更优的域泛化。

## 方法详解

### 整体框架

CMBRL 包含三个核心模块：（1）因果表示学习模块，将高维输入映射到结构化的隐因果空间；（2）马尔可夫毯发现模块，在隐空间中识别目标变量的 MB 特征；（3）不变预测模块，基于 MB 特征构建跨域不变的预测器。输入图像通过编码器得到隐表示，隐表示经过因果结构学习得到因果图，然后在因果图中识别 MB 特征，最后用 MB 特征做预测。

### 关键设计

1. **因果表示学习模块**:

    - 功能：从高维观测数据中恢复结构化的隐因果变量
    - 核心思路：使用基于 VAE 的架构，编码器将输入 $x$ 映射到隐变量 $z = (z_1, z_2, ..., z_d)$。在隐空间中引入稀疏的因果邻接矩阵 $A$，描述隐变量之间的因果关系。利用多个源域提供的数据分布变化作为"干预"信号来识别因果结构——不同域对隐变量的生成机制产生不同的"软干预"，使得因果关系可被识别。学习过程采用结构因果模型（SCM）框架，通过最大化观测数据的似然同时最小化因果图的稀疏性来联合学习隐变量和因果结构。
    - 设计动机：需要在隐空间中建立因果图，才能后续识别 MB。现有方法通常假设已知因果结构或采用简单的独立假设，CMBRL 则端到端地学习因果结构，更符合真实数据的复杂性。

2. **马尔可夫毯发现模块**:

    - 功能：在学习到的因果图中识别目标变量 $Y$ 的马尔可夫毯
    - 核心思路：给定因果邻接矩阵 $A$，MB($Y$) 由三部分组成——$Y$ 的父节点（直接原因）、$Y$ 的子节点（直接结果）、以及 $Y$ 的子节点的其他父节点（共因）。通过图操作从 $A$ 中提取这些节点：$MB(Y) = Pa(Y) \cup Ch(Y) \cup Pa(Ch(Y))$。为了在训练中使 MB 选择可微分，使用软注意力掩码替代硬选择——基于因果图结构计算每个隐变量的 MB 关系权重，得到连续的 MB 概率分配。
    - 设计动机：MB 的理论保障在于，它是使 $Y \perp\!\!\!\perp Z_{others} | Z_{MB}$ 成立的最小变量集。既不遗漏（充分性），也不冗余（最小性）。相比于仅选因果变量 $Pa(Y)$，MB 还包含了 $Y$ 的结果变量及其共因——这些变量虽非 $Y$ 的直接原因，但包含了关于 $Y$ 的额外判别信息。

3. **不变预测器**:

    - 功能：基于 MB 特征构建跨域不变的分类/回归头
    - 核心思路：不变预测的核心假设是 $P(Y|Z_{MB})$ 在所有域中保持一致（不变性条件）。训练时对每个域分别计算预测损失，并添加不变性正则化——约束不同域上的预测分布 $P_d(Y|Z_{MB})$ 之间的差异最小化。使用 IRM（Invariant Risk Minimization）风格的惩罚项，确保 MB 特征在所有域上对预测器都是最优输入。
    - 设计动机：只有 MB 特征才能同时满足"信息充分"和"跨域不变"两个条件。纯因果变量可能不变但信息不足；包含虚假相关的全部特征可能信息充分但不跨域不变。MB 在两者之间取得理论上最优的平衡。

### 损失函数 / 训练策略

总损失：$L = L_{recon} + \alpha L_{sparsity} + \beta L_{pred} + \gamma L_{inv}$

- $L_{recon}$：VAE 重构损失，确保隐变量保留输入信息
- $L_{sparsity}$：因果图稀疏性惩罚（L1 正则化），鼓励简洁的因果结构
- $L_{pred}$：基于 MB 特征的预测损失
- $L_{inv}$：不变性正则化，约束跨域预测一致性
- 训练采用两阶段策略：先学因果表示+结构，再固定因果图做 MB 发现+不变预测

## 实验关键数据

### 主实验

| 数据集 | 指标 | CMBRL | SOTA (之前最佳) | 提升 |
|--------|------|-------|-----------------|------|
| PACS | Acc (%) | 88.7 | 87.4 (CIRL) | +1.3 |
| VLCS | Acc (%) | 79.3 | 78.1 (CIRL) | +1.2 |
| OfficeHome | Acc (%) | 70.5 | 69.6 (IRM) | +0.9 |
| TerraIncognita | Acc (%) | 49.8 | 48.5 (CIRL) | +1.3 |
| DomainNet | Acc (%) | 43.2 | 42.6 | +0.6 |

### 消融实验

| 配置 | PACS Acc | VLCS Acc | 说明 |
|------|----------|----------|------|
| Full CMBRL (MB特征) | 88.7 | 79.3 | 完整模型 |
| 仅用因果变量 Pa(Y) | 86.9 | 77.8 | 丢失了子节点和共父节点的信息 |
| 仅用反因果变量 Ch(Y) | 87.1 | 78.0 | 丢失了父节点信息 |
| 用全部隐变量（不选择） | 86.2 | 76.5 | 引入了虚假相关特征 |
| w/o 因果结构学习 | 85.8 | 76.9 | 随机MB导致选择不准确 |

### 关键发现

- MB 特征比单独使用因果变量或反因果变量都更优，验证了 MB 作为"最小充分特征集"的理论优势（PACS 上分别高出 1.8 和 1.6 个百分点）。
- 不做特征选择（使用全部隐变量）的效果最差，说明冗余和虚假变量确实会损害泛化性能。
- 因果结构学习模块至关重要——没有准确的因果图就无法准确识别 MB。
- 在 TerraIncognita 这种域差异特别大的数据集上提升最显著，说明 MB 的不变性在大分布偏移下更有优势。

## 亮点与洞察

- **马尔可夫毯的引入解决了变量选择的理论缺口**：之前因果 DG 方法在"选因果还是选反因果"上争论不休，MB 作为统一框架给出了理论最优解。MB 的"最小充分性"保证了它既不像全部变量那样引入噪声，也不像仅用因果变量那样丢失有用信息。
- **将图模型中的经典概念引入深度学习**：MB 在概率图模型中是经典概念但在深度域泛化中首次被系统化应用。这种"经典理论 + 现代深度学习"的结合方式值得借鉴——很多图模型中的优雅概念可能在深度学习场景中焕发新生。
- **端到端因果发现与特征选择**：不需要先验因果知识，直接从多域数据中同时学习因果结构和识别 MB，这种端到端的设计使方法更具实用性。

## 局限与展望

- 隐因果变量的维度 $d$ 需要预设，不同的 $d$ 可能导致不同的因果图和 MB。自动确定隐空间维度是未解决的难题。
- 因果结构学习在有限数据下可能不准确，尤其当域数量较少时。方法依赖多域数据提供的"干预"信号，域数量不足可能导致因果不可识别。
- 两阶段训练可能导致早期因果结构错误传播到后续的 MB 发现。联合优化可能更好但更难训练。
- 实验数据集的域差异相对温和，在更极端的分布偏移（如 sim-to-real）下的表现未知。
- 计算开销较大：因果结构学习本身就是 NP-hard 问题，近似算法的效率和准确性需要权衡。

## 相关工作与启发

- **vs CIRL (Causal Invariant Representation Learning)**: CIRL 仅选择因果变量做预测。CMBRL 通过 MB 选择了更完整的特征集，在多个数据集上超越 CIRL。
- **vs IRM (Invariant Risk Minimization)**: IRM 追求跨域不变的预测器但不显式建模因果结构。CMBRL 将因果建模和不变预测结合，比 IRM 有更强的理论基础。
- **vs CausalVAE**: CausalVAE 在隐空间中做因果发现但不做有针对性的特征选择。CMBRL 在此基础上增加了 MB 发现模块，实现了有选择性的特征利用。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将马尔可夫毯引入域泛化是全新的视角，理论动机充分
- 实验充分度: ⭐⭐⭐⭐ 覆盖了 5 个标准 DG 基准，消融实验设计精当
- 写作质量: ⭐⭐⭐⭐ 理论与实践结合紧密，概念解释清晰
- 价值: ⭐⭐⭐⭐ 为因果域泛化提供了更优的变量选择理论框架

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ECCV 2024\] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](../../NeurIPS2025/causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](../../ACL2025/causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)

<!-- RELATED:END -->
