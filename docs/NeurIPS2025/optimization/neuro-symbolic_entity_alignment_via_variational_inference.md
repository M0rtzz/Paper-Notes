---
title: >-
  [论文解读] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference
description: >-
  [NeurIPS 2025][优化][实体对齐] 提出 NeuSymEA，一个基于变分 EM 算法的神经符号推理框架，将符号规则推理与神经网络嵌入统一在马尔可夫随机场中进行实体对齐，在 DBP15K 上实现了显著的性能提升和低资源鲁棒性。
tags:
  - NeurIPS 2025
  - 优化
  - 实体对齐
  - 神经符号推理
  - 变分推断
  - 知识图谱
  - 马尔可夫随机场
---

# NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference

**会议**: NeurIPS 2025  
**arXiv**: [2410.04153](https://arxiv.org/abs/2410.04153)  
**代码**: [GitHub](https://github.com/chensyCN/NeuSymEA-NeurIPS25)  
**领域**: 知识图谱、实体对齐  
**关键词**: 实体对齐, 神经符号推理, 变分推断, 知识图谱, 马尔可夫随机场

## 一句话总结

提出 NeuSymEA，一个基于变分 EM 算法的神经符号推理框架，将符号规则推理与神经网络嵌入统一在马尔可夫随机场中进行实体对齐，在 DBP15K 上实现了显著的性能提升和低资源鲁棒性。

## 研究背景与动机

实体对齐（Entity Alignment, EA）旨在通过识别等价实体对来合并两个知识图谱（KG）。现有方法分为两大类：

- **符号模型**（如 PARIS）：基于规则推理，精确且可解释，但在低度节点和子结构异构问题上表现不佳，导致召回率低
- **神经模型**（如 GCN 系列）：通过嵌入空间召回相似实体，但在实体池增大时难以区分相似表示，精度下降，且缺乏可解释性

现有神经符号方法（PRASE、EMEA）将两类模型简单拼接，缺乏统一优化目标。此外，跨 KG 规则的搜索空间随规则长度指数增长，高效推理是一大难题。

## 方法详解

### 整体框架

NeuSymEA 将所有可能实体对的真值分数建模为马尔可夫随机场中的联合概率分布，由一组加权规则约束，并通过变分 EM 算法迭代优化。E 步用神经模型参数化真值分数并推断缺失对齐；M 步根据观测和推断的对齐更新规则权重。

### 关键设计

1. **变分 EM 框架统一符号与神经模型**：将实体对齐形式化为概率推断问题。每个实体对 $(e, e')$ 关联一个二值变量 $v_{(e,e')}$，目标是最大化观测对齐的对数似然 $\log p_w(v_O)$。由于直接优化不可行，优化 ELBO 下界。E 步固定规则权重 $w$，用神经模型 $q_\theta$ 近似后验分布；M 步固定 $q_\theta$，更新规则权重。核心创新在于将原本独立的两类模型纳入同一优化目标。

2. **基于逻辑演绎的高效优化**：长度为 $L$ 的规则搜索空间指数增长，NeuSymEA 利用逻辑演绎将长规则分解为一系列单位长度的子规则。这样每步推理只需聚合邻居的对齐概率，并通过关系模式 $\eta(r)$（衡量关系唯一性）和子关系概率 $p_{sub}(r \subseteq r')$ 进行加权。参数复杂度线性于数据集大小，计算复杂度为二次方。

3. **可解释推理器（Explainer）**：通过反向规则分解过程，为每个对齐预测提取长规则路径作为显式证据，并恢复规则权重作为量化的置信度分数。支持两种模式：硬锚点模式（仅使用预对齐的锚点对）和软锚点模式（包含推断出的锚点对），后者提供更丰富的解释。

### 训练策略

- **E 步中的伪标签筛选**：神经模型计算所有隐藏对的匹配分数后，采用贪心一对一匹配策略，按置信度排序逐个标注正样本，若某实体已出现则跳过，有效减少假阳性
- **阈值 $\delta$ 控制**：符号模型预测概率超过 $\delta$ 时视为正样本，其余作为负采样候选
- **超参搜索**：$\delta \in \{0.6, ..., 0.99\}$，EM 迭代次数 1-9 轮

## 实验关键数据

### 主实验：DBP15K Full 版本

| 类别 | 模型 | JA-EN Hit@1 | FR-EN Hit@1 | ZH-EN Hit@1 | ZH-EN MRR |
|------|------|------------|------------|------------|-----------|
| 神经 | GCNAlign | 0.221 | 0.205 | 0.189 | 0.271 |
| 神经 | BootEA | 0.454 | 0.443 | 0.486 | 0.600 |
| 神经 | Dual-AMN | 0.627 | 0.652 | 0.650 | 0.732 |
| 神经 | LightEA | 0.736 | 0.782 | 0.725 | 0.779 |
| 符号 | PARIS | 0.589 | 0.618 | 0.603 | - |
| 神经符号 | PRASE | 0.611 | 0.647 | 0.652 | - |
| **Ours** | **NeuSymEA-D** | **0.806** | **0.827** | **0.801** | **0.843** |
| **Ours** | **NeuSymEA-L** | 0.781 | **0.834** | 0.785 | 0.825 |

NeuSymEA-D 在 ZH-EN 上 Hit@1 比最强基线 LightEA 提升 **7.6%**。

### 低资源实验（JA-EN Condensed 版本）

| 模型 | 1% Hit@1 | 5% Hit@1 | 10% Hit@1 | 20% Hit@1 |
|------|---------|---------|----------|----------|
| AlignE | 0.007 | 0.080 | 0.244 | 0.433 |
| PARIS | 0.145 | 0.340 | 0.450 | 0.565 |
| Dual-AMN | 0.239 | 0.509 | 0.652 | 0.750 |
| EMEA | 0.411 | 0.630 | 0.688 | 0.736 |
| **NeuSymEA-D** | 0.481 | 0.692 | 0.742 | 0.835 |
| **NeuSymEA-L** | **0.632** | **0.733** | **0.773** | **0.858** |

在仅 1% 种子对齐时，NeuSymEA-L 的 Hit@1 达到 0.632，远超所有基线。在 FR-EN 上 1% 种子对齐可达 73.7% Hit@1。

### 关键发现

- **统一优化目标的优势**：NeuSymEA 将符号和神经模型纳入同一概率框架联合优化，而非简单拼接，性能全面超越 PRASE 和 EMEA
- **Full vs Condensed 的互补特性**：神经模型在 Full 版本上性能大幅下降（低度实体增多），符号模型反而提升（长尾实体增加连接），NeuSymEA 两个版本均鲁棒
- **快速收敛**：EM 迭代中，规则推断对数量随迭代持续增长且精度保持高位，神经模型 MRR 在几轮内收敛
- **可扩展性**：在百万级实体的 DBP1M 上仍可运行，性能优于 LightEA

## 亮点与洞察

- 首次将变分 EM 从知识图谱补全扩展到跨 KG 实体对齐任务，设计了跨 KG 的加权规则和马尔可夫随机场联合概率建模
- 逻辑演绎分解将长规则推理复杂度从指数降至线性，是推理效率的关键创新
- Explainer 的设计使实体对齐不再是黑箱，为每个预测提供可追溯的规则路径和置信度
- 低资源场景下表现极其亮眼，仅 1% 种子对齐便大幅领先，说明符号推理对冷启动问题的缓解作用

## 局限与展望

- 符号推理组件的计算复杂度为二次方，虽然通过并行化和批处理优化，但在超大规模 KG 上仍有瓶颈
- 当前仅使用结构信息，未引入实体名称、属性等侧面信息，与利用多模态信息的最新方法存在差距
- 规则置信度计算采用乘积形式，长规则的置信度天然偏低，可能遗漏有价值的长距离对齐证据

## 相关工作与启发

- 变分 EM 在 KG 补全中的应用（如 pLogicNet）为本工作提供了理论基础，但跨 KG 的规则设计和双图结构推理是非平凡的扩展
- PARIS 等符号方法的规则挖掘思路被整合进统一框架，体现了"取长补短"的神经符号融合路线
- 对比 EMEA 的伪标签迭代策略，统一目标函数的收敛性更有保障

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 变分 EM 统一神经符号推理的思路新颖，逻辑演绎分解也有理论贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 Full/Condensed 数据集、低资源、大规模 KG、可解释性分析、收敛性分析
- **写作质量**: ⭐⭐⭐⭐ — 公式推导清晰，实验设计合理
- **价值**: ⭐⭐⭐⭐ — 对实体对齐领域有实际推动，低资源可用性强

<!-- RELATED:START -->

## 相关论文

- [VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [Least Squares Variational Inference](least_squares_variational_inference.md)
- [Brain-like Variational Inference](brain-like_variational_inference.md)
- [VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

<!-- RELATED:END -->
