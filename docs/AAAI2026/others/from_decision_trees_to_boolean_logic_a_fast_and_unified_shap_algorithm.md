---
title: >-
  [论文解读] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm
description: >-
  [AAAI 2026][SHAP] 本文提出Woodelf算法，通过将决策树集成模型转化为加权析取范式（WDNF）的伪布尔公式，在统一框架下实现了Background SHAP和Path-Dependent SHAP的线性时间计算，在大规模数据集上实现CPU 16-31倍、GPU 24-333倍的加速。
tags:
  - "AAAI 2026"
  - "SHAP"
  - "Shapley值"
  - "决策树集成"
  - "布尔逻辑"
  - "GPU加速"
---

# From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm

**会议**: AAAI 2026  
**arXiv**: [2511.09376](https://arxiv.org/abs/2511.09376)  
**代码**: [GitHub](https://github.com/ron-wettenstein/woodelf)  
**领域**: 可解释AI / 特征归因 / 博弈论  
**关键词**: SHAP, Shapley值, 决策树集成, 布尔逻辑, GPU加速

## 一句话总结

本文提出Woodelf算法，通过将决策树集成模型转化为加权析取范式（WDNF）的伪布尔公式，在统一框架下实现了Background SHAP和Path-Dependent SHAP的线性时间计算，在大规模数据集上实现CPU 16-31倍、GPU 24-333倍的加速。

## 研究背景与动机

SHAP（SHapley Additive exPlanations）是解释决策树集成模型（如XGBoost、Random Forest、CatBoost）最主流的方法之一，通过Shapley值为每个特征分配贡献值，广泛应用于金融、广告、医疗等领域。然而，SHAP计算有两大主流方式，此前被认为需要截然不同的算法来处理：

- **Path-Dependent SHAP**：利用树的结构和训练时的cover属性估计缺失特征的分布，效率较高但准确性欠佳
- **Background SHAP**：使用背景数据集估计特征分布，最准确但计算代价高——在大数据集上可能需要数年

现有最优方法各有局限：FastTreeShap v2加速了PD-SHAP，PLTreeShap将BG-SHAP降到$O(m+n)$线性复杂度，GPUTreeSHAP利用GPU并行。但它们都依赖自定义C++/CUDA代码，难以集成和扩展，且不支持Banzhaf值等其他博弈论指标。

本文的核心洞察是：**决策树的结构可以自然地编码为布尔逻辑中的WDNF公式，而WDNF上的Shapley值可以在线性时间内计算**。基于此，Woodelf构建了一个统一框架，用纯Python（NumPy/SciPy/CuPy）实现，同时支持多种SHAP变体和交互值。

## 方法详解

### 整体框架

Woodelf的pipeline分为四步：(1) 计算频率向量$f$（来自背景数据集或路径cover属性）；(2) 通过决策模式将每棵树的每个叶子转化为WDNF中的加权立方体，构建贡献矩阵$M$；(3) 利用矩阵乘法将$M$和$f$合并为预计算向量$s$；(4) 使用消费者的决策模式索引$s$，得到最终的SHAP值。整个流程将$O(nm)$的复杂度降到$O(n+m)$。

### 关键设计

1. **WDNF上的线性时间Shapley值公式**:

    - 功能：为伪布尔函数的加权析取范式提供封闭形式的Shapley值计算
    - 核心思路：对于WDNF公式$F(x_1,\dots,x_h)=\sum_{k=1}^m w_k \cdot c_k$，变量$i$的Shapley值为 $\phi_i(F) = \sum_{k=1}^m w_k \times \begin{cases} \frac{1}{|S_k^+| \binom{|S_k|}{|S_k^+|}} & i \in S_k^+ \\ \frac{-1}{|S_k^-| \binom{|S_k|}{|S_k^-|}} & i \in S_k^- \end{cases}$
    - 设计动机：利用Shapley值的线性性质和零玩家（NPO）性质，将指数级的枚举化简为对每个立方体的常数时间贡献计算
    - 类似地推导了Banzhaf值公式$\beta_i(F) = \sum_k \frac{w_k}{2^{|S_k|-1}}$，以及Shapley/Banzhaf交互值

2. **决策模式（Decision Pattern）**:

    - 功能：将消费者/基线的特征值在决策树上的行为编码为二进制序列
    - 核心思路：对叶子$l$的根到叶路径$(n_1, \dots, n_D)$，决策模式$p[i]=1$当且仅当消费者在节点$n_i$会沿路径方向走
    - 设计动机：决策模式完全刻画了消费者在树上的行为，且可以用向量化的BFS高效计算（CalcDecisionPatterns算法，复杂度$O(nL)$）

3. **从决策模式到WDNF的转换规则**:

    - 功能：将消费者和基线的决策模式对映射到WDNF中的加权立方体
    - 核心思路：四条规则——消费者走/基线不走→添加正文字；消费者不走/基线走→添加负文字；都走→立方体不变；都不走→立方体不可满足
    - MapPatternsToCube算法预计算所有可能的模式对到立方体的映射，避免重复计算

4. **$O(n+m)$的Background SHAP推导**:

    - 功能：将原始$O(nm)$的Background SHAP计算降为$O(n+m)$
    - 核心思路：关键观察是基线的贡献只取决于其决策模式，可以先对模式做value_counts聚合（$O(mL)$），然后用预计算的Shapley矩阵$M_{l,i}$和频率向量$f_l$做矩阵乘法得到预计算向量$s_{l,i}$，最后只需对每个消费者查表
    - 设计动机：矩阵乘法是天然GPU友好的操作，用NumPy/CuPy无需自定义CUDA代码

5. **GPU友好的纯Python实现**:

    - 功能：所有核心步骤用标准化的向量化操作实现
    - 核心思路：通过稀疏矩阵将$O(4^D)$降到$O(3^D)$、缓存具有相同特征模式的叶子、利用相邻叶子决策模式仅最后一位不同的性质加速、根据树深度选择适当的整数类型（uint8/16/32）
    - 设计动机：避免C++/CUDA的维护成本，利用CuPy实现CPU到GPU的无缝切换

### 损失函数 / 训练策略

本文不涉及模型训练，Woodelf是一种推理时的解释计算工具，直接作用于已训练好的决策树集成模型。

## 实验关键数据

### 主实验

在两个大规模工业数据集上的性能对比（XGBoost 100棵树，深度6）：

**IEEE-CIS 欺诈检测数据集** ($|B|=118K, |C|=472K, F=397$)

| 任务 | shap包 CPU | SOTA GPU | Woodelf CPU | Woodelf GPU | 加速(vs SOTA任意平台) |
|------|-----------|----------|------------|------------|---------------------|
| PD-SHAP | 151s | 16s | 6s | 3.3s | - |
| BG-SHAP | ~10天 | ~14小时 | 12s | 10s | 24×(GPU) |
| PD-SHAP IV | ~33小时 | 105s | 11s | 8s | 13×(GPU) |
| BG-SHAP IV | X | X | 19s | 12s | 首次可计算 |

**KDD Cup 1999 数据集** ($|B|=4.9M, |C|=3.0M, F=127$)

| 任务 | shap包 CPU | SOTA GPU | Woodelf CPU | Woodelf GPU | 加速(vs SOTA任意平台) |
|------|-----------|----------|------------|------------|---------------------|
| PD-SHAP | 51min | 7.9s | 96s | 3.3s | - |
| BG-SHAP | ~8年 | ~3个月 | 162s | 16s | 165×(GPU) |
| PD-SHAP IV | ~8天 | 229s | 193s | 6s | 38×(GPU) |
| BG-SHAP IV | X | X | 262s | 19s | 首次可计算 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 稀疏矩阵优化 | $O(3^D) \to O(4^D)$ | 利用MapPatternsToCube产生的稀疏性降低核心步骤复杂度 |
| 缓存机制 | 大幅减少矩阵计算 | 相同深度和特征模式的叶子复用计算结果 |
| 整数类型选择(uint8/16/32) | 计算速度提升 | 根据树深选择最小整数类型，优化SIMD利用率 |
| NumPy向量化索引(line 29) | 避免Python循环 | 利用向量化操作完成所有消费者的结果查找 |

### 关键发现
- Background SHAP在KDD数据集上的计算时间5年间从"估计8年"降到"16秒"，展示了算法进步的惊人速度
- Woodelf是首个能计算Background SHAP交互值的可用实现（此前无任何实现）
- 纯Python实现（无需C++/CUDA）依然能击败自定义原生代码的竞争方法
- 可证明Woodelf的推导仅依赖Shapley值的线性性质，因此天然适用于任何满足线性性的指标

## 亮点与洞察
- 论文的核心贡献不仅是一个更快的算法，而是揭示了"决策树→布尔逻辑→博弈论"之间的深层联系；提供了一个优美的理论框架
- WDNF上的线性时间Shapley值公式（Formula 2）极其简洁，仅需遍历每个立方体，根据正负文字集的大小直接计算贡献
- 统一框架同时覆盖Background和Path-Dependent两种SHAP变体，以及Shapley/Banzhaf值和交互值，此前认为它们需要不同的算法
- $O(n+m)$推导的关键步骤是将基线聚合移到内部循环之外（value_counts），然后利用矩阵乘法预计算，这是一个适用于更广泛场景的算法设计思路

## 局限与展望
- 当树非常深($D$大)或数据集较小时，$O(3^D)$的预计算步骤可能成为瓶颈，PLTreeShap和shap包可能更优
- 目前要求决策树集成中的特征分裂是单变量（标准XGBoost格式），对于oblique splits或其他变体需要扩展
- 论文仅在回归任务上实验，分类任务的结果未展示
- Background数据集的选择对SHAP值有显著影响，但论文未讨论如何选择最优的背景集
- 交互值的矩阵维度是$O(4^D \times 4^D)$，当树很深时内存可能成为瓶颈

## 相关工作与启发
- Lundberg等（2020）首次提出多项式时间SHAP算法，奠定了基础；Woodelf可视为其后续的极致优化
- PLTreeShap首次实现了$O(n+m)$的Background SHAP，但依赖自定义C++代码；Woodelf用纯Python达到更好性能
- 论文中将PB函数与博弈论的特征函数对应的框架，可能启发其他模型类型（如神经网络子结构）的高效Shapley值计算
- WDNF/WCNF形式的利用提示了可满足性求解和模型可解释性之间可能存在更多联系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[ICML 2026\] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](../../ICML2026/others/correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)

</div>

<!-- RELATED:END -->
