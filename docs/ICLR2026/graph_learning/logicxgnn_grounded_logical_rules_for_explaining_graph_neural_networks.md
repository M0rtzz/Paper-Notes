---
title: >-
  [论文解读] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks
description: >-
  [ICLR 2026][图学习][图神经网络可解释性] LogicXGNN 提出了一种从已训练的图神经网络中提取可解释一阶逻辑规则的 post-hoc 框架：通过图结构哈希和隐藏层嵌入模式识别谓词、用决策树确定判别式 DNF 规则结构、并将抽象谓词接地到输入空间，最终生成可替代原始 GNN 的规则化分类器，同时可作为可控的图生成模型。
tags:
  - ICLR 2026
  - 图学习
  - 图神经网络可解释性
  - 逻辑规则提取
  - 决策树
  - 知识发现
  - 图生成
---

# LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks

**会议**: ICLR 2026  
**arXiv**: [2503.19476](https://arxiv.org/abs/2503.19476)  
**代码**: 无  
**领域**: graph_learning  
**关键词**: 图神经网络可解释性, 逻辑规则提取, 决策树, 知识发现, 图生成

## 一句话总结

LogicXGNN 提出了一种从已训练的图神经网络中提取可解释一阶逻辑规则的 post-hoc 框架：通过图结构哈希和隐藏层嵌入模式识别谓词、用决策树确定判别式 DNF 规则结构、并将抽象谓词接地到输入空间，最终生成可替代原始 GNN 的规则化分类器，同时可作为可控的图生成模型。

## 研究背景与动机

图神经网络（GNN）在药物发现、欺诈检测、推荐系统等领域取得了显著成功，但其黑箱特性阻碍了在医疗健康等高可靠性场景中的应用。现有 GNN 可解释性方法存在明显不足：

- **局部归因方法**（如 GNNExplainer、PGExplainer）只能解释单个实例的预测，无法给出模型的全局行为描述
- **基于概念的全局方法**（如 GCNeuron、GLGExplainer）依赖预定义概念，且生成的规则仅描述单个类别的模式而无法有效区分不同类别
- GLGExplainer 的表达能力严重不足——例如在 Mutagenicity 数据集上仅使用 2 个谓词，性能接近随机分类器

本文的核心问题：能否从 GNN 中提取既可解释、又有判别力、同时模型无关的逻辑规则？

## 方法详解

### 整体框架

LogicXGNN 将问题分解为三个子任务，依次解决：(1) 识别隐藏谓词集合 $P$；(2) 确定 DNF 逻辑规则结构 $\phi_M$；(3) 将规则接地到输入特征空间。最终输出包括描述性规则 $\bar{\phi}_M$（用于图生成）和判别性规则 $\hat{\phi}_M$（用于分类）。

### 关键设计

1. **隐藏谓词识别（Section 3.1）**：

    - **结构模式**：对每个节点 $v$，利用其 1 到 $L$ 跳邻域构成的感受野子图，通过图哈希生成结构签名 $\text{Pattern}_{struct}(v) = \text{Hash}(\text{ReceptiveField}(v, A, L))$
    - **嵌入模式**：对最终层图嵌入训练决策树，找到最具判别力的维度集合 $K$ 和阈值 $T$，将阈值广播到节点嵌入层，对每个节点每个维度生成二值激活状态
    - **谓词定义**：谓词 $p_j$ 是结构模式和嵌入模式的组合元组，一个谓词对节点 $v$ 为真当且仅当两者都匹配
    - 设计动机：GNN 的 message passing 会导致 over-smoothing 使结构信息模糊，因此显式提取结构模式作为补充

2. **逻辑规则结构确定（Section 3.2）**：

    - 对每个类别 $c$，收集所有被正确分类实例的谓词激活模式，形成二值矩阵 $\Phi_c$
    - 每行代表一条合取子句（AND 规则），所有行的析取（OR）构成描述性规则 $\bar{\phi}_M^c$
    - 同时学习谓词之间的连通模式 $\psi_M$（邻接矩阵形式），用于图生成和 motif 级规则接地
    - 将 $\Phi$ 和标签 $Y$ 输入第二棵决策树，生成判别性规则 $\hat{\phi}_M$ 用于跨类区分

3. **输入空间接地（Section 3.3）**：

    - 对每个节点 $v$，构造子图输入特征 $Z_{v,L} = \text{CONCAT}(X_v, \text{Encode}(\{f(u) | u \in N^{(1)}(v)\}), ..., \text{Encode}(\{f(u) | u \in N^{(L)}(v)\}))$
    - 对结构同构但嵌入模式不同的谓词对，用决策树学习基于输入特征的区分规则
    - 对无同构对应的谓词，提取方差最小的特征维度作为代表性描述
    - 可结合连通模式 $\psi_M$ 生成 motif 级规则——例如从"两个节点度为2"的规则推导出"是否有环"

4. **LogicXGNN 作为生成模型**：

    - 从描述性规则 $\bar{\phi}_M$ 中采样一条合取子句
    - 结合连通模式 $\psi_M$ 构造谓词级图模板
    - 根据接地规则 $R$ 为每个谓词分配节点特征
    - 可控透明地生成保持类别属性的新图实例

### 损失函数 / 训练策略

LogicXGNN 是 post-hoc 方法，不改变原始 GNN 的训练。计算开销主要来自决策树训练（三处使用），与训练 DT 的复杂度相当。全部实验在 Ubuntu 22.04, 32GB RAM, 2.7GHz 处理器上完成。GNN 使用 2 层、隐藏维度 32。数据集 8:2 划分训练/测试集。使用 CART 算法训练所有决策树。

## 实验关键数据

### 主实验

| 数据集 | |P| | $|\hat{\phi}_M^0|$ | $|\hat{\phi}_M^1|$ | $\phi_M$ 训练准确率 | $\phi_M$ 测试准确率 | GNN 测试准确率 | GLGExplainer 测试准确率 |
|--------|-----|-----|-----|-------|-------|-------|-------|
| HIN | 2293 | 146 | 62 | 89.12 | 85.23 | 86.93 | 49.99 |
| BBBP | 254 | 20 | 22 | 81.18 | **81.37** | 81.13 | 42.90 |
| Mutagenicity | 314 | 198 | 167 | 78.12 | 76.50 | 76.04 | 54.26 |

### 消融实验

| 配置 | 关键特征 | 说明 |
|------|---------|------|
| 描述性规则 $\bar{\phi}_M$ | BBBP: 146+1044 clauses | 完整覆盖类内变异，可用于生成 |
| 判别性规则 $\hat{\phi}_M$ | BBBP: 20+22 clauses | 精简但准确率可比甚至超越 GNN |
| GLGExplainer | 2 个谓词 | 过度简化，效果近似随机 |
| GCNeuron | 依赖输入类别 | 无判别力，不同类别给出完全不同解释 |

### 关键发现

- **规则化模型可超越神经网络**：在 BBBP 上 $\hat{\phi}_M$ 测试准确率 81.37% 超过原始 GNN 的 81.13%，在 Mutagenicity 上也表现相当
- **化学知识发现准确**：
    - BBBP 数据集：发现含氧丰富基团的分子不易穿越血脑屏障（增加亲水性、降低亲脂性）
    - MUTAG 数据集：发现甲基（-CH3）与非致突变相关、芳环+硝基的组合对致突变关键——后者与原始论文 (Debnath et al., 1991) 的发现一致
    - 现有方法如 GCNeuron 和 GLGExplainer 将硝基和碳环视为独立实体，忽略了它们之间的协同效应
- **可控图生成优于 XGNN**：XGNN（基于 RL 的生成方法）产生的图结构不符合真实分子分布（如生成二部图），而 LogicXGNN 遵循学到的规则，可重建原始分子并生成保持关键属性的新实例

## 亮点与洞察

- 首个作为 GNN 功能等价物的可解释规则化模型——不只是解释，还能替代原始模型
- 三阶段设计非常模块化："发现谓词 → 组织规则 → 接地到输入"各自解耦
- 巧妙利用 decision tree 三次：(1) 提取嵌入关键维度，(2) 生成判别规则，(3) 接地到输入空间
- motif 级规则接地是亮点：通过连通模式将节点级规则升级为子图级语义理解
- 作为生成模型的应用很有想象空间，特别是在药物设计中的组合生成

## 局限与展望

- 实验规模较小（三个数据集），缺乏大规模图数据集的验证
- 图哈希在大图上可能产生大量唯一结构模式，谓词集合可能爆炸
- 描述性规则数量可能非常多（如 BBBP 的 class 1 有 1044 条子句），人类可读性受限
- 目前仅支持分类任务，未扩展到回归（如分子属性预测、3D 结构预测）
- 缺乏对方法可扩展性（节点数、类别数、层数增加时的行为）的分析
- 未与 GNNExplainer 等局部方法在解释质量上进行系统对比

## 相关工作与启发

- 灵感来自 NeuroLogic (Geng et al., 2025)，后者在 FC 网络和 CNN 上从隐藏层激活模式提取逻辑规则
- 与 GLGExplainer (Azzolin et al., ICLR 2023) 形成直接对比：后者依赖 PGExplainer 的局部解释，而 LogicXGNN 完全数据驱动
- 启发：将"神经网络→逻辑程序"的翻译与程序合成结合，可能进一步提升规则模型的性能

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [Hyperbolic Busemann Neural Networks](../../CVPR2026/graph_learning/hyperbolic_busemann_neural_networks.md)

<!-- RELATED:END -->
