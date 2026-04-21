---
title: >-
  [论文解读] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs
description: >-
  [NeurIPS 2025][图学习][文本属性图] DENSE 提出"文本捆绑"策略，将拓扑/语义相近的节点文本打包后查询 LLM 获取 bundle 级别标签，再用 entropy-based 和 ranking-based 损失监督 GNN 训练，并动态精炼 bundle 排除噪声节点，在 10 个 TAG 数据集上零样本推理全面超越 GPT-4o 和图基础模型。
tags:
  - NeurIPS 2025
  - 图学习
  - 文本属性图
  - 零样本推理
  - LLM
  - 图神经网络
  - Bundle监督
---

# Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs

**会议**: NeurIPS 2025  
**arXiv**: [2505.17599](https://arxiv.org/abs/2505.17599)  
**代码**: 无  
**领域**: graph_learning  
**关键词**: 文本属性图, 零样本推理, LLM, 图神经网络, Bundle监督

## 一句话总结
DENSE 提出"文本捆绑"策略，将拓扑/语义相近的节点文本打包后查询 LLM 获取 bundle 级别标签，再用 entropy-based 和 ranking-based 损失监督 GNN 训练，并动态精炼 bundle 排除噪声节点，在 10 个 TAG 数据集上零样本推理全面超越 GPT-4o 和图基础模型。

## 研究背景与动机

**领域现状**：文本属性图（TAG）中每个节点关联文本描述，零样本推理需要在无标注数据情况下预测节点类别。当前两条路线：(a) 将图拓扑编入语言模型（需大量训练数据，拓扑转序列有信息损失）；(b) 直接用 LLM 对单个节点文本生成伪标签，再监督 GNN。

**现有痛点**：
   - **信息不足**：LLM 只看到单个节点的孤立文本，缺乏图结构上下文，决策依据单薄。
   - **响应不可靠**：LLM 固有幻觉加上信息不足，生成的伪标签噪声大，直接作为监督信号会损害 GNN 训练。

**核心矛盾**：单节点查询让 LLM 既缺信息（图结构丢失）又不可靠（单点幻觉无法纠正），噪声在后续操作中被放大。

**本文目标**：如何让 LLM 获得更丰富的信息以做出更可靠判断？如何设计对噪声鲁棒的监督机制？

**切入角度**：从"单节点查询"转向"文本捆绑查询"——将多个相近节点打包成 bundle，查询 bundle 的众数类别（mode category），利用群体信息抑制单点噪声。

**核心 idea**：通过文本捆绑将 LLM 的零样本能力从节点级提升到 bundle 级，用 bundle 标签监督 GNN 实现鲁棒的零样本图推理。

## 方法详解

### 整体框架
给定文本属性图 $\mathcal{G}=\langle\mathcal{V},\mathcal{E},\mathcal{T},\mathcal{Y}\rangle$，DENSE 包含四个阶段：
1. **Bundle Sampling**：按拓扑或语义相似度采样节点捆绑
2. **Bundle Query**：将 bundle 内节点文本打包为 prompt 查询 LLM，获取 bundle 标签
3. **Bundle Supervision**：用 entropy-based 和 ranking-based 损失训练 GNN
4. **Bundle Refinement**：动态剔除 bundle 中与标签不一致的噪声节点

### 关键设计

1. **Bundle Sampling（捆绑采样）**:

    - 功能：构造包含 $n_B$ 个相近节点的 bundle，使 bundle 内大多数节点属于同一类别。
    - 核心思路：随机选核心节点 $v_c$，按两种策略采样邻居：
        - **拓扑近邻**（同质图）：$\mathcal{N}^k_{\mathcal{G}}(v_c)=\{i \mid 1\leq d^{\mathcal{G}}(v_i,v_c)\leq k\}$，$k$ 自适应选择使邻居数 $\geq n_B-1$。
        - **语义近邻**（异质图）：$\mathcal{B}=\{i \mid \bm{x}_i \in \mathcal{N}^{n_B}_{\mathcal{X}}(\bm{x}_c)\}$，取嵌入空间中 L2 距离最近的 $n_B$ 个节点。
    - 设计动机：保证 bundle 内大多数节点同类，LLM 预测众数类别更容易、更准确。同质图用拓扑、异质图用语义，灵活适配。

2. **Bundle Query（捆绑查询）**:

    - 功能：将 bundle 内节点文本拼接为单个 prompt，询问 LLM 该 bundle 的主要类别。
    - 核心思路：$\mathcal{P}(\mathcal{B})=\langle\text{dataset\_desc}\rangle\text{Concat}(\{t_i|i\in\mathcal{B}\})\langle\text{task\_desc}\rangle$
    - 设计动机：相比逐节点查询，bundle 查询让 LLM 看到多个相关文本，可以发现"持续主题"（persistent theme），做众数判断比单点分类更容易、更鲁棒。实验证明 bundle 查询准确率显著高于单节点查询。

3. **Bundle Supervision（捆绑监督）**:

    - 功能：用 bundle 标签 $\hat{y}^B$ 监督 GNN 训练，设计对离群节点鲁棒的损失函数。
    - 核心思路：GNN $g_\theta$ 输出节点概率 $\bm{p}_i=\text{softmax}(\bm{z}_i)$，计算 bundle 级分布 $\bm{p}(\mathcal{B})=\text{softmax}(\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\bm{z}_i)$。
        - **Entropy-based loss**: $\mathcal{L}_{BE}=\text{CE}(\bm{p}(\mathcal{B}), \hat{y}^B)$
        - **Ranking-based loss**: $\mathcal{L}_R=-\min(\log\bm{p}(\mathcal{B})_{\hat{y}^B}-\log\max_i\{\bm{p}(\mathcal{B})_i\}, 0)$
        - **总损失**: $\mathcal{L}=\mathcal{L}_{BE}+\mathcal{L}_R$
    - 设计动机：Theorem 3.1 证明 bundle 级交叉熵对离群节点的梯度惩罚 $\leq$ 逐节点监督的梯度惩罚，即 bundle 监督天然更容忍异类节点。Ranking loss 则确保只在 bundle 预测不符合标签时施加惩罚。

4. **Bundle Refinement（捆绑精炼）**:

    - 功能：训练过程中动态剔除 bundle 内与标签不一致的节点。
    - 核心思路：$\mathcal{B} \leftarrow \{i \mid i\in\mathcal{B} \wedge \bm{p}_{i,\hat{y}^B} > \min_{j\in\mathcal{B}}\bm{p}_{j,\hat{y}^B}\}$
    - 设计动机：多次精炼逐步移除置信度最低的节点，使 bundle 越来越纯净，监督信号噪声持续降低。

### 训练策略
- 默认 LLM: GPT-4o；bundle 大小 $n_B=5$；bundle 数量 $n_S=100$
- GNN 在 NVIDIA RTX 3090 上训练
- Theorem 3.2 证明 $\mathcal{L}_{BE}$ 的梯度有界、二阶导有界（$\frac{2(M+G^2)}{|\mathcal{B}|}$-smooth），Theorem 3.3 证明梯度下降收敛到稳定点

## 实验关键数据

### 主实验

| 数据集 | DENSE | LLM-BP (之前SOTA) | GPT-4o (直接) | 提升 vs LLM-BP |
|--------|-------|-------------------|---------------|----------------|
| Cora | **75.09** | 72.59 | 70.29 | +2.50 |
| CiteSeer | **72.37** | 69.51 | 64.77 | +2.86 |
| WikiCS | **71.03** | 67.75 | 66.10 | +3.28 |
| History | **67.31** | 59.86 | 53.30 | +7.45 |
| Children | **31.75** | 24.81 | 30.76 | +6.94 |
| Sportsfit | **75.88** | 61.92 | 66.35 | +13.96 |
| Cornell | **84.82** | 83.28 | 45.54 | +1.54 |
| Texas | **92.51** | 81.66 | 63.10 | +10.85 |
| Wisconsin | **87.17** | 77.75 | 56.60 | +9.42 |
| Washington | **81.66** | 73.14 | 48.90 | +8.52 |

### 消融实验

| 配置 | Cora | History | Sportsfit | Texas | 说明 |
|------|------|---------|-----------|-------|------|
| Full DENSE | **75.09** | **67.31** | **75.88** | **92.51** | 完整模型 |
| V1: Random Sampling | 70.48 | 61.80 | 65.60 | 88.24 | 随机采样 bundle 掉 5-14% |
| V2: Individual Query | 71.96 | 63.95 | 72.61 | 84.49 | 逐节点查询 LLM |
| V3: w/o $\mathcal{L}_{BE}$ | 70.11 | 64.49 | 65.29 | 91.44 | 去掉 entropy loss |
| V4: w/o $\mathcal{L}_R$ | 73.99 | 66.73 | 75.48 | 86.10 | 去掉 ranking loss |
| V5: w/ $\mathcal{L}_{IE}$ | 73.43 | 66.29 | 74.05 | 85.03 | 用逐节点监督替代 bundle 监督 |
| V6: w/o Bundle Refinement | 73.89 | 66.55 | 73.00 | 91.98 | 不做 bundle 精炼 |

### 关键发现
- **Bundle 采样**影响最大：在类别多的数据集（History 12类、Sportsfit 13类）上随机采样导致严重掉点（>10%），因为随机 bundle 内节点类别更分散，众数类别不够强。
- **Bundle 查询 vs 单节点查询**：LLM 对 bundle 查询的准确率显著高于单节点分类，特别是 CiteSeer 和 Cornell 数据集提升明显。
- **奇数 bundle 大小优于偶数**：奇数避免了 tie（如4节点 bundle 中2:2平局），$n_B=5$ 是最优选择。
- Bundle 数量 $n_S$ 越多效果越好，但边际递减；$n_S=100$ 是性价比平衡点。

## 亮点与洞察
- **"捆绑"思想很巧妙**：将噪声问题从"如何清洗单个伪标签"转化为"如何获取更可靠的群体标签"，利用统计众数天然抑制个体噪声。这个思路可迁移到任何需要 LLM 生成伪标签的弱监督场景。
- **理论分析完备**：Theorem 3.1 证明 bundle 监督对离群节点更鲁棒（梯度惩罚更小），Theorem 3.2-3.3 证明收敛性。理论和实验一致。
- **LLM backbone 无关性**：GPT-4o、GPT-3.5、DeepSeek-V3、Gemini 等均适用，方法随 LLM 进步自然受益。
- **同时适配同质图和异质图**：通过在采样策略中区分拓扑/语义近邻实现。

## 局限与展望
- 仅适用于节点关联**文本属性**的图，对于纯数值特征或图像属性的图不直接适用。
- Bundle 大小和数量需要调参，虽然 $n_B=5, n_S=100$ 是通用默认值，但在极端场景（如类别数百的大图）可能需要更精细的设置。
- 每个 bundle 查询一次 LLM，$n_S=100$ 意味着 100 次 API 调用，成本和延迟不可忽视。
- Bundle refinement 是单调缩小的（只剔除不加入），如果初始 bundle 质量差，精炼后可能剩余节点太少。

## 相关工作与启发
- **vs LLM-BP**: LLM-BP 也用 LLM 生成监督信号，但逐节点查询+直接伪标签监督。DENSE 通过 bundle 级查询+bundle 监督在所有 10 个数据集上超越，说明 bundle 策略系统性优于单节点策略。
- **vs GOFA/ZeroG**: 图基础模型需要大量预训练数据，且在分布外图上表现不稳定（如大学网页网络）。DENSE 无需图预训练，通过 LLM+GNN 管道更灵活。
- **vs GPT-4o 直接推理**: GPT-4o 在异质图上表现极差（Cornell 45.54%），因为缺乏图结构信息。DENSE 通过 GNN 消息传递弥补了这一缺陷。

## 评分
- 新颖性: ⭐⭐⭐⭐ "文本捆绑"视角新颖，将 bundle 概念引入 TAG 零样本推理，理论和直觉俱佳
- 实验充分度: ⭐⭐⭐⭐⭐ 10个数据集、15个基线、5种LLM、完整消融和超参分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，方法描述流畅，理论分析与实验紧密结合
- 价值: ⭐⭐⭐⭐ 零样本TAG推理是实际痛点，bundle策略简洁有效且可推广

<!-- RELATED:START -->

## 相关论文

- [Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)

<!-- RELATED:END -->
