---
title: >-
  [论文解读] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles
description: >-
  [ICML 2026][图学习][图基础模型] 这篇论文把图预训练中的“可迁移公共子结构”重新定义为表示空间中的行为不变性，并用神经向量丛、门控几何展平和 Dirichlet 损失构建 Gauge，使图模型在跨域少样本迁移、零样本链路预测和图同构任务上获得更强的结构泛化能力。
tags:
  - "ICML 2026"
  - "图学习"
  - "图基础模型"
  - "黎曼几何"
  - "神经向量丛"
  - "结构迁移"
  - "Dirichlet 能量"
---

# Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles

**会议**: ICML 2026  
**arXiv**: [2606.03270](https://arxiv.org/abs/2606.03270)  
**代码**: https://github.com/RiemannGraph/GAUGE  
**领域**: 图学习 / 图基础模型  
**关键词**: 图基础模型, 黎曼几何, 神经向量丛, 结构迁移, Dirichlet 能量  

## 一句话总结
这篇论文把图预训练中的“可迁移公共子结构”重新定义为表示空间中的行为不变性，并用神经向量丛、门控几何展平和 Dirichlet 损失构建 Gauge，使图模型在跨域少样本迁移、零样本链路预测和图同构任务上获得更强的结构泛化能力。

## 研究背景与动机
**领域现状**：图基础模型希望像语言和视觉基础模型一样，通过预训练学到可复用的结构规律，再迁移到新图和新任务。现有路线大致分为两类：一类借助 LLM 处理带文本属性的图，另一类在纯结构图中寻找 motif、tree、graphon 或结构词表等离散公共子结构。

**现有痛点**：离散子结构“出现得多”并不等于“功能可迁移”。同一个局部 motif 放在不同邻域中可能承担完全不同的结构角色，单纯匹配形状容易把上下文相关的模式误认为通用知识。另一方面，已有黎曼图学习通常预设双曲、球面或乘积空间等外在几何先验，难以刻画普通图模型表示本身诱导出的内在几何。

**核心矛盾**：图迁移真正需要复用的是结构行为，而不是孤立形状。若某个子结构的表示机制在新图中几乎无需调整，它应该表现为一种行为不变性；但这种不变性隐藏在模型的表示空间和局部邻域交互中，无法直接由离散频次判断。

**本文目标**：作者想回答“公共子结构是否真的可迁移”这个问题，并给出可训练模型：一方面建立行为不变性与内在几何平坦性的理论联系，另一方面设计能在预训练中学习这种几何的图神经架构，最后用它提升跨图迁移和结构辨识能力。

**切入角度**：论文选择黎曼几何中的向量丛作为表征语言。直观地说，图结构被放在一个抽象基流形上，每个节点附着一个局部纤维空间；如果相邻纤维之间的平行移动接近恒等变换，那么这一片局部结构的行为更稳定，也更可能迁移。

**核心 idea**：用神经向量丛显式学习图表示的局部坐标和纤维间传输，并通过降低 Dirichlet 能量找到几何平坦、行为不变的子结构。

## 方法详解
论文的方法可以分成两层：第一层是理论与表示框架，把图模型的局部结构行为写成向量丛上的几何对象；第二层是 Gauge 架构，把这个几何对象变成可预训练的网络和损失。

### 整体框架
给定图 $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$，Gauge 先用图编码器生成节点全局表示 $\mathbf{z}_i$，再为每个节点学习一组局部坐标 $\mathbf{Q}^{(i)}$。这些局部坐标可以理解为节点纤维空间的基，描述该节点附近结构如何在表示空间中展开。

在每一层中，模型先从节点与邻域的注意力残差构造局部纤维基，再计算相邻节点纤维之间的伪平行传输 $\mathbf{P}^{(i,j)}=(\mathbf{Q}^{(i)})^\top\mathbf{Q}^{(j)}$。如果 $\mathbf{P}^{(i,j)}$ 接近单位矩阵，说明两个邻域的局部坐标几何兼容；Gauge 会更强地聚合这些邻居的坐标，从而逐层展平同质的局部几何。

预训练时，Gauge 不依赖外部任务标签，而是用 Dirichlet 损失要求一个节点的局部表示能由邻居局部表示预测。损失低的连通区域就是近似行为不变区域，也就是作者定义的可迁移结构。下游适配时，预训练主体冻结，只微调输入编码和任务适配层，以减少对已学结构机制的破坏。

### 关键设计
1. **神经向量丛表示**:

    - 功能：把图中每个节点从单一嵌入点扩展为“全局表示 + 局部纤维坐标”，用于刻画模型表示诱导出的内在几何。
    - 核心思路：对每个节点，模型学习正交基 $\mathbf{Q}^{(i)}$，其转置相当于局部平凡化映射，可把全局表示投影到节点纤维的局部坐标中。相邻节点之间通过 $\mathbf{P}^{(i,j)}=(\mathbf{Q}^{(i)})^\top\mathbf{Q}^{(j)}$ 建立伪平行传输。
    - 设计动机：传统图表示只给每个节点一个向量，难以区分“表示值相似”和“局部结构行为一致”。向量丛让模型可以比较不同节点邻域的坐标系统，从而把迁移性问题变成几何兼容性问题。

2. **Gauge 的门控展平机制**:

    - 功能：在网络层内选择几何兼容的邻居进行局部坐标聚合，让可迁移区域逐层变得更平坦。
    - 核心思路：模型根据 $\mathrm{Tr}(\mathbf{I}_r-\mathbf{P}^{(i,j)})$ 计算边门控权重，平行传输越接近恒等变换，邻居 $j$ 的局部坐标越会被节点 $i$ 吸收。更新后再用 QR 分解保持局部基正交。
    - 设计动机：普通消息传递不区分邻居是否共享同一套局部几何，容易把不可迁移的上下文噪声一起聚合。门控展平把聚合对象限制在几何一致的邻域，契合“可迁移结构应具有平坦内在几何”的理论判断。

3. **Dirichlet 损失与不变子结构恢复**:

    - 功能：在无监督预训练中学习行为不变结构，并给出迁移开销的度量。
    - 核心思路：Dirichlet 能量衡量 $\|\mathbf{x}_i-\mathbf{P}^{(i,j)}\mathbf{x}_j\|^2$ 这类纤维间局部差异。论文将其改写为预测式损失：节点初始局部表示应能由邻居最终局部表示的均值预测。误差低的连通分量被解码为 $\tau$-invariant 子结构。
    - 设计动机：如果某个结构行为在邻域之间保持一致，它的局部表示应该能被邻域稳定预测；反过来，预测误差高说明该结构更依赖上下文，迁移时需要更多调整。

### 损失函数 / 训练策略
Gauge 的预训练目标是 Dirichlet loss。对每个图，损失近似为 $\mathcal{L}(\mathcal{G})=\sum_i\|(\mathbf{Q}^{(i)})^\top\hat{\mathbf{z}}^{(0)}_i-\frac{1}{|\mathcal{N}_i|}\sum_{j\in\mathcal{N}_i}(\mathbf{Q}^{(j)})^\top\mathbf{z}^L_j\|_2^2$，其中初始表示使用 stop-gradient，避免模型通过平凡缩放逃避目标。

下游适配使用 $\mathcal{L}_{ft}=\mathcal{L}_{task}+\lambda\mathcal{L}(\mathcal{G})$。论文给出参数稳定性结果：对预训练中识别出的不变结构，微调梯度会被任务梯度与结构预测误差共同约束，因此低误差结构更不容易被适配过程破坏。

## 实验关键数据

### 主实验
论文在跨域少样本节点分类、零样本链路预测和图同构分类上评估 Gauge。预训练图来自学术、社交和电商等域，目标图包括 PubMed、FacebookPagePage、Roman-empire 和 Photo；多数结果报告 10 次独立运行的均值与标准差。

| 任务 / 数据集 | 指标 | Gauge | 最强对比方法 | 关键结论 |
|--------|------|------|----------|------|
| PubMed 1-shot | Accuracy | 61.26±5.43 | RiemannGFM 56.82±9.00 | Gauge 在生物医学图跨域迁移中领先约 4.44 点 |
| PubMed 5-shot | Accuracy | 71.63±3.89 | GraphAny 70.19±2.20 | 少量标签下仍保持最佳 |
| Facebook 1-shot | Accuracy | 51.61±9.53 | GRACE 49.79±7.82 | 在社交图迁移中领先 |
| Photo 5-shot | Accuracy | 81.33±2.93 | GraphAny 78.45±0.84 | 对电商图的迁移优势较明显 |
| Roman 5-shot | Accuracy | 26.43±1.47 | GraphAny 26.72±1.14 | 异配图上与最优基线接近但不是第一 |

### 消融实验
论文没有只做单一模块剥离，而是用零样本链路预测、图同构和可视化来验证“内在几何 + Dirichlet 不变性”是否真的带来结构能力。

| 分析项 | 关键指标 | 说明 |
|------|---------|------|
| 零样本链路预测 PubMed | AUC 64.03 / AP 61.40 | 无需目标图微调，优于 RAGraph、GFT、RiemannGFM 等 GFM |
| 零样本链路预测 Facebook | AUC 93.88 / AP 90.73 | 在 Facebook 上大幅超过多数基线，说明预训练结构规律可直接转移 |
| 零样本链路预测 Roman | AUC 66.22 / AP 67.21 | 在异配图上仍取得最佳零样本表现 |
| 图同构 CSL | ACC 92.56±4.37 | 相比 SAMGPT 64.17±15.24，结构辨识能力显著增强 |
| 图同构 ZINC12K | MAE 0.1570±0.005 | 低于 GIN 0.1630±0.004，说明几何建模也能辅助分子图回归 |

### 关键发现
- 跨域迁移不是单纯由预训练规模带来。多个 GFM 基线也采用预训练-微调范式，但 Gauge 在大多数目标图和 few-shot 设置下更强，说明内在几何对结构迁移有独立贡献。
- 零样本链路预测是最直接的检验：模型完全不更新参数也能在 PubMed、Facebook、Roman 和 Photo 上取得最佳或近最佳结果，支持“行为不变结构可以复用”的论点。
- 图同构实验把任务从属性预测推向纯结构辨识。Gauge 在 CSL 和 MUTAG 上优于 GCN、GraphSAGE、GIN、GAT 与 SAMGPT，说明向量丛并非只服务于迁移，也增强了对结构等价性的表达。
- 可视化中，Gauge 在二叉树、网格、路径、星形等典型拓扑上恢复出与真实拓扑相符的不变子结构，给抽象的 Dirichlet 能量解释提供了直观证据。

## 亮点与洞察
- 最大亮点是把“子结构可迁移性”从离散匹配问题转成“函数行为是否在局部几何中保持不变”。这个视角避免了 motif 频次等启发式指标的含混性，也解释了为什么同形结构在不同上下文中未必可复用。
- 神经向量丛的建模很巧：它没有要求预设双曲或球面空间，而是从普通图模型的节点表示中学习局部坐标。这样既保留了几何工具的可解释性，又减少了外在几何先验不匹配的问题。
- Dirichlet loss 同时承担预训练目标和迁移度量，设计上比较统一。低损失区域不仅有助于训练表示，还能被解码为具体的不变子结构，用于解释模型学到了哪些结构机制。
- 这套思想可以迁移到其他结构化数据。比如分子、程序图、知识图谱中的“可复用模式”也常依赖上下文功能，而非孤立形状；用局部坐标一致性衡量迁移开销可能比手工 pattern 更稳健。

## 局限与展望
- 论文的理论框架较强，但实际系统依然依赖局部坐标维度、门控温度、Dirichlet 权重等超参数；不同图规模、噪声和异配程度下的敏感性仍需要更系统的报告。
- 实验覆盖多种图域，但目标任务主要集中在节点分类、链路预测和图同构。对于动态图、异构图、知识图谱推理或大规模工业图检索，神经向量丛的计算成本和稳定性还需要验证。
- Gauge 的几何解释很有吸引力，但可视化案例仍偏定性。后续可以把识别出的不变子结构和实际下游错误类型、领域知识或人工结构标签对应起来，增强可解释性的可检验性。
- 论文使用冻结主体加少量适配层的迁移方式，适合验证结构机制稳定性；但真实应用中常需要更大幅度微调，如何在强适配下保持不变结构不被破坏，是下一步值得探索的问题。

## 相关工作与启发
- **vs motif / graphon / structural vocabulary 方法**: 这些方法在离散结构空间中寻找公共模式，Gauge 则把公共性定义为表示空间中的行为不变性。前者更直观、易枚举，后者更能处理上下文相关的结构功能。
- **vs RiemannGFM**: RiemannGFM 也使用黎曼几何，但主要依赖预定义的外在几何空间；Gauge 学习的是模型表示诱导出的内在几何，因此更少受人工几何先验限制。
- **vs GraphMAE / GRACE 等自监督 GNN**: GraphMAE 和 GRACE 通过重构或对比学习获取通用表示，Gauge 可看作把“正样本对齐”推广为几何兼容邻域的 Dirichlet 平滑，目标更直接地面向结构迁移。
- **vs LLM-based graph foundation models**: LLM 图方法在文本属性图上很强，但纯结构图上受限明显。Gauge 的价值在于完全从图结构和节点表示出发，补足 text-free graph foundation model 的理论与架构空白。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从向量丛和内在几何角度定义图结构迁移，问题重述和模型设计都很有辨识度。
- 实验充分度: ⭐⭐⭐⭐ 覆盖跨域 few-shot、零样本链路预测、图同构和可视化，但模块级消融和大规模成本分析还可以更细。
- 写作质量: ⭐⭐⭐⭐ 理论叙事完整，几何动机清楚；部分公式和符号负担较重，读者需要一定黎曼几何背景。
- 价值: ⭐⭐⭐⭐⭐ 对图基础模型的核心问题给出可训练、可解释且效果强的方案，尤其适合启发后续结构迁移研究。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[ICML 2026\] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)

</div>

<!-- RELATED:END -->
