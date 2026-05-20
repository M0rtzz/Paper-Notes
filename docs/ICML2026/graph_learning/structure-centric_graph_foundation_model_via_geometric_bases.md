---
title: >-
  [论文解读] Structure-Centric Graph Foundation Model via Geometric Bases
description: >-
  [ICML 2026][图学习][结构中心 GFM] SCGFM 把跨域图基础模型重写为度量测度空间上的"三角测量"问题：学一组 $K$ 个可训练几何基 $\{B_k\}$，每个图用其与各基的 Gromov–Wasserstein 距离 $\delta_k$ softmax 得到一组结构坐标 $\mathbf…
tags:
  - "ICML 2026"
  - "图学习"
  - "结构中心 GFM"
  - "几何基"
  - "Sliced GW"
  - "结构坐标"
  - "特征再编码"
---

# Structure-Centric Graph Foundation Model via Geometric Bases

**会议**: ICML 2026  
**arXiv**: [2605.08689](https://arxiv.org/abs/2605.08689)  
**代码**: https://github.com/Xd-He/SCGFM  
**领域**: 图基础模型 / 跨域迁移 / Gromov–Wasserstein / 度量几何  
**关键词**: 结构中心 GFM、几何基、Sliced GW、结构坐标、特征再编码

## 一句话总结
SCGFM 把跨域图基础模型重写为度量测度空间上的"三角测量"问题：学一组 $K$ 个可训练几何基 $\{B_k\}$，每个图用其与各基的 Gromov–Wasserstein 距离 $\delta_k$ softmax 得到一组结构坐标 $\mathbf{w}$，再用基上的 OT plan 把节点特征汇聚到统一维度，从而摆脱"必须对齐节点特征空间"的传统 GFM 桎梏，在 in-domain 与 OOD 少样本图/节点分类上都打过 baseline。

## 研究背景与动机

**领域现状**：GFMs 的两条主流路线是 (a) 用 LLM/prompt 注入语言先验、(b) 用对比/生成目标在大图语料上预训 GNN。两者都假设节点特征空间在数据集间能对齐——通常靠 padding、降维或 dataset-specific adapter 实现。这种"特征对齐"在源/目标分布相近时还行，跨域时几乎必崩。

**现有痛点**：(i) 现有 GFM 强制固定节点特征维度（OFA、BRIDGE 等），把不同数据集的特征硬投影到同一空间，丢掉了图本质上的结构差异；(ii) Graph tokenization 方案（把图离散化成 token 当词处理）违反图的排列不变性，强加人工节点顺序；(iii) 没有"共享几何参照系"——图是非欧、置换不变的关系对象，无法像图片那样按像素对齐。

**核心矛盾**：图的可迁移知识在**结构（拓扑）**而非**特征**里，但现有 GFM 把对齐重心压在特征上，导致跨域时结构信息被压扁/扭曲；同时显式 GW barycenter 计算需要嵌套 OT 优化，理论上漂亮但实际不可行。

**本文目标**：建立一个**结构中心**的统一表征空间，使得 (1) 不依赖固定特征维度也能编码任意图；(2) 异构拓扑的图能被映射到同一连续坐标系；(3) 在少样本 in-domain 和 OOD 跨域设置都能强迁移。

**切入角度**：用度量测度（mm-）空间视角看图——每个图就是 $(\mathcal{V},d_G,\mu_G)$，与节点身份无关；进一步借 Gromov 紧致性定理，假设真实图位于 mm-空间的有界子集 $\mathcal{K}\subset\mathcal{X}$，则存在有限 $\epsilon$-cover。把这个 cover 学习化，就得到"几何基字典"。

**核心 idea**：把图表征学习重写成"用 $K$ 个可训练原型在 GW 距离下做三角测量"，每个图就是它对所有基的 GW 距离向量做 softmax 得到的结构坐标 $\mathbf{w}$，外加用 OT plan 重投的节点特征，组合成统一嵌入。

## 方法详解

### 整体框架
两阶段。**Pre-training**：在多源域图上联合优化 $K$ 个几何基 $B_k=([M],d_k,\mu_k)$（$M$ 节点的 mm-空间，$\mathbf{B}_k\in[0,1]^{M\times M}$ 对称无对角元、$\mu_k$ 设为 uniform），用 Sliced Gromov–Wasserstein (SGW) 把每个源图映射到结构坐标 $\mathbf{w}$，最小化结构重建 + 统计量重建 + 多样性正则三项联合损失。**Downstream Projection**：冻结预训练基，对目标图计算 $\mathbf{w}$，并用 GW 计算时的 OT plan $\mathbf{T}_{ik}\in\mathbb{R}^{N\times M}$ 把节点特征求和投到 $M$ 个基节点上得到 $\mathbf{H}(G_i)\in\mathbb{R}^{M\times F}$，最终拼接 $\mathbf{z}(G_i)=[\mathbf{w}\|f(\mathbf{w})\|\mathrm{vec}(\mathbf{H}(G_i))]\in\mathbb{R}^{K+r+MF}$ 喂给下游分类器。

### 关键设计

1. **学习化几何基 + 结构坐标 (Geometric Bases & Structural Coordinates)**:

    - 功能：构造一个共享的"图坐标参照系"，让任意图都能被表征成它和这组基的 GW 距离向量。
    - 核心思路：每个基 $B_k$ 用一个对称无对角元的 $[0,1]^{M\times M}$ 矩阵 $\mathbf{B}_k$ 参数化（不要求三角不等式，伪度量足够当 GW 距离核），测度 $\mu_k$ 固定为均匀分布以减少自由度。对输入图 $G_i$ 算 $\delta_k=d_{GW}(\mathbf{A}_i,\mathbf{B}_k)$（用 SGW 把复杂度从 $\mathcal{O}(N^3)$ 降到 $\mathcal{O}(N\log N)$），再 softmax 得结构坐标 $w_k=\exp(-\delta_k/\tau)/\sum_j\exp(-\delta_j/\tau)$。论文证明 Theorem 3.2：$\|\mathbf{w}-\mathbf{w}'\|_2\le L_w\eta$，即坐标对 GW 距离 Lipschitz 连续，保证结构相似的图坐标也接近。
    - 设计动机：直接学 GW barycenter 需要嵌套 OT，不可行；用"对一组基的距离向量"代替显式 barycenter，是 dictionary learning 在度量几何上的优雅实例。SGW 的一维投影把不可行的 $\mathcal{O}(N^3)$ 砍到 quasi-linear，是这套方案能 scale 的工程基础。

2. **线性代理 GW Barycenter + 多目标重建**:

    - 功能：避开真正 GW barycenter 的嵌套优化，同时让 $\mathbf{w}$ 在结构和统计量两个层面上"重建"原图，确保坐标含足够信息。
    - 核心思路：用线性组合 $\widetilde{\mathbf{B}}(G)=\sum_k w_k \mathbf{B}_k$ 当作 barycenter 的有限基展开，得到结构重建损失 $\mathcal{L}_{gw}=\mathbb{E}_G[d_{GW}(\mathbf{A},\widetilde{\mathbf{B}}(G))]$；再用 MLP decoder $f(\mathbf{w})$ 预测度数直方图 + 聚类系数直方图 + log-scaled motif 计数（三角形、短环），用 MSE 监督 $\mathcal{L}_{rec}=\mathrm{MSE}(\mathrm{FE}(G),f(\mathbf{w}))$；Corollary 3.3 保证 $\|f(\mathbf{w})-f(\mathbf{w}')\|_2\le L_f L_w \eta$，统计重建也对 GW 距离 Lipschitz。
    - 设计动机：线性代理虽不是严格的 mm-空间 barycenter，但作为字典扩展能跑通梯度，且和 softmax 坐标天然兼容；多目标重建保证了"坐标既能恢复邻接也能恢复粗粒度统计"，缓解 OT 自带的非唯一性。

3. **多样性正则 + 结构感知特征再编码**:

    - 功能：防止 $K$ 个几何基塌缩到同一原型；并把异构维度的节点特征 $\mathbf{X}_i\in\mathbb{R}^{N\times F}$ 统一投影到 $\mathbb{R}^{M\times F}$ 共享坐标系。
    - 核心思路：(a) 多样性损失 $\mathcal{L}_{div}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)}\max(0,m-\|\mathbf{B}_i-\mathbf{B}_j\|_F)$ 强制 Frobenius 距离至少为 $m$，让基张满结构空间；(b) 特征再编码用 GW 计算的 OT plan $\mathbf{T}_{ik}$ 把节点特征以 sum aggregation 投到基节点 $\mathbf{H}_k=N\cdot\mathbf{T}_{ik}^\top\mathbf{X}_i$（乘 $N$ 抵消 $\mathbf{T}$ 作为归一化测度引入的平均效应，保留 multiset injectivity），再用 $\mathbf{w}$ 加权得 $\mathbf{H}(G_i)=\sum_k w_k \mathbf{H}_k$。总目标 $\mathcal{L}_{total}=\mathcal{L}_{gw}+\alpha\mathcal{L}_{rec}+\beta\mathcal{L}_{div}$。
    - 设计动机：基塌缩是 prototype/dictionary 模型经典失效模式，hinge-style 多样性正则简单有效；用 OT plan 而非简单 padding/MLP 来对齐特征维度，让"结构相似 → 特征也按结构邻域聚合"，保持 feature-flexible 而不失语义对应。

### 损失函数 / 训练策略
Pre-training 仅在源域图上跑 $\mathcal{L}_{total}=\mathcal{L}_{gw}+\alpha\mathcal{L}_{rec}+\beta\mathcal{L}_{div}$，所有 GW 用 SGW 近似；下游冻结基与 $f(\cdot)$，仅训分类头。少样本 (5-shot) 评测重复 50 次取均值 ± 标准差。

## 实验关键数据

### 主实验
5-shot 图分类（节选自论文 Table 1，列出 in-domain 与 OOD 部分典型数值）：

| 训练 | 测试 | GCN | GIN | GraphCL | SCGFM (论文最佳值) |
|---|---|---|---|---|---|
| in-domain | COX2 | 49.84 | 54.31 | 54.68 | 论文 Table 1 加粗，最佳 |
| in-domain | NCI1 | 51.85 | 52.95 | 57.22 | 最佳 |
| in-domain | BZR | 54.41 | 51.29 | 60.28 | 最佳 |
| S1→COL-3 (OOD) | COL-3 | 9.53 | 9.25 | — | 大幅提升 |
| S2→COX2 (OOD) | COX2 | 50.33 | 55.16 | — | 持平或更高 |
| 平均 (Avg.) | — | 43.23 | 44.85 | — | 最高 |

（注：完整原始数值见原文 Table 1，本笔记仅列代表性切片以体现 in-domain / OOD 双重优势。）

### 消融实验

| 配置 | 关键变化 | 结论 |
|---|---|---|
| Full SCGFM | mean 最高 | 完整模型 |
| w/o 几何基（直接用结构特征） | 跨域大幅退化 | 结构坐标是迁移核心 |
| w/o $\mathcal{L}_{rec}$ | in-domain 仍可，OOD 退化 | 统计重建为跨域稳定性提供 inductive bias |
| w/o $\mathcal{L}_{div}$ | 基塌缩、有效维度下降 | 多样性正则不可省 |
| 把 SGW 换回 exact GW | 同精度但显存爆 | 验证 SGW 的可扩展性收益 |
| 改变基数 $K$ | 中等 $K$ 最优 | 太少欠表达，太多冗余 |

### 关键发现
- 学到的几何基有可解释的拓扑模式（链、星、稠密簇等），印证了"几何基近似 $\epsilon$-cover"的理论假设。
- 跨域时特征分布飘移很大，但结构坐标 $\mathbf{w}$ 几乎不变，能直接迁移；OT plan 让特征自然适配新维度。
- SGW 把训练时间和显存压到 quasi-linear，在百万节点级图上仍可扩展。

## 亮点与洞察
- **从 mm-空间视角统一图表征**：把"图基础模型如何对齐"这个长期悬而未决的问题，重新解释为度量几何里的 $\epsilon$-covering，这种重构能立即套到点云、3D 形状等其他非欧对象。
- **结构坐标 + Lipschitz 稳定性定理**：给了表征学习里少见的"几何一致性"保证——结构相似图必得相似坐标，相比无理论的对比学习 GFM 更可信。
- **特征再编码的 OT 投影**：把"特征对齐"从硬性 padding 变成"沿结构邻域的传输"，保留了多重集 injectivity，是 GFM 处理异构特征空间的优雅模板。
- **SGW 让方法可扩展**：把 $\mathcal{O}(N^3)$ GW 砍到 $\mathcal{O}(N\log N)$ 是 scale 的关键，证明 sliced OT 在图领域同样有用，是一个被低估的实用工具。

## 局限与展望
- 线性代理 barycenter 是工程妥协，无法对极端非高斯分布的图族保证最优；未来可探"非线性 GW barycenter 近似"。
- 基数 $K$、节点数 $M$ 是预先固定的超参数，未做自适应；可结合 covering-number 上界做数据驱动选择。
- 仅验证少样本分类任务，对节点级和链路级任务的覆盖较窄。
- 真实图常带边特征/时间戳，本文 mm-空间仅考虑节点测度 + 邻接，扩展到丰富图结构需重新设计基参数化。

## 相关工作与启发
- **vs OFA / BRIDGE / SAMGPT**：他们强制固定特征维度，跨域时受特征对齐限制；本文用结构坐标做对齐，特征维度可变。
- **vs 图 tokenization (GIT 等)**：他们破坏排列不变性，本文借 mm-空间天然置换不变。
- **vs FGW / GW-coarsening (Vayer 2019, Chen 2023)**：他们用 GW 做"成对比较或粗化"，本文用一组学习化基把它变成"统一坐标系"。
- **vs prototype/dictionary learning 方法**：他们在特征空间选原型，本文在 mm-空间选结构原型，更契合图的本质。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 mm-空间 + 学习化几何基重构 GFM，思路独立，有理论支撑
- 实验充分度: ⭐⭐⭐⭐ in-domain + 两组 OOD 设置 + 多项消融，缺更大规模/节点级 transfer 评测
- 写作质量: ⭐⭐⭐⭐ 几何 motivation 讲得清晰，定理与算法步骤可读性高
- 价值: ⭐⭐⭐⭐ 为 GFM 跨域迁移提供一种"结构中心、特征灵活"的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](../../ICLR2026/graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)

</div>

<!-- RELATED:END -->
