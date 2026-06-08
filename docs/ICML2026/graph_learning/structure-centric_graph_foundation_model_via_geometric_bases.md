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
SCGFM 要解决的是"跨域图怎么对齐到同一个表征空间"，它的答案是不去对齐特征、而是给所有图一套共享的结构参照系。具体做两阶段：预训练时在多源域图上联合学 $K$ 个可训练的几何基 $B_k=([M],d_k,\mu_k)$，每个图通过它对这组基的 Gromov–Wasserstein 距离得到一组结构坐标 $\mathbf{w}$；下游阶段冻结这些基，对目标图算出 $\mathbf{w}$ 并借 OT plan 把节点特征重投到基节点上，拼成统一嵌入喂给分类器。

### 关键设计

**1. 学习化几何基与结构坐标：用"对原型的距离"代替显式 barycenter**

跨域迁移的真正障碍是没有一个图与图共享的坐标系，而直接去学 mm-空间上的 GW barycenter 又要嵌套 OT 优化、根本跑不动。SCGFM 的做法是把 barycenter 换成一组离散原型：每个几何基 $B_k$ 用一个对称、无对角元、取值在 $[0,1]^{M\times M}$ 的矩阵 $\mathbf{B}_k$ 参数化（它只需当 GW 的距离核，不必满足三角不等式，伪度量就够），测度 $\mu_k$ 固定为均匀分布以压缩自由度。对输入图 $G_i$ 逐基算 GW 距离 $\delta_k=d_{GW}(\mathbf{A}_i,\mathbf{B}_k)$，再 softmax 成结构坐标 $w_k=\exp(-\delta_k/\tau)/\sum_j\exp(-\delta_j/\tau)$，于是任意图都被表达成它在这组原型下的"三角测量"向量。这套设计能成立、且能 scale，靠两个支点：一是论文的 Theorem 3.2 给出 $\|\mathbf{w}-\mathbf{w}'\|_2\le L_w\eta$，即坐标对 GW 距离 Lipschitz 连续，保证结构相近的图必然落到相近坐标，迁移才有几何依据；二是 GW 本身是 $\mathcal{O}(N^3)$ 的，这里全程用 Sliced GW (SGW) 的一维投影近似把它砍到 $\mathcal{O}(N\log N)$，让百万节点级的图也算得动。

**2. 线性代理 barycenter 与多目标重建：让坐标真的装得下原图信息**

光有坐标还不够，得保证 $\mathbf{w}$ 没把图的信息丢掉。SCGFM 不去求严格的 mm-空间 barycenter，而是用基的线性组合 $\widetilde{\mathbf{B}}(G)=\sum_k w_k \mathbf{B}_k$ 当它的有限基展开作代理，再要求这个重建图在结构上贴近原图，得到结构重建损失 $\mathcal{L}_{gw}=\mathbb{E}_G[d_{GW}(\mathbf{A},\widetilde{\mathbf{B}}(G))]$。但邻接重建对置换、对 OT 的非唯一性都不敏感，所以再加一路统计量监督：用 MLP decoder $f(\mathbf{w})$ 去预测原图的度数直方图、聚类系数直方图和 log-scaled motif 计数（三角形、短环），以 MSE 约束 $\mathcal{L}_{rec}=\mathrm{MSE}(\mathrm{FE}(G),f(\mathbf{w}))$。线性代理虽不是真正的 barycenter，却能跑通梯度、又和 softmax 坐标天然兼容；而"邻接 + 粗粒度统计"双目标合起来逼着坐标既能恢复连接关系也能恢复全局结构指纹，缓解了 OT 自带的非唯一性。Corollary 3.3 进一步保证 $\|f(\mathbf{w})-f(\mathbf{w}')\|_2\le L_f L_w \eta$，让统计重建这一路也对 GW 距离 Lipschitz，与坐标的稳定性一脉相承。

**3. 多样性正则与结构感知特征再编码：防基塌缩、并把异构特征投回共享坐标系**

原型/字典模型有个经典失效模式——$K$ 个基会塌缩到同一个原型，有效维度骤降。SCGFM 用一个 hinge 式多样性损失 $\mathcal{L}_{div}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)}\max(0,m-\|\mathbf{B}_i-\mathbf{B}_j\|_F)$ 逼任意两基的 Frobenius 距离至少为 $m$，把基撑开去张满结构空间。另一头要处理的是特征维度异构：不同数据集的节点特征 $\mathbf{X}_i\in\mathbb{R}^{N\times F}$ 维度、语义都不一样，传统做法靠 padding 或 MLP 硬投影。这里改用算 GW 时顺带得到的 OT plan $\mathbf{T}_{ik}\in\mathbb{R}^{N\times M}$ 做传输：把节点特征以求和方式投到 $M$ 个基节点上 $\mathbf{H}_k=N\cdot\mathbf{T}_{ik}^\top\mathbf{X}_i$，其中乘 $N$ 是为了抵消 $\mathbf{T}$ 作为归一化测度带来的平均效应、保住 multiset injectivity，再用结构坐标加权汇成 $\mathbf{H}(G_i)=\sum_k w_k \mathbf{H}_k$。这样"结构相似的节点就会沿同一结构邻域聚合特征"，既保持 feature-flexible（特征维度可变）又不丢语义对应。三项合起来就是总目标 $\mathcal{L}_{total}=\mathcal{L}_{gw}+\alpha\mathcal{L}_{rec}+\beta\mathcal{L}_{div}$；下游则拼接 $\mathbf{z}(G_i)=[\mathbf{w}\,\|\,f(\mathbf{w})\,\|\,\mathrm{vec}(\mathbf{H}(G_i))]\in\mathbb{R}^{K+r+MF}$ 作最终嵌入。

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

- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)

</div>

<!-- RELATED:END -->
