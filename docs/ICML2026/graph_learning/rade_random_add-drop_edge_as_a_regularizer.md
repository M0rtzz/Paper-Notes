---
title: >-
  [论文解读] RADE: Random Add-Drop Edge as a Regularizer
description: >-
  [ICML 2026][图学习][图神经网络] RADE 在 GNN 训练中同时随机删边与加边，并通过"期望保持"的聚合校正使训练-推理对齐，再用 GradNorm 自适应调节删/加比例，让一种增广同时缓解过拟合与过挤压。
tags:
  - "ICML 2026"
  - "图学习"
  - "图神经网络"
  - "过拟合"
  - "过挤压"
  - "随机图增广"
  - "训练-推理对齐"
---

# RADE: Random Add-Drop Edge as a Regularizer

**会议**: ICML 2026  
**arXiv**: [2606.00757](https://arxiv.org/abs/2606.00757)  
**代码**: https://github.com/Danial-sb/RADE  
**领域**: 图学习 / 图神经网络正则化  
**关键词**: GNN 正则化, 过拟合, 过挤压, 随机图增广, 训练-推理对齐

## 一句话总结
RADE 在 GNN 训练中同时随机删边与加边，并通过"期望保持"的聚合校正使训练-推理对齐，再用 GradNorm 自适应调节删/加比例，让一种增广同时缓解过拟合与过挤压。

## 研究背景与动机

**领域现状**：GNN 的消息传递机制（MPNN）天生面临两个痛点：一是与所有深度模型共享的**过拟合**，二是 MPNN 特有的**过挤压 (over-squashing)**——感受野指数膨胀但表示维度有限，长程信息被压缩坍缩。两个问题各自有成熟的对症方案：随机图增广（如 DropEdge）用扰动做隐式正则化对付过拟合；rewiring / virtual node / 图 Transformer 通过加边来缓解过挤压。

**现有痛点**：这两套方案被割裂使用。DropEdge 类删边增广不仅与过挤压无关（删边只会加剧瓶颈），还存在**训练-推理失配**——训练用的是被扰动的稀疏图，推理用的是原图，期望聚合不一致；而 rewiring 方法只构造一张确定性的稠密图，没有随机性可言，也就没有正则化作用。两条路线都还有同一个工程苦水：扰动强度 $p$ 必须按数据集精调。

**核心矛盾**：随机性与连通性增强被视为两件事，但其实"加边"既能注入随机性（→ 正则化），又能创造捷径（→ 缓解过挤压）。问题在于：训练时不停换扰动图，推理时却看到固定图，二者的聚合期望根本不匹配，这种 distribution shift 会反向损害泛化。

**本文目标**：(1) 设计一个统一的图增广，同时把"删边的正则化"和"加边的连通性增强"装进来；(2) 推导显式的聚合校正规则，使训练时的扰动聚合在期望意义下等于推理时的某个目标聚合；(3) 干掉对扰动率 $p, q$ 的手工调参。

**切入角度**：作者把图增广刻画成条件分布 $\mathcal{Q}(\cdot \mid \mathcal{G})$，并提出"期望保持聚合 (Expectation-Preserving Aggregation)"作为对齐准则：要求 $\mathbb{E}_{\mathcal{G}'}[\widetilde{\mathbf{a}}_i^{(\mathcal{G}',\ell)}] = \mathbf{a}_i^{(\mathcal{G},\ell)}$。一旦这个等式成立，随机扰动只贡献均值为零的聚合噪声，进而落入 Fang 等人的"方差型隐式正则化"框架 $\mathbb{E}[\tilde{L}_{\mathrm{BCE}}] \approx L_{\mathrm{BCE}} + \tfrac{1}{2}\sum_i z_i(1-z_i)\mathrm{Var}(\delta_i)$。

**核心 idea**：在删边的同时随机加边，并对每条边/非边的消息做显式校正——校正目标可以是原图聚合（仅治过拟合）或带"加边期望"的修改聚合（同治过挤压），从而把两类增广统一在一个 view-sampling 框架里。

## 方法详解

### 整体框架
RADE 想解决的尴尬是：DropEdge 这类随机增广在训练时删边、推理时却看原图，两边的聚合期望对不上，distribution shift 反过来害泛化；而 rewiring 只造一张确定的稠密图，没随机性也就没正则化。RADE 把这两条路线塞进同一个采样框架——每个 epoch 用 Bernoulli 采样得到扰动邻接矩阵 $\mathbf{A}'$（已有边按 $1-p$ 概率保留、非边按 $q$ 概率加入），在扰动图 $\mathcal{G}'$ 上做带权和聚合 $\widetilde{\mathbf{a}}_i^{(\mathcal{G}',\ell)} = \sum_{j\neq i} \alpha_{ij}^{\mathcal{G}'} \widetilde{\mathbf{m}}_{ij}^{(\ell-1)}$，但消息 $\widetilde{\mathbf{m}}_{ij}$ 先经过一道"期望保持"的校正，使训练聚合在期望意义上等于推理时的目标聚合。删率与加率 $(p,q)$ 不需手调，而是由一个 GradNorm 规则在线更新。

### 关键设计

**1. 随机加-删边扰动：让一种增广同时管正则化和连通性。**

DropEdge 只删边，删边只会加剧过挤压的瓶颈，所以它与长程信息坍缩无关。RADE 的出发点是：加边既能注入随机性（→ 正则化），又能创造捷径（→ 缓解过挤压），于是把删和加在一次扰动里一起做。具体对邻接矩阵每个元素独立采样——已有边 $A_{ij}=1$ 时 $A_{ij}' \sim \mathrm{Bernoulli}(1-p)$，非边 $A_{ij}=0$ 时 $A_{ij}' \sim \mathrm{Bernoulli}(q)$，再对称化 $A_{ji}'=A_{ij}'$ 且禁自环。这个家族把已有方法收成特例：$q=0$ 退化为 DropEdge，$p=0$ 退化为纯随机加边。工程上为避免枚举全部 $|\overline{E}|$ 条非边，按超几何分布无放回采样 $K = q|\overline{E}|$ 条加进来。删和加并非同一回事的两个方向：Proposition 4.4 证明 Drop-only 与 Add-only 一般不可互换，因为它们缩放的节点级统计量不同——这正说明二者是互补原语，必须同时引入，才能既得到不同的方差谱、又真正打开长程通信。

**2. 期望保持的聚合校正：把训练-推理对齐写成可推导的等式。**

随机增广失效的根因是训练聚合的期望和推理聚合对不上。RADE 把这件事形式化为对齐准则——要求 $\mathbb{E}_{\mathcal{G}'}[\widetilde{\mathbf{a}}_i^{(\mathcal{G}',\ell)}] = \mathbf{a}_i^{(\mathcal{G},\ell)}$，等式一旦成立，随机扰动就只贡献均值为零的聚合噪声，落进 Fang 等人的方差型隐式正则化框架。围绕"对齐到哪张图"，论文给出两套规则。**RADE-OF**（只治过拟合）让训练聚合在期望上等于原图聚合：已有边按 $\widetilde{\mathbf{m}}_{ij} = \frac{\alpha_{ij}^{\mathcal{G}}}{\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]} \mathbf{m}_{ij}$ 重缩放，非边则减去非邻居加权均值 $\widetilde{\mathbf{m}}_{ij} = \mathbf{m}_{ij} - \boldsymbol{\mu}_i$（其中 $\boldsymbol{\mu}_i = \tfrac{1}{Z_i}\sum_{j:A_{ij}=0}\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]\mathbf{m}_{ij}$），让加边的期望贡献正好抵消为零。**RADE-OFS**（同治过挤压）只校正删边期望，刻意把加边期望留下来，并在推理时写进修改聚合

$$\widehat{\mathbf{a}}_i^{(\mathcal{G},\ell)} = \mathbf{a}_i^{(\mathcal{G},\ell)} + \sum_{j:A_{ij}=0}\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]\mathbf{m}_{ij}$$

相当于推理时看到一张被"软稠密化"的图，长程捷径被显式保留下来。Proposition 4.1/4.2 证明两套规则都是期望保持。校正里那个期望项 $\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]$ 是否好算取决于聚合器：GIN 的 sum 聚合可解析得 $1-p$ 或 $q$，但 GCN（对称归一化）和 GAT（attention）的扰动度数 $d_i'$ 进了分母，只能用 delta-method 展开近似或经验估计。这也是为什么 DropEdge 仅靠"全局缩放"在 sum 聚合上侥幸成立、却在加权聚合上失效——显式校正补上了 sum 之外的统一规则，而"加边能否被推理利用"就由选 OF 还是 OFS 决定。

**3. GradNorm 自适应率调节：把"无超参"做实。**

传统增广对扰动强度 $p$ 极敏感——太弱没效果、太强毁信号，最优值还因数据集而异，逐集扫参是工程上的老苦水。RADE 不再手调，而是把方差型正则项 $R(B, p, q) = \tfrac{1}{2}\sum_{i\in B} z_i(1-z_i)\mathrm{Var}(\delta_i)$ 当作一个隐式 loss，要求它在共享参数 $\boldsymbol{\theta}$ 上的梯度幅度 $G_{\mathrm{reg}}^B = \|\nabla_{\boldsymbol{\theta}} R\|_2$ 与监督 loss 的梯度幅度 $G_{\mathrm{data}}^B$ 同量级。每个 mini-batch 后对

$$\mathcal{J}(p, q) = \left[\log\frac{G_{\mathrm{reg}}^B + \epsilon}{G_{\mathrm{data}}^B + \epsilon}\right]^2 + \lambda \left(\frac{q}{D(\mathcal{G})}\right)^2$$

走一步 Adam：第一项用对数比例匹配（取 log 是为了不让绝对差受量纲影响），第二项按图密度 $D(\mathcal{G})$ 归一化、惩罚过度加边；实现上对 $\rho = q/D(\mathcal{G})$ 重参数化优化，避免微小 $q$ 把密度算爆。这样"正则化强度"被原理性地绑到与监督 loss 同尺度的梯度上，而不是经验性地猜一个 $p$，且只在 task 级（节点/图分类）固定一个 $\lambda$，不再引入任何数据集级超参。

### 损失函数 / 训练策略
监督 loss 用标准的 BCE / Cross-Entropy；增广不显式写入 loss，而是通过随机采样的扰动图 + 校正消息隐式贡献方差型正则化项（Eq. 5）。优化器为 Adam，学习率 0.001，2 层 backbone（节点分类）或按各任务标准协议（图分类）。RADE 始终初始化 $p=0.5$、$q = D(\mathcal{G})$，之后由 GradNorm 自适应更新，无需扫调。

## 实验关键数据

### 主实验

节点分类（GCN backbone，8 个数据集，5 次平均，Flickr 报 micro-F1，其余报 accuracy %）：

| 方法 | Cora | CiteSeer | Computer | Physics | Flickr | ogbn-arxiv |
|------|------|----------|----------|---------|--------|------------|
| GCN | 80.10 | 69.12 | 89.63 | 96.52 | 51.89 | 70.51 |
| Dropout | 80.76 | 70.04 | 89.71 | 96.54 | 52.02 | 70.77 |
| DropEdge | 80.28 | 69.42 | 89.79 | 96.48 | 52.08 | 70.38 |
| DropMessage | 80.65 | 67.94 | 89.06 | 96.50 | 52.23 | 70.45 |
| **RADE-OF** | **81.08** | **70.20** | **90.22** | **96.58** | **52.25** | **71.09** |
| **RADE-OFS** | 80.82 | 70.12 | 89.94 | 96.53 | 51.92 | 70.95 |

图分类（GCN backbone，TU 报 accuracy %, ogbg-molhiv 报 ROC-AUC %, peptides-func 报 AP %）：

| 方法 | MUTAG | PROTEINS | IMDB-B | ogbg-molhiv | peptides-func |
|------|-------|----------|--------|-------------|---------------|
| GCN | 84.00 | 75.37 | 75.70 | 75.21 | 55.14 |
| DropEdge | 84.50 | 75.29 | 75.70 | 76.01 | 55.29 |
| DropMessage | 86.08 | 75.45 | 75.40 | 74.21 | 54.98 |
| RADE-OF | 86.16 | 75.62 | 76.20 | 76.09 | 56.12 |
| **RADE-OFS** | **86.24** | **75.84** | **76.60** | **76.28** | **56.97** |

### 消融实验（节点分类，RADE-OF / GCN backbone）

| 配置 | Cora | Flickr | ogbn-arxiv | 说明 |
|------|------|--------|------------|------|
| RADE-OF | **81.08** | **52.25** | **71.09** | 完整模型 |
| w/o GN | 80.90 | 51.42 | 70.52 | 去掉 GradNorm 自适应 → Flickr 掉 0.83 |
| w/o GN & EPC | 78.94 | 49.35 | 70.33 | 再去掉期望保持校正 → Cora 掉 2.14, Flickr 掉 2.90 |
| RADE-OF-Lin | 80.42 | 50.65 | 69.92 | 线性化 backbone |
| RADE-OF-Lin w/o GN & EPC | 77.80 | 50.09 | 68.72 | 全摘掉 → 多数据集大幅退化 |
| GCN-Lin (SGC) | 79.60 | 50.12 | 68.67 | 基线参考 |

### 关键发现
- **期望保持校正 (EPC) 贡献最大**：去掉 EPC 后 Cora 掉 ~2 个点、Flickr 掉近 3 个点，说明训练-推理对齐不是装饰品，而是 RADE 实际生效的支点。
- **GradNorm 自适应率确实优于固定调参**：在 Flickr / ogbn-arxiv 等大图上 w/o GN 配置稳定降 0.5–0.8 个点，且省掉了 $p$ 的扫调成本。
- **RADE-OFS 在长程任务上拉开差距**：peptides-func（LRGB 长程基准）上 RADE-OFS 比 GCN 提升 1.83 AP，比 RADE-OF 多 0.85 AP，验证了"保留加边期望进入推理"对缓解过挤压的实效。
- **加 + 删互补，非可互换**：Proposition 4.4 给出理论上互换需满足节点级方差比例严格成立的强约束；消融中 add-only / drop-only 也均弱于联合扰动。

## 亮点与洞察
- **把"期望保持聚合"作为对齐准则**：它既是 alignment criterion，又是从任何带权和聚合规则推导校正规则的构造性原理，比 DropEdge 仅靠"全局缩放"的特例方案更具一般性，能解释为什么 GCN/GAT 上简单 rescale 失效。
- **加边的双重身份**：作者重新发现"随机加边"既是噪声源（正则化）又是捷径（连通性），并通过 OF/OFS 两套校正目标显式控制把哪一面留给推理——这种"在校正规则里塞 inductive bias"的设计模式可迁移到任意带权和图算子（如 PageRank-style propagation, attention-based aggregation）。
- **把 GradNorm 用作"隐式 loss 与显式 loss 的梯度匹配"**：原 GradNorm 是 multi-task balancing 用的，这里改为"监督 loss vs. 方差正则化代理"的匹配，给所有"扰动强度难调"的随机正则化（DropEdge / DropMessage / Mixup）提供了一个干净的自动调参范式。

## 局限与展望
- 期望项 $\mathbb{E}[\alpha_{ij}^{\mathcal{G}'}]$ 对 GCN/GAT 没有闭式解，必须用 delta-method 近似或采样估计，逼近误差对小图与高方差节点可能放大。
- 理论框架的非线性 GNN 推导基于"线性 MPNN + 忽略非线性引入的偏置"的近似（Eq. 5 的脚注），对深度非线性堆叠未给出严格界。
- 互不可换性的刻画局限于 sum 聚合，作者没刻画联合 add-drop 扰动能实现的全部方差谱，对其他常见聚合的设计指导不完整。
- 工程上每个 mini-batch 都要做一步 GradNorm + 重采样扰动图 + 重算 EPC，复杂度与正则化强度的边际收益之间在巨型图（>10M 节点）上仍待验证。
- 没有讨论与图 Transformer / virtual node 类显式长程建模的组合潜力，OFS 在 peptides-func 这种小规模 LRGB 上有优势，能否推广到 PCQM-Contact 等更长程任务尚待回答。

## 相关工作与启发
- **vs DropEdge (Rong et al., 2020)**：DropEdge 只删边、没有显式期望对齐（仅 sum 聚合恰好成立全局缩放），且与过挤压无关。RADE-OF 严格包含 DropEdge 作为 $q=0$ 的特例，并补上一般加权聚合的校正规则。
- **vs DropMessage (Fang et al., 2023)**：DropMessage 把扰动施加在消息上而非边上，与本文同属方差正则化视角；但 DropMessage 不解决过挤压，也不能像 RADE-OFS 那样把加边期望搬到推理。
- **vs rewiring / virtual node (Alon & Yahav, 2021; Topping et al., 2022)**：这些方法在确定性图上做一次性结构修改，没有训练随机性、无正则化；RADE-OFS 把"加边的期望"用一个随机过程间接实现，等价于在期望意义下做了软 rewiring，同时获得了正则化。
- **vs GradNorm (Chen et al., 2018)**：把多任务梯度匹配的思路迁移到"显式 loss vs. 隐式正则项"的匹配上，是 RADE 真正做到"无超参"的关键，对其他随机正则化方法也具有启发性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把删/加边、期望对齐、自适应调参三件事拧成一个干净的统一框架，理论框架清晰且解决了 DropEdge 长期未解决的一般化问题。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 8 个节点分类、6 个图分类数据集，backbone 横跨 GCN/GIN/GAT，消融把 GN/EPC/线性化拆得很干净；不足是缺与图 Transformer / PCQM 长程基准的比较。
- 写作质量: ⭐⭐⭐⭐ 命题陈述与证明组织规整，OF 与 OFS 的对照表（Table 1）清晰，期望项的可解析性与近似策略也都标了出处。
- 价值: ⭐⭐⭐⭐ 给"训练时随机扰动 + 推理时确定图"这一常见尴尬提供了一个原理性的解，且无超参，对工程落地友好；GradNorm 调参范式可迁移到其他随机正则化方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](../../ACL2026/graph_learning/evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[ICML 2025\] Mixed-Curvature Decision Trees and Random Forests](../../ICML2025/graph_learning/mixed-curvature_decision_trees_and_random_forests.md)
- [\[AAAI 2026\] Kernelized Edge Attention: Addressing Semantic Attention Blurring in Temporal Graph Neural Networks](../../AAAI2026/graph_learning/kernelized_edge_attention_addressing_semantic_attention_blurring_in_temporal_gra.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[ICML 2026\] Physics-Informed Coarsening for Multigrid Graph Neural Surrogates](physics-informed_coarsening_for_multigrid_graph_neural_surrogates.md)

</div>

<!-- RELATED:END -->
