---
title: >-
  [论文解读] Singular Bayesian Neural Networks
description: >-
  [ICML 2026][低秩分解] 本文把权重矩阵直接参数化为 $W=AB^\top$ 而不是对 $W$ 本身做平均场分布，从而诱导出一个**关于 Lebesgue 测度奇异的低秩后验**，参数量从 $O(mn)$ 降到 $O(r(m+n))$…
tags:
  - "ICML 2026"
  - "低秩分解"
  - "奇异后验"
  - "PAC-Bayes"
  - "OOD 检测"
  - "平均场变分推断"
---

# Singular Bayesian Neural Networks

**会议**: ICML 2026  
**arXiv**: [2602.00387](https://arxiv.org/abs/2602.00387)  
**代码**: 无  
**领域**: 贝叶斯神经网络 / 变分推断 / 模型压缩 / 不确定性量化  
**关键词**: 低秩分解, 奇异后验, PAC-Bayes, OOD 检测, 平均场变分推断

## 一句话总结
本文把权重矩阵直接参数化为 $W=AB^\top$ 而不是对 $W$ 本身做平均场分布，从而诱导出一个**关于 Lebesgue 测度奇异的低秩后验**，参数量从 $O(mn)$ 降到 $O(r(m+n))$，PAC-Bayes 复杂度从 $\sqrt{mn}$ 收到 $\sqrt{r(m+n)}$，并在 MLP/LSTM/Transformer 三类架构上实现 OOD 检测胜过 5-成员 Deep Ensemble 同时参数少 $33\times$。

## 研究背景与动机

**领域现状**：贝叶斯神经网络 (BNN) 通过维持权重分布而非点估计来提供原则性的不确定性量化，对医疗、自动驾驶等高风险场景至关重要。主流近似方法是平均场变分推断 (MFVI)：每个权重 $w_{ij}$ 用一个独立的高斯 $\mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$，需要 **2 倍**于确定性模型的参数（均值 + 方差）。

**现有痛点**：(1) **参数爆炸**——MFVI 要 $O(mn)$ 个变分参数，让 BNN 长期被困在小模型上；(2) **独立性假设过强**——完全因子化后验抹掉了权重间的结构相关性，损害表达力；(3) Cinquin 等 (2021) 还指出 Transformer 上权重空间推断有根本病理（先验设定难、权重空间与函数空间映射困难）；(4) 现有低秩工作分三派但都有缺陷：post-hoc 低秩扰动 (Rank-1 Mult.) 依赖预训练 backbone 失去端到端不确定性、低秩协方差近似仍参数化全秩 $W$ 均值、LoRA 风贝叶斯变体只能 fine-tune 预训练模型。

**核心矛盾**：现代神经网络经验上具有**低内在维度**（Aghajanyan 等 2021；权重矩阵奇异值快速衰减），但 BNN 的全秩 + 独立参数化在结构上完全无视这个事实，既浪费参数又丢失相关性。

**本文目标**：(1) 把权重矩阵**直接**参数化为低秩乘积，让后验天然落在低秩流形上；(2) 建立 PAC-Bayes 紧化的理论保证，把泛化复杂度从 $\sqrt{mn}$ 降到 $\sqrt{r(m+n)}$；(3) 端到端训练，覆盖 MLP / LSTM / Transformer 三种主流架构；(4) 不依赖预训练 backbone，从头学不确定性。

**切入角度**：作者注意到，若对**因子** $A, B$ 而非 $W$ 做平均场，则诱导后验 $q_W$ 自动支撑在秩-$r$ 流形 $\mathcal{R}_r$ 上——而这个流形在 Lebesgue 测度下**零体积**。换句话说，得到的不是"近似低秩"，而是**严格**奇异于 Lebesgue 测度的后验。这一几何性质本身就是强归纳偏置：所有 $W_{ij}$ 通过共享因子 $A_{ik}, B_{jk}$ 耦合，自动产生结构化相关性。

**核心 idea**：把贝叶斯放在低秩因子上而不是权重上，让"奇异性"变成可量化的归纳偏置，并用 Eckart-Young-Mirsky 定理把近似误差用尾部奇异值 $\sum_{i>r} \sigma_i^2$ 严格刻画。

## 方法详解

### 整体框架
每个权重矩阵 $W \in \mathbb{R}^{m \times n}$ 参数化为 $W = AB^\top$，其中 $A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{n \times r}$。在因子上放尺度混合高斯先验 $p_A(A) = \prod_j [\pi \mathcal{N}(0, \sigma_1^2) + (1-\pi)\mathcal{N}(0, \sigma_2^2)]$（重尾促进稀疏）。变分后验 $q_A, q_B$ 都是平均场高斯，用重参数化技巧 $A = \mu_A + \log(1+\exp(\rho_A)) \circ \epsilon_A$ 让采样可微。ELBO 分解成数据拟合项 $\mathbb{E}_{q_A q_B}[\log p(\mathcal{D}|AB^\top)]$ 与正则项 $\beta(\text{KL}(q_A \| p_A) + \text{KL}(q_B \| p_B))$。三种架构的实例化：MLP 直接因子化全连接层；Transformer 因子化 Q/K/V 投影和 FFN，embedding 用 batch 稀疏只采样当前 token 对应的行；LSTM 因子化 $W_{ih}, W_{hh}$，每个 batch 采样一次 $A, B$ 然后缓存 $W$ 跨时间步。

### 关键设计

1. **诱导奇异后验与几何归纳偏置**:

    - 功能：把贝叶斯不确定性直接放在低秩流形上，避免 MFVI 全空间扩散。
    - 核心思路：对因子 $(A, B)$ 做变分推断，权重 $W = AB^\top$ 的分布通过 pushforward 得到。**Lemma 3.2** 证明 $q_W(\mathcal{R}_r) = 1$（支撑在秩-$r$ 矩阵集合上）；**Lemma 3.3** 证明当 $r < \min(m, n)$ 时 $\mathcal{R}_r$ 的 Lebesgue 测度为零；**Theorem 3.4** 直接得出 $q_W$ 奇异于 Lebesgue 测度。这意味着 $q_W$ 没有 Lebesgue 密度——这与 MFVI 的"处处正密度"形成根本性的几何对比。
    - 设计动机：Wilson & Izmailov (2020) 指出贝叶斯泛化取决于后验**支撑**和**归纳偏置**。MFVI 偏向"权重独立可自由调"，本文偏向"权重通过共享因子耦合"，更符合现代深度网络的低秩本质，并提供隐式正则化——更新 $W_{ij} = \sum_k A_{ik} B_{jk}$ 必须修改影响整行整列的共享因子，阻止局部记忆。

2. **结构化权重相关性（Lemma 3.5）**:

    - 功能：在低参数预算下捕获权重间的全局相关，弥补 MFVI 独立性假设的损失。
    - 核心思路：尽管 $A, B$ 自身平均场，但 $W$ 的元素**不独立**——$\text{Cov}(W_{ij}, W_{i'j'}) = \sum_k \text{Cov}(A_{ik}B_{jk}, A_{i'k}B_{j'k})$，只要两个权重共享潜在因子 $k$ 就有相关性。秩 $r$ 控制相关结构的丰富程度：高秩允许更复杂的块相关，参数仍是 $O(r(m+n))$。论文 Figure 1 实验对比显示 full-rank BBB 是对角相关、低秩则呈现块状结构。
    - 设计动机：过滤掉与主导低秩结构不一致的高频噪声，捕获 MFVI 看不到的"共享子空间"不确定性传播。

3. **理论保证：EYM 损失分解 + PAC-Bayes 收紧**:

    - 功能：把"低秩 ≠ 退化"用定理化语言写清楚，并量化复杂度收益。
    - 核心思路：**Theorem 3.6**（EYM 损失界）：在 $L$-Lipschitz 损失下，最优秩-$r$ 截断 SVD 与全秩最优的损失差被尾部奇异值控制 $|\mathbb{E}\ell(W^*x,y) - \mathbb{E}\ell(W^*_r x, y)| \le LR \sqrt{\sum_{i>r} \sigma_i^2(W^*)}$。**Theorem 3.7** 把学到的 $W = AB^\top$ 与全秩最优的误差分解为**学习误差** $\|W - W^*_r\|_F$ + **秩偏差** $\sigma_{>r}$。**Theorem 3.8** 给 PAC-Bayes 复杂度比 $\sqrt{r(m+n)/mn} \ll 1$；当 $r \ll \min(m, n)$ 时显著收紧。**Theorem 3.9** 用 Pinto 等 (2025) 的低秩 Gaussian complexity 给出补充非空泛化界。
    - 设计动机：让模型选择 $r$ 这件事有理论指导——可以用奇异值衰减分析或消融实验定 $r$，并能预测损失上界。

### 损失函数 / 训练策略
ELBO 三项全部 Monte Carlo 估计（尺度混合先验没有闭式 KL）；用 Adam 优化器；$\sigma = \log(1+\exp(\rho))$ 保正性；$\beta$ KL 温度调节。每层秩 $r_\ell$ 独立可调。预测时 Monte Carlo 平均多个权重样本。

## 实验关键数据

### 主实验
作者在 MIMIC-III（ICU 死亡率，MLP）、Beijing Air Quality（PM2.5 预测，LSTM）、SST-2（情感分类，Transformer）三个数据集上对比 Deterministic / Deep Ensemble (5)/ Full-Rank BBB / Low-Rank (本文) / LR-SVD init / Rank-1 Mult.。

| 数据集 (架构) | 指标 | 本文 Low-Rank | Full-Rank BBB | Deep Ens. (5) | 参数 |
|--------------|------|--------------|--------------|---------------|------|
| MIMIC-III (MLP) | AUC-OOD↑ | **0.802** | 0.770 | 0.738 | 13.6k vs 44.8k / 112k |
| MIMIC-III (MLP) | AUPR-In↑ | **0.824** | 0.807 | 0.721 | — |
| Beijing AQ (LSTM) | PICP↑ | **0.790** | 0.788 | 0.310 | 47k vs 132k / 330k |
| Beijing AQ (LSTM) | AUROC-OOD↑ | 0.710 | 0.492 | **0.730** | — |
| SST-2 (Transformer) | Acc↑ | 0.806 | 0.752 | **0.825** | **1.5M** vs 19.8M / 49.6M |
| SST-2 (Transformer) | AUROC-OOD↑ | 0.640 (2nd) | 0.622 | **0.657** | — |
| SST-2 训练耗时 | min | **8.2** | 23.1 | 64.7 | — |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Low-Rank (random init, r=15) | 最佳 OOD AUC=0.802 | 完整模型 |
| LR-SVD init | OOD AUC=0.713 | 用 SVD 初始化反而退化（过早锁死秩） |
| Rank-1 Mult. (post-hoc) | OOD AUC=0.705 | 验证端到端低秩 > 后处理低秩扰动 |
| Full-Rank BBB | OOD AUC=0.770 | 验证奇异后验贡献 |
| 不同 $r$ 扫描（见 PAC-Bayes Fig 3） | $r^* \approx 11$ 临界 | 超过临界值 PAC-Bayes 界变空泛 |

### 关键发现
- **OOD 检测 vs 似然校准 trade-off**：低秩模型在 OOD 检测和不确定性指标 (PICP/AUPR-Err) 上胜过 Deep Ensemble，但 in-distribution NLL/ECE 略逊于 Ensemble——结构化相关性更关心 epistemic 不确定性，集成更关心似然校准。
- 现代架构权重矩阵确实有快速奇异值衰减（embedding 衰减尤快），这为低秩参数化提供了强经验支撑。
- Transformer 上 Full-Rank BBB 性能反而最差（0.752 acc），印证了 Cinquin 等关于 Transformer 权重空间推断病理的结论；低秩约束反而稳定了训练。
- 一个秩-$r$ BNN 单模型即可匹配 5-成员 Deep Ensemble 的预测性能，参数省 $33\times$。

## 亮点与洞察
- **"奇异"是 feature 而不是 bug**：传统贝叶斯方法回避奇异后验，本文反过来主动构造它并量化其归纳偏置——是把"先验信念"几何化的优雅范式。
- **EYM 定理 + Pushforward**：把矩阵分析中最经典的工具引到贝叶斯深度学习的复杂度分析里，让"选择 $r$"这件事有明确的损失上界指导，非常实用。
- **架构无关的 drop-in 替换**：低秩变分层作为 Keras 标准层的直接替换，工程上极易落地——这对推动 BNN 在产业界普及有现实意义。
- 这种"贝叶斯放在低秩因子上"的思路可迁移到 LoRA fine-tuning（已被作者部分提及）、扩散模型权重不确定性、神经场参数等场景。

## 局限与展望
- 秩 $r$ 仍需人工选择或消融搜索；虽然奇异值衰减分析能辅助，但需要预训练 backbone 才能算 SVD，端到端训练时只能靠消融。
- Deep Ensemble 在 in-distribution likelihood 上仍有优势（NLL=0.300 vs 本文 0.433 在 MIMIC-III），说明结构化相关性带来的不是"全能更好"。
- 实验规模仍偏小（最大是 4-layer BERT-mini），未在真正 billion-scale 模型上验证；论文也承认这是"奠基"工作。
- 尺度混合先验 + Monte Carlo KL 引入额外采样成本，且超参 $\pi, \sigma_1, \sigma_2$ 需要调。
- 未来方向：与 SNGP/Laplace 等函数空间方法结合、扩展到 SSM/Mamba 架构、与生成模型权重不确定性结合做"安全生成"。

## 相关工作与启发
- **vs Rank-1 Multiplicative (Dusenberry 2020a)**：他们在确定性 backbone 上加 rank-1 乘性扰动，是 post-hoc 的；本文从初始化就低秩端到端学习，OOD 上明显胜出。
- **vs Low-Rank Covariance (Tomczak 2020)**：他们对协方差做低秩 + 对角，但权重均值仍是全秩；本文直接对 $W$ 本身做低秩。
- **vs LoRA Bayesian (Yang 2024)**：LoRA 需要预训练 backbone fine-tune；本文从头训练。
- **vs Deep Ensemble**：集成是"多个点估计采样"的"穷人贝叶斯"，参数 $5\times$；本文单模型，参数省 $5$–$33\times$，OOD 检测更好但 in-distribution 似然略差。
- **vs SNGP / Linearized Laplace**：他们做函数空间或最后一层不确定性，本文做权重空间端到端，互补关系。
- **vs Watanabe 的奇异学习理论**：本文的"奇异"是指诱导后验对 Lebesgue 测度奇异（几何性），与 Watanabe 的渐近模型奇异性概念不同。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "奇异后验"几何视角和 EYM 损失界框架是真正原创的贝叶斯深度学习范式
- 实验充分度: ⭐⭐⭐⭐ 覆盖 MLP/LSTM/Transformer 三大架构、多个 OOD 评估指标；缺大规模 LLM 验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论部分推导严谨自洽，定义—引理—定理结构清晰
- 价值: ⭐⭐⭐⭐ 让 BNN 真正可扩展到现代架构，工程上 drop-in 易落地；但 in-distribution 校准仍输 Ensemble

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](../../ICLR2026/others/addressing_divergent_representations_causal.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](../../ICLR2026/others/entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ACL 2025\] Meta-Learning Neural Mechanisms rather than Bayesian Priors](../../ACL2025/others/meta-learning_neural_mechanisms_rather_than_bayesian_priors.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](../../ICLR2026/others/on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](../../ICLR2026/others/learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)

</div>

<!-- RELATED:END -->
