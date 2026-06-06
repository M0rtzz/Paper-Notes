---
title: >-
  [论文解读] CoSiNE: 条件位点独立的抗体序列神经进化模型
description: >-
  [ICML2026][计算生物][抗体进化] CoSiNE 用神经网络参数化的条件位点独立连续时间马尔可夫链（CTMC）来建模抗体亲和力成熟过程，在保持可处理性的同时捕获位点间上位效应，并通过 Guided Gillespie 采样实现抗原特异性的抗体优化，在零样本变体效应预测上超越了现有语言模型和进化模型。
tags:
  - "ICML2026"
  - "计算生物"
  - "抗体进化"
  - "连续时间马尔可夫链"
  - "亲和力成熟"
  - "变体效应预测"
  - "分类器引导采样"
---

# CoSiNE: 条件位点独立的抗体序列神经进化模型

**会议**: ICML2026  
**arXiv**: [2602.18982](https://arxiv.org/abs/2602.18982)  
**代码**: https://github.com/thematrixmaster/cosine  
**领域**: 科学计算/计算生物学  
**关键词**: 抗体进化, 连续时间马尔可夫链, 亲和力成熟, 变体效应预测, 分类器引导采样  

## 一句话总结

CoSiNE 用神经网络参数化的条件位点独立连续时间马尔可夫链（CTMC）来建模抗体亲和力成熟过程，在保持可处理性的同时捕获位点间上位效应，并通过 Guided Gillespie 采样实现抗原特异性的抗体优化，在零样本变体效应预测上超越了现有语言模型和进化模型。

## 研究背景与动机

**领域现状**：抗体工程中的深度学习方法主要分为两类——蛋白质语言模型（如 ESM-2、AbLang-2）学习序列的边缘分布 $p(x)$，能捕获复杂的位点间上位效应，但将序列视为独立同分布样本，完全忽略了进化时间动力学；经典系统发育模型（如 WAG、LG）显式建模进化过程，但假设位点独立进化，无法捕获上位交互。

**现有痛点**：语言模型的性能部分源于对保守种系残基的记忆，而非真正理解亲和力成熟过程。经典进化模型的独立位点假设导致状态空间 $|\mathcal{A}|^L = 20^L$ 的矩阵指数计算不可行（$O(|\mathcal{S}|^3)$ 复杂度），因此必须假设位点独立来将复杂度降到 $O(L|\mathcal{A}|^3)$，但这又丢失了上位效应信息。

**核心矛盾**：表达能力与计算可行性之间的 trade-off——全序列空间的 CTMC 能捕获所有上位效应但计算不可行，独立位点模型可行但表达力不足，语言模型有表达力但缺乏进化时间建模。

**本文目标**：设计一个既能保持独立位点模型的计算效率，又能通过序列上下文条件化来捕获上位效应，同时显式建模连续时间进化动力学的统一框架。

**切入角度**：作者观察到，如果让每个位点的速率矩阵 $Q_\ell$ 依赖于完整的父序列 $x$（而非仅依赖该位点自身），就能在保持因子化转移概率的同时，通过神经网络隐式编码位点间的依赖关系。数学上，这构成了全序列空间顺序点突变过程的一阶近似，误差随分支长度二次增长——而抗体亲和力成熟恰好以短分支为主。

**核心 idea**：用神经网络输出以完整序列为条件的位点特异速率矩阵，实现"条件位点独立"的 CTMC，融合进化建模的时间动力学与语言模型的上位效应捕获能力。

## 方法详解

### 整体框架

CoSiNE 的输入是一个父抗体序列 $x$，神经网络（基于 ESM-2 150M 初始化）输出 $L$ 个位点特异的速率矩阵 $Q_\theta(x)_\ell \in \mathbb{R}^{|\mathcal{A}| \times |\mathcal{A}|}$。给定进化时间 $t$，通过矩阵指数计算每个位点的转移概率，再取乘积得到全序列转移概率。模型在约 200 万条从约 12 万个克隆树提取的进化转移（parent-child pairs）上训练。推理时通过 Gillespie 算法采样进化轨迹，并可通过分类器引导实现抗原特异性优化。

### 关键设计

1. **条件位点独立转移概率**:

    - 功能：在保持因子化可行性的同时捕获位点间上位效应
    - 核心思路：转移概率定义为 $p_\theta(y|x,t) = \prod_{\ell=1}^{L} \exp(t Q_\theta(x)_\ell)_{x_\ell, y_\ell}$，其中 $Q_\theta$ 是以完整父序列 $x$ 为条件的神经网络。当速率矩阵满足 $(Q_\theta(x)_\ell)_{x_\ell, y_\ell} = \mathbf{Q}_{x,y}$ 时，该模型构成全序列顺序点突变过程的一阶近似，转移概率的 $L_1$ 误差上界为 $(\lambda t)^2$（$\lambda$ 为最大离开速率）。这意味着短分支（抗体进化的典型情况）下近似误差很小
    - 设计动机：传统独立位点模型（WAG/LG）所有位点共享一个 $Q$，丢失所有上位信息；CoSiNE 让每个位点的 $Q_\ell$ 依赖全序列，理论保证一阶捕获上位效应

2. **选择-突变解耦的适应度推断**:

    - 功能：从学到的进化模型中提取纯选择信号，用于零样本变体效应预测
    - 核心思路：基于突变-选择框架，将观测到的转移速率分解为 $Q_{xy} = k \mu_{xy} P_{\text{fix}}(x \to y)$，其中 $\mu_{xy}$ 是中性体细胞超突变（SHM）速率。选择分数定义为 $\text{Score}(x \to y) = \log p_\theta(y|x,t) - \log q(y|x,t) \approx \log P_{\text{fix}}(x \to y) + C$，即 CoSiNE 对数似然与预训练 SHM 模型 Thrifty 对数似然的差值。这消除了 SHM 偏置的干扰，提取纯粹的自然选择信号
    - 设计动机：语言模型直接用困惑度评估适应度，但被种系残基的保守性混淆；DASM 需要手动截断选择分数来保持概率有效性；CoSiNE 通过对数似然比自然导出选择分数，无需启发式约束

3. **Guided Gillespie 抗原特异性采样**:

    - 功能：在推理时引导 CoSiNE 生成对特定抗原具有高亲和力的抗体序列
    - 核心思路：基于离散扩散模型的分类器引导理论，将引导速率矩阵定义为 $(\mathbf{Q}_z^{(\gamma)})_{x,y} = [p(z|y)/p(z|x)]^\gamma \mathbf{Q}_{x,y}$。用结合亲和力预测器的正态假设近似 $p(z|y)$，再通过 Taylor 近似（TAG）将代价从 $L \times (|\mathcal{A}|-1)$ 次预测器调用降至每步仅 1 次梯度计算，实现 500 倍加速。采用自适应阈值 $r_0 = \mu_{\theta_z}(x)$ 避免引导权重消失
    - 设计动机：CoSiNE 的训练数据不含抗原信息，无法直接生成针对特定靶标的抗体；与离散扩散/流匹配不同，CTMC 没有边界时间约束，因此预测器无需在噪声序列上训练，可直接使用朴素训练的序列-性质预测器

### 训练策略

模型基于 ESM-2 150M checkpoint 初始化，替换语言建模头为使用 softplus 激活的速率矩阵输出头。使用 AdamW 优化器（学习率 $2.5 \times 10^{-4}$），多项式衰减调度，BF16 混合精度训练，单 A100 GPU 约 1 天收敛。在重链和轻链之间插入 chain-break token 以同时处理配对抗体。

## 实验关键数据

### 零样本变体效应预测（DMS 评估）

在 FLAb2 benchmark 的 4 个 DMS 数据集上评估，指标为 Spearman 相关系数：

| 数据集 | CoSiNE | DASM | ESM2-150M | ProGen2-S | PRISM |
|--------|--------|------|-----------|-----------|-------|
| Koenig Expr (H) | **0.613** | 0.596 | 0.413 | 0.407 | 0.069 |
| Koenig Expr (L) | 0.508 | 0.474 | 0.485 | **0.513** | 0.129 |
| Adams Binding | **0.464** | 0.270 | -0.112 | -0.024 | 0.297 |
| Koenig Bind (H) | **0.456** | 0.415 | 0.112 | 0.098 | 0.005 |
| Koenig Bind (L) | **0.371** | 0.327 | 0.266 | 0.332 | 0.061 |

CoSiNE 在 7 个数据集中的 6 个上取得最佳，特别在跨物种（Adams 小鼠抗体）场景下大幅领先（0.464 vs 次优 0.297）。

### 消融与分析

| 消融配置 | 效果 | 说明 |
|---------|------|------|
| 不做 SHM 校正（仅用 $\log p_\theta$） | 所有数据集相关性下降 | 解耦突变-选择对 VEP 至关重要 |
| 仅输入单链（去掉配对链） | 部分数据集显著下降 | 链间上位效应对预测有贡献 |
| 从头训练（不用 ESM2 初始化） | 平均 $\Delta\rho = 0.041$ | 进化训练目标本身贡献了大部分预测力 |
| 不同分支长度 $t \in [0.1, 0.4]$ | $\Delta\rho \leq 0.045$ | 选择分数对 $t$ 选择鲁棒 |
| CDR 局部优化（5 突变预算） | $\Delta\text{Bind}_{\text{max}} = 0.395$（预算内最优） | 优于遗传算法和 PoE 方法 |
| Guided Gillespie（$\gamma=5$） | 生成抗体亲和力与真实结合物重叠 | 同时保持结构质量（pLDDT）和人源性（OASis） |
| TAG 近似 vs 精确引导 | 500 倍加速，无显著性能差异 | Taylor 一阶近似有效 |

## 亮点与洞察

1. **理论与实践的优美结合**：Proposition 4.1 给出了条件位点独立模型近似全序列 CTMC 的严格误差上界 $O(t^2)$，而抗体进化短分支的生物学特性恰好使这个近似特别适用
2. **首次连接离散扩散与经典序列进化模型**：将分类器引导从离散扩散模型移植到经典 CTMC 框架，且预测器无需在噪声数据上重训——这是因为 CTMC 没有边界时间约束
3. **Categorical Jacobian 分析**揭示了 CoSiNE 学到的链内和链间 CDR 区域上位耦合，与抗体抗原结合口袋的生物学结构一致

## 局限性 / 可改进方向

1. 一阶近似在长分支上的误差会增大，限制了对慢速进化蛋白的适用性
2. 当前框架忽略插入和缺失（indels），只能处理等长序列——对抗体来说可接受，但推广到一般蛋白质时成为瓶颈
3. Guided Gillespie 依赖亲和力预测器的质量，高引导强度（$\gamma \geq 10$）可能利用预测器不确定性生成过度优化的序列

## 相关工作与启发

- **DASM**（Matsen 2025）：同样解耦 SHM 与选择，但需手动截断选择分数；CoSiNE 通过对数似然比自然保持数学一致性
- **SiteRM**（Prillo 2024）：每位点独立速率矩阵在 ProteinGym 上效果好，但不做上下文条件化
- **PRISM**（Kim 2026）：用辅助头预测种系/突变状态，但将进化简化为二元区分
- 启发：条件位点独立的框架思路可推广到其他蛋白质家族的进化建模与设计

## 评分

- 新颖性: 9/10 — 首个融合神经 CTMC 与分类器引导的抗体进化模型，理论贡献扎实
- 实验充分度: 9/10 — VEP、引导采样、消融、跨物种泛化均有充分验证
- 写作质量: 9/10 — 理论推导清晰，实验组织系统
- 价值: 8/10 — 对抗体工程有直接应用价值，框架可推广但当前限于抗体场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)
- [\[ICML 2026\] Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design](cross-chirality_generalization_by_axial_vectors_for_hetero-chiral_protein-peptid.md)
- [\[ICML 2026\] DNAChunker: Learnable Tokenization for DNA Language Models](dnachunker_learnable_tokenization_for_dna_language_models.md)
- [\[ICML 2026\] What Makes a Representation Good for Single-Cell Perturbation Prediction?](what_makes_a_representation_good_for_single-cell_perturbation_prediction.md)
- [\[ICML 2026\] CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)

</div>

<!-- RELATED:END -->
