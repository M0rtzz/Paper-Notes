---
title: >-
  [论文解读] GenePheno: Interpretable Gene Knockout-Induced Phenotype Abnormality Prediction Framework
description: >-
  [AAAI 2026][可解释性][基因序列] 本文提出 GenePheno，首个从基因序列端到端预测基因敲除诱导表型异常的可解释多标签预测框架，通过对比式多标签学习捕获表型间相关性、互斥正则化强制生物学一致性、以及基因本体（GO）瓶颈层提供可解释性…
tags:
  - "AAAI 2026"
  - "可解释性"
  - "基因序列"
  - "表型异常预测"
  - "对比学习"
  - "互斥正则化"
  - "基因本体瓶颈层"
---

# GenePheno: Interpretable Gene Knockout-Induced Phenotype Abnormality Prediction Framework

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.09512](https://arxiv.org/abs/2511.09512)  
**代码**: 无  
**领域**: 生物信息学 / 多标签分类  
**关键词**: 基因序列, 表型异常预测, 对比学习, 互斥正则化, 基因本体瓶颈层

## 一句话总结

本文提出 GenePheno，首个从基因序列端到端预测基因敲除诱导表型异常的可解释多标签预测框架，通过对比式多标签学习捕获表型间相关性、互斥正则化强制生物学一致性、以及基因本体（GO）瓶颈层提供可解释性，在 4 个数据集上取得 SOTA 的基因中心 $F_{\max}$ 和表型中心 AUC。

## 研究背景与动机

理解基因序列如何塑造表型是生物学的根本挑战。现有方法分两类：(1) 变异效应预测方法从基因序列出发预测特定变异对有限表型的影响程度；(2) 大规模表型异常预测方法依赖人工整理的蛋白质-蛋白质相互作用（PPI）网络或基因本体（GO）注释等策展信息作为输入，可扩展性差，且对新发现或注释稀少的基因不适用。

这两类方法的共同限制是：(a) 大多数方法将多标签预测建模为独立二分类任务，忽略了表型之间的固有关联（如多效性——同一基因影响多个表型）；(b) 表型本体中存在语义互斥（如肌张力增高和肌张力减低不应同时出现），但现有方法不能保证逻辑一致性；(c) 缺乏对从基因信息到表型形成的中间功能机制的可解释性。

核心 idea：构建从 DNA 序列到表型的端到端预测框架，利用对比学习建模标签相关性，用互斥正则化强制生物约束，并通过 GO 瓶颈层提供关于表型形成机制的可解释概念。

## 方法详解

### 整体框架

GenePheno 接受两种输入：(1) 基因序列嵌入 $\mathbf{e}_i$（由 GENERator 编码器提取）；(2) 细粒度 GO 功能嵌入 $\mathbf{h}_i$（由 GoBERT 编码器生成）。两者通过交叉注意力机制融合，经注意力池化和拼接得到多模态嵌入 $\mathbf{x}_i$，最终通过 MLP 和 GO 瓶颈层预测多标签表型。

总训练目标为：$\mathcal{L} = \mathcal{L}_{\text{MLC}}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda_1 \mathcal{L}_{\text{ex}}(\hat{\mathbf{y}}, \mathbf{y}, \mathcal{E}) + \lambda_2 \mathcal{L}_{\text{MLC}}^{\text{GO}}(\hat{\mathbf{g}}, \mathbf{g})$

### 关键设计

1. **对比式多标签学习目标（Contrastive Multi-Label Learning）**:

    - 功能：替代标准 BCE 损失，用从 InfoNCE 推导而来的对比损失建模多标签预测
    - 核心思路：将正标签和负标签对应的 logit 视为对比学习中的正例和负例。正项 $\mathcal{L}_{\text{NCE}}^+ = \sum_{i \in \Omega_+} \log(e^{s_i/\tau} + \sum_{j \in \Omega_-} e^{-s_j/\tau}) - s_i/\tau$ 拉近正标签聚类并推离负标签；负项 $\mathcal{L}_{\text{NCE}}^-$ 对称地处理负标签
    - 设计动机：BCE 损失假设标签独立且产生高阶指数项加剧类别不平衡。对比损失隐式建模标签依赖（通过全局池化正负 logit），且在 $\tau=1$ 时可退化为 ZLPR 损失，继承其抗不平衡特性
    - 与传统方法的区别：从无监督对比学习 InfoNCE 推导到有监督多标签场景，是一种原理性扩展而非经验设计

2. **表型互斥正则化（Phenotype Exclusivity Regularization）**:

    - 功能：利用 LLM 自动识别表型本体中互斥的表型对，并通过正则化项防止模型同时预测互斥表型
    - 核心思路：对每对互斥表型 $(i,j) \in \mathcal{E}$，施加 Softplus 正则化 $\mathcal{L}_{\text{ex}} = \frac{1}{N}\sum_{n=1}^N \sum_{(i,j)\in\mathcal{E}} \log(1 + e^{s_i(\mathbf{x}_n) + s_j(\mathbf{x}_n)})$
    - 理论保证（Proposition 1）：在一阶稳定点处，互斥对中至少一个 logit $\leq 0$，即不会同时预测两个互斥表型
    - 理论保证（Theorem 1）：互斥正则化同时起到隐式范数正则化的作用，提供不弱于无正则化损失的泛化界；且冲突预测概率以不低于泛化误差的速率衰减
    - 设计动机：表型本体中的互斥关系（如 HPO 中的肌张力减低 vs 肌张力增高）代表重要的生物学先验，标准对比损失无法编码这些语义约束

3. **GO 功能瓶颈层（Gene Function Bottleneck Layer）**:

    - 功能：在网络倒数第二层设置瓶颈，其 $n$ 个特定节点的输出作为粗粒度 GO 功能的预测
    - 核心思路：细粒度 GO（深度 $d>2$ 的节点）作为输入通过交叉注意力与序列融合；粗粒度 GO（深度 $d=2$ 的节点）作为瓶颈层的监督目标，使用额外对比损失 $\mathcal{L}_{\text{MLC}}^{\text{GO}}$
    - 设计动机：双重作用——(a) 训练时引入生物学功能信息指导端到端学习；(b) 推理时瓶颈层的连接权重 $w_{ij}$ 量化 GO 功能与表型之间的关联，提供可解释的机制洞察
    - 与现有方法的区别：DeepPheno 等方法将 GO 功能作为输入特征，而 GenePheno 将其作为中间监督信号，实现了从序列到功能再到表型的信息通路建模

4. **双粒度 GO 信息整合**:

    - 功能：在输入层使用细粒度 GO 嵌入、在瓶颈层使用粗粒度 GO 标签
    - 核心思路：细粒度 GO 通过 UniEntrezDB 获取有明确实验证据的注释，由 GoBERT 生成嵌入后与序列交叉注意力融合；粗粒度 GO 通过沿"is_a"和"part_of"关系在 GO DAG 中传播得到二值向量
    - 设计动机：细粒度功能提供丰富的语义信息辅助特征提取，粗粒度功能代表一般生物学机制，适合作为可解释的中间概念

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{\text{MLC}} + \lambda_1 \mathcal{L}_{\text{ex}} + \lambda_2 \mathcal{L}_{\text{MLC}}^{\text{GO}}$，三项分别对应表型预测、互斥约束和功能瓶颈。温度参数 $\tau$ 控制对比损失的锐度。

## 实验关键数据

### 主实验

在 4 个数据集（MPO、HPO、GWAS、CAFA2 wPPI）上进行评估，按表型频率分层（11-30、31-100、101-300、≥301）。

| 数据集 | 指标 | GenePheno | 次优方法 | 提升 |
|--------|------|-----------|----------|------|
| MPO | $F_{\max}$ (All) | 31.14 | 29.66 (InterLabelGO) | +1.48 |
| MPO | AUC (All) | 67.86 | 59.89 (DeepPheno) | +7.97 |
| HPO | $F_{\max}$ (All) | 43.17 | 40.45 (InterLabelGO) | +2.72 |
| HPO | AUC (All) | 71.44 | 58.17 (DeepPheno) | +13.27 |
| GWAS | $F_{\max}$ (All) | 40.54 | 37.83 (InterLabelGO) | +2.71 |
| GWAS | AUC (All) | 56.34 | 53.53 (kmer2Vec) | +2.81 |
| CAFA2 wPPI | $F_{\max}$ (All) | 37.94 | 37.55 (DeepPheno) | +0.39 |
| CAFA2 wPPI | AUC (All) | 62.53 | 58.11 (GraphPheno) | +4.42 |

### 消融实验

| 配置 | HPO $F_{\max}$ | HPO AUC | MPO $F_{\max}$ | MPO AUC |
|------|----------------|---------|----------------|---------|
| 完整模型 | 43.17 | 71.44 | 31.14 | 67.86 |
| 去掉 GO 输入 | 39.96 | 56.60 | 28.21 | 54.19 |
| 去掉序列输入 | 40.99 | 60.23 | 29.13 | 64.72 |
| 去掉对比损失 | 41.83 | 65.75 | 29.99 | 64.64 |
| 去掉互斥损失 | 43.07 | 65.22 | 29.92 | 62.92 |
| 去掉瓶颈损失 | 42.94 | 68.17 | 30.13 | 64.67 |

### 关键发现
- 对比损失的移除造成最大性能下降，验证了建模表型相关性的重要性
- GO 输入的移除对 AUC 影响最大（HPO 上从 71.44 降到 56.60），说明功能信息是核心特征
- 仅用 GO（去掉序列）在 MPO 上 AUC 达到 64.72，但 $F_{\max}$ 仅 29.13，说明序列和GO互补
- 互斥损失和瓶颈损失各带来一致但温和的提升
- GenePheno 在低频表型（11-30次）上也有显著优势，说明对比损失有效缓解了类别不平衡
- Case study 中瓶颈权重揭示了生物学上合理的功能-表型关联（如异构酶活性与能量代谢、突触后膜与肌肉无力）

## 亮点与洞察
- 首个直接从基因序列预测大规模表型异常的端到端框架，填补了序列到表型的模态鸿沟
- 从 InfoNCE 到多标签对比损失的推导清晰且有原理性，不仅是经验设计
- 互斥正则化有完整理论支撑（稳定点分析 + 泛化/冲突概率界），不只是一个 trick
- 用 LLM 自动挖掘表型本体中的互斥对，方法论上很新颖且可扩展到其他本体
- GO 瓶颈层同时服务于性能和可解释性，是设计的亮点

## 局限与展望
- 在 GWAS 等小数据集上提升相对有限（$F_{\max}$ 仅 +2.71）
- CAFA2 数据集需要 PPI 信息来保证公平比较，限制了与不需要 PPI 的方法的对比范围
- 互斥对的质量依赖 LLM 的判断，可能引入噪声
- 推理时的可解释性主要依赖瓶颈权重的热力图，仍需领域专家验证
- 基因序列编码器（GENERator）是预训练模型，端到端微调的计算开销较大
- 当前仅考虑单基因敲除，多基因敲除的组合效应未涉及

## 相关工作与启发
- 从 InfoNCE 推导多标签损失的范式可直接推广到其他具有标签相关性的多标签分类任务
- 互斥正则化 + LLM 挖掘约束的方法论可应用于任何具有本体结构的分类问题（如疾病分类、物种分类）
- 瓶颈层提供可解释中间概念的思路借鉴了概念瓶颈模型（CBM），但在生物学场景下有自然的对应（GO功能）
- DeepPheno、SSLPheno 等依赖策展数据的方法提供了对比基线，GenePheno 的序列直接输入方式在可扩展性上有本质优势

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](../../ACL2026/interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2026\] IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](../../ACL2026/interpretability/idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)
- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](../../ICLR2026/interpretability/when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)
- [\[ICML 2026\] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions](../../ICML2026/interpretability/bridging_the_knowledge-prediction_gap_in_llms_on_multiple-choice_questions.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)

</div>

<!-- RELATED:END -->
