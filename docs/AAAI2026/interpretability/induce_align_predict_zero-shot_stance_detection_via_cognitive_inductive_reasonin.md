---
title: >-
  [论文解读] Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning
description: >-
  [AAAI 2026][零样本立场检测] 提出CIRF（Cognitive Inductive Reasoning Framework），受认知科学启发，从原始文本中无监督归纳一阶逻辑推理模式（schema），构建多关系schema图，用图核模型对齐输入与schema模板实现可解释的零样本立场推理，在SemEval-2016、VAST和COVID-19-Stance上达到SOTA，仅30%数据即可匹配全量。
tags:
  - AAAI 2026
  - 零样本立场检测
  - 认知推理
  - 一阶逻辑
  - 图核模型
  - schema
---

# Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning

**会议**: AAAI 2026  
**arXiv**: [2506.13470](https://arxiv.org/abs/2506.13470)  
**代码**: 无  
**领域**: NLP理解 / 立场检测  
**关键词**: zero-shot stance detection, cognitive schema, first-order logic, graph kernel, low-resource

## 一句话总结

提出CIRF框架，通过无监督schema归纳（USI）从LLM生成的一阶逻辑中抽象可迁移推理模式，再用schema增强图核模型（SEGKM）进行结构对齐实现可解释零样本立场推理，在三个基准上达到SOTA且仅需30%标注数据。

## 研究背景与动机

**领域现状**：零样本立场检测（ZSSD）需要对训练中未见过的目标判断文本立场，对分析快速涌现的极化社交媒体话题至关重要。

**现有痛点**：
- LLM零样本提示方法在复杂推理上表现不佳，泛化能力有限（GPT-3.5在SEM16上仅69.8 F1）
- LLM增强微调方法（KAI、FOLAR等）仍需大量标注数据，停留在实例级模式匹配
- 两类方法都缺乏可解释性和跨目标的推理泛化能力

**核心矛盾**：立场检测需要超越表面词汇的抽象推理（如"增加健康风险"和"降低经济稳定性"都实例化了"负面后果→反对"的推理模式），但现有方法要么做surface-level匹配要么过度依赖标注。

**切入角度**：认知科学中的schema理论——人类从具体经验中归纳出可泛化的推理模式（schema），并应用到新情境。将这种认知能力形式化为一阶逻辑模式的无监督归纳与图核对齐。

## 方法详解

### 整体框架

CIRF包含两个核心模块：(1) USI（无监督Schema归纳）：LLM生成FOL推理→解释抽象→聚类为schema图；(2) SEGKM（Schema增强图核模型）：输入构建FOL图→子图核匹配schema模板→层次化图表示→立场预测。

### 关键设计

**1. 无监督Schema归纳（USI）**

- **功能**：从原始文本中无监督归纳跨目标可迁移的抽象推理模式
- **为什么**：实例级FOL规则无法跨域泛化，需要更高层次的推理抽象
- **怎么做**（四阶段管线）：
    - **FOL推理生成**：对每个句子-目标对，提示LLM生成一阶逻辑推理链
    - **FOL解释与抽象**：提示LLM分析FOL内部逻辑，生成逻辑等价但结构不同的变体，然后总结为泛化模板。例如：`∀x, (is_robot(x) → (helps_humans(x) → must_be_safe(x)))` 抽象为 `∀x, ((is_target(x) ∧ meets_condition(x)) → entails_consequence(x))`
    - **Schema聚类与层次抽象**：按语义和推理模式相似度聚类FOL模板，大聚类用层次策略（先细分子簇→中间链→合并为schema模板）
    - **Schema图构建**：将归纳的schema作为节点构建多关系图，边表示因果、对比、蕴含等逻辑关系

**2. Schema增强图核模型（SEGKM）**

- **功能**：利用schema知识增强输入推理结构的表示，实现可解释的零样本推理
- **为什么**：标准GNN依赖局部消息传递，难以捕获可复用的高阶推理motif；图核通过显式结构匹配实现更好的泛化
- **怎么做**：
    - **FOL图构建**：输入(x,q)对生成FOL推理链，构建FOL图 $G_f=(V_f, E_f)$，节点为谓词，边为逻辑关系
    - **Schema子图滤波器**：从schema图 $G^{(j)}$ 提取以每个节点为中心的k-hop子图作为滤波器池 $\mathcal{H} = \bigcup_j H^{(j)}$
    - **关系感知节点嵌入**：BERT初始化节点/边嵌入，通过关系投影融合边语义：$x' = \text{ReLU}(x + \text{Proj}(e))$
    - **深度图核响应**：计算输入子图与schema滤波器的p步随机游走核：$\phi_{1,i}(v) = K_p(G_v^f, H_i^{(j)}) = \mathbf{s}^\top W A^{\times p} \mathbf{s}$，其中 $\mathbf{s} = \text{vec}(X_{G_v^f}' \cdot (X_{H_i^{(j)}}')^\top)$
    - **Schema图级选择**：聚合所有节点的核响应选择top-g schema图：$S^{(j)} = \sum_{v \in V_f} \frac{1}{|H^{(j)}|} \sum_{H_i^{(j)} \in H^{(j)}} \phi_{1,i}(v)$
    - **层次化图表示**：多层堆叠核特征提取，最终图表示：$\Phi(G_f) = \text{Concat}(\sum_{v \in G_f} \phi_l(v) | l=0,1,...,L)$

**3. 立场预测**

- 最终图表示输入全连接ReLU层进行三分类（Favor/Against/None），交叉熵损失端到端训练

### 损失函数 / 训练策略

- 损失函数：交叉熵损失
- 优化器：AdamW，batch size 32，学习率 $5 \times 10^{-4}$
- 早停策略（patience=10），最多20 epoch，每0.2 epoch验证一次
- Schema归纳完全无监督，SEGKM在源目标上训练
- 硬件：单张40GB A100 GPU
- LLM默认使用GPT-3.5，也测试了GPT-4o和DeepSeek-v3

## 实验关键数据

### 主实验

在SEM16、VAST、COVID-19三个基准上的零样本立场检测结果（Macro-F1）：

| 方法 | 类型 | SEM16 HC | SEM16 FM | SEM16 LA | VAST All | COVID AF | COVID WA |
|------|------|----------|----------|----------|----------|----------|----------|
| JointCL | BERT微调 | 54.4 | 54.0 | 50.0 | 71.2 | 57.6 | 63.1 |
| GPT-3.5 | LLM提示 | 78.9 | 68.3 | 62.3 | 65.1 | 69.2 | 57.8 |
| COLA | LLM提示 | 81.7 | 63.4 | 71.0 | 73.0 | 65.7 | 73.9 |
| KAI | LLM增强 | 76.4 | 73.7 | 69.4 | 76.3 | - | - |
| LCDA | LLM增强 | 79.8 | 70.0 | 69.4 | 80.3 | - | - |
| FOLAR | LLM增强 | 81.9 | 71.2 | 69.9 | 77.2 | 69.5 | 73.1 |
| LogiMDF | LLM增强 | 75.1 | 67.9 | 68.0 | 76.7 | 70.4 | 75.4 |
| **CIRF** | **Schema** | **80.1** | **74.7** | **73.9** | **80.9** | **74.1** | **81.0** |
| CIRF (GPT-4o) | Schema | 83.2 | 80.4 | 78.2 | 82.8 | 84.9 | 89.4 |

CIRF平均F1：SEM16上76.2（超FOLAR 1.9），VAST上80.9（超LCDA 0.6），COVID-19上超LogiMDF 3.7点。

### 消融实验

| 变体 | 效果描述 |
|------|---------|
| w/o Schema | **性能下降最大**，去除认知schema后跨目标泛化能力严重受损 |
| w/o SEGKM | 性能大幅下降，图核对齐是利用schema知识的关键 |
| w/o SE (边语义) | 中等下降，关系信息对结构匹配有帮助 |
| w/o USI (用简单聚类) | 中等下降，LLM驱动的语义归纳优于简单聚类 |

在VAST(10%)低资源设置下，各组件的性能差距更大，说明每个组件在数据稀缺时更加关键。

### 关键发现
- **三基准全面SOTA**，统计显著（p<0.05）
- **30%数据即可匹配全量方法**：COVID-19上10%数据即超LogiMDF 2.8点，SEM16上20%数据超FOLAR 0.6点
- **LLM可扩展性**：GPT-3.5→GPT-4o，VAST从80.9→82.8，WA从81.0→89.4
- FOL知识优于自然语言知识：CIRF和FOLAR（用FOL）整体优于KAI（用自然语言）
- Schema数量对性能影响小（波动<1点），说明推理可被少量schema充分抽象
- Top-g选择数从2到16性能稳定，模型对超参不敏感

## 亮点与洞察
- **认知科学→NLP的成功迁移**：schema理论形式化为FOL归纳+图核对齐，既有理论深度又有实践效果
- **无监督schema归纳的四阶段设计**精巧：生成→解释抽象→聚类→图构建，逐步从实例到抽象
- **图核替代GNN**的设计选择有说服力——GNN的局部消息传递难以捕获可复用的高阶推理motif
- **30%数据匹配全量**说明归纳的schema确实捕获了可迁移的推理结构，而非过拟合训练分布

## 局限与展望
- Schema归纳依赖LLM质量，在噪声大或极大规模语料上扩展性待验证
- FOL表示可能无法捕获修辞、反讽、隐喻等隐含立场表达
- 未探索多语言/跨文化场景的适用性
- Schema归纳的计算成本（多次LLM调用）未量化分析

## 相关工作与启发
- **vs FOLAR**：同样使用FOL知识，但FOLAR是实例级FOL规则，CIRF归纳为跨目标可迁移的schema，SEM16上超1.9点
- **vs LogiMDF**：LogiMDF也用逻辑推理但操作在谓词/单词层面，未建模关系结构，COVID-19上被超3.7点
- **vs KAI**：KAI用自然语言知识增强，CIRF证明FOL+schema的结构化知识更有效
- **vs 纯LLM提示**：GPT-3.5直接提示在VAST上仅65.1，CIRF达80.9，说明schema引导远超surface-level提示

## 评分
- 新颖性: ⭐⭐⭐⭐ 认知schema理论引入ZSSD是开创性的跨领域融合，USI+SEGKM框架设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三基准全面对比、消融完整、低资源分析、LLM可扩展性、超参敏感性分析、案例研究
- 写作质量: ⭐⭐⭐⭐ 从认知动机到形式化方法的推导清晰，公式符号一致
- 价值: ⭐⭐⭐⭐ 低资源ZSSD的实用方法，schema可迁移性为零样本NLP提供新范式
---
title: >-
  [论文解读] Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning
description: >-
  [AAAI 2026][零样本立场检测] 提出CIRF（Cognitive Inductive Reasoning Framework），受认知科学启发，从原始文本中无监督归纳一阶逻辑推理模式（schema），构建多关系schema图，用图核模型对齐输入与schema模板实现可解释的零样本立场推理，在SemEval-2016、VAST和COVID-19-Stance上达到SOTA，仅30%数据即可匹配全量。
tags:
  - AAAI 2026
  - 零样本立场检测
  - 认知推理
  - 一阶逻辑
  - 图核模型
  - schema
---

# Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning

**会议**: AAAI 2026  
**arXiv**: [2506.13470](https://arxiv.org/abs/2506.13470)  
**代码**: 无  
**领域**: NLP理解 / 立场检测  
**关键词**: 零样本立场检测, 认知推理, 一阶逻辑, 图核模型, schema

## 一句话总结
提出CIRF（Cognitive Inductive Reasoning Framework），受认知科学启发，从原始文本中无监督归纳一阶逻辑推理模式（schema），构建多关系schema图，用图核模型对齐输入与schema模板实现可解释的零样本立场推理，在SemEval-2016、VAST和COVID-19-Stance上达到SOTA，仅30%数据即可匹配全量。

## 研究背景与动机
**领域现状**：零样本立场检测（ZSSD）需要对未见目标判断文本立场。LLM零样本能力和LLM增强方法是两大方向。

**现有痛点**：(a) LLM提示在复杂推理上不足，泛化差；(b) LLM增强方法需大量标注，停留在实例级模式。

**核心矛盾**：立场检测需要跨目标的抽象推理，但现有方法要么surface-level匹配要么过度依赖标注。

**本文目标** 无监督归纳抽象推理模式，实现可解释的零样本立场推理。

**切入角度**：认知科学中的schema理论——人类从具体经验归纳推理模式并应用到新情境。

**核心 idea**：无监督归纳一阶逻辑模式为多关系schema图，通过图核对齐实现可解释零样本推理。

## 方法详解

### 整体框架
输入：文本+目标。输出：立场。Pipeline：文本→一阶逻辑→聚类为schema图→图核对齐→立场。

### 关键设计

1. **Schema归纳（Induce）**：

    - 功能：从文本无监督归纳抽象推理模式
    - 核心思路：文本→一阶逻辑表示→聚类为多关系schema图
    - 设计动机：schema跨目标可迁移

2. **结构对齐（Align）**：

    - 功能：用图核度量输入图和schema图的结构相似度
    - 设计动机：可解释的结构匹配

3. **立场预测（Predict）**：

    - 功能：基于最匹配schema推理立场

### 损失函数 / 训练策略
Schema归纳为无监督，图核匹配不需目标特定标注。

## 实验关键数据

### 主实验

| 数据集 | CIRF | 说明 |
|--------|------|------|
| SemEval-2016 | **SOTA** | 新最优 |
| VAST | **SOTA** | 新最优 |
| COVID-19-Stance | **SOTA** | 新最优 |
| 30%数据 | ≈全量 | 强低资源泛化 |

### 关键发现
- 三基准SOTA，30%数据匹配全量方法
- 可解释性：schema图提供清晰推理路径
- Schema具有跨目标可迁移性

## 亮点与洞察
- **认知科学→NLP成功迁移**：schema概念引入立场检测，提供超越表面匹配的抽象推理能力。
- **30%数据匹配全量**：证明归纳的schema捕获了可迁移的推理结构。
- **可解释性**：图结构匹配比黑盒LLM更透明。

## 局限与展望
- 论文全文不可用，详细消融数据未获取
- Schema归纳质量可能随文本域变化
- 一阶逻辑可能无法捕获修辞和隐喻

## 相关工作与启发
- **vs LLM零样本提示**：CIRF通过schema提供结构化推理
- **vs 传统ZSSD**：CIRF的schema跨目标可迁移，不需目标特定特征

## 评分
- 新颖性: ⭐⭐⭐⭐ 认知归纳推理schema引入零样本立场检测是新颖的跨领域结合
- 实验充分度: ⭐⭐⭐⭐ 三个基准数据集上达到SOTA，低资源场景也有验证
- 写作质量: ⭐⭐⭐⭐ 从认知科学到NLP方法的动机推导清晰
- 价值: ⭐⭐⭐⭐ 低资源零样本立场检测的实用方法，schema的跨目标迁移性有重要价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](../../ACL2026/interpretability/multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](../../NeurIPS2025/interpretability/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](../../CVPR2026/interpretability/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](../../CVPR2025/interpretability/lswag_zero_shot_nas.md)

<!-- RELATED:END -->
