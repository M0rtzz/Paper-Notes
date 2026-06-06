---
title: >-
  [论文解读] Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations
description: >-
  [ACL 2025][LLM/NLP][道德价值观对齐] 提出基于词语联想（word association）而非直接提问的 LLM 道德评估框架，构建人类和 LLM 的全局道德网络（GMN），发现两者在正面道德维度上高度一致，但 LLM 在负面道德概念上系统性地更抽象、更少情感化和具体性。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "道德价值观对齐"
  - "词语联想"
  - "道德基础理论"
  - "心理词库"
  - "全局道德网络"
---

# Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations

**会议**: ACL 2025  
**arXiv**: [2505.19674](https://arxiv.org/abs/2505.19674)  
**代码**: [https://github.com/ChunhuaLiu596/Word_Association_Generation](https://github.com/ChunhuaLiu596/Word_Association_Generation)  
**领域**: LLM/NLP / AI Safety  
**关键词**: 道德价值观对齐、词语联想、道德基础理论、心理词库、全局道德网络

## 一句话总结

提出基于词语联想（word association）而非直接提问的 LLM 道德评估框架，构建人类和 LLM 的全局道德网络（GMN），发现两者在正面道德维度上高度一致，但 LLM 在负面道德概念上系统性地更抽象、更少情感化和具体性。

## 研究背景与动机

**领域现状**：LLM 正被大规模部署到各类现实应用中，理解其内部编码的道德价值观对于 AI 安全至关重要。当前主流做法是直接用道德问卷（如 Moral Foundations Questionnaire）提示 LLM，观察其同意/不同意的回答来评估道德对齐。

**现有痛点**：直接提问方式存在四重缺陷——(1) 道德问卷可能已泄露到训练数据中，LLM 只是在"背答案"；(2) LLM 对提示措辞高度敏感，换个说法就给不同答案；(3) 二元回答（同意/不同意）无法捕捉细粒度的道德推理；(4) LLM 的 next-token prediction 本质使其倾向于输出"社会期望答案"，而非真实的道德倾向。Ji et al. (2024) 已经证实 LLM 对道德的理解是表面的、受训练数据短语主导的。

**核心矛盾**：直接评估 LLM 道德观的方法本身就不可靠——既无法区分"真正的道德理解"和"训练数据记忆"，也无法避免提示偏差，导致评估结果缺乏信度。

**本文目标** (1) 如何在不直接提问道德立场的情况下探测 LLM 的道德概念组织方式？(2) 如何系统化比较人类与 LLM 的道德价值差异并解释差异成因？

**切入角度**：心理学中词语联想测试（Word Association Test）已被证明能有效反映人类的道德推理过程。参与者对一个 cue 词自由联想，其响应模式能间接揭示其概念网络中的道德组织结构。Ramezani & Xu (2024) 已在人类联想数据上验证了这一范式的有效性，但仅使用局部子图、未扩展到 LLM。

**核心 idea**：用词语联想替代直接提问来间接探测 LLM 的道德概念结构，并通过全局图随机游走传播道德信息，实现人类—LLM 的系统化道德比较。

## 方法详解

### 整体框架

三阶段框架：(1) 从人类（已有 Small World of Words 数据集，~90K 参与者）和 LLM（提示 Llama-3.1-8B-Instruct）分别收集 12K cue 词的联想响应，构建两个联想图 wa-h 和 wa-l；(2) 基于 Moral Foundation Theory（MFT）的 626 个种子词，通过归一化随机游走在两个联想图中传播 5 维道德值（Care、Fairness、Loyalty、Authority、Sanctity），得到两个全局道德网络 gmn-h 和 gmn-l；(3) 从宏观维度相关性到微观概念级定性分析，系统比较两个道德网络的异同。

### 关键设计

1. **LLM 词语联想采集与温度校准**:

    - 功能：从 Llama 获取结构上可与人类联想数据对齐的大规模联想图
    - 核心思路：对 12K cue 词，每个提示 Llama 100 次（Monte-Carlo 近似联想概率分布），每次生成最多 3 个联想词，使用与人类实验 (Small World of Words) 完全相同的指令。关键超参数是温度 $T$：作者同时优化两个指标——多样性（response types 总数）和可靠性（split-half reliability，即 Spearman-Brown 公式 $r_{total} = 2r_{half}/(1 + r_{half})$），找到 $T = 2.1$ 时两个指标的差距同时最小化，使 wa-l 在结构上趋近 wa-h
    - 设计动机：LLM 默认温度下联想多样性远低于人类（Abramski et al., 2024 已指出此问题）。不校准温度，两个联想图在结构层面就不可比，后续所有道德分析都失去意义。同时优化两个指标（而非只调多样性）避免了"多样但不可靠"的问题

2. **全局道德网络传播（GMN）**:

    - 功能：将 MFD 种子词的 5 维道德标签扩散到联想图中所有 12K 节点，得到每个概念的道德评分向量
    - 核心思路：初始化道德矩阵 $F_0 \in \mathbb{R}^{|n| \times 5}$，仅 626 个 MFD 种子词有非零值（virtue=+1, vice=-1）。然后迭代传播 $F_{t+1} = \alpha S F_t + (1-\alpha) F_0$，其中 $S = D^{-1/2}WD^{-1/2}$ 是对称归一化邻接矩阵。实际使用闭合解 $F^* = (I - \alpha S)^{-1} F_0$。超参数 $\alpha$ 控制传播强度：gmn-h 最佳 $\alpha = 0.75$，gmn-l 最佳 $\alpha = 0.9$
    - 设计动机：相比 MAG（Ramezani & Xu, 2024）的局部子图方法，全局传播能捕获多跳远距离道德关联（如 "mother" → "birth" → "life"）。gmn-l 需要更大 $\alpha$ 是因为 LLM 联想图更稀疏（密度 0.007 vs 0.013、直径 4 vs 3、连通分量 77 vs 114），道德信息需要更强的"推力"才能传播到远端节点

3. **多粒度道德对齐分析**:

    - 功能：从维度级、概念级、语义特征级三个层面系统解释人类与 LLM 道德差异的成因
    - 核心思路：(a) 维度级：用 eMFD（2186 评估词）的 Spearman 相关对比 5 个道德维度的预测精度；(b) 概念级：对比 gmn-h/gmn-l 中 top 正面/负面道德概念的重叠和分歧，识别极性翻转概念（如 "abortion" 人类偏负而 LLM 偏正）；(c) 语义特征级：用 VAD-norms 情感词典和 Brysbaert 具体性词典量化两组联想响应在情感性（emotional intensity）和具体性（concreteness）上的系统差异
    - 设计动机：仅看维度级相关性无法解释"为什么"差异存在。通过逐层深入到具体概念和语义特征，能揭示 LLM 道德偏差的根本原因——基于文本共现的统计关联 vs 基于感官经验的人类联想

### 损失函数 / 训练策略

本文不训练模型。使用现成的 Llama-3.1-8B-Instruct（15T token 预训练 + RLHF），核心超参数为联想温度 $T = 2.1$ 和传播系数 $\alpha$（gmn-h: 0.75, gmn-l: 0.9）。$\alpha$ 使用 eMFD 中 277 个非评估词调参优化。

## 实验关键数据

### 主实验

**道德值预测（Spearman 相关）与 eMFD ground-truth 对比：**

| 道德维度 | MAG (baseline) | gmn-h (人类图) | gmn-l (LLM图) |
|---------|----------------|---------------|---------------|
| Care (n=1895) | 0.29 | **0.47** | 0.46 |
| Sanctity (n=1893) | 0.25 | 0.39 | **0.44** |
| Fairness (n=1514) | 0.23 | 0.29 | **0.32** |
| Authority (n=1737) | 0.21 | 0.19 | **0.25** |
| Loyalty (n=1714) | 0.30 | 0.26 | **0.30** |
| 总体 (n=8753) | 0.20 | 0.28 | **0.29** |

### 情感性与具体性对比（top-50 负面道德概念）

| 指标 | Care H/L | Fairness H/L | Loyalty H/L | Authority H/L | Sanctity H/L | 总体 H/L |
|------|----------|-------------|-------------|---------------|-------------|----------|
| 情感响应占比(%) | 72/**61*** | 67/**54*** | 69/**54*** | 67/**59*** | 69/**58*** | 66/**55*** |
| 情感强度 | **4.24**/4.1 | 3.71/3.77 | 3.8/3.82 | 3.78/**4.10*** | 3.81/**3.60*** | 3.30/**3.17*** |
| 具体响应占比(%) | **35**/24* | **24**/12* | **24**/12* | **29**/16* | **40**/33* | **42**/36* |
| 具体性评分 | **3.0**/2.7* | **2.6**/2.2* | **2.5**/2.3* | **2.7**/2.5* | **3.2**/3.0* | **3.1**/2.9* |

*注：H=gmn-h（人类），L=gmn-l（LLM），\* 表示 t 检验 p<0.05 显著差异*

### 关键发现

- **全局传播大幅优于局部方法**：GMN 在所有维度上均超越 MAG baseline，总体相关性从 0.20 提升至 0.28-0.29，验证了多跳全局传播捕获远距离道德关联的有效性
- **正面道德一致性远高于负面**：gmn-h 和 gmn-l 在 top 正面概念上高度重叠（church、religion、God、priest 等均共享），但 top 负面概念严重分歧——人类偏向感官/情感词（disgusting、vomit、hurt），LLM 偏向社会公正词（betrayal、prejudice、discrimination）
- **人类联想系统性地更情感化、更具体**：在所有 5 个道德维度上，人类联想的情感响应占比和具体性评分均显著高于 LLM。例如 "prejudice" 人类联想 "pride, black, race"（具体文化经验），LLM 联想 "stereotypes, biases, bigoted"（抽象概念）
- **极性翻转概念揭示 RLHF 偏差**：LLM 将 "abortion、immigrant、politician" 评为更正面，人类将 "jail、air、plastic" 评为更正面，暗示 RLHF 训练可能注入了特定的社会价值倾向
- **LLM 联想图更稀疏导致传播效率差异**：gmn-l 需要 $\alpha = 0.9$（vs gmn-h 的 0.75）才能达到最佳传播，因为 LLM 概念网络密度仅为人类的 54%

## 亮点与洞察

- **间接探测范式**：用词语联想替代直接道德提问，巧妙回避了训练数据泄露、提示敏感性和社会期望偏差三大问题。这一"不问道德却能推断道德"的思路可迁移到 LLM 的其他隐含属性评估
- **温度双目标校准**：同时优化多样性和可靠性两个互相矛盾的指标找到最优温度，确保 LLM-人类联想数据在结构层面可比。这一方法论可推广到任何需要 LLM 模拟人类行为分布的实验设计
- **图结构差异的解释力**：人类图更密集 → 传播更容易 → 需要更小的 $\alpha$，LLM 图更稀疏 → 需要更强传播力。这不仅是超参数调优的技术细节，更揭示了 LLM 和人类概念组织方式的根本性结构差异
- **定量+定性多层分析**：不停留在宏观相关性数字上，而是逐层深入到具体概念的联想词分析、情感性和具体性量化，提供了令人信服的差异解释

## 局限与展望

- 仅测试 Llama-3.1-8B-Instruct 一个模型，不同架构（GPT、Claude）、不同规模（70B、405B）的 LLM 道德概念组织可能截然不同
- 聚焦西方英语文化，MFT 框架本身存在争议（Atari et al., 2023 建议拆分 Fairness 维度），跨文化泛化性未知
- 随机游走传播可能受 hub 节点（高连接通用词）影响，稀释道德信号的传播精度
- Monte-Carlo 近似（100 次/cue）的充分性未严格验证，Precision@k 在 k>10 后明显下降，说明 LLM 联想长尾与人类仍有差距
- 概念级分析难以直接推广到句子/文档级道德推理场景

## 相关工作与启发

- **vs MAG (Ramezani & Xu, 2024)**：MAG 在局部 cue 子图上传播道德信息，本文扩展为全局图传播，在所有维度上均优于 MAG（总体 0.29 vs 0.20）。局部方法无法捕获多跳道德关联
- **vs 直接道德问卷 (Ji et al., 2024; Abdulhai et al., 2023)**：直接用 MFQ 提问 LLM 存在数据泄露和提示敏感性问题，本文通过间接联想范式避免了这些缺陷，获得更鲁棒的道德画像
- **vs LLM 联想研究 (Abramski et al., 2024)**：前人发现 LLM 联想多样性低于人类但未解决该问题，本文通过温度校准使结构对齐，并首次将联想应用于道德维度分析

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 词语联想+全局道德网络的方法论组合在 LLM 道德评估中带来了全新视角
- 实验充分度: ⭐⭐⭐ 主实验和分析扎实但仅测一个 LLM，跨模型泛化性未知
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、分析层次分明，从宏观到微观层层深入
- 价值: ⭐⭐⭐⭐ 为 LLM 道德对齐提供了新的间接评估范式，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans](psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)
- [\[ACL 2025\] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
