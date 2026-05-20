---
title: >-
  [论文解读] On the Mutual Influence of Gender and Occupation in LLM Representations
description: >-
  [ACL2025][LLM/NLP][gender bias] 通过在 LLM 嵌入空间中近似性别方向（gender direction），系统研究了名字的性别表征与职业上下文之间的双向影响：职业上下文会偏移名字的性别表征，而名字的性别表征反过来影响 LLM 在职业预测任务中的偏差行为…
tags:
  - "ACL2025"
  - "LLM/NLP"
  - "gender bias"
  - "occupation stereotype"
  - "LLM embeddings"
  - "gender direction"
  - "first name representation"
---

# On the Mutual Influence of Gender and Occupation in LLM Representations

**会议**: ACL2025  
**arXiv**: [2503.06792](https://arxiv.org/abs/2503.06792)  
**代码**: 无公开代码  
**领域**: LLM/NLP  
**关键词**: gender bias, occupation stereotype, LLM embeddings, gender direction, first name representation

## 一句话总结
通过在 LLM 嵌入空间中近似性别方向（gender direction），系统研究了名字的性别表征与职业上下文之间的双向影响：职业上下文会偏移名字的性别表征，而名字的性别表征反过来影响 LLM 在职业预测任务中的偏差行为，但二者的相关性仅为中等强度。

## 研究背景与动机
名字常被用作性别的代理变量（proxy），社会科学研究已证明性别相关的名字刻板印象会导致教育和就业中的歧视性待遇。LLM 被发现在招聘决策、推荐信撰写等任务中表现出类人的性别偏见，但现有工作主要采用黑箱方法，未深入探究偏差的内在机制。

本文的核心问题：
- LLM **内部**如何表征名字的性别？这些表征是否与真实世界的性别分布一致？
- 职业上下文（如"护士"vs"程序员"）是否会**改变**名字的性别表征？
- 这些内部性别表征能否**解释** LLM 在下游职业预测中的偏差行为？

与以往工作的区别：以往关于嵌入偏差的研究（Bolukbasi et al. 2016）主要在静态词嵌入或早期上下文化模型上进行，本文首次在现代 LLM（Llama-3.1、Mistral 等）中建立了**内部性别表征**与**外在偏差行为**之间的联系。

## 方法详解

### 整体框架
研究分三步推进：(1) 在 LLM 嵌入空间中近似性别方向并验证其质量；(2) 分析名字性别表征与真实世界统计及职业上下文的关系；(3) 在下游职业预测任务中研究性别表征对模型偏差行为的影响。

### 关键设计1：性别方向近似与验证
**近似方法**：采用 PCA 分解性别词对（如 she/he、woman/man 等 9 对，排除 Mary/John 以避免过拟合）的嵌入差异矩阵。具体步骤：
1. 从英文 Wikipedia 提取每个性别词的 3000 个上下文句子，并通过反事实替换生成配对句子
2. 计算每对性别词的平均上下文化嵌入差异
3. 对差异矩阵做 PCA，第一主成分即为近似的性别方向 $\vec{g}$

**验证方法**：设计二分类任务——用名字嵌入（或其与性别方向的点积）预测名字的关联性别。如果点积特征能保持与原始高维嵌入相当的分类精度，则说明性别方向有效捕获了性别信息。

PCA 结果显示，四个模型的第一主成分解释方差比例为 32-42%，显著高于后续成分。第一主成分作为性别方向在分类任务中表现与完整嵌入相当甚至更优（如 OLMo-7B 从 76.60% 提升至 80.57%），而第二主成分和平均方向均无法保持精度。

### 关键设计2：职业上下文对性别表征的影响
使用模板句 "{NAME} is a/an {OCC.}. {NAME} is " 构造包含职业信息的上下文，分析名字嵌入在提及职业前后的变化：
- 计算名字嵌入与性别方向的点积变化 $\Delta \text{DOT}(\vec{n}_{\text{temp}}, \vec{g})$
- 同时获取模型输出 "female"/"male" token 的概率变化

28 个职业来自 Bias in Bios 数据集，性别比例各异。另设"person"作为无刻板印象的基线。

### 关键设计3：下游职业预测中的偏差分析
在 Bias in Bios 数据集上进行零样本职业预测：
- 对 28 个职业各采样 135 篇男性和 135 篇女性传记
- 将传记中的名字占位符替换为 470 个候选名字
- 每个 LLM 共执行约 355 万次推理
- 用 **偏差系数**（Bias Coefficient，TPR 与名字女性化程度的 Pearson 相关）衡量外在偏差
- 用 **内部系数**（Internal Coefficient，嵌入性别表征与职业预测概率的 Spearman 相关）衡量内在表征的预测力

## 实验关键数据

### 模型与数据
- **模型**：Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, OLMo-7B, Phi-3.5-mini (3.8B)
- **名字**：470 个名字，按 SSA 数据集的女性比例分为 10 个桶，涵盖 4 个种族/民族
- **总推理次数**：超过 1200 万次

### Table 1: 性别方向验证——二分类准确率 (%)

| 模型 | 原始嵌入 | 第一主成分点积 | 第二主成分点积 | 随机方向点积 |
|---|---|---|---|---|
| Llama-3.1-8B | 75.46 | **75.18** | 50.78 | 47.09 |
| Mistral-7B | 74.04 | **67.80** | 58.44 | 55.18 |
| OLMo-7B | 76.60 | **80.57** | 55.46 | 56.03 |
| Phi-3.5-mini | 65.67 | **70.64** | 49.08 | 55.60 |

第一主成分点积在多数模型上保持甚至超过原始嵌入的分类精度，而第二主成分与随机方向接近随机水平，验证了性别方向近似的有效性。

### Table 2: 职业预测案例——名字性别对预测结果的影响（Llama-3.1-8B）

| 传记职业 | 名字 | 名字女性% | 模型预测 | 正确? |
|---|---|---|---|---|
| pastor | Luis | 0.53% | pastor | ✓ |
| pastor | Logan | 7.37% | pastor | ✓ |
| pastor | Jerre | 43.70% | pastor | ✓ |
| pastor | Alejandra | 99.00% | journalist | ✗ |
| pastor | Khadijah | 99.90% | journalist | ✗ |
| dietitian | Duc | 0.00% | personal trainer | ✗ |
| dietitian | Hunter | 5.02% | personal trainer | ✗ |
| dietitian | Ivory | 59.32% | dietitian | ✓ |
| dietitian | Bonnie | 98.78% | dietitian | ✓ |

同一传记仅改变名字，当名字性别与职业刻板印象不匹配时（如女性名字 + pastor，男性名字 + dietitian），模型更易预测错误。

### Figure 6: 偏差系数与内部系数的对比（Llama-3.1-8B）
- 偏差系数与内部系数的 Spearman 相关：**0.61**（Llama-3.1-8B）、**0.76**（Mistral-7B），均 $p < 0.001$
- 表明内部性别表征能**部分**解释外在偏差行为，但并非完全一致
- 某些职业出现不一致：如"nurse"有显著外在偏差但内部系数不显著，"physician"有显著内部系数但无外在偏差

## 关键发现

1. **性别表征与真实世界一致**：LLM 中名字嵌入的性别方向投影值与 SSA 数据集中的真实女性比例呈强线性相关（Pearson 相关显著），说明模型从训练数据中学到了与现实一致的名字-性别关联

2. **职业上下文偏移性别表征**：在提及女性主导职业（如 nurse, 90.9% 女性）后，名字嵌入向女性方向偏移；提及男性主导职业（如 comedian, 21.1% 女性）后向男性方向偏移。性别强指示名字受职业影响较小，性别模糊名字受影响最大

3. **偏差行为的内在解释有限**：内部性别表征与外在偏差行为存在中等相关（0.61-0.76），但存在"假阴性"（有外在偏差但内部系数不显著）和"假阳性"（内部系数显著但无外在偏差）的情况，印证了 intrinsic 与 extrinsic 偏差指标不完全对齐的已有发现

4. **跨模型一致性**：四个不同架构/训练方法的模型均表现出上述趋势，说明这些现象并非某个模型的特例

## 亮点与洞察

- **从黑箱到白箱**：首次将 Bolukbasi et al. 的性别子空间方法系统适配到现代 LLM，并建立了从内部表征到外在行为的完整分析链条，弥合了嵌入偏差研究与行为偏差研究之间的鸿沟
- **名字作为连续变量**：不仅研究强性别指示名字，还包含性别模糊名字，将性别表征视为连续谱而非二元分类，揭示了模型对性别确定性不同的名字有不同的上下文敏感度
- **实验规模充分**：1200 万+ prompts、470 个名字、28 个职业、4 个 LLM，结论建立在大规模统计基础上
- **方法论启示**：性别方向的点积作为单维特征可媲美甚至超越高维嵌入的性别分类精度，说明 LLM 的性别概念确实高度集中在一个主成分上

## 局限性

- **二元性别框架**：性别方向近似基于 female-male 二元定义，无法覆盖非二元性别身份
- **人口统计覆盖有限**：名字仅来自美国 SSA 和选民登记数据，仅包含 4 个种族/民族，缺乏跨文化和跨语言的验证
- **模型规模受限**：全部实验在 4B-8B 参数的小模型上进行（资源限制），更大模型（如 70B、GPT-4）的趋势未知
- **缺乏缓解方案**：论文聚焦于偏差的发现和解释，未提出具体的偏差缓解方法
- **内在-外在指标脱钩**：内部系数只能部分解释外在偏差，存在多个失败案例（如 nurse、physician），表明仅靠嵌入层面的性别表征不足以完整捕捉模型的偏差决策过程

## 相关工作与启发

- **嵌入偏差**：Bolukbasi et al. (2016) 的 Word2Vec 性别子空间方法是本文的直接基础；Basta et al. (2019) 将其扩展到上下文化嵌入；本文进一步适配到指令微调 LLM
- **名字与人口属性**：名字作为性别/种族代理的做法虽有局限（Gautam et al. 2024），但在公平性研究中被广泛使用，本文通过显式引入性别模糊名字增强了分析的细粒度
- **intrinsic vs extrinsic**：Goldfarb-Tarrant et al. (2021), Cao et al. (2022) 已指出内在和外在偏差指标不一致，本文在 LLM 的名字-职业场景中再次证实了这一点
- **启发**：如果要通过嵌入层面的干预（如去偏投影）来消除职业预测偏差，仅处理性别方向可能不够，需同时考虑更高阶的交互效应和注意力机制中的信息流

## 评分

- 新颖性: ⭐⭐⭐ — 方法（PCA 性别方向）并不新颖，贡献主要在于将已有方法系统适配到现代 LLM 并建立内外联系
- 实验充分度: ⭐⭐⭐⭐⭐ — 1200万+ prompts、4 模型、470 名字、28 职业，统计检验严谨，消融全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰逻辑连贯，伦理讨论充分，但数学符号较多增加阅读负担
- 价值: ⭐⭐⭐ — 发现有意义但偏描述性，内在表征仅能部分解释外在偏差的结论某种程度上削弱了方法的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](../../AAAI2026/llm_nlp/icl-router_in-context_learned_model_representations_for_llm_routing.md)

</div>

<!-- RELATED:END -->
