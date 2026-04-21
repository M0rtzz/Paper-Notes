---
title: >-
  [论文解读] Towards Fairness Assessment of Dutch Hate Speech Detection
description: >-
  [ACL 2025][AI安全][仇恨言论检测] 本文系统评估了荷兰语仇恨言论检测模型的反事实公平性，提出四种反事实数据生成方法（LLMdef、LLMlist、SLL、MGS），并通过在 BERTje 模型上微调验证了反事实数据增强对模型性能和公平性的改进效果。
tags:
  - ACL 2025
  - AI安全
  - 仇恨言论检测
  - 反事实公平性
  - 荷兰语NLP
  - 偏见缓解
  - 数据增强
---

# Towards Fairness Assessment of Dutch Hate Speech Detection

**会议**: ACL 2025  
**arXiv**: [2506.12502](https://arxiv.org/abs/2506.12502)  
**代码**: https://github.com/Juulba/Dutch-counterfactual-fairness  
**领域**: AI安全  
**关键词**: 仇恨言论检测, 反事实公平性, 荷兰语NLP, 偏见缓解, 数据增强  


## 一句话总结

本文系统评估了荷兰语仇恨言论检测模型的反事实公平性，提出四种反事实数据生成方法（LLMdef、LLMlist、SLL、MGS），并通过在 BERTje 模型上微调验证了反事实数据增强对模型性能和公平性的改进效果。

## 研究背景与动机

仇恨言论检测领域存在三个核心问题：

**语言偏向**：绝大多数研究聚焦英语，荷兰语等欧洲语言的研究严重不足

**公平性缺失**：现有荷兰语仇恨言论检测工作主要关注数据集构建和模型开发，几乎没有探讨模型的公平性问题

**反事实评估空白**：针对荷兰语的反事实公平性研究和相关数据集完全缺失

公平性问题的核心在于：如果将句子中的敏感属性（如国籍、种族、宗教）替换为其他属性，模型的预测结果不应发生变化。例如 "All Moroccans are troublemakers" 和 "All Dutch are troublemakers" 应获得相同的仇恨言论分类概率——但实际模型往往对前者预测 98% 仇恨概率，对后者仅 10%，暴露出严重的偏见。

本文填补了荷兰语仇恨言论检测公平性评估的空白，是首个为荷兰语提供反事实公平性数据集和系统评估的工作。

## 方法详解

### 整体框架

整体流程包含四个阶段：

1. **社会群体术语构建**：手动策划 85 个荷兰语社会群体术语（SGT）列表
2. **反事实数据生成**：使用四种方法生成反事实句子
3. **模型微调**：在 BERTje 基础上用反事实数据微调
4. **评估**：从性能和公平性两个维度全面评估

### 荷兰语社会群体术语（SGT）构建

不同于简单翻译英文 SGT 列表，作者考虑了荷兰语的特殊性：

- **社会语境适配**：如 "Moroccan" 是荷兰社会中的重要少数族群但在美国不是；"Sikhs" 在荷兰仇恨言论中不相关
- **语法变体处理**：荷兰语名词有形容词形式、形容词需要变位，如 "Nederlands"（形容词）和 "Nederlander"（名词）
- 最终策划 85 个 SGT，覆盖国籍、肤色、移民、性别、性取向、宗教、年龄和意识形态

### 四种反事实数据生成方法

**1. LLMdef（隐式 LLM 生成）**

- 指示 LLM 根据身份属性类别（性别、种族、宗教等）自动识别并替换句子中的 SGT
- 替换术语由 LLM 自行决定，更具上下文感知能力
- 生成 15,175 条反事实数据

**2. LLMlist（显式 LLM 生成）**

- 在 LLMdef 基础上提供预定义 SGT 列表
- LLM 从列表中选择替换术语，提供更强的可控性
- 生成 21,562 条反事实数据

**3. SLL（句子对数似然）**

- 基于 GPT-2 计算句子对数似然：$f(x) = \log(P(x)) = \sum_{i=1}^{n} \log P(x_i | x_0, x_1, \ldots, x_{i-1})$
- 将原句中的 SGT 替换为列表中所有其他 SGT，仅保留似然度大于等于原句的替换
- 生成 49,104 条反事实数据

**4. MGS（手动群组替换）**

- 基于身份群组和语法功能构建替换字典
- 仅在相同类别内替换：如 "woman" 只替换为其他性别/名词术语（"transgender"、"man"）
- 确保语法正确性，生成 20,393 条反事实数据

### 训练策略

- 基础模型：BERTje（荷兰语预训练 BERT 模型）
- 训练数据：IMSyPP 荷兰语仇恨言论数据集（25,720 训练 + 2,858 评估）
- 四分类标签：appropriate、inappropriate、offensive、violent
- 分别使用四种反事实数据对 BERTje 进行微调

### 公平性评估指标

- **CTF（反事实令牌公平性）**：$CTF(X, X_{cf}) = \sum_{x \in X} \sum_{x' \in X_{cf}} |g(x) - g(x')|$，值越低越公平
- **DPD（人口统计均等差异）**：衡量不同身份群组间预测正例率差异
- **EOD（均等机会差异）**：衡量不同群组间真正例率和假正例率差异

## 实验关键数据

### 模型整体性能对比

| 模型 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| Baseline BERTje | 0.75 | 0.78 | 0.75 | 0.75 |
| BERTje + LLMdef | 0.79 | 0.61 | 0.62 | 0.61 |
| BERTje + LLMlist | 0.77 | 0.65 | 0.52 | 0.61 |
| BERTje + SLL | **0.79** | **0.79** | **0.79** | **0.79** |
| BERTje + MGS | **0.79** | **0.79** | **0.79** | **0.79** |

### 反事实公平性评估（CTF，越低越公平）

| 模型 | Toxic | Non-Toxic | Average |
|------|-------|-----------|---------|
| Baseline | 0.11 | 0.36 | 0.24 |
| BERTje + LLMdef | 0.26 | 0.011 | 0.13 |
| BERTje + LLMlist | 0.32 | 0.001 | 0.16 |
| BERTje + SLL | 0.20 | 0.001 | **0.10** |
| BERTje + MGS | 0.28 | 0.003 | 0.14 |

### 群体公平性评估（越低越公平）

| 模型 | DPD | EOD |
|------|-----|-----|
| Baseline | 0.38 | 0.53 |
| BERTje + LLMdef | 0.09 | 0.18 |
| BERTje + LLMlist | 0.13 | 0.25 |
| BERTje + SLL | **0.06** | **0.11** |
| BERTje + MGS | 0.18 | 0.36 |

### 各类别检测性能（F1）

| 类别 | Baseline | LLMdef | LLMlist | SLL | MGS |
|------|----------|--------|---------|-----|-----|
| Appropriate | 0.81 | 0.85 | 0.84 | 0.85 | 0.85 |
| Inappropriate | 0.37 | 0.38 | 0.36 | 0.34 | 0.35 |
| Offensive | 0.75 | 0.78 | 0.78 | 0.79 | 0.79 |
| Violent | 0.53 | 0.43 | 0.45 | 0.50 | 0.52 |

## 亮点与洞察

1. **SLL 方法全面最优**：尽管 SLL 生成的反事实句子在语法和语义上质量不如 LLM 方法，但在性能（F1=0.79）和公平性（CTF=0.10, DPD=0.06, EOD=0.11）上均取得最佳结果——这揭示了数据质量与模型效果之间的非直觉关系
2. **公平性的"跷跷板效应"**：所有反事实模型在非毒性模板上大幅改善公平性（CTF 从 0.36 降至约 0.001），但在毒性模板上公平性反而恶化（从 0.11 升至 0.20-0.32），说明反事实数据增强可能引入新的偏见
3. **LLM 生成质量高但效果差**：LLM 能生成更符合语法的反事实句子，但其引入的噪声导致分类性能和公平性不如传统方法
4. **荷兰语特有挑战**：形容词变位、名词/形容词双重词性等语法特性使得简单替换方法难以生成正确荷兰语句子，这是英语研究中不存在的问题
5. **首个荷兰语反事实公平性数据集**：填补了荷兰语仇恨言论检测公平性研究的空白

## 局限性与可改进方向

1. **类别不平衡**：inappropriate 和 violent 类别样本量少，导致这两类的检测性能始终较低（F1 < 0.40 和 < 0.55），未来可采用过采样或类别权重策略
2. **毒性类反事实公平性退化**：反事实模型在毒性模板上的 CTF 反而升高，根本原因尚未探明，可能与不真实反事实引入的噪声有关
3. **单一模型评估**：仅评估了 BERTje 模型，未覆盖多语言模型（如 mBERT、XLM-R）或更大规模 LLM
4. **缺乏消融实验**：未研究训练数据量变化对性能和公平性的影响，难以区分"数据量增加"和"反事实数据质量"的各自贡献
5. **模板化评估局限**：公平性评估使用的模板数据仅 34 个短句模板，覆盖场景有限
6. **可扩展性**：方法可推广到其他欧洲小语种（如丹麦语、瑞典语），验证方法的跨语言泛化能力

## 相关工作与启发

- **Davani et al. (2021)**：英语反事实公平性仇恨言论检测的先驱工作，本文的 SGT 构建和 SLL 方法直接借鉴于此
- **Garg et al. (2019)**：提出 CTF 指标用于量化反事实公平性
- **Kusner et al. (2017)**：反事实公平性的理论基础，定义了"个体在反事实世界中属于不同人口群组时决策应一致"
- **Caselli et al. (2021)**：DALC v1.0 荷兰语辱骂语言语料库，为荷兰语仇恨言论研究提供基础
- 本文的核心启发：**数据增强的质量与下游任务效果并非线性正相关**，传统方法生成的"质量较低"数据可能因为引入更多样的分布偏移而促进模型公平性

## 评分

- 新颖性: ⭐⭐⭐ — 方法本身非原创（移植英语方法到荷兰语），但填补了荷兰语公平性评估的空白
- 实验充分度: ⭐⭐⭐⭐ — 四种方法对比充分，包含性能和多维公平性评估，但缺消融实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，案例分析详实，定性和定量评估结合得当
- 价值: ⭐⭐⭐ — 发布了首个荷兰语反事实数据集，但技术贡献有限，主要是已有方法的应用和对比

<!-- RELATED:START -->

## 相关论文

- [Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](../../ICLR2026/ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)
- [SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](speechfake_a_largescale_multilingual_speech_deepfake.md)
- [Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](gifi_gender_fairness.md)
- [Improving Fairness of Large Language Models in Multi-document Summarization](improving_fairness_of_large_language_models_in_multi-document_summarization.md)
- [FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)

<!-- RELATED:END -->
