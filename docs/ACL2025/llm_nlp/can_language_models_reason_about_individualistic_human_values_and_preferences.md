---
title: >-
  [论文解读] Can Language Models Reason about Individualistic Human Values and Preferences?
description: >-
  [ACL2025][LLM/NLP][individualistic alignment] 提出"个体价值对齐"范式，构建 IndieValueCatalog 数据集（基于世界价值观调查 WVS 的 9.3 万真人数据），评估并训练语言模型根据个人价值表达陈述推理其在新情境下的价值判断，揭示前沿 LM 仅达 55%-65% 准确率且存在显著的跨群体不公平性。
tags:
  - ACL2025
  - LLM/NLP
  - individualistic alignment
  - value reasoning
  - World Values Survey
  - pluralistic alignment
  - Value Inequity Index
---

# Can Language Models Reason about Individualistic Human Values and Preferences?

**会议**: ACL2025  
**arXiv**: [2410.03868](https://arxiv.org/abs/2410.03868)  
**代码**: [liweijiang/indievalue](https://github.com/liweijiang/indievalue)  
**领域**: llm_nlp  
**关键词**: individualistic alignment, value reasoning, World Values Survey, pluralistic alignment, Value Inequity Index

## 一句话总结

提出"个体价值对齐"范式，构建 IndieValueCatalog 数据集（基于世界价值观调查 WVS 的 9.3 万真人数据），评估并训练语言模型根据个人价值表达陈述推理其在新情境下的价值判断，揭示前沿 LM 仅达 55%-65% 准确率且存在显著的跨群体不公平性。

## 研究背景与动机

多元对齐（pluralistic alignment）近年受到关注，强调 AI 应服务于所有人的多样化需求。然而，现有方法存在根本性局限：

- **预定义分桶问题**：现有多元对齐框架依赖预先选定的多样性维度（人口统计、人格、文化、写作风格等），将个体强行归入粗粒度类别，抹杀了个体内部差异
- **刻板印象风险**：基于人口统计分组的方法可能强化刻板印象，如假设同一年龄、性别、地区的人持有相同价值观
- **组级推断的局限**：即使较细粒度的评估（如 Durmus et al. 2024）也仅做组级别的分布推断，无法捕捉个体变异

核心洞察：真正尊重多样性的方式是从个体出发（bottom-up），而非从预定义类别出发（top-down）。作者提出"个体价值对齐"（individualistic alignment），将多元对齐推向极致——在个人级别建模多样性。

关键挑战在于数据：获取足够丰富的、能代表个人整体价值偏好的多维数据极为困难。

## 方法详解

### 整体框架：IndieValueCatalog 数据集

作者将世界价值观调查（WVS）第七轮的原始问卷数据转化为统一的自然语言陈述格式，构建了 IndieValueCatalog：

- **问题统一化**：将 253 道原始问卷（多选、量表等不同格式）转换为标准化价值表达陈述。例如：原题"你最近觉得安全吗？"（答项"非常安全""比较安全"等）→ 陈述"I feel very secure these days"
- **两种粒度**：精细版（refined）929 条陈述 / 极性版（polar）567 条陈述
- **规模**：93,279 名真实受访者，来自 70+ 个国家，每人平均 242 条价值陈述 + 31 条人口统计陈述
- **组合空间极大**：精细版有 $1.65 \times 10^{139}$ 种组合，极性版有 $3.94 \times 10^{86}$ 种组合，使准确预测个体选择极具挑战

### 关键设计 1：评估框架与 Value Inequity Index

**评估设置**：将每个人的陈述分为示范集（50-200 条）和探测集（39 条，覆盖 13 个 WVS 问题类别）。模型需根据示范陈述，从探测问题的候选中选择最符合该个体价值观的陈述。采用三折交叉验证（200 条示范 + 39 条探测），取平均以避免对特定探测集的过拟合。从 IndieValueCatalog 中采样 800 名人口统计学上平衡的个体作为评估集。

**Value Inequity Index ($\sigma_{\text{Inequity}}$)**：衡量模型在不同人口群体间推理个体价值时的公平性。具体做法是：沿 13 个人口统计维度（收入、地区、年龄等），计算模型在各子群体上准确率的标准差，再对所有维度取平均：

$$\sigma\text{Inequity}_M = \frac{1}{|\mathbb{D}|} \sum_{\mathcal{D}^k \in \mathbb{D}} \sigma\left(\{Acc_M^{\mathbb{I}_{\text{eval}}^{g_{k_t}}} \mid \forall g_{k_t} \in \mathcal{D}^k\}\right)$$

$\sigma_{\text{Inequity}}$ 越低，模型越公平。这一指标与准确率互补：两个模型准确率相同但 $\sigma_{\text{Inequity}}$ 可能差异巨大。

### 关键设计 2：IndieValueReasoner 训练

基于 Llama-3.1-8B 微调，训练数据来自 $\mathbb{I}_{\text{train}}$。每条训练样本包含同一个体的 $d$ 条示范陈述和一个探测问题的候选项，模型输出该个体的选择。

关键训练配置消融：
- **探测粒度**：极性版 / 精细版 / 混合
- **示范数量**：固定 200 条 / 随机 50-200 条（mixed）/ 两者结合
- **训练规模**：每题 100/200/400/800/1600 个个体的数据
- **区域数据**：按洲训练，观察跨区域泛化能力

### 关键设计 3：统计基线对比

为理解 LM 的价值，设计了无需 LM 的统计基线：
- **全局多数投票**：选择全球最多人选的陈述
- **相似个体匹配（Resemble）**：找到训练集中示范陈述最相似的个体/群体，用其选择作为预测
- 统计基线可利用全部 92K 人的数据（oracle access），与仅用 1.6K 人训练的 LM 形成对比

## 实验关键数据

### Table: 前沿 LM 的零样本个体价值推理准确率

| 模型 | 准确率 (%) | $\sigma_{\text{Inequity}}$ ↓ |
|---|---|---|
| Random | ~36.6 | — |
| GPT-4o (0806, 无示范) | 54.8 | 3.03 |
| GPT-4o (0806, 200示范) | 63.5 | 3.03 |
| GPT-4o (0513) | 63.7 | 2.87 |
| GPT-4o-mini | ~58 | 2.55 |
| GPT-4-turbo | ~62 | 2.83 |
| Llama-3.1-8B | ~55 | 2.97 |
| Llama-3.1-70B | 63.7 | **1.94** |
| Claude-3.5 Sonnet | ~60 | 3.14 |
| Qwen2-72B | ~61 | 3.24 |

所有模型准确率仅 55%-65%，远未达到令人满意的水平。GPT-4o (0513) 和 Llama-3.1-70B 准确率相同（63.7），但后者的公平性指标显著更优（1.94 vs 2.87）。

### Table: IndieValueReasoner 结果（200 条示范，评估集）

| 方法 | Polar Acc | Refined Acc | 总体 Acc |
|---|---|---|---|
| Random | 45.37 | 27.87 | 36.62 |
| 全局多数投票 | 64.95 | 48.70 | 56.83 |
| Resemble (全部, top cluster) | 73.73 | 59.47 | 66.60 |
| GPT-4o (无示范) | 54.79 | 33.06 | 43.93 |
| GPT-4o (仅人口统计) | 60.31 | 40.69 | 50.50 |
| GPT-4o (200示范) | 63.46 | 35.59 | 49.52 |
| Llama-3.1-8B (200示范) | 54.34 | 37.97 | 46.16 |
| **IndieValueReasoner (最优)** | **74.74** | **60.60** | **67.67** |

IndieValueReasoner 较零样本 Llama-3.1-8B 提升 **46.6%**，较最优 GPT-4o 配置提升 **34.0%**。仅使用 1.6K 人/题的训练数据，即超越使用 92K 人的统计基线（67.67 vs 66.60）。

### Table: 人口统计 vs 价值陈述训练的影响

| 训练数据 | 人口统计评估 | 陈述评估 | 平均 |
|---|---|---|---|
| 仅人口统计 (400条/题) | 73.74 | 62.84 | 68.14 |
| 仅价值陈述 (800条/题) | 63.81 | 67.42 | 65.62 |
| 混合 (各400条/题) | **73.74** | **68.02** | **70.88** |

混合训练在两种评估场景下均表现最优，说明人口统计与价值陈述信息具有互补性。

## 关键发现

1. **人口统计不能替代个体价值**：仅提供人口统计信息时 GPT-4o 达 60.31%，而 50 条价值陈述即达 60.59%；随着价值陈述增多，准确率持续提升，而人口统计的边际收益微乎其微
2. **混合示范数量提升泛化**：训练时随机使用 50-200 条示范（而非固定 200 条）反而提升测试表现，因为数量多样性带来更强的泛化能力
3. **区域训练数据影响显著**：各洲模型通常对对应洲的测试数据表现最优；全球模型始终匹配或超越区域模型
4. **大洋洲例外**：由于数据全部来自新西兰，Shannon 均匀度指标最低，训练数据的同质性限制了模型捕捉多样价值模式的能力
5. **训练显著改善公平性**：IndieValueReasoner 的 $\sigma_{\text{Inequity}}$ 从零样本的 2.97 降至 2.22，尤其在原本表现最差的南美洲获得最大提升（+18.24%）

## 亮点与洞察

- **范式创新**：将多元对齐从"分组对齐"推向"个体对齐"，避免了预定义分桶的刻板印象风险，是一个有理论深度的研究方向
- **数据集设计精妙**：首次将 WVS 的个体级序列数据用于 LM 评估，统一化转换使之可直接用于 NLP 任务，组合空间之大保证了任务的挑战性
- **$\sigma_{\text{Inequity}}$ 指标**：提供了一个可量化的公平性度量，揭示了准确率相近的模型在公平性上的巨大差异，填补了现有评估框架的空白
- **反直觉发现**：小模型经过训练后可大幅超越未训练的大模型（Llama-3.1-8B 训练后 67.67 > GPT-4o 零样本 63.5），且仅需 1.6K 人数据即超越使用 92K 人的统计方法
- **对个性化 AI 的启示**：价值陈述比人口统计更能捕捉个体独特性，未来的个性化系统应更多依赖描述性偏好数据而非人口标签

## 局限性

- **数据局限**：WVS 基于静态问卷，问题抽象且缺乏真实交互的复杂性；标准化转换可能过度简化了价值表达的微妙之处
- **提示单一**：由于计算资源限制，仅使用单一精心设计的提示进行评估，未探索不同提示设计对结果的影响
- **训练规模受限**：每题仅使用 200-1600 个个体的数据进行训练，更大规模数据是否能带来进一步提升尚不明确
- **隐私风险**：个体对齐本质上需要深层个人价值数据，可能引发隐私侵犯和信息滥用问题
- **偏见传递**：虽然避免了分桶带来的刻板印象，但模型仍可能产生确认偏差、锚定偏差和框架效应等新型偏见
- **应用差距**：即使准确建模了个体价值，如何将其合理地应用于系统行为仍是非平凡问题

## 相关工作与启发

- **多元对齐**（Sorensen et al. 2024; Feng et al. 2024）：强调 AI 应服务于多样化人群，但多依赖预设维度进行分组 → 本文将其推向极致的个体层面
- **个性化 LM**（Han et al. 2024; Zhu et al. 2024）：针对特定应用（摘要、聊天、电影标注）做用户定制 → 本文聚焦更基础的价值体系建模，而非特定任务
- **文化与价值评估**（Durmus et al. 2024; Rao et al. 2024）：评估 LM 的文化偏差 → 但采用组级分布，无法触及个体差异
- **主动学习偏好获取**（Keswani et al. 2024; Piriyakulkij et al. 2024）：交互式高效获取偏好 → 未来可与个体价值数据采集结合
- **启发**：个体价值推理可能是通向真正"千人千面"AI 的关键能力，既有社会科学价值（理解人类价值动态），也有工程价值（心理健康聊天机器人、个性化推荐）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从多元对齐到个体对齐的范式推进有明确贡献，$\sigma_{\text{Inequity}}$ 指标有独立价值
- 实验充分度: ⭐⭐⭐⭐ — 10 个前沿 LM 零样本评估 + 系统消融训练实验 + 跨区域分析，但缺少下游应用验证
- 写作质量: ⭐⭐⭐⭐ — 动机论述充分，公式严谨，但符号体系略重，部分内容可更简洁
- 价值: ⭐⭐⭐⭐ — 对个性化AI和公平性评估有启发意义，数据集可复用性强，但距离实际应用仍有距离

<!-- RELATED:START -->

## 相关论文

- [Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)
- [Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'](can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)

<!-- RELATED:END -->
