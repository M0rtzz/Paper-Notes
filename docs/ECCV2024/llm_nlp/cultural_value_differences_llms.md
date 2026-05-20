---
title: >-
  [论文解读] Cultural Value Differences of LLMs: Prompt, Language, and Model Size
description: >-
  [ECCV 2024][LLM/NLP][文化价值观] 本文使用 Hofstede 文化维度问卷系统性地研究 LLM 表达文化价值观的行为模式，发现提示语言（中文 vs 英文）和模型规模对文化价值差异的影响远大于模型架构差异和问题顺序变化。
tags:
  - "ECCV 2024"
  - "LLM/NLP"
  - "文化价值观"
  - "大语言模型"
  - "Hofstede"
  - "多语言偏差"
  - "模型行为分析"
---

# Cultural Value Differences of LLMs: Prompt, Language, and Model Size

**会议**: ECCV 2024  
**arXiv**: [2407.16891](https://arxiv.org/abs/2407.16891)  
**代码**: 无  
**领域**: LLM / NLP / AI安全  
**关键词**: 文化价值观, 大语言模型, Hofstede, 多语言偏差, 模型行为分析

## 一句话总结

本文使用 Hofstede 文化维度问卷系统性地研究 LLM 表达文化价值观的行为模式，发现提示语言（中文 vs 英文）和模型规模对文化价值差异的影响远大于模型架构差异和问题顺序变化。

## 研究背景与动机

**领域现状**：LLM 在生成类人文本的同时，也内化了训练语料中的文化偏见和价值观。已有研究使用政治倾向测试、世界价值观调查等人类社会科学工具来评估 LLM 的价值取向。

**现有痛点**：(1) 大多数研究仅用英文评估，忽略了语言对价值表达的影响（Sapir-Whorf 假说）；(2) 缺乏对 prompt 变体（如选项顺序、身份模拟）影响的系统性研究；(3) 不同规模同系列模型间的文化价值差异未被充分探索。

**核心矛盾**：LLM 的文化价值表达到底是由训练数据决定的固有特性，还是可以被提示工程轻易操控的表面现象？

**本文目标**：系统性地识别影响 LLM 文化价值表达的关键因素——提示变体、语言、模型规模。

**切入角度**：使用 Hofstede VSM2013 问卷作为标准化工具，对 6 个 LLM 进行 54 种身份模拟 × 多语言 × 选项打乱的全面实验。

**核心 idea**：通过控制变量实验，揭示语言和模型规模是 LLM 文化价值差异的主要驱动因素，而非上下文中的身份信息。

## 方法详解

### 整体框架

使用 VSM2013 问卷（24 道题，6 个文化维度：权力距离 PDI、个人主义 IDV、不确定性规避 UAI、男性化 MAS、长期导向 LTO、放纵 IVR）。每个实验集由 LLM × 语言 × 选项顺序 三元组定义，每个集内测试 54 种模拟身份（9 国籍 × 2 性别 × 3 年龄），每题重复 10 次。

### 关键设计

1. **多维度实验设计**:

    - 功能：全面控制变量，识别各因素的独立影响
    - 核心思路：固定模型和语言变化选项顺序（RQ1），固定模型变化语言（RQ2），固定语言变化模型（RQ3）。6 个模型（Llama2-7B/13B/70B、Qwen-14B/72B、Mixtral-8x7B），2 种语言（中英），2 种选项顺序
    - 设计动机：社会科学中的标准控制变量方法论，确保结论的因果可解释性

2. **三层评估指标体系**:

    - 功能：从不同粒度量化文化价值差异
    - 核心思路：(1) VSM 原始分数的 Pearson 相关系数衡量集内一致性；(2) 标准差 $\sigma_m(v_i)$ 衡量国家间文化差异；(3) 提出 Model Cultural Disparity (MCD) = $D_m/D_h$，将模型表现的国家间差异与人类数据归一化对比。集间比较用 DBI、Silhouette Score 和新提出的 $SS_h$（以人类为参考的轮廓系数）
    - 设计动机：单一指标无法全面反映文化价值的多层次特征，需要集内/集间两个视角

3. **人类参照的轮廓系数 $SS_h$**:

    - 功能：以人类的文化差异为基准，衡量模型集间差异的"绝对大小"
    - 核心思路：标准 Silhouette Score 的分母用人类国家间平均距离 $a_h(n_i)$ 替代集内距离 $a(n_i)$，使得 $SS_h > 1$ 表示模型集间差异超过人类国家间差异
    - 设计动机：标准聚类指标只衡量相对分离度，$SS_h$ 提供以人类为锚点的绝对尺度

### 损失函数 / 训练策略

纯推理评估工作，不涉及训练。所有模型使用默认参数进行推理，每个问题重复 10 次取平均。

## 实验关键数据

### 主实验

| 因素 | 影响程度 | 证据 |
|------|---------|------|
| 提示语言 | **最大** | 同一模型中英文的 $SS_h$ 普遍 >1，远超选项打乱的影响 |
| 模型规模 | **显著** | Llama2 7B vs 70B 的文化差异 > Llama2 vs Mixtral 的差异 |
| 选项打乱 | 中等 | Mixtral $SS_h$=0.680 最敏感，Llama2-13B $SS_h$=0.228 最稳定 |
| 模拟身份 | **最小** | 54 种身份的 MCD 远小于 1，模型不会因模拟国籍而改变价值 |

### 消融实验

| 模型 | 英文 w/o shuffle vs w/ shuffle DBI↓ | 中文 vs 英文 $SS_h$↑ |
|------|-------------------------------------|---------------------|
| Llama2-7B | 1.837 | 1.52 |
| Llama2-70B | 0.658 | 0.87 |
| Qwen-14B | 0.981 | 1.33 |
| Mixtral-8x7B | 0.542 | 0.68 |

### 关键发现

- **语言是最关键因素**：同一模型用中文和英文提问时，在 PDI、IDV 等维度上表现出截然不同的文化价值倾向，差异甚至超过人类不同国家间的差异
- **模型规模 > 模型架构**：Llama2-7B 和 Llama2-70B 的价值差异大于 Llama2 和 Mixtral 的差异
- **模型不受身份模拟影响**：无论模拟日本人、美国人还是中国人，同一模型在同一语言下的回答几乎一致（MCD << 1）
- **选项位置偏差存在但可控**：大模型（70B+）对选项打乱更鲁棒

## 亮点与洞察

- **系统性实验设计**：54 身份 × 10 重复 × 多语言 × 多模型的大规模控制实验，总共 12960 × N 个回答，数据充分
- **$SS_h$ 指标的设计**：用人类文化差异做归一化基准，使得不同实验集间的比较有绝对尺度参照。可迁移到其他 LLM 行为评估场景
- **"语言决定文化"的发现**：与 Sapir-Whorf 假说一致，对多语言 LLM 的部署和安全审计有重要启示——不能只用英文评估模型的价值取向

## 局限与展望

- 仅测试 6 个模型、2 种语言，更多语言（阿拉伯语、日语等）和模型的验证不足
- VSM 问卷本身的心理学效度受到批评，可能不完全适用于 LLM
- 未分析训练数据组成与文化价值的因果关系
- 健康相关问题（Q15、Q18）直接赋中性值处理过于简单
- 可进一步研究 RLHF/DPO 对齐如何改变模型的文化价值表达

## 相关工作与启发

- **vs Arora et al.**: 使用 VSM + WVS 探索多语言 MLM 的文化价值，但仅用掩码模型且未控制模型规模；本文扩展到生成式 LLM 并增加规模维度
- **vs Kovač et al.**: 用心理问卷评估 LLM"人格"，发现上下文依赖性；本文进一步证明语言比上下文影响更大
- **vs Feng et al.**: 用政治倾向测试评估偏见，仅英文单模型；本文多维度多模型更全面

## 评分

- 新颖性: ⭐⭐⭐ 实验设计有新意但研究框架（用人类问卷评LLM）不算新
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模系统性实验，控制变量严格
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但数学符号偏多增加阅读负担
- 价值: ⭐⭐⭐⭐ 对多语言 LLM 部署和安全评估有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Convert Language Model into a Value-based Strategic Planner](../../ACL2025/llm_nlp/convert_language_model_into_a_value-based_strategic_planner.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](../../ACL2025/llm_nlp/unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ECCV 2024\] Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos](propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)
- [\[ACL 2025\] Cultural Learning-Based Culture Adaptation of Language Models](../../ACL2025/llm_nlp/cultural_learning-based_culture_adaptation_of_language_models.md)
- [\[ECCV 2024\] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)

</div>

<!-- RELATED:END -->
