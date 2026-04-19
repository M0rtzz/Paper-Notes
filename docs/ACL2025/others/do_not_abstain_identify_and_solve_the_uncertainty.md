---
title: >-
  [论文解读] Do not Abstain! Identify and Solve the Uncertainty
description: >-
  [ACL 2025][不确定性识别] 本文提出ConfuseBench基准和基于inquiry answer唯一性判断不确定性来源的方法，并通过InteractDPO在策略训练中动态生成偏好对来提升inquiry质量，使LLM能主动识别并解决不确定性而非简单回避。
tags:
  - ACL 2025
  - 不确定性识别
  - ConfuseBench
  - InteractDPO
  - 不确定性分类
  - 检索增强
---

# Do not Abstain! Identify and Solve the Uncertainty

**会议**: ACL 2025  
**arXiv**: [2506.00780](https://arxiv.org/abs/2506.00780)  
**代码**: [GitHub](https://github.com/somebodyhh1/ConfuseBench)  
**领域**: LLM 不确定性 / 交互式问答  
**关键词**: 不确定性识别, ConfuseBench, InteractDPO, 不确定性分类, 检索增强  

## 一句话总结

本文提出ConfuseBench基准和基于inquiry answer唯一性判断不确定性来源的方法，并通过InteractDPO在策略训练中动态生成偏好对来提升inquiry质量，使LLM能主动识别并解决不确定性而非简单回避。

## 研究背景与动机

**LLM过度自信问题**：大语言模型在面对不确定场景时常表现出过度自信，生成幻觉式回答，而现有解决方案主要采取保守策略——面对不确定性时简单回答"我不知道"。

**回避策略的局限性**：对于本质不可知的问题（如"2050年的天气"），回避是正确做法；但对于可解答但存在不确定性的问题，简单回避忽略了通过检索、推理链或澄清来解决不确定性的机会。例如，当模型面对"量子计算对气候建模的影响"这类问题信心不足时，应主动识别不确定性来源并采取相应策略。

**三类不确定性来源**：
   - **文档稀缺（Document Scarcity）**：模型缺少回答所需的事实信息，可通过检索补充
   - **能力不足（Limited Capacity）**：查询对模型来说太复杂，可通过CoT或更强模型解决
   - **查询歧义（Query Ambiguity）**：查询本身不清晰，需要用户进一步澄清

**现有研究的不足**：已有工作只关注单一不确定性来源（如只做迭代检索，或只做澄清），未能综合考虑不同来源并采取对应措施。

## 方法详解

### 整体框架

方法分为两个核心阶段：（1）基于inquiry answer的不确定性来源判断；（2）通过InteractDPO提升inquiry生成质量。

### 基准构建：ConfuseBench

| 数据集 | 文档稀缺 | 查询歧义 | 能力不足 |
|--------|----------|----------|----------|
| HotpotQA | 859 | 702 | 141 |
| AmbigQA | 543 | 537 | 167 |
| ExpertQA | 442 | 397 | 141 |
| TechQA | 470 | 683 | 140 |
| ToolBench | 479 | 590 | 144 |

- 覆盖三大LLM应用场景：基础问答、助手交互和工具使用
- 采用GPT-4o结合AMR（抽象语义表示）图来系统地引入歧义
- 基准测试集共650条（每数据集50+50+30条）
- 评估指标：Answer Quality (AQ)、Uncertainty Classification Accuracy (UCA)、Inquiry Quality (IQ)

### 关键设计1：基于Inquiry的不确定性判断

核心思路：不直接判断不确定性来源，而是先让模型生成follow-up inquiry，再通过inquiry的回答特征来判断不确定性类型。

**理论基础**（定理5.2）：inquiry持有的不确定性与原始query正相关，即 $|U_k(q) - U_k(x)| \leq -\log p(q^*|x,c,\theta)$，其中 $U_k$ 为知识不确定性。

**判断规则**：
- inquiry的回答指向**唯一客观事实** → 不确定性来源是**文档稀缺**，需检索
- inquiry的回答可以有**多个合理答案** → 不确定性来源是**查询歧义**，需澄清
- inquiry只是原问题的**简单复述**或逻辑不连贯 → 不确定性来源是**能力不足**，需CoT

**验证方法**：给模型一个预设答案，若模型只能重复该答案则说明答案唯一（客观事实），若能轻松生成替代答案则说明问题开放（需澄清）。

### 关键设计2：InteractDPO

传统DPO使用离线偏好数据，InteractDPO则在训练中动态生成偏好对：

1. 模型根据prompt生成inquiry
2. 通过与检索系统或模拟用户的**实时交互**获取文档或澄清
3. 模型基于交互结果生成回答
4. 若成功解决原始query → inquiry标记为"chosen"
5. 若未能解决 → inquiry标记为"rejected"

与OnlineDPO的区别：OnlineDPO让LLM判断哪个inquiry更好（缺乏真实监督信号），而InteractDPO通过实际交互获得真实反馈。

## 实验关键数据

### 初步测试结果（UCA准确率）

| 模型 | HotpotQA | AmbigQA | TechQA | ExpertQA | ToolBench | 平均 |
|------|----------|---------|--------|----------|-----------|------|
| GPT-4o | 0.531 | 0.377 | 0.477 | 0.400 | 0.685 | 0.494 |
| DeepSeek-V3 | 0.462 | 0.431 | 0.400 | 0.438 | 0.562 | 0.459 |
| Qwen2.5-72B | 0.631 | 0.592 | 0.431 | 0.408 | 0.700 | 0.552 |
| Qwen2.5-7B | 0.431 | 0.454 | 0.415 | 0.408 | 0.415 | 0.425 |

最好的模型也只有约55%的分类准确率，说明不确定性来源识别确实困难。

### 方法效果对比（UCA平均准确率）

| 方法 | GPT-4o | DeepSeek-V3 | Qwen2.5-72B | Llama-3-70B | Qwen2.5-7B | Mistral-7B |
|------|--------|-------------|-------------|-------------|------------|------------|
| prompt直接判断 | 0.494 | 0.459 | 0.552 | 0.408 | 0.425 | 0.454 |
| 基于inquiry判断 | 0.569 | 0.537 | 0.577 | 0.537 | 0.477 | 0.529 |
| 基于answer判断 | **0.606** | **0.554** | **0.603** | **0.548** | **0.500** | **0.548** |

所有模型在使用answer判断后都获得显著提升，平均提升约10个百分点。

### InteractDPO效果

| 训练方法 | 平均UCA |
|----------|---------|
| GPT-4o (上限) | 0.606 |
| 无训练 (vanilla) | 0.543 |
| SFT | 0.574 |
| DPO | 0.585 |
| OnlineDPO | 0.592 |
| **InteractDPO** | **0.606** |

InteractDPO使Qwen2.5-7B的性能达到GPT-4o水平。

### 关键发现

1. **模型偏向归因于歧义**：所有模型（特别是弱模型）倾向于将不确定性归因为查询歧义，ambig召回率高达85-97%，而doc召回率仅10-19%
2. **模型不承认能力不足**：类似于过度自信，模型在面对不确定性时宁可怪罪外部因素也不承认自身推理能力有限
3. **噪声文档的干扰**：当提供清晰query和噪声文档时，模型可能被噪声文档分散注意力，试图让用户修改query以匹配噪声文档

## 亮点与洞察

1. **范式创新**：从"LLM应否回答"转变为"LLM应如何解决不确定性"，提出了更积极的不确定性处理范式
2. **理论贡献**：证明了inquiry的不确定性与原query不确定性的正相关关系，为间接判断提供了理论支撑
3. **实用的判断流程**：利用inquiry answer的唯一性来区分检索需求和澄清需求，设计巧妙且直觉合理
4. **InteractDPO的在线反馈**：通过真实交互获取训练信号，比依赖LLM自判断更可靠

## 局限性

1. **不确定性类型有限**：仅考虑三类最常见的不确定性，实际中还有更细粒度的分类（如事实知识缺失 vs 背景知识缺失、歧义 vs 事实错误 vs 非法时间等）
2. **能力不足的解决方案单一**：仅使用CoT，而实际中可能需要Tree of Thought或MCTS等更高级推理方法
3. **依赖GPT-4o**：基准构建和评估依赖GPT-4o，可能引入系统性偏差

## 相关工作

- **不确定性识别**：Amayuelas et al. (2024) 提出模型应理解自己不知道什么；Deng et al. (2024) 训练模型给出不可回答的解释
- **不确定性解决**：Trivedi et al. (2022) 通过迭代推理和检索解决多跳查询；Qian et al. (2024) 构建IN3评估澄清问题生成能力
- **不确定性分解**：Yadkori et al. (2024) 通过分布偏移分解不确定性；本文的answer验证方法受其启发

## 评分

⭐⭐⭐⭐ — 提出了有价值的新问题和系统性的解决方案，ConfuseBench填补了不确定性来源识别评估的空白，InteractDPO是解决在线偏好学习的有效方法。不确定性类型覆盖可进一步丰富。

<!-- RELATED:START -->

## 相关论文

- [When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)
- [LoGU: Long-form Generation with Uncertainty Expressions](logu_longform_gen_uncertainty.md)
- [All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)

<!-- RELATED:END -->
