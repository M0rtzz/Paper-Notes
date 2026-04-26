---
title: >-
  [论文解读] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions
description: >-
  [ACL 2025][LLM/NLP][对话基础] 本文研究LLM在政治领域中处理直接知识问题和暗含错误预设的"loaded question"的能力，评估LLM是否能进行主动的对话基础（conversational grounding）来纠正用户的虚假信念，发现LLM在拒绝虚假预设和维护知识准确性方面存在重大缺陷。
tags:
  - ACL 2025
  - LLM/NLP
  - 对话基础
  - 预设性问题
  - 政治偏见
  - 知识边界
  - 错误信息
---

# Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions

**会议**: ACL 2025  
**arXiv**: N/A  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-long.728/)
**代码**: 无  
**领域**: LLM分析 / 对话Grounding / 政治偏见  
**关键词**: 对话基础, 预设性问题, 政治偏见, 知识边界, 错误信息

## 一句话总结

本文研究LLM在政治领域中处理直接知识问题和暗含错误预设的"loaded question"的能力，评估LLM是否能进行主动的对话基础（conversational grounding）来纠正用户的虚假信念，发现LLM在拒绝虚假预设和维护知识准确性方面存在重大缺陷。

## 研究背景与动机

**领域现状**：人类交流依赖于"对话基础"（conversational grounding）——交互双方不断校准彼此的理解，确保信息传递的准确性。在LLM广泛用于信息获取的今天，LLM能否像人类一样进行对话基础——特别是在它们"不确定"或"不知道"时——成为一个关键问题。

**现有痛点**：(1) 现有LLM在面对错误前提的问题时（loaded questions，如"特朗普在2020年赢得了大选吗？"），往往不能有效拒绝虚假预设，甚至可能顺着错误前提回答；(2) LLM的政治偏见可能影响其对不同政治派别相关事实的处理方式；(3) 缺乏系统性研究来评估LLM在政治领域的grounding能力。

**核心矛盾**：用户可能带着错误信念提问，一个负责任的AI系统应该识别并纠正这些错误信念（即进行grounding），而不是迎合用户的预设。但LLM天然倾向于生成"令用户满意"的回答，这与纠正错误的目标矛盾。

**本文目标**：(1) 评估LLM回答直接政治知识问题的准确性；(2) 测试LLM面对loaded questions（预设错误信息的问题）时能否进行主动grounding；(3) 分析LLM的知识水平和政治偏见如何影响其grounding行为。

**切入角度**：设计直接问题（Direct Questions）和预设错误信息的问题（Loaded Questions）配对实验，对比LLM在两种问题类型下的行为差异。

**核心 idea**：通过对比LLM对直接问题和loaded questions的回答来评估其对话grounding能力，揭示LLM在高风险政治信息场景中的可靠性问题。

## 方法详解

### 整体框架

实验设计分为三步：(1) 构建政治知识问题集（覆盖多国政治事实）；(2) 为每个直接问题创建对应的loaded question（植入虚假预设）；(3) 让多个LLM分别回答两种问题，分析回答的准确性和grounding行为。

### 关键设计

1. **直接问题与Loaded Question配对设计**:

    - 功能：系统评估LLM在标准知识查询和包含错误预设的查询中的行为差异
    - 核心思路：构建政治知识问题集（如"谁赢得了2020年美国总统选举？"），然后为每个问题创建包含错误事实预设的变体（如"拜登在2020年选举中舞弊后，最高法院是否采取了行动？"——预设了虚假的选举舞弊声称）。直接问题测试知识准确性，loaded questions测试grounding能力，配对设计允许精确对比
    - 设计动机：仅测试直接问答不够——真实场景中用户常带着偏见提问，LLM能否识别并纠正这些偏见是更关键的能力

2. **Grounding行为分类体系**:

    - 功能：对LLM的回答进行细粒度分类
    - 核心思路：将LLM对loaded questions的回应分为四类：(a) **主动grounding**——明确指出问题中的错误预设并纠正；(b) **部分grounding**——回答中提到了正确信息但未明确拒绝错误预设；(c) **顺应错误**——接受错误预设并基于此回答；(d) **回避**——拒绝回答但不纠正错误。通过人工标注对LLM的每个回答进行分类
    - 设计动机：区分不同类型的grounding行为可以揭示LLM失败的具体模式——是"不知道该纠正"还是"知道但不愿纠正"

3. **政治偏见与知识水平交叉分析**:

    - 功能：分析政治立场和知识掌握程度如何影响grounding行为
    - 核心思路：将问题按政治派别（关联左翼/右翼的虚假信息）分组，检查LLM是否对不同派别的虚假预设有不同的纠正倾向。同时比较LLM在直接问题上回答正确（说明"知道"正确答案）vs 回答错误（说明"不知道"）的情况下，面对loaded questions的grounding行为有何不同
    - 设计动机：如果LLM"知道"正确答案但仍然不纠正loaded questions中的错误预设，这暴露了alignment层面的问题而非知识层面的问题

### 评估设置

测试了GPT-4、Claude、LLaMA等主流LLM，问题覆盖美国、欧洲多国的政治事实。

## 实验关键数据

### 主实验

| 指标 | GPT-4 | Claude | LLaMA-3 | 中位水平 |
|------|-------|--------|---------|---------|
| 直接问题准确率 | 高 (~85%+) | 高 | 中等 | ~75% |
| 主动grounding率（loaded Q） | ~40-50% | ~35-45% | ~25-35% | ~35% |
| 顺应错误率（loaded Q） | ~15-25% | ~20-30% | ~30-40% | ~25% |
| 部分grounding率 | ~20-30% | ~20-25% | ~15-20% | ~20% |

### 消融实验

| 条件 | 主动grounding率 | 说明 |
|------|---------------|------|
| LLM "知道"正确答案 | ~50-60% | 知道但仍有40%+不纠正 |
| LLM "不知道"正确答案 | ~15-20% | 不知道时更容易顺应错误 |
| 左翼相关虚假信息 | ~45% | grounding相对积极 |
| 右翼相关虚假信息 | ~35% | grounding不够积极，暗示偏见 |
| 高争议性话题 | ~25% | 争议越大grounding越弱 |
| 低争议性话题 | ~55% | 明确事实更容易纠正 |

### 关键发现
- 即使LLM"知道"正确答案（直接问题回答正确），面对同一知识点的loaded questions仍有约40%+的概率不进行主动grounding，这说明问题不仅在知识层面，更在行为层面
- LLM对不同政治派别的虚假信息表现出不对称的纠正倾向，暗示存在潜在的政治偏见
- 高争议性话题上LLM的grounding能力急剧下降，可能因为RLHF训练让模型学会了"不得罪人"
- 所有测试的LLM在grounding能力上都远低于理想水平，这对其在政治信息场景中的可靠应用构成挑战

## 亮点与洞察
- **将conversational grounding理论引入LLM评估**是一个有洞察力的视角——不只是看"LLM知不知道"，而是看"LLM在面对用户的错误信念时能否主动纠正"
- **Direct vs Loaded配对设计**巧妙地分离了知识因素和行为因素——知道正确答案但不纠正说明了alignment的缺陷
- 发现LLM在政治敏感话题上"避免冲突"的倾向比以往研究更系统地量化了

## 局限与展望
- 问题集主要覆盖英美政治，其他地区（如中东、亚太）的政治知识覆盖有限
- "loaded question"的虚假预设是人工设计的，真实场景中用户的预设可能更微妙
- 未评估对话轮次的影响——如果用户被纠正后坚持错误预设，LLM是否会"让步"
- 改进方向：在RLHF训练中显式增加"纠正错误预设"的奖励信号

## 相关工作与启发
- **vs TruthfulQA**: TruthfulQA评估LLM是否会生成虚假但常见的说法，本文更进一步评估面对用户的虚假预设时的纠正能力
- **vs Political Compass测试**: 现有政治偏见测试主要评估LLM的观点倾向，本文关注偏见对grounding行为的影响
- **vs SimpleQA**: SimpleQA测试直接知识问答准确性，本文的loaded questions维度是重要补充

## 评分
- 新颖性: ⭐⭐⭐⭐ 将对话grounding与LLM知识边界结合的研究视角很独特
- 实验充分度: ⭐⭐⭐⭐ Direct/Loaded配对设计+多维度分析，方法论扎实
- 写作质量: ⭐⭐⭐⭐ 问题驱动的写作风格，分析层层深入
- 价值: ⭐⭐⭐⭐⭐ 对LLM在高风险信息场景中的可靠性提出了重要警示

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)

<!-- RELATED:END -->
