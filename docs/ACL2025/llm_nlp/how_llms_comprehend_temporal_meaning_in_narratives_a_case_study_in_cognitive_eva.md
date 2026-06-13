---
title: >-
  [论文解读] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs
description: >-
  [ACL 2025][LLM 其他][时态理解] 构建 Expert-in-the-Loop 探测管线，通过真值判断/词语补全/开放式因果提问三组认知语言学实验（16篇叙事×30种prompt变体×7个LLM），系统评估LLM对叙事中语法体貌（perfective vs imperfective）的理解能力，发现LLM在非原型体貌条件下准确率仅18%（人类71%），且缺乏远距因果推理能力。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "时态理解"
  - "语法体貌"
  - "认知评估"
  - "叙事理解"
  - "因果推理"
---

# How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs

**会议**: ACL 2025  
**arXiv**: [2507.14307](https://arxiv.org/abs/2507.14307)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 时态理解, 语法体貌, 认知评估, 叙事理解, 因果推理

## 一句话总结

构建 Expert-in-the-Loop 探测管线，通过真值判断/词语补全/开放式因果提问三组认知语言学实验（16篇叙事×30种prompt变体×7个LLM），系统评估LLM对叙事中语法体貌（perfective vs imperfective）的理解能力，发现LLM在非原型体貌条件下准确率仅18%（人类71%），且缺乏远距因果推理能力。

## 研究背景与动机

- **领域现状**: LLM展现出越来越强的语言能力，但这些行为究竟反映了类人认知理解还是高级统计模式匹配，仍是开放问题。近年兴起一批用认知科学实验范式评估LLM的研究（如Fan Effect、决策偏差、心智理论等），但对语言时态语义的认知评估仍属空白。
- **现有痛点**: 现有LLM评估多依赖NLP benchmark，缺乏与人类认知过程的精细对比；少数用人类实验的工作通常只用单一指标，且未充分控制prompt变异导致的伪发现。
- **核心矛盾**: LLM能流利阐述语法体貌（aspect）的学术定义（声明式知识），却在需要隐式应用体貌语义进行判断时严重失败——这种"知道规则但不会用"的分离，与二语学习者中级阶段的表现高度相似。
- **本文目标**: LLM能否像人类一样，利用语法体貌（完成体 vs 未完成体）的语义差异来构建叙事情境模型、保持工作记忆焦点、并进行远距因果推理？
- **切入角度**: 选择语言学中的"体貌"（aspect）作为探针——完成体（"washed"）表示事件已完成，未完成体（"was washing"）表示事件正在进行。在叙事中，未完成体使事件保持"开放状态"，更容易被保留在工作记忆中并与后续事件建立因果关联。这一机制在人类认知中已有大量实证，为LLM评估提供了理想的对照基准。
- **核心 idea**: 复用认知语言学已验证的叙事体貌实验材料，构建30种prompt变体的鲁棒测试管线，从语义→认知加工→语用三层递进地评估LLM与人类在体貌理解上的差距。

## 方法详解

### 整体框架

Expert-in-the-Loop探测管线：以认知语言学家参与的迭代式评估框架为核心，将人类实验材料（Schramm 1998的16篇叙事）转化为LLM可用的prompt，经过系统的prompt扰动（30种变体）后，在7个LLM上运行三组递进实验（真值判断→词语补全→开放式因果推理），最后用多层混合效应模型（lme4）做统计分析，与已发表的人类数据直接对比。

### 关键设计

**1. 叙事刺激材料与体貌操控**

- **功能**: 提供受控的实验材料，仅通过改变一个动词的语法体貌来操控整个叙事的因果推理方向
- **核心思路**: 每篇叙事包含两个潜在原因事件（C1、C2）和一个惊喜结果（Effect）。C1使用"完成类事件"（Accomplishment，如"wash the dishes"），其动词在完成体（"washed"）和未完成体（"was washing"）之间切换。当C1为未完成体时，事件保持开放状态，人类更倾向于将C1推断为Effect的原因；当C1为完成体时，事件被视为已结束，C2成为更可能的原因。共16篇叙事×2个版本=32个刺激。
- **设计动机**: 直接复用已在人类被试上验证过的语言学实验材料，确保与人类数据的严格可比性，同时利用Accomplishment+imperfective这一非原型搭配来测试LLM是否真正理解体貌语义而非仅依赖高频共现模式

**2. Prompt扰动与鲁棒性控制**

- **功能**: 消除prompt措辞对LLM响应的系统性影响，确保实验结论的可靠性
- **核心思路**: 对每个prompt进行两个维度的受控变异——(a) 通用指令改写：按话语句法变化和语义变化两类策略，生成3个指令版本；(b) 数据格式变异：基于FormatSpread协议引入10种格式变体（改变空格、大小写、排序、标点）。每个实验条件产生3×10=30种prompt变体，模型在所有变体上运行后取聚合结果。
- **设计动机**: Sclar et al. (2024)和Wahle et al. (2024)已证明prompt格式和措辞会显著影响LLM响应，单一prompt的结论可能是伪发现；30种变体提供了足够的统计覆盖

**3. 三层递进探测实验体系**

- **功能**: 从语义理解→认知加工→语用推理三个层面全面评估LLM的体貌处理能力
- **核心思路**: (a) **实验1-真值判断**：给LLM展示叙事后，要求判断事件最终状态的真假（如"Lena was running downstairs"→"Lena downstairs"是True/False），测试语义层面是否理解未完成体不蕴含事件完成；(b) **实验2-词语补全**：在叙事中插入部分字母（如"D I _ _ _ _"），要求LLM补全单词，通过匹配C1关键词的频率衡量工作记忆中的概念激活程度，并设置Near C1和Near Effect两个探测位置测试远距保持能力；(c) **实验3-开放式因果提问**：读完叙事后回答"What caused the effect?"，测量LLM选择C1为原因的频率，直接对标人类因果推理数据。
- **设计动机**: 单一指标无法全面反映认知过程；三个实验分别对应人类认知中的语义表征、工作记忆编码、情景记忆中的因果模型，形成收敛证据

## 实验关键数据

### 主实验

**实验1a——真值判断准确率（LLM vs 人类）**:

| 体貌 | 极性 | 正确答案 | LLM准确率 | 人类准确率 |
|------|------|---------|----------|----------|
| 完成体 | 正面 | True | 88% | 88% |
| 未完成体 | 负面 | True | **18%** | **71%** |
| 完成体 | 负面 | False | 89% | 93% |
| 未完成体 | 正面 | False | **35%** | **61%** |

统计检验：体貌主效应显著 F=66.5, p<.01；极性主效应显著 F=10363, p<.01；交互效应显著 F=661.5, p<.01。

**实验3——开放式因果推理（选择C1为原因的比率）**:

| 模型 | 未完成体条件 | 完成体条件 | 与人类差距 |
|------|------------|-----------|----------|
| 人类 | ~68% | ~33% | — |
| Qwen2-72B | ~60% | ~10% | 较小 |
| Llama3.1-70B | ~55% | ~8% | 中等 |
| GPT-4o | ~50% | ~12% | 中等 |
| 小模型（<10B平均） | ~35% | ~8% | 较大 |

体貌主效应显著 F=98.5, p<.01。

### 消融实验

| 对比条件 | 关键指标 | 结论 |
|---------|---------|------|
| 实验1b：有叙事上下文 vs 无上下文 | 去除叙事后未完成体准确率仍低 | 困难源于体貌处理本身而非叙事结构干扰；不同模型变化方向不一致 |
| 实验2：Near C1 vs Near Effect | 目标词匹配频率平均下降33% | LLM缺乏远距工作记忆保持，无法像人类将未完成体事件保持在焦点 |
| 大模型(70B+) vs 小模型(<10B) | 因果推理更接近人类 | 仅在实验3观察到scaling效果，实验1、2中大小模型无显著差异 |
| 模型家族对比（Gemma/Llama/Qwen/GPT-4o） | 家族间差异显著但方向不一致 | 没有任何模型在所有实验中维持人类水平表现 |

### 关键发现

1. **原型条件正常，非原型条件严重失败**: 完成体（原型搭配）的准确率88-89%与人类持平，但未完成体（非原型搭配）的准确率仅18%，远低于人类的71%
2. **过度依赖原型性**: 完成体+Accomplishment事件是叙事中最常见的搭配，LLM倾向于默认事件已完成，体现分布式表征而非意义驱动的理解
3. **远距因果推理缺失**: 词语补全从Near C1到Near Effect的激活下降33%，LLM无法在工作记忆中保持未完成体事件的焦点
4. **声明式知识vs隐式理解分离**: LLM能准确阐述体貌定义，但无法在实际判断中应用——类似二语学习者"能背规则但不会用"
5. **人类叙事解读更灵活**: 完成体条件下，人类有约1/3仍选C1为原因，LLM几乎不会——LLM缺乏人类那种追求叙事连贯性的灵活建模能力

## 亮点与洞察

- **研究设计精巧**: 直接复用认知语言学已验证的实验材料（Schramm 1998），确保与人类数据的严格可比性，而非另造benchmark
- **方法论贡献突出**: Expert-in-the-Loop管线+30种prompt变体的鲁棒性控制，是LLM认知评估的范式性方法，可直接迁移到其他认知域
- **三层递进设计形成收敛证据**: 语义（真值判断）→认知加工（词语补全/工作记忆）→语用（因果推理），逐层揭示LLM体貌处理的具体失败环节
- **"声明式vs隐式"洞察深刻**: LLM的表现与二语习得领域的L2学习者中级阶段高度一致（Salaberry 2024），暗示LLM对体貌的表征本质上是分布式的而非概念性的
- **统计分析严谨**: 使用多层混合效应模型控制叙事随机效应，Bonferroni校正多重比较，α=.01

## 局限与展望

1. 依赖LLM的"自我报告"式响应，无法直接观察内部处理机制；可结合attention分析或probing classifier获取隐式信号
2. 16篇叙事的规模较小，某些条件组合的统计power可能不足；扩展材料库可增强结论可靠性
3. 仅测试英语叙事中的体貌现象，其他语言（如中文、西班牙语）的体貌系统差异很大，跨语言泛化性未知
4. 未评估最新的reasoning模型（如o1、Claude 3.5）和chain-of-thought提示下的表现，可能低估了LLM的潜力
5. 实验3的开放式回答用GPT-4o自动标注（Cohen's κ=.93），虽然可靠性高但仍引入了LLM评估LLM的循环依赖

## 相关工作与启发

- **LLM认知评估框架**: 与Ivanova (2025)提出的评估原则一致，强调用多重指标和人类对照数据；Roberts et al. (2024)研究Fan Effect用token概率作为记忆检索难度的代理，本文则用多种行为指标研究更微妙的体貌效应
- **LLM语用理解**: Beuls & Van Eecke (2024)和Sravanthi et al. (2024)均指出LLM在语用推理上的不足，本文提供了叙事体貌这一新维度的证据
- **认知偏差与模式依赖**: 与Hagendorff et al. (2023)发现的LLM决策偏差一致——LLM倾向于依赖高频模式而非灵活推理
- **启示**: 需要在预训练或对齐中引入更丰富的时间/因果推理信号；区分LLM的声明式知识和隐式理解能力是评估LLM认知的关键方法论原则

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将认知语言学体貌实验精确移植到LLM评估，角度在NLP社区罕见
- 实验充分度: ⭐⭐⭐⭐⭐ 三组实验+30种prompt变体+7个模型+混合效应统计分析，极为严谨
- 写作质量: ⭐⭐⭐⭐⭐ 语言学背景介绍清晰，实验递进逻辑流畅，跨学科读者友好
- 价值: ⭐⭐⭐⭐ 对LLM认知能力边界提供有力证据，方法论可复用性高，但直接技术改进方向有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)

</div>

<!-- RELATED:END -->
