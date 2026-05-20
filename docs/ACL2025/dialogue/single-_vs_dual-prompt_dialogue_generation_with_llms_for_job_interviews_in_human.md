---
title: >-
  [论文解读] Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources
description: >-
  [ACL 2025 (GEM² Workshop)][对话系统][对话生成] 本文系统比较了使用 LLM 生成求职面试对话的两种策略——单提示（一次性生成完整对话）和双提示（两个 agent 分别扮演面试官和候选人轮流对话），发现双提示方法生成的对话在自然度上的胜率是单提示的 2-10 倍…
tags:
  - "ACL 2025 (GEM² Workshop)"
  - "对话系统"
  - "对话生成"
  - "LLM评估"
  - "单提示vs双提示"
  - "合成数据"
  - "人力资源"
---

# Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources

**会议**: ACL 2025 (GEM² Workshop)  
**arXiv**: [2502.18650](https://arxiv.org/abs/2502.18650)  
**代码**: 无  
**领域**: NLP / 对话生成  
**关键词**: 对话生成、LLM评估、单提示vs双提示、合成数据、人力资源

## 一句话总结

本文系统比较了使用 LLM 生成求职面试对话的两种策略——单提示（一次性生成完整对话）和双提示（两个 agent 分别扮演面试官和候选人轮流对话），发现双提示方法生成的对话在自然度上的胜率是单提示的 2-10 倍，但 token 成本增加约 6 倍。

## 研究背景与动机

**领域现状**：训练对话系统需要大量高质量对话数据，但在人力资源（HR）等专业领域，获取真实人类对话数据既昂贵又困难（涉及隐私、合规等问题）。因此，利用 LLM 合成对话成为主流替代方案。目前有两种主要的合成策略：单提示方法让 LLM 一次性生成完整对话，双提示方法让两个 LLM agent 分别扮演对话双方进行多轮交互。

**现有痛点**：尽管两种策略都被广泛使用，但此前缺乏系统性的对比研究来回答一个基本问题：哪种策略生成的对话更像真人对话？成本效益如何？不同 LLM 对结果有何影响？

**核心矛盾**：双提示方法直觉上更自然（因为模拟了真实对话的轮流过程），但需要多次 API 调用且每次都要传入完整历史，成本显著更高。研究者需要量化这种质量差异是否值得额外成本投入。

**本文目标**：（1）严格对比单提示和双提示策略的对话质量；（2）检验结论在不同生成模型（GPT-4o vs Llama 3.3 70B）间是否一致；（3）检验不同评判模型的一致性。

**切入角度**：作者采用 PairEval 框架，让 LLM 评判者对成对面试对话进行 AI 检测——判断哪个对话更可能是 AI 生成的。被判为"更不像 AI"的对话即为质量更高的对话。

**核心 idea**：以"对话真实性"作为质量代理指标，通过 LLM-as-a-judge 进行成对比较，系统量化两种生成策略的质量差异。

## 方法详解

### 整体框架

实验管线包含三个阶段：种子构建 → 对话生成 → 质量评估。首先从真实求职者的职业经历中构建 100 个匿名化的种子摘要，然后用每个种子分别通过 4 种配置（2 种策略 × 2 种 LLM）生成面试对话，最后用 LLM 评判者进行成对比较。

### 关键设计

1. **种子构建与对话生成**:

    - 功能：为对话生成提供真实且多样化的内容基础
    - 核心思路：从大规模职业经历数据集中随机选取 100 条真实简历，用 GPT-4T（temperature=1）生成匿名摘要作为种子。对于**双提示策略**，设置面试官 agent 和候选人 agent 各自的系统提示，面试官被要求发掘候选人的职业能力，候选人根据种子信息回答问题，双方都被要求"通过图灵测试"以模拟人类对话风格。对于**单提示策略**，直接要求 LLM 基于种子生成约 16 轮的完整面试对话
    - 设计动机：使用相同种子确保内容可变因素一致，唯一变量是生成策略和 LLM 选择。Turing test 要求促使模型生成更自然的对话

2. **LLM-as-Judge 成对评估**:

    - 功能：量化对话的真实感和质量
    - 核心思路：用 GPT-4o 和 Llama 3.3 分别作为评判者，对来自同一种子的两个对话进行成对比较。评判者需要判断哪个对话更可能是 AI 生成的，允许判平局。为消除顺序偏差，每对比较执行两次（交换顺序）。胜率计算公式为 $\text{Win Rate}(M_i) = \frac{\#\text{Wins}_{M_i}}{\#\text{Wins}_{M_i} + \#\text{Losses}_{M_i} + \#\text{Ties}_{M_i}}$
    - 设计动机：相比绝对打分，成对比较更稳定可靠。加入"先给理由再给判断"的顺序要求提升评判一致性。明确指示"不要考虑对话长度"来消除长度偏差

3. **长度偏差控制与评判一致性验证**:

    - 功能：确保评估结果的有效性
    - 核心思路：使用有序逻辑回归检验对话长度对评判结果的影响。结果显示对话整体长度的影响较小（且为负向，即长对话反而不被偏好），但单轮话语长度有正向影响。两个评判者之间的一致性通过"放松"标准（将平局视为一致）达到 85%+ 的高一致率
    - 设计动机：LLM 评判者已知存在长度偏差和自我偏好等问题，必须通过实验验证这些偏差是否影响核心结论

### 损失函数 / 训练策略

本文不涉及模型训练，仅评估不同生成策略的对话质量。

## 实验关键数据

### 主实验

GPT-4o 作为评判者时的胜率：

| 生成配置 | Dual 策略 | Single 策略 | 合计 |
|----------|----------|------------|------|
| GPT-4o 生成 | **0.49** | 0.18 | 0.36 |
| Llama 3.3 生成 | **0.62** | 0.09 | 0.33 |
| 合计 | **0.71** | 0.02 | — |

Llama 3.3 作为评判者时的胜率：

| 生成配置 | Dual 策略 | Single 策略 | 合计 |
|----------|----------|------------|------|
| GPT-4o 生成 | **0.54** | 0.24 | 0.39 |
| Llama 3.3 生成 | **0.81** | 0.08 | 0.43 |
| 合计 | **0.86** | 0.03 | — |

### 评判者一致性分析

| 对比配置 | 严格一致率 | 放松一致率 |
|----------|----------|----------|
| D,GPT vs D,Llama | 30.5% | **86.5%** |
| D,GPT vs S,GPT | 40% | **91%** |
| D,Llama vs S,GPT | 72% | **97.5%** |
| D,Llama vs S,Llama | 76.5% | **99%** |
| S,GPT vs S,Llama | 52.5% | **87%** |

### 关键发现

- **双提示策略全面碾压单提示**：无论用哪种 LLM 生成或评判，双提示的胜率都显著高于单提示，差距从 2 倍到 10 倍不等
- **生成模型选择影响不大**：GPT-4o 和 Llama 3.3 作为生成器时的总体胜率几乎相同（右列"合计"），这一发现出乎意料，因为通常 LLM 评判者倾向偏好自己的生成结果
- **Token 成本差异巨大**：双提示方法的平均 token 消耗约 6092-7583 tokens，而单提示仅 946-1143 tokens，约 6 倍差距。这是因为双提示每轮都需要传入完整历史
- **最优方案**：考虑到 Llama 3.3 成本更低且生成质量不逊于 GPT-4o，双提示 + Llama 3.3 是性价比最优的组合

## 亮点与洞察

- **简洁的实验设计**：整个研究只用了一个核心实验（成对比较），但通过巧妙的交叉设计（2策略 × 2生成器 × 2评判器 × 正反序）控制了多个混淆变量。这种简约但严谨的实验设计值得借鉴
- **反直觉发现**：GPT-4o 和 Llama 3.3 的生成质量几乎无差异，但价格差异巨大。这对实际应用有直接指导意义——用便宜的开源模型 + 双提示策略可能是最佳选择
- **长度偏差控制方法**：通过在评判提示中明确指出"不要考虑长度因素"来控制偏差，效果经过统计检验证实。这种简单的提示工程技巧可以广泛复用

## 局限与展望

- 实验仅覆盖求职面试这一种对话类型，结论是否推广到其他领域（客服、医疗问诊等）有待验证
- LLM-as-Judge 与人类判断的一致性约 65.74%（GPT-4 数据），并非完美替代
- 对话质量只用"AI 检测难度"这一维度衡量，未评估对话内容的准确性和面试问题的专业性
- Token 成本分析基于 API 调用次数，未考虑延迟和并发等实际工程因素
- 未来可扩展到更多 LLM（如 Claude、Gemini）和更多领域的对话生成

## 相关工作与启发

- **vs SODA (Kim et al., 2023)**: SODA 使用单提示策略生成百万级对话，本文证明双提示质量更高但成本也更高，具体选择取决于质量-成本权衡
- **vs BotChat (Duan et al., 2024)**: BotChat 首先提出用 PairEval 评估两个 LLM 的多轮对话能力，本文借用其评估框架但聚焦于生成策略的对比
- **vs DiaSynth (Suresh et al., 2025)**: DiaSynth 也是对话合成框架，但通过知识种子和单提示生成，本文补充了双提示的系统性对比

## 评分

- 新颖性: ⭐⭐⭐ 研究问题直观且重要，但方法上没有新的技术贡献，主要是实证对比
- 实验充分度: ⭐⭐⭐⭐ 变量控制严谨，统计检验充分，但仅限单一领域
- 写作质量: ⭐⭐⭐⭐ 写得非常清晰，逻辑链完整，数据呈现规范
- 价值: ⭐⭐⭐ 实用性强，但作为 workshop 论文深度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)
- [\[ACL 2025\] When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](../../ACL2026/dialogue/reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2025\] Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)
- [\[NeurIPS 2025\] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](../../NeurIPS2025/dialogue/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)

</div>

<!-- RELATED:END -->
