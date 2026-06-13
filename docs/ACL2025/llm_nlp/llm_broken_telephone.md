---
title: >-
  [论文解读] LLM as a Broken Telephone: Iterative Generation Distorts Information
description: >-
  [ACL 2025][LLM 其他][信息失真] 以翻译为测试床模拟 LLM 的"传话游戏"，发现信息在 100 次迭代翻译后严重失真——一辆卡车司机的罚款新闻经 100 轮英泰互译后变成"小汽车获得赔偿后发生爆炸"，而中间语言的选择、链条复杂度和解码温度是失真速度的关键调控因素。 领域现状：随着 LLM 生成的内容日益充斥…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "信息失真"
  - "迭代生成"
  - "模型坍缩"
  - "机器翻译"
  - "事实性"
---

# LLM as a Broken Telephone: Iterative Generation Distorts Information

**会议**: ACL 2025  
**arXiv**: [2502.20258](https://arxiv.org/abs/2502.20258)  
**代码**: [https://github.com/amr-mohamedd/LLM-as-a-Broken-Telephone](https://github.com/amr-mohamedd/LLM-as-a-Broken-Telephone)  
**领域**: LLM / NLP  
**关键词**: 信息失真, 迭代生成, 模型坍缩, 机器翻译, 事实性

## 一句话总结
以翻译为测试床模拟 LLM 的"传话游戏"，发现信息在 100 次迭代翻译后严重失真——一辆卡车司机的罚款新闻经 100 轮英泰互译后变成"小汽车获得赔偿后发生爆炸"，而中间语言的选择、链条复杂度和解码温度是失真速度的关键调控因素。

## 研究背景与动机

**领域现状**：随着 LLM 生成的内容日益充斥互联网，模型消费自身输出的循环不可避免。Model collapse（模型坍缩）研究已证明在合成数据上迭代训练会导致分布退化，但对推理阶段的迭代生成（非训练）的信息失真研究几乎空白。

**现有痛点**：
   - Model collapse 文献聚焦于训练循环，而非推理时的迭代处理
   - Perez et al. (2024) 研究了复述/续写链中的文本属性演化（毒性、正面性、难度），但忽略了翻译——最常见的迭代 LLM 应用之一
   - 已有研究仅用单模型单语言链，未考虑多模型协作（如多 agent 系统中模型 A 输出→模型 B 处理的场景）
   - 缺乏对事实性和文本相似度在迭代中退化速度的系统量化

**核心矛盾**：多 agent 系统和 AI 内容循环日益普遍，但我们对"信息经过 LLM 反复处理后还剩多少真实性"几乎一无所知

**本文目标** (1) 量化 LLM 迭代翻译中的信息失真积累 (2) 分析中间语言/链条复杂度/模型组合对失真的影响 (3) 探索缓解策略

**切入角度**：借鉴人类传话游戏（Telephone Game）的隐喻，将翻译链设计为可控的迭代生成实验——每轮 EN→中间语言→EN，100 轮后与原文对比

**核心 idea**：LLM 也是"坏掉的电话"——信息经迭代生成后渐进失真，且失真程度受语言选择、链条复杂度和解码策略的系统性影响。

## 方法详解

### 整体框架
3 个数据集（BookSum、ScriptBase、News2024）× 150 个文档 × 6 种中间语言（FR/DE/NL/VN/ZH/TH）× 2 个模型（Llama-3.1-8B / Mistral-7B）→ 100 轮迭代翻译 → 每轮用 5 类文本相关性指标 + FActScore 事实性指标与原文对比

### 关键设计

1. **三种实验设置的递进复杂度**：
    - **Exp1 双语自环**：单模型在 EN↔中间语言间反复翻译 100 轮，6 种中间语言 × 2 模型 × 3 数据集
    - **Exp2 双语双人**：两个不同模型交替参与同一翻译链（模拟多 agent 协作），EN↔FR 和 EN↔TH
    - **Exp3 多语多人**：2-4 种中间语言 + 2-3 个模型在同一链中随机排列，测试复杂度上限
    - 设计动机：从单一到复杂递进探测，模拟真实场景中从单人翻译到多 agent 多语言信息传播

2. **双轴评估体系**：
    - **文本相关性**：BLEU（词串精确匹配）+ ROUGE-1（词级覆盖）+ CHR-F（字符级）+ METEOR（释义变体）+ BERTScore（语义相似度）
    - **事实性**：FActScore 将长文本分解为原子事实，用 Claude 3.5 Sonnet 逐条验证——以原始文本为事实参考
    - 设计动机：文本相关性捕捉表层偏移，FActScore 捕捉深层事实扭曲

3. **梯度量化方法**：
    - 对 FActScore 随迭代次数的变化曲线计算平均梯度（average gradient），量化失真速度
    - 设计动机：不同语言/设置的最终得分可能趋同，但退化速度差异显著——梯度更能区分风险等级

### 消融实验
- 温度消融：1e-6, 0.25, 0.5, 0.75, 1.0 五档
- Prompt 约束度消融：simple / base / constrained 三种翻译 prompt
- 复述链消融：将翻译替换为同语言复述，验证失真是否仅限翻译

## 实验关键数据

### Exp1：FActScore 平均梯度（失真速度）

| 语言对 | Llama (News2024) | Mistral (News2024) |
|--------|-----------------|-------------------|
| EN↔FR | -0.004 ± 0.003 | -0.007 ± 0.004 |
| EN↔DE | -0.005 ± 0.003 | -0.011 ± 0.006 |
| EN↔NL | -0.005 ± 0.003 | -0.011 ± 0.006 |
| EN↔VN | -0.008 ± 0.005 | -0.027 ± 0.015 |
| EN↔ZH | -0.011 ± 0.006 | -0.024 ± 0.012 |
| EN↔TH | **-0.018 ± 0.009** | **-0.038 ± 0.022** |

### Exp3：链条复杂度影响（FActScore）

| 设置 | 语言数 | 模型数 | 第10轮 | 第100轮 | 平均梯度 |
|------|-------|-------|--------|---------|---------|
| Setting 1 | 3 | 2 | 0.063 | 0.04 | -0.036 ± 0.02 |
| Setting 2 | 3 | 3 | 0.075 | 0.04 | -0.034 ± 0.02 |
| Setting 3 | 5 | 2 | 0.054 | 0.04 | **-0.038 ± 0.02** |

### 温度消融（EN↔FR，Llama）

| 温度 | 前2轮后稳定性 | 100轮后 FActScore 趋势 |
|------|-------------|---------------------|
| 1e-6 | 几乎稳定 | 轻微下降后持平 |
| 0.25 | 缓慢下降 | 持续缓慢下降 |
| 0.50 | 明显下降 | 中等速度下降 |
| 1.00 | 最陡下降 | 持续剧烈发散 |

### 关键发现
- **语言相似度决定失真速度**：拉丁字母语言（FR/DE/NL）梯度接近 0，非拉丁字母语言（TH/ZH/VN）梯度大 3-10 倍
- **泰语是最"坏的电话"**：EN↔TH 在所有数据集和模型上失真最快，BookSum 上 Mistral 梯度达 -0.040
- **更多语言 > 更多模型**：Setting 3（5 种语言/2 模型）比 Setting 2（3 种语言/3 模型）退化更快——语言多样性比模型多样性更能放大失真
- **复述也会失真**：非翻译的纯复述链同样表现出信息退化，但速度慢于翻译链
- **低温度 + 约束 prompt 可有效缓解**：温度 1e-6 几乎冻结失真，constrained prompt 比 simple prompt 显著减缓退化

## 亮点与洞察
- **"传话游戏"的隐喻精准且具有广泛警示意义**：Table 1 的卡车→巴士→小车的渐变例子直观展示了失真过程，对所有使用 LLM 做迭代加工的流程（摘要→翻译→摘要）都有警示
- **对多 agent 系统的直接启示**：Agent A 的输出被 Agent B 处理再交给 Agent C——每一步都在积累失真。Chain-of-agents 的设计需要内建事实性校验
- **事实性退化比表层词汇变化更危险**：BLEU 下降可能只是换了同义词，但 FActScore 下降意味着事实被扭曲——这对新闻/医疗/法律场景风险极大

## 局限与展望
- **仅用 7-9B 模型**：更大的模型（70B+）或 GPT-4 级别模型是否更抗失真需要验证
- **数据集领域有限**：书籍/剧本/新闻三个领域特征相似，专业领域（医学/法律）的失真可能更严重
- **仅用默认解码参数**：贪心解码 vs beam search 的影响未探索
- **FActScore 的评估者偏差**：用 Claude 3.5 Sonnet 做 judge，但 judge 模型本身可能有系统性偏差

## 相关工作与启发
- **vs Shumailov et al. (2023) Model Collapse**：Model collapse 是训练循环问题；本文是推理循环问题——两者共同构成 AI 内容循环的完整风险图景
- **vs Perez et al. (2024)**：Perez 研究复述链的毒性/正面性演化；本文增加翻译场景 + 多模型链 + 事实性量化，视角更完整
- **vs Peterson (2024) Knowledge Collapse**：Knowledge collapse 是宏观概念；本文用翻译链提供了微观可量化的信息退化证据

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](longdpo_unlock_better_long-form_generation_abilities_for_llms_via_critique-augme.md)
- [\[ACL 2025\] Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons](knockout_llm_assessment_using_large_language_models_for_evaluations_through_iter.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)

</div>

<!-- RELATED:END -->
