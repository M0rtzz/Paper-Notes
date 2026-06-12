---
title: >-
  [论文解读] (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding
description: >-
  [ACL 2025 (Main)][比喻语言理解] 本文提出 (RSA)² 框架，通过在概率语用学的 RSA 框架中显式建模说话者的修辞策略（如反讽、夸张），使 LLM 能够在不建模说话者动机的情况下正确理解非字面义，在反讽理解数据集 PragMega+ 上达到 SOTA。
tags:
  - "ACL 2025 (Main)"
  - "比喻语言理解"
  - "语用学"
  - "理性言语行为框架"
  - "反讽解释"
  - "修辞策略"
---

# (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding

**会议**: ACL 2025 (Main)  
**arXiv**: [2506.09301](https://arxiv.org/abs/2506.09301)  
**代码**: 无  
**领域**: 其他  
**关键词**: 比喻语言理解, 语用学, 理性言语行为框架, 反讽解释, 修辞策略

## 一句话总结

本文提出 (RSA)² 框架，通过在概率语用学的 RSA 框架中显式建模说话者的修辞策略（如反讽、夸张），使 LLM 能够在不建模说话者动机的情况下正确理解非字面义，在反讽理解数据集 PragMega+ 上达到 SOTA。

## 研究背景与动机

**领域现状**：比喻语言（figurative language）在人类交流中无处不在——反讽、夸张、低调陈述等修辞手法使得话语的字面意义与意图意义不一致。Rational Speech Act (RSA) 框架是概率语用学中最广泛使用的理论模型，通过递归地模拟说话者和听话者之间的意图推理来解释话语的语用含义。

**现有痛点**：标准 RSA 框架有一个根本性限制——它只能产生与字面义一致的解释，无法为非字面话语分配非零概率。已有的解决方案（affect-aware RSA）要求显式建模说话者使用比喻语言的"动机"（如表达愉悦或恼怒的情感），但这种建模方式需要为每种场景单独设计情感空间，通用性差。另一方面，LLM 虽然具有强大的语言能力，但在比喻语言理解上表现不佳，存在严重的字面义偏好偏差。

**核心矛盾**：要让模型正确理解比喻语言，需要考虑非字面义解释的可能性，但标准 RSA 从数学上就排除了这种可能性。现有的 affect-aware RSA 虽然能处理这个问题，但要求预先定义说话者的情感动机，而实际中一个人使用反讽的原因可能是多种多样的（幽默、讽刺、习惯性表达等），很难统一建模。

**本文目标**：设计一种不需要建模说话者动机、但能自然产生非字面义解释的RSA框架扩展。

**切入角度**：非字面语言的使用遵循系统性的模式，这些模式可以被归纳为"修辞策略"（rhetorical strategy）。例如反讽的特征是意图义与字面义相反，夸张的特征是意图义弱于字面义。这些策略是说话者"如何"非字面地表达的规律，与"为什么"非字面表达（动机）是正交的。

**核心 idea**：将修辞策略作为显式的潜变量引入 RSA 框架，通过边缘化修辞策略来获得非字面义解释，从而绕过对说话者动机的建模。

## 方法详解

### 整体框架

(RSA)² 的输入是一个上下文（context）和一个话语（utterance），输出是对话语意图义的概率分布。框架核心是在 RSA 的 literal listener → pragmatic speaker → pragmatic listener 三层递归推理基础上，在每一层引入修辞策略变量 $r \in \mathcal{R}$。推理时，修辞策略作为潜变量被边缘化，使得最终的意图义分布能同时考虑字面和非字面解释。在 LLM 实验中，框架进一步与预训练语言模型结合，利用 LLM 来估计各条件概率分布。

### 关键设计

1. **修辞函数泛化 (Rhetorical Function Generalization)**:

    - 功能：将标准 RSA 中的二值语义理解函数 $\mathbb{1}_{m \in [\![u]\!]}$ 泛化为连续的修辞函数 $f_r: \mathcal{C} \times \mathcal{M} \times \mathcal{U} \to [0,1]$
    - 核心思路：对于每种修辞策略 $r$，定义一个修辞函数 $f_r$ 来描述该策略下特定意义与话语的兼容程度。例如，对于反讽策略，"天气真棒"在暴风雪语境下与"天气糟糕"的兼容度为1，与"天气真棒"的兼容度为0。这个函数替换原始 literal listener 方程中的语义指示函数：$P_{L_0}(m|c,u,r) \propto f_r(c,m,u) \cdot P(m|c)$
    - 设计动机：标准 RSA 的语义函数是二值的（兼容/不兼容），这从数学上确保了非字面义概率为零。泛化为连续函数后，非字面义可以通过合适的修辞函数获得非零概率

2. **修辞策略潜变量边缘化 (Rhetorical Strategy Marginalization)**:

    - 功能：通过对修辞策略的后验概率进行边缘化，将条件于不同修辞策略的 listener 分布混合为统一的意图义分布
    - 核心思路：最终的 pragmatic listener 分布为 $P_{L_1}(m|c,u) = \sum_{r'} P_{L_1}(m|c,u,r') \cdot P(r'|c,u)$。这意味着模型会同时考虑"如果说话者在使用字面义/反讽/夸张/…，那么意图义分别是什么"，然后根据各修辞策略的后验概率加权混合。在 LLM 实验中，修辞策略后验通过 prompt 从 LLM 估计
    - 设计动机：在实际场景中，听话者不确定说话者正在使用哪种修辞策略，因此需要对所有可能的策略进行概率加权。这是标准的贝叶斯推理方式，理论上干净优雅

3. **LLM 整合方案 (LLM Integration with Prompt-based Probability Estimation)**:

    - 功能：将 (RSA)² 与 LLM 结合，利用 LLM 来估计框架中需要的各条件概率
    - 核心思路：使用两个 LLM：一个 instruction-tuned 模型 $N$（如 Mistral-7B-Instruct）通过多选题格式的 prompt 估计 $P_N(m|c,u)$、$P_N(r|c,u)$、$P_N(m|c,u,r)$ 等分布；一个 base 模型 $G$（如 Llama-8B）用于生成替代话语集合及其先验概率。为避免位置偏差，对选项顺序随机排列10次取平均
    - 设计动机：将 RSA 的概率语用推理与 LLM 的语言理解能力相结合——LLM 提供语义先验，RSA 提供结构化的语用推理框架，两者互补

### 损失函数 / 训练策略

(RSA)² 核心部分不涉及端到端训练。在 ironic weather utterances 实验中，使用 2层神经网络（16×16×5，sigmoid 激活）学习修辞函数 $f_r$，训练 500 epochs，Adam 优化器（lr=0.001, weight decay=0.001），交叉熵损失。在 LLM 实验中，所有概率通过 prompt 工程估计，不需要微调 LLM。

## 实验关键数据

### 主实验

**RSA 模型实验（非字面数字表达 + 反讽天气话语）**：

| 模型 | 非字面数字 $L_0$ MAD↓ | 非字面数字 $L_1$ MAD↓ | 天气话语 $L_0$ MAD↓ | 天气话语 $L_1$ MAD↓ |
|------|-----|-----|-----|-----|
| Affect-Aware RSA | - | 0.0436 | 0.2377 | 0.1278 |
| (RSA)² | 0.0438 | 0.0467 | 0.1647 | **0.1229** |

**LLM 反讽理解实验（PragMega+ 数据集）**：

| 模型 | 正确义概率↑ | 错误义概率↓ | 干扰义概率↓ |
|------|-----------|-----------|-----------|
| LLM RSA $L_0$ | 0.73 | 0.24 | 0.02 |
| LLM RSA $L_1$ | 0.76 | 0.22 | 0.01 |
| LLM (RSA)² $L_0$ (with $I(r|c,u)$) | **0.85** | **0.13** | 0.01 |
| LLM (RSA)² $L_1$ (with $I(r|c,u)$) | 0.84 | 0.13 | 0.01 |

### 消融实验

| 配置 | 正确义概率 | 相对变化 | 说明 |
|------|-----------|---------|------|
| LLM RSA $L_1$ (完整) | 0.76 | - | 基线 |
| LLM RSA $L_1$ (去除 $P(m|c)$) | 0.44 | -42.7% | 意义先验是关键 |
| LLM RSA $L_1$ (去除 $P(u|c)$) | 0.78 | +1.8% | 话语先验影响小 |
| LLM (RSA)² $L_0$ with $I$ (完整) | 0.85 | - | 最优模型 |
| LLM (RSA)² $L_0$ with $I$ (去除 $P(m|c)$) | 0.51 | -39.4% | 先验仍然重要 |
| LLM (RSA)² $L_0$ with $I$ (去除 $P(u|c)$) | 0.84 | +0.2% | 话语先验几乎无影响 |

### 关键发现

- (RSA)² 在反讽场景下效果最佳（意图义概率 >0.8），在天气话语反讽数据集上超越 affect-aware RSA
- 修辞策略后验的质量很关键：使用指示函数 $I(r|c,u)$ 比使用 LLM 估计的连续概率 $P_N(r|c,u)$ 效果更好
- 意义先验 $P(m|c)$ 是性能的最大贡献因素，RSA 的递归推理过程本身贡献有限（这是因为替代话语生成倾向于产生字面义的同义改写）
- LLM 在判断修辞策略方面呈现不对称性：对反讽场景判断准确（$P(r=\text{irony}|c,u)=0.88$），但对字面场景判断较差（$P(r=\text{literal}|c,u)=0.55$）

## 亮点与洞察

- **修辞策略 vs 说话者动机**：将"如何非字面表达"（修辞策略）与"为什么非字面表达"（动机）解耦是核心创新。这不仅简化了建模，还更符合语言学直觉——听话者解读比喻语言时，确实更多是在推断使用了什么修辞手法，而非深入分析说话者的情感动机
- **数学证明的理论贡献**：论文证明了 affect-aware RSA（更准确地说是 QUD-RSA）是 (RSA)² 的特例，但反之不成立，这从理论上确立了 (RSA)² 的表达能力更强
- **LLM + 概率语用学的结合范式**：展示了用 prompt 从 LLM 中提取概率估计并嵌入到结构化推理框架中的方法，这种范式可以推广到其他需要结构化推理的 NLP 任务

## 局限与展望

- 数据集规模较小且仅限英语，跨语言和跨文化的比喻理解尚未验证
- 替代话语的生成方式导致 RSA 递归推理效果有限——生成的替代话语往往是字面义的同义改写，无法有效区分不同意图义
- 修辞策略集合需要预先定义，如何自动发现适合特定场景的修辞策略仍是开放问题（附录中的聚类方法效果不佳）
- LLM 的修辞策略判断在字面场景下不够准确，改进策略分类器可能进一步提升性能

## 相关工作与启发

- **vs Affect-Aware RSA (Kao et al., 2014/2015)**: 它们通过情感投影来实现非字面义解释，需要为每种场景定义情感空间。(RSA)² 通过修辞策略替代情感，更通用且不需要场景特定的设计
- **vs LLM 直接推理**: 直接用 LLM 理解比喻语言会受到严重的字面义偏好偏差。(RSA)² 通过结构化的语用推理框架来纠正这种偏差
- **vs Tsvilodub et al. (2025)**: 他们将 affect-aware RSA 与 LLM 结合用于数字表达理解。(RSA)² 提出了更通用的框架，且不依赖情感变量

## 评分

- 新颖性: ⭐⭐⭐⭐ 将修辞策略作为潜变量引入 RSA 是理论上优雅的创新，且有严格的数学支撑
- 实验充分度: ⭐⭐⭐ 数据集规模较小，场景有限（主要是反讽），但包含了理论证明和多层次消融
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构非常清晰，数学推导严谨，从动机到方法到实验的叙述逻辑流畅
- 价值: ⭐⭐⭐⭐ 对概率语用学领域有重要理论贡献，对 LLM 比喻理解有实际参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)

</div>

<!-- RELATED:END -->
