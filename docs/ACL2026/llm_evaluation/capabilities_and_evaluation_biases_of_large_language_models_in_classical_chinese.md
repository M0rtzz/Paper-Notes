---
title: >-
  [论文解读] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry
description: >-
  [ACL 2026][LLM评测][古诗生成] 本文提出了一个三步评估框架（计算特征提取 + LLM-as-Judge + 人类专家验证）来系统评估六种 LLM 在唐诗生成上的能力，发现了关键的"回声室"效应：LLM 系统性地高估模仿统计模式但违反格律规则的机器生成诗歌，与人类专家判断显著偏离。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "古诗生成"
  - "唐诗"
  - "LLM评估偏差"
  - "回声室效应"
  - "人机评估"
---

# Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry

**会议**: ACL 2026  
**arXiv**: [2510.15313](https://arxiv.org/abs/2510.15313)  
**代码**: [https://github.com/boleima/Tang-Poetry](https://github.com/boleima/Tang-Poetry)  
**领域**: LLM评测  
**关键词**: 古诗生成, 唐诗, LLM评估偏差, 回声室效应, 人机评估

## 一句话总结

本文提出了一个三步评估框架（计算特征提取 + LLM-as-Judge + 人类专家验证）来系统评估六种 LLM 在唐诗生成上的能力，发现了关键的"回声室"效应：LLM 系统性地高估模仿统计模式但违反格律规则的机器生成诗歌，与人类专家判断显著偏离。

## 研究背景与动机

**领域现状**：LLM 在文本生成（包括创意写作）上展示了令人印象深刻的能力。古典中国诗歌（特别是唐诗）因其严格的韵律、声调约束和深厚的文化内涵，构成了 AI 创造力的极端挑战。

**现有痛点**：(1) LLM 在诗歌生成中仍常出现行间不连贯、意象缺乏原创性、或复现记忆诗句等问题；(2) 传统自动指标（BLEU、ROUGE）无法捕捉韵律、意象和美学价值；(3) LLM-as-Judge 方法可能存在系统性偏差——模型可能膨胀自身输出或与同伴趋同。

**核心矛盾**：诗歌生成需要兼顾结构正确性和美学质量，而当前的自动评估方法无法可靠地衡量这两个维度，特别是在文化敏感的创意任务中。

**本文目标**：建立系统性的 LLM 唐诗生成和评估研究，揭示 LLM 在诗歌生成中的能力边界和评估中的偏差。

**切入角度**：以唐诗为测试平台，设计包含五个维度（体裁、诗人风格、主题、情感、意象）的生成任务，通过三步框架提供多层次评估。

**核心 idea**：LLM 生成的诗歌可能在表面统计特征上接近人类作品，但在严格格律遵守上存在系统性缺陷，而 LLM 评估者无法识别这些缺陷，形成"回声室"。

## 方法详解

### 整体框架

(1) **大规模生成**——6 种 LLM 各生成约 2,500 首诗（共 15,000 首），覆盖五个诗歌维度；(2) **三步评估**——Step 1 自动计算特征提取（格律合规率等），Step 2 LLM 交叉评估（每个模型评估其他模型的输出），Step 3 人类专家验证（古诗文领域专家）。

### 关键设计

1. **多维度诗歌生成设计**:

    - 功能：系统性地覆盖唐诗创作的各关键维度
    - 核心思路：定义五个维度——体裁（五/七言绝句/律诗）、诗人风格（李白/杜甫/白居易/王维/李商隐）、主题（山水/乡愁/怀古/田园/离别）、情感（悲伤/宁静/豪放/浪漫/喜悦）、意象（风/花/柳/月/雁）。使用显式提示指定维度，温度 T=0.4
    - 设计动机：控制变量的实验设计使不同模型和维度间的对比具有科学性

2. **计算特征提取（Step 1）**:

    - 功能：客观量化诗歌的格律合规性
    - 核心思路：自动检测平仄格式、对仗、押韵等格律规则的遵守情况，计算格律合规率。这是唐诗评估中最客观可量化的维度
    - 设计动机：格律规则是唐诗的硬性约束，违反格律的诗歌在专业角度不合格，但 LLM 评估者可能忽略这些违反

3. **LLM 交叉评估与人类专家验证（Step 2 & 3）**:

    - 功能：揭示自动评估与人类判断之间的偏差
    - 核心思路：Step 2 让每个 LLM 从主题相关性、情感一致性、意象/结构、语言真实性等多维度评估其他模型生成的诗歌；Step 3 由古诗文专家对相同样本进行独立评估。通过对比两者发现"回声室"效应
    - 设计动机：LLM 评估的可靠性是当前热点问题，诗歌领域提供了文化敏感性和格律约束的独特测试场景

### 损失函数 / 训练策略

不涉及模型训练。生成使用温度 T=0.4，评估在零样本设置下进行。

## 实验关键数据

### 主实验

**六种 LLM 生成能力分层**

- 第一梯队：Qwen2.5-7B-Instruct（格律合规率最高，整体质量最佳）
- 第二梯队：GLM-4-9B-Chat、DeepSeek-V2-Lite-Chat
- 第三梯队：Baichuan2-7B-Chat、Gemma-2-9B-it、Mistral-7B（中文诗歌能力较弱）

### 消融实验

**"回声室"效应**：LLM 评估者系统性地给机器生成的诗歌打高分，即使这些诗歌违反了严格的格律规则。人类专家则能准确识别格律违反并显著降低评分。LLM 自评分与交叉评分之间存在偏向自身输出的倾向。

### 关键发现

- 以中文为强项的模型（Qwen、GLM、DeepSeek）在唐诗生成上显著优于以英文为主的模型
- LLM 评估者倾向于高估模仿统计模式但违反格律的诗歌——"回声室"效应
- 格律合规率是最有区分度的质量指标，但恰恰是 LLM 评估者最容易忽略的
- 不同维度的生成难度不同，风格模仿比格律遵守更容易

## 亮点与洞察

- 首次系统研究 LLM 在古典中国诗歌生成和评估中的"回声室"效应
- 三步评估框架可推广到其他文化敏感的创意生成任务
- 对 LLM-as-Judge 方法的可靠性敲响了警钟，特别是在需要专业领域知识的评估中
- 数据集和代码开源，可复现性强

## 局限与展望

- 仅评估了 6 种开源模型，未包含商业闭源模型
- 人类评估受限于专家可用性，评估规模有限
- 仅聚焦唐诗，未扩展到其他诗歌形式
- 未来可探索格律感知的微调策略来改善 LLM 的诗歌生成能力

## 相关工作与启发

- 与 LLM-as-Judge 领域的偏差研究（如 Clark 等人的发现）形成诗歌领域的具体验证
- 为中文 NLP 社区提供了重要的创意生成基准
- 格律合规率的自动检测方法可推广到其他有严格形式约束的文本生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究 LLM 唐诗生成的回声室效应
- 实验充分度: ⭐⭐⭐⭐ 6 模型 × 5 维度 × 3 步评估的全面设计
- 写作质量: ⭐⭐⭐⭐ 研究设计严谨，图表清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](../../ACL2025/llm_evaluation/mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)

</div>

<!-- RELATED:END -->
