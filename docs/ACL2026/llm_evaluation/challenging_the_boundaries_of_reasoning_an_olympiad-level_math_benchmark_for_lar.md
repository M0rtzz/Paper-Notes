---
title: >-
  [论文解读] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
description: >-
  [ACL 2026][LLM评测][数学推理基准] 提出 OlymMATH，首个统一自然语言评估和形式化定理证明的奥赛级数学基准，包含350题双语（中英文）题目，涵盖OlymMATH-EASY/HARD（200题数值答案）和OlymMATH-LEAN（150题Lean 4形式化），揭示最强模型在HARD子集上仅58.4%准确率。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "数学推理基准"
  - "奥赛数学"
  - "形式化验证"
  - "Lean4"
  - "双语评估"
---

# Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models

**会议**: ACL 2026  
**arXiv**: [2503.21380](https://arxiv.org/abs/2503.21380)  
**代码**: [GitHub](https://github.com/RUCAIBox/OlymMATH)  
**领域**: LLM数学推理  
**关键词**: 数学推理基准, 奥赛数学, 形式化验证, Lean4, 双语评估

## 一句话总结

提出 OlymMATH，首个统一自然语言评估和形式化定理证明的奥赛级数学基准，包含350题双语（中英文）题目，涵盖OlymMATH-EASY/HARD（200题数值答案）和OlymMATH-LEAN（150题Lean 4形式化），揭示最强模型在HARD子集上仅58.4%准确率。

## 研究背景与动机

**领域现状**：推理模型（DeepSeek-R1、o3-mini、Gemini 2.5 Pro等）的快速进步已使GSM8K、MATH等现有数学基准趋于饱和，急需更具挑战性的评估框架。

**现有痛点**：（1）奥赛级基准规模不足（如AIME仅30题，单题差异影响3.33%准确率）；（2）部分基准依赖LLM-as-judge评估证明题，存在评估幻觉风险；（3）数据泄露问题严重——从AoPS等在线源爬取的题目可能已在预训练数据中；（4）几乎所有基准仅支持英文，缺乏多语言评估。

**核心矛盾**：需要一个同时满足高难度、大规模、低泄露、双语、双范式（答案验证+过程验证）的数学推理基准——但现有基准无一满足所有条件。

**本文目标**：构建首个统一自然语言和形式化证明双范式的奥赛级双语数学基准。

**切入角度**：从印刷出版物（非网络来源）手动收集题目以最小化数据泄露。

**核心idea**：OlymMATH-EASY/HARD提供计算题用sympy规则验证（结果评估），OlymMATH-LEAN提供Lean 4形式化题目用定理证明器验证（过程评估），两者互补。

## 方法详解

### 整体框架

OlymMATH 要解决的核心问题是：现有数学基准要么趋于饱和、要么规模太小、要么只验证答案不验证推理过程。它把一套奥赛级题库切成三个互不重叠的子集，从两个维度协同评估——一边用带数值答案的计算题做客观的结果验证，一边用 Lean 4 形式化题目做严格的过程验证，输入一道奥赛题、输出一个既能判定"答案对不对"又能判定"推理严不严谨"的双重信号。三个子集中 OlymMATH-EASY（100 题较易）、OlymMATH-HARD（100 题困难）均为可由 sympy 自动校验的数值题并配中英文平行版本，OlymMATH-LEAN（150 题）则在 Lean 4 中形式化、附双语自然语言陈述与解答，整体覆盖数论、代数、组合、几何四大领域。

### 关键设计

**1. 双范式评估统一：结果验证与过程验证互补覆盖**

仅看答案对错有一个隐蔽盲点——模型可能不经严谨推导，靠"启发式猜测"直接试出正确数值，纯结果评估会把这种行为误判为真推理。OlymMATH 因此让两种范式分工互补：计算题用 sympy 按规则比对最终答案，客观、可扩展但只管结果；形式化证明题交给 Lean 4 验证器逐步检查证明过程，严格、可审计但要求模型具备形式化能力。前者负责覆盖广度、后者专门堵住"猜答案"的漏洞，合起来才能同时回答"答案对不对"和"过程严不严谨"。

**2. 数据泄露防护：题目全部取自印刷出版物**

现有基准的泄露风险相当现实：Omni-MATH 从 AoPS 爬题、PolyMath 直接沿用 AIME/CNMO，这些题目很可能早已躺在模型的预训练语料里，使评估结果虚高、反映的是记忆而非推理。OlymMATH 索性绕开网络来源，所有题目从专业杂志和教科书等印刷出版物手动收集。n-gram 泄露分析印证了这一策略——OlymMATH 的泄露指标显著低于 PolyMath，评分更贴近模型的真实推理水平。

**3. 双语平行评估：每题保留语义等价的中英双版本**

多语言推理是 LLM 落地的现实需求，但此前几乎所有数学基准只有英文，无法回答"换一种语言模型还会不会做"。OlymMATH 为每道题准备语义等价的英文与中文两版，把语言当成一个可控变量。实验中两版之间出现了稳定且一致的性能差距，而这种差距在单语基准上根本无从暴露。

## 实验关键数据

### 主实验（OlymMATH-HARD EN）

| 模型 | 准确率 |
|------|--------|
| Gemini 2.5 Pro | **58.4%** |
| o3-mini | 31.2% |
| DeepSeek-R1 | 19.5% |

### 关键发现
- 最强模型在HARD子集上仅58.4%准确率，说明奥赛级数学仍极具挑战性
- 所有模型英文版本准确率一致高于中文，揭示多语言推理能力差距
- 案例分析发现模型存在"启发式猜测"行为——跳过严谨推导直接试探答案
- n-gram泄露分析确认OlymMATH相比PolyMath有更低的数据泄露风险
- 开源582K+推理轨迹支持社区深入分析

## 亮点与洞察
- **双范式统一是关键创新**：首次在单一基准中融合结果评估和过程评估
- **"启发式猜测"的发现重要**：模型可能通过非严谨途径得到正确答案，仅答案验证会高估能力
- **印刷出版物sourcing策略有效**：最小化泄露风险的同时保证题目质量
- **582K推理轨迹是宝贵资源**：支持社区分析不同模型的推理模式

## 局限与展望
- **规模仍然有限**：350题虽优于AIME但仍不算大
- **Lean 4形式化门槛高**：限制了题目类型和数量扩展
- **仅覆盖四大领域**：概率、统计等数学分支未覆盖
- 未来方向：扩展到更多数学分支、自动化形式化管线、基于推理轨迹的训练数据构建

## 相关工作与启发
- **vs AIME**：仅30题且英文单一，统计可靠性差且难度天花板不足
- **vs Omni-MATH**：规模大但从AoPS爬取有泄露风险，且依赖LLM-as-judge评估证明
- **vs miniF2F**：形式化基准但仅英文且题目来自知名竞赛，泄露风险高

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个双范式+双语的奥赛级数学基准，设计理念清晰
- 实验充分度: ⭐⭐⭐⭐ 多模型评估、泄露分析、案例研究、582K推理轨迹开源
- 写作质量: ⭐⭐⭐⭐⭐ 与前作比较充分，问题动机清晰
- 价值: ⭐⭐⭐⭐⭐ 为数学推理评估设定新标准，双范式设计有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](../../ICML2025/llm_evaluation/leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)

</div>

<!-- RELATED:END -->
