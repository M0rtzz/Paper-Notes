---
title: >-
  [论文解读] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models
description: >-
  [ACL 2026][LLM推理][数学推理基准] 提出 OlymMATH，首个统一自然语言评估和形式化定理证明的奥赛级数学基准，包含350题双语（中英文）题目，涵盖OlymMATH-EASY/HARD（200题数值答案）和OlymMATH-LEAN（150题Lean 4形式化），揭示最强模型在HARD子集上仅58.4%准确率。
tags:
  - ACL 2026
  - LLM推理
  - 数学推理基准
  - 奥赛数学
  - 形式化验证
  - Lean4
  - 双语评估
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

OlymMATH包含三个不重叠子集：（1）OlymMATH-EASY（100题，较易奥赛级）和（2）OlymMATH-HARD（100题，困难奥赛级），均为计算题有数值答案，支持sympy自动验证，每题有中英文平行版本；（3）OlymMATH-LEAN（150题），在Lean 4中形式化，配有双语自然语言陈述和解答。覆盖数论、代数、组合、几何四大领域。

### 关键设计

1. **双范式评估统一**：

    - 功能：同时评估结果正确性和推理过程质量
    - 核心思路：计算题用sympy规则验证答案（可扩展、客观但不评估推理质量）；形式化证明题用Lean 4验证器检查证明过程（严格、可审计但需要专业形式化能力）。两种范式互补覆盖
    - 设计动机：仅答案验证无法发现"启发式猜测"行为（模型不通过严谨推导直接猜答案），形式化验证弥补这一盲点

2. **数据泄露防护**：

    - 功能：确保评估结果反映真实推理能力
    - 核心思路：所有题目从印刷出版物（专业杂志和教科书）手动收集，刻意排除网络来源。n-gram泄露分析显示OlymMATH的泄露指标显著低于PolyMath
    - 设计动机：现有基准（Omni-MATH从AoPS爬取、PolyMath直接使用AIME/CNMO）面临严重泄露风险

3. **双语平行评估**：

    - 功能：揭示模型在不同语言上的推理差异
    - 核心思路：每道题都有英文和中文两个版本，保持语义等价。实验发现一致的中英文性能差距
    - 设计动机：多语言推理能力是LLM部署的现实需求，但此前缺少系统性评估

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

## 相关论文

- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](../../ICLR2026/llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)

<!-- RELATED:END -->
