---
title: >-
  [论文解读] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs
description: >-
  [ICLR 2026][LLM推理][几何推理] 形式化Program-to-Geometry任务并提出GeoGramBench（500题），按三级几何复杂度分类法评估19个前沿LLM从过程式绘图代码构建几何表征并推理的能力，发现即使GPT-5在最高抽象级别也仅39.26%准确率，揭示了LLM空间抽象的根本性短板。
tags:
  - ICLR 2026
  - LLM推理
  - 几何推理
  - 程序转几何
  - benchmark
  - 空间推理
  - Asymptote代码
---

# GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs

**会议**: ICLR 2026  
**arXiv**: [2505.17653](https://arxiv.org/abs/2505.17653)  
**代码**: [GitHub](https://github.com/LiAuto-DSR/GeoGramBench)  
**领域**: llm_reasoning  
**关键词**: 几何推理, 程序转几何, benchmark, 空间推理, Asymptote代码

## 一句话总结

形式化Program-to-Geometry任务并提出GeoGramBench（500题），按三级几何复杂度分类法评估19个前沿LLM从过程式绘图代码构建几何表征并推理的能力，发现即使GPT-5在最高抽象级别也仅39.26%准确率，揭示了LLM空间抽象的根本性短板。

## 研究背景与动机

**领域现状**：空间推理是人类认知和AI的基础能力，支撑机器人、自动导航、自动设计等应用。LLM在解释几何变换和空间关系方面引起广泛关注，但从过程式代码进行几何推理的能力被严重忽视。

**现有痛点**：既有benchmark（如MathVerse, GeoSense, Euclid）聚焦视觉几何理解；MATH-500和AIME24虽包含少量Asymptote代码题，但缺乏系统性的Program-to-Geometry评测。更关键的是，现有基准未识别到代码中的**答案泄露**问题——代码参数可直接或间接暴露答案。

**核心矛盾**：初步研究表明LLM在从代码到空间推理的过渡中存在显著性能下降。DeepSeek-R1在含Asymptote代码的几何题（$\mathbb{P}_{TC}$）上比纯文本题（$\mathbb{P}_T$）准确率骤降23.5%（AIME24）和10.9%（MATH-500）。GPT-o1和QwQ-32B也呈现类似趋势。

**本文方案**：形式化Program-to-Geometry任务定义，提出GeoGramBench（500题精心策划的几何题含过程式绘图代码），配套三级几何复杂度分类法而非传统推理难度分类。

## 方法详解

### 整体框架

**任务定义**：给定文本描述和几何绘图代码（Asymptote/Matplotlib），模型需解析代码构建内部几何表征，并基于该表征进行推理得到数值答案（长度/面积/体积/角度/比例/计数）。

**分类法**基于几何复杂度而非推理步骤：
1. **Primitive Recognition（基元识别）**：1-2个几何基元（点/线/弧/圆/多边形），聚焦长度/面积/角度等基本属性
2. **Local Relation Composition（局部关系组合）**：多个局部几何元素，需识别和整合子组件间的空间关系
3. **Global Abstract Integration（全局抽象整合）**：涉及空间方向/参数化/递归/3D对象/复合结构/高级几何操作（旋转/折叠/投影）

### 关键设计1: 答案泄露防护

**功能**：确保模型无法通过检查代码直接获得答案，必须真正进行几何推理。

**核心思路**：识别两类泄露——
- **直接泄露**：答案显式编码为坐标值（如圆半径、线段长度），处理方法为重缩放坐标同时保持几何形状
- **间接泄露**：答案可从代码参数或公式计算得出，处理方法为修改或遮蔽关键代码参数

**设计动机**：在MATH-500中发现大量Asymptote代码直接包含答案的情况，不解决此问题将使评估失效。经4位专家（硕士及以上数学背景）两轮人工验证确保每道题的答案不可从代码检查获得。

### 关键设计2: 基于几何复杂度的分类法验证

**功能**：证明几何复杂度而非推理步骤数是Program-to-Geometry任务的主要挑战。

**核心思路**：在QwQ-32B上对比MATH-500的推理复杂度（按MATH-500标注）和几何复杂度：
- 纯文本题（$\mathbb{P}_T$）：准确率随推理复杂度增加而下降——符合传统benchmark
- 含代码题（$\mathbb{P}_{TC}$）：准确率**与推理复杂度基本无关**，但随几何复杂度增加显著下降

**设计动机**：传统基于推理步骤的分类法（高中→竞赛级）不适用于此任务。几何复杂度分类法能更准确捕获模型瓶颈。

### 数据构建流程

从3个开源数学数据集（NuminaMath-1.5, HARP, Omni-MATH）聚合约905K候选题 → 过滤含Asymptote代码的9,260题 → n-gram去重至1,782题 → GPT-4o筛选几何题得1,247题 → 两轮人工验证（格式规范化+质量提升：去污染/答案泄露防护/准确性校验）→ 392题 → 增补AIME24(5题)/MATH-500(42题)/Mathverse(61题固体几何手写matplotlib代码) → **最终500题**。

## 实验关键数据

### 主实验：19个LLM在GeoGramBench上的表现

| 模型 | Primitive | Compositional | Abstract | 总平均 |
|------|-----------|--------------|----------|-------|
| **GPT-5** | **90.44%** | **84.59%** | 39.26% | **75.01%** |
| Qwen3-235B-Think | 89.09% | 79.12% | **49.05%** | 74.00% |
| GPT-o1 | 85.92% | 76.12% | 44.67% | 70.92% |
| GPT-o3-mini | 83.49% | 76.10% | 42.67% | 70.00% |
| DeepSeek-R1 | 84.68% | 75.13% | 40.86% | 69.17% |
| QwQ-32B | 85.17% | 73.12% | 37.92% | 67.12% |
| GPT-4o | 40.02% | 21.36% | 4.51% | 21.40% |
| DeepScaleR-1.5B | 65.44% | 47.89% | 15.76% | 43.83% |

**所有模型在Abstract级别均低于50%**，GPT-5仅39.26%。

### 消融实验：绘图语言影响

| 基准 | Asymptote代码 | Matplotlib代码 | 差异 |
|------|-------------|---------------|------|
| AIME24 (QwQ-32B) | ~X% | ~X% | < 1% |
| MATH-500 (QwQ-32B) | ~X% | ~X% | < 1% |

绘图语言的选择几乎不影响性能，瓶颈在于空间抽象而非代码语法理解。

### 关键发现

- **最难子类型**：Primitive/Compositional级别中角度题最难（需重建和推理隐式空间关系）；Abstract级别中面积和体积最难（需完整3D空间理解）
- **CoT推理效果有限**：Token Budget Forcing增加77.4%的token数（10,544→18,710）仅提升0.30%准确率（54.60%→54.90%），说明瓶颈不在推理长度而在空间表征构建能力
- **领域数据微调饱和效应**：添加100个GeoGramBench训练样本可提升3.02%，但从100增至300样本仅额外提升0.58%，瓶颈在模型架构而非数据量
- **常见失败模式**：（1）偏好代数方法而非几何构造；（2）极少引入辅助线/点；（3）空间方向（顺/逆时针）判断困难；（4）符号-几何元素映射混淆

## 亮点与洞察

- 首次形式化Program-to-Geometry任务并构建专用大规模benchmark
- 几何复杂度分类法的验证实验极具说服力——证明此任务的难度来源与传统数学推理不同
- 答案泄露问题的识别和系统性防护是重要贡献，提升了评估的有效性
- 行为分析（RQ1-3）提供了对LLM内部几何推理机制的深入洞察
- 假设的"多阶段内部几何表征过程"（附录H）为未来研究提供了有价值的框架

## 局限与展望

- 仅覆盖2D和简单3D几何，未涉及真实世界3D场景
- 失败模式分析主要基于定性观察，缺乏自动化的系统性诊断工具
- 500题规模虽为最大Program-to-Geometry评测集，但各子类型分布不均（Volume仅27题）
- 仅测试zero-shot设置，未探索few-shot或in-context learning的潜力
- 微调实验仅使用s1.1-32B单一模型

## 相关工作与启发

- **SGP-Bench**（Qiu et al., 2024）和**SVGenius**（Chen et al., 2025）：聚焦SVG代码理解，GeoGramBench进一步关注几何推理而非仅代码解析
- **s1: Simple Test-time Scaling**（Muennighoff et al., 2025）：Token Budget Forcing方法在GeoGramBench上效果有限，说明test-time scaling对空间推理帮助不大
- 对多模态模型设计的启发：当前LLM的空间抽象能力是根本瓶颈，增加数据和推理长度无法解决，需要架构级创新

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个专用Program-to-Geometry评测基准，任务定义清晰且分类法有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 19个模型广泛评测，含行为分析、微调消融、CoT分析、绘图语言对比
- 写作质量: ⭐⭐⭐⭐ 结构完整，研究问题驱动分析，图表丰富清晰
- 价值: ⭐⭐⭐⭐ 揭示LLM空间推理的根本性短板，对未来模型设计有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [TopoBench: Benchmarking LLMs on Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)
- [RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)
- [Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)

<!-- RELATED:END -->
