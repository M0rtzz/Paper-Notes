---
title: >-
  [论文解读] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics
description: >-
  [ICLR 2026][LLM推理][数学推理] 提出 ContextMATH 基准，通过将 AIME/MATH-500 抽象数学题转化为情景嵌入（SG）和复杂度缩放（CS）两种变体，揭示即使是 GPT-5 和 DeepSeek-R1 等顶级模型在上下文数学推理中也出现 13-34% 的准确率下降…
tags:
  - "ICLR 2026"
  - "LLM推理"
  - "数学推理"
  - "上下文推理"
  - "问题建模"
  - "benchmark"
  - "LLM 评估"
  - "AIME"
---

# From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

**会议**: ICLR 2026  
**arXiv**: [2601.23048](https://arxiv.org/abs/2601.23048)  
**代码**: 未公开  
**领域**: LLM推理  
**关键词**: 数学推理, 上下文推理, 问题建模, benchmark, LLM 评估, AIME

## 一句话总结

提出 ContextMATH 基准，通过将 AIME/MATH-500 抽象数学题转化为情景嵌入（SG）和复杂度缩放（CS）两种变体，揭示即使是 GPT-5 和 DeepSeek-R1 等顶级模型在上下文数学推理中也出现 13-34% 的准确率下降，且错误主要由问题建模（formulation）而非计算推理导致。

## 研究背景与动机

LLM 在数学 benchmark 上已接近满分（AIME、MATH-500），甚至达到 IMO 金牌水平。但这些成功局限于**格式规整的抽象问题**——直接给出方程和条件。

现实世界中的数学应用（金融分析、科学研究、工程设计）很少以现成方程呈现，通常需要从具体叙事场景中**提取数学核心再求解**。这种能力被作者定义为**上下文数学推理（Contextual Mathematical Reasoning）**。

现有 benchmark 几乎全部聚焦抽象问题（GSM8K、MATH、AIME），即使包含简单叙事（"Jack had 8 pens..."）也是浅层的。这留下一个关键问题：**LLM 在抽象 benchmark 上的强劲表现能否迁移到情景化的、需要建模的数学问题中？**

收集真实世界数学问题成本高昂且难以规模化，因此作者采用**受控转换策略**——基于已有 benchmark（保证正确性），系统地将每个问题转化为上下文变体。

## 方法详解

### 整体框架

ContextMATH 基于 AIME 2024、AIME 2025 和 MATH-500（仅保留难度 ≥3 的题目），将每个原题转化为两种变体：

1. **Scenario Grounding (SG)**：将抽象数学结构嵌入具体叙事，但不增加推理复杂度
2. **Complexity Scaling (CS)**：将显式条件隐藏在子问题中，需额外推理步骤才能恢复原始条件

### 关键设计

**SG 构建**：通过多步提示引导 LLM（o1-mini）将所有抽象数学元素映射到现实实体（如"变量 $x$"→"初始油桶数"），然后定义实体间的交互规则。保证数学核心不变，仅增加叙事语境。

**CS 构建**：将直接条件编码为简单自包含子问题的输出。策略包括：
- 将数值编码为数论/组合问题的解（如"25 个指示灯"→"指示灯唯一配对数恰好为 300"）
- 用需从数据点确定的变量替代显式函数/常数
- 将几何关系改述为物理/结构描述

**质量控制**：三位专家（计算机科学高级学位 + 竞赛数学背景）独立审核每个题目：
- 评估叙事合理性和清晰度
- 独立从场景中建模抽象数学问题验证等价性
- 在 Gemini 和 GPT-5 上测试
- 不一致时由最强数学背景的审核员主持讨论

SG 和 CS 平均长度分别为 133 和 176 词，在现有 LLM 处理能力范围内。

### 建模分析框架

除准确率外，定义三个指标评估问题建模能力：

**建模准确率**：模型正确将场景翻译为数学公式的比率

**建模必要性（Necessity）**：

$$P(F=\text{True} \mid R=\text{True})$$

即正确推理在多大程度上依赖正确建模。

**建模充分性（Sufficiency）**：

$$P(R=\text{True} \mid F=\text{True})$$

即正确建模在多大程度上导致正确推理。

### 训练策略

训练实验基于 Qwen3-Base 系列，三种训练设置：
- $\text{SFT}_{\text{Ori}}$：仅原始数据（50k）
- $\text{SFT}_{\text{Syn}}$：仅合成场景数据（50k）
- $\text{SFT}_{\text{Mix}}$：两者混合（100k）

同时探索了训练专用建模模型的方案。

## 实验关键数据

### 主实验

**顶级闭源模型在 AIME 上的表现（单次评估准确率 %）**：

| 模型 | AIME24 Ori | AIME24 SG | AIME24 CS | AIME25 Ori | AIME25 SG | AIME25 CS |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| DeepSeek-R1 | 93.3 | 70.0 (-25%) | 66.7 (-29%) | 86.7 | 73.3 (-15%) | 53.3 (-38%) |
| GPT-5 | 90.0 | 83.3 (-7%) | 80.0 (-11%) | 90.0 | 80.0 (-11%) | 66.7 (-26%) |
| Gemini 2.5 Pro | 83.3 | 73.3 (-12%) | 76.7 (-8%) | 83.3 | 56.7 (-32%) | 50.0 (-40%) |
| o3 | 83.3 | 70.0 (-16%) | 66.7 (-20%) | 76.7 | 70.0 (-9%) | 60.0 (-22%) |
| QwQ-plus | 86.7 | 56.7 (-35%) | 46.7 (-46%) | 73.3 | 53.3 (-27%) | 43.3 (-41%) |

**开源模型（16 样本平均准确率 %）**：

| 模型 | AIME24 Ori | AIME24 SG | AIME24 CS | AIME25 SG | AIME25 CS |
|------|-----------|-----------|-----------|-----------|-----------|
| Qwen3-32B | 81.2 | 67.9 (-16%) | 57.1 (-30%) | 54.4 (-22%) | 45.0 (-36%) |
| Qwen3-8B | 73.8 | 61.5 (-16%) | 42.9 (-42%) | 48.3 (-25%) | 35.8 (-45%) |
| Qwen3-4B | 70.4 | 52.5 (-25%) | 34.6 (-51%) | 39.6 (-38%) | 33.8 (-47%) |
| AReaL-boba-2-32B | 81.5 | 65.4 (-20%) | 58.3 (-29%) | 55.0 (-29%) | 43.8 (-43%) |

平均而言，开源模型在 SG 上下降 13%，在 CS 上下降 34%；闭源模型分别下降 13% 和 20%。

### 消融实验

**建模能力分析（关键模型）**：

| 模型 | 建模准确率 Avg | 建模必要性 Avg | 建模充分性 Avg |
|------|--------------|--------------|--------------|
| Qwen3-0.6B | 42.8 | 56.1 | 13.5 |
| Qwen3-4B | 61.6 | 79.2 | 61.3 |
| Qwen3-8B | 73.8 | 83.8 | 60.7 |
| Qwen3-32B | 75.0 | 81.9 | 64.9 |
| GPT-5 | 81.4 | 85.6 | 82.7 |

**训练实验（Qwen3-14B-Base，平均准确率 %）**：

| 设为 | Average |
|------|---------|
| Base | 29.4 |
| + SFT_Ori | 55.5 (+26.1%) |
| + SFT_Syn | 60.4 (+31.0%) |
| + SFT_Mix | **61.3** (+31.9%) |

**训练专用建模模型失败**：

| 推理模型 | 无建模 | 未调建模 8B | 调优建模 8B |
|---------|-------|-----------|-----------|
| Qwen3-8B | 53.9 | 48.9 | 20.8 |
| Qwen3-14B | 57.7 | 51.8 | 21.8 |

训练后的建模模型性能反而崩溃，从 scenario-original 配对数据中难以有效学习建模能力。

### 关键发现

1. **上下文复杂性是普遍瓶颈**：即使 GPT-5 在 AIME25-CS 上也下降 26%
2. **规模缓解但不解决问题**：1.5B 下降 77% vs 32B 下降 29%（CS），但差距仍然显著
3. **错误分析：建模错误占 ~80%**，远超计算、逻辑等其他类型
4. **建模是必要条件**：必要性一致高于准确率（Qwen3-8B: 83.8% vs 73.8%）
5. **建模非充分条件**：充分性滞后于必要性，即使 GPT-5 也仅 82.7%
6. **后续 RL 专门化可能有害**：进一步 SFT/RL 提升了原始题得分但加大了上下文下降
7. **场景数据训练有效但不够**：SFT_Mix 最优但仍有大量未解决的差距

## 亮点与洞察

1. **benchmark 设计理念优秀**：SG 和 CS 形成递进式探针，分离"语境理解"和"条件恢复"两种能力
2. **三层定量分析框架**（准确率-必要性-充分性）清晰地刻画了建模与推理的双重瓶颈
3. **揭示了一个反直觉现象**：专门的 RL 后训练可能过度拟合规范格式，削弱上下文推理
4. **负面结果同样有价值**：训练专用建模模型失败，说明建模能力不可从配对数据简单学习
5. **评估规模宏大**：61 个模型（46 开源 + 15 闭源），包括 GPT-5

## 局限性 / 可改进方向

1. **基准规模有限**：基于 AIME（30 题/年）和 MATH-500 子集，数据量较小
2. **构建依赖 LLM + 人工审核**：难以大规模扩展
3. **MATH-500 未构建 CS 变体**：部分简单题不适合进一步转换
4. **闭源模型仅单次评估**：API 限制导致无法多次采样
5. 可扩展到其他领域（物理、经济学）的上下文推理评估
6. 可探索训练时同时暴露抽象和上下文变体的课程学习策略

## 相关工作与启发

- **GSM8K/MATH/AIME**：ContextMATH 直接基于这些 benchmark 构建上下文变体
- **Math-Perturb** (Huang et al., 2025)：改变表面参数测试泛化，ContextMATH 更深层——改变呈现方式
- **SWE-bench/WebArena**：其他领域的真实场景评估，ContextMATH 是数学领域的类似尝试
- 启发：抽象能力 ≠ 应用能力，这一差距在数学领域尤为突出

## 评分

- **新颖性**: ⭐⭐⭐⭐ — SG/CS 双维度设计和建模分析框架新颖
- **技术深度**: ⭐⭐⭐⭐ — 必要性/充分性分析框架严谨
- **实验充分性**: ⭐⭐⭐⭐⭐ — 61 模型评估 + 训练实验 + 建模分析，极其全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，insight 凝练
- **实用价值**: ⭐⭐⭐⭐ — 对评估和训练 LLM 数学能力有直接指导意义
- **综合推荐**: ⭐⭐⭐⭐⭐ (4.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](../../ACL2026/llm_reasoning/chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ICML 2026\] Biases in the Blind Spot: Detecting What LLMs Fail to Mention](../../ICML2026/llm_reasoning/biases_in_the_blind_spot_detecting_what_llms_fail_to_mention.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](../../ICML2026/llm_reasoning/how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)

</div>

<!-- RELATED:END -->
