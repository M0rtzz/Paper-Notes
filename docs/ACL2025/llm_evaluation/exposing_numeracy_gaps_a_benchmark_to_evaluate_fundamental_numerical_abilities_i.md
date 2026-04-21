---
title: >-
  [论文解读] Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models
description: >-
  [ACL2025][numerical reasoning] 提出 NumericBench 综合基准，通过 6 类数据集评估 LLM 的 6 种基本数值能力（数字识别、算术运算、上下文检索、比较、汇总、逻辑推理），发现包括 GPT-4o、DeepSeek-V3 在内的 SOTA 模型在简单数值任务上仍表现极差，并深入分析了 5 种根因。
tags:
  - ACL2025
  - numerical reasoning
  - benchmark
  - arithmetic
  - number understanding
  - LLM evaluation
  - tokenizer
---

# Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models

**会议**: ACL2025  
**arXiv**: [2502.11075](https://arxiv.org/abs/2502.11075)  
**代码**: [TreeAI-Lab/NumericBench](https://github.com/TreeAI-Lab/NumericBench)  
**领域**: llm_nlp  
**关键词**: numerical reasoning, benchmark, arithmetic, number understanding, LLM evaluation, tokenizer

## 一句话总结
提出 NumericBench 综合基准，通过 6 类数据集评估 LLM 的 6 种基本数值能力（数字识别、算术运算、上下文检索、比较、汇总、逻辑推理），发现包括 GPT-4o、DeepSeek-V3 在内的 SOTA 模型在简单数值任务上仍表现极差，并深入分析了 5 种根因。

## 研究背景与动机

**领域现状**：LLM 在文本生成、语义理解甚至奥赛数学上表现出色，但在简单数值任务（基本乘法、数字比较、数字检索）上出人意料地失败。

**现有痛点**：语义类基准（GLUE、SuperGLUE）只测语言能力，数学类基准（GSM8K、MathBench）聚焦结构化代数问题，两者都忽视了真实场景中非结构化数据里的基本数值推理需求。

**核心矛盾**：LLM 依赖表面统计模式，将数字视为离散 token 而非连续量值，导致数值语义理解的根本缺陷。

**本文目标** 构建系统化基准，全面评估 LLM 在 6 种基本数值能力上的表现，暴露其数值推理短板。

**切入角度**：融合合成数据（数字列表、算术运算）和真实世界爬取数据（股票、天气），覆盖从简单识别到多步推理的完整数值能力谱。

**核心 idea**：LLM 在简单数值任务上的系统性失败揭示了 tokenization、训练范式、位置编码等架构层面的根本限制。

## 方法详解

### 整体框架
6 个数据集（数字列表、股票、天气、数列序列、算术运算、混合字符串）× 6 种能力（数字识别、算术、检索、比较、汇总、逻辑推理）→ 12+ 个 SOTA LLM 评测 → 5 种根因分析（tokenizer / 训练语料 / 训练范式 / 位置编码 / Transformer 架构）。

### 关键设计 1：六种基本数值能力的系统化定义
- **功能**：将 LLM 数值能力分解为 6 个互补维度进行独立测试。
- **核心思路**：数字识别（混合字符串中计数数字）、算术运算（加减乘除含小数）、上下文检索（从长序列/结构化数据中定位特定数值）、比较（判断大小关系）、汇总（计算均值/统计趋势）、逻辑推理（识别数列模式并预测下一个值）。
- **设计动机**：现有基准要么过简单要么过复杂，缺少对基础数值操作的细粒度诊断；6 种能力覆盖从感知到推理的完整链路。

### 关键设计 2：真实世界数据集引入噪声与长上下文
- **功能**：爬取东方财富股票数据（18 个属性）和 Open-Meteo 天气数据，构建含噪声的长上下文评测。
- **核心思路**：平均 token 数达 27K+；通过添加 k∈{2,4,6} 个无关属性模拟噪声场景；短/长上下文分别评测以衡量长距离依赖能力。
- **设计动机**：真实场景中数值常嵌入在大量无关信息中，纯合成数据无法反映这种复杂性。

### 关键设计 3：算术运算的双问法与位数递增
- **功能**：设计 $Q_{oper}$（符号表达如 a+b）和 $Q_{context}$（自然语言如"a 加 b"）两种问法；数字位数从 1 到 6 递增。
- **核心思路**：12000 对数字 × 4 种运算 × 2 种问法，系统测试算术精度随位数增长的衰减。
- **设计动机**：揭示 LLM 的"从高位到低位"生成模式与实际算术"从低位到高位"进位逻辑的根本冲突。

### 损失函数
本文为评测基准，不涉及训练损失函数。评估指标统一使用 **Accuracy**，其中算术数据集答案保留两位小数，其余数据集采用单选题（8 选 1，随机基线 12.5%）。

## 实验关键数据

### 主实验（检索/比较/汇总/逻辑推理，Table 2 精选）

| 模型 | 数字列表-检索 | 股票-检索 | 天气-比较 | 数列-逻辑推理 |
|------|-------------|----------|----------|-------------|
| Random | 12.5 | 12.5 | 12.5 | 12.5 |
| Llama-3.1-8B | 22.8 | 14.4 | 13.7 | 18.2 |
| Llama-3.3-70B | 44.4 | 19.4 | 35.8 | 18.6 |
| DeepSeek-V3 | 47.2 | **47.5** | 35.8 | 15.8 |
| GPT-4o | 41.7 | 37.5 | **64.2** | 14.6 |
| o3-mini | **96.8** | 68.6 | 83.9 | **66.4** |
| DeepSeek-R1 | 73.6 | **81.3** | **98.8** | 65.4 |
| Human | 100 | 100 | 100 | 52.6 |

### 消融实验（CoT 对 Llama-3.3-70B 的影响，Table 4）

| 方法 | 数字列表-检索 | 数字列表-比较 | 股票-检索 | 股票-比较 |
|------|-------------|-------------|----------|----------|
| Base | 44.4 | 31.5 | 19.4 | 13.8 |
| Plain-CoT | 65.2 | 39.4 | 24.8 | 27.7 |
| PS-CoT | 65.4 | 40.0 | 24.3 | 16.7 |
| Table-CoT | 65.8 | 38.4 | 27.6 | 29.1 |

### 混合字符串数字识别（Table 3）

| 模型 | 50字符 | 100字符 | 150字符 | 200字符 |
|------|--------|---------|---------|---------|
| GPT-4o | 18.2 | 6.4 | 4.0 | 4.2 |
| DeepSeek-V3 | 13.2 | 4.0 | 3.2 | 2.0 |
| Human | 100 | 100 | 100 | 100 |

### 关键发现
1. **普遍失败**：即使是 GPT-4o 在大多数数值任务上也远低于人类水平（检索 41.7 vs 100，汇总 11.6 vs 100）。
2. **推理型模型显著领先**：o3-mini 和 DeepSeek-R1 大幅超越标准 LLM，检索任务达 73-97%，说明 CoT 式推理对数值任务至关重要。
3. **乘法是最难运算**：LLM 在加减除上表现尚可，但乘法准确率极低。
4. **长上下文性能骤降**：几乎所有模型在长上下文中表现显著劣于短上下文。
5. **噪声显著干扰**：添加无关属性后模型性能持续下降。
6. **CoT 帮助有限**：CoT 在简单任务上略有提升，但在复杂股票汇总任务上甚至引入噪声。
7. **SFT 仅对简单数据有效**：QLoRA 微调在数字列表上提升显著（检索 24.4→62.8），但对股票/天气等复杂数据无改善。

## 亮点与洞察
1. **诊断深度出色**：不只报告"LLM 数值差"，而是系统归因到 tokenizer/训练范式/位置编码/架构 5 个层面，每个都给出具体机制解释。
2. **人类基线的意外发现**：逻辑推理任务人类仅 52.6%，说明某些数列模式识别本身就很难，不应苛求 LLM。
3. **推理模型 vs 标准模型的巨大差距**：o3-mini / R1 的表现证明显式推理链是弥补数值能力缺陷的有效路径。
4. **实用导向的数据集设计**：股票和天气数据直接来自真实 API，贴近生产场景。

## 局限性
1. 仅覆盖 6 类数值任务，缺少交通、医疗等更多真实领域。
2. 未评测 Claude 系列和 o1 等模型（因 API 成本约 $15,000）。
3. 算术数据集最大 6 位数，未探索更大数值范围。
4. 数列模式任务人类基线仅 52.6%，说明数据集设计可能偏难或标注存在歧义。
5. 缺少对模型内部表征的分析（如 probing 实验），根因分析主要是推测性的。

## 相关工作与启发

### vs GSM8K / MathBench
GSM8K 测试多步数学应用题，MathBench 覆盖代数/几何等结构化问题；NumericBench 聚焦**更基础的数值操作**（检索、比较、识别），揭示了即使在小学级运算上 LLM 也存在系统性缺陷——这是高阶数学推理的前提条件。

### vs MATH-Perturb / GSM-Symbolic
这些工作通过扰动数学题中的数字证明 LLM 依赖模式匹配而非真正理解；NumericBench 进一步将这一现象扩展到**非数学场景**（股票分析、天气预报），证明数值理解缺陷是通用的而非数学特有的。

### 启发
- Tokenizer 改进（如数值感知分词）可能是提升数值能力成本最低的切入点。
- 从低位到高位的生成顺序（reverse generation）值得探索以解决算术冲突。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个系统化评估 LLM 基本数值能力的综合基准，6 种能力 × 6 种数据集的组合设计有新意
- 实验充分度: ⭐⭐⭐⭐ — 12+ 模型、多场景（噪声/长上下文/CoT/SFT）消融全面；缺少 Claude 等模型稍有遗憾
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题定义精准，根因分析有理有据
- 价值: ⭐⭐⭐⭐⭐ — 揭示了 LLM 一个被严重低估的根本缺陷，对 tokenizer/架构/训练范式改进具有重要指导意义

| 逻辑推理 | ~40% | ~35% | ~25% |

### 原因分析

| 原因 | 影响程度 | 说明 |
|------|---------|------|
| Tokenization | 高 | 数字被拆成多个 token |
| 训练数据 | 高 | 数值训练数据不足 |
| 位置编码 | 中 | 长序列位置偏差 |
| 架构 | 中 | 注意力机制不适合精确计算 |

### 关键发现
- **所有模型在所有任务上都远未达到完美**
- **汇总和逻辑推理是最大弱点**
- **真实数据比合成数据更难**（噪声和上下文干扰）
- **GPT-4o 在简单 3 位数乘法上仍有 ~40% 错误率**

## 亮点与洞察
- **六种能力的系统分类**为数值理解研究提供了完整框架
- **根因分析**将表面症状（任务失败）与底层原因（tokenization/架构）联系起来

## 局限与展望
- 未测试计算器工具调用的缓解效果
- 改进方向：数值感知 tokenization、混合神经-符号方法

## 相关工作与启发
- **vs GSM8K**：GSM8K 测试数学推理，NumericBench 测试更基础的数值操作
- **vs CUTE/EXECUTE**：它们测字符理解，NumericBench 测数值理解——互补方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 六种能力的系统评估 + 根因分析
- 实验充分度: ⭐⭐⭐⭐ 多模型 × 6 数据集 × 真实+合成
- 写作质量: ⭐⭐⭐⭐ 分类清晰
- 价值: ⭐⭐⭐⭐ 对数值感知 LLM 开发有直接指导

<!-- RELATED:START -->

## 相关论文

- [SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](../../NeurIPS2025/llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)

<!-- RELATED:END -->
