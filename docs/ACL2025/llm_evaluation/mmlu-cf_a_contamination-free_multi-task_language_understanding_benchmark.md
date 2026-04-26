---
title: >-
  [论文解读] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark
description: >-
  [ACL 2025][数据污染] 提出 MMLU-CF，一个包含 20,000 道题的无数据污染多任务语言理解基准，通过从更广泛的来源收集数据并应用三条去污染规则（改写题目、打乱选项、随机替换选项）来避免无意和恶意的数据泄露，最强模型 GPT-4o 在该基准上仅获得 73.4%（MMLU 上为 88.0%）。
tags:
  - ACL 2025
  - 数据污染
  - 基准评测
  - MMLU
  - 多任务语言理解
  - 去污染
---

# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark

**会议**: ACL 2025  
**arXiv**: [2412.15194](https://arxiv.org/abs/2412.15194)  
**代码**: 无（测试集闭源，验证集公开）  
**领域**: 其他（基准评测）  
**关键词**: 数据污染, 基准评测, MMLU, 多任务语言理解, 去污染

## 一句话总结

提出 MMLU-CF，一个包含 20,000 道题的无数据污染多任务语言理解基准，通过从更广泛的来源收集数据并应用三条去污染规则（改写题目、打乱选项、随机替换选项）来避免无意和恶意的数据泄露，最强模型 GPT-4o 在该基准上仅获得 73.4%（MMLU 上为 88.0%）。

## 研究背景与动机

MMLU 是评估 LLM 通用知识理解能力的标准基准，但面临严重的数据污染问题：

**问题1：无意泄露**
- LLM 训练数据来源广泛，不可避免地包含 MMLU 等公开基准的数据
- 模型可能通过记忆而非推理来回答问题

**问题2：恶意泄露**
- 由于基准完全公开，存在被故意加入训练数据的风险
- 实验证据：当仅给出 MMLU 的题目时，某些 LLM 能直接给出与原始完全相同的选项，说明模型已经记住了这些题目

**MMLU 难度不足**：
- GPT-4o、Gemini-1.5-Pro、Claude 等前沿模型在 MMLU 上已达到 86%-88%
- 近三分之一题目的难度低于 4（满分 9），大量简单题目是高分的原因
- 评估已到达瓶颈，无法有效区分新一代模型的能力

## 方法详解

### 整体框架

MMLU-CF 的构建经过五步流程：
1. **MCQ 收集（MCQ Collection）**
2. **MCQ 清理（MCQ Cleaning）**
3. **难度采样（Difficulty Sampling）**
4. **LLM 审核（LLMs Checking）**
5. **去污染处理（Contamination-Free Processing）**

最终产出：10,000 道闭源测试题 + 10,000 道开源验证题

### 关键设计

**数据来源多样化**：
- 从超过 2000 亿公开网页文档中提取原始选择题
- 覆盖 3000+ 个不同网站域名
- 远比 MMLU 仅依赖少数来源更加多样
- 涵盖 14 个领域（健康、数学、物理、商业、化学、哲学、法律、工程等）

**数据清理**：
- 从 270 万原始题目经过格式标准化、去重、过滤后缩减到 166 万
- 包括统一选项标签（A/B/C/D）、去除非四选一题、去除低质量短题等

**难度采样**：
- 使用 GPT-4o 对 MMLU 和候选题目进行 0-9 级难度分类
- 采用以难度 6 为中心的正态分布进行采样，确保整体难度高于 MMLU
- 166 万题 → 5 万题

**三模型审核**：
- 使用 GPT-4o、Gemini、Claude 三个 LLM 从质量和安全两方面审核
- 质量维度：上下文清晰度、逻辑一致性、事实准确性、选项互斥性、正确答案存在性
- 安全维度：无仇恨、无色情、无自残、无暴力
- 三模型评分均值 > 4（满分 5）的题目才保留
- 5 万题 → 2 万题

**三条去污染规则**：

1. **Rule 1：改写题目（Rephrase Question）**

    - 减少模型对训练中遇到的题目的依赖
    - 不改变题目含义，但改变表述方式

2. **Rule 2：打乱选项（Shuffle Choices）**

    - 防止模型通过记忆选项顺序来作答
    - "以上都不是"类选项保持在最后

3. **Rule 3：随机替换选项（Random Replace Choices）**

    - 以 50% 概率将某个选项替换为"None of the other choices"
    - 替换正确选项时仍保持有效，但需要更多推理
    - 替换错误选项时作为干扰项

**测试集/验证集设计**：
- 闭源测试集防止恶意污染
- 开源验证集便于独立验证
- 两者难度和类别分布相似
- 若未来验证集与测试集的差距增大，说明验证集正在被污染

### 损失函数 / 训练策略

本文是评测基准论文，不涉及模型训练。评估方法：
- 使用 OpenCompass 平台进行标准化评测
- 支持 5-shot 和 0-shot 评估
- 用户可通过项目主页提交 Hugging Face 模型或 API 进行评测

## 实验关键数据

### 主实验

**40+ 模型在 MMLU-CF 上的表现（5-shot Test %）**：

| 分类 | 模型 | MMLU | MMLU-CF (5-shot) | 下降 |
|------|------|------|------------------|------|
| API | GPT-4o | 88.0 | 73.4 | -14.6 |
| API | GPT-4-Turbo | 86.5 | 70.4 | -16.1 |
| API | GPT-4o-mini | 81.8 | 65.5 | -16.3 |
| Large | Qwen2.5-72B-instruct | 85.3 | 71.6 | -13.7 |
| Large | Llama-3.3-70B-instruct | 86.3 | 68.8 | -17.5 |
| Medium | Qwen2.5-32B-instruct | 83.9 | 69.7 | -14.2 |
| Medium | Phi-4-14B | 84.8 | 67.8 | -17.0 |
| Small | Qwen2.5-7B-instruct | 75.4 | 61.3 | -14.1 |
| Mini | Phi-3.5-mini-3.8B | 69.1 | 57.9 | -11.2 |

所有模型在 MMLU-CF 上的得分显著低于 MMLU，平均下降约 14-17 个百分点。

**验证集与测试集一致性**：
- 绝大多数模型在两个集合上差异 < 1%（$\Delta$ 列）
- 说明两个集合的难度和分布确实相似
- 为未来检测验证集是否被污染提供了基准

### 消融实验

**三条去污染规则的累积效果（0-shot）**：

| 规则组合 | GPT-4o | GPT-3.5 | Llama-3.1-8B |
|----------|--------|---------|--------------|
| 无规则 | 79.8 | 65.3 | 63.8 |
| +Rule 1（改写） | 78.6 | 63.1 | 62.3 |
| +Rule 1+2（+打乱） | 77.9 | 62.8 | 61.8 |
| +Rule 1+2+3（+替换） | 73.4 | 58.2 | 57.1 |

每条规则都进一步降低了模型得分：
- Rule 1 使 GPT-4o 下降 1.2 分，弱模型下降更多
- Rule 3 的效果最大（GPT-4o 下降 4.5 分），说明随机替换选项是最有效的去污染策略

### 关键发现

1. **MMLU 存在严重污染**：所有模型在 MMLU-CF 上的表现显著下降，证实 MMLU 评估结果被数据污染显著抬高
2. **GPT-4o 仍是最强模型**：即使在去污染后依然保持领先，但优势缩小（88% → 73.4%）
3. **参数效率模型表现突出**：Phi-4-14B（67.8%）超越了多个更大模型如 Qwen2-72B（63.7%），展示了架构效率的重要性
4. **Qwen2.5 系列整体表现最佳**：在各个尺寸类别中都名列前茅
5. **0-shot vs 5-shot 差异小**：大多数模型在两种设置下表现接近，说明 MMLU-CF 主要测试真实知识理解而非 few-shot 适应能力

## 亮点与洞察

1. **系统性去污染方案**：区分无意泄露和恶意泄露，分别用规则和闭源策略应对，设计全面
2. **三模型交叉审核**：用 GPT-4o、Gemini、Claude 三个 LLM 联合审核避免单一模型偏差
3. **污染监控机制**：验证集公开+测试集闭源的设计，使得可以通过两者差距的变化来追踪验证集的污染程度
4. **难度校准**：以 MMLU 的难度分布为参照，有意提升整体难度，使基准更有区分力
5. **超大规模数据来源**：从 2000 亿文档中筛选，覆盖 3000+ 域名，极大降低了数据已被纳入训练集的概率

## 局限与展望

- 去污染规则可能意外改变了题目的认知要求（特别是 Rule 3 的随机替换可能改变题目难度分布）
- 闭源测试集虽防止了泄露，但也限制了学术界的独立验证和复现
- 14 个领域的覆盖范围是否足够全面值得商榷（如缺少编程相关领域）
- 依赖 GPT-4o 进行难度分类和质量审核，引入了对特定模型的依赖
- 未分析去污染处理对不同题目类型（事实记忆 vs 推理）的影响差异

## 相关工作与启发

- **MMLU**（Hendrycks et al.）和 **MMLU-Pro**（Wang et al., 2024）是直接的前序工作
- **GSM1K**（Zhang et al., 2024）采用类似策略为 GSM8K 创建无污染版本
- **LiveBench**（White et al., 2024）通过持续更新确保无污染，但成本高
- **LatestEval**（Li et al., 2024）从最新文本动态生成评估题
- 启发思考：基准污染问题的根本解决可能需要动态生成 + 闭源评测的结合

## 评分

- **创新性**：⭐⭐⭐ — 去污染方法较为直观，但系统性执行到位
- **实验完整性**：⭐⭐⭐⭐⭐ — 评测了 40+ 模型，分析全面且细致
- **实用价值**：⭐⭐⭐⭐⭐ — 直接回应了 LLM 评估中的核心痛点，测试集持续维护中
- **写作质量**：⭐⭐⭐⭐ — 结构清晰，数据丰富，图表设计规范

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)

<!-- RELATED:END -->
