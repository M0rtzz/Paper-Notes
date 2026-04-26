---
title: >-
  [论文解读] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models
description: >-
  [ACL 2025][Chinese bias] 提出首个多任务中文偏见评估基准 McBE，包含 4,077 条偏见评估实例（BEI），覆盖 12 种偏见类别和 82 个子类别，通过 5 种评估任务（偏好计算/子类别分类/场景选择/偏见分析/偏见评分）多角度量化 LLM 中的中文偏见，并揭示"参数越大偏见越强"的传统结论可能源于单任务评估的局限性。
tags:
  - ACL 2025
  - Chinese bias
  - benchmark
  - fairness
  - stereotypes
  - multi-task evaluation
---

# McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2507.02088](https://arxiv.org/abs/2507.02088)  
**代码**: [GitHub](https://github.com/) (论文附带数据集和代码)  
**领域**: LLM/NLP  
**关键词**: Chinese bias, benchmark, fairness, stereotypes, multi-task evaluation

## 一句话总结

提出首个多任务中文偏见评估基准 McBE，包含 4,077 条偏见评估实例（BEI），覆盖 12 种偏见类别和 82 个子类别，通过 5 种评估任务（偏好计算/子类别分类/场景选择/偏见分析/偏见评分）多角度量化 LLM 中的中文偏见，并揭示"参数越大偏见越强"的传统结论可能源于单任务评估的局限性。

## 研究背景与动机

**领域现状**：LLM 在各类 NLP 任务中广泛应用，但训练数据中的偏见（如性别与职业关联、地域刻板印象）不可避免地被模型继承，给社会公平带来风险。

**现有痛点**：(1) 绝大多数偏见评估数据集基于英语和北美文化（WinoBias、StereoSet、CrowS-Pairs、BBQ），无法覆盖中国社会特有偏见；(2) 已有中文偏见数据集（CHBias、CBBQ）类别有限，且仅支持单一评估任务（如仅 QA），无法多角度测量偏见。

**核心矛盾**：中国文化中存在大量独特偏见类型（地域偏见、世界观偏见、亚文化偏见等），但缺少针对性的多维度评估工具。

**本文目标** 构建一个覆盖面广、任务多样的中文偏见评估基准，从多个角度系统量化 LLM 的中文偏见。

**切入角度**：提出偏见评估实例（BEI）概念作为基本评估单元，每个 BEI 包含上下文、句子模板、替换词表、子类别、解释和偏见评分六个属性，原生支持 5 种评估任务。

**核心 idea**：多偏见类别 × 多评估任务 = 更全面、更准确的中文偏见评估。

## 方法详解

### 整体框架

McBE 的核心设计是**偏见评估实例（Bias Evaluation Instance, BEI）**。每个 BEI 包含 6 个属性：
- **Context**：提供语境帮助模型理解句子
- **Sentence Template**：包含占位符 `[PLH]` 的句子模板
- **Substitution List**：替换词列表，用于生成不同群体的完整句子
- **Bias Subcategories**：人工标注的偏见子类别
- **Explanation**：偏见的详细解释（先由 3 名标注者独立撰写，再由 ChatGLM 合并，最后由 2 名标注者审核）
- **Bias Score**：偏见严重程度评分（0-10 分，6 名标注者的均值）

全部 4,077 个 BEI 覆盖 12 类偏见、82 个子类别，数据来源包括社交平台（51.85%）、个人经历（35.79%）和其他数据集翻译改编（12.36%）。

### 关键设计

1. **12 种偏见类别 + 82 子类别的全面覆盖**

    - 通用类别：性别（7 子类）、种族（11 子类）、宗教（9 子类）、年龄（8 子类）
    - 中国特色类别：地域偏见（省份/城乡/本外地）、世界观（婚育观/生活观/消费观）、亚文化（外观/爱好/个人属性）
    - 类别依据：部分基于中国劳动法和残疾人保护法中的受保护群体，部分基于社会多样性需求
    - 设计动机：区分文化差异（中性事实）与偏见（歧视性表达），确保评估对象是真正的偏见

2. **5 种评估任务的多维度测量**

    - **Task 1: 偏好计算（Preference Computation, PC）**：计算模型对替换词列表中各句子的负对数似然（NLL），通过 NLL 方差衡量偏好差异，方差=0 表示无偏见，使用指数衰减函数映射到 0-100 分
    - **Task 2: 子类别分类（Subcategory Classification, SC）**：让模型判断句子属于哪个偏见子类别，正确率 × 100 即为得分
    - **Task 3: 场景选择（Scenario Selection, SS）**：从 BEI 生成句子对，让模型判断哪句更可能发生，统计选择频率的方差
    - **Task 4: 偏见分析（Bias Analysis, BA）**：模型阅读句子并分析其中的偏见，由 GLM4-AIR 作为评委从准确性/潜在含义/文化差异/亮点 4 个维度带权打分
    - **Task 5: 偏见评分（Bias Scoring, BS）**：模型为句子的偏见程度打分，计算与人类标注分数的平均绝对差，差值越小得分越高
    - 设计动机：PC 和 SS 测量模型的内在偏好，SC/BA/BS 测量模型对偏见的理解和价值对齐

### 损失/评分函数

- PC/SS 任务使用指数衰减函数：$\text{Score} = 100 \cdot e^{-r \cdot V}$，其中 $r = \frac{2e}{3}$，$V$ 为 NLL 方差
- BA 任务使用加权评分：$\text{Final Score} = \frac{\sum_{i=1}^{4} s_i \cdot w_i}{\sum_{i=1}^{4} w_i}$（准确性权重 3.5，潜在含义 1.5，文化差异 2.5，亮点 0.5）
- BS 任务使用均绝对差：$\text{Final Score} = 100 - 10 \cdot \frac{1}{n}\sum_{i=1}^{n}|d_i|$

## 实验关键数据

### 主实验 -- 跨偏见类别的模型表现

| 模型 | 宗教 | 地域 | 国籍 | 种族 | 整体趋势 |
|------|------|------|------|------|---------|
| InternLM2.5-7B | 较高 | 较高 | 中等 | 较低 | 7B 模型中最优 |
| Qwen2.5-32B | 较高 | 较高 | 中等 | 较低 | 大参数显著提升 |
| GLM4-AIR/0520 | 中等 | 中等 | 较低 | 较低 | 低于部分 7B 模型 |
| Llama2-7B-hf | PC/SS 高 | — | — | — | SC/BA/BS 极低 |
| Mistral-7B | — | — | — | — | 优于 Llama2 但趋势类似 |

- 所有模型在**宗教和地域**类别得分最高（偏见最小），在**国籍和种族**类别得分最低（偏见最大）

### 参数规模实验 -- Qwen2.5 系列 (0.5B → 1.5B → 7B → 32B)

| 参数量 | SS 得分 | SC/BA/BS 趋势 | 解读 |
|--------|---------|-------------|------|
| 0.5B | 87.69 | 最低 | SS 高分源于随机选择 |
| 1.5B | 80.49 | 中低 | 0.5B→1.5B 提升最大 |
| 7B | 77.82 | 中高 | 边际收益递减 |
| 32B | 77.11 | 最高 | SS 低但理解力最强 |

### 关键发现

1. **传统结论的反转**：CBBQ、Rubia 等仅用 SS 类任务得出"参数越大偏见越强"的结论，但 McBE 的 SC/BA/BS 任务表明，小模型的 SS 高分来自随机选择而非真正的无偏见，大模型在偏见理解和价值对齐上表现更好
2. **文化特异性**：多语言模型（Llama2、Mistral）在 PC/SS 上表现尚可但在 SC/BA/BS 上严重不足，说明以英语为主训练的模型难以理解中文文化偏见
3. **GLM4 系列反常**：尽管参数更大，GLM4-AIR/0520 得分低于部分 7B 模型，暗示训练数据中偏见内容较多

## 亮点

- **首创 BEI 概念**：偏见评估实例是原子化的评估单元，天然支持多任务，可扩展到其他语言
- **挑战传统认知**：通过多任务评估揭示"大模型偏见更强"的结论可能是单一评估范式的假象
- **标注严谨**：30 名多背景标注者、1:1 性别比、跨省籍、社会学专家参与，较 CHBias（3人）和 IndiBias（5人）显著更具代表性
- **12 类 82 子类别**的覆盖广度在中文偏见基准中无出其右

## 局限性

- **PC 任务不适用于黑盒模型**：偏好计算依赖 NLL 概率分布，对 API-only 模型（如 GPT-4）无法使用
- **偏见定义的主观性**：尽管有专家审核，偏见边界（尤其是偏见 vs 文化差异）仍存在主观判断空间
- **静态数据集**：社会偏见随时间演变，数据集需要动态更新机制
- **BA 任务依赖 LLM 评委**：GLM4-AIR 作为评委可能引入自身偏见

## 相关工作

- **vs CrowS-Pairs**：CrowS-Pairs 覆盖 9 类英文偏见、仅支持反事实输入评估；McBE 覆盖 12 类中文偏见、5 种评估任务，且针对中国文化定制了地域/世界观/亚文化等类别
- **vs CBBQ**：CBBQ 是 BBQ 的中文版，仅支持 QA 评估任务；McBE 引入 BEI 概念和 5 种任务，能多角度测量偏见，并揭示了单任务评估的局限性
- **vs CEB**：CEB 提出组合式偏见分类法但依赖 Perspective API 评分，对 API 未覆盖的偏见无效；McBE 使用人工标注和多任务评估，不依赖外部 API

## 评分

- 新颖性: ⭐⭐⭐⭐ 首创 BEI + 5 任务范式，对"大模型偏见更强"的传统结论提出有力反驳
- 实验充分度: ⭐⭐⭐⭐ 多系列多参数量模型、12 类别 × 5 任务的全矩阵评估
- 写作质量: ⭐⭐⭐⭐ 分类体系清晰，偏见 vs 文化差异的界定有深度
- 价值: ⭐⭐⭐⭐ 对中文 LLM 公平性研究有重要基准意义，BEI 概念可推广到其他语言

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)

<!-- RELATED:END -->
