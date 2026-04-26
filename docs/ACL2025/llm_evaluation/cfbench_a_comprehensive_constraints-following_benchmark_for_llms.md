---
title: >-
  [论文解读] CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
description: >-
  [ACL 2025][约束遵循] 提出 CFBench——一个包含 1000 条精标样本、覆盖 200+ 真实场景和 50+ NLP 任务的中文大规模约束遵循基准，系统性地定义了 10 大类 25+ 子类的约束分类体系，并设计结合约束满足率（CSR）、指令满足率（ISR）和需求优先级满足率（PSR）的多维评估框架，揭示当前顶级 LLM 在约束遵循方面仍存在显著提升空间。
tags:
  - ACL 2025
  - 约束遵循
  - 指令遵循
  - 中文基准
  - 多维评估
  - 约束分类系统
---

# CFBench: A Comprehensive Constraints-Following Benchmark for LLMs

**会议**: ACL 2025  
**arXiv**: [2408.01122](https://arxiv.org/abs/2408.01122)  
**代码**: 未公开（论文称将发布）  
**作者**: Tao Zhang, Chenglin Zhu, Yanjun Shen, Wenjing Luo, Yan Zhang, Hao Liang, Tao Zhang, Fan Yang, Mingan Lin, Yujing Qiao, Weipeng Chen, Bin Cui, Wentao Zhang, Zenan Zhou
**机构**: Baichuan Inc., 北京大学
**领域**: LLM评测 / 指令遵循  
**关键词**: 约束遵循, 指令遵循, 中文基准, 多维评估, 约束分类系统

## 一句话总结

提出 CFBench——一个包含 1000 条精标样本、覆盖 200+ 真实场景和 50+ NLP 任务的中文大规模约束遵循基准，系统性地定义了 10 大类 25+ 子类的约束分类体系，并设计结合约束满足率（CSR）、指令满足率（ISR）和需求优先级满足率（PSR）的多维评估框架，揭示当前顶级 LLM 在约束遵循方面仍存在显著提升空间。

## 研究背景与动机

**领域现状**：LLM 在真实应用中需要理解并遵循用户指令中的各种约束条件（格式、字数、风格、内容等），现有评估基准主要关注碎片化约束或有限场景。

**现有方法的不足**：
   - IFEval 仅关注 25 种可程序验证的指令，泛化性不足
   - FollowBench 通过增加约束数量来提升难度，但仅覆盖 5 种约束类型、75 个实例，数据量有限
   - ComplexBench 关注约束组合关系，但只设计了 4 种类型
   - 缺乏从用户多维度视角出发的评估方法，评估标准与用户感知不一致

**核心问题**：
   - **Q1**: 如何构建高质量、全面覆盖的评估数据？
   - **Q2**: 如何从用户视角进行精细、准确的评估？

## 方法详解

### 整体构建流程

CFBench 的构建包含三大阶段：约束系统构建 → 数据集组装 → 多维评估方法设计。

### 约束分类系统

通过对数百万真实在线指令进行挖掘、过滤和聚类，提炼出 5000+ 原子约束，由领域专家综合分类学和统计学原则整理为 **10 大类 25+ 子类**：

| 编号 | 约束类型 | 说明 | 子类示例 |
|------|----------|------|----------|
| C1 | 内容约束 | 控制输出范围和深度 | 词汇约束、元素约束、语义约束 |
| C2 | 数值约束 | 长度和数量要求 | 词级、句级、段级、篇级 |
| C3 | 风格约束 | 赋予独特风格和格调 | 语气、正式度、受众、作者风格 |
| C4 | 格式约束 | 规范化表达 | 基础格式、定制格式、专业场景 |
| C5 | 语言约束 | 控制语言内部特征 | 语用、句法、形态、音韵 |
| C6 | 情境约束 | 通过背景参数引导回复 | 角色、任务、复杂上下文 |
| C7 | 示例约束 | 利用有限样本中的模式 | 上下文约束学习 |
| C8 | 反向约束 | 通过间接排除缩小空间 | 排除性指令 |
| C9 | 矛盾约束 | 互斥条件 | 常见于在线日志中被忽略 |
| C10 | 规则约束 | 定义逻辑流程或动作 | 条件逻辑、流程控制 |

### 数据集构建

1. **数据源与筛选**：从真实场景和 NLP 任务中收集初始指令，利用 LLM 评估约束类型和数量，过滤不合理约束，平衡场景和类型分布，最终得到 2000 条候选指令
2. **迭代精炼**：专业标注人员反复审核和修改，确保约束合理性和黄金答案质量，每个样本包含高质量指令、理想答案、评估标准（checklist）、约束类型和优先级
3. **最终规模**：1000 条样本 = Easy Set (500) + Hard Set (500)

### 数据统计

| 指标 | Easy Set | Hard Set | Full Set |
|------|----------|----------|----------|
| 平均指令长度 | 413 | 605 | 509 |
| 平均主需求数 | 1.69 | 1.98 | 1.84 |
| 平均约束数 | 3.59 | 4.89 | 4.24 |
| 平均约束类型数 | 2.83 | 3.58 | 3.20 |

### 评估方法

#### 评估标准
将复杂指令拆解为多个简单、独立的 **checkpoints**（checklist），标注约束类型和优先级，使用 GPT-4o 逐项判定。

#### 三个评估指标

1. **CSR（约束满足率）**：各指令中约束满足比例的平均值——反映约束级别的性能
2. **ISR（指令满足率）**：完全满足所有约束的指令占比——反映指令级别的严格性
3. **PSR（需求优先级满足率）**：引入主/次需求优先级的加权评分
    - 当所有主要需求满足时，score = 0.5 + 0.5 × A（A 为次要需求平均分）
    - score > 0.8 时 PSR_i = 1，否则为 0
    - 任何主要需求未满足则 PSR_i = 0

## 实验

### 评估设置

评估了 50+ 个主流模型，涵盖 API 模型和开源模型，推理最大长度设为 2048，使用 GPT-4o 作为评估模型（temperature=0）。

### 主实验结果

| 模型 | CSR(Full) | ISR(Full) | PSR(Full) | PSR(Hard) |
|------|-----------|-----------|-----------|-----------|
| DeepSeek-R1 | 0.908 | 0.699 | 0.783 | 0.672 |
| DeepSeek-V3 | 0.890 | 0.648 | 0.740 | 0.616 |
| GPT-4o | 0.886 | 0.653 | 0.735 | 0.582 |
| o1-preview | 0.870 | 0.634 | 0.718 | 0.592 |
| Claude-3.5-Sonnet | 0.871 | 0.626 | 0.723 | 0.564 |
| Qwen2-72B-Instruct | 0.867 | 0.589 | 0.705 | 0.530 |
| GLM-4-0520 | 0.862 | 0.596 | 0.694 | 0.536 |
| Llama-3-8B-Instruct | 0.609 | 0.211 | 0.297 | 0.238 |

**关键发现**：
- DeepSeek-R1 在各指标上均排名第一，CSR 达 0.908
- 即便最强模型，Hard Set 上 PSR 仅 0.672，说明复杂约束仍是挑战
- CSR 与 ISR/PSR 的差距巨大（例如 GPT-4o CSR=0.886 vs ISR=0.653），说明模型虽满足部分约束但难以完全满足所有约束

### 约束类型分析

- **矛盾约束（C9）** 对多数模型最具挑战性
- 在词汇约束、词计数、句计数等细粒度数值约束上表现普遍较差
- 文档级数值约束和受众风格约束表现较好
- 没有单一模型在所有约束类型上都保持领先

### 领域与任务分析

- 就业和心理学等领域表现较差，技术和招聘领域是多数模型的优势领域
- NLP 任务类型中，GPT-4o 在句间关系任务上表现优异，Qwen2-72B 在序列标注上较强

### 影响因素分析

四个因素对 ISR 有正相关影响：
1. 提示长度
2. **约束数量**（影响最大）
3. 约束类型数
4. 主需求数

对于 PSR：约束数量和类型数不完全正相关，主需求数量影响更大——约束少时用户对未满足约束更敏感，约束多时对次要约束更宽容。

### 与其他基准的对比

| 基准 | 样本数 | 类型数 | 系统性 | 优先级 |
|------|--------|--------|--------|--------|
| IFEval | 541 | 4 | ✗ | ✗ |
| FollowBench | 820 | 5 | ✗ | ✗ |
| ComplexBench | 1150 | 4 | ✔ | ✗ |
| **CFBench** | **1000** | **10-25** | **✔** | **✔** |

### 提升策略探索

- **SFT**：经过指令微调的模型效果显著提升（如 Qwen 系列）
- **模型规模**：Qwen2-72B 较 Qwen2-7B 有 40% 的相对 PSR 提升
- **复杂约束训练**：复制 Conifer 方法，使用复杂约束指令微调可进一步提升性能

## 亮点与洞察

1. **系统性约束分类体系**：首次提出基于分类学和统计学方法论的指令约束框架（10+25），远比现有基准全面
2. **需求优先级机制**：PSR 引入主/次需求概念，更贴近真实用户对 LLM 输出的容忍度判断
3. **CFBench 排名与 MMLU/GSM8K 排名不完全一致**，说明约束遵循是独立于知识能力和数学能力的一种能力维度
4. **实验规模大**：50+ 模型的全面横评提供了丰富的模型能力画像
5. **发现"矛盾约束"是通用短板**——即使最强模型也难以优雅处理互斥需求

## 局限性

1. 主要关注中文能力较强的模型，缺乏更广泛的英文模型调查
2. 对中英双语指令遵循差异的分析不够深入
3. 评估主要依赖 GPT-4o 作为评判模型，可能存在评估偏差
4. 对推理类模型（如 DeepSeek-R1）的约束遵循增强机制缺乏深入分析

## 相关工作

- **指令遵循训练**：Alpaca 式 SFT → 复杂指令（Xu et al., 2023）→ 约束数量和多样性提升（Sun et al., 2024a; He et al., 2024a）
- **约束遵循评估**：IFEval（可验证指令）→ FollowBench（多级约束）→ ComplexBench（约束组合）→ CFBench（系统性+优先级）

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 约束分类系统和 PSR 优先级评估是新颖设计
- **实验充分性**: ⭐⭐⭐⭐⭐ — 50+ 模型全面评测，消融和影响因素分析详尽
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富
- **实用价值**: ⭐⭐⭐⭐ — 对约束遵循的改进方向有明确指导
- **局限**: 数据集偏向中文场景，可能限制了对多语言研究的直接适用

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)

<!-- RELATED:END -->
