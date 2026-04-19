---
title: >-
  [论文解读] RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation
description: >-
  [ICML 2025 (Poster)][推理评估] 受 Pearl 因果阶梯启发，提出 RE-IMAGINE 框架，通过将问题转化为中间符号表示（代码）并在计算图上执行多层级变异，生成不可通过记忆化解决的基准变体，系统评估 LLM 的真实推理能力。
tags:
  - ICML 2025 (Poster)
  - 推理评估
  - 因果阶梯
  - 符号表示
  - 基准变异
  - 记忆化检测
  - 计算图
---

# RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation

**会议**: ICML 2025 (Poster)

**arXiv**: [2506.15455](https://arxiv.org/abs/2506.15455)

**作者**: Xinnuo Xu, Rachel Lawrence, Kshitij Dubey, Atharva Pandey, Risa Ueno, Fabian Falck, Aditya V. Nori, Rahul Sharma, Amit Sharma, Javier Gonzalez

**领域**: 因果推理 / LLM推理评估

**关键词**: 推理评估, 因果阶梯, 符号表示, 基准变异, 记忆化检测, 计算图

## 一句话总结

受 Pearl 因果阶梯启发，提出 RE-IMAGINE 框架，通过将问题转化为中间符号表示（代码）并在计算图上执行多层级变异，生成不可通过记忆化解决的基准变体，系统评估 LLM 的真实推理能力。

## 研究背景与动机

近年来 LLM 在推理基准（如 GSM8K）上的准确率持续攀升，但一个核心疑问始终未解：

> **模型的高表现究竟源于真正的推理能力，还是对训练集的统计记忆？**

现有的评估方法存在以下问题：

**静态基准泄露**：固定的基准一旦发布，可能被纳入训练数据

**变异方式零散**：之前的工作（如 GSM-Symbolic）仅探索了有限的变异类型，缺乏统一层级框架

**难度与记忆纠缠**：性能下降可能源于题目变难而非记忆失效

RE-IMAGINE 的核心思想借鉴 Judea Pearl 的**因果阶梯** (Ladder of Causation)，将推理能力划分为三个递进层级，并构建可扩展的自动化变异管道。

## 方法详解

### 整体框架

RE-IMAGINE 的管道包含四个步骤：

```
自然语言问题 → 符号表示(代码) → AST变异 → 自然语言变体问题
```

1. **NL-to-Symbolic**：将自然语言问题转化为可执行的 Python 代码（计算图）
2. **Symbolic Mutation**：在 AST（抽象语法树）上执行确定性变异操作
3. **Mutation-to-NL**：将变异后的代码翻译回自然语言问题
4. **Answer Verification**：通过执行变异后代码获得确定性答案

### 推理层级体系（因果阶梯映射）

受 Pearl 因果阶梯（关联 → 干预 → 反事实）启发，定义三个推理层级：

| 层级 | Pearl 对应 | 变异类型 | 核心测试能力 |
|:---|:---|:---|:---|
| Level-1 | 关联 (Association) | 原始问题 | 基线性能 |
| Level-2 | 干预 (Intervention) | SampleValues, UselessInfo, OverWriteValue | 保持推理路径不变的泛化 |
| Level-3 | 反事实 (Counterfactual) | AddDependence, InsertConditional, CounterFactual | 需要修改推理路径 |

### 关键变异操作

**Level-2 变异**（保持原始推理路径）：

- **SampleValues**：替换计算图中的叶节点数值（$x_i \to x_i + \delta$, $\delta \in [-10, 10]$）
- **UselessInfo**：在计算图中添加不影响答案的冗余节点
- **OverWriteValue**：覆盖中间变量值

**Level-3 变异**（修改推理路径）：

- **AddDependence**：添加新的依赖节点，增加一步推理
- **InsertConditional**：插入条件分支（if-else），改变计算逻辑
- **CounterFactual**：假设某条件与事实相反，重新推导答案

### 损失函数与质量保证

变异过程的正确性通过以下机制保证：

1. **NL-to-Code 验证**：确保代码中的常量与问题中的数值一一对应
2. **Code-to-NL 回译验证**：使用 GPT-4o 回译并执行，验证答案一致性
3. **人工抽样检查**：报告每种变异的错误率

核心模型选择：
- GSM8K NL→Code: Mixtral-8x7B
- GSM8K Code→NL: GPT-4o
- CLadder: 原始 causal engine + Meta-Llama-70B-Instruct
- Loop/CruxEval: 直接从符号表示出发，无需 LLM

## 实验关键数据

### 主实验：GSM8K 上各模型的变异性能

| 模型 | Raw | SampleValues (L2) | OverWriteValue (L2) | UselessInfo (L2) | AddDependence (L3) | InsertConditional (L3) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| QwQ-32B | 100 | 98.2 | 92.6 | 99.1 | **59.6** | 95.6 |
| R1-Distill-Qwen-32B | 98.3 | 93.5 | 85.9 | 97.6 | **63.2** | 85.0 |
| GPT-o3-mini | 97.4 | 90.3 | 84.0 | 93.5 | **77.3** | 91.6 |
| GPT-4.5 | 97.5 | 89.8 | 81.3 | 95.5 | **61.5** | 89.3 |

### 消融实验：按计算步数分析性能下降

控制题目难度（计算步数）后的平均准确率，证明性能下降并非仅因难度增加：

| 变异类型 | 2步 | 3步 | 4步 | 5步 | 6步 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Raw | 0.95 | 0.94 | 0.84 | 0.91 | 0.83 |
| SampleValues (L2) | 0.87 | 0.84 | 0.75 | 0.74 | 0.80 |
| UselessInfo (L2) | 0.91 | 0.90 | 0.90 | 0.81 | 0.88 |
| CounterFactual (L3) | 0.74 | 0.71 | 0.75 | 0.62 | 0.67 |
| InsertConditional (L3) | 0.62 | 0.68 | 0.65 | 0.61 | 0.57 |
| AddDependence (L3) | 0.57 | 0.47 | 0.46 | 0.45 | 0.42 |

### CruxEval 代码基准结果

| 模型 | Raw | Mutate String (L2) | Mutate Value (L2) | Redefine Function (L2) | Replace Operator (L2) | Swap Conditional (L2) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4.5 | 45.5 | 23.6 | 32.5 | 29.8 | 29.1 | 32.1 |
| GPT-o3-mini | 56.9 | 36.0 | 59.8 | 56.4 | 58.5 | 50.0 |

### 关键发现

1. **Level-3 变异引发最大性能下降**：AddDependence 使 QwQ-32B 从 100% 骤降至 59.6%，即使是最强模型也难以应对需要修改推理路径的变异
2. **Same-step 下仍有显著下降**：在相同计算步数下，Level-3 变异的 2 步准确率（0.57）低于 Raw 的 6 步准确率（0.83），说明性能下降并非难度增加所致
3. **SampleValues 的普遍影响**：仅改变数值就导致 5-10% 的性能下降，表明模型对特定数值有记忆倾向
4. **代码推理更脆弱**：CruxEval 上 GPT-4.5 的 Mutate String 变异后仅剩 23.6%，接近一半性能损失

## 亮点与洞察

- **因果阶梯的创新映射**：将 Pearl 因果框架引入 LLM 推理评估，提供了理论基础而非 ad-hoc 的难度定义
- **确定性答案保证**：通过符号表示（可执行代码）获得变异后问题的确定性答案，避免了人工标注或 LLM 标注的噪声
- **跨域统一框架**：同一框架覆盖数学（GSM8K）、代码（CruxEval/Loop）和逻辑（CLadder）三大推理领域
- **模块化设计**：管道的每个步骤都可独立替换，适配新基准仅需编写约 150 行代码（50 行变异定义 + 100 行 Code→NL 提示）
- **动态基准范式**：每次评估可生成不同变体，从根本上解决基准泄露问题

## 局限性

1. **NL→Code 步骤依赖 LLM**：对于复杂推理问题，NL→Code 转化可能引入错误，且需要针对每个基准定制提示
2. **变异与难度的纠缠**：虽然作者做了 same-step 分析，但 Level-3 变异本质上增加了推理步骤，难以完全分离难度效应
3. **因果阶梯映射的松散性**：审稿人指出，Level-3 变异与严格的反事实定义存在差距，更像是"干预"而非"反事实"
4. **无法检测非记忆化的推理失败**：如果模型既没记忆化也没推理能力，性能下降不能唯一归因于记忆化
5. **仅覆盖零/少样本设置**：未探索微调后模型在变异基准上的表现

## 相关工作与启发

- **GSM-Symbolic** (Mirzadeh et al., 2024)：仅探索了 SampleValues 等 Level-2 变异，RE-IMAGINE 扩展至 Level-3
- **GSM-IC** (Shi et al., 2023)：研究无关信息对推理的干扰，对应 RE-IMAGINE 的 UselessInfo
- **iGSM** (Ye et al., 2024)：使用符号结构生成合成管道，但非统一的多级框架
- **Pearl's Ladder of Causation** (2009)：提供理论基础，将推理能力层级化
- **Reasoning Elicitation via Counterfactual Feedback** (Hüyük et al., 2024)：Level-3 Bi-Counterfactual 的灵感来源

**启发**：将因果推理理论引入 LLM 评估是一个有前景的方向。未来可探索：(1) 在训练中加入变异数据是否提升真实推理能力；(2) 机械可解释性视角下模型如何处理不同层级的变异。

## 评分

| 维度 | 分数 (1-10) | 评价 |
|:---|:---:|:---|
| 创新性 | 7 | 因果阶梯映射和统一变异框架有新意，但 Level-2 变异与先前工作重叠 |
| 技术深度 | 7 | 管道设计完整、验证机制严谨，但理论分析可更深入 |
| 实验充分性 | 8 | 四个基准、多个模型家族、消融分析全面，审稿人评价 "extremely thorough" |
| 写作质量 | 7 | 框架描述清晰，但审稿人反映存在重复段落等排版问题 |
| 实用价值 | 8 | 动态基准生成和模块化设计有很高的实际应用价值 |
| 总体推荐 | 7.5 | 扎实的系统性工作，连接因果理论与 LLM 评估，值得关注 |

<!-- RELATED:START -->

## 相关论文

- [Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](../../ACL2025/causal_inference/reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)
- [Causal Graph based Event Reasoning using Semantic Relation Experts](../../ACL2025/causal_inference/causal_graph_based_event_reasoning_using_semantic_relation_experts.md)
- [A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)

<!-- RELATED:END -->
