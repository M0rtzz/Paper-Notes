---
title: >-
  [论文解读] Learning to Generate Structured Output with Schema Reinforcement Learning
description: >-
  [ACL 2025][强化学习][structured generation] 提出 SchemaBench 基准（约4万条 JSON schema）和 Schema Reinforcement Learning (SRL) 训练框架，通过细粒度 schema 验证器提供密集奖励信号…
tags:
  - "ACL 2025"
  - "强化学习"
  - "structured generation"
  - "JSON schema"
  - "reinforcement-learning"
  - "LLM"
  - "benchmark"
---

# Learning to Generate Structured Output with Schema Reinforcement Learning

**会议**: ACL 2025  
**arXiv**: [2502.18878](https://arxiv.org/abs/2502.18878)  
**代码**: [https://github.com/thunlp/SchemaReinforcementLearning](https://github.com/thunlp/SchemaReinforcementLearning)  
**领域**: 强化学习  
**关键词**: structured generation, JSON schema, reinforcement-learning, LLM, benchmark

## 一句话总结

提出 SchemaBench 基准（约4万条 JSON schema）和 Schema Reinforcement Learning (SRL) 训练框架，通过细粒度 schema 验证器提供密集奖励信号，结合 Thoughts of Structure (ToS) 推理机制，将 LLM 的复杂 JSON 生成准确率提升高达16%，同时不损害通用推理能力。

## 研究背景与动机

大语言模型在实际应用中越来越多地需要生成结构化输出（尤其是 JSON 格式），以便与自动化系统和 API 对接。当前有几种主流方法：

**Prompting**：直接提示生成，对简单 schema 有效，但在复杂逻辑下容易出错

**Tool Calls**：把模型输出转换为 JSON，但常遗漏 schema 特定语法

**Constrained Decoding**（如 Outlines、SGLang）：通过限制解码空间保证有效 JSON，但可能降低输出质量

**核心痛点**：

- 缺乏全面的基准来评估模型在复杂 JSON schema 下的生成能力
- 现有模型（包括 GPT-4o）在复杂 schema 下的整体正确率仅约 61%
- SFT 方法面临高质量训练数据匮乏的问题——自动生成器和模型 prompting 均难以为复杂 schema 规模化生成合规 JSON
- 即使经过 SFT，模型在某些场景下仍无法学会基本的 JSON 语法

## 方法详解

### 1. SchemaBench 基准构建

从 JSON Schema Store 和 GitHub 爬取 108,528 个 schema 文件，经过外部 URI 过滤（保留 46,280 个）和语法合规验证（移除 5,574 个），最终得到约 40,706 个有效 schema（训练集 36,960 / 测试集 3,746）。平均字符长度约 35,754，平均嵌套深度 16.7 层。

**任务一：Schema-only Generation**（三个子任务）：

- **Complex Schema**：测试模型在复杂引用（\）和逻辑组合（anyOf/oneOf）下生成有效 JSON 的能力
- **Custom Formats**：测试模型理解自然语言描述的自定义格式约束（电话号码、文件路径、RGB 颜色、base64 编码等），通过 const/pattern 字段验证
- **Escape Translation**：测试模型正确处理转义字符（\"、\\、\n 等）的能力，任何转义错误会导致整个 JSON 无效

**任务二：Schema-constrained Reasoning**：在 GSM8K、MATH-500、MMLU、ARC-Challenge 上要求模型在 schema 约束下生成答案，同时评估推理能力和 JSON 合规性。

### 2. Schema Reinforcement Learning (SRL)

基于 PRIME 框架的在线强化学习方法，分三个阶段迭代进行：

**采样阶段**：使用当前策略模型为每个 schema 任务生成 K 个候选响应。引入 **Thoughts of Structure (ToS)** 机制——受 CoT 启发，模型在生成 JSON 之前先以 JSON5 注释形式进行结构推理，阐述每个键值对的生成理由。验证时忽略注释，仅验证最终 JSON。

**奖励阶段**：

- **细粒度 schema 验证器**：不使用简单的二值奖励（通过/不通过），而是计算 correctness ratio（正确 token 数 / 总 token 数）。对于仅部分有效的 JSON，计算有效部分的正确比例；对于解析失败的 JSON，在错误位置截断并填充控制字符继续验证剩余部分。这大幅缓解了奖励稀疏问题。
- **优势估计**采用 leave-one-out 方式：将当前响应的奖励与其余 K-1 个响应的平均奖励做差。

**更新阶段**：使用 Cross Entropy loss 更新奖励模型，PPO 算法（带 clipping）更新策略模型。

### 3. 关键设计选择

- 训练使用 SchemaBench 训练集的 37K schema，batch size 32，学习率 5e-7
- 补充训练数据（Collected JSON）包括 UltraChat 6K、UltraInteract 6K、xLAM 20K、Glaive 20K、ToolACE 10K
- Tool-calling 数据集中的工具定义被转换为 JSON schema 格式

## 实验关键数据

### 表1：SchemaBench 主实验结果（%）

| 模型 | Complex | Custom | Escape | Overall | GSM8K | MATH-500 |
|------|---------|--------|--------|---------|-------|----------|
| GPT-4o | 84.47 | 61.56 | 37.14 | 61.06 | 97.80 | 41.40 |
| Qwen-2.5 7B | 72.42 | 43.60 | 11.11 | 42.38 | 94.54 | 38.60 |
| LLaMA-3.1 8B | 64.26 | 33.07 | 12.02 | 36.45 | 95.91 | 85.60 |
| LLaMA-3.1 8B SFT | 74.56 | 46.64 | 60.58 | 60.59 | 89.46 | 63.80 |
| LLaMA-3.1 8B **SRL** | **90.48** | **78.67** | **69.86** | **79.67** | 90.90 | 88.00 |
| LLaMA-3.2 3B | 49.84 | 27.31 | 8.37 | 28.51 | 80.97 | 35.40 |
| LLaMA-3.2 3B SFT | 71.71 | 45.52 | 52.21 | 56.48 | 82.94 | 44.40 |
| LLaMA-3.2 3B **SRL** | **82.25** | **66.13** | **69.10** | **72.50** | 84.23 | 43.20 |

**关键发现**：SRL 在所有 schema-only 子任务上大幅超越 SFT。LLaMA-3.1 8B SRL 达到 79.67%，显著超过 GPT-4o 的 61.06%。

### 表2：BFCL-Live 下游任务结果（%）

| 模型 | Simple | Multiple | Parallel | Multi-Para. | Overall |
|------|--------|----------|----------|-------------|---------|
| GPT-4o Tool | 36.43 | 37.22 | 18.75 | 41.67 | 59.13 |
| LLaMA-3.1 8B | 0.39 | 0.00 | 0.00 | 0.00 | 24.08 |
| LLaMA-3.1 8B SFT | 72.09 | 68.76 | 50.00 | 66.67 | 52.69 |
| LLaMA-3.1 8B **SRL** | 72.09 | 73.10 | **75.00** | 50.00 | **70.10** |
| LLaMA-3.2 3B | 4.26 | 13.11 | 0.00 | 0.00 | 35.72 |
| LLaMA-3.2 3B SFT | **74.03** | **74.64** | **68.75** | **58.33** | **64.10** |
| LLaMA-3.2 3B SRL | 65.50 | 64.22 | 50.00 | 29.17 | 57.00 |

LLaMA-3.1 8B SRL 在 BFCL 下游 function calling 任务中达到 70.10%，超过所有对比方法。

### 消融实验

| 设置 | Schema (%) | MATH-500 | ARC-C |
|------|-----------|----------|-------|
| LLaMA-3.2 3B baseline | 28.51 | 35.40 | 79.27 |
| + ORM | 31.15 | 39.40 | 78.92 |
| + ToS | **44.89** | 36.60 | 80.38 |
| + Fine-grained Validator | 35.59 | 35.60 | 79.10 |

ToS 贡献最大（+13.74%），细粒度验证器也有显著提升（+4.44%），两者互补。

## 亮点

- **SchemaBench 是首个大规模、高复杂度的 JSON schema 生成评测基准**，涵盖3类挑战任务，schema 平均嵌套深度 16.7，贴近真实应用场景
- **细粒度 schema 验证器**解决了结构化生成中的稀疏奖励问题，对部分正确的 JSON 也能给出有效梯度信号，显著优于简单的二值奖励
- **Thoughts of Structure (ToS)** 机制新颖，让模型在生成 JSON 前先推理结构，利用 JSON5 注释保持格式兼容性，消融实验中贡献最大
- **SRL 不仅大幅提升结构化生成能力，还较好地保留了通用推理能力**（MATH-500、ARC-C 未明显下降），与 SFT 掉分形成对比
- RL 训练效率高，约训练到一半时即可超过 SFT 基线

## 局限与展望

1. **格式局限**：目前仅面向 JSON schema，未扩展到 YAML、XML、TOML 等其他结构化格式
2. **效率瓶颈**：SRL 的在线采样阶段计算开销大，需要反复生成和验证
3. **小模型泛化受限**：LLaMA-3.2 3B SRL 在 BFCL 下游任务上反而不如 SFT（57.00% vs 64.10%），说明 SRL 在极小模型上的迁移能力有待提升
4. **缺少更多对比**：未评测 Claude/Gemini 等更强模型，也未与 GRPO、DPO 等其他 alignment 方法对比
5. **ToS 注释质量**：未分析 ToS 注释的实际内容质量，也未探讨推理长度与性能的关系

## 与相关工作的对比

| 方法类别 | 代表工作 | 与本文关系 |
|---------|---------|-----------|
| Prompting 方法 | OpenAI Structured Outputs | 本文基准证明 prompting 在复杂 schema 下不足 |
| Constrained Decoding | Outlines, SGLang, XGrammar | 可与 SRL 互补，但可能降低输出质量 |
| Tool Call 方法 | ToolLLM, Toolformer | 依赖后处理，难以对齐标准 schema |
| 结构化生成基准 | BFCL, StructuredBench | SchemaBench 更大更复杂（40K schemas） |
| RL for LLM | PRIME, PPO, RLHF | SRL 在 PRIME 基础上引入 schema 验证器和 ToS |

## 评分

- 新颖性: ⭐⭐⭐⭐ — SchemaBench 填补了复杂 JSON 生成评测空白，ToS 和细粒度验证器是有意义的创新
- 实验充分度: ⭐⭐⭐⭐ — 多模型、多任务对比全面，消融实验清晰，但缺少更强闭源模型和更多 RL 方法对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，问题动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ — JSON 结构化生成是 LLM 应用中的核心问题，基准和方法具有很高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](../../ICLR2026/reinforcement_learning/learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](../../ICML2025/reinforcement_learning/robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [\[ACL 2025\] Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering](prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)
- [\[NeurIPS 2025\] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/learning_to_focus_prioritizing_informative_histories_with_structured_attention_m.md)

</div>

<!-- RELATED:END -->
