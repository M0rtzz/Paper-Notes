---
title: >-
  [论文解读] FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving
description: >-
  [ACL 2025][LLM推理][审慎推理] 提出 FineReason——一个基于逻辑谜题的推理基准，通过"状态检查"（判断当前状态是否可解）和"状态转换"（决定下一步操作）两个任务，对LLM的审慎推理能力（反思、回溯、纠错）进行原子级粒度评估，并证明在谜题数据上的训练可迁移提升数学推理能力（GSM8K 提升 5.1%）。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "审慎推理"
  - "逻辑谜题"
  - "状态检查"
  - "状态转换"
  - "回溯"
  - "中间步骤验证"
---

# FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving

**会议**: ACL 2025  
**arXiv**: [2502.20238](https://arxiv.org/abs/2502.20238)  
**代码**: [https://github.com/DAMO-NLP-SG/FineReason](https://github.com/DAMO-NLP-SG/FineReason)  
**作者**: Guizhen Chen, Weiwen Xu, Hao Zhang, Hou Pong Chan, Chaoqun Liu, Lidong Bing, Deli Zhao, Anh Tuan Luu, Yu Rong  
**机构**: Nanyang Technological University, DAMO Academy (Alibaba), Hupan Lab  
**领域**: LLM推理  
**关键词**: 审慎推理, 逻辑谜题, 状态检查, 状态转换, 回溯, 中间步骤验证

## 一句话总结

提出 FineReason——一个基于逻辑谜题的推理基准，通过"状态检查"（判断当前状态是否可解）和"状态转换"（决定下一步操作）两个任务，对LLM的审慎推理能力（反思、回溯、纠错）进行原子级粒度评估，并证明在谜题数据上的训练可迁移提升数学推理能力（GSM8K 提升 5.1%）。

## 研究背景与动机

**领域现状**：LLM 的推理能力正从 System 1（快速直觉）向 System 2（慢速分析）转变，OpenAI-o1, DeepSeek-R1 等推理模型通过迭代反思和纠正展现了强大的推理能力。

**现有方法的不足**：
   - 现有推理基准（MATH, GSM8K, HumanEval）**仅关注最终答案正确率**，无法评估中间推理过程
   - 模型可能通过有缺陷的推理达到正确结论（Zelikman et al., 2022; Lightman et al., 2024）
   - 模型可能利用训练数据中的表面模式"作弊"（Roelofs et al., 2019）
   - 无法区分模型到底是在"真正推理"还是"模式匹配"

**核心动机**：需要一个基准能够评估推理过程的每一步，特别是**反思**（检查当前状态）和**纠错**（回溯到正确路径）的能力。逻辑谜题天然适合这一目的——每步操作可分解为原子步骤，且有明确规则可自动验证。

## 方法详解

### 整体框架

FineReason 包含四类逻辑谜题，两个评估任务，以及一个训练集：

**四类谜题**：

| 谜题 | 状态定义 | 最小操作 | 数据来源 |
|------|---------|---------|---------|
| 数独（Sudoku） | 部分/完整 9×9 棋盘 | 添加/移除一个数字 | Kaggle 数据集 |
| 图着色（Graph Coloring） | 部分/完整着色的图 | 着色/去色一个顶点 | 随机图生成 + 回溯算法 |
| 24 点（Game of 24） | 部分/完整算术表达式 | 对两个数执行/撤销一次运算 | Yao et al. (2023) |
| 逻辑网格谜题（Grid Puzzles） | 部分/完整网格 | 根据线索分配/移除属性 | Tyagi et al. (2024) |

### 关键设计一：基于树的谜题分解

- 将谜题求解过程表示为**搜索树**：节点是中间状态，边是状态转换
- 从初始状态 $s_1$ 进行深度优先搜索（DFS），每步仅执行最小操作
- 边是双向的——支持前进探索和回溯
- 将规则转为可执行代码，自动验证每个状态的合法性
- 对于逻辑网格谜题，定义三个辅助函数 $r(v)$, $c(v)$, $T(i,j)$ 将文本线索编码为可验证的约束条件

### 关键设计二：两个评估任务

**状态检查（State Checking）**：
- 给定当前状态 $s_i$，判断是否存在可达的解 $s_n$
- 从树中均匀采样可解和不可解状态
- 评估两个层面：(1) 检查已有步骤是否违反规则（回顾性），(2) 预判未来是否会进入死胡同（前瞻性）

**状态转换（State Transition）**：
- 给定当前状态和状态检查结果，决定下一步操作
- 可解状态 → 探索未访问的子状态
- 不可解状态 → 回溯到父状态
- 评估时提供真实的状态检查标签，消除状态检查错误的干扰
- 提供部分不可解子状态，测试模型是否能有效规避

### 训练数据

构建谜题训练集，包含状态检查和状态转换数据，用于增强通用推理能力。

## 实验关键数据

### 实验设置

- 测试实例：每类谜题 500 个中间状态 × 4 类 = 2000 个实例/任务
- 评估方式：0-shot CoT prompt，明确禁止使用编程求解
- 模型：推理模型（o1, Gemini-2.0-Flash-Thinking）+ 通用模型（GPT-4o, GPT-3.5, Gemini-2.0-Flash, Qwen2.5-72B-Inst）

### 端到端谜题求解准确率

| 谜题 | GPT-4o | Gemini-F | Gemini-FT | o1 |
|------|--------|----------|-----------|-----|
| 数独 | 0 | 5.9 | 0 | 0 |
| 图着色 | 7.8 | 35.3 | 80.4 | 78.4 |
| 24点 | 15.3 | 83.7 | 48.0 | 54.1 |
| 网格谜题 | 2.2 | 10.9 | 34.8 | 45.7 |

端到端结果不一致——Gemini-F 在数独和 24 点上优于 Gemini-FT，但在其他谜题上大幅落后，表明端到端准确率不足以可靠评估推理能力。

### 状态检查 + 状态转换结果（核心结果）

| 谜题 | 模型 | 状态检查 | 状态转换 | 平均 |
|------|------|---------|---------|------|
| 数独 | Random | 50.0 | - | - |
| | GPT-4o | 52.4 | 38.8 | 45.6 |
| | Gemini-FT | 69.2 | 48.8 | 59.0 |
| | **o1** | **81.0** | **70.2** | **75.6** |
| 图着色 | GPT-4o | 56.4 | 49.4 | 52.9 |
| | Gemini-FT | 92.6 | 46.4 | 69.5 |
| | **o1** | **94.6** | **65.0** | **79.8** |
| 24点 | GPT-4o | 82.6 | 23.0 | 52.8 |
| | Gemini-FT | 96.0 | 48.6 | 72.3 |
| | **o1** | **97.4** | **86.6** | **92.0** |
| 网格谜题 | GPT-4o | 52.4 | 10.0 | 31.2 |
| | Gemini-FT | 89.0 | 51.4 | 70.2 |
| | **o1** | **88.8** | **77.6** | **83.2** |

**关键发现**：
- o1 和 Gemini-FT 之间存在 **19.7%** 的显著差距，而在其他数学/代码基准上二者差距很小
- 通用模型（GPT-4o 等）在数独和网格谜题的状态检查上接近随机猜测
- Gemini-FT 在状态检查上接近 o1，但在状态转换上显著落后——暴露了其纠错能力的短板

### 模型行为分析

**状态检查精度/召回率**（以不可解状态为正例）：
- 通用模型在深树谜题（数独、图着色）中召回率极低，倾向于"过度乐观"——遇到超出能力的问题时默认判定为可解
- GPT-4o 和 Qwen2.5 精度高但召回低——非常保守，只在非常确信时才判定不可解
- 推理模型在两个指标上均表现良好

### 训练迁移效果

在 DeepSeek-R1-Distill-Qwen-7B 上的实验：

| 训练数据 | GSM8K | MATH-500 |
|---------|-------|----------|
| 仅数学数据 | 82.3% | - |
| 数学 + **谜题数据** | **87.4%** | 提升 |

谜题数据带来 **5.1%** 的 GSM8K 提升，证明回溯和约束验证等技能可从谜题迁移到通用推理。

## 亮点与洞察

1. **评估范式创新**：从"答案对不对"转向"中间每一步对不对"——这是评估推理能力的正确方向
2. **逻辑谜题 = 完美的评估载体**：规则明确、步骤可原子化分解、可自动验证
3. **揭示通用模型的"过度乐观"倾向**：GPT-4o 等模型在面对困难状态时默认判定为可解，从不回溯
4. **区分度远超现有基准**：o1 vs Gemini-FT 差距 19.7%，而在 MATH 等基准上二者接近饱和
5. **训练迁移效果**证明了谜题训练可作为通用推理能力的增强工具——类似于下棋训练提升人类策略思维

## 局限性

1. 四类谜题均为组合优化/约束满足类型，可能无法全面代表所有推理类型（如因果推理、类比推理）
2. "禁止编程"的指令依赖模型遵从，实际上模型可能仍隐式利用记忆中的算法
3. 训练迁移实验仅在一个 7B 模型上验证，更大/更小模型的效果未知
4. 逻辑网格谜题的文本线索转代码依赖 GPT-4o 的 one-shot 翻译 + 人工验证，可扩展性受限
5. 未讨论不同难度谜题的性能分层分析（如不同棋盘大小、不同图密度）

## 相关工作

- **推理基准**：GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), HumanEval (Chen et al., 2021)
- **推理模型**：OpenAI o1 (2024), DeepSeek-R1 (2025), Gemini-2.0-Flash-Thinking (Google, 2024)
- **过程奖励/中间步骤评估**：PRM (Lightman et al., 2024)
- **LLM + 搜索**：Tree-of-Thoughts (Yao et al., 2023)

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是一篇出色的基准构建工作。问题定义清晰（从最终答案到中间过程），载体选择巧妙（逻辑谜题的可原子化验证性），实验发现深刻（揭示模型的过度乐观和纠错短板），且训练迁移实验为"谜题训练提升推理"提供了有力证据。对于理解和提升 LLM 推理能力具有重要参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[ACL 2025\] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](../../ICML2025/llm_reasoning/improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](../../ICML2026/llm_reasoning/evaluating_relational_reasoning_in_llms_with_rel.md)
- [\[ICML 2025\] MARGE: Improving Math Reasoning for LLMs with Guided Exploration](../../ICML2025/llm_reasoning/marge_improving_math_reasoning_for_llms_with_guided_exploration.md)

</div>

<!-- RELATED:END -->
