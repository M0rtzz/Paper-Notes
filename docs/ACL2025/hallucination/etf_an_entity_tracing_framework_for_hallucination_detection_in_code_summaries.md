---
title: >-
  [论文解读] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries
description: >-
  [ACL 2025][幻觉检测][代码摘要] 提出 Entity Tracing Framework (ETF)，一种通过静态程序分析提取代码实体、再用 LLM 验证这些实体在生成摘要中是否被正确描述的幻觉检测框架，配合首创的 CodeSumEval 数据集（~10K样本）…
tags:
  - "ACL 2025"
  - "幻觉检测"
  - "代码摘要"
  - "实体追踪"
  - "程序分析"
  - "LLM评估"
---

# ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries

**会议**: ACL 2025  
**arXiv**: [2410.14748](https://arxiv.org/abs/2410.14748)  
**代码**: 有 (GitHub)  
**领域**: 幻觉检测  
**关键词**: 代码摘要, 幻觉检测, 实体追踪, 程序分析, LLM评估

## 一句话总结

提出 Entity Tracing Framework (ETF)，一种通过静态程序分析提取代码实体、再用 LLM 验证这些实体在生成摘要中是否被正确描述的幻觉检测框架，配合首创的 CodeSumEval 数据集（~10K样本），在代码摘要幻觉检测上达到 73% F1。

## 研究背景与动机

随着 LLM 在代码摘要（Code Summarisation）任务中的广泛应用（如 Amazon Q Developer、IBM Watsonx），生成的代码文档虽然语言流畅但却可能包含幻觉——即摘要内容与源代码实际功能不符。这个问题特别危险，因为：

**代码摘要的幻觉很难被发现**：生成文本看起来合理但实际上错误，甚至可能误导新手开发者

**代码和自然语言的交互复杂**：模型可能基于函数名推测功能（如看到 `getJobID()` 就臆想数据库查询），即"标识符名称偏差"

**现有幻觉检测方法不直接适用**：NL领域的幻觉检测通常不需要对照参考文本（代码），而代码摘要必须以源代码为ground truth

论文指出，此前没有专门用于代码摘要幻觉检测的数据集和方法，这是一个重要的空白。

## 方法详解

### 整体框架

ETF 的核心思想受人类代码审查行为启发：开发者验证文档时，会逐行阅读代码，追踪文档中提到的实体是否在代码中有对应、描述是否准确。这是一个自底向上（bottom-up）的过程，与工作记忆理论（Working Memory Theory）一致——人的工作记忆容量有限（3-4个有依赖关系的对象），因此需要聚焦于局部实体验证。

框架包含两大步骤：

**步骤一：实体验证（Entity Verification）**  
检查摘要中的实体是否存在于源代码中→检测**外在幻觉（Extrinsic Hallucination）**

**步骤二：实体意图验证（Entity-Intent Verification）**  
对于匹配到的实体，验证其在摘要中的描述是否准确→检测**内在幻觉（Intrinsic Hallucination）**

### 关键设计

1. **代码端实体提取**：使用 Javalang 静态分析工具将 Java 代码解析为 AST（抽象语法树），提取变量名、类名、函数名等所有代码实体。这一步完全不依赖 LLM，准确可靠

2. **摘要端实体提取**：采用 LLM（GPT-4-Omni）做 NER，基于 Tabassum et al. (2020) 的 NER tagset。之所以用 LLM 而非规则，是因为摘要中存在严重的多义性问题——如 "list", "while", "if" 可以是代码实体也可以是自然语言词汇。同时加入过滤步骤移除 LLM 虚构的实体

3. **实体匹配**：通过字符串匹配启发式规则将摘要实体映射回代码。未匹配的实体所在句子标记为外在幻觉。匹配到的实体进入下一步验证

4. **意图提取与验证**：对每个匹配实体，用字符串匹配（而非prompt）从摘要中提取描述该实体的相关句子。选择字符串匹配而非 LLM 是因为 prompt 方法会引入额外幻觉（虚构句子、提取不准确等）。然后构造 <代码, 实体, 相关句子> 三元组，用 LLM zero-shot 验证描述准确性

5. **摘要级聚合**：如果一个摘要中有 ≥1 个实体被标记为幻觉，则整条摘要判定为幻觉。阈值设为1是基于人工标注数据——被评为 FAIR/POOR 的摘要平均有 1.33 个幻觉实体

### CodeSumEval 数据集

从 CodeXGLUE 数据集中选取 600 个 Java 代码片段，用 7 个不同 LLM 生成摘要（包括 Granite-20B/34B, Llama3-8B/70B, CodeLlama-7B/34B, Mistral-7B），每个片段平均产生 411 条摘要，细分为 9933 个实体级样本。

标注过程由 8 位 Java 专家完成（均有硕士以上学历，4+ 年 Java 经验），包含三层标注：NER、实体描述验证（CORRECT/INCORRECT/IRRELEVANT）、整体摘要质量评分。Cohen's Kappa = 0.72，表示高度一致。

## 实验关键数据

### 主实验：ETF vs Direct 方法（实例级幻觉检测）

| 模型 | Direct P/R/F1 | ETF P/R/F1 |
|------|---------------|------------|
| GPT4-Omni | 0.48/0.50/0.28 | **0.72/0.74/0.73** |
| Gemini-2.0 | 0.51/0.50/0.42 | **0.64/0.65/0.64** |
| Mixtral-8x22B | 0.48/0.48/0.45 | **0.62/0.61/0.61** |
| Llama-3.1-70B | 0.57/0.54/0.38 | **0.62/0.62/0.54** |
| Llama3-8B | 0.59/0.51/0.26 | **0.51/0.55/0.50** |
| Mistral-7B | 0.16/0.50/0.24 | **0.51/0.50/0.41** |

### 消融：各模型生成摘要的实体映射统计

| 生成模型 | 平均摘要长度 | 实体数 | 映射率↑ | 未映射率↓ |
|----------|------------|--------|---------|----------|
| Llama3-70B | 313.58 | 10.01 | **88.44%** | 8.69% |
| Granite-34B | 214.50 | 6.55 | 85.69% | 12.41% |
| Llama3-8B | 257.21 | 9.45 | 84.50% | 12.77% |
| Granite-20B | 148.95 | 7.10 | 79.54% | **19.64%** |

### 关键发现

1. **ETF 显著优于 Direct 方法**：GPT4-Omni 上 F1 从 0.28 提升至 0.73（+160%），证实了细粒度实体追踪优于笼统的 <代码,摘要> 判断
2. **代码复杂度是最主要的幻觉原因**：在分类体系中，Code Complexity 占比最高，其次是 Insufficient Knowledge
3. **大模型的grounding更好**：Llama3-70B 的实体映射率最高（88.44%），而 Granite-20B 未映射率最高（19.64%），即小模型更容易产生外在幻觉
4. **函数名偏差导致的幻觉尤其难检测**：当代码中引用了未定义的外部函数时，模型和人类标注者都表现出歧义
5. 几乎所有 LLM 在 NER 提取时都会"虚构"摘要中不存在的实体（CodeLlama-7B 虚构率高达 35.68%），凸显了过滤步骤的必要性

## 幻觉分类体系

论文提出了代码摘要幻觉的四类成因：
- **HC1 标识符名称偏差**：模型基于变量名/函数名/库名推断语义而非实际代码逻辑
- **HC2 知识不足**：对未见过的库/语言特性理解错误
- **HC3 代码复杂度**：长代码、复杂逻辑（嵌套条件、递归、多函数调用）导致理解失败
- **HC4 自然语言上下文**：代码中的注释、日志语句等自然语言干扰了模型理解

## 亮点与洞察

- **受认知科学启发的框架设计**：将人类代码审查的 bottom-up 认知过程形式化为实体追踪流程，设计思路新颖且直觉强
- **首创数据集**：CodeSumEval 填补了代码摘要幻觉检测领域的空白，标注质量高（Kappa=0.72）
- **实用价值高**：框架能精确定位幻觉发生在摘要的哪个位置、涉及哪个实体，相比binary的"有/无幻觉"判断提供了更丰富的可解释性
- **工程选择务实**：意图提取最终选用字符串匹配而非 LLM prompt，因为 prompt 方法引入更多幻觉——这种务实的设计决策值得学习

## 局限与展望

1. 仅支持 Java 语言——依赖 Javalang parser，对低资源语言（COBOL、Perl）不适用
2. 实体检测和验证高度依赖大规模 LLM（GPT-4），计算成本高且不够可扩展
3. 多阶段 pipeline 涉及多次 LLM 调用，实时部署到开发者工具中有挑战
4. 对"创造性摘要"（准确但更加elaborate的描述）可能产生误判
5. 实体形式变化（如 "PreparedStatement" → "prepared statement"）可能导致匹配遗漏

## 相关工作与启发

- **NL幻觉检测**（Manakul et al., 2023; Dhuliawala et al., 2023）：无参考文本grounding，不直接适用于代码场景
- **代码生成幻觉**（Liu et al., 2024; Tian et al., 2024）：关注生成代码的幻觉，而非代码摘要的幻觉，方向不同
- **启发**：实体追踪范式可推广到其他需要grounding验证的场景（如科学论文摘要对照原文验证）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个专门针对代码摘要幻觉的检测框架和数据集，受认知科学启发的设计有新意
- **实验充分度**: ⭐⭐⭐⭐ — 7个生成模型+6个验证模型，多层次评估（实体级+实例级），定量+定性分析丰富
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，案例丰富易理解，分类体系直观
- **价值**: ⭐⭐⭐⭐ — 对LLM辅助代码文档生成的质量保障具有直接实用价值，数据集有长期影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](cracking_hallucination_vhd.md)
- [\[ACL 2025\] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge](correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)
- [\[NeurIPS 2025\] Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs](../../NeurIPS2025/hallucination/benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ACL 2025\] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)

</div>

<!-- RELATED:END -->
