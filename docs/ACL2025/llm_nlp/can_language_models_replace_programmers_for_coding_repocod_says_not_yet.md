---
title: >-
  [论文解读] Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'
description: >-
  [ACL2025][LLM 其他][Code Generation] 构建了 RepoCod——一个包含980个来自11个大型 Python 项目的复杂代码生成任务的基准，具有真实的仓库级依赖和平均314个开发者测试用例，揭示了即使最先进的 LLM 也仅能达到不到30%的 Pass@1，远未能替代程序员完成真实编码任务。
tags:
  - "ACL2025"
  - "LLM 其他"
  - "Code Generation"
  - "Repository-Level"
  - "Benchmark"
  - "RAG"
  - "LLM Evaluation"
---

# Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'

**会议**: ACL2025  
**arXiv**: [2410.21647](https://arxiv.org/abs/2410.21647)  
**代码**: 有（论文中提到）  
**领域**: LLM NLP / 代码生成基准  
**关键词**: Code Generation, Repository-Level, Benchmark, RAG, LLM Evaluation  

## 一句话总结

构建了 RepoCod——一个包含980个来自11个大型 Python 项目的复杂代码生成任务的基准，具有真实的仓库级依赖和平均314个开发者测试用例，揭示了即使最先进的 LLM 也仅能达到不到30%的 Pass@1，远未能替代程序员完成真实编码任务。

## 研究背景与动机

### 问题定义
LLM 在 HumanEval 等自包含基准上已达到 90%+ 的 Pass@1，但一个核心问题是：这些高分能否代表 LLM 在真实软件开发中的能力？现有仓库级代码生成基准未能回答这一问题，因为它们存在系统性不足。

### 现有基准的四大不足

**1. 非真实世界任务**：
HumanEval, MBPP 是人工构造的编程题，无法代表真实项目需求驱动的开发任务。

**2. 任务和仓库复杂度低**：
- CrossCodeEval, RepoBench, Long-Code-Arena 仅做单行补全
- CoderEval, DevEval 的函数较短（平均108/86 tokens），仓库规模小（平均152/164文件）
- 基准饱和风险高

**3. 缺乏仓库级依赖**：
- 真实代码中只有27%的函数是自包含的
- CoderEval 和 DevEval 中仅10-31%的任务需要跨文件依赖
- 现有基准以自包含或文件级任务为主

**4. 评估指标不当**：
- CodeBLEU, BLEU 等相似度指标无法判断功能等价性
- 与人类评估的不匹配率很高

## 方法详解

### 整体框架：数据构建流水线

RepoCod 采用三阶段自动化数据收集流程：

**Step I - 仓库选择**：
- Python 为主语言（≥70%）
- ≥2K stars 的开源仓库
- 克隆最新版本（2024年10月）
- 最终选择11个流行项目

**Step II - 目标函数选择**：
结合静态分析和动态分析：
- **静态分析**：使用 tree-sitter 解析测试函数，收集被调用的函数
- **动态分析**：使用 Python `trace` 模块执行测试，捕获间接函数调用
- **过滤条件**：≥10行 docstring + ≥2行函数体

**Step III - 相关测试用例收集**：
两步收集确保完整覆盖：
1. 执行全部测试建立参考结果
2. 对每个目标函数，将其替换为 assertion failure，重新运行所有测试
3. 如果测试结果从通过变为失败，则该测试与目标函数相关

### 基准结构

每个实例包含：
- **目标函数描述**：开发者提供的 docstring
- **仓库快照**：完整源代码（移除目标函数体）
- **相关测试用例**：开发者编写的测试
- **标准答案**：开发者编写的函数体

### 关键统计数据

| 指标 | RepoCod | DevEval | CoderEval | RepoBench |
|------|---------|---------|-----------|-----------|
| 实例数 | 980 | 1,825 | 230 | 23,561 |
| 平均Token数 | **331.6** | 86.3 | 108.2 | 13.7 |
| 圈复杂度 | **9.0** | 3.5 | 4.7 | 1.0 |
| 测试用例/实例 | **313.5** | 2.1 | - | 0 |
| 仓库文件数 | **2,610** | 164 | 152 | - |
| 仓库代码行 | **290,110** | 36,640 | 48,821 | - |

### 上下文复杂度分布

| 基准 | 仓库级 | 文件级 | 自包含 |
|------|--------|--------|--------|
| CoderEval | 10.0% | 53.5% | 36.5% |
| DevEval | 31.3% | 41.2% | 27.5% |
| **RepoCod** | **50.8%** | 18.1% | 31.1% |

RepoCod 有超过一半的任务需要仓库级上下文——远超其他基准。

### 测试执行优化

通过精细化测试收集，将每个实例的测试用例从平均17,974个(全仓库)减少到313个(相关测试)，执行时间从216.9小时降至22.6小时（仅为原来的10.4%）。

## 实验

### 实验设置

**检索方法**：
- **RAGBM25**：基于BM25的稀疏检索
- **RAGDense**：基于 text-embedding-3-small 的密集检索
- **Current File**：仅使用目标函数所在文件作为上下文
- **Baseline**：仅提供签名和 docstring
- **Callees**（Oracle）：提供真实被调用的函数
- **RAGDense-oracle**（Oracle）：用标准答案作为查询

**模型**：10个 SOTA LLM（GPT-4o, DeepSeek-V2.5, Claude 3.5 Sonnet 等）

### 主实验结果

| 模型 | RAGBM25 | RAGDense | Current-File |
|------|---------|----------|-------------|
| CodeLlama-7B | 10.7 | 10.4 | 5.7 |
| CodeLlama-34B | 12.4 | 12.8 | 9.6 |
| DeepSeekCoder-6.7B | 14.0 | 14.1 | 10.9 |
| DeepSeekCoder-33B | 16.7 | 17.1 | 14.9 |
| Claude 3.5 Sonnet | 14.4 | 17.5 | 19.8 |
| DeepSeek-V2.5 | 18.5 | 20.7 | 27.0 |
| GPT-4o-Mini | 15.1 | 15.0 | 18.7 |
| **GPT-4o** | **27.4** | **27.0** | **26.8** |

**核心发现**：即使最好的 GPT-4o 也仅达到27.4% Pass@1——远低于 HumanEval 上的~90%。

### Oracle 设置结果（GPT-4o）

| 方法 | Self | File | Repo | Overall |
|------|------|------|------|---------|
| Baseline | 23.6 | 11.3 | 3.8 | 11.3 |
| RAGBM25 | 39.3 | 31.1 | 18.7 | 27.3 |
| RAGDense | 44.6 | 36.7 | 12.9 | 27.0 |
| Current-File | 39.3 | 35.0 | 16.3 | 26.8 |
| Callees（Oracle） | 35.1 | 31.1 | 12.2 | 22.8 |
| **RAGDense-oracle** | **45.2** | **34.5** | **16.3** | **28.6** |

即使使用标准答案做检索查询（RAGDense-oracle），Pass@1 也仅 28.6%。

### 复杂度对性能的影响

**上下文复杂度**：所有模型在仓库级任务上表现最差，Pass@1 随上下文复杂度升高而下降。

**圈复杂度**：最复杂的设置下（M≥11），最好的 LLM 仅 7.0% Pass@1。

**Token 长度**：长度>232 tokens 时，GPT-4o 也不到10% Pass@1。

### 检索召回率的影响

| 模型 | 召回率=0 | 召回率(0, 0.5] | 召回率(0.5, 1] |
|------|---------|---------------|----------------|
| GPT-4o (BM25) | 20.3 | 16.4 | **41.1** |
| GPT-4o (Dense) | 16.7 | 14.2 | **38.0** |

高召回率（>0.5）的检索能显著提升性能，但低召回率甚至不如不检索。

### 关键发现总结

1. **LLM 在真实仓库级代码生成上远未达标**（最高仅28.6% Pass@1）
2. **更高的依赖召回率能提升性能**，但低召回率的检索可能无帮助
3. **直接提供依赖函数并非最优**——Callees 设置不如 RAG 和 Current-File
4. **即使自包含函数，额外上下文也有帮助**
5. **不同模型有独特解题能力**——每个模型都有其唯一解出的任务
6. **商业模型在 Current-File 设置下表现更优**，而开源模型在 RAG 设置下更好

### 失败原因分析

对30个 GPT-4o 失败样本的分析发现两大类原因：
1. **输入验证不足**：忽略或错误处理意外输入
2. **核心逻辑实现不完整/不正确**：选择次优算法、错误管理对象状态、错误使用并行处理策略

## 亮点与洞察

1. **真正贴近现实的基准**：平均331 tokens的标准答案、2,610文件的仓库、314个测试用例/实例——这些数字远超现有基准，真正反映了软件工程的复杂性
2. **自动化测试收集的巧妙设计**："替换为assertion failure"的方法简单有效地识别了所有相关测试，无需人工干预
3. **Callees 不如 RAG 的反直觉发现**：直接提供依赖函数反而不如检索相似函数，可能因为检索结果提供了更丰富的上下文模式和编码风格信息
4. **50.8% 仓库级依赖**——这一比例远超其他基准，确保基准不会很快被饱和
5. **实用的基准扩展性**：完全自动化的标注流程，使基准容易扩展到更多项目

## 局限性

1. 仅来自11个仓库，Python 单一语言
2. 仅评估了10个模型，代表性有限
3. 数据可能随时间过期（仓库持续演进）
4. 自动化测试收集可能遗漏某些间接调用场景
5. Pass@1 作为唯一指标可能忽略了"接近正确"的解决方案
6. 未考虑 API 版本演化对代码生成的影响（虽然论文讨论了这一相关方向）

## 相关工作

- **自包含基准**：HumanEval, MBPP, APPS
- **仓库级基准**：CrossCodeEval（单行补全）, RepoBench（单行）, CoderEval（短函数）, DevEval（人工标注）, R2E-Eval1（LLM生成测试）, RepoEval
- **版本感知基准**：LibEvolutionEval, GitChameleon, CodeUpdateArena, VersiCode
- **代码LLM**：CodeLlama, DeepSeek-Coder, GPT-4o 等
- **测试收集**：CoderEval（静态分析+人工）, R2E-Eval1（LLM生成）

## 评分 ⭐⭐⭐⭐

构建了目前最具挑战性和最贴近真实世界的代码生成基准，设计理念清晰，统计指标全面碾压现有基准。揭示了 LLM 在真实编码任务上的巨大差距（~90% vs ~28%），对社区有重要的校准价值。测试收集流程的自动化设计值得借鉴。不足之处在于仅覆盖11个Python项目，模型评估数量有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)
- [\[ACL 2025\] Not Quite Sherlock Holmes: Language Model Predictions Do Not Reliably Differentiate Impossible from Improbable Events](not_quite_sherlock_holmes_language_model_predictions_do_not_reliably_differentia.md)

</div>

<!-- RELATED:END -->
