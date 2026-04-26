---
title: >-
  [论文解读] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models
description: >-
  [ACL2025][LLM/NLP][tool learning] ToolCoder 将工具学习重新建模为代码生成任务，借鉴软件工程的需求分析、模块化设计、代码复用和错误诊断原则，让 LLM 通过生成并执行结构化 Python 代码来调用外部工具，在 RestBench 和 API-Bank 基准上显著超越 ReAct、CodeAct 等现有方法。
tags:
  - ACL2025
  - LLM/NLP
  - tool learning
  - code generation
  - LLM agent
  - software engineering
  - error reflection
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models

**会议**: ACL2025  
**arXiv**: [2502.11404](https://arxiv.org/abs/2502.11404)  
**代码**: [GitHub](https://github.com/dhx-2020/ToolCoder)  
**领域**: llm_nlp  
**关键词**: tool learning, code generation, LLM agent, software engineering, error reflection

## 一句话总结

ToolCoder 将工具学习重新建模为代码生成任务，借鉴软件工程的需求分析、模块化设计、代码复用和错误诊断原则，让 LLM 通过生成并执行结构化 Python 代码来调用外部工具，在 RestBench 和 API-Bank 基准上显著超越 ReAct、CodeAct 等现有方法。

## 背景与动机

1. **工具学习是 LLM 关键能力**：LLM 需要与外部工具/API 交互才能完成复杂的现实任务，工具学习（tool learning）已成为 LLM agent 的核心研究方向。
2. **现有方法依赖手工提示**：ReAct、Chameleon 等方法采用"计划-执行-观察"的顺序范式，严重依赖精心设计的自然语言 prompt，在复杂多步任务中规划能力不足。
3. **缺乏精确的错误诊断机制**：当工具调用执行失败时，现有框架无法精确定位错误原因，也无法针对性地进行修正，只能简单重试或放弃。
4. **无法积累和复用经验**：每个查询都是孤立处理的，之前成功执行的方案不会被保存和复用，导致对相似问题反复从零求解。
5. **代码预训练提升推理能力**：已有研究表明代码预训练能显著增强 LLM 的 chain-of-thought 能力，代码范式天然适合结构化推理和任务分解。
6. **软件工程原则的启发**：软件工程中的需求分析、模块化设计、单元测试、版本管理等原则为工具学习提供了系统性方法论，但尚未被充分利用。

## 方法详解

ToolCoder 框架包含四个核心阶段，类比软件工程的完整开发流程：

### 阶段一：Task-to-Code Transformation（需求分析）

将自然语言查询 $q$ 转换为结构化 Python 函数脚手架 $c$：

- 生成描述性函数名和参数列表，明确任务的输入输出规范
- 编写 docstring 记录函数用途、参数描述和返回值类型
- 函数体留空，作为后续阶段的占位符
- 同时生成 `__main__` 入口代码，定义具体调用参数

例如查询"查找 Sofia Coppola 导演的电影数量"会被转化为 `get_directed_movie_count(director_name: str) -> int` 函数框架。

### 阶段二：Subtask Planning and Tool Selection（模块化设计）

基于函数脚手架 $c$ 和候选工具集 $\mathcal{T}$，进行子任务规划和工具选择：

- **子任务分解**：将高层任务拆解为具体可执行的子任务 $\{s_1, s_2, \dots, s_m\}$，以代码注释的形式嵌入脚手架
- **工具匹配**：分析每个子任务所需的 API 功能，从工具集中选择匹配的工具，生成包含工具调用序列和数据流的伪代码 $c_p$
- 每个工具调用用子函数占位符表示，具体实现延迟到下一阶段

### 阶段三：Implementation and Execution（编码实现）

- **代码生成**：为每个子函数占位符生成具体实现代码 $F = \mathcal{M}_{CG}(c_p, \mathcal{T}_s, \mathcal{F})$，参考工具 API 文档和可复用代码库 $\mathcal{F}$
- **代码执行**：在 Python 环境中执行完整程序，获得最终响应 $r$ 或异常 $e$
- **函数仓库**：成功执行的子函数被存入可复用函数仓库 $\mathcal{F}$，供后续类似任务直接调用，避免重复开发

### 阶段四：Error Reflection（错误诊断与修复）

- **计划重制（Plan Reformulation）**：检测计划中是否包含无效或不存在的工具调用，通过交叉引用工具集进行纠正
- **代码审查（Code Review）**：利用 Python 的 traceback 机制精确定位执行错误位置和原因，进行针对性修复
- 最多迭代 3 次直至执行成功或达到上限

## 实验关键数据

### 表1：RestBench & API-Bank 主实验结果（gpt-4o-mini）

| 方法 | TMDB Success% | TMDB Accuracy% | TMDB Path% | Spotify Success% | Spotify Path% | API-Bank LV1% | API-Bank LV2% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ReAct | 76.0 | 48.0 | 50.0 | 68.42 | 52.63 | 73.93 | 56.30 |
| Chameleon | 75.0 | 45.0 | 52.0 | 70.18 | 63.16 | 74.87 | 37.04 |
| CodeAct | 80.0 | 56.0 | 67.0 | 71.93 | 66.67 | 75.94 | 54.07 |
| **ToolCoder** | **85.0** | **78.0** | **83.0** | **87.72** | **78.95** | **83.08** | **62.41** |
| w/o Reusable Repo | 83.0 | 71.0 | 78.0 | 78.95 | 71.93 | - | - |
| w/o Error Reflection | 75.0 | 65.0 | 77.0 | 73.68 | 70.18 | 79.85 | 58.02 |

- 在 Spotify 上，ToolCoder 相比 CodeAct 成功率提升 15.79%，准确率提升 22%
- 消融实验表明：移除错误反射使 TMDB 成功率下降 10%，移除函数复用使准确率下降 7%

### 表2：ToolAlpaca Real-world 泛化实验

| 方法 | Procedure | Response | Overall |
|:---|:---:|:---:|:---:|
| ReAct | 64.86 | 60.81 | 54.05 |
| CodeAct | 68.92 | 58.11 | 56.94 |
| **ToolCoder** | **78.38** | **75.68** | **72.97** |

- 在真实 API 场景下 Overall 得分较 CodeAct 提升 16.03 个百分点

### 表3：开源模型实验（部分）

| 模型 | 方法 | Success | Accuracy | Path |
|:---|:---|:---:|:---:|:---:|
| Qwen2.5-Coder-32B | CodeAct | 92.0 | 82.0 | 85.0 |
| Qwen2.5-Coder-32B | **ToolCoder** | **96.0** | **90.0** | **91.0** |

- 在开源代码模型上优势更加明显，验证了代码特化能力对 ToolCoder 的增益

## 亮点

- **视角新颖**：将工具学习系统性地映射为软件工程流程（需求分析→模块设计→编码实现→测试调试），提供了清晰的方法论框架
- **错误诊断精准**：利用 Python traceback 进行结构化错误定位，比自然语言反思更加可靠和可解释
- **函数复用机制实用**：通过维护可复用函数仓库实现经验积累，随使用量增加性能持续提升（累计成功率曲线证明）
- **效率不降反升**：在大幅提升性能的同时，平均 API 调用次数与 ReAct 基本持平（7.69 vs 7.78）
- **代码模型加成显著**：在 Coder 系列模型上提升幅度明显大于通用模型，说明框架能充分利用代码特化能力

## 局限与展望

- **依赖高质量 API 文档**：当工具文档不完整、模糊或不一致时，模型生成正确代码的能力会显著下降
- **全局规划缺乏动态适应**：采用一次性全局规划策略，难以应对实时变化的任务需求或部分可观测环境
- **工具规模扩展性有限**：当工具数量大且存在复杂依赖关系时，组合搜索空间可能导致次优方案
- **仅在闭源模型上验权主实验**：主要结果基于 gpt-4o-mini，开源模型仅在一个数据集上验证
- **代码生成可能引入安全风险**：自动生成和执行代码的范式在生产环境中需要额外的沙箱和权限控制

## 与相关工作的对比

### vs ReAct (Yao et al., 2023)

ReAct 通过交替生成推理轨迹和动作来完成工具调用，是文本型工具学习的代表方法。ToolCoder 的核心区别在于：(1) 用结构化代码替代自然语言推理，提供更精确的任务规划；(2) 利用 Python traceback 进行错误诊断而非语言反思；(3) 支持函数复用积累经验。实验表明 ToolCoder 在 TMDB 上成功率比 ReAct 高 9%，准确率高 30%。

### vs CodeAct (Wang et al., 2024)

CodeAct 同样使用代码作为动作空间，是代码型工具学习的最强基线。但 CodeAct 采用单次代码生成，缺乏迭代优化和意图对齐。ToolCoder 通过完整的软件工程流水线（需求→设计→实现→测试）对其进行了系统性增强，在 Spotify 上成功率提升 15.79%，在 API-Bank LV2 上提升 8.34%。

### vs RestGPT (Song et al., 2023)

RestGPT 引入粗到细的在线规划模块，但仍局限于文本范式。ToolCoder 在结构化规划和执行可靠性上全面超越，且 API 调用效率更优（7.69 vs 9.04）。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将软件工程完整流程映射到工具学习的思路清晰且有启发性
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 3 个基准、7 个基线、多模型规模和消融实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示丰富，类比软件工程易于理解
- 价值: ⭐⭐⭐⭐ — 提供了实用且可扩展的工具学习框架，对 LLM agent 开发有直接参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [\[ACL 2025\] Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'](can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)

<!-- RELATED:END -->
