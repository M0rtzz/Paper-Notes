---
title: >-
  [论文解读] FuncBenchGen: 面向可靠基准测试的无污染可控评估框架
description: >-
  [ICLR 2026][视频理解][工具增强LLM] 提出 FuncBenchGen 框架，通过将多步函数调用建模为 DAG 图遍历问题，实现无数据污染、可精细控制任务难度的 LLM 工具使用能力评估，并揭示了推理模型在长调用链和连接型干扰函数下的关键失败模式。
tags:
  - ICLR 2026
  - 视频理解
  - 工具增强LLM
  - 多步函数调用
  - benchmark
  - 数据污染
  - DAG图遍历
---

# FuncBenchGen: 面向可靠基准测试的无污染可控评估框架

**会议**: ICLR 2026  
**arXiv**: [2509.26553](https://arxiv.org/abs/2509.26553)  
**领域**: 视频理解  
**关键词**: 工具增强LLM, 多步函数调用, benchmark, 数据污染, DAG图遍历

## 一句话总结

提出 FuncBenchGen 框架，通过将多步函数调用建模为 DAG 图遍历问题，实现无数据污染、可精细控制任务难度的 LLM 工具使用能力评估，并揭示了推理模型在长调用链和连接型干扰函数下的关键失败模式。

## 研究背景与动机

现有的工具增强语言模型（TaLM）评估基准面临两大核心问题：

**数据污染风险**：现有基准（如 API-Bank、BFCLv4、ToolBench 等）的问答对可能在预训练数据或测试时网页搜索中被泄露，导致评估结果不可靠

**任务复杂度不可控**：现有基准缺乏对任务难度的精细控制，无法系统地分析哪些因素最显著影响模型性能

| 基准 | 无污染 | 函数集大小控制 | 依赖深度控制 | 干扰函数类型控制 |
|------|--------|----------------|--------------|------------------|
| API-Bank | ✗ | ✗ | ✗ | ✗ |
| BFCLv4 | ✗ | ✓ | ✗ | ✗ |
| ToolBench | ✗ | ✓ | ✗ | ✗ |
| **FuncBenchGen** | **✓** | **✓** | **✓** | **✓** |

## 方法详解

### 整体框架

FuncBenchGen 将多步函数调用形式化为**有向无环图（DAG）遍历问题**。给定函数集 $\mathcal{F}=\{f_1, f_2, \ldots, f_n\}$，输入变量集 $\mathcal{V}_{input}$ 和目标变量 $v_T$，LLM 需要通过迭代执行函数调用序列来确定 $v_T$ 的值。

### 关键设计

**1. 图结构生成**：接受四个控制参数：
- $n^{\text{core}}$：核心节点数（解决任务所需的函数数量）
- $d$：依赖深度
- $n^{\text{conn}}$：连接型无关节点数（CIN，与核心节点共享类型兼容变量）
- $n^{\text{dis}}$：断开型无关节点数（DIN，与核心节点无连接）

**2. 函数模式创建**：每个 DAG 节点转化为函数定义，包含随机生成的函数名、类型标注的输入/输出参数和自然语言描述。函数通过语义类型和子类型匹配建立连接。

**3. 确定性执行**：每个变量被赋予三位随机整数值。函数仅在输入值完全正确时返回正确输出，否则返回随机错误值，模拟真实 API 的静默失败行为。

### 缓解策略

针对最常见的失败模式（使用未知/错误值），提出简单的**变量值重述策略**：每次函数返回时，不仅返回输出值，还附带当前所有已知变量值列表。

## 实验关键数据

### 主实验：不同核心节点数下的成功率

| 模型 | 5 核心节点 | 10 核心节点 | 20 核心节点 |
|------|-----------|------------|------------|
| GPT-5 | 72.5% | 38.2% | 15.0% |
| Gemini-2.5-Pro | 46.5% | 14.4% | 6.0% |
| GPT-5-mini | 16.0% | 7.6% | 4.2% |
| Qwen3 | 11.0% | 8.2% | 3.8% |
| GPT-4.1 | 12.0% | 2.2% | 0.2% |

### 失败类型分析

| 失败类型 | GPT-5 | Gemini-2.5-Pro | Qwen3 | GPT-4.1 |
|---------|-------|----------------|-------|---------|
| 函数不存在 | 0.0% | 2.4% | 0.0% | 0.0% |
| 输入参数数量错误 | 0.0% | 0.2% | 0.1% | 0.0% |
| 使用未知值 | 79.6% | 69.1% | 74.0% | 73.2% |
| 使用错误值 | 20.4% | 28.3% | 25.8% | 26.8% |

### 依赖深度影响

- GPT-5 在深度为 1（星型结构）时接近 90% 成功率，深度增至 4-8 时降至不到 30%
- 路径结构（深度 8-9）相比中等分支结构（深度 5-7）略有改善，表明分支少的序列化调用链更易处理
- 更大思考预算（medium vs minimal）在复杂场景中显著提升性能

### 关键发现

1. **推理模型显著优于通用模型**：GPT-5 在 5 核心节点时达到 72.5%，而 GPT-4.1 仅 12.0%
2. **性能随序列长度急剧下降**：GPT-5 从 72.5%（5 节点）降至 15.0%（20 节点）
3. **连接型干扰函数（CIN）危害最大**：因共享类型兼容变量，模型难以区分相关/无关函数
4. **缓解策略显著有效**：变量重述使 GPT-5 成功率从 62.5% 提升至 81.3%
5. **GPT-5 调用效率不佳**：即使成功，也多调用约 10% 的冗余函数
6. **充足推理预算至关重要**：minimal 思考预算下，GPT-5 在有干扰函数时成功率低于 20%

## 亮点与洞察

1. **优雅的形式化**：将工具使用抽象为 DAG 遍历问题，实现评估维度的正交分解
2. **失败分析深刻**：揭示所有模型最大的短板是**状态追踪**而非语法理解——79.6% 的 GPT-5 错误来自使用未知变量值
3. **简单有效的缓解**：仅重述已知变量值（不提供新信息），就能大幅提升性能，说明 LLM 的工作记忆是多步工具使用的核心瓶颈
4. **对 MCP 生态的警示**：即使是断开型干扰函数在函数集增大到 40 时也严重降低 GPT-5 性能（<10%），意味着当前 LLM 尚未准备好处理大规模 MCP 服务器
5. **失败模式差异揭示模型性格**：失败时 GPT-5 倾向于多次尝试（调用更多函数），而 Gemini-2.5-Flash 则倾向于放弃（调用更少函数）

## 局限性

1. 合成函数与真实 API 存在差距，真实场景中函数语义更复杂
2. 仅考虑 DAG 结构，未覆盖条件逻辑和循环等更复杂控制流
3. 每个函数固定一个输出变量，不支持多输出函数
4. 未评估开源小模型在该任务上的能力
5. 函数间通过类型匹配建立连接，缺乏自然语言语义推理的评估
6. 未考虑模型在调用失败后的修复和重试能力

## 评分 ⭐⭐⭐⭐

系统性强、分析深入的评估框架工作。核心贡献在于揭示了 LLM 多步工具使用中的状态追踪瓶颈，对 Agent 系统设计有重要指导意义。DAG 建模的抽象优雅，缓解策略虽简单但洞察深刻。不足之处在于合成任务与真实场景仍有距离，且分类为视频理解领域但论文主题更偏向 LLM Agent 评估。

<!-- RELATED:START -->

## 相关论文

- [VideoNSA: Native Sparse Attention Scales Video Understanding](videonsa_native_sparse_attention_scales_video_understanding.md)
- [From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)
- [A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [Mamba-3: Improved Sequence Modeling using State Space Principles](mamba-3_improved_sequence_modeling_using_state_space_principles.md)

<!-- RELATED:END -->
