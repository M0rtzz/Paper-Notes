---
title: >-
  [论文解读] MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios
description: >-
  [ACL 2025][代码调试] 本文提出 MLDebugging——首个面向**多库 Python 代码调试**的综合基准，涵盖 126 个 Python 库和 7 种 bug 类型（共 1175 个样本），系统评估主流开源和闭源 LLM 在多库调试场景下的能力，发现当前 LLM 在此任务上仍有很大提升空间。
tags:
  - ACL 2025
  - 代码调试
  - 多库交互
  - LLM评测
  - benchmark
  - Python
---

# MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios

**会议**: ACL 2025  
**arXiv**: [2506.13824](https://arxiv.org/abs/2506.13824)  
**代码**: [有 (GitHub)](https://github.com/hjyTsuki/MLDebugging)  
**领域**: NLP / 代码智能 / 软件工程  
**关键词**: 代码调试, 多库交互, LLM评测, benchmark, Python

## 一句话总结

本文提出 MLDebugging——首个面向**多库 Python 代码调试**的综合基准，涵盖 126 个 Python 库和 7 种 bug 类型（共 1175 个样本），系统评估主流开源和闭源 LLM 在多库调试场景下的能力，发现当前 LLM 在此任务上仍有很大提升空间。

## 研究背景与动机

代码调试是软件工程中的关键任务。虽然 LLM 在代码调试上已取得显著进展，但现有研究和基准几乎完全聚焦于**无库或单库**设定：

- **DebugBench**：来自 LeetCode 算法题的 bug，不涉及库交互
- **xCodeEval / MdEval**：多语言但无多库场景
- **QuickBugs**：来自算法竞赛，同样不涉及库

而在真实世界的软件开发中，使用多个库协作是常态。多库调试引入了两个独特挑战：（1）**理解多个库以定位 bug**；（2）**利用多个库的知识来修复 bug**。例如一个涉及 pandas 和 numpy 之间数据传递的 bug，需要模型理解两个库的 API 及变量类型适配关系。

这启发了 MLDebugging 基准的构建：填补多库代码调试评测的空白。

## 方法详解

### 整体框架

基准构建流水线：源代码收集 → LLM 注解与调试 → Bug 类别平衡 → 质量控制。

### 关键设计

1. **源代码收集**：

    - 从 BigCodeBench 获取涉及 2+ 个库的 Python 编程任务
    - 使用 GPT-4o 生成 1038 个多库代码片段
    - 通过执行测试用例自动识别出 609 个含 bug 的代码

2. **7 种 Bug 分类体系**（从三个视角出发）：

    - **变量传递视角**：Type Mismatch (TM) / Data Transfer Issues (DTI)
    - **库函数参数视角**：Function Parameter Errors (FPE) / Parameter Configuration Errors (PCE) / Function Misuse (FM)
    - **功能理解视角**：Requirement Misunderstanding (RM) / Import Errors (IE)

3. **LLM 辅助注解和调试**：

    - 使用 GPT-4o、DeepSeek-V3、Claude-3.5-sonnet 三个 LLM 分类 bug 并生成修复代码
    - 对失败修复最多重试 5 次（受 test-time scaling 启发）

4. **Bug 类别平衡**：

    - 使用 AST 捕获多库代码结构
    - 从不平衡类别中等比例采样生成新 bug，最终每类约 200 个
    - 成功注入 566 个额外 bug

5. **质量控制**：

    - 4 名经验丰富的程序员交叉检查
    - 修正 119 个 bug 描述、340 个分类错误、手动修复 185 个样本、移除 356 个不合理样本

### 数据分布验证

通过与 StackOverflow 真实 bug 数据的文本嵌入分布对比，验证 MLDebugging 比 DebugBench 更接近真实世界的 bug 分布（余弦相似度 0.731 vs 0.660）。

## 实验关键数据

### 主实验：不同规模 LLM 的调试通过率（%）

| 类别 | Qwen2.5-7B | Qwen2.5-C-7B | Llama3.1-7B | Qwen2.5-72B | DS-V3 | GPT-4 |
|------|-----------|-------------|------------|------------|-------|-------|
| TM | 47.6 | 40.0 | 39.7 | 52.9 | 60.0 | 55.3 |
| DTI | 36.1 | 33.8 | 30.5 | 47.2 | 52.8 | 49.1 |
| PFE | 48.4 | 48.8 | 43.2 | 62.9 | 67.0 | 67.1 |
| PCE | 57.6 | 58.0 | 49.8 | 70.4 | 76.3 | 70.4 |
| FM | 38.2 | 40.4 | 38.8 | 53.8 | 56.2 | 53.0 |
| RM | 12.6 | 7.0 | 5.6 | 16.1 | 23.8 | 21.0 |
| IE | 26.1 | 8.7 | 13.0 | 26.1 | 34.8 | 30.4 |
| **AVG** | 42.7 | 40.6 | 36.7 | 53.7 | **58.7** | 55.6 |

### 消融：相关性分析

| 变量 | 相关系数 | P 值 |
|------|---------|------|
| 代码行数 | -0.0071 | 0.9654 |
| 库数量 | -0.2113 | 0.1906 |
| **库流行度** | **0.4094** | **0.0087** |

### 关键发现

1. **所有 LLM 在 MLDebugging 上都面临挑战**：最高通过率仅 58.7%（DeepSeek-V3），远低于简单代码任务的表现。

2. **规模增加的边际收益递减**：从 7B 到 32B 性能显著提升，但 32B 到 72B 提升趋于平缓甚至下降。这表明多库调试不能仅通过扩大模型规模来解决。

3. **不同 bug 类型的能力差异巨大**：
    - 函数级 bug（TM, DTI, PFE, PCE, FM）通过率 30-76%
    - 库级推理 bug（RM, IE）通过率仅 5-34%，差距近 20%
    - **需求误解（RM）是最难的类别**，最高仅 25.2%（Claude）

4. **库流行度是影响调试难度的最显著因素**（相关系数 0.41，p = 0.0087），代码长度和库数量则不显著。LLM 对训练数据中常见的库掌握更好。

5. **CoT 提示显著提升调试性能**，但**基于蒸馏的推理模型（DeepSeek-R1-Distill）反而下降**，说明单纯 SFT 蒸馏不足以增强此能力。

6. **测试用例和运行时错误信息缺一不可**：同时提供两种反馈时效果最佳最稳定。

## 亮点与洞察

- **填补空白的基准**：首个专注于多库代码调试的评测数据集，126 个库的覆盖面广。
- **7 种 bug 分类体系**：从变量传递、函数参数、功能理解三个维度系统化地刻画多库 bug。
- **分布验证**：通过与 StackOverflow 真实数据的嵌入对比证明基准的真实性。
- **深入的模型行为分析**：按库使用场景（通用算法、数据处理、网络通信等）和库流行度分析 LLM 能力。
- **发现了蒸馏推理模型的局限**：这一反直觉发现对 CoT 蒸馏研究有参考价值。

## 局限与展望

- 数据主要由模型自动生成，虽经人工校验但与真实 bug 仍有差异，未来可引入更多真实数据
- 评测流程需要配置大量外部依赖和复杂环境，耗时较长
- 仅覆盖 Python，未扩展到其他语言的多库场景
- Bug 注入方式可能引入系统性偏差（模型倾向生成某些类型的错误）
- 未探索 agentic 调试方法（如多轮交互、工具调用等）

## 相关工作与启发

- **DebugBench** (Tian et al., 2024)：首个 LLM 调试数据集，基于 LeetCode
- **xCodeEval** (Khan et al., 2024)：多语言多任务代码评测
- **MdEval** (Liu et al., 2024b)：18 种语言的多语言调试
- **BigCodeBench** (Zhuo et al., 2024)：多库代码生成基准，本文的数据来源
- 测试时缩放（test-time scaling）思想启发了重试策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多库调试场景之前完全未被系统研究，分类体系有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ — 20+ 模型覆盖 7B-72B 及闭源模型，按类别/场景/库流行度多维分析，含 CoT 和蒸馏消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，质量控制流程透明
- **价值**: ⭐⭐⭐⭐ — 为代码 LLM 的评测提供了更贴近现实的基准，发现了当前模型的显著短板

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)
- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](../../ACL2026/code_intelligence/precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)

<!-- RELATED:END -->
