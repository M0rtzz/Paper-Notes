---
title: >-
  [论文解读] CodeMEnv: Benchmarking Large Language Models on Code Migration
description: >-
  [ACL 2025][LLM评测][code migration] 提出 CodeMEnv，首个系统评估 LLM 跨环境代码迁移能力的基准，包含 922 个样本、19 个 Python/Java 包、3 个层次化任务（定位不兼容函数→描述变更→迁移代码），9 个 LLM 的平均 Pass@1 仅 26.50%，GPT-4o 最高 43.84%，揭示 LLM 更熟悉新版本函数且存在版本推理逻辑不一致问题。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "code migration"
  - "benchmark"
  - "LLM evaluation"
  - "API versioning"
  - "software engineering"
---

# CodeMEnv: Benchmarking Large Language Models on Code Migration

**会议**: ACL 2025  
**arXiv**: [2506.00894](https://arxiv.org/abs/2506.00894)  
**代码**: [GitHub](https://github.com/xdshen-ai/Benchmark-of-Code-Migration)  
**领域**: LLM评测  
**关键词**: code migration, benchmark, LLM evaluation, API versioning, software engineering

## 一句话总结

提出 CodeMEnv，首个系统评估 LLM 跨环境代码迁移能力的基准，包含 922 个样本、19 个 Python/Java 包、3 个层次化任务（定位不兼容函数→描述变更→迁移代码），9 个 LLM 的平均 Pass@1 仅 26.50%，GPT-4o 最高 43.84%，揭示 LLM 更熟悉新版本函数且存在版本推理逻辑不一致问题。

## 研究背景与动机

**领域现状**: LLM 在代码生成、代码翻译（跨语言）等软件工程任务上表现出色，GPT-4、DeepSeek-V3、CodeLlama 等在 HumanEval 等基准上持续刷新记录。

**被忽视的场景**: 代码迁移（adapting code to run in different environments）是实际开发中的高频痛点——用户从 GitHub 获取代码后常因库版本不兼容而需大量手动适配，但该场景几乎未被系统研究。

**核心矛盾**: 库的持续演进导致 API 变更频繁（如 NumPy 1.26→2.0 中 `compare_chararrays` 从顶层模块移至 `numpy.char` 子模块），同一功能在不同版本中的实现方式可能完全不同，而现有基准几乎全部聚焦跨语言翻译而非跨版本迁移。

**已有尝试不足**: Google 研究者 Ziftci et al. (2025) 探索了自动化迁移，Amazon Q Developer 针对 Java 8/11→17 提供了工具，但缺乏面向 LLM 的全面评估基准。

**切入角度**: 从 API 函数变更的三种类型（新增、废弃、替换）出发，设计覆盖"定位→理解→修复"全链路的层次化评估体系。

**核心 idea**: 代码迁移能力需要 LLM 同时具备版本感知的 API 知识、跨版本推理能力和代码生成能力，三者缺一不可，需要专门的基准来分别衡量。

## 方法详解

### 基准构建流程

**Step 1 — 数据收集**: 从 19 个包（11 Python + 8 Java）的官方文档版本发布说明中系统梳理函数变更，收集 212 个 Python 函数变更和 114 个 Java 函数变更，并通过在多个版本上实际执行来确定函数的兼容版本范围。

**Step 2 — 代码生成**: 用 GPT-4 基于函数变更信息和使用说明生成目标代码。对"新增"类变更生成 New2Old 样本（新环境代码→迁移到旧环境），对"废弃"类变更生成 Old2New 样本，对"替换"类变更双向生成。

**Step 3 — 测试用例生成**: GPT-4 为每个代码样本生成 3 个测试用例，在原始代码上执行获取 ground-truth 输出；若测试用例有误则迭代修复最多 3 轮，仍失败则丢弃该样本。

### 三个层次化任务

1. **Task-1 定位不兼容函数**: 给定代码片段和目标环境版本，模型需精确指出所有不兼容函数。分 easy（仅 1 个不兼容函数）和 hard（2-3 个不兼容函数）两个难度。
2. **Task-2 描述函数变更**: 对每个不兼容函数，模型需回答变更类型（新增/废弃/替换）、变更版本号（允许 ±0.5 误差）、替换函数名。
3. **Task-3 代码迁移**: 修改代码使其在目标环境中运行正确，以通过全部 3 个单元测试为准。分 Old2New 和 New2Old 两个方向。

### 评估方法

- Task-1/Task-2: Agent-based 评估，将模型预测与 ground-truth 精确比对。
- Task-3: 单元测试评估，迁移后代码需在 3 个测试用例上输出与原始代码完全一致的结果；报告 Pass@1 和 Pass@5。

## 实验关键数据

### 表1: Task-1 & Task-2 准确率（%）

| 模型 | Task-1 Python(easy) | Task-1 Python(hard) | Task-1 Java | Task-1 平均 | Task-2 平均 |
|------|-----------|-----------|------|------|------|
| GPT-3.5-Turbo | **85.10** | **32.98** | 80.89 | **66.32** | 34.13 |
| GPT-4o | 70.71 | 25.65 | 81.19 | 59.18 | 37.02 |
| DeepSeek-v3 | 78.48 | 26.17 | 82.08 | 62.24 | **42.06** |
| Llama-3.1-70B | 75.51 | 29.84 | 81.19 | 62.18 | 35.44 |
| Llama-3.1-8B | 70.71 | 21.99 | 67.16 | 53.29 | 23.79 |

### 表2: Task-3 代码迁移 Pass@1（%）

| 模型 | Old2New easy | Old2New hard | New2Old easy | New2Old hard |
|------|-------------|-------------|-------------|-------------|
| GPT-4o | **43.84** | **26.83** | **31.60** | **22.94** |
| DeepSeek-v3 | 41.20 | 20.73 | 29.60 | 14.68 |
| Llama-3.1-70B | 32.88 | 19.51 | 28.80 | 17.43 |
| Code Llama-34B | 35.62 | 21.95 | 29.60 | 15.76 |
| GPT-3.5-Turbo | 26.03 | 7.32 | 24.80 | 7.34 |
| Qwen2.5-Coder-7B | 32.19 | 14.63 | 29.20 | 8.26 |

### 关键发现

- **整体表现低**: 9 个 LLM 在迁移任务上的平均 Pass@1 仅 26.50%，说明跨版本代码迁移远未解决。
- **新版本偏好**: 所有模型在 Old2New 方向上的表现均显著优于 New2Old（如 GPT-4o easy: 43.84% vs 31.60%），说明 LLM 训练数据中新版本函数占比更高。
- **定位 ≠ 迁移**: GPT-3.5-Turbo 在 Task-1（定位不兼容函数）上最强（66.32%），但迁移任务（Task-3 hard）仅 7.32%，说明"找到问题"和"解决问题"是截然不同的能力。
- **版本推理不一致**: 案例研究中 Llama-3.1-8B 和 GPT-3.5-Turbo 在目标环境为 NumPy 1.16 时错误引用了 1.17/1.18 版本的变更，暴露版本顺序推理的系统性弱点。
- **多不兼容函数难度陡增**: hard 集（2-3 个不兼容函数）的 Pass@1 典型下降 50-70%，模型难以同时处理多个不兼容点。
- **错误类型分布**: CallError（仍调用不兼容函数）占比最大（如 Llama-3.1-8B 达 50.8%），其次是 RunError（无限循环，DeepSeek-v3 达 33.0%）和 WrongAnswer（GPT-4o 达 19.4%）。

## 亮点与洞察

- **首个全面的代码迁移基准**: 填补了 LLM 评估中跨版本迁移场景的空白，3 个层次化任务设计精巧地分解了迁移能力的不同维度。
- **真实数据来源**: 函数变更从 19 个包的官方文档手动整理而非合成，保证了评估的实用性和可信度。
- **任务间能力解耦发现**: 首次定量证明了"定位不兼容 API"与"实际完成代码迁移"是完全不同的能力，为未来改进 LLM 提供了明确方向。
- **New2Old 方向的独特价值**: 揭示了一个反直觉的挑战——将新代码适配到旧环境比反过来更难，这对维护遗留系统的开发者有重要启示。

## 局限性

- **数据规模偏小**: 总共 922 个样本（Python 587 + Java 335），Java 部分仅有 easy 难度，对某些包的覆盖可能不足。
- **语言覆盖有限**: 仅支持 Python 和 Java，未涉及 JavaScript/TypeScript、Rust、Go 等广泛使用的语言。
- **代码由 GPT-4 生成**: 测试代码和测试用例均由 GPT-4 生成而非从真实项目提取，可能引入分布偏差且不够反映真实迁移场景的复杂度。
- **评估局限**: Task-3 仅通过 3 个测试用例判断正确性，可能遗漏部分边界情况；Agent-based 评估的准确性依赖评估 agent 自身的能力。

## 相关工作对比

### vs. CodeUpdateArena (Liu et al. 2024)
CodeUpdateArena 聚焦于 LLM 在 API 更新后的知识编辑，关注模型是否"知道"API 变了。CodeMEnv 范围更广，不仅要求模型知道变更，还要求能定位不兼容代码并完成实际迁移，覆盖了从识别到修复的完整链路。

### vs. Amazon Q Developer (Code Migration Tool)
Amazon Q 是面向 Java 8/11→17 升级的生产级工具，聚焦单一语言的特定版本跳跃。CodeMEnv 是评估基准而非工具，覆盖 19 个包的多种版本组合，且支持双向迁移（Old2New + New2Old），评估范围更系统全面。

### vs. 跨语言代码翻译基准（Yuan et al. 2024, Eniser et al. 2024）
跨语言翻译关注不同编程语言间的转换（如 Python→Rust），不涉及库版本兼容性。CodeMEnv 关注的是同一语言内因库版本变化而需要的代码适配，是一个互补但本质不同的问题维度。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统化的跨版本代码迁移基准，问题定义清晰且任务层次设计合理
- 实验充分度: ⭐⭐⭐ 测试了 9 个主流 LLM 且分析全面，但数据规模偏小、语言覆盖有限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，案例分析和错误分析直观有力
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 在实际代码迁移上的关键短板，对评估社区和工具开发者均有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ACL 2025\] Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
