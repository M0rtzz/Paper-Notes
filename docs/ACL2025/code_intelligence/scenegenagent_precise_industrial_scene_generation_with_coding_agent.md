---
title: >-
  [论文解读] SceneGenAgent: Precise Industrial Scene Generation with Coding Agent
description: >-
  [ACL 2025][代码智能][工业场景生成] 提出 SceneGenAgent，一个基于 LLM 的代码生成 Agent，通过结构化布局规划、布局验证和迭代优化流程，利用 C# 代码精确生成工业场景，在真实工业任务上达到 81% 成功率，并构建 SceneInstruct 数据集使开源 LLM 接近 GPT-4o 水平。
tags:
  - "ACL 2025"
  - "代码智能"
  - "工业场景生成"
  - "LLM代码生成"
  - "布局规划"
  - "迭代优化"
  - "场景建模"
---

# SceneGenAgent: Precise Industrial Scene Generation with Coding Agent

**会议**: ACL 2025  
**arXiv**: [2410.21909](https://arxiv.org/abs/2410.21909)  
**代码**: [https://github.com/THUDM/SceneGenAgent](https://github.com/THUDM/SceneGenAgent)  
**领域**: 代码智能  
**关键词**: 工业场景生成, LLM代码生成, 布局规划, 迭代优化, 场景建模  

## 一句话总结

提出 SceneGenAgent，一个基于 LLM 的代码生成 Agent，通过结构化布局规划、布局验证和迭代优化流程，利用 C# 代码精确生成工业场景，在真实工业任务上达到 81% 成功率，并构建 SceneInstruct 数据集使开源 LLM 接近 GPT-4o 水平。

## 研究背景与动机

**领域现状**：工业场景建模是仿真和制造流程的基础需求。近年来，大语言模型（LLM）在从文本描述生成 3D 场景方面取得了显著进展，尤其是在家居、室内等通用场景中表现良好。

**现有痛点**：然而，工业场景与通用场景有本质区别——工业场景对精确测量和定位有严格要求，需要精确的尺寸、间距和空间排布。直接使用通用场景生成方法无法满足工业级精度需求，LLM 在空间推理和精确数值计算上的短板成为主要瓶颈。

**核心矛盾**：LLM 擅长理解自然语言描述和推理高层语义，但在需要精确数值计算、空间几何布局的场景中表现不佳。直接让 LLM 输出坐标和尺寸往往不精确，而工业场景不允许任何布局误差。

**本文目标**：设计一个 LLM-based Agent 系统，(1) 将场景生成转化为结构化代码生成问题以确保精确性；(2) 引入自动验证和迭代修正机制提高生成质量；(3) 构建训练数据集让开源模型也具备工业场景生成能力。

**切入角度**：作者观察到代码天然支持精确的数值计算和逻辑控制。将场景生成任务转化为 C# 代码编写问题，利用代码的可执行性来保证布局的精确性，同时利用编译器反馈自动验证和修正生成结果。

**核心 idea**：用 LLM 生成 C# 代码来描述工业场景布局，结合"规划-验证-修正"的 Agent 闭环来保证精确性。

## 方法详解

### 整体框架

SceneGenAgent 是一个多阶段的 LLM Agent 框架，整体流程为：输入工业场景的自然语言描述 → LLM 进行结构化布局规划（生成可计算的布局格式）→ 将布局规划转化为 C# 代码 → 执行代码并进行布局验证 → 如果验证失败则进入迭代修正直到满足要求 → 输出最终的工业场景。系统核心依赖三个模块：结构化布局规划、自动布局验证和迭代优化修正。

### 关键设计

1. **结构化可计算布局规划（Structured & Calculable Layout Planning）**:

    - 功能：将自然语言场景描述转化为结构化的、可精确计算的布局表示
    - 核心思路：定义一套结构化的布局描述格式，让 LLM 输出的布局信息不是自由文本，而是包含精确数值的结构化数据（如物体类型、位置坐标、尺寸参数等）。这种格式天然可以用代码表达和计算，避免了自然语言描述的模糊性。LLM 在生成 C# 代码时，可以利用编程语言的计算能力来执行精确的空间排布计算。
    - 设计动机：工业场景要求毫米级精度，自然语言描述无法承载精确的数值布局信息。结构化格式使得布局可以被程序化验证，同时代码形式保证了数值计算的精确性。

2. **自动布局验证（Layout Verification）**:

    - 功能：自动检测生成的场景布局是否满足工业要求
    - 核心思路：将生成的 C# 代码编译执行后，系统自动检查物体间的碰撞检测、间距约束、边界条件等。利用编译器的报错信息和运行时检查结果来判断布局的合法性和正确性。验证模块会产生结构化的错误报告，为后续修正提供精确的反馈信号。
    - 设计动机：工业场景中，物体之间的碰撞、不满足安全距离等都是不允许的。自动验证替代人工检查，实现端到端的自动化生成流程，同时为迭代修正提供可靠的反馈。

3. **迭代优化修正（Iterative Refinement）**:

    - 功能：根据验证反馈自动修正布局错误
    - 核心思路：当布局验证检测到问题时，系统将错误信息反馈给 LLM，LLM 根据具体的错误类型和位置信息重新调整代码。这个过程可以多轮进行，每轮修正后再次验证，直到所有约束都被满足或达到最大迭代次数。
    - 设计动机：一次性正确生成复杂工业场景非常困难，迭代修正机制允许模型逐步完善布局，类似于人类工程师的反复调整过程。利用编码 Agent 的强项——代码可以精确修改、局部调整，而不需要重新生成整个场景。

### 训练策略

为了让开源模型也具备工业场景生成能力，作者构建了 SceneInstruct 数据集。该数据集包含工业场景描述和对应的高质量 C# 代码对，用于微调开源 LLM（如 Llama3.1-70B）。微调的目标是让模型学会从场景描述直接生成结构化的布局代码，并理解工业场景的空间约束。

## 实验关键数据

### 主实验

实验在真实工业场景生成任务上进行评估，由工程师编写场景描述并人工检查生成结果的正确性。

| 模型 | 方法 | 成功率 |
|------|------|--------|
| GPT-4o | 直接生成（无Agent） | ~50% |
| GPT-4o | + SceneGenAgent | **81.0%** |
| Llama3.1-70B | 直接生成 | 较低 |
| Llama3.1-70B | + SceneInstruct 微调 + SceneGenAgent | 接近 GPT-4o |
| 其他开源LLM | + SceneInstruct 微调 | 显著提升 |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| Full SceneGenAgent | 81.0% | 完整框架 |
| w/o 布局验证 | 下降明显 | 去掉验证后无法保证精确性 |
| w/o 迭代修正 | 中等下降 | 只有一次机会生成正确布局 |
| w/o 结构化格式 | 大幅下降 | 自由文本布局精度无法保证 |
| SceneInstruct 微调 | 大幅提升 | 开源模型受益最大 |

### 关键发现

- 结构化布局格式是性能提升最关键的设计，去掉后成功率大幅下降，因为精确的数值计算完全依赖代码的可执行性
- 迭代修正机制平均需要 2-3 轮即可收敛，说明大部分错误可以通过少量修正解决
- SceneInstruct 微调对开源模型效果显著，Llama3.1-70B 微调后在 SceneGenAgent 框架下接近 GPT-4o 的表现，显示领域专用数据对开源模型的巨大价值
- 工程师评估表明 SceneGenAgent 生成的场景能满足大多数实际工业需求

## 亮点与洞察

- **代码即精度保证**：将场景生成转化为代码生成是非常巧妙的设计。代码不仅可以表达精确的数值关系，还可以通过编译器自动发现错误——这是自然语言输出无法做到的。这个思路可以迁移到所有需要精确数值输出的任务中（如电路设计、建筑布局等）。
- **验证-修正闭环**：Agent 的核心在于反馈闭环。与纯生成式方法不同，SceneGenAgent 利用编译器和运行时检查构建了自动反馈机制，让修正过程可以精确到具体的代码行和数值。
- **数据飞轮效应**：SceneInstruct 展示了一个实际可行的"强模型标注→弱模型学习"的知识蒸馏路径，用 GPT-4o 生成的高质量代码来训练开源模型。

## 局限与展望

- 当前评估主要基于人工检查，缺乏自动化的评估指标体系，规模化评估成本高
- C# 代码与特定工业软件（如 Unity 等）绑定，泛化到其他工业软件需要适配
- SceneInstruct 的规模和多样性可能有限，对非常复杂或罕见的工业场景覆盖不足
- 多轮迭代修正的时间开销在实时性要求高的场景中可能成为瓶颈
- 未来可以探索多模态输入（如示意图+文本描述）来增强场景理解能力

## 相关工作与启发

- **vs 通用 3D 场景生成方法**：SceneScape、Text2Room 等方法面向通用场景，不关注精确度。SceneGenAgent 专为工业精确度设计，是场景生成从"看起来对"到"测量正确"的关键进步。
- **vs 代码生成 Agent（如 SWE-Agent）**：SWE-Agent 做代码修复，SceneGenAgent 将代码 Agent 范式迁移到场景生成，验证了代码 Agent 在精确控制类任务中的通用性。
- 这篇论文启发了一个思路：对于需要精确输出的任务，都可以考虑引入"代码中间表示+编译验证"的范式。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将代码生成引入工业场景建模是新视角，但 Agent 闭环框架本身并不新颖
- 实验充分度: ⭐⭐⭐ 真实工业任务评估是亮点，但规模较小且依赖人工评判
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机讲述得好
- 价值: ⭐⭐⭐⭐ 对工业场景生成有直接应用价值，代码-Agent 范式可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ACL 2025\] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)
- [\[ACL 2025\] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)
- [\[ACL 2025\] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)

</div>

<!-- RELATED:END -->
