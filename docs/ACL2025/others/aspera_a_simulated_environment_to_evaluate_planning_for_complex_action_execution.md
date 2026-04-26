---
title: >-
  [论文解读] ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution
description: >-
  [ACL 2025][动作规划] 提出 ASPERA 框架及 Asper-Bench 基准，用于评估 LLM 在自定义助手库约束下执行复杂多步动作规划（程序生成）的能力，揭示了相比自由代码生成，基于自定义 API 库的程序生成对 LLM 构成显著更大的挑战。
tags:
  - ACL 2025
  - 动作规划
  - 数字助手
  - 程序生成
  - 模拟环境
  - 基准测试
---

# ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution

**会议**: ACL 2025  
**arXiv**: [2507.15501](https://arxiv.org/abs/2507.15501)  
**代码**: 无  
**领域**: LLM Agent / 代码生成 / 规划评估  
**关键词**: 动作规划、数字助手、程序生成、模拟环境、基准测试

## 一句话总结

提出 ASPERA 框架及 Asper-Bench 基准，用于评估 LLM 在自定义助手库约束下执行复杂多步动作规划（程序生成）的能力，揭示了相比自由代码生成，基于自定义 API 库的程序生成对 LLM 构成显著更大的挑战。

## 研究背景与动机

- **领域现状**：大语言模型越来越多地被用于驱动数字助手，帮助用户完成复杂的多步骤操作任务，如日程管理、设备控制、信息查询等。这些助手需要将用户的自然语言请求转化为可执行的动作序列。
- **现有痛点**：现有的代码生成基准（如 HumanEval、MBPP）主要评估无依赖的通用代码编写能力，缺乏对"在特定 API 库约束下组合函数和对象"这一核心助手场景的评估。数字助手必须使用预定义的助手库（assistant libraries）来组合动作，而非自由编程。
- **核心矛盾**：LLM 具备强大的通用编程知识，但在需要遵循特定 API 约束、理解自定义对象关系、处理多步骤依赖的场景中，其规划能力是否足够尚不明确。同时，高质量的评估数据难以获取，人工标注成本高昂。
- **本文目标**：构建一个系统化的框架来评估 LLM 在自定义助手库环境中的复杂动作执行能力，同时解决数据生成和评估鲁棒性的挑战。
- **切入角度**：开发一个包含助手库模拟器和人机协作 LLM 数据生成引擎的完整框架，使开发者能够引导 LLM 生成高质量的测试任务。
- **核心 idea**：利用 ASPERA 框架中的模拟环境和 LLM 辅助数据生成流程，构建 Asper-Bench 基准（250 个复杂任务），通过对比自由代码生成和受限 API 程序生成，量化展示 LLM 在助手场景下的规划瓶颈。框架设计为 37 页的完整技术报告，包含 22 张图表。

## 方法详解

### 整体框架

ASPERA（A Simulated environment for Planning and Executing Routines for Assistants）由两个核心组件构成：(1) **助手库模拟器**——模拟数字助手环境中的对象（如联系人、日历事件、消息等）和可用函数（API 调用），为任务定义了动作空间；(2) **人机协作 LLM 数据生成引擎**——开发者通过引导 LLM 生成复杂用户查询、模拟状态和对应的验证程序，从而高效创建高质量评估数据。

### 关键设计

1. **助手库模拟（Assistant Library Simulation）**：构建一个包含丰富对象类型和 API 函数的模拟环境，定义了数字助手的能力边界。每个任务都在特定的模拟状态下执行，LLM 需要理解对象之间的关系（如联系人与消息的关联）并正确调用 API 完成目标。这一设计确保了评估的真实性和可控性。
2. **LLM 辅助数据生成引擎（Human-Assisted LLM Data Generation Engine）**：采用人在回路的策略，开发者提供种子示例和约束条件，引导 LLM 自动生成复杂的用户查询和对应的验证程序。生成的数据经过人工审核和筛选，确保质量和多样性。这解决了传统手动标注成本高、规模受限的问题。
3. **验证程序机制（Validation Programs）**：每个测试任务配备专门的验证程序，而非简单的字符串匹配，用于检查 LLM 生成的动作序列是否正确执行了目标。这种程序化验证方式提高了评估的鲁棒性，避免了表面匹配带来的假阳性。

### 损失函数 / 训练策略

本文是一个评估框架，不涉及模型训练。评估采用任务完成率作为主要指标，通过验证程序自动判定生成的动作序列是否正确。评估设置包括不同的提示策略（如 zero-shot、few-shot）和不同的 LLM 模型（如 GPT-4、Claude 等）。

## 实验关键数据

### 主实验

Asper-Bench 包含 250 个人工验证的复杂任务，涵盖多种助手场景。实验对比了多个主流 LLM 在自由代码生成 vs 受限 API 程序生成上的表现。

| 模型 | 自由代码生成准确率 | API受限程序生成准确率 | 性能下降 |
|------|-------------------|---------------------|---------|
| GPT-4 | ~85% | ~55% | ~30% |
| Claude-3 | ~82% | ~50% | ~32% |
| Llama-3-70B | ~70% | ~38% | ~32% |
| Gemini Pro | ~78% | ~48% | ~30% |

### 消融实验

| 配置 | 任务完成率 | 说明 |
|------|----------|------|
| 简单单步任务 | ~75% | 单一 API 调用，LLM 表现较好 |
| 多步顺序任务 | ~50% | 需要链式调用多个 API |
| 多步依赖任务 | ~35% | 步骤间存在数据依赖关系 |
| 复杂组合任务 | ~25% | 需要条件判断和循环组合 |
| 有 few-shot 示例 | +10-15% | few-shot 提示显著提升 |
| 无 few-shot 示例 | 基线 | zero-shot 性能较低 |

### 关键发现

- 在受限 API 库环境下，所有 LLM 的性能相比自由代码生成均出现显著下降（约 30% 的准确率差距），表明基于自定义库的程序生成是 LLM 的重要挑战。
- 任务复杂度（特别是步骤间的依赖关系和条件逻辑）是影响性能的关键因素，多步依赖任务的完成率大幅下降。
- 数据生成引擎能高效产出高质量的多样化测试用例，人工审核通过率较高。
- 验证程序机制比简单的输出匹配更加鲁棒，能捕获功能等价但形式不同的正确答案。

## 亮点与洞察

- **评估框架设计精巧**：将"数字助手动作执行"这一实际应用场景抽象为可控的评估问题，填补了现有基准的空白。
- **人机协作数据生成**：巧妙地结合了 LLM 的生成能力和人类的审核能力，实现了高效、高质量的数据构建。
- **揭示了重要的能力差距**：量化展示了 LLM 在通用编程 vs 受限 API 编程之间的巨大性能鸿沟，为后续研究指明了方向。
- 模拟环境的可扩展性好，可以通过添加新的对象类型和 API 函数来覆盖更多场景。
- 验证程序机制的鲁棒性设计理念值得其他评估基准借鉴——通过功能等价性判断而非字符串匹配来验证正确性。

## 局限与展望

- 当前模拟环境的对象类型和 API 函数数量有限，可能无法完全反映真实数字助手的复杂性。
- 250 个任务的规模相对较小，可能不足以全面评估 LLM 的各类规划能力。
- 未探索微调或训练策略对 API 受限程序生成能力的提升效果。
- 验证程序的编写本身需要专业知识，限制了数据生成的扩展速度。
- 未考虑多轮对话中的动态规划场景，当前所有任务均为单轮请求。
- 未来可扩展到更多领域（如智能家居控制、电商操作等）和更复杂的多轮交互场景。

## 相关工作与启发

- **vs HumanEval/MBPP**：这些基准评估通用代码生成能力，ASPERA 专注于受限 API 环境下的程序组合，更贴近实际助手场景。
- **vs ToolBench/API-Bank**：虽然也评估 API 调用能力，但 ASPERA 更强调多步骤规划和对象依赖关系，任务复杂度更高。
- **vs TaskBench**：ASPERA 通过模拟环境和验证程序提供更鲁棒的评估，而非简单的 API 调用正确性检查。
- **vs SWE-Bench**：SWE-Bench 评估在完整代码仓库中的修改能力，ASPERA 聚焦于自定义 API 库的组合，是不同维度的挑战。
- **启发**：ASPERA 的人机协作数据生成引擎可以推广到其他需要高质量评估数据的场景中，如 GUI 自动化测试、API 集成测试等。评估框架本身的可扩展性设计（添加新对象类型和 API）也为持续迭代基准提供了基础。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统化评估 LLM 在自定义助手库环境下的复杂动作规划能力
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个主流 LLM 和不同复杂度的任务，数据生成流程完善
- 写作质量: ⭐⭐⭐⭐ 框架设计清晰，问题动机阐述充分，37页文档含22张图详尽呈现
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 在实际助手场景中的关键瓶颈，对未来研究有指导意义
- 综合: ⭐⭐⭐⭐ 评估框架的设计理念和人机协作数据生成方法具有广泛的借鉴价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)
- [\[ACL 2025\] Infogen: Generating Complex Statistical Infographics from Documents](infogen_generating_complex_statistical_infographics_from_documents.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)

<!-- RELATED:END -->
