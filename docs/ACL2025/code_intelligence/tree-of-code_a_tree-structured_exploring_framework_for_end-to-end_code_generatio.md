---
title: >-
  [论文解读] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation
description: >-
  [ACL 2025][代码生成] 提出 Tree-of-Code（ToC）框架，通过树结构组织端到端的完整代码程序（CodeProgram）节点，结合基于执行结果的反思机制和提示/模型随机探索策略，在无需标注数据的零样本设置下，以不到 1/4 的交互轮次实现了比 CodeAct 高近 20% 的复杂任务准确率。
tags:
  - ACL 2025
  - 代码生成
  - 树搜索
  - LLM Agent
  - 工具调用
  - 端到端推理
  - 自监督
---

# Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation

**会议**: ACL 2025  
**arXiv**: [2412.15305](https://arxiv.org/abs/2412.15305)  
**代码**: 未公开  
**领域**: nlp_generation  
**关键词**: 代码生成, 树搜索, LLM Agent, 工具调用, 端到端推理, 自监督

## 一句话总结

提出 Tree-of-Code（ToC）框架，通过树结构组织端到端的完整代码程序（CodeProgram）节点，结合基于执行结果的反思机制和提示/模型随机探索策略，在无需标注数据的零样本设置下，以不到 1/4 的交互轮次实现了比 CodeAct 高近 20% 的复杂任务准确率。

## 研究背景与动机

代码作为 LLM Agent 的行为空间已展现出处理复杂任务的巨大潜力，其中 CodeAct 是代表性工作，将代码块作为 agent 的动作单元。然而 CodeAct 存在四个关键局限：

**依赖 Ground Truth 终止**：CodeAct 假设 GT 已知并用于匹配终止条件，在真实在线场景中不可行

**碎片化思考低效**：逐步生成代码块导致逻辑链断裂，随着轮次增加，重复整合历史上下文引发上下文过载和幻觉累积

**缺乏多路径探索**：仅沿单一推理路径交互，无法探索多种解题策略

**轨迹数据难以复用**：多轮交互产生的轨迹无法直接组合为完整程序用于 SFT，且因缺乏过程监督而不适合强化学习

核心动机：既然中间状态的监督信号难以获取，能否将每次端到端的完整执行视为一个原子状态？通过并行探索多个可行解并投票选优，避免对中间过程标注的依赖。

## 方法详解

### 1. CodeProgram：端到端代码生成范式

ToC 的核心创新在于将树的每个节点定义为一个 **CodeProgram**——一次性生成解决整个任务的完整代码程序，而非像 CodeAct 那样逐步生成单个动作。

具体流程：
- LLM 首先进行全局推理（Global Reasoning），分析和分解问题
- 然后生成包含完整推理逻辑的端到端代码
- 思维和代码分别用 `<thought>` 和 `<execute>` 标签封装
- 代码的执行成功/失败作为唯一的过程监督信号

CodeProgram 的一个轮次等价于 CodeAct 的多个轮次，显著提高了效率。代码本身既是推理载体（code-as-reasoning），又是可执行的验证器。

### 2. 两个辅助工具

为保持端到端流程的连贯性，引入两个辅助函数：
- **res_handler**：通用结果处理器，在所有工具调用完成后生成符合要求的最终摘要
- **next_action**：专用于网页浏览任务，根据页面内容、已访问 URL 和任务查询决定下一步浏览动作

### 3. 树结构与自生长机制

ToC 表示为 $\mathbf{T} = (\mathbf{N}, \mathbf{S})$，其中 $\mathbf{N}$ 为节点集合，$\mathbf{S}$ 为连接节点的茎（反思推理过程）。

**树扩展规则**：
- 采用广度优先搜索（BFS），每个父节点分支为 $M=3$ 个子节点
- 节点的生长取决于自身执行状态：执行成功则停止并收集，执行失败则继续生长
- 最大深度 $L=3$

**基于执行的反思**：当执行失败时，将错误信息反馈给 LLM 进行自我反思和代码修正，生成下一层新节点。反思过程利用祖先节点的完整历史（思维、代码、执行结果）。

### 4. 探索策略：提示池 + 模型池

为增强解的多样性，借鉴随机森林思想：
- **模型探索**：从10个LLM中随机采样，温度统一设为0.1
- **提示探索**：从精心设计的提示池中随机选择。提示池构建过程包括：用10个LLM从根提示进化出10个多样化提示，人工筛选6个，再进行格式调整、示例添加、组件重排等修改

### 5. 最终结果生成

收集所有执行成功节点的输出，通过多数投票（majority vote）和摘要确定最终答案。

## 实验关键数据

### 表1：ToC vs CodeAct vs ReAct 整体对比

| 方法 | M3 平均轮次 | M3 准确率 | M3 输出词数 | API-Bank L3 平均轮次 | API-Bank L3 准确率 | API-Bank L3 输出词数 |
|------|-----------|----------|------------|-------------------|-------------------|---------------------|
| ReAct | 8.2 | 38.1% | 1.86k | 9.5 | 8.2% | 1.66k |
| CodeAct | 7.0 | 49.4% | 1.91k | 8.9 | 19.2% | 1.82k |
| **ToC (3-3)** | **1.7** | **67.1%** | **0.44k** | **2.1** | **38.0%** | **0.39k** |

ToC 在两个数据集上均以不到 1/4 的交互轮次和输出词数，实现了约 20% 的准确率提升。

### 表2：不同模型上 CodeProgram 与多轮方法的对比（部分模型）

| 模型 | M3 ReAct | M3 CodeAct | M3 ToC(1-1) | M3 ToC(1-3) | API-Bank CodeAct | API-Bank ToC(1-3) |
|------|----------|-----------|-------------|-------------|-----------------|-------------------|
| claude-3-5-sonnet | 48.8% (7.7轮) | 73.2% (5.7轮) | **73.2%** (1轮) | **82.9%** (1轮) | 32.0% (7.8轮) | **52.0%** (1轮) |
| gpt-4-1106-preview | 54.9% (7.5轮) | 75.6% (5.4轮) | 72.0% (1轮) | 73.2% (1轮) | 30.0% (8.2轮) | 38.0% (1轮) |
| qwen2.5-72b | 50.0% (7.9轮) | 70.7% (5.6轮) | 51.2% (1轮) | 59.8% (1轮) | 30.0% (8.2轮) | 32.0% (1轮) |
| 10模型平均 | 38.05% (8.24轮) | 49.37% (7.04轮) | 43.53% (1轮) | 50.98% (1轮) | 19.2% (8.89轮) | 24.4% (1轮) |

关键发现：单轮 CodeProgram（ToC 1-1）在部分模型上已超越多轮 CodeAct，加上提示探索（ToC 1-3）后平均准确率进一步超过 CodeAct。claude-3-5-sonnet 在 ToC 中表现最佳，因其强大的指令遵循能力。

### 表3：不同树规模的性能（claude-3-5-sonnet，M3）

| 层数 \ 每层节点数 | 1 | 2 | 3 |
|----------------|------|------|------|
| 1 | 73.2% (1轮) | 75.6% (1轮) | 82.9% (1轮) |
| 2 | 73.2% (1.4轮) | 76.8% (1.4轮) | 84.1% (1.5轮) |
| 3 | 74.4% (1.8轮) | 79.3% (1.7轮) | 84.1% (1.6轮) |

节点宽度（breadth）比层深度（depth）对准确率的提升更大——更多节点引入了更多样化的提示和模型变体。

### 表4：提示探索的消融实验（M3）

| 设置 | 平均轮次 | 准确率 | 下降 |
|------|---------|--------|------|
| ToC（随机模型） | 1.7 | 67.1% | — |
| ToC 去掉提示探索 | 1.9 | 63.4% | -3.7% |
| ToC（固定最佳模型） | 1.6 | 84.1% | — |
| ToC 去掉模型+提示探索 | 1.8 | 75.6% | -8.5% |

提示探索在多样性较低的场景（固定模型）中贡献更为显著（-8.5% vs -3.7%）。

## 核心发现

- CodeProgram 的端到端范式将任务粒度从"动作级"提升到"程序级"，单轮即可匹配甚至超越多轮交互方法
- 提示多样性对性能的贡献大于模型多样性，且在固定模型设置下效果更为突出
- 树的宽度（探索更多独立解）比深度（迭代反思修正）更重要
- 在 M3 的五类任务中，ToC 在除网页浏览外的所有任务上接近完美准确率
- 不同基线方法的最佳模型因数据集而异——CodeAct 中 GPT-4 在 M3 上最佳，GPT-4o 在 API-Bank 上最佳；ToC 中 claude-3-5-sonnet 一致最优

## 亮点

- **端到端自监督**：仅依赖代码执行成功/失败作为监督信号，完全避免了对 GT 标注的依赖，适合真实在线场景
- **效率与效果双赢**：以不到 1/4 的轮次和词数实现近 20% 的准确率提升，大幅降低 API 调用成本
- **随机森林式探索**：提示池 + 模型池的随机组合策略简单而有效，为 agent 系统引入了结构化多样性
- **通用框架设计**：树结构可灵活集成任意反思方法，CodeProgram 生成的数据可直接用于后续 SFT/ReFT 训练
- **广泛验证**：在两个数据集（M3ToolEval、API-Bank level-3）上使用10个主流零样本 LLM 进行了全面评估

## 局限与展望

1. **不适合完全开放探索场景**：如机器人在未知环境中导航等需要逐步感知的任务，无法一次性生成完整解决方案
2. **反思机制较为基础**：仅依赖执行反馈，未探索更精细的搜索策略（如自适应剪枝、MCTS）
3. **提示池设计初步**：主要依赖简单的提示进化和人工调整，缺乏系统化的提示池构建方法论
4. **树规模受限**：默认 3×3 树，更大规模树的效果-效率权衡有待深入研究
5. **数据集规模有限**：M3 仅 82 个任务、API-Bank level-3 仅 50 个任务，验证规模偏小
6. **未探索与微调的结合**：虽然讨论了 CodeProgram 数据可用于 SFT/ReFT，但未进行实际训练实验

## 与相关工作的对比

| 方法 | 类型 | 粒度 | 需要GT | 多路径探索 | 端到端 |
|------|------|------|--------|-----------|--------|
| ReAct | 推理+动作 | 动作级 | 是 | ✗ | ✗ |
| CodeAct | 代码即动作 | 代码块级 | 是 | ✗ | ✗ |
| Tree-of-Thoughts | 思维树 | 思维级 | 否 | ✓（同方案内） | ✗ |
| CodeTree | 多智能体树搜索 | 代码级 | 否 | ✓ | ✗ |
| Self-Repair Trees | 修复树 | 程序级 | 是 | ✓ | ✗ |
| MCTS 方法 | 蒙特卡罗树搜索 | 步骤级 | 是 | ✓ | ✗ |
| **ToC (本文)** | **自生长树** | **程序级** | **否** | **✓（跨方案）** | **✓** |

ToC 与 ToT 名字相似但含义不同——ToT 探索同一解题路径内的不同思维步骤，ToC 探索多个完全独立的完整程序解。ToC 更接近"代码随机森林"的概念。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 端到端 CodeProgram + 执行驱动自生长树的组合有良好创新性
- 实验充分度: ⭐⭐⭐⭐ — 10个模型、两个数据集、多维消融实验，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机分析透彻，图示丰富
- 实用价值: ⭐⭐⭐⭐ — 零样本无GT设置贴近真实应用，效率提升显著

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](../../AAAI2026/code_intelligence/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)
- [\[ACL 2025\] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi_agent_tree_search_via_reinforcement_learning_for_code_genera.md)

<!-- RELATED:END -->
