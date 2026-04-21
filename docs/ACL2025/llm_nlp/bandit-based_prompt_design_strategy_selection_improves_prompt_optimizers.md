---
title: >-
  [论文解读] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers
description: >-
  [ACL 2025][LLM/NLP][提示学习] 首次提出 prompt 设计策略的显式选择机制 OPTS，将 11 种策略（CoT、角色提示、情感提示等）建模为多臂老虎机的臂，用 Thompson 采样自动选择最合适的策略并集成到 EvoPrompt 优化器中，在 BIG-Bench Hard 的 23 个任务上用 GPT-4o mini 实现最高 50% 的性能提升。
tags:
  - ACL 2025
  - LLM/NLP
  - 提示学习
  - Thompson采样
  - 多臂老虎机
  - BIG-Bench Hard
---

# OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers

**会议**: ACL 2025  
**arXiv**: [2503.01163](https://arxiv.org/abs/2503.01163)  
**代码**: [GitHub](https://github.com/shiralab/OPTS)  
**领域**: Prompt 优化  
**关键词**: Prompt策略选择, Thompson采样, 多臂老虎机, EvoPrompt, BIG-Bench Hard

## 一句话总结

首次提出 prompt 设计策略的显式选择机制 OPTS，将 11 种策略（CoT、角色提示、情感提示等）建模为多臂老虎机的臂，用 Thompson 采样自动选择最合适的策略并集成到 EvoPrompt 优化器中，在 BIG-Bench Hard 的 23 个任务上用 GPT-4o mini 实现最高 50% 的性能提升。

## 研究背景与动机

**Prompt 优化能自动搜索有效 prompt，但结果常不如人类专家设计。** EvoPrompt 等方法通过 LLM 模拟进化算法搜索 prompt 空间，虽然发现了有效 prompt，但这些 prompt 往往缺乏人类专家常用的设计策略（如 CoT 推理链、角色设定、分步指令等），与精心设计的专家 prompt 存在质量差距。

**Prompt 设计策略并非总有益。** CoT 和角色提示在某些 LLM 和任务组合上反而降低性能。这意味着不能简单地给所有 prompt 都加上所有策略。APET 方法将全部策略描述一起喂给 LLM 做隐式选择，但 LLM 本身的优化能力有限，隐式选择可能次优。

**核心矛盾：策略有价值但什么时候用什么策略是未知的。** 这本质上是一个探索-利用(explore-exploit)问题——经典多臂老虎机正好适用。**OPTS 的核心 idea 是将每种 prompt 设计策略视为一个"臂"，用 Thompson 采样在优化过程中动态学习哪种策略对当前任务最有效**，实现了首次显式的策略选择机制。

## 方法详解

### 整体框架

OPTS 作为模块插入到现有 prompt 优化器（EvoPrompt）的突变/交叉步骤之后：EvoPrompt 生成候选 prompt → OPTS 选择一种策略 → LLM 把策略应用到候选 prompt → 评估性能并反馈给多臂老虎机更新。

### 关键设计

1. **多臂老虎机建模**:
    - 功能：将 K=11 种 prompt 设计策略 + 1 个"不使用策略"的 inaction arm 建模为 K+1=12 个臂
    - 核心思路：每种策略的价值不确定且可能随任务变化，用 bandit 框架在线学习最优策略。Inaction arm 确保不强制使用任何策略——可能所有策略都无益
    - 设计动机：显式选择比隐式选择可控且可优化

2. **Thompson 采样选择 (OPTS-TS)**:
    - 功能：用 Beta 分布先验和后验更新实现高效的探索-利用平衡
    - 核心思路：每个臂维护 $\text{Beta}(\alpha_k, \beta_k)$ 分布，每次从各臂分布中采样，选择采样值最大的臂。奖励定义为 $r = \mathbf{1}[s > \max \tilde{S}]$，即新 prompt 得分是否超过父 prompt 的最高分
    - 设计动机：Thompson 采样有理论保证的渐近最优性，在 bandit 文献中实践效果最优

3. **策略应用机制**:
    - 功能：将选中策略的文字描述传给 prompt 设计 LLM，让其修改候选 prompt
    - 核心思路：选中前 K 个臂之一时，将对应策略描述和待修改 prompt 一起输入 LLM；选中 inaction arm 时不做任何修改。使用与 APET 相同的 meta-prompt 格式
    - 设计动机：策略以自然语言描述传递，LLM 自然理解如何应用

### 11 种 Prompt 设计策略

包括 ExpertPrompting（专家角色）、CoT（推理链）、Tree-of-Thought（推理树）、Emotion Prompting（情感）、Re-Reading（重读）、Style Prompting（风格）、Rephrase and Respond（改述再答）、Avoiding Bias（避免偏见）、Making Prompt Specific（具体化）、Shortening（精简）、Adding Necessary Information（补充信息）。

### 与 EvoPrompt 的集成

OPTS 可以同时与 EvoPrompt(DE) 和 EvoPrompt(GA) 集成。在 DE 变体中，OPTS 插入在交叉+突变之后、选择之前：先用 DE 操作生成 $p'_i$，再用 OPTS 选择策略修改为 $p''_i$，最后与父代 $p_i$ 比较保留更优者。

## 实验关键数据

### 主实验（GPT-4o mini 做生成和解题）

| 方法 | BIG-Bench Hard 23任务平均准确率 | vs EvoPrompt(DE) |
|------|-------------------------------|-----------------|
| Manual Prompt | 56.95 | - |
| APET | 57.93 | - |
| EvoPrompt(DE) | 60.11 | baseline |
| +OPTS(APET) | 62.36 | +2.25 |
| +OPTS(US) | 63.04 | +2.93 |
| +OPTS(TS) | **64.15** | **+4.04** |

### Llama-3-8B-Instruct 做解题

| 方法 | 平均准确率 | vs EvoPrompt(DE) |
|------|-----------|-----------------|
| EvoPrompt(DE) | 46.52 | baseline |
| +OPTS(TS) | **49.83** | **+3.31** |

### 消融实验

| 配置 | 说明 | 结果 |
|------|------|------|
| OPTS(TS) vs OPTS(US) | TS vs 均匀采样 | TS 在大多数任务上更优 |
| OPTS(TS) vs OPTS(APET) | TS vs 隐式选择 | TS 一致更优 |
| EvoPrompt(GA)+OPTS(TS) | GA变体 | 同样有效，不依赖特定优化算法 |
| Inaction arm 去除 | 强制使用策略 | 性能下降——有些任务不需要策略 |

### 关键发现

- **Thompson 采样一致最优**：在 GPT-4o mini 和 Llama-3 两个模型上都超越其他选择机制
- **最高 50% 提升在单个任务上**：某些任务上 OPTS(TS) 相比 EvoPrompt 提升 50%
- **Inaction arm 重要**：不是所有任务都能从策略中获益，保留"不使用"选项至关重要
- **不同任务偏好不同策略**：Thompson 采样能自动学习每个任务的最优策略分布

## 亮点与洞察

- **首次将多臂老虎机引入 prompt 策略选择**——概念简单但将经典 RL 工具用到了正确的问题上
- **"策略可能有害"的认识重要**——对所有 prompt 工程实践者都有意义，不应盲目叠加策略
- **模块化设计**：OPTS 是独立模块，可即插即用到任何 prompt 优化器中
- **显式 > 隐式**：OPTS(TS) 的显式选择一致优于 APET 的隐式选择，说明让 LLM 做优化不如用专门的优化算法

## 局限与展望

- **仅在 BIG-Bench Hard 验证**：其他类型任务（生成、对话、代码等）的效果未知
- **策略集合固定**：11 种预定义策略无法覆盖所有可能的有效策略，未探索自动发现新策略
- **单策略选择**：每次只应用一种策略，未探索策略组合的效果
- **上下文窗口限制**：策略描述占用 prompt 空间，当策略描述过长时可能压缩有效信息

## 相关工作与启发

- **vs APET (隐式选择)**：将所有策略描述喂给 LLM 让其隐式选择；OPTS 显式选择更可控，性能更优
- **vs PromptWizard (固定策略)**：总是应用 CoT 和角色提示；OPTS 按需选择避免有害策略
- **vs EvoPrompt (无策略)**：缺乏人类设计知识；OPTS 将最佳实践注入进化优化过程

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次显式策略选择机制，Thompson 采样在 prompt 优化中的新应用
- 实验充分度: ⭐⭐⭐⭐ BIG-Bench Hard 23 任务 × 2 模型 × 3 选择机制 × 2 优化算法变体
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，算法描述完整，图示直观
- 价值: ⭐⭐⭐⭐ 对 prompt 优化从业者有直接实用价值，模块化设计便于采用

<!-- RELATED:START -->

## 相关论文

- [What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)
- [JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)
- [Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](goal_hijacking_attack.md)

<!-- RELATED:END -->
