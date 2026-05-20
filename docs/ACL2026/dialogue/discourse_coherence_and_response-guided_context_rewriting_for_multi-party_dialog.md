---
title: >-
  [论文解读] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation
description: >-
  [ACL 2026][对话系统][多方对话] 本文提出 DRCR，首个将上下文改写引入多方对话生成的框架，使用话语连贯性和回复质量双反馈信号构建偏好数据，通过动态自演化学习让改写器和回复器在迭代训练中相互增强。
tags:
  - "ACL 2026"
  - "对话系统"
  - "多方对话"
  - "上下文改写"
  - "话语连贯性"
  - "偏好学习"
  - "动态自演化"
---

# Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation

**会议**: ACL 2026  
**arXiv**: [2604.06784](https://arxiv.org/abs/2604.06784)  
**代码**: 无  
**领域**: 对话系统 / 多方对话  
**关键词**: 多方对话, 上下文改写, 话语连贯性, 偏好学习, 动态自演化

## 一句话总结

本文提出 DRCR，首个将上下文改写引入多方对话生成的框架，使用话语连贯性和回复质量双反馈信号构建偏好数据，通过动态自演化学习让改写器和回复器在迭代训练中相互增强。

## 研究背景与动机

**领域现状**：多方对话生成（MDG）涉及多个角色和复杂的话语结构（跨越多个话语的发言关系），比双方对话困难得多。已有方法通过编码对话结构信息来辅助生成。

**现有痛点**：(1) 对话中的口语化表达和不完整话语（如指代、省略）损害了话语连贯性，进而影响对话结构的表示质量；(2) 先前方法直接用有缺陷的对话上下文编码结构，未尝试先改善上下文质量；(3) 在多方对话中这些问题更加突出——多个说话者增加了指代和省略的复杂度。

**核心矛盾**：对话结构编码的质量取决于上下文的连贯性，但原始上下文中的口语表达和省略破坏了连贯性。简单改写可能无法兼顾话语连贯性和下游回复生成的质量。

**本文目标**：通过对话上下文改写提升多方对话生成质量，同时保证改写既提高话语连贯性又有利于生成高质量回复。

**切入角度**：用话语连贯性质量和回复生成质量作为双反馈信号构建偏好数据，训练改写器生成既连贯又有利于回复的上下文。

**核心 idea**：改写器和回复器通过迭代训练相互增强——更好的改写产生更好的回复，更好的回复反馈引导更好的改写。

## 方法详解

### 整体框架

DRCR 包含两个模块：改写器（Rewriter）和回复器（Responder），通过三个阶段训练：(1) 监督微调——分别训练改写器和回复器的基础能力；(2) 偏好数据构建——用话语连贯性和回复质量双信号对改写结果排序；(3) 动态自演化——改写器和回复器在迭代训练中通过相互反馈持续提升。

### 关键设计

1. **话语连贯性反馈**:

    - 功能：评估改写后上下文的话语结构质量
    - 核心思路：使用话语连贯性评估模型为不同改写结果打分，连贯性更高的改写作为偏好数据中的"优选"样本。连贯性衡量改写是否消除了指代歧义、补全了省略、理顺了话语关系
    - 设计动机：对话上下文的连贯性直接影响话语结构编码的质量，进而影响回复生成

2. **回复质量反馈**:

    - 功能：确保改写有利于生成高质量回复
    - 核心思路：将不同改写的上下文输入回复器，比较生成回复的质量（相关性、信息量、连贯性）。产生更好回复的改写被标记为"优选"
    - 设计动机：改写的最终目标是提升回复质量，仅优化话语连贯性可能不足以保证下游生成效果

3. **动态自演化学习**:

    - 功能：让改写器和回复器在迭代中相互增强
    - 核心思路：在每次迭代中，改写器用当前回复器的反馈更新，更新后的改写器产生更好的上下文，回复器在更好的上下文上进一步提升。多轮迭代直到收敛
    - 设计动机：单次训练可能陷入次优——改写器不知道什么样的改写真正有利于当前回复器，动态交互允许两者协同优化

### 损失函数 / 训练策略

改写器和回复器均使用 DPO 风格的偏好学习。偏好数据由双反馈信号（话语连贯性 + 回复质量）构建。迭代训练直到改写和回复质量稳定。

## 实验关键数据

### 主实验

**四个多方对话数据集上的 BLEU/ROUGE 分数**

| 方法 | 数据集1 | 数据集2 | 数据集3 | 数据集4 |
|------|--------|--------|--------|--------|
| SS-MPC（先前SOTA） | 基线 | 基线 | 基线 | 基线 |
| LLM 直接生成 | 中等 | 中等 | 中等 | 中等 |
| **DRCR** | **超越** | **超越** | **超越** | **超越** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅话语连贯性反馈 | 提升但有限 | 缺少下游信号 |
| 仅回复质量反馈 | 提升 | 直接优化目标 |
| 双反馈 | 最优 | 两个信号互补 |
| 无自演化（单次训练） | 次优 | 缺少协同优化 |
| 有自演化 | 最优 | 迭代提升 |

### 关键发现

- DRCR 在全部四个多方对话数据集上超越先前 SOTA
- 双反馈信号优于单一信号——话语连贯性和回复质量提供互补视角
- 动态自演化的迭代训练显著优于单次训练——改写器和回复器的协同效应
- 上下文改写有效消除了指代和省略导致的理解障碍

## 亮点与洞察

- 首次将上下文改写引入多方对话生成——解决了被忽视的口语化问题
- 双反馈+自演化的设计形成了优雅的闭环优化
- 改写作为生成的前处理步骤，与现有生成方法正交，可叠加使用

## 局限与展望

- 改写增加了推理时的计算开销（额外的改写步骤）
- 自演化的迭代次数和收敛条件需要实验确定
- 仅在中文多方对话数据集上验证，跨语言效果待确认
- 改写可能引入信息偏差，尤其在涉及模糊意图的场景

## 相关工作与启发

- **vs SS-MPC**: SS-MPC 直接编码原始对话结构，DRCR 先改写再编码
- **vs 查询改写**: 搜索中的查询改写启发了对话上下文改写，但多方对话的结构更复杂

## 评分

- 新颖性: ⭐⭐⭐⭐ 上下文改写+双反馈自演化在多方对话中首次应用
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、详细消融
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，示例直观
- 价值: ⭐⭐⭐⭐ 为多方对话生成提供了新的预处理范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[ACL 2026\] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)

</div>

<!-- RELATED:END -->
