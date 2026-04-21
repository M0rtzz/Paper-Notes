---
title: >-
  [论文解读] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning
description: >-
  [NeurIPS 2025][机器人][神经符号推理] 提出 NeSyPr（Neurosymbolic Proceduralization），一种将符号规划器的声明性知识编译为可组合过程化表示的框架，使紧凑语言模型在无需在线符号引导的情况下完成高效具身推理，在 PDDLGym、VirtualHome 和 ALFWorld 三个基准上超越大规模推理模型和符号规划器。
tags:
  - NeurIPS 2025
  - 机器人
  - 神经符号推理
  - 过程化知识编译
  - 具身推理
  - 语言模型规划
  - 知识蒸馏
---

# NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.19429](https://arxiv.org/abs/2510.19429)  
**代码**: 无  
**领域**: 具身智能 / 机器人推理  
**关键词**: 神经符号推理, 过程化知识编译, 具身推理, 语言模型规划, 知识蒸馏

## 一句话总结

提出 NeSyPr（Neurosymbolic Proceduralization），一种将符号规划器的声明性知识编译为可组合过程化表示的框架，使紧凑语言模型在无需在线符号引导的情况下完成高效具身推理，在 PDDLGym、VirtualHome 和 ALFWorld 三个基准上超越大规模推理模型和符号规划器。

## 研究背景与动机

将语言模型（LM）应用于动态环境中的具身任务面临关键挑战：在线访问大规模推理引擎或符号规划器受限于延迟、连接性和计算资源。在物理系统部署中，这些约束尤为严峻——机器人不可能每一步都调用云端大模型或运行完整的 PDDL 求解器。

现有方法的困境：(1) 大型推理模型（如 GPT-4 级别）虽能力强但延迟高、资源需求大；(2) 符号规划器（如 PDDL solver）能保证最优性但缺乏灵活性，且计算复杂度随状态空间指数增长；(3) 小型 LM 推理能力不足，无法可靠完成多步结构化推理。

NeSyPr 的灵感来自认知科学中的"知识编译"理论：人类通过反复练习将"陈述性知识"（如交通规则）编译为"过程性知识"（如驾驶技能），后者执行速度更快、认知负荷更低。NeSyPr 将这一认知过程人工实现——将符号规划器的多步推理"编译"为 LM 的单步推理。

## 方法详解

### 整体框架

NeSyPr 分为两个阶段：

**编译阶段**（离线）：对给定任务域，使用符号工具（PDDL 规划器）基于声明性知识生成多种任务变体的最优计划。将这些计划转换为可组合的过程化表示——编码了计划中的隐式产生规则（production rules），使其可无缝集成到 LM 的推理过程中。

**推理阶段**（在线）：编译后的过程化知识被整合到紧凑 LM 中，将原本需要多步符号搜索的推理压缩为单步 LM 推理。这类似于人类的程序性记忆——无需每次回忆规则，而是直接执行编译后的程序。

### 关键设计

1. **声明性到过程化的知识转换**: 符号规划器利用领域描述（PDDL 格式的前置条件、效果、动作定义）和搜索算法生成任务解。NeSyPr 不直接使用这些行动序列，而是将其转化为"过程化表示"——编码了从状态到行动的隐式映射规则。这些表示是可组合的，意味着在新的状态配置下，LM 可以通过组合已知的过程片段来处理未见任务。设计动机是绕过符号系统的刚性：声明性知识需要搜索，过程性知识可以直接执行。

2. **过程化表示的 LM 集成**: 将编译好的过程化表示融入 LM 的推理过程，使 LM 不仅依赖其预训练知识，还能利用结构化的符号知识。这种集成方式使得推理在测试时不依赖外部符号引导，所有必要的推理能力都内化在 LM 中。关键优势是推理延迟极低——一次前向传播替代了多步符号搜索。

3. **跨任务泛化的可组合性**: 过程化表示被设计为可组合的单元，而非针对特定任务实例的固定序列。当面对新任务时，LM 可以组合已有的程序片段来构建新的解决方案。这类似于程序语言中的函数复用——每个过程化单元封装了特定的状态-动作转换逻辑，可被不同任务组合调用。

### 损失函数 / 训练策略

论文摘要层面描述了该方法的设计理念但未公开详细训练细节。核心训练过程应包括：
- 使用符号规划器生成大量（状态, 计划）对作为监督信号
- 通过微调或知识蒸馏将这些过程化知识注入紧凑 LM
- 可能采用课程学习策略，先学习简单任务的过程化表示再组合处理复杂任务

## 实验关键数据

### 主实验

三个具身推理基准上的评估：

**PDDLGym**: 基于 PDDL 定义的经典规划问题集。NeSyPr 使用紧凑 LM 即可达到接近符号规划器的最优解成功率，推理延迟远低于大规模 LM。

**VirtualHome**: 家庭活动模拟环境，要求执行多步家务任务（如"做早餐"需要打开冰箱→取食材→放到台面→开火→烹饪等）。NeSyPr 在任务完成率上超越大规模推理模型。

**ALFWorld**: TextWorld 风格的文本环境中执行家务指令。NeSyPr 展示了高效推理能力，以更小的模型规模达到或超越大型模型基线。

核心量化结果：NeSyPr 使用更紧凑的 LM，在推理效率上显著优于大规模推理模型和在线符号规划器，同时在任务成功率等指标上保持竞争力或更优。

### 消融实验

从摘要和论文描述推断，关键消融应包括：
- 声明性 vs 过程化表示的影响
- 不同规模 LM 作为载体的效果
- 可组合性对新任务泛化的贡献
- 符号训练数据量对性能的影响

### 关键发现

- 过程化知识编译成功将多步符号推理压缩为紧凑 LM 的单步推理，验证了认知科学"知识编译"理论在 AI 系统中的可行性
- 紧凑 LM 经过 NeSyPr 编译后超越了远大于自身的推理模型——规模不是高效推理的唯一路径
- 该框架适合延迟敏感、资源受限的物理部署场景，是"边缘 AI 推理"的有力方案

## 亮点与洞察

- **认知科学启发**: 将"声明性→过程性"的人类知识编译过程显式建模为 AI 系统的设计原则，这种跨学科借鉴令人启发
- **小模型大能力**: 证明了在正确的知识注入方式下，紧凑 LM 可以在结构化推理任务上超越大模型——不是所有任务都需要 GPT-4
- **实用性强**: 离线编译+在线高效推理的范式非常适合实际机器人系统的部署需求
- 神经符号方法的新范式——不是让神经网络和符号系统运行时协作，而是将符号知识在编译期"烧入"神经网络

## 局限与展望

- 编译阶段依赖完整的符号领域描述（PDDL 定义），限制了对新领域的快速适应
- 过程化表示的可组合性边界不明确——何时组合失效需要更多分析
- 仅在文本/仿真环境验证，未扩展到真实机器人的感知-动作闭环
- 环境动态变化（非稳态转移函数）时编译好的过程可能失效
- 论文信息量有限（仅有摘要级细节），具体的编译算法、LM 微调策略、规模效应等有待论文正文补充

## 相关工作与启发

- **SayCan / Inner Monologue**: 用 LLM 做高层规划+低层技能执行，但依赖在线 LLM 调用；NeSyPr 将规划能力编译到本地 LM
- **LLM + PDDL 规划**: 如 LLM+P 将自然语言转为 PDDL 再调用求解器；NeSyPr 反其道而行——将求解器知识编译到 LM 中
- **知识蒸馏**: NeSyPr 可视为从符号系统到神经网络的跨范式蒸馏，比传统神经→神经蒸馏更具结构性
- 核心启发：符号 AI 和连接主义 AI 不必是运行时竞争关系，可以是编译期→运行期的互补关系

## 评分

⭐⭐⭐⭐ — 思路精巧（过程化知识编译），实用价值高（低延迟、小模型），三个基准一致验证了有效性。从认知科学借鉴设计原则是一大亮点。受限于当前信息量，详细机制的评估有待论文正文确认。

<!-- RELATED:START -->

## 相关论文

- [Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)

<!-- RELATED:END -->
