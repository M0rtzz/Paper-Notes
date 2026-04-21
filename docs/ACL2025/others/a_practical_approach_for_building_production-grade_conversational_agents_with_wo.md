---
title: >-
  [论文解读] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs
description: >-
  [ACL 2025][对话代理] 提出基于有向无环图(DAG)的工作流框架，通过将LLM agent的复杂业务约束分解到图中不同状态节点，并结合响应掩码微调策略，构建满足生产级要求的电商对话代理，在任务准确率和格式遵循方面均大幅超越GPT-4o基线。
tags:
  - ACL 2025
  - 对话代理
  - 工作流图
  - DAG框架
  - 电商agent
  - 微调策略
---

# A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs

**会议**: ACL 2025  
**arXiv**: [2505.23006](https://arxiv.org/abs/2505.23006)  
**代码**: 无  
**领域**: LLM Agent / 对话系统  
**关键词**: 对话代理, 工作流图, DAG框架, 电商agent, 微调策略

## 一句话总结

提出基于有向无环图(DAG)的工作流框架，通过将LLM agent的复杂业务约束分解到图中不同状态节点，并结合响应掩码微调策略，构建满足生产级要求的电商对话代理，在任务准确率和格式遵循方面均大幅超越GPT-4o基线。

## 研究背景与动机

**领域现状**: LLM在搜索、推荐、聊天等服务场景展现出强大能力，基于tool calling的对话代理日益普及。

**现有痛点**: 将LLM部署到真实工业场景面临三大挑战：(1) LLM的概率生成特性导致无法严格遵守业务规则(如禁止向未成年人推荐烟酒)；(2) 移动端场景要求特定输出格式(长度限制、emoji等)，LLM难以稳定遵循；(3) 将所有要求堆砌到system prompt会导致提示词过长，反而降低延迟和准确率。

**核心矛盾**: LLM需要同时保持灵活对话能力和严格遵守服务特定约束，但这两者因LLM的概率本质而本质冲突。

**本文目标**: 如何在真实电商场景中构建可控、可靠、可扩展的对话代理，同时满足灵活对话和严格合规的双重需求。

**切入角度**: 采用混合架构——用DAG工作流图将agent行为结构化，每个节点拥有独立的system prompt、工具和约束规则，避免单一长提示词的问题；并设计针对图结构的数据收集和微调方法。

**核心 idea**: 用DAG工作流图将复杂业务约束分散到各节点，配合响应掩码微调，让小模型在结构化agent架构中超越GPT-4o。

## 方法详解

### 整体框架

系统将电商对话agent建模为一个有向无环图(DAG)，包含LLM调用节点(绿色)和工具调用节点(粉色)。从初始聊天节点出发，LLM根据用户意图路由到对应的任务节点(如商品推荐、购买消息等)，每个节点独立管理约束和格式要求。整个流程包括：(1) 设计工作流图 → (2) 用GPT-4o构建原型agent收集数据 → (3) 微调自有模型。

### 关键设计

1. **多状态DAG框架**
    - 功能：将业务约束分散到图的各个状态节点
    - 核心思路：每个LLM调用节点拥有独立的system prompt(只包含该状态的约束)和自定义的历史对话操作函数(modify_history)，工具节点有输入/输出schema
    - 设计动机：避免将所有约束堆砌到单一system prompt中导致注意力分散和遵循率下降

2. **对话历史操纵(modify_history)**
    - 功能：根据节点需求裁剪输入的对话历史
    - 核心思路：如购买消息节点(purchase_message)会移除所有历史只保留最后一轮的购买信息
    - 设计动机：通过限制对无关先验信息的访问来减少幻觉

3. **原型agent数据收集流程**
    - 功能：三步法高效收集高质量标注数据
    - 核心思路：先用GPT-4o搭建原型agent → 标注员与原型agent交互并记录全部图遍历过程 → 标注员审核修正错误响应，辅以自动检查器(如JSON静态类型检查)
    - 设计动机：标注员难以独立生成涉及多步推理和工具调用的复杂响应，需要原型agent辅助

4. **响应掩码微调(Response Masking)**
    - 功能：解决图结构下多轮训练的prompt冲突问题
    - 核心思路：在训练时对来自其他节点的响应施加loss masking，只对当前节点生成的响应计算loss
    - 设计动机：同一对话历史中的不同响应可能来自不同节点(不同system prompt)，标准多轮训练会引入冲突监督信号

### 损失函数 / 训练策略

采用chatbot风格的序列格式化，每个节点独立构建训练样本 $(s_v, x_1, o_1, ..., x_n, o_n)$，其中 $s_v$ 为该节点的system prompt。关键是在loss计算中掩码掉非当前节点生成的响应，防止不同节点指令间的梯度干扰。工具调用输出使用constrained decoding确保格式正确。

## 实验关键数据

### 主实验

| 模型 | 架构 | Accuracy | Format Adherence | Response Validity |
|------|------|----------|-----------------|-------------------|
| Qwen 2.5 (32B) | Basic | 0.578 | 0.734 | 2.816 |
| Qwen 2.5 (32B) | WG | 0.616 | 0.813 | 2.831 |
| Qwen 2.5 (32B) | WG-FT | 0.884 | **0.969** | 2.880 |
| Gemma 3 (27B) | WG-FT | 0.887 | 0.966 | **2.911** |
| Internal (27-32B) | WG-FT | **0.890** | 0.987 | 2.953 |
| GPT-4o | WG | 0.888 | 0.964 | 2.882 |

### 消融实验

| 评估维度 | Internal ≥ GPT-4o 比例 |
|---------|----------------------|
| 常规聊天 | 42.42% |
| 安全性 | 60.53% |
| 商品推荐 | 82.42% |
| 信使功能 | 60.61% |
| 总体 | 63.29% |

### 关键发现

- WG框架让格式遵循率提升最高45%(内部模型从0.655到0.951)
- 准确率提升最高14%
- WG-FT使开源模型(27-32B)在所有指标上达到或超过GPT-4o
- 人工评测中内部模型在除常规聊天外的所有类别均优于GPT-4o，语言流畅度是影响聊天类偏好的关键因素

## 亮点与洞察

- 将工业级agent的核心问题精准定位为"灵活性 vs 合规性"的矛盾，并通过DAG结构优雅解决
- 响应掩码微调是图结构agent训练的关键技巧——简单但有效地解决了多节点prompt冲突
- 原型agent辅助数据收集的思路具有高度实用性，降低了复杂agent标注数据收集的门槛
- 证明了中小规模开源模型通过合理的架构设计+微调可以在特定领域超越GPT-4o

## 局限与展望

- 数据收集高度依赖人工标注，成本高且存在人口统计偏差(标注员的性别和年龄分布有限)
- LLM-as-Judge评估可能无法完全反映人类偏好(如语言流畅度的影响)
- 未探讨LLM模拟用户进行自动数据收集的可能性
- 仅在电商场景验证，跨领域泛化性有待进一步验证

## 相关工作与启发

- 与LangGraph等图结构agent框架定位不同，本文聚焦于如何在图结构中实现生产级响应质量
- 启发：对于任何需要严格遵循多种约束的agent场景，将约束分散到独立节点是比堆砌prompt更好的工程实践
- MARCO等方案依赖独立guardrail组件验证+重试，增加延迟；本文通过结构化设计从源头减少违规

## 评分

- **新颖性**: ⭐⭐⭐ (DAG思路不新，但响应掩码微调和整体工业化方案有实用创新)
- **实验充分度**: ⭐⭐⭐⭐ (多模型对比+人工评测+真实部署，但缺少跨领域验证)
- **写作质量**: ⭐⭐⭐⭐ (工业论文风格，表述清晰，案例详实)
- **价值**: ⭐⭐⭐⭐ (对构建生产级agent有很高参考价值)

<!-- RELATED:START -->

## 相关论文

- [Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)
- [ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)
- [Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [REP: Keys to Robust Edits — From Theoretical Insights to Practical Advances](rep_robust_knowledge_editing.md)

<!-- RELATED:END -->
