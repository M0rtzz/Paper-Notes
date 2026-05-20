---
title: >-
  [论文解读] LegalAgentBench: Evaluating LLM Agents in Legal Domain
description: >-
  [ACL2025][LLM Agent][法律领域] 提出 LegalAgentBench，一个面向中国法律领域的 LLM Agent 综合评测基准，包含 17 个真实语料库、37 个工具和 300 个覆盖多跳推理与写作的任务，通过关键词匹配和过程进度率实现细粒度评估。
tags:
  - "ACL2025"
  - "LLM Agent"
  - "法律领域"
  - "benchmark"
  - "工具调用"
  - "多跳推理"
---

# LegalAgentBench: Evaluating LLM Agents in Legal Domain

**会议**: ACL2025  
**arXiv**: [2412.17259](https://arxiv.org/abs/2412.17259)  
**代码**: [CSHaitao/LegalAgentBench](https://github.com/CSHaitao/LegalAgentBench)  
**领域**: LLM Agent  
**关键词**: LLM Agent, 法律领域, benchmark, 工具调用, 多跳推理

## 一句话总结
提出 LegalAgentBench，一个面向中国法律领域的 LLM Agent 综合评测基准，包含 17 个真实语料库、37 个工具和 300 个覆盖多跳推理与写作的任务，通过关键词匹配和过程进度率实现细粒度评估。

## 研究背景与动机
- LLM Agent 在法律领域的应用潜力巨大，但现有通用领域 benchmark（如 AgentBench、ToolBench）无法捕捉真实司法认知与决策的复杂性和微妙之处
- 法律领域已有数据集大多聚焦于相对基础的任务（如案例检索、判决预测），而实际法律实践涉及深度案件分析、法律推理和基于大量法律判例的综合判断
- **核心问题**：缺乏专门针对法律场景评估 LLM Agent 工具调用、多步推理和领域知识运用能力的标准化 benchmark
- **解决思路**：构建一个基于真实法律数据、包含丰富工具和多层次任务的综合评测框架

## 方法详解

### 1. 环境设计：语料库与工具
- **17 个真实语料库**：14 个结构化表格数据库（公司基本信息、注册信息、子公司信息、法律案件文书、法院信息、律所信息、地址信息、限高案件、终本案件、失信案件、行政处罚案件等）+ 3 个文档检索库（法律知识、法条、指导性案例）
- **37 个专业工具**，分四类：
    - **文本检索器**（3 个）：从文档库中检索与查询相关的内容，使用 Embedding-3 作为默认检索器
    - **数学工具**（5 个）：执行加减乘除、排序、求最大/最小值等运算
    - **数据库工具**（28 个）：从特定数据库中根据查询条件提取列内容
    - **系统工具**（1 个）：Finish 工具，解析执行反馈并返回答案

### 2. 可扩展的任务构建框架（6 步流程）
1. **规划树构建**：基于工具间的调用关系构建规划树，根节点为未知实体（任务起点），分支对应可用工具，子节点包含工具调用后获得的信息
2. **路径选择**：分层采样 + 最大覆盖策略，从规划树中选取不同深度（1-hop 到 5-hop）和广度的路径，确保任务类型和难度的多样性
3. **实体选择**：遍历所有可能实体，选择能成功完成预定路径的实体
4. **问题改写**：使用 GPT-4 将模板化问题改写为更自然、更贴近真实使用习惯的表述，同时隐藏解题路径
5. **答案生成**：通过已知实体和工具链从语料库中程序化提取正确答案
6. **人工验证**：人工校验所有问题、解题路径和答案的正确性

### 3. 任务形式化定义
- 在每个时间步 t，Agent 执行动作 a_t，接收观测 o_t，更新状态 s_{t+1} = u(s_t, a_t, o_t)
- 动作由决策策略决定：a_t = pi(s_t, o_1, o_2, ..., o_{t-1})
- 迭代直到任务完成或达到最大迭代限制 T=10

### 4. 细粒度评估指标
- **成功率（Success Rate）**：提取工具调用结果中的关键词 key_answer，计算 Agent 输出与关键词的重合比例
- **过程进度率（Process Rate）**：额外标注中间步骤关键词 key_middle，综合 key_middle 和 key_answer 评估各阶段完成情况
- **BERTScore**：计算生成答案与参考答案的文本相似度

## 实验关键数据

### 表1：任务统计

| 属性 | 1-hop | 2-hop | 3-hop | 4-hop | 5-hop | Writing | ALL |
|---|---|---|---|---|---|---|---|
| 任务数 | 80 | 80 | 60 | 40 | 20 | 20 | 300 |
| 平均查询长度 | 88.29 | 87.90 | 99.37 | 118.33 | 110.25 | 1059.95 | 160.65 |
| 平均答案长度 | 74.20 | 40.84 | 45.53 | 63.48 | 86.20 | 678.75 | 99.24 |
| 平均 key_answer 数 | 1.88 | 1.44 | 1.20 | 1.40 | 2.25 | 10.25 | 2.14 |

### 表2：各模型在 LegalAgentBench 上的成功率（ReAct 方法）

| 模型 | 1-hop | 2-hop | 3-hop | 4-hop | 5-hop | Writing | ALL |
|---|---|---|---|---|---|---|---|
| GPT-4o | **0.926** | **0.840** | **0.750** | **0.642** | **0.612** | 0.654 | **0.791** |
| Qwen-max | 0.906 | 0.792 | 0.633 | 0.583 | 0.608 | 0.666 | 0.742 |
| GLM-4-Plus | 0.913 | 0.810 | 0.642 | 0.617 | 0.430 | 0.766 | 0.750 |
| Claude-sonnet | 0.895 | 0.698 | 0.475 | 0.479 | 0.457 | 0.657 | 0.658 |
| GPT-4o-mini | 0.933 | 0.650 | 0.400 | 0.421 | 0.258 | 0.609 | 0.616 |
| GLM-4 | 0.879 | 0.677 | 0.417 | 0.388 | 0.243 | 0.594 | 0.606 |
| GPT-3.5 | 0.642 | 0.285 | 0.117 | 0.100 | 0.133 | 0.085 | 0.299 |
| LLaMA3.1-8B | 0.602 | 0.154 | 0.075 | 0.071 | 0.060 | 0.087 | 0.236 |

**关键发现**：
- GPT-4o 在 ReAct 方法下取得最佳整体成功率 79.08%，且 token 消耗相对较少
- 随着 hop 数增加，所有模型性能显著下降（1-hop 最高 93% -> 5-hop 最高 61%），验证任务难度梯度有效
- ReAct 方法在多跳问题上通常优于 Plan-and-Solve 和 Plan-and-Execute，但 token 消耗更高
- 在 Writing 任务上，ReAct 反而表现不佳，因其逐步解决机制不适合需要并行处理的写作类任务
- GPT-3.5 和 LLaMA3.1-8B 成功率低于 30%，工具使用能力严重不足

## 亮点
- **首个法律领域 LLM Agent 评测基准**：填补了垂直领域 Agent benchmark 的空白
- **可扩展的任务构建框架**：基于规划树的 6 步流程可方便地扩展到新知识库和工具
- **细粒度评估**：过程进度率（Process Rate）不仅评估最终结果，还衡量中间步骤完成情况，提供更深入的诊断信息
- **真实数据**：17 个语料库均来自真实法律场景，可随时间更新以避免模型过拟合

## 局限与展望
- 当前仅覆盖中国法律体系，未来需扩展至多语言和多法律体系
- 300 个任务的规模相对有限，可能不足以全面评估所有法律场景
- 评估主要依赖关键词匹配，对语义等价但措辞不同的答案可能存在漏判
- 任务构建依赖 GPT-4 改写问题，可能引入特定偏好
- 未开源评测中使用的具体 prompt 模板的细节（仅在附录中部分公开）
- Writing 任务仅 20 个，样本量较少，可能不具代表性

## 与相关工作的对比
- **vs AgentBench**：AgentBench 是通用多环境评测平台，LegalAgentBench 专注法律垂直领域，提供领域特有的语料库和工具
- **vs ToolBench/ToolQA**：ToolBench 覆盖通用 API 调用，ToolQA 跨 8 个通用领域；LegalAgentBench 深入法律领域，工具与语料库高度专业化
- **vs AgentBoard**：AgentBoard 关注多轮交互中的细粒度进度率评估，LegalAgentBench 借鉴了此思路并将其应用于法律场景
- **vs 已有法律 NLP 数据集**：已有法律数据集聚焦单一任务（检索/判决预测），LegalAgentBench 要求多跳推理和工具调用的综合能力

## 评分
- 新颖性: ⭐⭐⭐⭐ (首个法律领域 Agent benchmark，填补重要空白)
- 实验充分度: ⭐⭐⭐⭐ (8 个模型 x 3 种方法，多维度指标分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，任务构建流程详尽)
- 价值: ⭐⭐⭐⭐ (对法律 AI 社区有重要参考价值，方法论可迁移至其他领域)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)
- [\[ACL 2025\] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)
- [\[ACL 2026\] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](../../ACL2026/llm_agent/mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)

</div>

<!-- RELATED:END -->
