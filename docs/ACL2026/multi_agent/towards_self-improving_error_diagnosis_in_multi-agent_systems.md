---
title: >-
  [论文解读] Towards Self-Improving Error Diagnosis in Multi-Agent Systems
description: >-
  [ACL 2026][多智能体][多智能体故障归因] 提出 ErrorProbe 框架，通过 MAST 分类驱动的结构化分解、症状驱动的后向追踪和验证式记忆机制，在多智能体系统中实现自改进的语义故障归因，尤其在步骤级错误定位上大幅超越基线。
tags:
  - "ACL 2026"
  - "多智能体"
  - "多智能体故障归因"
  - "错误定位"
  - "自改进诊断"
  - "验证记忆"
  - "后向追踪"
---

# Towards Self-Improving Error Diagnosis in Multi-Agent Systems

**会议**: ACL 2026  
**arXiv**: [2604.17658](https://arxiv.org/abs/2604.17658)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 多智能体故障归因, 错误定位, 自改进诊断, 验证记忆, 后向追踪

## 一句话总结

提出 ErrorProbe 框架，通过 MAST 分类驱动的结构化分解、症状驱动的后向追踪和验证式记忆机制，在多智能体系统中实现自改进的语义故障归因，尤其在步骤级错误定位上大幅超越基线。

## 研究背景与动机

**领域现状**：基于 LLM 的多智能体系统（MAS）已在软件工程、Web 导航、科学推理等领域展现强大能力，但其调试问题日益突出。当系统由多个角色（架构师、工程师、测试员等）协作完成任务时，一旦失败，需要回答"哪个 agent 导致了错误？错误源于哪一步？"

**现有痛点**：现有诊断方法有三类缺陷：（1）基于分类学的人工标注方法（如 MAST）需要大量专家工作，难以规模化；（2）基于训练数据的专用追踪器依赖昂贵的数据生成管线，且需不断重训；（3）LLM-as-a-Judge 范式在长上下文的步骤级定位中表现不佳，特别是错误延迟显现的场景。

**核心矛盾**：MAS 中的错误归因面临多重挑战——交互轨迹极长（数十至上百轮）、错误延迟显现（早期错误在后期才暴露）、智能体间的因果依赖链复杂、故障模式多样化。这使得单次 LLM 判断无法有效穿透长上下文定位根因。

**本文目标**：设计一种无需人工标注、可自我改进的多智能体故障归因框架，能精确识别责任 agent 和错误起源步骤。

**切入角度**：模拟人类专家的调试过程——先将问题分解为多个专业角色（假设生成、验证执行、仲裁决策），通过后向追踪剪枝无关上下文，并利用经过验证的记忆库实现跨域模式复用。

**核心 idea**：将 MAST 分类法操作化为轻量检测器提供局部异常线索，结合症状驱动的后向追踪压缩上下文，再由"策略师-调查员-仲裁者"三人团队通过工具执行验证假设，最终通过验证门控更新记忆库实现自我改进。

## 方法详解

### 整体框架

ErrorProbe 是一个三阶段管线：输入为失败的多智能体交互轨迹和故障症状描述，输出为责任 agent、错误起源步骤和故障类型。首先通过 MAST 分类法检测局部异常标签，然后从症状出发进行后向追踪剪枝上下文，最后由三个专业 agent 协作诊断并更新验证式记忆。

### 关键设计

1. **MAST 引导的结构化分解**:

    - 功能：将原始交互轨迹转化为结构化表示，检测局部异常信号
    - 核心思路：先解析轨迹提取每步的 agent 身份、角色和动作类型，再用分类法条件提示检测步骤级偏差（如"工具输出被忽略"、"推理-动作不匹配"）。这些弱信号作为启发式先验，将搜索空间从 $L$ 步缩小到少量候选区域
    - 设计动机：原始轨迹噪声大且无结构，直接分析容易迷失。通过 MAST 的14种错误模式（规范问题、对齐失败、验证缺陷）提供语义锚点

2. **症状驱动的后向追踪**:

    - 功能：从故障症状出发反向重建因果链，压缩无关上下文
    - 核心思路：构建消息间的依赖图 $G=(V,E)$，从症状节点 $v_L$ 做广度优先搜索确定错误的有效感受野，屏蔽不相关的并行分支。这将原始长轨迹 $x$ 压缩为因果子集 $x' \subset x$
    - 设计动机：根因（如第5步的错误参数）和症状（如第50步的崩溃）之间可能跨越数十步，直接处理全部历史会导致"中间迷失"现象

3. **验证式记忆与三 Agent 诊断团队**:

    - 功能：通过"策略师-调查员-仲裁者"团队执行诊断，并维护经验证的错误模式记忆库
    - 核心思路：策略师检索记忆库中的历史模式并生成假设集合；调查员必须通过工具（CodeExec 沙箱、LogicProbe 条件验证）为每个假设提供可执行证据；仲裁者聚合证据做出最终判断，并决定是否将模式写入记忆。记忆更新需满足严格验证门控：$\text{Verify}(E_t) \land c_t > \tau$，防止幻觉污染
    - 设计动机：单纯的 LLM 判断容易产生错误归因幻觉，工具执行验证可提供客观证据；验证门控防止分布偏移下的记忆腐败

### 损失函数 / 训练策略

ErrorProbe 是一个无需训练的推理时框架。通过流式处理失败任务，每次诊断后根据验证结果选择性更新记忆状态 $\mathcal{M}_i \leftarrow \text{Update}(\mathcal{M}_{i-1}, x_i, \hat{y}_i, \text{Verify}(\hat{y}_i))$，实现自我改进。记忆检索使用结构匹配与质量加权的 RFI-Δ 评分组合，冷启动时退化为第一原理推理。

## 实验关键数据

### 主实验

| 基准 | 方法 | Agent 准确率 | Step 准确率 |
|------|------|-------------|------------|
| TracerTraj | LLM-as-a-Judge (Claude) | 67.7% | 8.7% |
| TracerTraj | ErrorProbe+Memory (Claude) | 73.2% | 39.4% |
| Who&When-Algo | LLM-as-a-Judge (Claude) | 55.6% | 41.3% |
| Who&When-Algo | ErrorProbe+Memory (Claude) | 60.3% | 59.5% |
| 三基准平均 | ErrorProbe+Memory (Claude) | 59.6% | 42.7% |
| 三基准平均 | LLM-as-a-Judge (Claude) | 57.0% | 21.3% |

### 消融实验

| 配置 | Agent 平均 | Step 平均 | 说明 |
|------|-----------|----------|------|
| LLM-as-a-Judge | 57.0% | 21.3% | 单次判断基线 |
| Agent-as-a-Judge (基线) | 46.4% | 24.7% | 工具增强但无结构化 |
| ErrorProbe (无记忆) | 56.3% | 41.9% | 有分解+追踪 |
| ErrorProbe (有记忆) | 59.6% | 42.7% | 完整框架 |

### 关键发现

- 步骤级定位是最大亮点：ErrorProbe 将 Claude 的 Step 准确率从 21.3% 提升至 42.7%，提升超过一倍
- 记忆模块对弱模型帮助更大：GPT-OSS-120B 从 25.8% 提升至 31.1%，Qwen3-32B 从 29.2% 提升至 34.9%
- 跨域迁移有效：从 KodCode 学到的模式可改善 TracerTraj 的诊断，验证门控成功过滤域特异性噪声
- GSM8K 域内记忆增益最大（Step +35%），因为该域错误模式重复性高

## 亮点与洞察

- **验证门控设计精巧**：只有经工具执行确认的诊断模式才写入记忆，避免了朴素缓存在分布偏移下的记忆腐败问题，这一思路可迁移到其他需要经验积累的 LLM agent 系统
- **后向追踪解决"中间迷失"**：通过依赖图剪枝将长轨迹压缩为因果子集，这一方法适用于所有需要在长上下文中定位因果关系的场景
- **三 Agent 团队模拟人类调试流程**：假设生成-证据收集-仲裁决策的分工设计使各环节可独立优化

## 局限与展望

- 依赖显式故障信号，对"静默失败"（技术正确但语义错误的输出）无法检测
- 多 Agent 诊断团队的推理开销较大，不适用于超低延迟场景
- 仅在三个模型族上验证，未覆盖更多架构
- 未来可引入测试时预言反馈机制来暴露潜在错误

## 相关工作与启发

- **vs LLM-as-a-Judge**: LLM-as-a-Judge 在步骤定位上严重不足（<10% on TracerTraj），ErrorProbe 通过结构化分解+后向追踪解决了长上下文中的因果定位难题
- **vs TracerTraj 训练式追踪器**: 训练式方法依赖昂贵的反事实重播数据，且需持续重训。ErrorProbe 无需训练，通过验证记忆实现渐进式改进

## 评分

- 新颖性: ⭐⭐⭐⭐ 验证式记忆和后向追踪的组合很有新意，但核心思路（多 Agent 协作诊断）不算全新
- 实验充分度: ⭐⭐⭐⭐ 三个基准+三个模型+丰富消融+记忆缩放分析，较为充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细，但部分内容略显冗长

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ICML 2026\] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](../../ICML2026/multi_agent/mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)

</div>

<!-- RELATED:END -->
