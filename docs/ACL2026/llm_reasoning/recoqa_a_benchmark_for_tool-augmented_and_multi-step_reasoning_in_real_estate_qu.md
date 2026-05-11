---
title: >-
  [论文解读] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering
description: >-
  [ACL 2026][LLM推理][工具增强推理] 本文构建了 ReCoQA——一个包含 29,270 个房地产问答对的大规模基准，要求模型融合数据库查询和地图 API 调用进行混合多源推理，并提出层次化多 Agent 框架 HIRE-Agent 作为强基线，系统性地揭示了现有 LLM 在垂直领域复杂推理中的瓶颈。
tags:
  - ACL 2026
  - LLM推理
  - 工具增强推理
  - 多步推理
  - 房地产问答
  - 多Agent框架
  - 基准数据集
---

# ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering

**会议**: ACL 2026  
**arXiv**: [2604.17944](https://arxiv.org/abs/2604.17944)  
**代码**: [https://github.com/Husky-989/ReCoQA](https://github.com/Husky-989/ReCoQA)  
**领域**: LLM推理  
**关键词**: 工具增强推理, 多步推理, 房地产问答, 多Agent框架, 基准数据集

## 一句话总结
本文构建了 ReCoQA——一个包含 29,270 个房地产问答对的大规模基准，要求模型融合数据库查询和地图 API 调用进行混合多源推理，并提出层次化多 Agent 框架 HIRE-Agent 作为强基线，系统性地揭示了现有 LLM 在垂直领域复杂推理中的瓶颈。

## 研究背景与动机

**领域现状**：在房地产决策中，用户需要在多个平台间切换——在一个网站比较房源、在地图应用计算通勤时间、在政府网站查看学区信息。这种碎片化的信息获取方式造成了巨大的时间成本和认知负荷。AI Agent 是解决这一问题的有力方案，但现有的 QA 基准无法有效评估这种混合推理能力。

**现有痛点**：经典数据集如 Spider 只关注结构化查询，近期的 Agent 基准评估通用工具使用，但它们都将数据库查询和外部 API 调用视为独立能力。现有基准无法模拟实际场景中的混合工作流——比如数据库查询的输出（候选小区列表）需要动态地作为 API 调用（距离计算）的输入。此外，大多数地图 QA 数据集假设地理信息是静态的，忽略了通勤时间等动态值。

**核心矛盾**：真实世界的垂直领域决策需要异构信息源的紧密耦合和多步推理，但缺乏系统性评估这种能力的基准。

**本文目标**：（1）构建一个涵盖静态数据库查询、动态 API 调用和多步推理的端到端基准；（2）建立分层多 Agent 基线并系统分析各模块的瓶颈。

**切入角度**：以房地产购房咨询为切入点——这是一个天然需要数据库查询（房源属性）和地图 API（通勤距离、周边设施）紧密结合的垂直场景。

**核心 idea**：设计包含三种递进难度问题类型的大规模基准（简单查询、联合查询、多步推理），并用"理解-规划-执行"的层次化 Agent 架构作为强基线。

## 方法详解

### 整体框架
ReCoQA 包含两个贡献：数据集和 HIRE-Agent 框架。数据集覆盖 8 个中国主要城市，包含小区信息、POI 数据和位置配对数据，存储在 PostgreSQL 中，并提供 4 种地图 API 功能。HIRE-Agent 采用"前端 Agent + 监督 Agent + 专家 Agent"的三层架构。

### 关键设计

1. **三级递进难度的问题类型设计**:

    - 功能：系统性评估从简单到复杂的推理能力
    - 核心思路：Type 1（简单查询）仅需数据库直接查询；Type 2（联合查询）需要同时使用数据库和地图 API；Type 3（多步推理）需要链式推理——先查数据库获取坐标，再调 API 计算距离，最后基于 API 结果做比较推理。每个样本都带有完整的中间步骤标注（SLU 标签、SQL 语句、API 调用序列）
    - 设计动机：递进难度设计可以精确定位模型在推理链的哪个环节失败，中间步骤标注实现了可解释的评估

2. **前端 Agent（SLU 模块）**:

    - 功能：将用户的自然语言查询解析为结构化的意图和槽位表示
    - 核心思路：微调 BERT 模型进行意图检测（16 种预定义意图）和槽位填充（19 种槽位类型，IOB 标注），将"天河区通勤到集满佳广场 30 分钟以内"解析为具体的地点、交通方式和时间约束
    - 设计动机：将意图理解与执行解耦是必要的——消融实验表明，没有 SLU 标签时 Type 3 准确率平均仅 0.2921，加入后提升至 0.6535

3. **监督 Agent（任务编排）**:

    - 功能：接收结构化输入后进行任务分解和编排
    - 核心思路：使用 CoT 提示生成执行计划，按计划依次向专家 Agent 分派子任务，根据反馈动态调整（成功则继续、失败则重规划），设有最大步数限制防止死循环
    - 设计动机：集中式编排确保了多步推理的全局一致性，重规划机制提高了鲁棒性

### 损失函数 / 训练策略
SLU 模块使用交叉熵损失微调 BERT；LLM Agent 部分使用 ICL（5-shot）引导，不做额外训练。API 结果通过预缓存实现确定性和免费执行。

## 实验关键数据

### 主实验

| 模型 | 方法 | Type 1 Acc | Type 2 Acc | Type 3 Acc | Overall Acc |
|------|------|-----------|-----------|-----------|-------------|
| Qwen2.5-72B | Standard | 0.8082 | 0.7110 | 0.3973 | 0.5855 |
| Qwen2.5-72B | HIRE-Agent | 0.8862 | 0.6581 | **0.6211** | **0.6989** |
| Qwen3-30B A3B | Standard | 0.7394 | 0.5871 | 0.3512 | 0.5131 |
| Qwen3-30B A3B | HIRE-Agent | 0.7659 | **0.8645** | **0.8371** | **0.8260** |
| 平均 | Standard | 0.7741 | 0.6090 | 0.3559 | 0.5299 |
| 平均 | HIRE-Agent | **0.8453** | **0.7658** | **0.6535** | **0.7323** |

### 消融实验（瓶颈分析）

| 组件 | 关键指标 | 说明 |
|------|---------|------|
| 无 SLU → 有 SLU | Type 3: 0.2921 → 0.6535 | SLU 模块贡献最大，提升 +0.3614 |
| GT SQL 标签 | Qwen2.5-72B 提升 +0.1407 | SQL 生成是该模型的主要瓶颈 |
| GT API 标签 | Qwen3-8B 提升 +0.0660 | 工具调用是该模型的主要瓶颈 |
| 全部 GT 标签 | 平均准确率仅 0.8864 | 暴露了全局规划和最终推理的合成差距 |

### 关键发现
- 层次化架构平均提升 Overall 准确率 20.24 个百分点，在 Type 3 多步推理上提升最大（+29.76 个百分点）
- 即使提供全部中间步骤的 GT 标签，准确率仍只有 0.8864，说明存在"合成差距"——模型无法完美整合多个子任务的结果
- Qwen3-30B 出现了"过度思考"现象：在简单问题上表现反而差（复杂推理能力干扰了简单查询的直接执行）
- 各模型的瓶颈不同：Qwen2.5-72B 瓶颈在 SQL 生成，Qwen3-8B 瓶颈在工具调用，揭示了模型能力的异质性

## 亮点与洞察
- **渐进式瓶颈分析方法**极具价值——通过逐步替换 GT 标签来定位各模块的性能贡献，这种诊断方法可以迁移到任何多模块系统的评估
- **API 缓存策略**实现了基准的可复现性和零成本使用——将实时 API 调用结果预存储为本地数据库查询，既保证了结果的确定性又消除了 API 费用
- 发现了大模型的"过度思考"现象——强推理能力在简单任务上反而成为负担，这对 Agent 设计有重要启示

## 局限与展望
- 数据集目前仅覆盖中国 8 个城市的房地产场景，地理和文化局限性较大
- 问题基于 41 个模板生成，虽然经过改写但仍可能存在模式重复
- API 缓存策略虽然提高了可复现性，但无法反映实时 API 的延迟和错误处理
- 未来可以扩展到更多垂直领域（医疗、法律、金融），验证框架的通用性

## 相关工作与启发
- **vs Spider**: Spider 只评估 Text-to-SQL，不涉及外部 API 调用和多源融合；ReCoQA 要求数据库和 API 的紧密耦合
- **vs RETQA**: RETQA 引入了 SLU 但局限于静态数据库，不支持动态地理信息查询
- **vs MACT**: MACT 展示了复杂 Agent 协作，但依赖 Pandas 进行数据操作，存在内存和扩展性问题；HIRE-Agent 使用 SQL + API 的组合更具扩展性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性评估混合数据库-API 推理的垂直领域基准
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 LLM、多层消融、瓶颈分析、真实场景测试，极为详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰、分析深入
- 价值: ⭐⭐⭐⭐ 数据集和分析方法对 Agent 研究社区有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[ACL 2025\] Entropy-based Exploration Conduction for Multi-step Reasoning](../../ACL2025/llm_reasoning/entropy-based_exploration_conduction_for_multi-step_reasoning.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](../../ICLR2026/llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)

</div>

<!-- RELATED:END -->
