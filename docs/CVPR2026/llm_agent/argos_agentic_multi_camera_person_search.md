---
title: >-
  [论文解读] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search
description: >-
  [CVPR 2026][LLM Agent][多摄像头搜索] 本文提出 ARGOS，首个将多摄像头行人搜索重新定义为交互式推理问题的基准和框架，智能体通过与目击者进行多轮对话、调用时空工具并在信息不对称下推理排除候选人，包含 2,691 个任务、3 个渐进式赛道。
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "多摄像头搜索"
  - "智能体推理"
  - "时空拓扑图"
  - "交互式对话"
  - "行人搜索"
---

# ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search

**会议**: CVPR 2026  
**arXiv**: [2604.12762](https://arxiv.org/abs/2604.12762)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: 多摄像头搜索, 智能体推理, 时空拓扑图, 交互式对话, 行人搜索

## 一句话总结
本文提出 ARGOS，首个将多摄像头行人搜索重新定义为交互式推理问题的基准和框架，智能体通过与目击者进行多轮对话、调用时空工具并在信息不对称下推理排除候选人，包含 2,691 个任务、3 个渐进式赛道。

## 研究背景与动机

1. **领域现状**：多摄像头行人搜索是监控领域的基础需求。传统行人重识别依赖清晰的视觉查询，文本驱动和交互式方法仅使用外观描述。现有空间推理基准和智能体评估框架局限于单图或通用场景。
2. **现有痛点**：现有方法缺乏主动提问规划能力，无法利用目击者提供的时空线索（如"我在仓库看到他们，几分钟后在大厅附近"）。没有方法同时整合多模态交互、空间定位和时间推理。
3. **核心矛盾**：真实世界的行人搜索本质上是一个主动推理问题——需要在信息不对称下决定"问什么、何时调用工具、如何解释模糊回答"，但现有基准和方法都将其简化为被动的视觉匹配。
4. **本文目标**：定义交互式多摄像头行人搜索任务，构建包含语义感知（Who）、空间推理（Where）和时间推理（When）的渐进式基准。
5. **切入角度**：将摄像头网络编码为时空拓扑图（STTG），作为任务构建的结构骨架和智能体的定位工具，支持基于物理约束的时间可行性推理。
6. **核心 idea**：用 LLM 驱动的四模块智能体（分析→规划→访谈→解释）在 STTG 上进行多轮对话推理，通过工具调用消除不可行的候选人。

## 方法详解

### 整体框架
智能体接收初始目击者陈述和行人库 $\mathcal{G}$，在有限回合内通过多轮对话确定目标行人。对话中可选择三类动作：询问视觉属性、查询空间位置、调用时间推理。STTG 提供摄像头连接关系和经验验证的转移时间。

### 关键设计

1. **时空拓扑图（STTG）**:
    - 功能：编码摄像头网络的物理连接和时间约束
    - 核心思路：有向加权图 $\mathcal{T} = (V, E)$，节点为摄像头（含区域标签），边携带类型（OVERLAP=共享视野、SOFT_ADJ=软相邻、TRAVEL=远距离）和转移时间统计 $(t_{\min}, t_{\text{med}}, t_{\max}, n)$。OVERLAP 边的连通分量定义区域。STTG 双重角色：基准从中生成真值任务，智能体将其作为环境表示进行推理。
    - 设计动机：将原本模糊的"从A到B需要多久"转化为可计算的图约束，使时间推理有据可循

2. **三赛道渐进式基准**:
    - 功能：分层评估从感知到时空推理的递进能力
    - 核心思路：Track 1 Who（989 任务）测试语义感知——智能体接收完整对话记录，提取属性过滤行人库；Track 2 Where（550 任务）测试空间推理——目击者报告在某区域看到目标，智能体通过空间和属性问题定位具体子区域（oracle 平均 2.02 轮）；Track 3 When（1,152 任务）测试时间推理——目击者报告两次不同时间地点的目击，智能体利用 STTG 排除转移时间不可行的候选人（oracle 平均 1.89 轮）。采用 Turn-Weighted Success (TWS) 指标联合衡量正确率和轮次效率。
    - 设计动机：渐进式设计允许精确诊断能力瓶颈；TWS 借鉴具身导航中 SPL 的设计思想

3. **四模块 LLM 智能体**:
    - 功能：通过结构化推理管线高效完成行人搜索
    - 核心思路：Analyst 查询行人库并计算属性消除力；Planner 决定下一步动作；Interviewer 通过工具执行动作（8 种工具：行人库查询、区域结构检索、目击者交互、STTG 时间可行性检查、过滤/预测）；Interpreter 解析目击者回答并应用过滤。关键设计是"信息边界"：智能体不知道目击者能回答哪些属性（21 个中仅 3 个可观测），必须在不确定性下做策略决策。
    - 设计动机：将复杂的推理任务分解为可控的模块化步骤，每个模块有明确职责

### 损失函数 / 训练策略
无训练，使用冻结的 LLM 骨干（GPT-5.2、GPT-4o、GPT-5-mini、Claude Sonnet 4）直接推理。温度 0.0，20 轮预算。

## 实验关键数据

### 主实验

| 骨干模型 | Track 2 TWS | Track 2 Top-1 | Track 3 TWS | Track 3 Top-1 |
|---------|------------|--------------|------------|--------------|
| Oracle | 1.000 | 100.0% | 1.000 | 100.0% |
| GPT-5.2 | 0.338 | 73.1% | **0.590** | **88.2%** |
| Claude Sonnet 4 | **0.383** | **76.0%** | 0.548 | 82.8% |
| GPT-4o | 0.323 | 74.5% | 0.567 | 80.6% |

### 消融实验

| 配置 | Track 3 TWS | 说明 |
|------|------------|------|
| 完整工具集 | 0.590 | GPT-5.2 |
| 移除时空工具 | ~0.30 | 下降 49.6 百分点 |
| 移除属性分析工具 | ~0.45 | 策略选择变差 |

### 关键发现
- **基准远未被解决**：最佳 TWS 仅 0.383（Track 2）和 0.590（Track 3），Oracle 均为 1.0
- **工具移除造成巨大性能下降**（49.6 百分点），证明领域特定工具对任务至关重要
- **空间推理是最大瓶颈**：Track 2 的 TWS 远低于 Track 3，因为空间消歧需要更多轮次且更依赖策略规划

## 亮点与洞察
- **将行人搜索重定义为交互式推理**是视角创新：从被动的视觉匹配转为主动的对话推理，更贴近真实安防场景中人与系统的交互模式
- **STTG 的双重角色**设计巧妙：既是数据集构建的结构骨架（保证任务有明确真值），又是智能体的推理工具（提供可计算的时空约束）
- **信息边界设计**增加了任务的策略深度：智能体不知道目击者能回答什么，必须在有限预算下智能探索

## 局限与展望
- 目击者模拟器是确定性的（固定模板），缺乏真实人类回答的噪声和歧义
- 仅使用 3 个可观测属性（性别、上衣颜色、下衣颜色），实际场景中目击者可能提供更丰富的描述
- 16 个摄像头的规模较小，未验证在大规模摄像头网络上的可扩展性
- 未来可引入视觉理解能力，让智能体直接从摄像头画面中提取信息

## 相关工作与启发
- **vs 传统 Re-ID**: Re-ID 是给定图像查找匹配，ARGOS 是通过对话和推理主动缩小候选范围，信息获取方式根本不同
- **vs GT-Bench / VS-Bench**: 这些是多智能体博弈基准，ARGOS 是单智能体在结构化环境中的推理基准，侧重于工具使用和时空约束

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将多摄像头行人搜索定义为交互式推理问题，STTG 设计原创
- 实验充分度: ⭐⭐⭐⭐ 四个 LLM、三个赛道，但缺少与传统 Re-ID 方法的对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，赛道设计逻辑连贯
- 价值: ⭐⭐⭐⭐ 开辟了行人搜索的新范式，基准有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](../../ACL2026/llm_agent/bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ACL 2026\] Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](../../ACL2026/llm_agent/rethinking_reasoning-intensive_retrieval_evaluating_and_advancing_retrievers_in_.md)
- [\[AAAI 2026\] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](../../AAAI2026/llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)

</div>

<!-- RELATED:END -->
