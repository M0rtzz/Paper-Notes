---
title: >-
  [论文解读] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents
description: >-
  [ACL 2026][LLM推理][深度研究] 本文提出 FS-Researcher，一个基于文件系统的双 Agent 深度研究框架，通过 Context Builder 构建层次化知识库、Report Writer 分节撰写报告，利用持久化工作空间突破上下文窗口限制…
tags:
  - "ACL 2026"
  - "LLM推理"
  - "深度研究"
  - "文件系统"
  - "测试时扩展"
  - "知识库构建"
  - "双Agent框架"
---

# FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents

**会议**: ACL 2026  
**arXiv**: [2602.01566](https://arxiv.org/abs/2602.01566)  
**代码**: [https://github.com/Ignoramus0817/FS-Researcher](https://github.com/Ignoramus0817/FS-Researcher)  
**领域**: LLM推理  
**关键词**: 深度研究, 文件系统, 测试时扩展, 知识库构建, 双Agent框架

## 一句话总结

本文提出 FS-Researcher，一个基于文件系统的双 Agent 深度研究框架，通过 Context Builder 构建层次化知识库、Report Writer 分节撰写报告，利用持久化工作空间突破上下文窗口限制，在 DeepResearch Bench 上达到 53.94 RACE（SOTA），并展示了上下文构建计算量与报告质量的正相关测试时扩展效应。

## 研究背景与动机

**领域现状**：深度研究（Deep Research）是 LLM Agent 的前沿代表性任务，要求 Agent 从互联网系统性地收集证据并综合成长篇报告。OpenAI、Google、Anthropic 等已推出商业深度研究产品，展现了人类级别的性能。

**现有痛点**：(1) 模型上下文长度有限，而深度研究的长轨迹任务容易超出上下文容量，导致 Agent 执行中断；(2) 现有方法（静态管线、单 Agent 流程）中 thoughts、tool observations 和报告草稿竞争有限的 token 预算，导致覆盖不全和过早综合；(3) 当前的压缩策略（如摘要化 tool 观察）虽延长了轨迹，但引入有损瓶颈——细粒度证据和来源可能丢失，且仍受上下文硬限制约束。

**核心矛盾**：深度研究任务需要的信息量（数百个网页、数万 token 的报告）与模型上下文窗口容量之间存在根本性矛盾。现有方法要么截断信息，要么有损压缩，无法真正实现测试时扩展（分配更多计算以提升质量）。

**本文目标**：(1) 设计一个可扩展至上下文窗口之外的深度研究框架；(2) 验证框架能否通过增加计算来持续提升报告质量；(3) 在多个基准上超越闭源和开源 SOTA。

**切入角度**：受编程 Agent 和 AI IDE（Cursor、Claude Code）的启发——文件系统工作空间是长时间工具使用和迭代开发的有效基础设施。将此范式迁移到深度研究，用文件系统作为持久外部记忆。

**核心 idea**：用文件系统替代上下文窗口作为 Agent 的记忆基础设施——信息存入文件而非保留在上下文中，按需加载，支持无限扩展和跨 session 迭代优化。

## 方法详解

### 整体框架

FS-Researcher 是双 Agent 框架，分为两个阶段：(1) Context Builder（上下文构建器）接收研究主题，像图书管理员一样浏览互联网、撰写结构化笔记、归档原始网页，构建层次化知识库；(2) Report Writer（报告撰写器）以知识库为唯一事实来源，分节撰写报告。两个 Agent 共享同一文件系统工作空间，支持独立的迭代优化。工作空间包含交付物（知识库/报告）和控制文件（TODO、Checklist、Log）。

### 关键设计

1. **文件系统工作空间**:

    - 功能：提供持久化外部记忆，突破上下文窗口限制
    - 核心思路：工作空间包含两类文件：交付物（index.md、knowledge_base/、sources/、report.md）和控制文件（todos、checklist、logs）。所有文件以 Markdown 格式存储。Agent 在每个 session 开始时检查工作空间状态，制定计划并执行。session 结束时根据 checklist 审查，将未达标项标记为 [IN-PROGRESS]。工具集包括文件系统工具（ls、grep、read_file、insert/delete/replace）和网络浏览工具（search_web、read_webpage）
    - 设计动机：文件系统有三大优势：(a) 镜像人类处理复杂任务的原生环境；(b) 存储量远超上下文窗口，按需访问无溢出；(c) 中间产物持久可回溯，支持跨 session 迭代优化

2. **Context Builder（上下文构建器）**:

    - 功能：系统性收集、蒸馏和归档信息到知识库
    - 核心思路：交付物包含 index.md（目录，含主题分解和 KB 结构）、knowledge_base/（树状结构的笔记目录，每条陈述附引用指向 sources/）和 sources/（归档的原始网页）。工作流非线性——index.md 和 knowledge_base/ 随浏览过程动态更新。每个 session 结束时进行自检，识别知识库中的错误、缺口或冲突，标记为待处理。可迭代运行直到达到 session 预算或通过审查
    - 设计动机：与直接在上下文中累积事实不同，将信息外化到文件系统允许知识库增长到远超上下文容量，且结构化组织便于 Report Writer 按需检索

3. **Report Writer（报告撰写器）**:

    - 功能：基于知识库分节撰写高质量研究报告
    - 核心思路：移除网络浏览工具，仅允许从知识库读取事实。采用多 session 写作流程：第一个 session 创建大纲（同时作为 TODO），后续每个 session 选择一个章节撰写。每节完成后进行节级审查（根据 checklist），全部完成后进行报告级审查。若发现问题则重标相关章节为 [IN-PROGRESS]。无 session 预算限制
    - 设计动机：一次性生成整篇报告往往变成事实罗列，缺乏深度分析。分节写作提供频繁的重新锚定机会，结合知识库进行局部规划和自纠正

### 损失函数 / 训练策略

本文为框架工作，不涉及模型训练。使用标准 ReAct 架构驱动两个 Agent：$T_i, A_i = M_\theta(T_{j<i}, A_{j<i}, O_{j<i}, P)$，$O_i = Execute(A_i)$。支持 GPT-5、Claude-Sonnet-4.5、Gemini-2.5-Pro 等多种骨干模型。文件 I/O 延迟可忽略（<0.03% 总时间）。

## 实验关键数据

### 主实验

**DeepResearch Bench 性能对比**

| 方法 | 骨干模型 | Comp. | Insight | Instr. | Read. | RACE |
|------|---------|-------|---------|--------|-------|------|
| OpenAI-DeepResearch | - | 46.46 | 43.73 | 49.39 | 47.22 | 46.45 |
| Gemini-2.5-Pro-DR | - | 49.51 | 49.45 | 50.12 | 50.00 | 49.71 |
| WebWeaver | Qwen3-235B | 51.45 | 51.39 | 50.26 | 48.98 | 50.80 |
| RhinoInsight | Gemini-2.5-Pro | 50.51 | 51.45 | 51.72 | 50.00 | 50.92 |
| **FS-Researcher** | Claude-Sonnet-4.5 | **54.25** | **55.85** | **52.47** | **51.54** | **53.94** |
| **FS-Researcher** | GPT-5 | 51.96 | 54.44 | 52.14 | 51.26 | 52.76 |

**DeepConsult 性能对比**

| 方法 | Win% | Tie% | Lose% | Avg Score |
|------|------|------|-------|-----------|
| OpenAI-DeepResearch | 0.00 | 100.00 | 0.00 | 5.00 |
| WebWeaver | 66.16 | 12.14 | 21.68 | 6.94 |
| **FS-Researcher** (Claude) | **80.00** | 10.42 | 9.58 | **8.33** |

**BrowseComp 准确率**

| 方法 | 准确率 |
|------|-------|
| Claude-Sonnet-4.5 (官方) | 43.9% |
| FS-Researcher (Claude) | **55.0%** |
| GPT-5 (官方) | 54.9% |
| FS-Researcher (GPT-5) | **68.0%** |

### 消融实验

**模块消融（GPT-5 骨干，10 个采样查询）**

| 配置 | Comp. | Insight | Instr. | Read. | RACE |
|------|-------|---------|--------|-------|------|
| FS-Researcher (完整) | 51.96 | 54.44 | 52.14 | 51.26 | 52.76 |
| - 持久化工作空间 | 48.38(-3.58) | 46.49(-7.95) | 50.78 | 49.92 | 48.69(-4.07) |
| - 双Agent→单Agent | 40.90(-11.06) | 37.55(-16.89) | 46.30 | 44.78 | 42.41(-10.35) |
| - 分节写作→一次性生成 | 47.06(-4.90) | 45.64(-8.80) | 50.50 | 46.46 | 47.63(-5.13) |

### 关键发现

- FS-Researcher 在三个基准上一致超越闭源和开源 SOTA，证明文件系统范式的框架级优势独立于骨干模型
- 双 Agent 消融影响最大（RACE -10.35），说明证据收集与报告撰写的分离是核心设计
- 增加 Context Builder 轮次（3→5→10）持续提升报告质量（Insight 从 49.48 到 55.88），但可读性在 5 轮后略有下降，因为信息密度增加导致写作风格更技术化
- 持久化工作空间对 Insight 影响最大（-7.95），说明结构化知识库对深度分析至关重要
- 用更小的摘要模型压缩上下文可降低 Context Builder 成本 47%，质量损失可忽略

## 亮点与洞察

- 文件系统作为 Agent 外部记忆的范式转换——从"信息放在上下文中"到"信息放在文件中按需加载"，是一个简洁但深刻的架构创新
- 双 Agent 分离解决了一个根本问题：信息收集和报告撰写需要不同的认知模式，混合在一起会导致过早综合和浅层探索
- 测试时扩展效应（更多计算→更好报告）的成功验证为 Agent 系统的 scaling law 提供了初步证据

## 局限与展望

- 框架依赖较强的骨干模型——需要强大的多轮规划、网络搜索和长文写作能力，小模型可能频繁提前终止
- 可读性与全面性之间存在权衡——更丰富的知识库导致更技术化的写作风格
- 未研究多 Agent 协作（如多个 Context Builder 并行搜索不同子主题）
- 存储原始网页可能涉及版权和隐私问题

## 相关工作与启发

- **vs OpenAI/Google Deep Research**: 商业产品技术不透明，FS-Researcher 是可复现的开源替代方案，且在多个基准上超越
- **vs LangChain Open Deep Research**: 在相同 GPT-5 骨干下，FS-Researcher RACE 提升 +2.16，证明框架贡献独立于模型
- **vs 摘要压缩方法**: 摘要压缩是有损的且仍受上下文限制，文件系统方法是无损的且无上限

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 文件系统作为 Agent 记忆的范式创新简洁而有效，测试时扩展效应验证有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准、三个骨干模型、三个消融实验、scaling 分析和案例研究
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、方法描述详尽、消融设计合理
- 价值: ⭐⭐⭐⭐⭐ 为深度研究 Agent 提供了可复现的 SOTA 框架和设计原则

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)
- [\[ACL 2026\] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)
- [\[ACL 2026\] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)

</div>

<!-- RELATED:END -->
