---
title: >-
  [论文解读] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration
description: >-
  [ACL 2026][LLM Agent][上下文窗口扩展] 提出 ExtAgents 多智能体框架，通过全局知识同步（所有Seeking Agent间交换信息）和知识累积推理（逐步向Reasoning Agent注入筛选后的知识）两个机制，解决现有多智能体方法在扩展外部知识输入超出上下文窗口时性能不升反降的瓶颈，在多跳QA和长综述生成任务上显著提升。
tags:
  - ACL 2026
  - LLM Agent
  - 上下文窗口扩展
  - 多智能体协作
  - 外部知识扩展
  - 多跳问答
  - 知识同步
---

# Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration

**会议**: ACL 2026  
**arXiv**: [2505.21471](https://arxiv.org/abs/2505.21471)  
**代码**: [GitHub](https://github.com/THUNLP-MT/ExtAgents)  
**领域**: LLM长上下文  
**关键词**: 上下文窗口扩展, 多智能体协作, 外部知识扩展, 多跳问答, 知识同步

## 一句话总结

提出 ExtAgents 多智能体框架，通过全局知识同步（所有Seeking Agent间交换信息）和知识累积推理（逐步向Reasoning Agent注入筛选后的知识）两个机制，解决现有多智能体方法在扩展外部知识输入超出上下文窗口时性能不升反降的瓶颈，在多跳QA和长综述生成任务上显著提升。

## 研究背景与动机

**领域现状**：随着后训练推理和信息检索技术的进步，LLM在上下文窗口内可整合更多检索知识来解决复杂任务，且更多知识通常带来更好的效果。

**现有痛点**：当外部知识量超出上下文窗口时，直接截断导致信息丢失；RAG受限于排序误差会遗漏关键证据；上下文压缩丢弃细微线索。多智能体分布式方法（如LLM×MapReduce）是新范式，但实验发现它们在知识量增加时性能不升反降。

**核心矛盾**：现有多智能体编排存在两个瓶颈——（1）知识同步带宽小，每个Agent仅能访问邻居的2个消息，需要多轮才能同步全局信息；（2）推理上下文冗余，把所有消息都塞入推理Agent导致信息过载。

**本文目标**：设计一个可扩展的多智能体框架，使任务性能随外部知识输入量持续提升，即使超出上下文窗口。

**切入角度**：简化Agent角色为两类（Seeking + Reasoning），针对两个瓶颈分别设计全局同步和累积推理机制。

**核心idea**：Seeking Agent全局交换并评分chunk相关性（带宽=N），Reasoning Agent通过多轮逐步增加top-k知识进行累积推理，避免一次性信息过载。

## 方法详解

### 整体框架

ExtAgents将输入分成N个chunk，每个分配给一个Seeking Agent。流程：（1）全局知识同步——所有Seeking Agent共享消息并评分chunk与查询的相关性；（2）知识累积推理——Reasoning Agent从最相关的chunk开始，每轮增加一倍知识量（top-2^s），逐步积累直到可回答或知识耗尽。高度可并行化。

### 关键设计

1. **全局知识同步（带宽=N）**：

    - 功能：每个Agent能访问所有其他Agent的信息
    - 核心思路：所有Seeking Agent的消息全局共享，每个Agent消化本地chunk并评价与查询的相关性。与Chain of Agents（带宽=2）和LLM×MapReduce（带宽=O(L/|m|)）相比，带宽直接等于Agent数N，实现真正的全局信息交换。可选地排除不相关chunk
    - 设计动机：小带宽需要多轮同步，信息在传递中退化；全局访问一次性完成同步

2. **知识累积推理**：

    - 功能：避免一次性信息过载，渐进式集成知识
    - 核心思路：Reasoning Agent按相关性排序接收chunk。第s轮接收top-$2^s$个chunk的消息。每轮推理后判断是否可回答，若不可回答则扩展知识量继续推理。最终输出答案或"无法回答"
    - 设计动机：一次性注入所有消息（如LLM×MapReduce）导致冗余信息淹没关键证据；渐进式注入让推理Agent在每轮聚焦于最相关的信息

3. **∞Bench+ 增强基准**：

    - 功能：消除现有长上下文基准的偏差
    - 核心思路：过滤掉仅需8k token窗口扫描即可回答的样本，保留真正需要跨文档信息聚合的多跳问题。结果En.QA从351样本降至157+大文档样本=294，Zh.QA从189降至56+大文档样本=184
    - 设计动机：发现原始∞Bench中大量问题可被简单截断上下文回答，不能真正测试长上下文能力

## 实验关键数据

### 主实验（∞Bench+ En.QA, gpt-4o-mini）

| 方法 | 8k input | 32k input | 128k input | 256k+ input |
|------|---------|----------|-----------|------------|
| 截断 | ~30 | ~35 | ~38 | N/A |
| LLM×MapReduce | ~32 | ~33 | ~34 | ~32 |
| ExtAgents | ~33 | ~38 | ~43 | **~46** |

### 关键发现
- ExtAgents是唯一在知识量增加时性能持续提升的方法，即使超出128k上下文窗口
- LLM×MapReduce在超出上下文窗口后性能反而低于截断，暴露了瓶颈
- ExtAgents在HotpotQA（大知识库多跳QA）上同样有效，验证泛化性
- 在长综述生成任务上也展现优势
- 高并行性保证了效率——Seeking Agent完全可并行

## 亮点与洞察
- **问题定义清晰有价值**：首次明确提出"扩展外部知识输入超出上下文窗口"的问题并构建评测框架
- **瓶颈分析精准**：将现有方法的失败归因到同步带宽和推理冗余两个具体瓶颈
- **设计简洁有效**：仅两类Agent + 两个机制，易于理解和实现
- **∞Bench+的构建有独立价值**：消除了现有长上下文基准的测量偏差

## 局限与展望
- **依赖LLM API**：需要多次调用LLM，成本较高
- **chunk分割策略简单**：仅用简单分割，未探索更智能的语义分块
- **评测覆盖有限**：主要在QA和综述生成上验证，其他长上下文任务未测试
- 未来方向：更智能的chunk策略、与RAG结合、后训练Agent协作能力

## 相关工作与启发
- **vs LLM×MapReduce**：SOTA多智能体方法但在扩展知识时性能不升反降；ExtAgents通过全局同步和累积推理克服
- **vs Chain of Agents**：带宽=2的序列式方法，扩展性差
- **vs RAG**：受限于检索排序误差，不能保证关键证据被选中

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义有价值，全局同步+累积推理的双重设计针对性强
- 实验充分度: ⭐⭐⭐⭐ 多任务多模型验证，有∞Bench+构建和效率分析
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，瓶颈分析系统
- 价值: ⭐⭐⭐⭐ 为LLM超长上下文推理提供了实用的无训练方案

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](../../AAAI2026/llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)
- [\[AAAI 2026\] A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis](../../AAAI2026/llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in.md)
- [\[ACL 2025\] Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents](../../ACL2025/llm_agent/beyond_numeric_rewards_in-context_dueling_bandits_with_llm_agents.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](../../NeurIPS2025/llm_agent/lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](../../ACL2025/llm_agent/metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)

<!-- RELATED:END -->
