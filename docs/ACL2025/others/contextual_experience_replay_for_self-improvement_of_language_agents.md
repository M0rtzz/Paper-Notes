---
title: >-
  [论文解读] Contextual Experience Replay for Self-Improvement of Language Agents
description: >-
  [ACL 2025][经验回放] CER（Contextual Experience Replay）提出了一个无需训练的语言 agent 自我改进框架，通过将过去的交互经验累积并合成为动态记忆缓冲区，在推理时让 agent 检索相关知识来增强新任务的决策…
tags:
  - "ACL 2025"
  - "经验回放"
  - "自我改进"
  - "语言智能体"
  - "上下文学习"
  - "网页导航"
---

# Contextual Experience Replay for Self-Improvement of Language Agents

**会议**: ACL 2025  
**arXiv**: [2506.06698](https://arxiv.org/abs/2506.06698)  
**代码**: 无  
**领域**: 其他  
**关键词**: 经验回放, 自我改进, 语言智能体, 上下文学习, 网页导航

## 一句话总结

CER（Contextual Experience Replay）提出了一个无需训练的语言 agent 自我改进框架，通过将过去的交互经验累积并合成为动态记忆缓冲区，在推理时让 agent 检索相关知识来增强新任务的决策，在 WebArena 上相对 GPT-4o baseline 提升了 51.0% 的成功率。

## 研究背景与动机

**领域现状**：LLM agent 在网页导航等序列决策任务中展现了潜力，典型方法包括 ReAct、SteP、AgentQ 等。这些方法通常直接将 LLM 放入环境中，通过 prompt 定义任务和可用操作，让 LLM 逐步决策。然而大多数方法将每个任务视为独立的——agent 在完成一个任务后不记住任何经验，下次遇到类似情况仍然可能犯同样的错。

**现有痛点**：（1）**缺乏环境特定知识**——LLM 的预训练知识是通用的，对特定网站的布局、操作逻辑、常见陷阱缺乏了解，导致在复杂环境中频繁失败；（2）**无法从经验中学习**——当前 agent 没有机制在推理阶段持续积累和利用过去的经验，每次都是"从零开始"；（3）**训练成本高**——通过微调来注入经验知识需要大量标注数据和计算资源。

**核心矛盾**：人类在操作网站时会自然地积累经验（如"这个网站的搜索功能在左上角"、"提交订单前需要先登录"），但 LLM agent 的上下文窗口在每个任务开始时都是空白的。如何在不训练的情况下让 agent 在推理时动态积累和利用经验，是提升 agent 实用性的关键。

**本文目标**：设计一个 training-free 的框架，让 LLM agent 能够在上下文窗口内实现经验的积累、合成和检索，从过去的任务执行中持续学习并改进后续任务的表现。

**切入角度**：作者从强化学习中经典的"经验回放"（experience replay）概念出发，将其适配到 LLM agent 的上下文学习范式中——不通过梯度更新来存储经验，而是将经验以文本形式存入一个动态记忆缓冲区，在新任务时通过检索相关经验来增强 agent 的决策。

**核心 idea**：将 RL 的经验回放思想迁移到 LLM 的上下文窗口中，通过积累、合成和检索过去的任务经验（环境动态、决策模式），在不训练的情况下实现 agent 的持续自我改进。

## 方法详解

### 整体框架

CER 的工作流程为：agent 在执行任务时与环境交互，完成（不论成功与否）后将交互轨迹处理并存入记忆缓冲区。在执行新任务时，根据当前任务描述从缓冲区中检索最相关的经验，将其作为 in-context 示例添加到 prompt 中，辅助 agent 做出更好的决策。随着任务执行的增多，缓冲区不断充实，agent 的表现也持续提升。

### 关键设计

1. **动态记忆缓冲区（Dynamic Memory Buffer）**:

    - 功能：存储和管理 agent 累积的交互经验，支持高效检索
    - 核心思路：缓冲区中的每条经验不是简单的交互轨迹记录，而是经过合成处理的结构化知识。包含两类信息：（1）环境动态知识——特定网站/页面的操作逻辑、元素位置、常见问题等；（2）决策模式——在特定场景下什么操作序列是有效的。缓冲区采用动态更新策略，新经验可以补充或替换旧经验，避免信息过时
    - 设计动机：直接存储完整轨迹会占用大量上下文窗口，且包含大量冗余信息。通过合成和结构化，可以在有限的上下文空间内存储更多有用的经验

2. **经验合成（Experience Synthesis）**:

    - 功能：从原始交互轨迹中提炼出可复用的知识
    - 核心思路：任务完成后，利用 LLM 对交互轨迹进行反思和总结，提取出关键的环境知识和决策规律。例如从"在 Reddit 上搜索帖子"的轨迹中提炼出"Reddit 的搜索框在顶部导航栏，需要先点击再输入关键词"。成功的轨迹提取正面经验，失败的轨迹提取教训和避坑指南。合成后的经验比原始轨迹更紧凑、更可复用
    - 设计动机：raw trajectory 包含大量步骤级细节，直接作为 few-shot 示例效率极低。经验合成将"What happened"转化为"What to know"，使知识更加泛化和紧凑

3. **上下文检索增强（Contextual Retrieval Augmentation）**:

    - 功能：在新任务执行时快速找到最相关的历史经验
    - 核心思路：给定新任务的描述和当前环境状态，通过相似度匹配（如基于任务描述的语义相似性、涉及的网站/功能的重合度等）从缓冲区中检索 top-k 最相关的经验。检索到的经验被格式化后插入到 agent 的 prompt 中，作为决策参考。随着缓冲区的增长，检索的质量也随之提升
    - 设计动机：不同任务需要不同的经验支撑。盲目地将所有经验塞入上下文既浪费 token 又可能引入噪声。通过检索只提供相关经验，在效率和效果之间取得平衡

### 损失函数 / 训练策略

CER 是完全 training-free 的框架，不涉及梯度更新或损失函数。所有的"学习"都发生在上下文窗口内，通过累积经验和调整 prompt 来实现。这使得 CER 可以直接应用于任何现成的 LLM，无需额外训练。

## 实验关键数据

### 主实验

在 WebArena 和 VisualWebArena 基准上的成功率对比：

| 方法 | WebArena 成功率 (%) | VisualWebArena 成功率 (%) | 是否需要训练 |
|------|-------------------|------------------------|------------|
| GPT-4o (baseline) | ~24.3 | ~26-28 | 否 |
| ReAct | ~15-20 | - | 否 |
| SteP | ~28-30 | - | 需微调 |
| AgentQ | ~35+ | - | 需 RL 训练 |
| **CER (本文)** | **36.7** | **31.9** | **否** |

CER 在 WebArena 上达到 36.7%，相对 GPT-4o baseline 提升了 **51.0%**。

### 消融实验

| 配置 | WebArena 成功率 | 说明 |
|------|----------------|------|
| Full CER | 36.7% | 完整框架 |
| w/o Experience Synthesis | 显著下降 | 直接用 raw trajectory，知识利用效率低 |
| w/o Retrieval (全部塞入) | 下降 | 无关经验引入噪声 |
| 仅使用成功轨迹 | 略降 | 失败经验也有价值（避坑指南） |
| 仅使用失败轨迹 | 明显下降 | 成功经验是主要收益来源 |
| 随任务数增加 | 持续提升 | 经验越多效果越好，验证了持续学习的有效性 |

### 关键发现

- **经验合成是关键**：去掉经验合成模块后性能显著下降，说明将 raw trajectory 提炼为结构化知识极其重要，直接拿轨迹做 few-shot 效率太低
- **失败经验也很有价值**：虽然成功经验贡献更大，但加入失败经验的"教训总结"仍能带来额外提升。这与人类学习的直觉一致——从错误中也能学到东西
- **经验越多越好，但边际递减**：随着缓冲区中经验条数增加，agent 表现持续提升，但增长速度逐渐放缓。这暗示了更智能的经验管理策略（如去重、压缩）的潜在价值
- **Training-free 也能接近训练方法的效果**：CER 不需要任何训练，但在 WebArena 上的表现接近甚至超过了需要 RL 训练的 AgentQ，证明了上下文学习范式的潜力

## 亮点与洞察

- **RL 概念到 LLM 上下文的优雅迁移**：将经验回放从参数空间搬到上下文空间是一个巧妙的类比。不需要改变模型参数，只通过管理 prompt 中的经验就能实现"学习"，大幅降低了部署成本
- **51% 的相对提升说明环境知识极度匮乏**：GPT-4o 作为强大的通用模型，仅因为缺少环境特定知识就表现不佳。CER 通过填补这个"知识空白"就获得了巨大提升，说明网页导航的核心挑战不在推理能力而在环境知识
- **经验合成比经验存储更重要**：这个发现启示我们，agent 的记忆系统关键不在于"记住多少"，而在于"提炼出什么"。这与认知科学中"schema theory"的理念不谋而合

## 局限与展望

- **上下文窗口限制**：所有经验都存储在上下文中，受限于 LLM 的上下文长度。经验量一旦超过窗口容量，就需要更复杂的管理策略
- **依赖 LLM 的合成能力**：经验合成的质量取决于 LLM 能否正确从轨迹中提炼关键知识，如果 LLM 本身对某些操作理解有误，合成的知识也会有误
- **环境特异性**：CER 积累的经验是环境特定的，跨网站/平台的迁移能力未验证
- **API 成本**：虽然不需要训练，但每次任务执行都需要额外的 LLM 调用来进行经验合成和检索，增加了推理成本
- **改进方向**：可以探索跨环境的经验迁移、更高效的经验压缩算法、以及将 CER 与训练方法结合以获得更大提升

## 相关工作与启发

- **vs ReAct**: ReAct 使用思考—行动—观察的循环来改进单个任务内的决策，但不跨任务积累经验。CER 在 ReAct 的基础上增加了跨任务的经验复用维度
- **vs SteP**: SteP 通过微调来注入经验，需要训练数据和计算资源。CER 获得了类似效果但完全 training-free，部署门槛更低
- **vs AgentQ**: AgentQ 使用 RL 训练来学习决策策略，是"参数空间的经验利用"。CER 是"上下文空间的经验利用"，两者互补，理论上可以结合
- **vs Reflexion**: Reflexion 也利用 LLM 反思来改进，但侧重单任务内的试错循环。CER 的经验缓冲区支持跨任务的知识迁移，更符合"持续学习"的理念

## 评分

- 新颖性: ⭐⭐⭐⭐ 经验回放到上下文学习的迁移思路新颖且优雅，training-free 的设计降低了使用门槛
- 实验充分度: ⭐⭐⭐⭐ WebArena + VisualWebArena 双基准评估，消融分析全面，51%的提升令人信服
- 写作质量: ⭐⭐⭐⭐ 20页的充实论文，方法描述清晰，分析深入
- 价值: ⭐⭐⭐⭐⭐ training-free 自我改进的实用价值极高，对 agent 持续学习研究方向有很好的推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](../../ICML2025/others/truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)

</div>

<!-- RELATED:END -->
