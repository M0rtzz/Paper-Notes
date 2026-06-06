---
title: >-
  [论文解读] SynthAgent: Adapting Web Agents with Synthetic Supervision
description: >-
  [ACL 2026][LLM Agent][合成数据] 本文提出 SynthAgent，一个完全基于合成监督的 Web Agent 适应框架，通过分类探索系统覆盖网页功能区域以合成多样化任务，再通过任务精炼（冲突检测触发修正幻觉）和轨迹精炼（全局视角去噪）的双重精炼策略提升合成数据质量…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "合成数据"
  - "Web Agent"
  - "双重精炼"
  - "分类探索"
  - "轨迹质量"
---

# SynthAgent: Adapting Web Agents with Synthetic Supervision

**会议**: ACL 2026  
**arXiv**: [2511.06101](https://arxiv.org/abs/2511.06101)  
**代码**: [GitHub](https://github.com/aiming-lab/SynthAgent)  
**领域**: LLM Agent / Web Agent 适应  
**关键词**: 合成数据, Web Agent, 双重精炼, 分类探索, 轨迹质量

## 一句话总结

本文提出 SynthAgent，一个完全基于合成监督的 Web Agent 适应框架，通过分类探索系统覆盖网页功能区域以合成多样化任务，再通过任务精炼（冲突检测触发修正幻觉）和轨迹精炼（全局视角去噪）的双重精炼策略提升合成数据质量，在 WebArena 和 Online-Mind2Web 上显著优于现有合成方法。

## 研究背景与动机

**领域现状**：LLM 驱动的 Web Agent 在标准化基准上展现出强大的网页交互能力，但在部署到训练中未见过的新网站时性能急剧下降。适应新环境需要环境特定的任务和演示数据，而人工标注成本高昂且无法规模化。

**现有痛点**：(1) Self-Instruct 直接让 LLM "想象" 任务，缺乏环境基础，任务简单且重复；(2) OS-Genesis 从单步观察反向合成任务，上下文不足导致大量幻觉（引用不存在的元素或状态）；(3) Explorer 在执行中持续精炼任务，但频繁改变任务意图（平均 8.6 次），导致 68.3% 的轨迹超出步数预算。

**核心矛盾**：任务合成需要环境基础（grounding）来避免幻觉，但在执行过程中过度基础化任务会引入轨迹噪声——这是合成监督中一个根本性的设计张力。

**本文目标**：设计一个完全合成的监督框架，在无需人工参与和测试集泄露的前提下，高效适应 Web Agent 到新环境。

**切入角度**：将任务精炼和轨迹精炼解耦为协同互补的两阶段：任务精炼确保可行性但引入噪声，轨迹精炼随后消除噪声。

**核心 idea**：双重精炼（dual refinement）——在执行中仅当检测到显式冲突时才精炼任务（冲突触发式，而非持续式），执行后利用全局上下文精炼轨迹，从而同时保证任务可行性和轨迹质量。

## 方法详解

### 整体框架

SynthAgent 由四个阶段组成：(1) **分类探索任务合成**——将网页元素按功能分组，均匀采样交互三元组 $(o_t, a_t, o_{t+1})$，让 LLM 基于真实界面转换提出多步任务；(2) **冲突触发任务精炼**——在轨迹收集中检测任务与观察的冲突，仅在触发时精炼任务（平均仅 2.0 次）；(3) **全局轨迹精炼**——事后利用完整轨迹和最终任务 $\tau^{\star}$ 移除噪声和不对齐动作；(4) **Agent 微调**——在精炼后的合成数据上 SFT 开源模型。训练损失为标准自回归交叉熵：$\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(\tau^{\star}, h^{\star}) \sim \mathcal{D}} \left[ -\sum_{t=1}^{T} \log p_\theta(a_t | \tau^{\star}, o_{\leq t}, a_{<t}) \right]$。

### 关键设计

1. **分类探索（Categorized Exploration）**:

    - 功能：系统覆盖网页的功能区域，提升任务多样性
    - 核心思路：在每个页面 $o_t$ 上，LLM 将交互元素按语义角色分类（如"账户管理"、"搜索过滤"），每个类别均匀采样最多 2 个未访问元素进行交互。每个类别的采样预算上限防止单个密集区域主导探索
    - 设计动机：随机探索（OS-Genesis）常反复访问冗余元素而遗漏重要功能区域；分类探索将其转化为功能感知的覆盖问题，每页平均发现 6.0 个功能类别

2. **冲突触发任务精炼（Conflict-Triggered Task Refinement）**:

    - 功能：修正任务中的幻觉，同时最小化对轨迹的干扰
    - 核心思路：定义轻量级冲突谓词 $\mathcal{C}(h_t, \tau_t) = \neg\textsf{ExistsUI} \vee \textsf{MissingArgs} \vee \textsf{Stall}$，分别检测 UI 元素不存在、参数缺失和执行停滞。仅在冲突触发时调用 LLM 精炼任务，遵循四条原则：具体化缺失细节、对齐实际观察、降低范围、保持类别
    - 设计动机：与 Explorer 每步持续精炼（8.6 次/平均，68.3% 超时）不同，冲突触发式精炼仅 2.0 次/平均，6.3% 超时——初始任务已充分指定，精炼聚焦于"修正"而非"重新定义"

3. **全局轨迹精炼（Trajectory Refinement with Global Context）**:

    - 功能：事后去除轨迹中的噪声和不对齐片段
    - 核心思路：离线审查完整轨迹 $h_T$ 和最终任务 $\tau^{\star}$，执行四类编辑操作：Remove(i) 删除无关/冗余步骤，Reorder(i,j) 交换可交换步骤，Drop($h_T$) 丢弃过于嘈杂的轨迹，Keep($h_T$) 保留良好轨迹。设计刻意偏向精确性——不确定的交换被拒绝而非冒险破坏因果依赖
    - 设计动机：任务精炼引入的噪声需要事后全局视角来清理；Reorder 仅占编辑操作的 4.1%，但被重排序的轨迹获得更高的偏好胜率（42% vs 27%）

### 损失函数 / 训练策略

标准 SFT 范式，历史上下文窗口设为最近 3 步。每个网站合成最多 500 个任务-轨迹对，混合五个网站的数据训练单一模型（学习率 1e-5，batch size 32，3 epochs）。

## 实验关键数据

### 主实验

**WebArena（5 个网站）- Qwen2.5-VL-7B 骨干**

| 方法 | 训练数据 | Shopping | CMS | Reddit | Gitlab | Maps | Overall |
|------|---------|---------|-----|--------|--------|------|---------|
| Base Qwen | - | 13.71 | 8.24 | 9.43 | 6.18 | 5.50 | 8.80 |
| +Self-Instruct | 合成 | 18.18 | 8.77 | 3.85 | 12.50 | 9.38 | 11.50 |
| +OS-Genesis | 合成 | 14.55 | 10.53 | 11.54 | 16.07 | 12.50 | 13.27 |
| +Explorer | 合成 | 10.91 | 3.51 | 0.00 | 1.82 | 3.12 | 4.44 |
| **+SynthAgent** | **合成** | **20.00** | **21.05** | **15.38** | **19.64** | **28.12** | **20.80** |

**Online-Mind2Web（136 个真实网站）**

| 方法 | GPT-4.1 Judge | GPT-5.1 Judge | WebJudge | 平均 |
|------|-------------|-------------|---------|------|
| Self-Instruct | 17.67 | 13.00 | 19.67 | 16.78 |
| OS-Genesis | 19.53 | 11.00 | 19.33 | 16.62 |
| **SynthAgent** | **31.67** | **15.67** | **23.33** | **23.56** |

### 消融实验

| 配置 | Overall | 变化 |
|------|---------|------|
| SynthAgent (完整) | 20.80 | - |
| w/o 分类探索 | 17.26 | -3.54 |
| w/o 任务精炼 | 15.93 | -4.87 |
| w/o 轨迹精炼 | 16.81 | -3.99 |
| w/o 双精炼 | 15.93 | -4.87 |

### 关键发现

- Explorer 性能甚至低于 base 模型——持续精炼产生了过长、不对齐的"负面监督"轨迹
- 合成数据质量：SynthAgent 轨迹质量 82.6 远超 Explorer 的 36.4 和 OS-Genesis 的 52.0
- SynthAgent 轨迹完成率 96.5% vs Explorer 30.5%，API 成本更低（$0.13 vs $0.22/轨迹）
- 在 Qwen3 更强骨干上仍有提升（15.93→24.34），验证方法的模型无关性

## 亮点与洞察

- "任务精炼和轨迹精炼是协同的"这一设计洞察精准——前者确保可行性但引入噪声，后者消除噪声
- 冲突触发式 vs 持续式精炼的对比揭示了关键设计原则：初始任务质量决定精炼策略
- 分类探索将随机探索转化为结构化覆盖问题，简单但有效

## 局限与展望

- 仅在离线和有限在线环境验证，未探索真实活跃网站的合成
- 任务和轨迹合成完全依赖 GPT-4.1，未探索更先进 LLM 或参数优化
- 仅使用标准 SFT，未探索 DPO 或在线 RL 等更高级训练方法

## 相关工作与启发

- **vs OS-Genesis**: OS-Genesis 从单步观察合成任务导致幻觉；SynthAgent 通过分类探索+冲突触发精炼解决
- **vs Explorer**: Explorer 持续精炼导致意图漂移和轨迹过长；SynthAgent 的冲突触发式精炼保持意图一致性
- **vs AgentTrek**: AgentTrek 依赖离线教程可能过时；SynthAgent 直接与环境交互合成

## 评分

- 新颖性: ⭐⭐⭐⭐ 双重精炼的协同设计和冲突触发机制是清晰的创新点
- 实验充分度: ⭐⭐⭐⭐⭐ 两个基准+多骨干+详细消融+数据质量分析+规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，设计张力分析深入
- 价值: ⭐⭐⭐⭐ 为 Web Agent 无监督适应提供了实用且高质量的合成数据方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Towards Scalable Oversight via Partitioned Human Supervision](../../ICLR2026/llm_agent/towards_scalable_oversight_via_partitioned_human_supervision.md)
- [\[ACL 2026\] Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces](dont_click_that_teaching_web_agents_to_resist_deceptive_interfaces.md)

</div>

<!-- RELATED:END -->
