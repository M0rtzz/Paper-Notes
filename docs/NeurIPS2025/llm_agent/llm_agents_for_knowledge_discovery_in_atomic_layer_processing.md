---
title: >-
  [论文解读] LLM Agents for Knowledge Discovery in Atomic Layer Processing
description: >-
  [NeurIPS 2025][LLM Agent][LLM agent] 通过让 LLM Agent 控制模拟化学反应器（黑盒函数），证明 Agent 能在无先验知识下通过试错探索、发现并总结未知化学系统的规则，揭示了 Agent 进行开放式科学发现的能力与局限。
tags:
  - NeurIPS 2025
  - LLM Agent
  - LLM agent
  - knowledge discovery
  - atomic layer processing
  - scientific exploration
  - tool-augmented reasoning
---

# LLM Agents for Knowledge Discovery in Atomic Layer Processing

**会议**: NeurIPS 2025  
**arXiv**: [2509.26201](https://arxiv.org/abs/2509.26201)  
**代码**: [https://github.com/awwerbro/ALDReactor](https://github.com/awwerbro/ALDReactor)  
**领域**: llm_agent  
**关键词**: LLM agent, knowledge discovery, atomic layer processing, scientific exploration, tool-augmented reasoning

## 一句话总结
通过让 LLM Agent 控制模拟化学反应器（黑盒函数），证明 Agent 能在无先验知识下通过试错探索、发现并总结未知化学系统的规则，揭示了 Agent 进行开放式科学发现的能力与局限。

## 研究背景与动机

LLM Agent 在材料科学中的应用主要集中在两类：

**知识整合**：综合、验证和整理领域知识

**自驱动实验室**：作为自动化实验流程的组件，追求特定优化目标

这两类应用都难以区分 Agent 的**训练时知识回忆**、**潜在知识发现**和**全新知识发现**三种能力。

本文提出一个关键问题：**LLM Agent 能否在没有具体目标函数的情况下，仅通过"探索系统"来发现全新知识？**

为解决评估问题，作者巧妙地设计了**完全虚构的系统**（自定义规则的"外星市场"和虚构化学反应器），确保 Agent 无法依赖训练时习得的知识，从而真正测试其发现能力。

这一问题高度相关于半导体工业的**原子层处理（ALP）**技术：发现和表征新型化学反应是耗时且昂贵的过程，AI/ML 有望加速这一过程。

## 方法详解

### 整体框架

核心思想：将 LangGraph 的工具功能重新用于提供**黑盒函数**，让 Agent 自由探索而不给定其他目标。

```
LLM Agent
    ├── 工具 1: perform_experiment(recipe) → 传感器数据摘要
    ├── 工具 2: retrieve_experiment(id) → 完整实验数据
    └── 目标: 描述系统规则（无其他优化目标）
```

**两个测试系统**：

1. **外星市场**（概念验证）：Agent 需要通过试错发现规则——外星人不卖名字中含"p"或"m"的东西
2. **ALP 反应器模拟**（核心实验）：Agent 控制虚拟化学反应器，通过压力传感器和石英晶体微量天平（QCM）来探索虚构化学物质之间的反应

### 关键设计

**ALP 反应器模拟系统**：

反应器建模为管道，离散化为 $N$ 段，温度均匀。核心方程：

气相物种反应输运方程：
$$\frac{\partial c_i(x,t)}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} - v\frac{\partial c_i}{\partial x} - \frac{4}{d}\sum_j r_{ij}$$

表面覆盖率变化：
$$\frac{\partial \theta_i}{\partial t} = \frac{1}{\sigma_i}\sum_j r_{ij}$$

反应速率 $r_i = k_i c_i \theta_i \sigma$，其中 $k_i$ 遵循 Arrhenius 动力学。

**三种实验配置**：

| 配置 | 化学物质数 | 可发现反应 | 难度 |
|-----|-----------|-----------|------|
| I | 2 (A, B) | ALD 沉积 | 简单，0.5s 曝光即可饱和 |
| II | 2 (A, B) | ALD 沉积（动力学更慢） | 困难，需 40s 曝光 + 升温 |
| III | 4 (A, B, C, D) | ALD + ALE + 钝化 + CVD | 复杂，大实验空间 |

配置 II 中，反应 2 的 Arrhenius 前因子降低 4 倍，化学物 B 的蒸气压降低（Antoine 方程 B 参数减半），要求 Agent 更有耐心地探索。

**信息限制设计**：
- Agent 仅能看到**压力**和**QCM 质量变化**信号（类似真实实验人员的有限探测能力）
- 完整反应器状态（560 维向量）不对 Agent 开放
- 实验输出经确定性叙述生成 + LLM 摘要，而非原始数据直接传递

### 损失函数 / 训练策略

无传统意义上的训练/损失函数。Agent 评估基于：
- **外星市场**：规则发现的正确性评分（每识别对一个字母得 1 分，多余规则扣 0.5 分）
- **ALP 反应器**：通过 UMAP 投影将 Agent 的实验轨迹与专家轨迹对比，评估覆盖的实验空间

模型选择：外星市场测试 gpt-5、gemini-2.5-pro、gpt-5-mini、gemini-2.5-flash、gemini-2.0-flash；ALP 反应器使用 gemini-2.5-pro。

## 实验关键数据

### 主实验

**外星市场结果**：

| 模型 | 无实验数限制 | 指定实验数后 |
|------|------------|------------|
| gpt-5 | 表现最佳（自主做更多实验） | 进一步改善 |
| gemini-2.5-pro | 较差（过早停止） | 显著改善 |
| gpt-5-mini | 较差 | 显著改善 |
| gemini-2.5-flash | 较差 | 显著改善 |
| gemini-2.0-flash | 较差 | 中等改善 |

关键发现：gpt-5 的优势不在于更强的推理能力，而在于它**自发进行了更多实验**；其他模型在被要求执行足够数量的实验后性能大幅提升。

**ALP 配置 I（简单双化学物质）**：3/3 次迭代均发现自限反应本质，成功执行 ALD 沉积，并进一步探索了动力学极限和分解生长。

**ALP 配置 II（困难双化学物质）**：

| 提示条件 | 时间 | 结果 |
|---------|------|------|
| IIa（标准提示） | 3600s | 3/3 陷入 CVD 低增长局部最小值 |
| IIb（标准提示） | 7200s | 3/3 仍陷于局部最小值 |
| IIc（+QCM 参考值） | 7200s | 1/3 局部最小值, 1/3 错误标记, 1/3 成功发现 |

**ALP 配置 III（四化学物质，完整系统）**：

| 迭代 | 发现的反应 | 未发现的反应 |
|------|-----------|------------|
| 1 | ALD (B+C), 钝化 (D vs B-C), CVD (高温) | ALE (未识别，仅用 A 作"清洁剂") |
| 2 | ALD, ALE, 钝化 (D vs 单脉冲 B/C) | CVD 共剂量, 完整钝化 |
| 3 | ALE (A+C), C 分解 | ALD (B+C), CVD |

### 消融实验

**持久性的影响**：
- 不指定实验数时，大多数模型过早停止探索
- 增加实验时间的效果有限（配置 IIa→IIb 无改善）
- 提供额外上下文（QCM 参考值）比单纯增加时间更有效

**路径依赖性**：
- 外星市场中几乎所有 Agent 都从 "apple" 开始，有时导致错误的"双 p 规则"
- ALP 配置 III 的三次迭代探索了实验空间的不同部分
- 这启发了群体策略——让多个子 Agent 分别探索，由监督 Agent 汇总

### 关键发现

1. **坚持是关键**：Agent（和人类一样）需要足够的实验资源才能克服噪声和局部最小值
2. **信号触发好奇心**：AG 需要检测到某些信号才会深入探究，无信号时两种策略有效：增加资源或提供更多上下文
3. **路径依赖本质**：不同起始点导致不同的发现轨迹，这不是缺陷而是特征——可用于群体探索策略
4. **知识发现的可信度**：由于系统完全虚构，可以确认 Agent 进行的是真正的新知识发现而非记忆回忆

## 亮点与洞察

1. **实验设计精妙**：使用虚构系统完美解决了"Agent 是否在使用训练时知识"的评估难题
2. **真正的开放式探索**：不给 Agent 具体目标（如优化某个属性），而是要求它描述系统——这更接近真实的科学发现
3. **对现实实验的直接可迁移性**：反应器配方格式与真实 LabVIEW 控制系统兼容，从模拟切换到物理实验仅需更换 API 端点
4. **"不知道"也有价值**：Agent 忽略先验知识可能开辟更新颖的探索路径，这是一个反直觉但深刻的洞见
5. **与 Duan et al. 的联系**：同期在生物系统中的类似工作表明，LLM Agent 的科学发现能力是跨领域的

## 局限与展望

1. **仅使用单 Agent**：未探索多 Agent 协作探索（虽然提到了群体策略的可能性）
2. **仅验证了模拟环境**：未在真实实验室中部署
3. **评估主观性**：ALP 部分的结论依赖人工检查 Agent 的陈述和实验轨迹
4. **模型温度未系统调优**：提到了温度可能影响探索多样性，但未充分实验
5. **输入空间仍然有限**：4 种化学物质已经具有挑战性，真实化学空间远更复杂
6. **缺少 cost-benefit 分析**：与人类专家相比，Agent 需要多少实验资源/时间来达到类似覆盖

## 相关工作与启发

- **与 Boiko et al. (2023) 的区别**：Coscientist 等工作聚焦于自动化已知实验流程，本文关注无目标的开放式发现
- **与 Duan et al. 的互补**：Duan 让 LLM 用 Python 建模生物系统的反应，本文让 LLM 通过实验探索化学系统——两种互补路径
- **"没有目标的伟大"（Stanley & Lehman 2015）**：作者引用这本书的核心理念——最重要的发现往往不是目标驱动的
- **启发**：可用类似框架评估 Agent 在任何模拟环境中的发现能力（物理、生物、经济模型等）

## 评分
- 新颖性: ⭐⭐⭐⭐ 用虚构系统测试 LLM 的知识发现能力是巧妙的方法论创新
- 实验充分度: ⭐⭐⭐ 多种配置和提示策略，但每种仅 3 次迭代，统计力度有限
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，实验描述详尽，讨论部分深刻
- 价值: ⭐⭐⭐⭐ 为 AI 辅助科学发现开辟了"无目标探索"这一重要研究方向

<!-- RELATED:START -->

## 相关论文

- [GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning](../../ICML2025/llm_agent/guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning.md)
- [NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search](../../ICML2025/llm_agent/improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati.md)
- [Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](../../ICML2025/llm_agent/evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)

<!-- RELATED:END -->
