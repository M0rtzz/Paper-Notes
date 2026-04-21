---
title: >-
  [论文解读] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies
description: >-
  [NeurIPS 2025][LLM/NLP][多智能体规划] 提出 SYMPHONY，一个基于 MCTS 的多智能体规划框架，通过异构 LLM 池的多样性驱动搜索、UCB 自适应调度、熵调制置信度评估和池级记忆共享，显著提升了 LLM 规划的多样性和效率。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 多智能体规划
  - MCTS
  - 异构模型
  - LLM协作
  - 树搜索
---

# SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies

**会议**: NeurIPS 2025  
**arXiv**: [2601.22623](https://arxiv.org/abs/2601.22623)  
**代码**: https://github.com/ZHUWEI-hub/SYMPHONY (有)  
**领域**: LLM Agent / 模型压缩  
**关键词**: 多智能体规划, MCTS, 异构模型, LLM协作, 树搜索

## 一句话总结
提出 SYMPHONY，一个基于 MCTS 的多智能体规划框架，通过异构 LLM 池的多样性驱动搜索、UCB 自适应调度、熵调制置信度评估和池级记忆共享，显著提升了 LLM 规划的多样性和效率。

## 研究背景与动机
将 LLM 与 MCTS 结合用于复杂任务规划是近年热点（RAP、LATS、MASTER 等），但现有方法几乎都采用**单模型范式**：反复查询同一个 LLM 来生成搜索分支。问题在于，单个 LLM 多次采样的输出往往高度相似，反映同一种主导推理模式，导致：
- 搜索树中充满冗余的相似分支
- 探索能力受限，容易陷入局部最优
- 需要额外的采样和 token 消耗来覆盖解空间

**核心矛盾**：MCTS 需要多样的 rollout 来有效探索，但单一 LLM 的变异性不足以支持这种多样性。

**本文切入角度**：用一组异构 LLM（不同预训练来源、不同推理风格）替代单一模型，从"模型级多样性"角度提升搜索树的分支多样性。配合自适应调度和协作记忆保证高效协调。

## 方法详解

### 整体框架
SYMPHONY 在标准 MCTS 框架中引入四个关键组件：(1) 异构智能体池提供多样化的分支生成；(2) UCB 调度策略动态选择当前最优智能体；(3) 熵调制置信度评分 (EMCS) 校准节点价值估计；(4) 池级记忆共享支持跨智能体的反思学习。

### 关键设计

1. **异构智能体池**：

    - 维护 $\mathcal{M}^{(k)} = \{M_1^{(k)}, \cdots, M_n^{(k)}\}$ 多个异构 LLM
    - SYMPHONY-S（消费级硬件）：Qwen2.5-7B + Mistral-7B + Llama-3.1-8B
    - SYMPHONY-L（API级）：GPT-4 + Qwen-Max + DeepSeek-V3
    - 统一输入输出接口，支持模块化替换
    - 从理论上证明：从集成中以非零概率采样多个智能体，期望误差严格低于确定性选择单一智能体

2. **UCB 自适应调度**：

    - 将智能体选择建模为多臂老虎机问题
    - $\text{UCB}(M_i^{(k)}) = \bar{Q}(M_i^{(k)}) + \alpha \cdot \sqrt{\frac{\ln N_{total}}{N(M_i^{(k)})+1}}$
    - 兼顾对高分智能体的利用和对低频智能体的探索
    - 用于 MCTS 的扩展、评估和反思三个阶段

3. **池级记忆共享**：

    - 失败轨迹触发 UCB 选出的智能体生成自然语言反思 $\mathcal{R}_i^k$
    - 反思广播到全部智能体，作为共享记忆块注入 prompt
    - 固定大小缓冲区 + FIFO 策略管理记忆
    - 无需参数更新，通过 prompt 级记忆实现行为调整

4. **熵调制节点评估 (EMCS)**：

    - 智能体评估节点产生价值 $Z(s_t) \in [0,1]$ 和置信度 $C(s_t) \in (0,1)$
    - 用伯努利熵惩罚不确定预测：$R(s_t) = Z(s_t) \cdot (1 - E(s_t))$
    - 其中 $E(s_t) = -C(s_t)\ln C(s_t) - (1-C(s_t))\ln(1-C(s_t))$
    - 在 $C=0.5$ 时惩罚最大（最不确定），保留高置信度评估

### 损失函数 / 训练策略
- 无需训练，纯推理时协作框架
- 超参数包括 UCB 探索系数 $\alpha$、MCTS 展开宽度 $n$ 和搜索预算 $K$

## 实验关键数据

### 主实验

| 任务 | 指标 | SYMPHONY-S | SYMPHONY-L | LATS | MASTER |
|------|------|-----------|-----------|------|--------|
| HotpotQA | EM | 0.59 | **0.79** | 0.71 | 0.76 |
| WebShop | Score/SR | 0.82/0.56 | **0.88/0.72** | 0.76/0.38 | 0.80/– |
| MBPP (Python) | Pass@1 | 0.927 | **0.965** | 0.811 | 0.910 |
| MBPP (Rust) | Pass@1 | 0.946 | **0.974** | – | – |

### 消融实验

| 配置 | HotpotQA(EM) | WebShop(SR) | MBPP(Pass@1) |
|------|-------------|-------------|-------------|
| SYMPHONY-S 完整 | 0.59 | 0.56 | 0.927 |
| w/o Agent Scheduling | 0.51 | 0.48 | 0.906 |
| w/o Memory Sharing | 0.45 | 0.46 | 0.871 |
| w/o EMCS | 0.51 | 0.49 | 0.892 |

### 关键发现
- **多样性驱动性能**：三模型组合的 4-Unique 分支比例在 MBPP 上超过 80%，单模型不到 20%，性能提升超 30%
- **搜索效率**：SYMPHONY-L 在 HotpotQA 上仅需 9.47 个节点展开（LATS@50 需要 66.65），效率提升 7×
- **成本优化**：SYMPHONY-L 中 GPT-4 仅占 40% 调用量，但性能优于纯 GPT-4 基线
- **消费级可行**：SYMPHONY-S（纯开源 7B/8B 模型）在多个任务上已超越单模型 GPT-4 基线

## 亮点与洞察
- **异构 > 同构**：即使是同规模的不同开源模型组合，也比单一强模型的多次采样更有效——模型级多样性是被低估的资源
- **记忆共享的去中心化设计**：不同于需要显式通信协议的多智能体系统，通过自然语言反思实现轻量级知识传播
- **EMCS 的优雅性**：用信息论中的伯努利熵来校准 LLM 的自评分，简单且有效

## 局限与展望
- 异构模型池的选择缺乏系统性指导，目前靠经验选择
- MCTS 的搜索预算 $K$ 和展开宽度 $n$ 需手动调节
- 记忆共享的 FIFO 策略较简单，可能丢失重要反思
- 对抗性或高噪声环境下的鲁棒性未充分验证

## 相关工作与启发
- **vs MASTER (Gan et al.)**: MASTER 用同一 LLM 构建多智能体并改 UCT 公式，SYMPHONY 用异构模型池从根本上增加多样性
- **vs LATS (Zhou et al.)**: LATS 单模型搜索，SYMPHONY 多模型 + 记忆共享，在更小预算下达到更好效果
- **vs AgentCoder (Huang et al.)**: AgentCoder 固定角色分工（程序员/测试员），SYMPHONY 自适应调度更灵活

## 评分
- 新颖性: ⭐⭐⭐⭐ 异构智能体池 + UCB调度的组合在 MCTS-LLM 场景中是新的
- 实验充分度: ⭐⭐⭐⭐⭐ 三个不同类型任务，两种配置，消融和效率分析全面
- 写作质量: ⭐⭐⭐⭐ 框架清晰，但数学符号较多
- 价值: ⭐⭐⭐⭐⭐ 为"如何组合多个LLM做规划"提供了有原则的方案，消费级硬件可用性强

<!-- RELATED:START -->

## 相关论文

- [Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [Planning-Driven Programming: A Large Language Model Programming Workflow](../../ACL2025/llm_nlp/planning-driven_programming_a_large_language_model_programming_workflow.md)
- [Red-Teaming LLM Multi-Agent Systems via Communication Attacks](../../ACL2025/llm_nlp/red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)
- [StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)

<!-- RELATED:END -->
