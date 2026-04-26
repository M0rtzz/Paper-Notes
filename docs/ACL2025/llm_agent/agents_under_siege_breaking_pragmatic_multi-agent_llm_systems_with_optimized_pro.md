---
title: >-
  [论文解读] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks
description: >-
  [ACL 2025][LLM Agent][多智能体安全] 本文首次系统研究了在带宽约束、延迟和安全机制的现实多智能体LLM系统中的对抗攻击问题，提出基于最大流最小费用的拓扑优化和排列不变蒙骗损失（PIEL）的攻击方法，在多个LLM架构上实现了高达7倍于传统攻击的成功率。
tags:
  - ACL 2025
  - LLM Agent
  - 多智能体安全
  - 越狱攻击
  - 排列不变性
  - 最大流最小费用
  - 对抗提示传播
---

# Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks

**会议**: ACL 2025  
**arXiv**: [2504.00218](https://arxiv.org/abs/2504.00218)  
**代码**: 无  
**领域**: LLM Agent / LLM安全  
**关键词**: 多智能体安全、越狱攻击、排列不变性、最大流最小费用、对抗提示传播

## 一句话总结
本文首次系统研究了在带宽约束、延迟和安全机制的现实多智能体LLM系统中的对抗攻击问题，提出基于最大流最小费用的拓扑优化和排列不变蒙骗损失（PIEL）的攻击方法，在多个LLM架构上实现了高达7倍于传统攻击的成功率。

## 研究背景与动机

**领域现状**：多Agent LLM系统通过分布式推理和集体智慧增强任务性能，被越来越多地应用于自动化系统和AI治理等场景。关于LLM安全的研究主要集中在单Agent设置下的越狱攻击和防御。

**现有痛点**：多Agent系统引入了全新的安全风险维度——Agent间通信可以被利用为攻击向量。现有的多Agent攻击研究（如Evil Geniuses、Prompt Infection）假设攻击者可以不受限制地向Agent发送消息，忽略了实际系统中的通信约束：token带宽限制、消息延迟和安全过滤机制。

**核心矛盾**：在受限的多Agent系统中，攻击者面临三重约束：（1）每条通信边有token数量限制，完整的对抗提示无法通过单条边传输；（2）不同路径的消息到达顺序不确定；（3）部分通信边上部署了安全检测机制。如何在这些约束下成功实施攻击是一个未被研究的开放问题。

**本文目标**：在带有token带宽限制、异步消息到达和安全机制的现实多Agent系统中，设计最优的对抗提示传播策略。

**切入角度**：将攻击路径优化建模为图论中的最大流最小费用问题，同时设计排列不变的对抗损失确保攻击在任意chunk顺序下都有效。

**核心 idea**：将对抗提示分成多个chunk，通过最大流最小费用算法找到最优传播路径（最大化token流量、最小化检测风险），并用排列不变蒙骗损失优化chunk内容使其在任意排列下都能触发越狱。

## 方法详解

### 整体框架
方法分为两个解耦的模块：（1）Topological Optimization——给定多Agent系统的通信拓扑、带宽约束和安全机制部署，求解最优的攻击路径和chunk分配；（2）Permutation-Invariant Evasion Loss——在确定的chunk方案下，优化每个chunk的token内容使攻击在所有可能的到达顺序下都有效。

### 关键设计

1. **最大流最小费用拓扑优化**:

    - 功能：在有约束的通信网络中找到最优的对抗提示传播路径
    - 核心思路：定义流函数 $f: \mathcal{E} \to \mathbb{R}_{\geq 0}$ 表示每条边传输的对抗token数量。优化目标是最小化总风险 $\min \sum_{(u,v)} G(u,v) \cdot f(u,v)$，约束为token容量 $0 \leq f(u,v) \leq F(u,v)$、流守恒和源汇约束。使用NetworkX实现的标准MFMC算法求解，输出每条边应传输的chunk长度
    - 设计动机：直觉上，攻击者需要将一个完整的对抗提示"分装"到多条路径上传送到目标Agent，同时要避开有安全检测的"危险"路径。这本质上就是网络流问题

2. **排列不变蒙骗损失（PIEL）**:

    - 功能：确保分块的对抗提示在任意到达顺序下都能触发越狱
    - 核心思路：将对抗提示分为 $K$ 个chunk $\mathcal{C} = \{C_1, ..., C_K\}$，损失函数定义为所有可能排列的平均负对数似然：$\mathcal{L}(\mathcal{C}) = \frac{1}{K!} \sum_{\pi \in S_K} -\log p(x^*_{n+1:n+L} | \phi)$，其中 $\phi$ 是按排列 $\pi$ 拼接chunk后的输入，$x^*$ 是目标有害输出。使用GCG（Greedy Coordinate Gradient）方法迭代优化每个chunk中的token
    - 设计动机：由于网络延迟，不同路径传输的chunk到达目标Agent的顺序是不确定的。如果对抗提示只在特定顺序下有效，攻击会因到达顺序随机性而频繁失败

3. **随机PIEL（S-PIEL）**:

    - 功能：降低PIEL的计算复杂度
    - 核心思路：当chunk数 $K$ 较大时，$K!$ 个排列的遍历不可行。S-PIEL随机采样 $M$ 个排列近似完整期望：$\tilde{\mathcal{L}}(\mathcal{C}) = \frac{1}{|\tilde{S}_K|} \sum_{\pi \in \tilde{S}_K} -\log p(x^*|  \phi)$
    - 设计动机：$K=5$ 时有120种排列，遍历计算成本已经很高。实验显示采样约50%的排列即可达到良好效果

### 损失函数 / 训练策略
使用GCG方法优化token选择：对每个token计算PIEL梯度，选择梯度最大的位置进行替换。每轮迭代中计算所有排列的损失、聚合梯度、执行token替换。在500步GCG迭代中优化，搜索宽度64。

## 实验关键数据

### 主实验

| 目标模型 | 方法 | JailbreakBench ASR | AdversarialBench ASR | In-the-wild ASR |
|---------|------|-------------------|---------------------|-----------------|
| Llama-2-7B | Vanilla | 0.000 | 0.000 | 0.144 |
| Llama-2-7B | GCG | 0.017 | 0.160 | 0.201 |
| Llama-2-7B | **Ours** | **0.726** | **0.533** | **0.561** |
| Mistral-7B | Vanilla | 0.000 | 0.000 | 0.215 |
| Mistral-7B | GCG | 0.324 | 0.212 | 0.203 |
| Mistral-7B | **Ours** | **0.812** | **0.543** | **0.627** |

### 消融实验

| 因素 | ASR变化 | 说明 |
|------|---------|------|
| Chain拓扑 | ~60% | 连通性最差，攻击最难 |
| Complete拓扑 | ~78% | 全连接最脆弱 |
| S-PIEL M=8 | 0% (不收敛) | 采样太少无法逼近 |
| S-PIEL M=64 | 56% | 约50% K!时效果良好 |
| PIEL (M=K!) | 72.6% | 完整排列最优 |

### 关键发现
- 本文方法在Llama-2-7B上将ASR从GCG的1.7%提升到72.6%（JailbreakBench），提升了约43倍
- 现有安全机制（Llama-Guard各版本、PromptGuard）对本文攻击的检测F1-score下降约30%，说明分块传输的对抗提示更难检测
- 攻击具有良好的跨模型迁移性——在Llama-2上优化的攻击对Mistral和Gemma也有效（ASR 61-71%）
- 拓扑结构影响显著：Complete Graph最脆弱（78% ASR），Chain最安全（60% ASR）

## 亮点与洞察
- 将多Agent攻击建模为最大流最小费用是非常巧妙的形式化，完美捕捉了带宽约束和安全检测风险的权衡。这个建模方式也可以反向用于防御——设计拓扑使最大流最小
- PIEL损失的排列不变性设计应对了异步通信的核心挑战，这个思想可以迁移到其他需要排列不变的场景（如分布式推理）
- 本文揭示了多Agent系统"增加连通性反而增加脆弱性"的反直觉现象，对系统安全设计有重要启示

## 局限与展望
- 仅在开源模型上评估，未测试GPT-4等商业模型
- 假设攻击者知道网络拓扑和安全机制位置，部分知识假设可能过强
- 静态安全机制和固定带宽的假设不适用于动态自适应的防御系统
- 仅考虑文本交互，多模态Agent系统的攻击面更广

## 相关工作与启发
- **vs Evil Geniuses**: Evil Geniuses研究基于角色的对抗，本文关注通信拓扑层面的攻击，更系统化
- **vs Agent Smith**: Agent Smith研究指数级传播但假设通信无约束，本文实际考虑了带宽和延迟约束
- **vs Prompt Infection**: Prompt Infection研究自复制提示注入，本文研究优化的提示分发

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统化研究受限多Agent系统的攻击，MFMC建模和PIEL都很创新
- 实验充分度: ⭐⭐⭐⭐⭐ 五种模型、三个基准、多种拓扑的全面评估
- 写作质量: ⭐⭐⭐⭐ 技术描述详尽但篇幅较长
- 价值: ⭐⭐⭐⭐ 对多Agent安全防御有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/llm_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] iAgent: LLM Agent as a Shield between User and Recommender Systems](iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)
- [\[ACL 2025\] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](multi_agent_dialect_bias_privacy_qa.md)
- [\[ICML 2025\] TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](../../ICML2025/llm_agent/tamas_benchmarking_adversarial_risks_in_multi-agent_llm_systems.md)

<!-- RELATED:END -->
