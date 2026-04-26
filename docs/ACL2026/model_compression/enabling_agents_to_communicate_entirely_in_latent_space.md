---
title: >-
  [论文解读] Enabling Agents to Communicate Entirely in Latent Space
description: >-
  [ACL 2026][模型压缩][潜空间通信] 本文提出 Interlat，一个让 LLM 智能体完全在潜空间中通信的框架——发送方直接传递最后一层隐状态作为"思维"的表示，接收方通过通信适配器解释这些潜空间消息，并通过潜空间推理进一步压缩到仅 8 个 token 同时保持竞争性能，实现高达 24× 的通信加速。
tags:
  - ACL 2026
  - 模型压缩
  - 潜空间通信
  - 多智能体
  - 隐状态传递
  - 信息压缩
  - 推理加速
---

# Enabling Agents to Communicate Entirely in Latent Space

**会议**: ACL 2026  
**arXiv**: [2511.09149](https://arxiv.org/abs/2511.09149)  
**代码**: [GitHub](https://github.com/XiaoDu-flying/Interlat)  
**领域**: 多智能体 / 潜空间通信  
**关键词**: 潜空间通信, 多智能体, 隐状态传递, 信息压缩, 推理加速

## 一句话总结

本文提出 Interlat，一个让 LLM 智能体完全在潜空间中通信的框架——发送方直接传递最后一层隐状态作为"思维"的表示，接收方通过通信适配器解释这些潜空间消息，并通过潜空间推理进一步压缩到仅 8 个 token 同时保持竞争性能，实现高达 24× 的通信加速。

## 研究背景与动机

**领域现状**：基于 LLM 的多智能体系统通过自然语言通信协调任务。尽管自然语言具有人类可读性，但它是一种有损的通信媒介——将高维内部状态下采样为离散 token 丢失了大量信息。

**现有痛点**：(1) 自然语言通信的信息带宽有限（约 15 bits/token vs 隐状态约 40k bits/hidden-state），大量推理路径和细微信息在 token 化过程中被丢弃；(2) 生成的大量文本用于语言连贯性而非任务相关信息，造成冗余；(3) 语言通信固有的歧义性是多智能体协调失败的主要来源；(4) 现有隐状态通信方法依赖单次激活嫁接或与语言轨迹耦合，需要特定层选择。

**核心矛盾**：LLM 的大部分计算在连续的潜空间中进行，内部隐状态包含极其丰富的信息——但通信时必须将其压缩为离散 token，造成巨大的信息损失。

**本文目标**：让智能体之间的通信完全在潜空间中进行——直接传递连续隐状态而非离散 token，并通过压缩实现高效通信。

**切入角度**：类比"心灵感应"——绕过符号语言直接传递内部表示。利用 LLM 生成过程中产生的最后一层隐状态序列作为"思维"的连续表示进行传输。

**核心 idea**：用时间对齐的最后一层隐状态序列作为潜空间通信消息，通过条件思维分离损失确保接收方利用而非忽略潜空间信息，再通过潜空间推理模型将长序列压缩为极短的潜空间消息。

## 方法详解

### 整体框架

Sender-Receiver 二智能体设置：推理智能体（Sender）生成计划及其隐状态 $H \in \mathbb{R}^{L \times d}$ → 通信适配器（轻量级自注意力+投影层）处理隐状态 → 执行智能体（Receiver）接收隐状态并生成动作。训练后可进一步训练压缩模型将 $H_L$ 压缩为 $H_K$（$K \ll L$）。

### 关键设计

1. **潜空间通信与条件思维分离**:

    - 功能：确保接收方真正利用潜空间消息中的任务相关信息
    - 核心思路：传输发送方生成过程中时间对齐的最后一层隐状态序列 $H = [h_1, ..., h_L]$，用特殊 token `<bop>` 和 `<eop>` 标记通信边界。训练时最小化条件思维分离损失：比较匹配潜空间 $H$ 和不匹配潜空间 $\tilde{H}$（来自不同任务）下接收方输出分布的 Jensen-Shannon 散度，迫使模型区分两者
    - 设计动机：简单的 SFT 可能让模型忽略潜空间信息而仅依赖提示——条件分离损失显式鼓励模型利用潜空间中的任务特定信息

2. **计划对齐正则化**:

    - 功能：防止条件分离训练中的退化模式
    - 核心思路：最大化分离可能导致模型将概率质量移向增加散度但损害任务效用的怪异 token。用对应的语言空间计划 $P$ 的输出分布来正则化——KL 散度约束潜空间条件下的输出与语言计划条件下的输出一致，加上 logit 余弦相似度对齐
    - 设计动机：确保潜空间通信的效果不弱于语言通信——它应该传达相同或更多的信息

3. **潜空间推理压缩**:

    - 功能：将长潜空间消息压缩为极短序列
    - 核心思路：训练独立的推理模型 $M_\phi$ 在潜空间中自回归生成紧凑消息 $H_K$（$K \ll L$），通过将自身的隐状态反馈为下一个输入嵌入。训练时冻结接收方，优化三部分损失：任务损失（保持下游性能）+ 不确定性加权一致性损失（在潜空间有信息增益的位置对齐压缩和完整消息的分布）+ 潜空间几何对齐损失（保持全局语义方向）
    - 设计动机：完整隐状态序列可能有数百步，造成通信延迟。自回归潜空间推理可以将信息"蒸馏"到少数步骤中

### 损失函数 / 训练策略

主训练：$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_S \mathcal{L}_{sep} + \lambda_A \mathcal{L}_{align}$，使用随机 token-latent 混合课程学习稳定训练。压缩训练：$\mathcal{L}_{compress} = \lambda_{task}\mathcal{L}_{task} + \lambda_{pref}\mathcal{L}_{pref} + \lambda_{geom}\mathcal{L}_{geom}$，冻结接收方仅更新压缩模型。

## 实验关键数据

### 主实验

**Qwen2.5-7B 在 Seen/Unseen 任务上的成功率**

| 方法 | Seen 成功率 | Unseen 成功率 |
|------|-----------|-------------|
| No-Comm（无通信） | 62.14 | 62.19 |
| Text（语言通信 + SFT） | 64.29 | 62.44 |
| CoT (full) | 67.14 | - |
| **Interlat（潜空间通信）** | **70.48** | **65.42** |

### 消融实验

**通信压缩（Qwen2.5-7B，Seen 任务）**

| 压缩 token 数 K | 成功率 | 加速比 |
|----------------|--------|-------|
| 完整 L | 70.48 | 1× |
| 64 | ~70 | ~4× |
| 32 | ~69 | ~8× |
| 16 | ~68 | ~16× |
| **8** | **~66** | **24×** |

**跨模型异构通信**

| Sender → Receiver | 潜空间通信 | 语言通信 |
|-------------------|----------|---------|
| Qwen-7B → Qwen-0.5B | 61.19 | 54.52 |
| LLaMA-8B → LLaMA-8B | 70.71 | 62.86 |

### 关键发现

- 潜空间通信（70.48%）显著优于语言通信（64.29%）和无通信（62.14%）——隐状态确实携带了语言无法表达的有用信息
- 即使跨异构模型（不同架构/大小）也有效——说明最后一层隐状态的信息结构具有一定的跨模型通用性
- 压缩到 8 个 token 时性能仅损失约 4%（~66% vs 70.48%），但通信速度提升 24×
- 分析发现使用潜空间通信的智能体展现出更多探索性行为——它们利用的是潜空间中的任务相关信息而非表面模式匹配
- 条件分离损失是关键——没有它模型倾向于忽略潜空间输入

## 亮点与洞察

- "心灵感应"的类比虽然浮夸但确实抓住了核心——LLM 之间通信不需要经过人类可读的中间表示
- 潜空间推理压缩是一种新颖的"信息蒸馏"方式——在连续空间中做自回归推理而不解码为 token
- 24× 的通信加速对多智能体系统的实际部署有重要意义

## 局限与展望

- 仅在 Sender-Receiver 二智能体场景中验证，未扩展到更复杂的多智能体拓扑
- 通信适配器需要训练，增加了部署复杂性
- 潜空间通信失去了人类可解释性——难以调试和审计智能体间的"对话"
- 未探索潜空间通信在安全性方面的影响

## 相关工作与启发

- **vs COCONUT/Thought-of-Thought**: 这些工作在单模型内做潜空间推理，Interlat 将其扩展到多智能体间通信
- **vs Ramesh & Li (2025)**: 他们用单次激活嫁接，Interlat 传递完整的时间对齐隐状态序列
- **vs Tang et al. (2025)**: 他们的潜空间通信与语言轨迹耦合，Interlat 完全在潜空间中

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 完全潜空间通信+潜空间推理压缩是全新范式
- 实验充分度: ⭐⭐⭐⭐ 多模型多任务评估，但场景限于二智能体
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，数学公式完整
- 价值: ⭐⭐⭐⭐⭐ 为多智能体系统的高效通信开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)

<!-- RELATED:END -->
