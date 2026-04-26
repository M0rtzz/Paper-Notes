---
title: >-
  [论文解读] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents
description: >-
  [ACL 2026][模型压缩][信息引出] 提出信息引出代理（IEA）作为新的对话范式，发布了首个大规模（2,281 段对话，26M token）人与人信息引出对话数据集 YIELD，将信息引出形式化为有限视野 POMDP，并设计了专门的评估指标（Conformity、Progression、TLR），实验表明在 YIELD 上微调能显著提升 LLM 与真实引出行为的对齐。
tags:
  - ACL 2026
  - 模型压缩
  - 信息引出
  - 对话数据集
  - 强化学习
  - 会话代理
  - POMDP
---

# YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents

**会议**: ACL 2026  
**arXiv**: [2604.10968](https://arxiv.org/abs/2604.10968)  
**代码**: https://github.com/infosenselab/yield (有)  
**领域**: 对话系统  
**关键词**: 信息引出, 对话数据集, 强化学习, 会话代理, POMDP

## 一句话总结

提出信息引出代理（IEA）作为新的对话范式，发布了首个大规模（2,281 段对话，26M token）人与人信息引出对话数据集 YIELD，将信息引出形式化为有限视野 POMDP，并设计了专门的评估指标（Conformity、Progression、TLR），实验表明在 YIELD 上微调能显著提升 LLM 与真实引出行为的对齐。

## 研究背景与动机

**领域现状**：大多数对话代理（CA）设计用于满足用户需求的用户驱动交互——用户控制议程和方向，代理提供帮助。常见数据集如 MultiWOZ、SGD 都面向这一范式。

**现有痛点**：许多现实场景（学术访谈、司法程序、新闻调查）需要代理主动从用户中引出信息，以支持代理方的机构或任务目标。这种"信息引出"与传统 CA 存在根本差异：(1) 对话主导权不同——代理需要主动提问引导方向；(2) 目标不同——成功的定义是获取了多少有价值的信息而非解决了用户的问题；(3) 没有单一最优问题，而是多种可能方向，每条都可能产生有价值的信息。

**核心矛盾**：尽管对 IEA 的需求明确，但缺乏专门的数据集、形式化框架和评估指标来支持研究。现有对话数据集对话轮次短（DSTC2 平均 14.49 轮），无法捕捉长程引出策略。

**本文目标**：(1) 定义 IEA 作为新的对话范式；(2) 构建首个大规模 IEA 数据集；(3) 形式化信息引出问题；(4) 设计专用评估指标。

**切入角度**：从真实的人与人引出对话（口述历史、司法听证、学术访谈、新闻调查）中收集数据，确保数据反映真实的引出行为模式。

**核心 idea**：将信息引出建模为 POMDP，利用实体新颖性作为代理奖励信号，通过离线强化学习（AWR）微调 LLM 使其学会像人类引出者一样提问。

## 方法详解

### 整体框架

YIELD 包含四个核心组件：(1) 大规模数据集——2,281 段跨 4 个领域的人与人引出对话；(2) POMDP 形式化——将引出过程建模为代理在部分可观测环境中的序贯决策；(3) 离线强化学习训练——使用 AWR 和实体新颖性奖励；(4) 专用评估指标——Conformity、Progression、Turn-Length Ratio。

### 关键设计

1. **POMDP 形式化与状态表示**:

    - 功能：将信息引出过程形式化为可优化的序贯决策问题
    - 核心思路：受访者拥有不可观测的隐藏信息（状态 $X_t$），引出者采取行动 $A_t$（自然语言提问），环境返回观测 $O_{t+1}$（受访者回答）。由于隐藏状态空间无界，使用因果语言模型的隐层表示 $S_t = f_\theta(H_t^s)$ 替代传统的信念状态。奖励定义为受访者回答中新出现的命名实体数量：$R_{t+1} = |\mathcal{E}_{t+1} \setminus \mathcal{E}_{\leq t}|$
    - 设计动机：POMDP 框架自然契合引出场景——引出者永远无法完全观测受访者的知识状态，只能通过提问间接获取信息。实体新颖性奖励虽然简单，但有效量化了每次提问带来的信息增量

2. **离线强化学习训练（AWR）**:

    - 功能：使 LLM 学会优先生成能引出更多新信息的提问
    - 核心思路：使用 Advantage-Weighted Regression，训练线性价值头估计状态价值 $v_\psi(S_t)$，计算优势函数来调整每个训练样本的权重。高优势（引出更多新信息）的引出者发言获得更高的训练权重。使用 LoRA 进行参数高效微调，联合优化策略损失和价值损失：$\mathcal{L}(\theta, \psi) = \mathcal{L}_\pi(\theta) + \mathcal{L}_v(\psi)$
    - 设计动机：与标准 SFT 不同，AWR 考虑了每次发言对整个后续对话的影响，使模型学习长程引出策略而非逐轮模仿

3. **专用评估指标体系**:

    - 功能：从多个维度评估 IEA 的引出能力
    - 核心思路：(1) **Conformity**——通过困惑度和回复长度衡量模型输出是否符合真实引出者的分布模式；(2) **Progression**——衡量对话是否持续向前推进而非停滞在同一话题，使用衰减余弦距离窗口计算；(3) **Turn-Length Ratio**——受访者平均回复长度与引出者的比值，有效引出者应言简意赅地提问以引出长回复
    - 设计动机：传统指标（BLEU、任务成功率）无法捕捉引出对话的本质——对话的前进动量、引出者的提问效率以及是否符合真实引出行为的风格

### 损失函数 / 训练策略

AWR 加权策略损失：$\mathcal{L}_\pi(\theta) = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \bar{w}_i \log \pi_\theta(A_i | S_i)$，其中权重 $\bar{w}_i$ 根据优势函数缩放。使用滑动窗口（6 轮）分段数据，折扣因子 $\gamma=0.9$，温度 $\alpha=0.25$。在 3 张 A6000 GPU 上训练。

## 实验关键数据

### 主实验

Conformity 评估（困惑度和回复长度对比）：

| 模型 | 学术困惑度 | 司法困惑度 | 学术回复长度 | 真实回复长度 |
|------|-----------|-----------|-------------|-------------|
| Llama-3.1-8B Prompt | 46.9 | 22.6 | 39.5 tokens | 16.9 tokens |
| Llama-3.1-8B SFT | 10.9 | 10.9 | 11.2 tokens | 16.9 tokens |
| Llama-3.1-8B ORL | 12.5 | 11.3 | 11.6 tokens | 16.9 tokens |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| Prompt-only vs SFT/ORL | 困惑度下降 3-4 倍 | 微调显著提升与真实引出行为的对齐 |
| SFT vs ORL | SFT 困惑度略低 | ORL 牺牲逐 token 似然以优化长程策略 |
| DeepSeek-R1 Prompt | 回复长度 414-472 tokens | 推理模型的冗长元思考严重偏离引出风格 |
| 3B vs 8B 模型 | 性能接近 | 说明 YIELD 数据而非模型规模是关键 |

### 关键发现

- 提示方法（Prompt-only）产生的引出者发言远长于真实引出者（39-53 vs 17-39 tokens），且提示本身也很长（540-648 tokens），极不经济
- DeepSeek-R1 的推理模式在引出任务上完全不适用——生成超长的元推理前缀，微调后才恢复正常
- ORL 训练的模型在 Progression 指标上与 SFT 竞争，且回复长度分布更接近真实引出者
- 人工评估证实了自动指标的发现，经过 YIELD 训练的模型在引出质量上显著优于提示方法

## 亮点与洞察

- **IEA 概念的提出**具有开创性——明确定义了"代理主动引出信息"这一对话范式，与传统"用户提问代理回答"形成鲜明对比，这个框架可以统一学术访谈、司法听证、新闻调查等多种场景
- **实体新颖性奖励**简洁有效——用 NER 提取新实体数作为引出成功的代理指标，避免了定义"信息价值"的主观性问题，同时通过多种约束防止奖励作弊
- **数据集本身的构建方法论**值得学习——从多种公域/CC 授权的真实人与人对话中手工标注，平均 171 轮/对话远超现有数据集（13-20 轮）

## 局限与展望

- 实验仅在离线设置下评估，未与真实用户交互测试，IEA 的实际引出效果未知
- 实体新颖性奖励过于简单，无法衡量信息的"深度"或"相关性"，更精细的奖励设计是关键方向
- 数据集全为英文，且某些领域数据量较小（新闻调查仅 129 段对话）
- IEA 的伦理边界需要更多讨论——在何种程度上代理的引出行为是合适的

## 相关工作与启发

- **vs MultiWOZ/SGD**: 这些数据集面向用户驱动的任务完成对话，平均 13-20 轮/对话。YIELD 面向代理驱动的信息引出，平均 171 轮/对话，规模和范式都完全不同
- **vs LLM-as-Interviewer**: 现有工作多聚焦于 LLM 模拟面试官，但缺乏系统的数据集和形式化框架。YIELD 提供了从数据到理论到评估的完整研究基础设施

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 定义了信息引出代理这一新范式，数据集、形式化和评估指标都是首创
- 实验充分度: ⭐⭐⭐⭐ 多模型对比和人工评估充分，但缺乏在线交互实验
- 写作质量: ⭐⭐⭐⭐⭐ 概念定义清晰，论文结构严谨，从动机到方法到评估的逻辑链完整
- 价值: ⭐⭐⭐⭐ 开创性数据集和框架对社区有长远价值，但应用范围相对专业

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)
- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[ICLR 2026\] Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](../../ICLR2026/model_compression/rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)

<!-- RELATED:END -->
