---
title: >-
  [论文解读] Ad-Hoc Human-AI Coordination Challenge (AH2AC2)
description: >-
  [ICML2025][LLM推理][Human-AI Coordination] 提出 AH2AC2 挑战——基于 Hanabi 合作卡牌游戏，通过行为克隆+正则化强化学习构建人类代理智能体，并开源有限人类数据集，为 Human-AI 临时协作研究提供标准化、可复现的评估框架。
tags:
  - ICML2025
  - LLM推理
  - Human-AI Coordination
  - Ad-Hoc Teamplay
  - Hanabi
  - Behavioural Cloning
  - Regularised RL
  - Human Proxy Agents
  - benchmark
---

# Ad-Hoc Human-AI Coordination Challenge (AH2AC2)

**会议**: ICML2025  
**arXiv**: [2506.21490](https://arxiv.org/abs/2506.21490)  
**代码**: [FLAIROx/ah2ac2](https://github.com/FLAIROx/ah2ac2)  
**领域**: llm_reasoning / multi-agent RL / human-AI coordination  
**关键词**: Human-AI Coordination, Ad-Hoc Teamplay, Hanabi, Behavioural Cloning, Regularised RL, Human Proxy Agents, benchmark

## 一句话总结

提出 AH2AC2 挑战——基于 Hanabi 合作卡牌游戏，通过行为克隆+正则化强化学习构建人类代理智能体，并开源有限人类数据集，为 Human-AI 临时协作研究提供标准化、可复现的评估框架。

## 研究背景与动机

### 核心问题
AI 智能体在现实世界中需要与人类临时协作（ad-hoc teamplay），但现有方法面临两大困局：

**自博弈过拟合**：传统 self-play 训练的智能体形成专属通信协议，无法与未见过的搭档（尤其是人类）配合

**评估不可复现**：此前的 Human-AI 协调评估依赖闭源数据集和私有代理智能体，研究结果难以复现和对比

### 为什么选择 Hanabi？
Hanabi 是一个理想的测试平台，因为它同时具备：

- **不完全信息**：玩家看不到自己的手牌
- **受限通信**：只能通过有限的提示传递信息
- **心智理论（Theory of Mind）**：需要推理搭档的意图和知识
- **协作行动**：所有玩家共享同一目标（最高得分 25 分）

### 研究动机
现有文献中缺乏标准化的 Human-AI 协调评估方法。各研究使用不同的代理智能体和数据集，导致方法之间无法公平比较。本文旨在填补这一空白，提供统一的评估协议和公开的基准线。

## 方法详解

### 整体框架：AH2AC2 挑战设计

AH2AC2 包含两部分评估任务：

1. **Part 1 — 与人类代理协调**：参与者开发智能体，与论文提供的人类代理智能体（Human Proxy）进行 1000 局 Hanabi 对局
2. **Part 2 — 人类行动预测**：在闭源人类对局数据上预测人类玩家的行动（使用交叉熵损失评估）

数据来源为 hanab.live 平台：共收集 101,096 局双人对局和 46,525 局三人对局。开源其中 3,079 局（1,858 双人 + 1,221 三人）供参赛者使用，刻意限制数据量以鼓励数据高效方法的研究。

### 关键设计：HDR-IPPO（Human-Data-Regularised IPPO）

人类代理智能体的训练分两步：

**第一步：行为克隆（BC）**

- 使用 LSTM 架构建模策略 $\pi_\theta^{BC}$，输入为局部动作-观察历史（AOH）
- 通过标准交叉熵损失做监督学习
- 每个训练 epoch 结束后评估自博弈得分，保存最优参数 $\theta'$

**第二步：正则化强化学习（Regularised RL）**

- 从 BC 最优参数 $\theta'$ 初始化人类代理策略 $\pi_\theta^{HP}$
- 使用 IPPO（Independent PPO）进行自博弈训练
- 添加 KL 正则化项，约束最终策略不偏离人类行为：

$$\mathcal{L}_t^{\text{HDR-IPPO}}(\theta) = (1 - \lambda) \cdot \mathcal{L}_t^{\text{IPPO}}(\theta) + \lambda \cdot D_{\text{KL}}(\pi_{\theta'}^{BC} \| \pi_\theta^{HP})$$

其中 $\lambda \in [0, 1]$ 控制正则化强度。这使得最终策略在提升博弈能力的同时保持人类风格。

### 训练策略

- 使用 LSTM 架构捕捉动作-观察的序列特征
- 虽然模型参数固定，但代理的行为是动态的——它们依据完整博弈历史（包括搭档行为）进行决策
- 共训练 4 个人类代理（2 个双人设定 + 2 个三人设定），通过不同超参数和随机种子确保多样性
- 评估 API 隐藏代理权重以防过拟合，参赛需预注册实验

### 评估协议的防过拟合设计

- 人类代理不公开发布，仅通过 API 交互
- 每次评估限制 1000 局
- 需要预注册实验才能获取 API 访问权限
- 评估结果公布在排行榜上

## 实验关键数据

### 开源数据集统计

| 设定 | 指标 | 最小值 | 最大值 | 平均值 | 中位数 | 标准差 |
|------|------|--------|--------|--------|--------|--------|
| 双人 (1858局) | 得分 | 13 | 25 | 23.37 | 24 | 1.86 |
| 双人 (1858局) | 对局长度 | 52 | 76 | 65.45 | 66 | 3.35 |
| 三人 (1221局) | 得分 | 14 | 25 | 23.25 | 24 | 1.91 |
| 三人 (1221局) | 对局长度 | 45 | 67 | 57.86 | 58 | 3.38 |

### 人类代理自博弈性能（5000局评估）

| 指标 | HP1 (2P) | HP2 (2P) | HP3 (3P) | HP4 (3P) |
|------|----------|----------|----------|----------|
| 平均自博弈得分 | 22.55±0.03 | 22.97±0.03 | 20.88±0.03 | 21.21±0.03 |
| 相对BC提升 | +3.0 | +4.0 | +15.7 | +13.9 |
| 满分比例 | 23.86% | 29.66% | 2.76% | 3.88% |
| BC满分比例 | 16.12% | 19.88% | 1.34% | 1.80% |
| 零分比例 | 0.10% | 0.04% | 0.34% | 0.20% |
| BC零分比例 | 11.42% | 17.70% | 75.82% | 66.02% |

三人设定下提升尤为显著：BC 策略零分比例高达 66-76%，而 HDR-IPPO 将其降至不足 0.5%。

### AH2AC2 排行榜主结果

| 玩家数 | 方法 | Mean | Median | 交叉熵 |
|--------|------|------|--------|--------|
| 2P | OBL (L4) | **21.04** | **22** | 1.33 |
| 2P | BR-BC | 19.41 | 20 | 10.82 |
| 2P | FCP | 14.01 | 16 | 3.52 |
| 2P | OP | 13.91 | 19 | 7.81 |
| 2P | HDR-IPPO | 12.76 | 15 | 0.96 |
| 2P | IPPO | 10.16 | 14 | 12.60 |
| 2P | DeepSeek-R1 (H-Group) | 9.91 | 0 | - |
| 2P | DeepSeek-R1 | 5.43 | 0 | - |
| 2P | BC | 2.12 | 0 | 0.86 |
| 3P | DeepSeek-R1 (H-Group) | **14.62** | **18** | - |
| 3P | HDR-IPPO | 14.03 | 16 | 0.80 |
| 3P | OP | 12.87 | 18 | 6.40 |
| 3P | BR-BC | 11.89 | 12 | 29.89 |
| 3P | FCP | 11.55 | 6 | 5.97 |
| 3P | IPPO | 6.34 | 0 | 8.60 |
| 3P | BC | 3.31 | 0 | 0.70 |

### 消融：人类代理行为分析

| 设定 | 来源 | IPP (信息/打出牌) | Communicativeness (提示倾向) |
|------|------|-------------------|------------------------------|
| 2P | 人类数据集 | 0.44 | 0.47 |
| 2P | HP1 | 0.43 | 0.45 |
| 2P | HP2 | 0.44 | 0.48 |
| 3P | 人类数据集 | 0.42 | 0.49 |
| 3P | HP3 | 0.44 | 0.47 |
| 3P | HP4 | 0.44 | 0.46 |

人类代理在 IPP 和 Communicativeness 两个行为指标上与真实人类数据几乎完全一致，验证了其人类相似性。

### 行动预测性能

| 指标 | HP1 (2P) | HP2 (2P) | HP3 (3P) | HP4 (3P) |
|------|----------|----------|----------|----------|
| 准确率 | 0.63 | 0.63 | 0.43 | 0.44 |
| 相比BC准确率差 | -0.03 | -0.08 | -0.08 | -0.07 |
| 交叉熵损失 | 0.53 | 0.54 | 0.63 | 0.60 |
| Top-10% 准确率 | 0.82 | 0.82 | 0.71 | 0.73 |
| Top-20% 准确率 | 0.95 | 0.95 | 0.87 | 0.88 |

人类代理在保持高自博弈性能的同时，行动预测准确率仅比纯 BC 下降 3-8 个百分点。

## 亮点与洞察

1. **BC + 正则化 RL 的威力**：仅靠行为克隆的策略在三人设定下零分率高达 76%，加入 KL 正则化的 IPPO 后零分率降至 0.3%，证明正则化 RL 是弥补 BC 泛化缺陷的有效手段
2. **OBL 的意外优势**：不使用任何人类数据的 OBL 在双人设定中反而取得最高分（21.04），揭示了一个重要研究缺口——现有方法无法有效利用少量人类数据来提升协调能力
3. **LLM 的初步能力**：DeepSeek-R1 在三人设定中超越了所有其他基线，展现了 LLM 内在的协调能力潜力，但在双人设定中表现不佳，且中位数为 0 说明表现极不稳定
4. **评估协议设计精巧**：通过 API + 预注册 + 有限对局数的三重机制防止过拟合，体现了严谨的实验科学精神
5. **数据效率的迫切需求**：开源数据集刻意限量（仅 3079 局 vs 训练用 14.7 万局），明确指向数据高效学习这一关键研究方向

## 局限与展望

1. **仅支持 2/3 人设定**：Hanabi 支持 2-5 人，但受限于数据可用性，挑战仅覆盖双人和三人场景
2. **缺乏直接人类评估**：人类代理虽然在行为指标上接近人类，但未进行与真人玩家的直接对局验证
3. **数据来源单一**：所有数据均来自 hanab.live 的 H-Group 约定玩家群体，可能不能代表更广泛的人类玩法多样性
4. **LLM 评估不充分**：DeepSeek-R1 仅评估了 100 局（其他方法为 1000 局），统计显著性存疑
5. **理论分析缺乏**：HDR-IPPO 的有效性主要基于经验验证，缺乏为何 KL 正则化能保持人类兼容性的理论解释
6. **未涵盖游戏变体**：未测试彩虹牌等 Hanabi 变体，无法评估方法的泛化能力
7. **FCP 在复杂部分可观察环境中表现不佳**，表明基于种群的方法可能不适合此类场景

## 相关工作与启发

- **Hanabi 基准**：Bard et al. (2019) 提出 Hanabi Challenge，SPARTA 在自博弈中接近满分（24.61/25），但临时协调仍是开放问题
- **正则化 RL 路线**：Bakhtin et al. (2022) 在 Diplomacy 中使用人类数据正则化 RL；Cornelisse & Vinitsky (2024) 将此方法拓展到自动驾驶场景；本文将其应用于 Hanabi
- **零样本协调（ZSC）**：OBL (Hu et al., 2021) 和 OP (Hu et al., 2020) 探索无需人类数据的协调方法
- **FCP**：Strouse et al. (2021) 提出种群方法，本文首次在 Hanabi 中评估其表现
- **LLM 作为博弈智能体**：DeepSeek-R1 的评估为 LLM 在合作博弈中的心智理论能力研究开辟了新方向

## 评分
- 新颖性: ⭐⭐⭐⭐ (4/5) — 首个标准化开源的 Hanabi Human-AI 协调评估框架，评估协议设计严谨
- 实验充分度: ⭐⭐⭐⭐ (4/5) — 基线丰富（含 LLM），人类相似性多维验证，但 LLM 评估量有限
- 写作质量: ⭐⭐⭐⭐ (4/5) — 结构清晰，问题定义和动机明确
- 价值: ⭐⭐⭐⭐⭐ (5/5) — 为 Human-AI 协调研究提供了急需的标准化基准，开源数据+API评估是重要社区贡献

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](../../ACL2026/llm_reasoning/explicit_trait_inference_for_multi-agent_coordination.md)
- [\[ACL 2025\] Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation](../../ACL2025/llm_reasoning/towards_safety_reasoning_in_llms_ai-agentic_deliberation_for_policy-embedded_cot.md)
- [\[AAAI 2026\] Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](../../AAAI2026/llm_reasoning/dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)
- [\[AAAI 2026\] BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](../../AAAI2026/llm_reasoning/blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)
- [\[ACL 2026\] Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](../../ACL2026/llm_reasoning/decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)

<!-- RELATED:END -->
