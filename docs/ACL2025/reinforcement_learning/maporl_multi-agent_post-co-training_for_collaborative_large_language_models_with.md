---
title: >-
  [论文解读] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning
description: >-
  [ACL 2025][强化学习][多智能体RL] 提出 MAPoRL——一种基于多智能体强化学习的后训练范式，通过让多个 LLM 在辩论框架中共同训练（co-training），配合验证器评分和协作激励机制，显著提升多 LLM 协作的效果，并展现出跨任务的泛化能力。 领域现状：多 LLM 协作（如多智能体辩论）近年受到关注…
tags:
  - "ACL 2025"
  - "强化学习"
  - "多智能体RL"
  - "LLM协作"
  - "后训练"
  - "多轮辩论"
  - "协作奖励塑形"
---

# MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning

**会议**: ACL 2025  
**arXiv**: [2502.18439](https://arxiv.org/abs/2502.18439)  
**代码**: [https://github.com/chanwoo-park-official/MAPoRL](https://github.com/chanwoo-park-official/MAPoRL)  
**作者**: Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman Ozdaglar, Kaiqing Zhang, Joo-Kyung Kim  
**机构**: MIT, Stanford, Amazon, UMD  
**领域**: 强化学习 / 多智能体协作  
**关键词**: 多智能体RL, LLM协作, 后训练, 多轮辩论, 协作奖励塑形

## 一句话总结

提出 MAPoRL——一种基于多智能体强化学习的后训练范式，通过让多个 LLM 在辩论框架中共同训练（co-training），配合验证器评分和协作激励机制，显著提升多 LLM 协作的效果，并展现出跨任务的泛化能力。

## 研究背景与动机

**领域现状**：多 LLM 协作（如多智能体辩论）近年受到关注，但现有方法主要依靠 prompting 开箱即用的 LLM，期望它们"天然"具备协作能力。

**现有方法的不足**：
   - 多轮辩论不一定能改善性能，对于能力不够强的模型甚至可能导致性能下降（Huang et al., 2024）
   - LLM 在预训练时从未被显式训练过如何协作，单纯通过 prompt 无法激发真正的合作行为
   - 单智能体训练（如 SFT 或单独 RL）不足以产生有效协作——一个未经训练的、非策略性的对手无法驱动合作行为的产生

**核心动机**：需要一个交互式训练环境，让多个 LLM 同时训练、动态优化各自策略，从而显式地学习协作行为。博弈论分析表明，共同训练的智能体能够达到展现合作行为的均衡点。

## 方法详解

### 整体框架

MAPoRL 以协作辩论（Collaborative Debate）为基础框架，其流程为：

1. **独立生成**：每个 LLM agent 独立生成对问题的初始回答
2. **多轮讨论**：agents 进行 T 轮讨论，每轮基于所有 agents 的历史回答生成新回答
3. **验证器评分**：MAPoRL 验证器同时评估答案正确性和讨论质量
4. **多智能体 RL**：以验证器分数为奖励，通过多智能体 PPO 最大化每个 agent 的价值函数

### 关键设计一：影响感知验证奖励（Influence-aware Verification Reward）

普通的奖励只看当前回答的正确性，而 MAPoRL 的奖励函数考虑了 agent 对未来所有 agents 回答的影响：

$$R_{\theta}(q, s_{ta}) = \mathbb{E}\left[\frac{1}{\sum_{t' \in [t,T]} \gamma^{t'-t}} \left(\text{Verifier}(q, s_{ta}) + \sum_{t' \in [t+1,T]} \sum_{j \in [A]} \frac{1}{A} \gamma^{t'-t} \text{Verifier}(q, s_{t'j})\right)\right]$$

- 不仅考虑当前回答的分数，还纳入未来所有 agents 回答的折扣分数
- 鼓励 agent 不只追求当前得分，还要为后续讨论提供有价值的信息
- $\gamma$ 为折扣因子，控制对未来影响的重视程度

### 关键设计二：协作激励机制（Reward Shaping）

引入四个激励参数以强化不同的协作行为：

| 参数 | 含义 |
|------|------|
| $\alpha_0$ | 奖励从错误答案中提取有用信息（批判性推理） |
| $\alpha_1$ | 奖励被正确信息说服（可说服性） |
| $\beta_0$ | 奖励提供虽错但有用的答案（建设性影响） |
| $\beta_1$ | 奖励用正确答案说服他人（说服力） |

具体实现：根据 agent 在连续轮次中答案正确性的变化以及多数投票的变化方向，分配正/负激励。

### 关键设计三：多智能体 PPO

- 每个 agent 在每轮拥有独立的策略网络和价值网络
- 状态定义为完整的多 agent 交互历史
- 损失函数 = PPO surrogate loss + 价值函数 MSE loss
- 加入 KL 正则化防止策略偏离参考模型过远
- 使用 GAE（Generalized Advantage Estimation）估计优势函数

### 博弈论分析

论文用简化模型（2 轮、2 agent、2 动作：协作/独立行动）证明：
- **Observation 1**：单智能体训练时，如果对手的协作概率不够高，最优策略是不协作
- **Observation 2**：两个智能体共同训练时，只要协作收益 $R_{syn}$ 足够大，均衡点自然导向合作

## 实验关键数据

### 数据集
- **GSM8K**：高中数学推理（7463 训练 + 12800 训练 MAPoRL + 1319 测试）
- **ANLI**：对抗自然语言推理（10000 训练验证器 + 12800 训练 MAPoRL + 1200 测试）

### 模型
- 主模型：Phi-3-mini-128k-instruct (3.4B)，QLoRA 量化微调
- 辅助模型：Qwen2.5-3B-instruct, Llama-3-8B-instruct

### 实验1：开箱即用 vs. MAPoRL 训练

| 设置 | GSM8K (T1→T2→T3) | ANLI (T1→T2→T3) |
|------|-------------------|------------------|
| Phi-3 开箱即用 | 性能无提升甚至下降 | 性能无提升甚至下降 |
| MAPoRL 训练 | 随轮次持续提升 | 随轮次持续提升 |

关键发现：单独测试 MAPoRL 训练模型（不协作时），性能与原始模型相当（GSM8K: 0.609 vs 0.604/0.611），证明提升来自**协作能力**而非任务知识。

### 实验2：激励参数分析

- 增大 $\alpha_1$=2 时，$\Delta_1$ 提升 9.5%——跟随正确多数意见的能力增强
- 增大 $\beta_0$=2 时，$\Delta_0$ 提升 17.2%——提供"有建设性的错误答案"的能力显著增强
- $\beta_1$ 的增大反而无效，说明直接奖励"用正确答案说服"不如奖励"提供有用信息"

### 实验3：跨域迁移

| 训练→评估 | Turn 1 | Turn 2 | Turn 3 |
|-----------|--------|--------|--------|
| ANLI→GSM8K（开箱即用） | 0.677 | 0.688 | 0.640 |
| ANLI→GSM8K（MAPoRL） | 0.677 | 0.712 | **0.720** |
| GSM8K→ANLI（开箱即用） | 0.482 | 0.486 | 0.468 |
| GSM8K→ANLI（MAPoRL） | 0.482 | 0.499 | **0.507** |

证明协作能力可以跨任务迁移。

### 实验4：异构模型协作

Phi-3 (3.4B) + Qwen2.5 (3B) 和 Phi-3 + Llama-3 (8B) 的异构配对通过 MAPoRL 共同训练后效果优于单模型，不同能力的模型协同效果更好。

## 亮点与洞察

1. **首次系统性**地用多智能体 RL 训练多 LLM 协作系统（非 SFT），且提供了博弈论分析支持
2. **激励设计的反直觉发现**：奖励"有建设性的错误"（$\beta_0$）比奖励"正确说服"（$\beta_1$）更有效
3. **协作能力是元能力**：MAPoRL 学到的不是特定任务知识，而是跨域的协作策略
4. **实用价值高**：可应用于任何有评分器的多 LLM 系统

## 局限性

1. 由于计算资源限制，主要在 3-4B 的量化模型上实验，更大模型的效果未验证
2. 仅测试了辩论框架，其他多 agent 协作框架（如共识、分工）未探索
3. 激励参数的最优设置需要手动调节，缺乏自动化选择机制
4. 验证器的质量直接影响训练效果，但验证器本身的训练未深入讨论

## 相关工作

- **多 LLM 协作**：Du et al. (2024) 多智能体辩论, Li et al. (2023) 基于 prompt 的多 agent 系统
- **LLM 后训练**：RLHF/PPO 范式的单智能体后训练
- **同期工作**：Subramaniam et al. (2025) 和 Zhao et al. (2025) 用迭代 SFT（非 RL）训练多 LLM 协作

## 评分

⭐⭐⭐⭐ (4/5)

理论分析（博弈论视角）与实验验证结合扎实，首次提出多智能体 RL co-training 的后训练范式。但实验规模受限于计算资源（仅 3-4B 模型，QLoRA），且仅在两个 benchmark 上验证。激励机制的分析很有启发性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](../../NeurIPS2025/reinforcement_learning/repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ACL 2025\] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)
- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](../../ACL2026/reinforcement_learning/why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ACL 2025\] Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)

</div>

<!-- RELATED:END -->
