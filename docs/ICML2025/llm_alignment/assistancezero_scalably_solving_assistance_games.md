---
title: >-
  [论文解读] AssistanceZero: Scalably Solving Assistance Games
description: >-
  [ICML 2025][LLM对齐][assistance game] 提出 AssistanceZero，首次将 assistance game 扩展到复杂环境（Minecraft 建筑辅助，$10^{400}$ 种可能目标），通过扩展 AlphaZero 增加 reward 预测头和人类行为预测头，在 MCTS 下进行不确定性规划，显著优于 PPO 和模仿学习基线，人类实验证明能有效减少用户操作并展现挖地基、推断屋顶、从纠正中学习等涌现行为。
tags:
  - ICML 2025
  - LLM对齐
  - assistance game
  - AlphaZero
  - MCTS
  - human modeling
  - Minecraft
  - cooperative AI
  - POMDP
---

# AssistanceZero: Scalably Solving Assistance Games

**会议**: ICML 2025  
**arXiv**: [2504.07091](https://arxiv.org/abs/2504.07091)  
**代码**: https://github.com/cassidylaidlaw/minecraft-building-assistance-game  
**领域**: 对齐RLHF  
**关键词**: assistance game, AlphaZero, MCTS, human modeling, Minecraft, cooperative AI, POMDP

## 一句话总结

提出 AssistanceZero，首次将 assistance game 扩展到复杂环境（Minecraft 建筑辅助，$10^{400}$ 种可能目标），通过扩展 AlphaZero 增加 reward 预测头和人类行为预测头，在 MCTS 下进行不确定性规划，显著优于 PPO 和模仿学习基线，人类实验证明能有效减少用户操作并展现挖地基、推断屋顶、从纠正中学习等涌现行为。

## 研究背景与动机

### RLHF 的局限性

当前主流 AI 助手训练范式（预训练 + SFT + RLHF）存在结构性问题：

**欺骗激励**：标注者可被欺骗给出正向反馈，激励模型产生欺骗/操纵行为

**不鼓励不确定性维护**：单轮高评分的目标不鼓励助手询问澄清问题或对冲回答

**非协作性**：自动补全类助手（如 Copilot）无法考虑人机协作的互补性——助手行为应与用户行为互补而非替代

### Assistance Game 的优势

Assistance game 是一个双人博弈：助手和用户在共享环境中行动，共享 reward function，但助手**无法观测**目标参数 $\theta$。这一框架：

- 去除欺骗激励（reward 依赖真实隐含 reward 而非人类反馈）
- 激励助手通过交互解决不确定性
- 产生与用户行动互补的最优联合行为

### 为何 Assistance Game 此前未被广泛研究？

两大挑战：(1) 不确定性下的决策问题计算上不可解；(2) 需要准确的人类行为模型。此前工作仅限于 ≤10 个离散 reward 参数的简单环境。

## 方法详解

### 整体框架

**环境设计：Minecraft Building Assistance Game (MBAG)**

- 状态：3D 方块网格（11×10×10）+ 玩家位置 + 背包
- 动作空间：无操作、六方向移动、放置方块、打破方块（>20,000 种可能动作）
- 目标参数 $\theta$：目标建筑的方块网格（基于 CraftAssist 数据集）
- $|\Theta| \approx 10^{400}$——远超此前工作的不到 20 种
- Reward $R(s, a_H, a_R; \theta) = d(s', \theta) - d(s, \theta)$（编辑距离变化）

### 关键设计

**PPO 失败分析**：PPO 在 MBAG 上几乎不起作用（assistant goal % ≈ 0%）。原因：

- reward 信号高度噪声化（reward 同时依赖人类和助手动作）
- 即使期望有益的动作也可能得到负 reward（因目标不确定）
- 长序列决策进一步放大 reward-to-go 噪声
- PPO 学到的主信号是"放置/打破 = 负 reward"→收敛到什么都不做

**AssistanceZero 的核心思想**：将目标预测与行动选择**分离**

循环神经网络具有四个头：
1. **策略头** $\pi_\phi(a_R | h)$：选择助手动作
2. **价值头** $\hat{V}_\phi(h)$：估计状态价值
3. **Reward 参数预测头** $\hat{p}_\phi(\theta | h)$：预测目标建筑每个位置的方块类型分布
4. **人类动作预测头** $\hat{p}_\phi(a_H | h)$：预测人类下一步动作

MCTS 通过采样 reward 参数和人类动作来模拟未来轨迹，实现不确定性下的规划。

### 损失函数/训练策略

AssistanceZero 的完整损失函数：

$$L(\phi) = \frac{1}{n} \sum_{t=1}^{n} \left[ \lambda_{\text{policy}} D_{\text{KL}}(\pi_t^{\text{MCTS}} \| \pi_\phi(\cdot|h_t)) + \lambda_{\text{value}} (\hat{V}_\phi(h_t) - \sum_{t'=t}^{T} \gamma^{t'-t} R_{t'})^2 - \lambda_{\text{reward}} \log \hat{p}_\phi(\theta|h_t) + \lambda_{\text{prev-rew}} D_{\text{KL}}(\hat{p}_\phi(\theta|h_t) \| \hat{p}_t(\theta)) - \lambda_{\text{action}} \log \hat{p}_\phi(a_H^t | h_t) \right]$$

五项损失分别训练四个头，其中 $\lambda_{\text{prev-rew}}$ 项防止 reward 预测头过拟合到最近见到的目标。

**MCTS 的 reward 估计**中使用低方差技巧：利用 reward 可分解为 $R = R_H + R_R$，助手 reward 用当前时刻估计，人类 reward 用下一时刻估计。

**人类建模**：
- Reward-based（PPO/AlphaZero）：预测差、建造过快
- BC（行为克隆）：准确预测但累积误差
- **piKL（最优选择）**：MCTS + BC 先验策略，平衡预测准确性和任务表现

## 实验关键数据

### 主实验

**Table 1：固定人类模型评估**

| 方法 | 总目标完成率 | 人类动作数 | 助手完成率 |
|------|------------|-----------|-----------|
| PPO baseline | 71.6% | 203 | 0.0% |
| PPO + reward engineering | 74.0% | 200 | 3.5% |
| PPO + aux loss | 74.1% | 191 | 7.2% |
| **AssistanceZero** | **79.8%** | **158** | **27.0%** |
| 人类模型独自 | 70.8% | 200 | — |

AssistanceZero 减少 42 次人类动作，助手自主完成 27% 目标。

**Table 3：不同训练范式对比**

| 方法 | 总目标完成率 | 人类动作数 | 助手完成率 |
|------|------------|-----------|-----------|
| Pretraining（类 Copilot） | 89.8% | 240 | 2.3% |
| SFT（类 RLHF 第一阶段） | 90.4% | 241 | 2.9% |
| **Assistance Game** | **92.6%** | **179** | **26.0%** |

### 消融实验

- 移除 LSTM：目标率从 77.5% 暴降至 69.0%，助手完成率从 25.2% 降至 -0.6%
- 移除 KL 正则（$\lambda_{\text{prev-rew}}$）：目标率从 77.5% 降至 76.8%，助手完成率从 25.2% 降至 18.1%
- 去除测试时 MCTS：性能不降（80.2% vs 79.8%），说明优势不来自额外推理计算

### 关键发现

**人类实验（16 名被试）**：
- AssistanceZero 助手帮助度评分 3.1/5 vs SFT 1.7/5 vs 人类助手 4.0/5
- 显著减少参与者的放置/打破操作（$p < 0.05$）
- **涌现行为**：
    - 挖地基：观察人类勾勒轮廓后自动清理内部
    - 推断屋顶：从人类开始建造的几块推断屋顶结构并完成
    - 从纠正学习：建墙太高时，人类打掉一块，助手自动打掉其余多余方块

## 亮点与洞察

1. **首次将 assistance game 扩展到复杂环境**（$10^{400}$ 种目标），证明其可行性
2. **PPO 失败的深层原因分析**精辟：reward 噪声 + 目标预测与行动耦合是核心瓶颈
3. **分离预测与行动**的设计理念：AlphaZero 框架天然适合 POMDP 中的 belief 维护
4. **人类建模的实证发现**：纯 reward-based 模型不预测人类行为，BC 有累积误差，piKL 是最佳折中
5. **涌现行为**展示 assistance game 框架的本质优势——助手学会了语用沟通，而非简单模仿
6. **对 LLM 后训练的展望**：将对话视为多轮 assistance game，可解决 RLHF 的欺骗激励和不确定性回避问题

## 局限与展望

1. **环境简化**：MBAG 是极度简化的 Minecraft，真实世界复杂度远高于此
2. **人类模型数据量有限**：仅 5 名被试 18 个 episode 训练 BC 模型
3. **未与完整 RLHF 对比**：因 RLHF 在多智能体环境难以直接应用，仅与 SFT 对比
4. **计算开销**：AssistanceZero 需 MCTS 模拟（训练时 100 次/步），计算成本高
5. **人类实验规模**：16 名被试，人类助手仅 1 名，统计效力有限
6. **LLM 扩展距离远**：从 Minecraft 到 LLM 对话的迁移路径仍是愿景

## 相关工作与启发

- **Assistance game 理论线**：Fern et al. 2014（hidden-goal MDP）→ Hadfield-Menell et al. 2016（CIRL）→ 本文（首次大规模求解）
- **人类建模**：BC 有累积误差（DAgger 问题），piKL（Jacob et al. 2022）是合理的混合方案
- **AlphaZero 扩展**：与 MuZero 的区别在于处理部分可观测性和随机性；与 POMCP 的区别在于使用学习模型
- **Learned Belief Search**（Hu et al. 2021）：类似从 rollout 学习 belief 分布的技术
- **启发**：assistance game 可能是 RLHF 的"下一代"对齐范式，但需要更好的人类模型和更高效的规划算法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次大规模 assistance game 求解，全新的方法路径)
- 实验充分度: ⭐⭐⭐⭐ (仿真+人类实验+丰富消融，但人类实验规模有限)
- 写作质量: ⭐⭐⭐⭐⭐ (论文结构清晰，图示精美，涌现行为展示极有说服力)
- 价值: ⭐⭐⭐⭐⭐ (开辟 assistance game 大规模应用方向，对 AI 对齐有深远启示)

<!-- RELATED:START -->

## 相关论文

- [Diverging Preferences: When do Annotators Disagree and do Models Know?](diverging_preferences_when_do_annotators_disagree_and_do_models_know.md)
- [MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift](right_now_wrong_then_non-stationary_direct_preference_optimization_under_prefere.md)

<!-- RELATED:END -->
