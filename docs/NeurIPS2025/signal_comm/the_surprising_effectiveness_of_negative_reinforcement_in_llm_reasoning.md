---
title: >-
  [论文解读] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning
description: >-
  [NeurIPS 2025][RLVR分解] 将可验证奖励的强化学习（RLVR）分解为正样本强化（PSR，增强正确回答概率）和负样本强化（NSR，惩罚错误回答），发现仅用 NSR 就能在整个 Pass@k 谱上持续提升推理性能且通常匹配或超越 PPO/GRPO，据此提出 Weighted-REINFORCE（降低 PSR 权重至 0.1）在 MATH/AIME 2025/AMC23 上取得全面最优。
tags:
  - NeurIPS 2025
  - RLVR分解
  - 负样本强化
  - 正样本强化
  - Pass@k
  - Weighted-REINFORCE
---

# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2506.01347](https://arxiv.org/abs/2506.01347)  
**代码**: [GitHub](https://github.com/TianHongZXY/RLVR-Decomposed)  
**领域**: LLM 推理 / 强化学习  
**关键词**: RLVR分解, 负样本强化, 正样本强化, Pass@k, Weighted-REINFORCE

## 一句话总结

将可验证奖励的强化学习（RLVR）分解为正样本强化（PSR，增强正确回答概率）和负样本强化（NSR，惩罚错误回答），发现仅用 NSR 就能在整个 Pass@k 谱上持续提升推理性能且通常匹配或超越 PPO/GRPO，据此提出 Weighted-REINFORCE（降低 PSR 权重至 0.1）在 MATH/AIME 2025/AMC23 上取得全面最优。

## 研究背景与动机

**领域现状**：可验证奖励的强化学习（RLVR）已成为训练 LLM 推理能力的核心技术，DeepSeek-R1、Kimi K1.5 通过 RLVR 展现了涌现式长链推理和自反思。RLVR 使用二元奖励（正确 +1 / 错误 -1），概念简单且不需要复杂的奖励模型。

**现有痛点**：（1）RLVR 的底层学习机制不清楚——模型如何分别从正确和错误样本中学习？（2）现有评估偏重 Pass@1（贪心准确率），忽略大 k 时的 Pass@k，掩盖了模型行为深层变化。近期研究（Yue et al. 2025）发现 RL 训练后模型在 Pass@k（大 k）上反而不如基座模型，暗示多样性丧失。

**核心矛盾**：RLVR 同时在做两件事（强化正确+惩罚错误），两者效果纠缠。直觉上正强化应是主要信号，但它可能过度收窄输出分布；负强化看似辅助，但其效果可能被低估。

**本文目标** 将 RLVR 解耦为 PSR 和 NSR，分别研究对 Pass@k 推理缩放的影响，并设计更好的训练目标。

**切入角度**：RLVR 目标精确分解为 $\mathcal{L}_{RLVR} = \mathcal{L}_{PSR} + \mathcal{L}_{NSR}$。通过单独训练并用 Pass@k 全谱（k=1到256）评估，可以观察各自独立效果。再通过 token 级梯度分析解释机制。

**核心 idea**：仅惩罚错误回答就能有效提升推理——它通过抑制错误并按模型先验重新分配概率来精炼已有知识。

## 方法详解

### 整体框架

RLVR 目标 $\mathcal{L} = -\mathbb{E}[r(\bm{x}, \bm{y})]$（$r \in \{-1, +1\}$）按奖励符号分解为 PSR（正样本似然最大化）和 NSR（负样本似然最小化）两个子目标。分别独立训练 Qwen2.5-Math-7B、Qwen3-4B 和 Llama-3.1-8B-Instruct，在 MATH/AIME/AMC23 上评估 Pass@k。

### 关键设计

1. **RLVR 目标分解与独立评估**:

    - 功能：分离正强化和负强化信号的贡献
    - 核心思路：$\mathcal{L}_{PSR} = -\mathbb{E}[\sum_{r=1} \pi_\theta(\bm{y}|\bm{x})]$；$\mathcal{L}_{NSR} = \mathbb{E}[\sum_{r=-1} \pi_\theta(\bm{y}|\bm{x})]$。训练时仅用对应奖励的样本更新策略。PSR-only 和 NSR-only 各自比 PPO/GRPO 使用更少的样本（各仅用半批数据）
    - 设计动机：解耦后可直接归因 RLVR 中各信号的贡献，回答"正强化和负强化谁更重要"。Pass@k 全谱提供了比 Pass@1 更全面的能力评估

2. **Token 级梯度分析**:

    - 功能：从梯度层面解释 PSR 和 NSR 对输出分布的不同影响
    - 核心思路：PSR 对采样 token $y_t$ 的梯度方向为 $\propto \pi_v(1-\pi_v)$（增大），其余 token $\propto -\pi_{y_t}\pi_v$（减小）——分布锐化，多样性下降。NSR 对错误 token 方向为 $\propto -\pi_v(1-\pi_v)$（减小），其余 token $\propto \pi_{y_t} \cdot \pi_v$（增大）。关键：**NSR 的概率重分配与当前概率成正比**，即模型按先验知识自动找到替代方案
    - 设计动机：揭示 NSR 的"自校准"机制——不教新行为，而是移除错误后让模型自身先验浮现。PSR 则强制集中到已观察到的正确路径，压制其他可能正确的答案

3. **Weighted-REINFORCE**:

    - 功能：利用 PSR/NSR 分析洞察设计改进的 RL 目标
    - 核心思路：$\mathcal{L}_{W-REINFORCE} = \lambda \cdot \mathcal{L}_{PSR} + \mathcal{L}_{NSR}$，设 $\lambda = 0.1$ 大幅降低正强化权重
    - 设计动机：PSR 提升 Pass@1 但损害大 k 的 Pass@k，NSR 保持全谱但 Pass@1 略逊。$\lambda=0.1$ 在准确率和多样性之间取得最佳平衡

### 损失函数 / 训练策略

训练集 MATH 7500 题，prompt batch 1024，每 prompt 8 rollouts，学习率 1e-6。Qwen2.5-Math-7B 和 Llama 最大长度 4096，Qwen3-4B 最大长度 32768。评估时采 256/64 个回答，用 Chen et al. 2021 的无偏 Pass@k 估计器。

## 实验关键数据

### 主实验

| 方法 | MATH P@1 | MATH P@256 | AIME P@1 | AIME P@256 | AMC P@1 | AMC P@256 |
|------|---------|-----------|---------|-----------|--------|----------|
| Base | 63.2 | 96.9 | 6.1 | 46.7 | 41.0 | 100.0 |
| PPO | 76.6 | 96.3 | 8.5 | 43.3 | 62.0 | 97.5 |
| GRPO | 76.3 | 95.5 | 10.3 | 50.0 | 61.7 | 97.5 |
| PSR | 74.1 | 91.2 | 11.6 | 43.3 | 62.6 | 92.5 |
| NSR | 75.7 | **96.9** | 10.0 | **53.3** | 60.9 | **100.0** |
| W-REINF | **76.6** | 96.7 | **10.6** | **56.7** | **62.0** | 97.5 |

### 消融实验

| 训练动态 | PSR 效果 | NSR 效果 |
|---------|---------|---------|
| 测试集熵 | 急剧下降 → 多样性丧失 | 保持接近基座 → 多样性保留 |
| 训练集正确比例 | 快速上升 → 过拟合 | 缓慢上升 → 积极但不过度 |
| 全部正确占比 | 最高 → 过度自信 | 最低 → 保持不确定性 |
| Pass@1 趋势 | 快速提升后饱和 | 稳步持续提升 |

### 关键发现

- **NSR 最核心发现**：仅用负样本训练即达 MATH Pass@1=75.7（接近 PPO 76.6），Pass@256=96.9 完全匹配基座——不引入正确样本也能既提升准确率又保持多样性
- **PSR 的代价明确**：MATH Pass@256 从 96.9 掉到 91.2，k>8 后 Pass@k 低于基座。PSR 导致分布坍缩
- **Qwen3-4B 案例**：PSR 完全无法激活非思考模式的潜在推理能力；NSR 将 Pass@1 从 ~80% 提升到 94%，接近思考模式的 94.5%
- **Llama 案例**：所有 RL 方法都退化，但 NSR 退化最小——基座模型先验质量决定 RL 收益
- **W-REINFORCE**：AIME Pass@256=56.7 大幅超越 GRPO 50% 和 PPO 43.3%

## 亮点与洞察

- **"负强化比正强化更重要"的反直觉结论**：传统认为模型需正面示范才能进步，但本文证明仅排除错误就能有效学习。核心机制是 NSR 的概率重分配遵循模型先验——"帮模型清除干扰，让模型自己找到正确答案"。对整个 RLHF/RLAIF 领域有启示
- **Pass@k 全谱评估范式**：仅看 Pass@1 会误导——PPO/GRPO 看似提升准确率，实际牺牲了推理覆盖能力。Pass@k 全谱才能完整反映模型能力边界
- **Weighted-REINFORCE 的简约之美**：单参数 $\lambda=0.1$ 一致超越复杂的 PPO 和 GRPO，挑战了"更复杂 RL 算法更好"的假设
- NSR 梯度中 $(1-\pi_v)$ 因子提供自然停止机制：当错误 token 概率已很低时梯度趋零，避免过度惩罚

## 局限与展望

- **模型依赖性强**：NSR 在 Qwen 上效果显著但 Llama 全面退化，方法有效性与基座先验质量高度相关
- 仅在数学推理验证，代码/科学推理等任务是否类似需进一步确认
- $\lambda=0.1$ 选择可能非普遍最优，不同模型/任务可能需要不同值
- 未分析 NSR 在训练不同阶段的动态：早期 vs 后期是否应调整 PSR/NSR 权重
- 与 DPO/KTO 等偏好学习方法的联系未讨论

## 相关工作与启发

- **vs Yue et al. 2025**: 他们发现 RL 后 Pass@k 下降。本文解释来源——PSR 导致多样性坍缩，NSR 提供避免退化的路径
- **vs Dang et al. 2025**: 他们通过权重插值恢复 SFT 后多样性。本文从训练目标源头降低 PSR 权重
- **vs DeepSeek-R1**: R1 将 RLVR 视为整体。本文分解视角暗示可通过 W-REINFORCE 进一步提升 R1 的推理缩放性能

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 分解视角简洁深刻，发现反直觉且重要
- 实验充分度: ⭐⭐⭐⭐ 三模型三基准全谱评估 + 梯度分析
- 写作质量: ⭐⭐⭐⭐⭐ 叙事流畅，图表精炼
- 价值: ⭐⭐⭐⭐⭐ 对 RLVR 机制理解有范式性影响

<!-- RELATED:START -->

## 相关论文

- [Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [Angular Steering: Behavior Control via Rotation in Activation Space](angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [ConTextTab: 语义感知的表格上下文学习器](contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)

<!-- RELATED:END -->
