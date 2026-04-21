---
title: >-
  [论文解读] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization
description: >-
  [NeurIPS 2025][目标检测][自监督强化学习] 针对自监督强化学习中 LLM 策略崩溃和熵崩溃问题，提出动量锚定的 GRPO（M-GRPO）框架和基于 IQR 的低熵轨迹过滤方法，实现稳定训练和 SOTA 性能。
tags:
  - NeurIPS 2025
  - 目标检测
  - 自监督强化学习
  - 策略崩溃
  - 动量锚定
  - GRPO
  - 熵过滤
---

# M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2512.13070](https://arxiv.org/abs/2512.13070)  
**代码**: [GitHub](https://github.com/Bai-Bizhe/M_GRPO) (有)  
**领域**: 大语言模型训练 / 强化学习  
**关键词**: 自监督强化学习, 策略崩溃, 动量锚定, GRPO, 熵过滤

## 一句话总结

针对自监督强化学习中 LLM 策略崩溃和熵崩溃问题，提出动量锚定的 GRPO（M-GRPO）框架和基于 IQR 的低熵轨迹过滤方法，实现稳定训练和 SOTA 性能。

## 研究背景与动机

**领域现状**: 带可验证奖励的强化学习（RLVR）已成为 LLM 后训练增强推理能力的核心方法，但依赖大量人工标注数据和奖励模型基础设施，成本高且领域受限。因此学界探索自监督/无标签 RL 信号，如自一致性、自确定性作为奖励的方法（SRT、TTRL、Intuitor 等）。

**现有痛点**: 自监督 RLVR（SS-RLVR）在长期训练中存在致命的**策略崩溃**问题——训练奖励先升后急剧下降，验证集准确率同步恶化。同时伴随**策略熵崩溃**，模型过早变得过度自信。

**核心矛盾**: 增加 rollout 数量只能延缓而非阻止崩溃；自奖励机制内在不稳定，缺乏稳定的训练目标。

**本文目标**: 在不依赖真值标签的前提下，稳定 SS-RLVR 的训练过程，防止策略崩溃和熵崩溃。

**切入角度**: 从对比学习中动量编码器的成功经验（MoCo）出发，将动量机制引入策略优化。

**核心 idea**: 用缓慢演化的动量模型提供稳定的伪标签训练目标，用 IQR 过滤低熵轨迹保持策略多样性。

## 方法详解

### 整体框架

M-GRPO 由两个关键组件构成：
1. **动量锚定的自监督 RL 框架**: 双模型（当前策略 $\pi_{\theta_q}$ + 动量模型 $\pi_{\theta_k}$）联合生成 rollout 并通过多数投票产生伪标签
2. **IQR 自适应熵过滤器**: 动态修剪低熵轨迹，防止策略过早收敛

### 关键设计

1. **动量模型更新机制**:

    - 动量模型参数 $\theta_k$ 不通过反向传播更新，而是当前策略参数的指数移动平均：
    $\pi_{\theta_k} \leftarrow m \cdot \pi_{\theta_k} + (1-m) \cdot \pi_{\theta_q}$
    - $m \in [0,1)$ 为动量系数（如 0.99），确保动量模型缓慢演化，提供稳定参考
    - 灵感来自 MoCo 系列对比学习工作，首次将动量对比思想迁移到 RL 策略优化

2. **联合 Rollout 与多数投票**:

    - 当前策略生成 $M$ 个响应 $\{y_i^q\}_{i=1}^M$，动量模型生成 $N$ 个响应 $\{y_j^k\}_{j=1}^N$
    - 合并为 $G = M + N$ 大小的池
    - 通过多数投票选出伪真值 $y_v$：答案一致性最高的响应
    - 动量模型的 rollout 加入投票池是关键——减少纯由快速变化的当前策略产生伪标签的噪声

3. **归一化优势估计**: 基于伪真值计算当前策略 $M$ 个 rollout 的二值奖励（一致=1，不一致=0），按 GRPO 框架逐 prompt 归一化：
    $\hat{A}_i = \frac{r(y_v, y_i^q) - \text{mean}(\{r(y_v, y_j^q)\}_{j=1}^M)}{\text{std}(\{r(y_v, y_j^q)\}_{j=1}^M)}$

4. **IQR 自适应熵过滤**:

    - 对每个输入的 $G$ 条轨迹计算轨迹级熵
    - 用四分位距法检测低熵异常值：$T_{IQR} = Q_1 - k \cdot (Q_3 - Q_1)$，$k=0.75$
    - 熵低于阈值的轨迹被剪除
    - 相比静态阈值（如去掉底部 10%），IQR 方法自适应训练过程中熵分布的动态变化

### 损失函数 / 训练策略

学习目标为最大化优势加权的对数似然：
$$\mathcal{J}(\theta_q) = \mathbb{E}_{x \sim D, \{y_i^q\} \sim \pi_{\theta_q}}\left[\sum_{i=1}^M \hat{A}_i \log \pi_{\theta_q}(y_i^q | x)\right]$$

训练细节：
- 骨干模型: Qwen3-4B-Base
- 训练数据: MATH 训练集（无真值标签）
- batch size: 8 questions，每题 32 rollouts，温度 1.1
- 优化器: AdamW (lr=1e-6, cosine warmup)
- KL 损失系数: 0.005
- 动量模型 rollout 数: $N = G/4$

## 实验关键数据

### 主实验

M-GRPO 在 Qwen3-4B-Base 上的表现（无真值标签训练）：

| 方法 | MATH500 | AIME24 | AIME25 | GPQA Dia | GPQA | LiveCode |
|------|---------|--------|--------|----------|------|----------|
| 原始模型 | 61.50% | 0.83% | 5.00% | 34.41% | 29.91% | 9.61% |
| SRT_Best (手选) | 79.20% | 12.50% | 11.67% | 38.26% | 35.04% | 19.69% |
| SRT_Final (崩溃后) | 47.50% | 7.50% | 8.75% | 28.54% | 25.89% | 16.12% |
| **M-GRPO+IQR_Final** | **79.75%** | **14.58%** | **14.17%** | **39.65%** | 35.49% | **27.12%** |

M-GRPO 的最终检查点性能超过 SRT 的最佳手选检查点，无需人工干预。

### 消融实验

Rollout 数量缩放分析（M-GRPO+IQR）：

| 配置 | MATH500 | AIME24 | AIME25 | GPQA Dia | mbpp |
|------|---------|--------|--------|----------|------|
| G=8 | 77.60% | 11.25% | 10.42% | 39.02% | 68.60% |
| G=16 | 79.75% | 14.43% | 10.00% | 39.65% | 70.40% |
| G=32 | 79.75% | 14.58% | 14.17% | 39.65% | 70.60% |
| G=256 | 79.50% | 16.67% | 14.17% | 40.66% | 70.40% |

性能从 G=8 到 G=32 显著提升，之后趋于饱和。

### 关键发现

- SRT 方法在所有 rollout 配置下最终都会崩溃，增加 rollout 只是延缓
- M-GRPO 在 Qwen3-1.7B、4B、8B 三个规模上均保持训练稳定
- 在 GPQA 上提升 +5.05%，LiveCode 上提升 +7.43%（相对 SRT_Best）
- IQR 过滤器有效维持更高的策略熵水平，熵下降更缓慢平稳

## 亮点与洞察

- **问题诊断清晰**: 系统性地揭示了 SS-RLVR 中策略崩溃和熵崩溃的现象及其因果关系
- **动量机制迁移巧妙**: MoCo 的动量思想从视觉对比学习迁移到 RL 策略稳定化，类比清晰
- **实用性强**: 无需人工选择检查点，训练全程稳定，最终模型即最佳模型
- **IQR 自适应过滤**: 比固定阈值更鲁棒，随训练动态调整

## 局限与展望

- 仅在 MATH 数据集上验证，未扩展到其他自监督 RL 场景（如代码生成、对话等）
- 动量系数 $m$ 的选择对性能影响未详细讨论
- 实验仅使用 Qwen3 系列模型，其他模型家族的泛化性未知
- 双模型架构增加了约 25% 的推理计算开销（动量模型的 $N$ 次 rollout）
- 多数投票假设正确答案是主流答案，可能在分布外或困难样本上失效

## 相关工作与启发

- **SRT** (Sheikh et al.): 自训练 RL 方法，本文的直接基线，揭示了其崩溃问题
- **GRPO/DAPO**: 组相对策略优化框架，M-GRPO 在此基础上引入动量机制
- **MoCo** (He et al., 2020): 动量对比学习，M-GRPO 核心灵感来源
- 对其他需要自监督信号的 RL 场景（如多模态推理、工具使用学习）有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 动量+IQR 的组合设计简洁有效，问题分析深入
- 实验充分度: ⭐⭐⭐⭐ 多 benchmark、多规模验证，缩放分析完整
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，图表直观
- 价值: ⭐⭐⭐⭐⭐ 解决了 SS-RLVR 中的关键实践障碍，对 LLM 自进化训练意义重大

<!-- RELATED:START -->

## 相关论文

- [XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)
- [Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)

<!-- RELATED:END -->
