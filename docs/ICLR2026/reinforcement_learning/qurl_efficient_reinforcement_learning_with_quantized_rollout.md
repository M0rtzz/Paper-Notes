---
title: >-
  [论文解读] QuRL: Efficient Reinforcement Learning with Quantized Rollout
description: >-
  [ICLR 2026][量化推理] 提出 QuRL 方法，通过量化 actor 模型加速 RL 训练中的 rollout 阶段，设计自适应裁剪范围（ACR）解决量化导致的训练崩溃问题，并提出更新感知量化（UAQ）解决权重更新远小于量化误差的尺度失配问题，实现 20%~80% 的推理吞吐量提升且不损失性能。
tags:
  - ICLR 2026
  - 量化推理
  - RL加速
  - PPO
  - GRPO
  - importance sampling
---

# QuRL: Efficient Reinforcement Learning with Quantized Rollout

**会议**: ICLR 2026  
**arXiv**: [2602.13953](https://arxiv.org/abs/2602.13953)  
**代码**: 无  
**领域**: 强化学习 / 模型量化  
**关键词**: 量化推理, RL加速, PPO, GRPO, importance sampling

## 一句话总结

提出 QuRL 方法，通过量化 actor 模型加速 RL 训练中的 rollout 阶段，设计自适应裁剪范围（ACR）解决量化导致的训练崩溃问题，并提出更新感知量化（UAQ）解决权重更新远小于量化误差的尺度失配问题，实现 20%~80% 的推理吞吐量提升且不损失性能。

## 研究背景与动机

**领域现状**：RLVR（基于可验证奖励的强化学习）已成为训练推理型 LLM 的主流范式（如 DeepSeek-R1、OpenAI-O1），但 rollout 阶段因自回归解码的序列依赖性占用了约 70% 的训练时间。

**现有痛点**：(1) rollout 是 RL 训练的效率瓶颈，推理型任务需要更长的 CoT 进一步加剧；(2) 直接将量化应用于 rollout 会导致 importance sampling 偏差和训练不稳定；(3) RL 的信任域约束使权重更新量级（$\sim 10^{-7}$）远小于量化误差，导致量化模型几乎无法感知训练动态。

**核心矛盾**：量化可以显著加速推理，但量化 actor 与全精度 actor 之间的策略分歧会破坏 PPO/GRPO 的 importance sampling 和 clipping 机制。

**本文目标** 如何在保持 RL 训练质量的同时，利用量化加速 rollout 推理。

**切入角度**：结合 Decoupled PPO 分离行为策略和近邻策略，在此基础上解决量化带来的两个独特挑战：clipping 不稳定和权重更新被量化淹没。

**核心 idea**：用量化模型做 rollout 但用全精度模型做 clipping 和梯度更新，通过自适应裁剪范围和不变缩放技术弥合量化带来的差距。

## 方法详解

### 整体框架

RL 训练循环中：(1) 将旧 actor $\theta_{\text{old}}$ 量化为 $\hat{\theta}_{\text{old}}$ 用于 rollout 生成响应；(2) 用全精度 $\theta_{\text{old}}$ 计算近邻策略的 clipping ratio $R_{i,t}$；(3) 用 Decoupled PPO 目标函数训练，ACR 动态调节裁剪范围；(4) UAQ 通过不变缩放预处理权重，减小量化误差并放大权重更新。

### 关键设计

1. **自适应裁剪范围 (Adaptive Clipping Range, ACR)**:
    - 做什么：解决量化 rollout 导致的长期训练崩溃问题
    - 核心思路：在 Decoupled PPO 中，行为策略（量化 actor $\pi_{\hat{\theta}_{\text{old}}}$）与近邻策略（全精度 $\pi_{\theta_{\text{old}}}$）分离。FlashRL 的 TIS 方法通过截断 $\min(\pi_{\theta_{\text{prox}}}/\pi_{\theta_{\text{behav}}}, C)$ 稳定训练，但隐含了一个缩放因子 $r_{i,t}$，对正 advantage 序列过度裁剪。ACR 将上界调整为 $(1+\epsilon)/r_{i,t}$：$\mathcal{J}_{\text{ACR}} = \tilde{\mathbb{E}}[\min(\pi_{\text{prox}}/\pi_{\text{behav}}, C) \cdot \min(R_{i,t}A_{i,t}, \text{clip}(R_{i,t}, 1-\epsilon, (1+\epsilon)/r_{i,t})A_{i,t})]$
    - 设计动机：在训练后期（>1000步），量化actor与全精度actor的KL散度从0.002增长到0.025（12×），TIS的固定截断导致偏差梯度估计。ACR根据策略分歧动态放宽裁剪上界，让更多正 advantage token 得到训练

2. **更新感知量化 (Update-Aware Quantization, UAQ)**:
    - 做什么：解决权重量化变化量与权重更新量的尺度失配
    - 核心思路：利用线性层的不变缩放 $WX = (W/s) \cdot (sX)$，选择 $s > 1$ 使量化误差$\propto |\theta|/(s \cdot 2^b)$ 降低 $s$ 倍，同时权重更新$\propto s \cdot \alpha G$ 放大 $s$ 倍，产生 $s^2$ 的信噪比改善。$s$ 列式应用于 $W$，行式应用于前一层激活（可合并到 LayerNorm）
    - 设计动机：RL 中学习率 $\alpha \sim 10^{-6}$，梯度 $G \sim 0.1$-$1.0$，权重更新量级 $\sim 10^{-7}$ 远小于量化误差（权重范数 $\sim 0.001$-$0.1$）。实验显示 INT8 量化几乎掩盖了所有权重更新，量化模型实际上在"冻结"

3. **系统集成与工程实现**:
    - 做什么：将 ACR 和 UAQ 集成到 VeRL RL 训练框架
    - 核心思路：UAQ 是一次性的权重预处理（RL训练前执行），不增加训练开销。ACR 仅修改 clipping 逻辑，计算开销可忽略。利用 vLLM 的 INT8/FP8 矩阵乘法核加速推理
    - 设计动机：QuRL 介于 PTQ 和 QAT 之间——不像 QAT 显式优化量化，但参数通过量化模型产出的梯度隐式更新。需要简单但有效的量化策略

### 损失函数 / 训练策略

采用 GRPO/DAPO 目标函数的 decoupled 变体。通道级权重量化 + token级激活量化（INT8 或 FP8）。UAQ 缩放因子 $s = 1.5$。TIS 截断阈值 $C$ 沿用 FlashRL 设置。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(QuRL) | FlashRL | BF16基线 | 说明 |
|--------|------|------|------|------|------|
| GSM8K(INT8) | Accuracy | 53.55 | 51.40 | 55.35 | 差距从4%缩小到1.8% |
| GSM8K(FP8) | Accuracy | 54.28 | 53.60 | 55.35 | 差距仅1.1% |
| AIME2024(INT8) | Avg@32 | 31.25 | 30.29 | 31.67 | w/ UAQ几乎无损 |
| AIME2024(FP8) | Avg@32 | 33.27 | 32.60 | 31.67 | FP8超过BF16！ |
| DeepScaleR(INT8) | Avg5任务 | 55.48 | 53.80 | 56.40 | 差距从4.1%缩小到0.9% |
| DeepScaleR(INT8) | AIME24 | 40.52 | 36.77 | 40.73 | 几乎匹配全精度 |

### 消融实验

| 配置 | AIME24 Avg@32 | 说明 |
|------|------|------|
| QuRL w/o UAQ (INT8) | 30.63 | 基准 |
| QuRL w/ UAQ s=1.5 | 31.25 | +0.62, 最优缩放 |
| QuRL w/ s=2.0 | 29.15 | 过大导致不稳定 |
| 直接增大学习率 1.5× | 29.06 | 不如 UAQ 有效 |
| 直接增大学习率 2× | 26.66 | 严重退化 |

### 关键发现

- 7B模型INT8量化可加速20~30%，32B模型可加速70~90%（H100上），较大模型因矩阵乘法瓶颈更受益于量化
- 朴素INT8 RL在DAPO任务上奖励直接崩溃为0，ACR是稳定训练的关键
- UAQ的$s^2$信噪比改善在7B+DeepScaleR上将差距从1.61%缩小到0.92%

## 亮点与洞察

- 精准诊断了量化RL的两个核心问题：clipping失效和权重更新被淹没。特别是后者（权重更新量级$10^{-7}$ vs 量化误差$10^{-3}$~$10^{-1}$）是一个此前未被认识的根本性挑战。
- UAQ利用不变缩放同时"缩小分母、放大分子"获得$s^2$改善的设计非常巧妙，单一操作同时解决两个问题，且几乎零计算开销。

## 局限与展望

- 仅验证 INT8/FP8 两种精度，4-bit 量化可带来更大加速但挑战更大，未探索
- FP8 KV cache 量化在当前 vLLM 中未优化，实际加速效果受限
- 实验模型最大 32B，对更大模型（如 70B+）的适用性未验证

## 相关工作与启发

- **vs FlashRL**: FlashRL 提出 TIS+Decoupled PPO 用于量化 rollout，但在训练后期仍有性能差距；QuRL 的 ACR 解决了其长期训练崩溃问题
- **vs 标准 PTQ/QAT**: QuRL 处于 PTQ 和 QAT 之间的新位置——每轮一次性量化但隐式受梯度影响，需要专门设计的量化策略
- **vs GPTQ 等复杂 PTQ**: 虽然 GPTQ 可能捕获更精细变化，但每步重校准代价过高，不适合 RL 场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究量化对RL训练的影响，ACR和UAQ设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 覆盖PPO/GRPO/DAPO三种算法、多种模型规模和精度格式
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，失败模式可视化清晰
- 价值: ⭐⭐⭐⭐ 直接解决RL训练的核心效率瓶颈，实用性强

<!-- RELATED:START -->

## 相关论文

- [Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](entropy-preserving_reinforcement_learning.md)
- [ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)

<!-- RELATED:END -->
