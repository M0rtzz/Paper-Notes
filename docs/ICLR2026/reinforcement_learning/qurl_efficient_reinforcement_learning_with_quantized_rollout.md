---
title: >-
  [论文解读] QuRL: Efficient Reinforcement Learning with Quantized Rollout
description: >-
  [ICLR 2026][强化学习][量化推理] 提出 QuRL 方法，通过量化 actor 模型加速 RL 训练中的 rollout 阶段，设计自适应裁剪范围（ACR）解决量化导致的训练崩溃问题，并提出更新感知量化（UAQ）解决权重更新远小于量化误差的尺度失配问题…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "量化推理"
  - "RL加速"
  - "PPO"
  - "GRPO"
  - "importance sampling"
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

QuRL 把 RL 训练循环里的角色拆成两套精度：每轮先把旧 actor $\theta_{\text{old}}$ 量化成 $\hat{\theta}_{\text{old}}$，只用它跑 rollout 生成响应（吃掉约 70% 训练时间的瓶颈环节）；而 clipping ratio 和梯度更新仍交给全精度的 $\theta_{\text{old}}$。在此 Decoupled PPO 骨架上，ACR 负责修正量化带来的裁剪偏差，UAQ 则在训练前一次性预处理权重，让全精度更新不再被量化误差淹没。

### 关键设计

**1. 自适应裁剪范围 (Adaptive Clipping Range, ACR)：止住量化 rollout 引发的后期训练崩溃**

Decoupled PPO 把行为策略（量化 actor $\pi_{\hat{\theta}_{\text{old}}}$）与近邻策略（全精度 $\pi_{\theta_{\text{old}}}$）分开，FlashRL 的 TIS 用截断 $\min(\pi_{\theta_{\text{prox}}}/\pi_{\theta_{\text{behav}}}, C)$ 来稳定二者的重要性采样。问题在于这一截断隐含了缩放因子 $r_{i,t}$，会把正 advantage 序列过度裁剪。我们观察到训练后期（>1000 步）量化 actor 与全精度 actor 的 KL 散度从 0.002 涨到 0.025（12×），固定截断越往后偏差越大，最终把梯度估计带崩。ACR 的做法是把裁剪上界从固定的 $1+\epsilon$ 改成随策略分歧动态放宽的 $(1+\epsilon)/r_{i,t}$，目标函数写作 $\mathcal{J}_{\text{ACR}} = \tilde{\mathbb{E}}[\min(\pi_{\text{prox}}/\pi_{\text{behav}}, C) \cdot \min(R_{i,t}A_{i,t}, \text{clip}(R_{i,t}, 1-\epsilon, (1+\epsilon)/r_{i,t})A_{i,t})]$。分歧越大上界越松，更多正 advantage token 因此能继续贡献梯度，而不是被一刀切掉——这正是朴素 INT8 RL 在 DAPO 上奖励直接崩为 0 而 ACR 能稳住的原因。

**2. 更新感知量化 (Update-Aware Quantization, UAQ)：让微小的权重更新别被量化误差吞掉**

RL 的信任域约束让每步权重更新量级只有 $\sim 10^{-7}$（学习率 $\alpha \sim 10^{-6}$、梯度 $G \sim 0.1$–$1.0$），而量化误差对应的权重范数却在 $0.001$–$0.1$ 量级，于是 INT8 量化几乎抹掉了所有更新，量化模型实质上在"冻结"训练。UAQ 借线性层的不变缩放恒等式 $WX = (W/s) \cdot (sX)$ 来破局：取 $s > 1$ 后，量化误差 $\propto |\theta|/(s \cdot 2^b)$ 被压低 $s$ 倍，而被搬到激活侧、再经反量化还原的权重更新 $\propto s \cdot \alpha G$ 被放大 $s$ 倍，一进一出换来 $s^2$ 的信噪比改善。具体实现上 $s$ 按列作用于 $W$、按行作用于前一层激活（可直接折进 LayerNorm），所以它只是 RL 训练前的一次性权重预处理，不带来任何训练时开销。这也解释了为什么单纯把学习率放大 1.5×/2× 远不如 UAQ——前者同时放大了噪声，后者才真正改善了信噪比。

**3. 系统集成与工程实现：把 ACR 与 UAQ 接进现成 RL 框架且几乎零额外开销**

QuRL 集成在 VeRL 训练框架上，rollout 直接调用 vLLM 的 INT8/FP8 矩阵乘法核加速。两个改动都刻意做轻：UAQ 是训练前执行的一次性预处理，ACR 只改动 clipping 逻辑、计算量可忽略。这种轻量化是被 QuRL 的定位逼出来的——它介于 PTQ 和 QAT 之间：不像 QAT 显式优化量化目标，但参数又通过量化模型产出的梯度被隐式更新，因此需要的是简单且对 RL 友好的量化策略，而非 GPTQ 那种每步重校准、代价过高的精细方案。

### 损失函数 / 训练策略

采用 GRPO/DAPO 目标函数的 decoupled 变体，配通道级权重量化 + token 级激活量化（INT8 或 FP8）。UAQ 缩放因子取 $s = 1.5$（消融显示 $s=2.0$ 反而因过大失稳），TIS 截断阈值 $C$ 沿用 FlashRL 的设置。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] EchoRL: Reinforcement Learning via Rollout Echoing](../../ICML2026/reinforcement_learning/echorl_reinforcement_learning_via_rollout_echoing.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICML 2026\] DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](../../ICML2026/reinforcement_learning/darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)

</div>

<!-- RELATED:END -->
