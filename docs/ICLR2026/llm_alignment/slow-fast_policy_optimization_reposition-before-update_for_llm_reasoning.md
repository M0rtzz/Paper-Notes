---
title: >-
  [论文解读] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning
description: >-
  [ICLR 2026][LLM对齐][强化学习] 提出 SFPO（Slow-Fast Policy Optimization），通过将每个训练步分解为"快速轨迹—重定位—慢速校正"三阶段结构，在不修改目标函数和 rollout 过程的前提下即插即用地增强 GRPO 的稳定性和样本效率…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "强化学习"
  - "GRPO"
  - "策略优化"
  - "数学推理"
  - "样本效率"
---

# Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning

**会议**: ICLR 2026  
**arXiv**: [2510.04072](https://arxiv.org/abs/2510.04072)  
**代码**: [slow-fast-po.github.io](https://slow-fast-po.github.io/)  
**领域**: LLM对齐  
**关键词**: 强化学习, GRPO, 策略优化, 数学推理, 样本效率

## 一句话总结

提出 SFPO（Slow-Fast Policy Optimization），通过将每个训练步分解为"快速轨迹—重定位—慢速校正"三阶段结构，在不修改目标函数和 rollout 过程的前提下即插即用地增强 GRPO 的稳定性和样本效率，在数学推理基准上平均提升最高 2.80 分，rollout 减少最多 4.93 倍。

## 研究背景与动机

- 强化学习（RL）已成为提升 LLM 推理能力的核心手段，GRPO 是广泛使用的无 critic 策略梯度方法
- **GRPO 的局限性**：
    - 训练早期 rollout 质量差，随机奖励导致**高方差梯度**，更新不稳定
    - 每批 rollout 只做**单步更新**（one-shot），浪费了可进一步利用的梯度信息
    - 简单复用 rollout 数据会引入 off-policy 偏差，后期反而降低性能
- 需要一种能稳定梯度方向、提高样本利用率、同时控制分布偏移的更新机制

## 方法详解

### 整体框架

SFPO 将每个训练迭代分解为三个协调阶段：

1. **Stage I: Fast Trajectory（快速轨迹）** — 在同一批 rollout 上执行 $K$ 步内循环更新
2. **Stage II: Reposition（重定位）** — 插值回起点以控制 off-policy 漂移
3. **Stage III: Slow Correction（慢速校正）** — 在插值点处执行一步额外梯度更新

### Stage I: 快速轨迹

从参数 $\theta^{s,0}$ 出发，执行 $K$ 步内循环更新：

$$\theta^{s,k+1} = \theta^{s,k} - \eta \nabla_\theta \mathcal{L}(\theta^{s,k}), \quad k=0,\ldots,K-1$$

- 与单步更新不同，位移 $\theta^{s,K} - \theta^{s,0} = -\eta \sum_{k=0}^{K-1} \nabla_\theta \mathcal{L}(\theta^{s,k})$ 累积了 $K$ 个梯度的效果
- 在二阶近似下，这等价于一个**曲率感知的低通滤波器**：沿平坦方向稳步前进，沿高曲率方向自动抑制振荡

### Stage II: 重定位

受 Lookahead Optimizer 启发，将快速轨迹终点插值回起点：

$$\widetilde{\theta}^{s,K} = \theta^{s,0} + \alpha(\theta^{s,K} - \theta^{s,0}), \quad \alpha \in [0,1]$$

- $\alpha$ 控制 off-policy 漂移程度：较小的 $\alpha$ 隐含更强的近端正则化
- 等价于求解以 $\theta^{s,0}$ 为中心的线性化近端子问题，$\alpha$ 充当隐式信赖域半径

### Stage III: 慢速校正

在插值点处执行一步额外梯度更新：

$$\theta^{s+1} = \widetilde{\theta}^{s,K} - \eta \nabla_\theta \mathcal{L}(\widetilde{\theta}^{s,K})$$

形成 predictor-corrector 结构，确保最终更新对齐局部曲率。

### 统一更新公式

$$\theta^{s+1} = \theta^{s,0} - \eta \left[\alpha \sum_{k=0}^{K-1} \nabla_\theta \mathcal{L}(\theta^{s,k}) + \nabla_\theta \mathcal{L}(\widetilde{\theta}^{s,K})\right]$$

### 自适应 $\alpha$ 调度

- 通过监控策略熵 $H_s$ 的滚动 z-score $Z_s = (H_s - \mu_s) / \sigma_s$
- 当 $|Z_s| \geq \tau$ 时触发 $\alpha \to 0$，此后退化为标准 GRPO
- 早期利用快速轨迹加速收敛，后期回到纯 on-policy 更新保稳定性

### 损失函数

SFPO 不改变底层损失函数，直接使用 GRPO 目标：

$$\mathcal{J}_{GRPO}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}) - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$$

## 实验关键数据

### 主实验：数学推理基准（DAPO+Math 训练集）

| 模型 | 方法 | Math-500 | AIME24 | AIME25 | AMC | Minerva | Olympiad | Avg |
|------|------|----------|--------|--------|-----|---------|----------|-----|
| Qwen2.5-Math-1.5B | GRPO | 77.15 | 16.67 | 11.67 | 53.31 | 31.89 | 39.42 | 38.35 |
| | **SFPO** | **78.35** | **20.00** | **15.00** | **56.02** | **32.07** | **39.72** | **40.19** |
| DS-Qwen-1.5B | GRPO | 84.65 | 30.00 | 23.33 | 66.86 | 31.71 | 49.85 | 47.73 |
| | **SFPO** | **86.10** | **32.50** | **30.83** | **70.28** | **32.81** | **50.67** | **50.53** |
| DS-Qwen-7B | GRPO | 91.70 | 50.00 | 35.83 | 80.42 | 43.65 | 61.24 | 60.47 |
| | **SFPO** | **92.60** | **54.17** | **37.50** | **83.75** | **44.49** | **65.73** | **63.04** |

### 效率分析消融

| 模型 | Rollout 减少倍数 | 训练时间减少倍数 |
|------|-----------------|-----------------|
| DS-Qwen-1.5B | 3.21× | 2.62× |
| Qwen3-4B-Base | 3.50× | 2.65× |
| DS-Qwen-7B | **4.93×** | **4.19×** |

### 关键发现

1. SFPO 在所有 5 个模型、6 个基准上一致优于 GRPO，小模型增益最大（+2.80 on DS-Qwen-1.5B）
2. 训练动态分析表明 SFPO 避免了 GRPO 的响应长度崩塌问题
3. SFPO 不引入额外 GPU 显存开销，因为不需要存储额外优化器状态
4. 在更大训练集 Skywork-OR1（105K 数据）上同样保持一致增益

## 亮点与洞察

- **即插即用设计**：完全不改变损失函数、rollout 生成和正则化，可直接替换 GRPO 的更新步骤
- **理论直觉清晰**：快速轨迹=曲率感知低通滤波，重定位=隐式信赖域，慢速校正=对齐局部曲率
- **自适应退出机制**：基于熵监控的 $\alpha$ 调度在收敛阶段自动退化为 GRPO，兼顾效率与稳定性
- **显著的样本效率提升**：最高 4.93× 更少 rollout 达到相同精度

## 局限性

- 引入了 $K$、$\alpha_0$、$\omega$、$\tau$ 等额外超参数，虽然实验表明对超参选择不敏感
- 理论分析主要基于 L-smooth 假设下的近似推导，LLM 损失景观的实际性质更复杂
- 仅在数学推理任务上验证，尚未在代码生成、多模态推理等其他推理任务上测试

## 相关工作

- 策略梯度增强：DAPO、Dr.GRPO 等关注不同角度的 GRPO 改进
- Lookahead Optimizer：SFPO 的重定位机制受其启发
- 样本效率：ReMax、RLOO 等方法同样关注 rollout 利用率

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 快速-重定位-慢速的三阶段结构是新颖的策略优化范式
- **技术深度**: ⭐⭐⭐⭐ — 理论推导完整，从曲率分析到近端优化再到自适应调度
- **实验充分性**: ⭐⭐⭐⭐⭐ — 5 个模型、6 个基准、2 种训练集、效率与训练动态全面分析
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用、无额外显存、显著提速，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](../../ACL2025/llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)

</div>

<!-- RELATED:END -->
