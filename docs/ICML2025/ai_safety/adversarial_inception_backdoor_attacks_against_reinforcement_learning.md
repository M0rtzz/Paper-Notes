---
title: >-
  [论文解读] Adversarial Inception Backdoor Attacks against Reinforcement Learning
description: >-
  [ICML 2025][AI安全][backdoor attack] 提出"inception"后门攻击框架——通过在 RL 智能体的训练轨迹中插入触发器并将高回报动作替换为目标对抗动作，首次在严格奖励约束下实现 100% 攻击成功率，同时保持智能体在正常任务上的表现。
tags:
  - "ICML 2025"
  - "AI安全"
  - "backdoor attack"
  - "强化学习"
  - "数据投毒"
  - "动作操纵"
  - "奖励约束"
---

# Adversarial Inception Backdoor Attacks against Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2410.13995](https://arxiv.org/abs/2410.13995)  
**代码**: [https://github.com/ethanrathbun/Q-Incept](https://github.com/ethanrathbun/Q-Incept)  
**领域**: AI安全  
**关键词**: backdoor attack, 强化学习, 数据投毒, 动作操纵, 奖励约束

## 一句话总结
提出"inception"后门攻击框架——通过在 RL 智能体的训练轨迹中插入触发器并将高回报动作替换为目标对抗动作，首次在严格奖励约束下实现 100% 攻击成功率，同时保持智能体在正常任务上的表现。

## 研究背景与动机

**领域现状**：DRL 在安全关键领域（自动驾驶、网络防御、机器人）的广泛应用使其成为对抗攻击的目标。后门攻击在训练阶段操纵智能体，使其在部署时遇到特定触发器时执行预定义的对抗行为。

**现有痛点**：现有后门攻击（TrojDRL、SleeperNets）假设攻击者可以任意控制奖励信号——注入远超环境自然范围的极端奖励值。这导致：(a) 奖励裁剪/归一化即可使攻击失效；(b) 异常大的奖励值易被简单的规则检测器发现。

**核心矛盾**：当奖励被约束在环境自然范围 $[\inf[R], \sup[R]]$ 内时，攻击者无法仅通过奖励操纵使对抗动作的期望回报超过最优动作——特别是当最优动作的累积回报很高时（如接近 $\gamma/(1-\gamma)$），需要无界奖励才能抵消。

**本文目标**：如何在严格奖励约束下实现有效的 RL 后门攻击？

**切入角度**：不操纵奖励值本身，而是操纵训练数据中的动作——将高回报时刻的最优动作替换为目标对抗动作，让智能体"曾经以为"对抗动作带来了高回报。

**核心 idea**：在训练轨迹中"植入记忆"（inception）——让智能体相信目标动作是高回报的，因为它在历史数据中看到了目标动作与高回报的关联。

## 方法详解

### 整体框架
攻击者在训练过程中：
1. 观察智能体生成的完整轨迹 $H = \{(s, a, r)_t\}$
2. 以概率 $\beta$ 选择时间步注入触发器 $\delta(s_t)$ 到观察中
3. **关键创新**：用 DQN 估计各时间步的 Q 值，选择高回报时间步，将其动作替换为目标动作 $a^+$
4. 修改后的轨迹被智能体用于策略优化

### 关键设计

1. **Inception 动作操纵（区别于 Forced Action Manipulation）**:

    - 功能：将历史高回报时刻的动作替换为目标对抗动作
    - 核心思路：选择 $Q(s_t, a_t)$ 最高的时间步，将 $a_t$ 替换为 $a^+$，使智能体在训练时"看到"$a^+$ 带来了高回报
    - 设计动机：与 TrojDRL 的"强制动作操纵"不同——强制操纵只是增加探索，不改变动作价值；而 inception 通过替换数据中的动作直接改变 Q 值估计，使 $Q(s_p, a^+)$ 偏高
    - 形式化证明：对任意 MDP，inception 攻击都能使毒策略 $\pi^+$ 在被毒状态中选择 $a^+$

2. **Q-Incept 在线攻击**:

    - 功能：基于 DQN 估计动态选择最佳投毒时间步
    - 核心思路：维护一个辅助 DQN $Q_\theta$ 估计当前策略的动作价值，选择 $Q_\theta(s_t, a_t)$ 最高的 $\lfloor \beta \cdot |\text{episode}| \rfloor$ 个时间步进行 inception 操纵
    - 设计动机：比随机选择时间步更有效率——选择高价值时刻极大化对抗动作的期望回报

3. **奖励约束遵守**:

    - 功能：确保所有注入的奖励在环境自然范围内
    - 核心思路：被毒状态的奖励保持不变（$R'(\delta(s), a, s') = R(s, a, s')$），无需注入异常奖励
    - 设计动机：因为 inception 通过替换动作而非修改奖励来达到目的，天然满足奖励约束

### 损失函数 / 训练策略
- 攻击者使用标准 DQN 训练辅助 Q 网络来估计动作价值
- 投毒率 $\beta$ 控制攻击强度和隐蔽性的权衡
- 攻击在 outer-loop 威胁模型下操作（在 episode 完成后修改轨迹数据）

## 实验关键数据

### 主实验
在奖励约束下的攻击成功率（ASR）：

| 环境 | Q-Incept ASR | SleeperNets ASR | TrojDRL ASR | Q-Incept 任务性能 |
|------|-------------|-----------------|-------------|-----------------|
| Atari Q*bert | 100% | ~20% | ~15% | ≈无损 |
| CybORG (网络防御) | 100% | <50% | <30% | 最小影响 |
| Highway (自动驾驶) | 100% | 失败 | 失败 | ≈无损 |
| Safety-Gym | 100% | 失败 | 失败 | ≈无损 |

### 消融实验

| 配置 | ASR | 说明 |
|------|-----|------|
| Q-Incept (β=0.1) | 100% | 低投毒率仍有效 |
| Q-Incept (β=0.05) | ~95% | 进一步降低仍然较高 |
| 随机时间步选择 | ~60% | 不如 Q 值引导的选择 |
| 无 inception（仅触发器+奖励） | ~20% | 验证了动作操纵的必要性 |

### 关键发现
- 在奖励裁剪到 [0,1] 后，SleeperNets 和 TrojDRL 攻击成功率从 ~100% 骤降至 <50%
- Q-Incept 在所有测试环境中均保持 100% ASR，即使在严格奖励约束下
- 投毒率 β 对隐蔽性的影响很小——智能体在正常任务上的表现几乎不受影响

## 亮点与洞察
- **从奖励操纵转向动作操纵**的范式转换极其巧妙——绕过了奖励约束这一自然防御机制
- "inception"命名很精准——像电影《盗梦空间》一样在训练数据中"植入虚假记忆"
- 形式化证明了先前方法在奖励约束下必然失败，为新方法提供了理论动机
- 揭示了 RL 系统中一个被忽视的安全隐患：奖励裁剪/归一化并非万能防御

## 局限与展望
- 仅考虑离散动作空间中的目标攻击（固定 $a^+$），连续动作空间更具挑战
- outer-loop 威胁模型假设攻击者可访问完整训练轨迹，实际中可能有限制
- 未提出有效的防御方法——仅分析了攻击能力
- Q 值估计的质量取决于辅助 DQN，训练初期估计可能不准

## 相关工作与启发
-  **vs TrojDRL/SleeperNets**: 它们依赖无约束奖励操纵，裁剪后攻击失效；Q-Incept 通过动作操纵绕过此限制
- **vs 测试时攻击**: 测试时攻击直接改变部署时的动作/观察，而 Q-Incept 是训练时攻击，更难检测
- 对安全 RL 系统的设计有重要启示：不应仅依赖奖励范围检查作为安全保障

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在奖励约束下实现有效 RL 后门攻击，范式转换
- 实验充分度: ⭐⭐⭐⭐ 四种不同领域环境，对比充分
- 写作质量: ⭐⭐⭐⭐ 理论+实验结合好，定义清晰
- 价值: ⭐⭐⭐⭐⭐ 揭示了 RL 系统重要安全漏洞

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](../../ICLR2026/ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[ICML 2025\] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](../../CVPR2025/ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[CVPR 2025\] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](../../CVPR2025/ai_safety/invisible_backdoor_attack_against_self-supervised_learning.md)
- [\[ICML 2025\] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)

</div>

<!-- RELATED:END -->
