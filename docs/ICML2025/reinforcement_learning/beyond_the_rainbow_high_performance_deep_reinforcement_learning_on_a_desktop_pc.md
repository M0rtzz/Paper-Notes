---
title: >-
  [论文解读] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC
description: >-
  [ICML 2025][深度强化学习] 提出 BTR（Beyond The Rainbow）——整合 6 项 RL 改进到 Rainbow DQN 中，在单台桌面 PC 上 12 小时内训练 Atari-60 达到 IQM 7.4（Rainbow 为 1.9），并首次成功训练智能体玩马里奥银河、马里奥赛车和真人快打等 3D 游戏。
tags:
  - ICML 2025
  - 深度强化学习
  - Rainbow DQN
  - Atari
  - 计算效率
  - 3D游戏
---

# Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC

**会议**: ICML 2025  
**arXiv**: [2411.03820](https://arxiv.org/abs/2411.03820)  
**代码**: https://github.com/VIPTankz/BTR  
**领域**: 强化学习  
**关键词**: 深度强化学习, Rainbow DQN, Atari, 计算效率, 3D游戏

## 一句话总结
提出 BTR（Beyond The Rainbow）——整合 6 项 RL 改进到 Rainbow DQN 中，在单台桌面 PC 上 12 小时内训练 Atari-60 达到 IQM 7.4（Rainbow 为 1.9），并首次成功训练智能体玩马里奥银河、马里奥赛车和真人快打等 3D 游戏。

## 研究背景与动机

**领域现状**：SOTA RL 算法（MuZero、Agent57）需要分布式计算集群和数周训练时间，远超小型实验室和爱好者的能力。

**现有痛点**：Rainbow DQN 需要 34,200 GPU 小时（1435 天），新算法的计算需求更高——RL 不像 NLP/CV 有基础模型可微调，每个环境都需从头训练。

**核心矛盾**：高性能与计算可及性的矛盾。

**本文目标**：设计一个在桌面 PC 上就能快速训练的高性能 RL 算法。

**切入角度**：延续 Rainbow DQN 的方法论——精选独立改进组件并组合。

**核心 idea**：选择 6 个兼顾性能和计算效率的改进，组合成 BTR。

## 方法详解

### 整体框架
BTR 在 Rainbow DQN 基础上替换/新增 6 个组件：
1. Distributional RL（保留）
2. N-step returns（保留）
3. Adam 优化器替代 RMSProp
4. 网络重置（Layer Reset）解决可塑性损失
5. 大批量训练（batch size 512）
6. 高更新-采样比（UTD ratio = 8）

### 关键设计

1. **网络重置（Periodic Layer Reset）**:

    - 功能：周期性重新初始化网络的最后几层
    - 核心思路：RL 训练中网络逐渐失去可塑性（dormant neurons 增加），重置恢复学习能力
    - 设计动机：代替增大网络的方法，保持计算效率

2. **高更新-采样比（High UTD）**:

    - 功能：每收集 1 步环境交互执行 8 次网络更新
    - 核心思路：提高样本效率，减少环境交互次数
    - 设计动机：在桌面 PC 上 GPU 计算比 CPU 环境模拟便宜

3. **2-hot 编码的分布式 RL**:

    - 功能：用更稳定的 2-hot 编码替代 C51 的概率质量表示
    - 核心思路：每个回报值分配给最近的两个 bin（类似插值），用 cross-entropy loss
    - 设计动机：比 C51 的 KL 散度损失更稳定

### 损失函数 / 训练策略
- 分布式 RL + 2-hot encoding + cross-entropy loss
- Adam 优化器，学习率 6.25e-5
- 环境帧跳过 4，灰度化 + 帧堆叠标准预处理

## 实验关键数据

### 主实验
Atari-60 benchmark（200M frames）：

| 算法 | IQM ↑ | 训练时间 | 设备 |
|------|-------|---------|------|
| DQN | 0.5 | ~35h | 桌面PC |
| Rainbow DQN | 1.9 | 35h | 桌面PC |
| Dreamer-v3 | 2.5 | - | GPU集群 |
| **BTR** | **7.4** | **<12h** | **桌面PC** |

超过人类水平：52/60 游戏

### 消融实验

| 去掉组件 | IQM变化 | 说明 |
|---------|---------|------|
| -Layer Reset | -2.3 | 最重要组件 |
| -High UTD | -1.8 | 样本效率关键 |
| -2-hot Distributional | -1.2 | 稳定性改进 |
| -Adam | -0.5 | 温和改进 |

### 关键发现
- BTR 在 3D 游戏上首次成功：马里奥银河（最终关卡）、马里奥赛车（彩虹之路）、真人快打
- 网络重置是最关键的单一改进——直接解决 RL 中的可塑性损失问题
- 高 UTD 比增加环境交互更有效率

## 亮点与洞察
- **"普惠 RL"的理念**——强调计算可及性，让小型实验室和爱好者也能做前沿 RL 研究
- 组件选择的原则值得借鉴：不选复杂的（世界模型/搜索），选简单高效的
- 3D 游戏的成功说明方法的泛化性，不局限于 Atari

## 局限与展望
- 仅适用于离散动作空间（DQN 家族限制）
- 未测试持续学习/多任务设置
- 3D 游戏结果仍是初步的（单环境、单关卡）

## 相关工作与启发
- **vs Rainbow DQN**: 同样的组合方法论，但组件更现代，性能 4× 提升
- **vs Dreamer-v3**: 世界模型方法，需更多计算但理论上更样本高效
- **对 RL 研究的启示**: 简单方法的组合可以胜过复杂方法

## 评分
- 新颖性: ⭐⭐⭐ 组合现有方法，无本质创新
- 实验充分度: ⭐⭐⭐⭐⭐ 60个游戏 + 3D游戏 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰实用
- 价值: ⭐⭐⭐⭐⭐ 使高性能RL对所有人可及

<!-- RELATED:START -->

## 相关论文

- [The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification](../../NeurIPS2025/reinforcement_learning/kimina_lean_server_a_high-performance_lean_server_for_large-scale_verification.md)
- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/reinforcement_learning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)
- [Confounding Robust Deep Reinforcement Learning: A Causal Approach](../../NeurIPS2025/reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)

<!-- RELATED:END -->
