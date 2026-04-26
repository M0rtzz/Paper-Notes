---
title: >-
  [论文解读] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning
description: >-
  [ICML 2025][自动驾驶][多智能体强化学习] 提出 R3DM 框架，通过最大化智能体角色、历史轨迹与未来预期行为之间的互信息，利用动力学模型驱动的内在奖励实现角色多样性与协调性的平衡，在 SMAC/SMACv2 环境中将胜率提升最高 20%。
tags:
  - ICML 2025
  - 自动驾驶
  - 多智能体强化学习
  - 角色发现
  - 动力学模型
  - 对比学习
  - 内在奖励
---

# R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2505.24265](https://arxiv.org/abs/2505.24265)  
**代码**: [有](https://github.com/UTAustin-SwarmLab/R3DM)  
**领域**: 自动驾驶  
**关键词**: 多智能体强化学习, 角色发现, 动力学模型, 对比学习, 内在奖励

## 一句话总结

提出 R3DM 框架，通过最大化智能体角色、历史轨迹与未来预期行为之间的互信息，利用动力学模型驱动的内在奖励实现角色多样性与协调性的平衡，在 SMAC/SMACv2 环境中将胜率提升最高 20%。

## 研究背景与动机

多智能体强化学习（MARL）在交通控制、自动驾驶、协作机器人等领域取得重要进展。现有方法主要面临以下矛盾：

**共享参数 vs 行为多样性**：CTDE 范式（如 QMIX、MAPPO）通过共享策略参数提高样本效率，但阻碍了个体智能体学习差异化行为

**多样性 vs 协调性**：多样性驱动的方法（如 CDS）虽然鼓励个体差异，但往往牺牲团队协调效果

**现有角色方法的局限**：ROMA、RODE、ACORM 等角色方法仅从智能体的**过去经验**推导角色，忽略了角色对**未来行为**的影响

**核心矛盾举例**：在消防无人机场景中，如果两架无人机初始观测相似，基于历史的角色推断会使它们获得相同角色，导致都飞向同一个火点，无法有效分工。

**本文核心 idea**：智能体的角色应当塑造其未来行为——采取不同角色的智能体在任意时刻之后应自然展现出不同的轨迹。因此需要将角色与未来预期行为通过动力学模型建立联系。

## 方法详解

### 整体框架

R3DM 在 CTDE 框架下提出基于信息论的目标函数，最大化智能体角色 $m_i^t$、观测-动作历史 $\tau_i^t$ 与未来轨迹 $\tau_i^{t+1:t+k}$ 之间的互信息。通过 Theorem 4.1 将该不可解目标分解为两个可优化的子目标：

$$I(\tau_i^{t+k}; m_i^t) \geq \mathbb{E}_{e_i^t, z_i^t, m_i^t}\left[\log\frac{p(z_i^t \mid e_i^t)}{p(z_i^t)}\right] + I(\tau_i^{t+1:t+k}; z_i^t)$$

- **第一项**：从历史中学习中间角色嵌入 → 对比学习优化
- **第二项**：确保角色嵌入能引导未来行为多样性 → 内在奖励优化

### 关键设计

1. **对比学习角色嵌入（优化第一项）**: 用轨迹编码器 $f_{\theta_e}$ 将观测-动作历史编码为嵌入 $e_i^t$，再通过角色编码器 $f_{\theta_r}$ 获得角色嵌入 $z_i^t$。通过 K-means 将智能体嵌入聚类为 $|M|$ 个角色组，同组为正例、跨组为负例，用双线性打分函数 $g(z_i^t, e_i^t)$ 计算相似度。核心公式（Theorem 4.2）：

   $$\mathbb{E}\left[\log\frac{p(z_i^t | e_i^t)}{p(z_i^t)}\right] \geq \log|M| + \mathbb{E}\left[\log\frac{g(z_i^t, e_i^t)}{g(z_i^t, e_i^t) + \sum_{m_i^{t*}} g(z_i^t, e_i^{t*})}\right]$$

   设计动机：复用 ACORM 的成熟对比学习框架获得中间角色表示，作为后续内在奖励的基础。

2. **策略内在奖励（Policy Intrinsic Reward）**: 通过 Theorem 4.3 将未来轨迹-角色互信息分解为策略项和动力学项。策略内在奖励衡量角色对动作选择的影响：

   $$r_{i,\text{pol}}^t = \sum_{l=t}^{t+k-1} \mathbb{D}_{KL}\left(\text{SoftMax}(Q_i(\cdot|\tau_i^l, z_i^t; \phi_Q)) \| p(\cdot|\tau_i^l)\right)$$

   其中 $p(\cdot|\tau_i^l) = \mathbb{E}_{z_i^t}[\text{SoftMax}(Q_i(\cdot|\tau_i^l, z_i^t; \phi_Q))]$ 是所有角色下的平均动作概率。该 KL 散度鼓励不同角色产生差异化的策略分布。

3. **动力学内在奖励（Dynamics Intrinsic Reward）**: 学习两个 DreamerV3 风格的 RSSM 世界模型——角色条件模型 $q_\psi(o_i^{l+1}|\tau_i^l, z_i^t, a_i^l)$ 和角色无关模型 $p(o_i^{l+1}|\tau_i^l, a_i^l)$。RSSM 包含序列模型、观测编码器、动力学预测器和观测解码器四个组件。动力学内在奖励为两模型的对数似然之差：

   $$r_{i,\text{dyn}}^t = \sum_{l=t}^{t+k-1}\left(\beta_1[\log q_{\psi_{\text{dec}}}(\cdot) + \beta_2 \log q_{\psi_{\text{dyn}}}(\cdot)] - [\text{role-agnostic terms}]\right)$$

   设计动机：当角色条件模型的预测显著优于角色无关模型时，说明角色嵌入确实对未来轨迹有预测力。$\beta_1$ 平衡跨角色轨迹多样性与角色-轨迹一致性。

### 损失函数 / 训练策略

总内在奖励：$r_{\text{int}}^t = \sum_{i \in I} \beta_3 r_{i,\text{pol}}^t + r_{i,\text{dyn}}^t$

最终 TD 学习目标：

$$\mathcal{L}_{TD}(\theta) = \left[r^t + \alpha r_{\text{int}}^t + \gamma \max_{a^{t+1}} Q_{\text{tot}}(s^{t+1}, a^{t+1}; \phi^{-}) - Q_{\text{tot}}(s^t, a^t; \phi)\right]^2$$

其中 $\alpha$ 平衡任务奖励与内在奖励，$\phi^{-}$ 为冻结目标网络参数。默认想象步长 $k=1$，$\epsilon$-greedy 探索从 1.0 线性衰减到 0.02。

## 实验关键数据

### 主实验

在 SMAC（6 个 hard/super-hard 地图）和 SMACv2（6 个环境）上评测，与 QMIX、CDS、EMC、CIA、GoMARL、ACORM 比较。

| 场景 | 指标 | R3DM | ACORM (SOTA) | 提升 |
|--------|------|------|----------|------|
| 3s5z_vs_3s6z (SMAC) | 测试胜率 | ~55% | ~35% | +20% |
| Corridor (SMAC) | 测试胜率 | ~90% | ~80% | +10% |
| 6h_vs_8z (SMAC) | 测试胜率 | ~30% | ~20% | +10% |
| protoss_10_vs_11 (SMACv2) | 测试胜率 | 最优 | 次优 | 边际 |
| protoss_5_vs_5 (SMACv2) | 累积奖励 | 最优 | 接近 | 策略更高效 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| k=1（默认） | 最佳胜率 | 单步想象即可 |
| k=10 | 性能明显下降 | 局部观测世界模型累积误差 |
| \|M\|=3 | 收敛最快 | 平衡协调与专业化 |
| \|M\|=8 | 类似最终性能但收敛慢 | 过度特化 |
| 无对比学习 | 低于完整版但优于 ACORM | 内在奖励是核心贡献 |
| 无内在奖励（=ACORM） | 基线水平 | 证实动力学奖励有效 |

### 关键发现

1. 内在奖励是核心：去掉对比学习后仍优于 ACORM，但去掉内在奖励则退化为 ACORM
2. 短 horizon 预测更好：基于局部观测的世界模型在多步预测时误差累积严重
3. 定性分析（3s_vs_5z）：R3DM 的一个 stalker 学会"诱敌"角色，引开 3 个 zealot，主力分两队歼灭弱化敌军；ACORM 全员冲锋最终落败
4. SMACv2 中 R3DM 累积奖励优势明显，即使胜率相近也学到了更高效的获胜策略

## 亮点与洞察

- 核心 insight 简洁有力：角色应塑造未来行为，而非仅从过去推断——直击现有方法的根本局限
- 将 DreamerV3 世界模型引入 MARL 角色学习是新颖的跨领域组合
- 信息论推导严谨，从 MI 目标到可操作下界到具体奖励设计，每步都有定理支撑
- 定性分析（战术演示）非常有说服力，直观展示了角色分化带来的战术协调优势

## 局限与展望

1. **角色数量需预设**：$|M|$ 是超参数，未来可探索从 replay buffer 动态推导
2. **世界模型基于局部观测**：仅用 ego agent 观测建模，未考虑其他智能体的动作/角色影响
3. **仅在 SMAC 类环境验证**：未在连续动作空间或真实自动驾驶场景中测试
4. **计算开销**：需训练两个 RSSM 模型（角色条件 + 角色无关），增加计算和内存负担

## 相关工作与启发

- **vs ACORM**：R3DM 在 ACORM 基础上增加动力学内在奖励，3s5z_vs_3s6z 胜率从 35% 提升到 55%
- **vs CDS**：CDS 过度强调个体多样性损害协调，R3DM 通过角色约束的多样性避免此问题
- **vs MAVEN**：MAVEN 用隐变量促进探索但不学习显式角色，R3DM 更可解释
- **启发**：世界模型 + 内在奖励的组合可扩展到更多 MARL 场景，未来可用全局世界模型提升精度

## 评分

- 新颖性: ⭐⭐⭐⭐ 世界模型引入 MARL 角色学习是创新组合，MI 分解严谨
- 实验充分度: ⭐⭐⭐⭐ SMAC/SMACv2 全面评测含消融和定性分析，但环境类型单一
- 写作质量: ⭐⭐⭐⭐ 定理推导清晰，消防无人机例子贯穿全文，直观易懂
- 价值: ⭐⭐⭐⭐ 为 MARL 角色学习提供连接历史与未来行为的新范式

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[ICML 2025\] GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction](goirl_graph-oriented_inverse_reinforcement_learning_for_multimodal_trajectory_pr.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[CVPR 2025\] Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels](../../CVPR2025/autonomous_driving/learning_to_detect_objects_from_multi-agent_lidar_scans_without_manual_labels.md)

<!-- RELATED:END -->
