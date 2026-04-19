---
title: >-
  [论文解读] Offline Reinforcement Learning with Generative Trajectory Policies
description: >-
  [ICLR 2026][图像生成][离线强化学习] 提出生成轨迹策略（GTP），通过统一视角将扩散模型、流匹配、一致性模型视为ODE解映射的特例，学习完整的连续时间轨迹解映射，并引入分数近似和优势权重两个适配技术，在D4RL基准上实现SOTA。
tags:
  - ICLR 2026
  - 图像生成
  - 离线强化学习
  - 生成式策略
  - ODE轨迹
  - 一致性模型
  - 流匹配
---

# Offline Reinforcement Learning with Generative Trajectory Policies

**会议**: ICLR 2026  
**arXiv**: [2510.11499](https://arxiv.org/abs/2510.11499)  
**代码**: 无（计划开源）  
**领域**: image_generation  
**关键词**: 离线强化学习, 生成式策略, ODE轨迹, 一致性模型, 流匹配

## 一句话总结

提出生成轨迹策略（GTP），通过统一视角将扩散模型、流匹配、一致性模型视为ODE解映射的特例，学习完整的连续时间轨迹解映射，并引入分数近似和优势权重两个适配技术，在D4RL基准上实现SOTA。

## 研究背景与动机

1. **领域现状**: 离线RL中生成模型因能捕获复杂多模态行为分布而成为强大的策略类。扩散策略在表达力方面表现优异，一致性策略在推理效率上有优势。

2. **现有痛点**: 扩散策略需要迭代去噪过程，推理代价高昂；一致性策略通过1-2步生成虽快但性能常退化。两者形成表达力与效率的根本矛盾。

3. **核心矛盾**: 现有方法各自独立发展，缺乏统一视角来理解和超越各自的局限。是否可能设计一种既表达力强又计算高效的策略类？

4. **本文目标**: 打破生成式策略中表达力与效率的权衡，设计一种灵活的多步生成策略，即使少量采样步骤也能实现高性能。

5. **切入角度**: 发现扩散模型、流匹配、一致性模型等都可视为学习连续时间ODE轨迹 $\frac{d\bm{x}_t}{dt} = f(\bm{x}_t, t)$ 的解映射 $\Phi(\bm{x}_t, t, s)$，以此统一视角设计新策略。

6. **核心 idea**: 将策略表示为ODE解映射（flow map），结合高效的分数近似和优势加权目标，实现兼具表达力和效率的离线RL策略。

## 方法详解

### 整体框架

GTP是一个actor-critic框架。Actor是生成轨迹策略 $\Phi_\theta(s, a_t, t, \tau)$，学习将噪声动作映射到干净动作的解映射。Critic是标准双Q网络 $Q_\varphi$。训练通过两个互补目标（瞬时流损失 + 轨迹一致性损失）优化Actor，并通过优势加权实现策略提升。推理时从高斯噪声出发，经多步迭代得到最终动作。

### 关键设计

**1. 统一ODE轨迹框架**

- **功能**: 提供理论基础，统一理解扩散、流匹配、一致性模型
- **核心思路**: 定义理想流映射 $\Phi(\bm{x}_t, t, s) = \bm{x}_t + \int_t^s f(\bm{x}_\tau,\tau)d\tau$ 及其重参数化形式 $\phi(\bm{x}_t, t, s)$。通过两个互补目标训练：瞬时流损失（局部正确性，对应扩散去噪器和流匹配速度场）和轨迹一致性损失（全局一致性，$\Phi(\bm{x}_t,t,s) \approx \Phi(\Phi(\bm{x}_t,t,u),u,s)$）
- **设计动机**: 现有模型各为ODE学习的特例，统一视角提供更清晰的策略设计空间

**2. 高效稳定的分数近似**

- **功能**: 解决ODE积分的计算代价和自引用监督的训练不稳定性
- **核心思路**: 用封闭形式的代理 $\tilde{f}(\bm{x}_t,t) = (\bm{x}_t - \bm{x})/t$ 替代需要多步积分的真实分数 $f^*(\bm{x}_t,t)$。定理1证明此替代引起的目标误差为 $O(h^p)$（$p$为求解器阶数），随步长趋于零而消失
- **设计动机**: 直接用自身不准确的早期估计做监督形成恶性循环（类似TD学习的自举问题），锚定到离线数据的解析信号消除了误差传播

**3. 优势加权的价值驱动引导**

- **功能**: 将生成模型从行为克隆提升为策略改进
- **核心思路**: 定理2证明KL正则化策略优化的最优解为 $\pi^*(a|s) \propto \pi_{BC}(a|s)\exp(\eta A(s,a))$。在生成损失中加入指数优势权重 $w(s,a) = \exp(\eta \cdot \max(0, A(s,a))/(\text{std}(A)+\epsilon))$，优先模仿高优势动作
- **设计动机**: 纯生成目标只是行为克隆，无法实现策略改进。优势加权是理论上正确的价值引导方式

### 损失函数 / 训练策略

Actor总损失：$\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{Consistency}} + \lambda_{\text{Flow}} \cdot \mathcal{L}_{\text{Flow}}$

- 轨迹一致性损失：$\mathcal{L}_{\text{Consistency}} = \mathbb{E}[w(s,a)\|\Phi_\theta(s,a_t,t,\tau) - \Phi_{\theta^-}(s,\tilde{a}_u,u,\tau)\|_2^2]$
- 瞬时流损失：$\mathcal{L}_{\text{Flow}} = \mathbb{E}[w(s,a)\|a - \phi_\theta^{\text{inst}}(s,a_t,t)\|_2^2]$

Critic使用标准双Q网络的TD误差损失，目标网络通过EMA更新。

## 实验关键数据

### 主实验

D4RL行为克隆（BC）性能对比，GTP-BC使用5步采样：

| 任务 | Diffusion-BC | Consistency-BC | GTP-BC (本文) |
|------|-------------|----------------|---------------|
| Gym平均 | 76.3 | 69.7 | **82.3** |
| AntMaze平均 | 41.7 | 44.1 | **66.3** |
| halfcheetah-mr | 41.7 | 34.4 | **46.3** |
| hopper-mr | 67.3 | 99.7 | **100.5** |

D4RL离线RL性能对比（基于完整actor-critic框架）：

| 任务 | IDQL | DIPO | D-QL | C-QL | GTP (本文) |
|------|------|------|------|------|------------|
| AntMaze-large-diverse | 47.5 | — | 47.3 | 51.0 | **100.0** |
| AntMaze-medium-diverse | — | — | — | — | **100.0** |

### 消融实验

| 配置 | Gym平均 | AntMaze平均 | 说明 |
|------|---------|-------------|------|
| GTP完整模型 | 最佳 | 100.0 | 两个损失+优势加权+分数近似 |
| 去除轨迹一致性损失 | 下降 | 下降 | 全局一致性对长时任务关键 |
| 去除瞬时流损失 | 下降 | 下降 | 局部动力学锚定不可或缺 |
| 使用真实分数(ODE积分) | 训练不稳定 | 差 | 验证分数近似的必要性 |
| 去除优势加权 | 退化为BC | 退化 | 无策略改进能力 |

### 关键发现

- GTP在多个AntMaze难题上**首次达到满分**（100.0），显著超越所有先前方法
- 在BC设定下GTP-BC已经大幅优于扩散-BC和一致性-BC，证明了轨迹策略本身的强表达力
- 分数近似使训练加速且更稳定，理论误差界 $O(h^p)$ 得到实验验证
- 5步采样即可达到优异性能，展示了效率与质量的良好平衡

## 亮点与洞察

- **统一视角的理论贡献**: 将扩散、流匹配、一致性模型纳入同一ODE框架，为策略设计提供清晰的设计空间
- **双目标的互补性设计**: 瞬时流损失保证局部准确性，轨迹一致性损失保证全局一致性
- **分数近似的优雅**: 用简单的封闭形式替代复杂的ODE积分，有理论保证且效果更好
- **AntMaze满分**: 在离线RL社区中具有标志性意义

## 局限与展望

- 主要在D4RL标准基准上验证，未测试更复杂的真实世界任务
- ODE框架的统一视角虽然优美，但实际采样步骤的最优选择缺乏理论指导
- 可探索GTP在在线RL和model-based RL中的应用
- 可与近期的token-level生成策略方法结合

## 相关工作与启发

- Consistency Trajectory Models (CTM) 提供了学习ODE解映射的基础，GTP将其扩展到RL领域
- 优势加权与AWR/AWAC等方法一脉相承，但在生成模型框架下有新的理论解释
- 启发: 统一不同生成模型的理论视角可以产生超越各个方法的新方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一ODE轨迹视角+两个理论原则性适配
- 实验充分度: ⭐⭐⭐⭐ D4RL全面评估，消融充分，AntMaze满分引人注目
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，框架图清晰
- 价值: ⭐⭐⭐⭐⭐ 对离线RL中的生成式策略研究有深远影响

<!-- RELATED:START -->

## 相关论文

- [Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)

<!-- RELATED:END -->
