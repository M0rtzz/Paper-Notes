---
title: >-
  [论文解读] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles
description: >-
  [NeurIPS 2025][自动驾驶][城市路由基准] 本文提出 URB——首个面向城市混合交通（人类+CAV）路由问题的大规模 MARL 基准环境，整合 29 个真实交通网络、微观交通仿真器 SUMO 和真实出行需求模式，实验发现当前 SOTA MARL 算法很难超越人类驾驶员的路由表现，揭示了该领域亟需算法突破。
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "城市路由基准"
  - "多智能体强化学习"
  - "交通仿真"
  - "博弈论"
---

# URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles

**会议**: NeurIPS 2025  
**arXiv**: [2505.17734](https://arxiv.org/abs/2505.17734)  
**代码**: [GitHub](https://github.com/COeXISTENCE-PROJECT/URB)  
**领域**: 自动驾驶 / 强化学习路径规划  
**关键词**: 城市路由基准, 多智能体强化学习, 自动驾驶, 交通仿真, 博弈论

## 一句话总结

本文提出 URB——首个面向城市混合交通（人类+CAV）路由问题的大规模 MARL 基准环境，整合 29 个真实交通网络、微观交通仿真器 SUMO 和真实出行需求模式，实验发现当前 SOTA MARL 算法很难超越人类驾驶员的路由表现，揭示了该领域亟需算法突破。

## 研究背景与动机

联网自动驾驶车辆（CAV）有望通过优化路由决策来缓解城市拥堵，但核心问题是用什么算法来做集体路由决策。强化学习（RL）可以自然地将路由问题建模为决策任务，但目前缺乏标准化、真实的基准来评测和比较不同算法。

现有研究的痛点包括：(1) 城市路由的巨大动作空间（路径数随网络规模指数增长）、非平稳性（智能体竞争有限资源）和高仿真成本；(2) 先前工作仅在简化场景下测试，缺乏统一评估；(3) 忽视了 CAV 部署对人类驾驶员的影响。

本文的切入角度是构建一个全面的基准框架，让 RL 研究者能在真实交通场景上测试算法，同时让交通研究者能用 SOTA RL 方法评估 CAV 部署的影响。核心发现是当前 MARL 算法在大规模城市路由问题上远未成熟，甚至难以超越随机策略。

## 方法详解

### 整体框架

URB 基于 Agent-Environment Cycle（AEC）博弈模型，将城市路由问题形式化为：人类驾驶员先学习并稳定路由策略 → 部分车辆转为 CAV → CAV 用 MARL 算法训练路由策略 → 策略测试。环境仿真使用微观交通仿真器 SUMO，人类行为使用通用学习模型（HLM），CAV 根据观察选择路由。

### 关键设计

1. **真实交通网络与需求**:

    - 包含 29 个真实交通网络：28 个法兰西岛子区域 + 1 个 Ingolstadt
    - 每个网络配有基于经验数据的合成出行需求模式（AM 高峰时段）
    - 网络规模从小型（St. Arnoult）到大型（Ingolstadt）覆盖不同复杂度

2. **灵活的参数化方案**:

    - CAV 市场份额（0-100%）可配置
    - CAV 行为画像：自私（最小化自身旅行时间）、合作（最小化群体时间）、利他、甚至恶意
    - 观察空间：CAV 能看到先出发智能体的路由选择
    - 动作空间：从预计算的 $K_i$ 条路径中选择

3. **人类行为建模**:

    - 使用经典的 day-to-day 路由学习模型（Gawron 1998）
    - 人类驾驶员每天根据最近经历更新期望通行时间，选择主观最优路径
    - 经过足够学习后趋近用户均衡（Nash 均衡的特例）

4. **综合评估指标体系**:

    - 核心指标：通行时间 $t$（$t^{pre}$, $t^{train}$, $t^{test}$, $t_{CAV}$, $t_{HDV}$）
    - 训练代价 $c_{all}$：衡量训练过程中对系统的负面影响
    - 变化分析：平均速度变化 $\Delta_V$ 和里程变化 $\Delta_L$
    - 胜率 $WR$：CAV 训练后旅行时间 < 人类基线的实验比例

### 损失函数 / 训练策略

基准实现了 4 种 MARL 算法：
- **IQL（独立 Q-Learning）**: 每个智能体独立训练 DQN
- **IPPO（独立 PPO）**: 每个智能体独立使用 PPO
- **MAPPO（多智能体 PPO）**: 集中式 critic，去中心化 actor
- **QMIX**: 通过混合网络分解联合 Q 函数

协作算法（MAPPO, QMIX）使用合作奖励（最小化群体旅行时间），独立算法（IQL, IPPO）使用自私奖励。基线包括全有或全无（AON）、随机和人类基线。

## 实验关键数据

### 主实验

**Scenario 1：40% CAV 市场份额（5 次重复平均）**：

| 算法 | St. Arnoult $t^{test}$ | Provins $t^{test}$ | Ingolstadt $t^{test}$ | St. Arnoult WR |
|------|----------------------|-------------------|-----------------------|---------------|
| 人类基线 | 3.15 | 2.80 | 4.21 | 100% |
| AON | 3.15 | **2.67** | **4.11** | 100% |
| 随机 | 3.38 | 2.93 | 4.40 | 0% |
| IPPO | 3.28 | **2.90** | 4.37 | 0% |
| IQL | 3.36 | 2.91 | 4.41 | 0% |
| MAPPO | 3.35 | 2.91 | 4.38 | 0% |
| QMIX | **3.24** | 2.94 | 4.47 | 80% |

### 消融实验

| 配置 | 说明 |
|------|------|
| QMIX 6000 episodes | 仅在最小网络 St. Arnoult 有 80% WR |
| QMIX 20000 episodes | 延长训练收益递减，在大网络仍表现差 |
| 不同 CAV 份额 | 更多 CAV 不一定带来更好系统表现 |
| 固定回合数 | MARL 训练收敛极慢，每个 episode 需完整 SUMO 仿真 |

### 关键发现
- **MARL 算法很少超越人类**：仅 QMIX 在最小的 St. Arnoult 网络上以 80% 的胜率超越人类，但在更大的网络上甚至不如随机策略
- **CAV 部署伤害人类驾驶员**：所有场景中 $t_{HDV} > t^{pre}$，说明 CAV 的存在增加了人类驾驶员的通行时间
- **训练代价巨大**：训练过程中系统效率持续下降（速度降低、里程增加）
- **简单的 AON 基线在部分场景表现最优**：暴露了 MARL 方法的不成熟

## 亮点与洞察

- **"负面结果"的价值**：勇于报告 MARL 算法表现不佳的结果，比展示虚假成功更有意义，为社区指明了真正需要突破的方向
- **跨学科设计**：将交通工程（SUMO 仿真）、运输工程（人类路由行为建模）和机器学习（MARL）有机结合
- **社会影响评估**：不仅评估 CAV 性能，还追踪 CAV 部署对人类驾驶员的影响，体现了负责任的 AI 研究
- **开放可扩展**：模块化设计允许社区自由添加新算法、网络和任务

## 局限与展望

- 仿真效率瓶颈：每个 episode 需完整 SUMO 仿真，大规模训练极其耗时
- 当前仅支持出行前路由决策，未考虑途中改路（en-route rerouting）
- 人类行为模型使用标准化模型，未捕捉异质性和非理性行为
- 当前仅使用离散路由选择，未探索连续路由优化
- 未考虑通信延迟和信息不完全等实际 CAV 部署约束

## 相关工作与启发

- **vs FLOW / RESCO**: 这些交通 RL 基准关注信号灯控制而非路由决策，URB 填补了城市路由 MARL 基准的空白
- **vs RouteRL**: URB 在 RouteRL 基础上扩展，增加了 29 个网络、系统评估指标和基准方法
- **对 MARL 研究的警示**: 强调了真实世界大规模非平稳多智能体问题与常见 MARL 基准的巨大差距

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个大规模城市路由 MARL 基准，问题定义有创新性但核心方法使用已有算法
- 实验充分度: ⭐⭐⭐⭐ 3 个网络、4 种算法、多指标评估，但仅展示了一种场景配置
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，跨学科内容组织良好，背景知识充足
- 价值: ⭐⭐⭐⭐⭐ 对 MARL 和自动驾驶路由规划领域有重要的基准性价值，"负面结果"极具启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Extrapolated Urban View Synthesis Benchmark](../../ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)
- [\[CVPR 2025\] Towards Autonomous Micromobility through Scalable Urban Simulation](../../CVPR2025/autonomous_driving/towards_autonomous_micromobility_through_scalable_urban_simulation.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](../../AAAI2026/autonomous_driving/tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration](simworld-robotics_synthesizing_photorealistic_and_dynamic_urban_environments_for.md)

</div>

<!-- RELATED:END -->
