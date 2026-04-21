---
title: >-
  [论文解读] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning
description: >-
  [NeurIPS 2025][遥感][轨道动力学] 本文提出OrbitZoo，一个基于工业级Orekit轨道动力学库构建的多智能体RL环境，支持碰撞规避、霍曼转移、星座协调等真实轨道任务，通过PettingZoo接口实现标准化MARL训练，并在Starlink真实星历数据验证中达到低误差组24米RMSE（16.6小时传播）。
tags:
  - NeurIPS 2025
  - 遥感
  - 轨道动力学
  - 多智能体RL
  - 卫星操控
  - 碰撞规避
  - 仿真环境
---

# OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2504.04160](https://arxiv.org/abs/2504.04160)  
**代码**: 开源（未注明具体链接）  
**领域**: 遥感 / 强化学习  
**关键词**: 轨道动力学, 多智能体RL, 卫星操控, 碰撞规避, 仿真环境

## 一句话总结

本文提出OrbitZoo，一个基于工业级Orekit轨道动力学库构建的多智能体RL环境，支持碰撞规避、霍曼转移、星座协调等真实轨道任务，通过PettingZoo接口实现标准化MARL训练，并在Starlink真实星历数据验证中达到低误差组24米RMSE（16.6小时传播）。

## 研究背景与动机

**领域现状**：地球轨道上约有2万颗卫星（约50%仍在运行），以及约1.4亿个碎片物体。随着Starlink等大规模星座的部署，低地球轨道(LEO)拥堵问题日益严重。传统的卫星操控高度依赖人工决策，面对不断增长的复杂性已变得不可持续。RL在自主卫星操控中展现出潜力。

**现有痛点**：现有的RL轨道环境存在三大问题：(1) 大多使用简化的动力学模型（如三体问题、仅牛顿引力），难以捕捉真实轨道扰动；(2) 多为定制环境从零构建，缺乏标准化和可复现性——动力学验证工作量大且容易出错；(3) 缺少对多智能体场景、连续控制、真实推力建模和可视化的同时支持（Table 1列举了14个现有环境，没有一个兼具所有功能）。

**核心矛盾**：RL需要大量的仿真交互来训练策略，这要求环境既要保真（高保真动力学）又要高效（快速传播）。同时，sim-to-real gap是将RL策略部署到真实卫星的核心障碍——仿真必须经过真实数据验证。

**本文目标** (1) 提供一个兼具高保真动力学和RL标准接口的统一环境；(2) 支持多智能体协作/竞争场景；(3) 通过真实星历数据验证仿真精度。

**切入角度**：站在Orekit（工业级开源轨道力学库）的肩膀上获得高保真动力学，同时封装PettingZoo标准接口使RL研究者无需了解轨道力学细节。

**核心 idea**：将工业级轨道传播器与标准MARL框架组合，构建"既够真实又够易用"的卫星RL基准环境。

## 方法详解

### 整体框架

OrbitZoo的架构分为三层：底层是基于Orekit的高保真轨道传播引擎（处理引力场、大气阻力、太阳辐射压、第三体效应等）；中间层是任务定义层（定义观测空间、动作空间、奖励函数、终止条件）；上层是PettingZoo标准的MARL接口（每个卫星作为独立agent，支持部分可观测POMDP建模）。

### 关键设计

1. **高保真数据生成引擎**:

    - 功能：提供与真实轨道行为高度一致的仿真环境
    - 核心思路：基于Orekit的数值传播器，支持Holmes-Featherstone谐波引力场、基于历史天气数据的大气阻力、考虑日/月遮挡的太阳辐射压(SRP)、来自太阳系所有行星及日/月/地月质心的第三体引力。支持笛卡尔坐标、开普勒轨道要素和春分点轨道要素三种状态表示。使用Dormand-Prince变步长积分器提供高精度传播。通过并行计算加速多体传播
    - 设计动机：已有RL环境多用简化模型（如仅J2扰动或仅牛顿引力），无法训练出可迁移到真实环境的策略

2. **标准化MARL接口**:

    - 功能：使RL研究者可以专注于算法设计而非环境实现
    - 核心思路：基于PettingZoo框架实现POMDP结构，每个卫星agent有独立的观测空间和动作空间。推力动作采用极坐标参数化(T, θ, φ)——推力大小和RSW坐标系下的方向角。支持合作、竞争和混合场景的配置。可与MARLlib、EPyMARL等主流MARL库直接集成
    - 设计动机：Table 1显示大部分现有环境未同时支持多智能体交互和工业级仿真器

3. **模块化奖励框架**:

    - 功能：支持灵活的、物理约束的奖励设计
    - 核心思路：提供体间度量（相对距离、碰撞概率PoC、视线条件）和个体度量（燃料消耗、质量变化）两类信号。支持从稠密到稀疏的不同奖励模式，以及多目标权衡（性能vs安全vs效率）。碰撞概率基于Akella方法计算，PoC > 10^{-6}被视为高风险
    - 设计动机：轨道控制中奖励设计面临延迟反馈和耦合动力学的固有挑战，模块化框架让研究者更容易实验不同的奖励策略

### 可视化组件

OrbitZoo提供基于Python的实时3D可视化工具，据作者所知是首个在RL框架中内置实时轨道可视化的Python实现。可用于策略检查、故障诊断和行为解释。

## 实验关键数据

### 主实验1：单智能体霍曼转移

| 指标 | 结果 |
|------|------|
| 任务 | 30km高度提升 |
| 算法 | PPO (连续动作) |
| 结果 | 近最优轨道转移，半长轴与理论值匹配 |
| 发现 | 智能体自适应了理论解不考虑的扰动力 |

### 主实验2：单智能体碰撞规避(CAM)

| 算法 | 训练动力学 | 评估动力学 | PoC降低效果 |
|------|-----------|-----------|------------|
| DQN (离散) | 牛顿+阻力 | 全扰动 | 有效但泛化较弱 |
| PPO (连续) | 牛顿+阻力 | 全扰动 | 更优，泛化能力更强 |

### 主实验3：Starlink真实数据验证

| 组别 | 卫星数 | 平均RMSE (米) | 传播时长 |
|------|--------|--------------|---------|
| 低RMSE组 | - | 24.14 | 16.6小时 |
| 中RMSE组 | - | 83.75 | 16.6小时 |
| 高RMSE组 | - | 1924.90 | 16.6小时 |
| 共计 | 31颗 | - | - |

### 关键发现

- 连续动作空间(PPO)在真实扰动下泛化能力显著优于离散动作(DQN)——因为轨道机动本质上是连续的
- 在简化动力学下训练的策略可以在全扰动环境下评估，但泛化存在gap——这正是OrbitZoo试图帮助研究者缩小的
- 4颗GEO星座卫星的合作PPO学会了维持等角间距同时最小化燃料消耗，策略还能泛化到训练中未见的扰动（第三体力、SRP、阻力）
- 高RMSE组（1924.90米）卫星偏差较大，可能因为缺少其物理特性（阻力系数、反射系数等）的精确信息——通过贝叶斯优化调参可以改善

## 亮点与洞察

- 综合性是OrbitZoo最大的优势：Table 1对比14个现有环境，只有OrbitZoo同时满足所有7项能力（多智能体、工业仿真器、高保真动力学、连续控制、真实推力建模、可视化、代码开源）。这种"全能选手"的定位填补了领域空白
- 将Orekit（Java生态）与PettingZoo（Python生态）桥接的工程贡献不容忽视，使得轨道力学领域的工业级仿真能力对RL社区变得触手可及
- 真实Starlink数据验证提供了仿真精度的可信证据——低RMSE组在16.6小时传播后仅24米误差，这对2小时内的操作决策窗口已足够准确

## 局限与展望

- 高RMSE组偏差较大（1924米），说明对未知物理特性的卫星仿真精度仍有提升空间，需要更好的参数估计方法
- RL实验主要展示了环境的可用性，但未与其他环境在同一任务上进行严格的对比实验
- 大规模星座（数千颗卫星）的可扩展性虽然提到了并行化支持，但未给出具体的计算性能数据
- 仅在LEO和GEO场景下验证，深空或月球轨道等更复杂场景未测试
- 缺少与Model Predictive Control等传统控制方法的对比基准

## 相关工作与启发

- **vs ColAvGym (Kazemi 2024)**：使用真实CDM数据但仅单智能体CAM场景；OrbitZoo支持多智能体且场景更丰富
- **vs Poliastro**：Python原生但保真度不如Orekit，缺少高阶引力场和多种扰动支持
- **vs STK**：商业软件保真度高但成本高、不开源，无法直接集成RL训练循环
- OrbitZoo的设计思路（标准化环境 + 高保真仿真 + 真实数据验证）对其他安全关键领域的RL环境构建有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性整合了高保真仿真与MARL框架，填补了缺少全面轨道RL环境的空白
- 实验充分度: ⭐⭐⭐⭐ 涵盖多种任务场景和真实数据验证，但RL实验深度可进一步加强
- 写作质量: ⭐⭐⭐⭐ 结构清晰，背景知识铺陈充分，便于RL和航天两个社区的读者理解
- 价值: ⭐⭐⭐⭐ 开源环境对推动空间自主操控的RL研究有重要基础设施价值

<!-- RELATED:START -->

## 相关论文

- [CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)
- [GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](../../CVPR2026/remote_sensing/geoflow_real-time_fine-grained_cross-view_geolocalization.md)
- [ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)

<!-- RELATED:END -->
