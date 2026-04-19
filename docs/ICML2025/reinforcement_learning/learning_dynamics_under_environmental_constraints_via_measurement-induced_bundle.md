---
title: >-
  [论文解读] Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures
description: >-
  [ICML 2025][fiber bundle] 提出一种几何框架，利用测量过程自然诱导的纤维丛结构统一处理测量不确定性、系统约束和动力学学习：在纤维丛上定义测量感知控制屏障函数(mCBF)，结合Neural ODE学习连续时间动力学，在三个机器人控制任务上实现96.3%成功率和99.3%约束满足率。
tags:
  - ICML 2025
  - fiber bundle
  - measurement-aware CBF
  - Neural ODE
  - safe learning control
  - geometric constraints
---

# Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures

**会议**: ICML 2025  
**arXiv**: [2505.19521](https://arxiv.org/abs/2505.19521)  
**代码**: [GitHub](https://github.com/ContinuumCoder/Measurement-Induced-Bundle-for-Learning-Dynamics/)  
**领域**: 安全学习控制 / 几何深度学习  
**关键词**: fiber bundle, measurement-aware CBF, Neural ODE, safe learning control, geometric constraints

## 一句话总结

提出一种几何框架，利用测量过程自然诱导的纤维丛结构统一处理测量不确定性、系统约束和动力学学习：在纤维丛上定义测量感知控制屏障函数(mCBF)，结合Neural ODE学习连续时间动力学，在三个机器人控制任务上实现96.3%成功率和99.3%约束满足率。

## 研究背景与动机

**领域现状**：在机器人等领域，在环境约束下学习未知动力学是基础问题。控制屏障函数(CBF)是确保约束满足的有力工具，但经典CBF框架将测量视为外部观测而非系统几何结构的内在组成部分。

**现有痛点**：

1. 经典CBF要求完整的全局约束知识，在仅有局部传感器测量时不可行

2. 概率滤波方法（如Kalman滤波）将测量不确定性视为外部扰动，而非利用其内在几何信息

3. 现有学习方法（Neural CBF、SafeLearn等）将测量视为完美观测，限制了实际部署的鲁棒性

4. 物理信息和几何方法虽保持结构但需要全局几何信息，无法处理局部测量不确定性

**核心洞察**：局部传感器测量（如力传感器、距离检测器）即使不完美，也包含了关于约束和动力学的充分几何信息——关键是正确利用这种局部结构而非要求全局知识。

## 方法详解

### 整体框架

状态空间流形M + 测量空间Y → 投影映射π诱导纤维丛结构E = M×Y → 在纤维丛上定义连接∇（编码状态-测量几何关系）→ 定义基于纤维丛的安全证书Φ → 测量感知控制屏障函数mCBF → Neural ODE在纤维丛上学习动力学 → 安全约束策略更新。

### 关键设计

1. **测量诱导纤维丛结构**

    - 状态x∈M与测量y=h(x)+v的关系自然构成投影π: E→M
    - 每个状态x的纤维π⁻¹(x)表征该状态下所有可能测量（受噪声约束δ_v限制）
    - 纤维丛上的连接∇通过反馈增益算子K(x)耦合状态与测量动力学
    - 对称性通过兼容Lie群作用捕捉，支持降维

2. **测量感知控制屏障函数(mCBF)**

    - 在纤维丛E上定义光滑函数b: E→R，满足三个条件：
    - (1) b(x,y)≥0 蕴含 x∈安全集S₀
    - (2) 沿可容许向量场的Lie导数条件：inf_u [L_f b + (L_g b)w + α(b)] ≥ 0
    - (3) Lipschitz连续性：|b(x,y₁)-b(x,y₂)| ≤ L_b·d(y₁,y₂)——确保安全证书对测量噪声的灵敏度有界
    - 安全保证：P(x(t)∈S₀, ∀t≥0) ≥ 1-exp(-c/δ_v²)

3. **不确定性加权学习**

    - Neural ODE学习目标中引入协方差矩阵Σ_i加权：对高不确定性数据点降低权重
    - 学习收敛保证：‖f̂-f‖ ≤ c₁exp(-λ₁t) + c₂δ_v（指数收敛+界由测量噪声决定）
    - 安全约束策略更新通过投影到安全策略集实现

### 损失函数 / 训练策略

- 底层RL框架：Soft Actor-Critic (SAC)
- 网络架构：三层MLP（128-64-32, ReLU），屏障函数输出层加tanh保证有界
- Adam优化器，混合精度训练，NVIDIA RTX 3090
- 三个实验任务的物理模拟基于Genesis物理引擎

## 实验关键数据

### 主实验

**三个机器人任务的综合对比（蠕虫500+机械臂400+四旋翼300试验的汇总）**

| 方法 | 成功率↑ | 路径长度↓ | 约束满足率↑ |
|------|---------|----------|------------|
| **Ours** | **96.3%** | **18.5±0.7m** | **99.3%** |
| Neural-CBF | 84.0% | 22.3±0.8m | 98.7% |
| SafetyNet | 86.7% | 22.1±0.7m | 98.7% |
| SafeLearn | 82.3% | 23.8±0.9m | 98.3% |
| GEM | 76.3% | 24.0±0.9m | 98.0% |
| GPMPC | 67.7% | 26.9±0.9m | **99.7%** |
| RobustSafe | 70.0% | 26.5±0.8m | 98.7% |
| SafeRL | 70.3% | 26.4±0.8m | 98.7% |

### 消融实验

| 组件 | 成功率↑ | 说明 |
|------|---------|------|
| 完整方法 | 96.3% | 纤维丛+mCBF+Neural ODE |
| 去除纤维丛结构 | ~88% | 退化为标准CBF+学习 |
| 去除不确定性加权 | ~85% | 等权对待所有数据点 |
| 去除mCBF的Lipschitz条件 | ~90% | 安全证书对噪声过度敏感 |

### 关键发现

- 相比最强基线SafetyNet（86.7%），成功率提升9.6个百分点，路径长度从22.1m缩短至18.5m
- GPMPC虽有最高约束满足率（99.7%）但成功率仅67.7%——过度保守导致路径太长
- 纤维丛结构使安全边界能根据测量质量自适应：测量不确定性高的区域更保守，低的区域更激进
- 该方法在三个截然不同的任务（软体蠕虫/刚体机械臂/空中四旋翼）上均表现最优，泛化能力强

## 亮点与洞察

- 将测量不确定性从"需要处理的噪声"提升为"有价值的几何信息"——这一视角转换是核心贡献
- 纤维丛框架提供了统一处理测量、约束、动力学的数学语言，比拼凑式的多组件方法更优雅
- mCBF的Lipschitz条件提供了安全保证优雅退化（graceful degradation）的形式化——不是非此即彼，而是安全概率随噪声平滑变化
- 理论保证（指数收敛+概率安全界限）和实验性能同时优秀

## 局限与展望

- 全部实验在仿真中完成，缺少真实世界硬件验证
- 纤维丛框架假设已知测量映射h(x)的函数形式，实际中传感器特性可能未知
- 理论分析假设有界噪声（亚高斯），对重尾噪声的鲁棒性未验证
- 计算开销相比简单CBF更高（需要维护纤维丛结构和连接计算）
- 三层MLP网络较浅，对更复杂系统可能需要更强的函数逼近器

## 相关工作与启发

- **vs Neural-CBF/SafeLearn**：这些方法学习安全证书但将测量视为完美观测，本文将测量不确定性内嵌到几何框架中
- **vs GPMPC**：高斯过程MPC方法有强不确定性量化但产生过于保守的轨迹（成功率仅67.7%）
- **vs GeoPath/GEM**：这些几何方法需要全局几何信息，本文仅需局部传感器测量
- 启发：将控制理论中的纤维丛/联络/对称性等经典概念与深度学习结合是一个有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 纤维丛框架统一测量/约束/学习的视角高度原创
- 实验充分度: ⭐⭐⭐ 三个仿真任务+12个baseline对比充分，但缺少真实世界验证
- 写作质量: ⭐⭐⭐ 理论部分数学密集，可读性对非几何背景读者有挑战
- 价值: ⭐⭐⭐⭐ 为安全学习控制提供了新的理论框架和实践方案

<!-- RELATED:START -->

## 相关论文

- [MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation](../../ICLR2026/reinforcement_learning/momagen_generating_demonstrations_under_soft_and_hard_constraints_for_multi-step.md)
- [Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics](safety_certificate_against_latent_variables_with_partially_unidentifiable_dynami.md)
- [Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics](../../NeurIPS2025/reinforcement_learning/massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)
- [Octopus: Embodied Vision-Language Programmer from Environmental Feedback](../../ECCV2024/reinforcement_learning/octopus_embodied_vision-language_programmer_from_environmental_feedback.md)
- [EvoLM: In Search of Lost Language Model Training Dynamics](../../NeurIPS2025/reinforcement_learning/evolm_in_search_of_lost_language_model_training_dynamics.md)

<!-- RELATED:END -->
