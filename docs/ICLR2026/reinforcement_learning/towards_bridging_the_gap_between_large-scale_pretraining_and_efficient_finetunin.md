---
title: >-
  [论文解读] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control
description: >-
  [ICLR 2026][人形机器人控制] LIFT提出预训练-微调三阶段框架：(i) 大规模并行SAC预训练实现零样本部署；(ii) 基于拉格朗日动力学的物理先验世界模型离线预训练；(iii) 确定性动作执行+世界模型内随机探索的高效微调，在Booster T1和Unitree G1人形机器人上验证了从仿真到真实世界的全流程。
tags:
  - ICLR 2026
  - 人形机器人控制
  - 大规模预训练
  - 高效微调
  - SAC
  - 物理先验世界模型
  - 仿真到现实
---

# Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control

**会议**: ICLR 2026  
**arXiv**: [2601.21363](https://arxiv.org/abs/2601.21363)  
**代码**: https://lift-humanoid.github.io  
**领域**: 强化学习  
**关键词**: 人形机器人控制, 大规模预训练, 高效微调, SAC, 物理先验世界模型, 仿真到现实

## 一句话总结
LIFT提出预训练-微调三阶段框架：(i) 大规模并行SAC预训练实现零样本部署；(ii) 基于拉格朗日动力学的物理先验世界模型离线预训练；(iii) 确定性动作执行+世界模型内随机探索的高效微调，在Booster T1和Unitree G1人形机器人上验证了从仿真到真实世界的全流程。

## 研究背景与动机

**领域现状**：PPO因大规模并行GPU仿真的鲁棒收敛成为人形机器人控制的主流方法，可实现零样本部署。但on-policy方法的低样本效率限制了安全适应新环境的能力。

**现有痛点**：(1) 直接用off-policy方法做大规模并行训练关注不足；(2) 微调时随机探索可能损坏执行器或导致不安全状态，对支撑面小的人形机器人尤其危险；(3) 从头训练model-based方法耗时极长且容易陷入局部最优。

**核心矛盾**：大规模预训练需要on-policy方法的稳定性和并行效率，而高效微调需要off-policy方法的样本效率和model-based方法的数据高效性。

**本文目标**：如何统一预训练和微调阶段的算法选择，同时保证安全性和效率？

**切入角度**：以SAC为统一backbone，预训练阶段用高UTD大批量并行训练，微调阶段将随机探索限制在世界模型内，环境中只执行确定性动作。

**核心 idea**：SAC贯穿预训练-微调全流程，物理先验世界模型桥接仿真与真实，确定性执行+模型内探索实现安全高效微调。

## 方法详解

### 整体框架
LIFT分三阶段：(i) 基于JAX的大规模SAC预训练（1024并行环境，高UTD=10），在MuJoCo Playground中快速收敛；(ii) 用预训练数据离线训练物理先验世界模型；(iii) 在新环境中用确定性策略收集数据，在世界模型内用随机策略探索生成合成轨迹，交替微调策略和世界模型。

### 关键设计

1. **大规模SAC预训练**:

    - 功能：在GPU上并行训练SAC，实现与PPO可比的wall-clock效率
    - 核心思路：JAX实现的全编译SAC，固定tensor形状实现高效OP fusion，大批量更新（batch=1024）和高UTD（=10）在1024并行环境中无额外通信开销
    - 使用asymmetric actor-critic——actor接收本体感知状态 $s_t$，critic接收特权状态 $s_t^p$
    - 设计动机：SAC的off-policy特性使其比PPO更自然地与model-based方法集成，状态依赖的随机策略在世界模型rollout中提供更好的探索多样性

2. **物理先验世界模型**:

    - 功能：结合刚体动力学先验和学习残差的混合模型
    - 核心思路：基于拉格朗日方程 $M(q_t)\ddot{q_t} + C(q_t,\dot{q_t}) + G(q_t) = B\tau_t + J^\top F^e_t + \tau^d_t$，其中 $M,C,G,B$ 已知（取决于机器人几何/惯性参数），用残差网络 $\tau_\phi(s_t,a_t) \approx J^\top F^e_t + \tau^d_t$ 学习未知的接触力和耗散项
    - 损失函数：$\mathcal{L}_\phi = \frac{1}{B}\sum_{b=1}^{B}((\hat{s}^p_{b,t+\Delta t} - s^p_{b,t+\Delta t})^2 \odot \exp(-\log\sigma^2_{b,t}) + \log\sigma^2_{b,t})$
    - 设计动机：纯神经网络世界模型在有限数据下泛化能力差，会产生物理不合理的预测导致critic损失爆炸

3. **安全微调策略**:

    - 功能：在新环境中安全高效地微调策略
    - 核心思路：环境中执行确定性动作（action mean），随机探索仅在世界模型rollout中进行。从replay buffer采样初始状态，在世界模型中展开 $H_{wm}=20$ 步轨迹用于训练actor-critic
    - 安全重置：当基座高度、速度、姿态角或关节状态超出预设阈值时立即终止rollout
    - 设计动机：人形机器人支撑面小，单支撑相对扰动极度敏感，随机探索可能导致摔倒

### 损失函数 / 训练策略
- 预训练：标准SAC目标 + Optuna超参搜索（约10小时），Booster T1训练时间从7小时降至30分钟
- 世界模型：高斯负对数似然损失，端到端梯度通过归一化、坐标变换、PD控制器和Euler积分反传
- 微调：多epoch自回归训练增强样本效率，长度2-4的自回归损失稳定学习

## 实验关键数据

### 预训练实验（6个人形任务）
- LIFT在所有flat terrain任务上达到与PPO/FastTD3可比的peak return
- 在rough terrain上更快稳定在peak性能
- 单GPU（RTX 4090）30分钟内完成Booster T1预训练

### 微调实验（Brax环境，8个seed）

| 场景 | LIFT | SAC | PPO | FastTD3 | SSRL |
|------|------|-----|-----|---------|------|
| In-Distribution (0.6 m/s) | ✓ 收敛 | 发散 | 初始还行后崩溃 | 强振荡后崩溃 | 有收敛迹象但不达标 |
| Long-Tail (1.0 m/s) | ✓ 收敛 | 发散 | 崩溃 | 崩溃 | 不收敛 |
| OOD (1.5 m/s) | ✓ 收敛 | 发散 | 崩溃 | 崩溃 | 不收敛 |

微调仅需 $4 \times 10^4$ 环境步（约800秒在线交互时间）。

### 消融实验

| 配置 | 结果 |
|------|------|
| 完整LIFT（SAC预训练+WM预训练） | 4×10⁴步内收敛到目标速度 |
| 去掉WM预训练 | 仍能收敛但明显慢 |
| 去掉SAC+WM预训练（=SSRL） | 只学会站立，几乎零前进速度 |
| MBPO ensemble替代物理先验WM | 不收敛，critic损失爆炸 |

### 真实世界微调
- 从80-590秒真实数据收集后，机器人展现更直立的姿态、更平滑的步态和更稳定的前进速度
- 限制：依赖Vicon动捕估计基座高度，IMU积分有漂移

## 亮点与洞察
- **统一backbone的优势**：SAC贯穿预训练到微调避免了算法切换导致的目标不一致和遗忘
- **物理先验的关键性**：消融实验定量证明纯神经网络世界模型在有限数据下完全无法工作，物理先验提供了必要的泛化归纳偏置
- **安全探索范式**：确定性执行+模型内随机探索是一个可推广的范式，对任何需要安全微调的机器人系统都有参考价值
- **工程贡献**：从SSRL代码中发现并修正了状态映射错误，增加基座高度到特权状态——对人形机器人是关键的

## 局限与展望
- 当前仅使用本体感知观测，不支持视觉输入
- 真实世界微调依赖外部运动捕捉系统和IMU积分
- 微调pipeline是同步的（数据收集→训练），异步pipeline可显著提高效率
- 动作修正可能unbounded（vs ASAP的delta-action方法）

## 相关工作与启发
- **vs PPO**: PPO在确定性数据收集+有限数据下逐渐退化崩溃，不适合微调场景
- **vs SSRL**: LIFT本质是SSRL的预训练增强版，验证了从头训练model-based在人形上不可行
- **vs FastTD3**: FastTD3虽实现了大规模off-policy训练，但缺乏微调验证和sim-to-real
- **vs DreamerV3**: Dreamer使用latent world model+learned reward，在确定性数据收集下不稳定

## 评分
- 新颖性: ⭐⭐⭐⭐ 预训练-微调框架结合物理先验世界模型，针对人形机器人的系统性解决方案
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实、多平台（T1/G1）、多场景（in/out分布）、详细消融
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，消融设计合理，但文章较长
- 价值: ⭐⭐⭐⭐⭐ 提供了完整的开源pipeline，对人形机器人学习社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [RLP: Reinforcement as a Pretraining Objective](rlp_reinforcement_as_a_pretraining_objective.md)
- [WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)

<!-- RELATED:END -->
