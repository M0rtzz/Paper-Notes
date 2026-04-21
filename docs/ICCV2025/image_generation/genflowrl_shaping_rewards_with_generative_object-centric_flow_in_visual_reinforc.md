---
title: >-
  [论文解读] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning
description: >-
  [ICCV 2025][图像生成][强化学习] 提出 GenFlowRL，通过从跨具身数据集训练的流生成模型中提取的 δ-flow 表示进行奖励塑形，将生成式物体中心光流与强化学习结合，实现了鲁棒且可泛化的机器人操控策略学习，在 10 个操控任务上显著优于流式模仿学习和视频引导 RL 方法。
tags:
  - ICCV 2025
  - 图像生成
  - 强化学习
  - 物体中心流
  - 奖励塑形
  - 机器人操控
  - 跨具身
  - 视频生成模型
---

# GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning

**会议**: ICCV 2025  
**arXiv**: [2508.11049](https://arxiv.org/abs/2508.11049)  
**代码**: [Project Page](https://colinyu1.github.io/genflowrl)  
**领域**: image_generation  
**关键词**: 强化学习, 物体中心流, 奖励塑形, 机器人操控, 跨具身, 视频生成模型

## 一句话总结

提出 GenFlowRL，通过从跨具身数据集训练的流生成模型中提取的 δ-flow 表示进行奖励塑形，将生成式物体中心光流与强化学习结合，实现了鲁棒且可泛化的机器人操控策略学习，在 10 个操控任务上显著优于流式模仿学习和视频引导 RL 方法。

## 研究背景与动机

近年来，视频生成基础模型在机器人学习中展示了巨大潜力——通过逆动力学从生成的未来帧中推导动作。但现有方法存在两大核心问题：

**开环策略缺乏鲁棒性**：完全依赖生成的未来帧学习策略，不与环境交互，在精细操控任务中表现不佳

**视频生成质量瓶颈**：大规模机器人数据收集成本高昂，且生成的视频存在显著伪影，限制了其作为 RL 奖励信号的有效性

强化学习通过环境交互提供鲁棒性，但直接将视频生成模型用于 RL 奖励塑形存在挑战：视频是高维信号，难以从中提取精细操控特征。

**核心观察**：物体中心光流（Object-Centric Flow）是一种低维度、跨具身的表示，能够保留关键操控特征，同时抽象掉无关细节。相比原始视频帧、末端执行器关键点等表示，光流在 RL 兼容性和几何复杂度建模方面具有全面优势（见 Table 1），特别是同时支持可变形物体和关节物体。

## 方法详解

### 整体框架

GenFlowRL 包含三个核心阶段：

1. **任务条件化的物体中心光流生成**：从跨具身数据集训练流生成模型
2. **混合奖励模型**：结合 δ-flow 稠密匹配奖励和稀疏状态感知奖励
3. **流条件化的策略学习**：基于混合奖励模型训练可泛化策略

### 光流生成过程

流生成分三步：

- **流数据集构建**：使用 Grounding-DINO 检测初始帧物体边界框，CoTracker 跟踪均匀采样的 128 个关键点，生成光流表示 $\mathcal{F}_0 \in \mathbb{R}^{3 \times T \times H \times W}$
- **生成模型适配**：基于 AnimateDiff 进行两阶段微调——先微调解码器适配流数据，再用 LoRA 注入运动模块学习时序动态
- **后处理**：通过运动滤波器去除静态关键点，SAM 语义滤波器去除非物体关键点

### δ-flow 表示（关键创新）

将原始关键点流压缩为三个统计量：

$$\bar{\mathbf{P}}^t = \frac{1}{N}\sum_{i=1}^{N}\mathbf{P}_i^t, \quad \boldsymbol{\delta}_{tr}^t = \bar{\mathbf{P}}^t - \bar{\mathbf{P}}^1$$

$$\boldsymbol{\delta}_{rot}^t = \frac{1}{N}\sum_{i=1}^{N}\left[(\mathbf{P}_i^t - \bar{\mathbf{P}}^t) \times (\mathbf{P}_i^1 - \bar{\mathbf{P}}^1)\right]$$

δ-flow 的本质是蒙特卡洛估计，将冗余的多关键点轨迹压缩为位移和旋转的统计特征，有效减少了不可靠关键点的噪声影响。

### 混合奖励模型

**稠密流匹配奖励**：将生成和观测的 δ-flow 建模为高斯分布，用 KL 散度度量对齐程度，简化为均值匹配：

$$R_{\delta}^t = 1 - \text{clip}\left(\frac{(\mathcal{T}_R^t - \mathcal{T}_G^t)^2}{C}, 0, 1\right)$$

**整体奖励设计**（分阶段，任务无关）：

$$R^t = \begin{cases} \alpha \cdot (1 - \tanh(\tau \cdot d_{grip})), & \text{接近阶段} \\ \alpha, & \text{完成子目标} \\ \alpha + \beta \cdot R_{\delta}^t, & \text{子目标后} \\ 1.0, & \text{任务完成} \end{cases}$$

其中 $\alpha=0.25, \beta=0.75, \tau=10$。

### 策略设计

策略输入包含六部分：当前机器人状态、当前关键点质心、当前观测 δ-flow、k 步前瞻生成质心、k 步前瞻生成 δ-flow、初始帧 3D 质心位置。输出为 6D 位姿位移，通过逆运动学转换为关节命令。使用 DrQv2 算法优化。

### 损失函数

策略通过最大化混合奖励进行优化，采用 DrQv2 的经验回放策略：
- 学习率 $10^{-4}$，折扣因子 $\gamma=0.99$
- 探索标准差从 1.0 线性衰减到 0.1

## 实验关键数据

### 主实验：流式 RL vs 流式 IL（Table 2）

| 方法 | PickNP. | Pour | Open | Fold | Pivot |
|------|---------|------|------|------|-------|
| Heuristic | 70 | 50 | 30 | 0 | 0 |
| Im2Flow2Act | 100 | 95 | 95 | 90 | 60 |
| **GenFlowRL** | **100** | **100** | **100** | **95** | **90** |

语言条件下的优势更加显著：Fold 任务从 35→80（+45），Pivot 从 45→85（+40）。

### 与视频奖励 RL 的比较（Fig. 4）

在 MetaWorld 5 个最具挑战性的任务上：
- GenFlowRL 在 Assembly、Lever Pull、Stick Pull 等难任务上显著优于 VIPER 和 Diffusion Reward
- 收敛速度更快，成功率更高
- 纯稀疏奖励（PSR）和 RND 在简单任务尚可，但在复杂任务上挣扎

### 消融实验（Fig. 6）

| 变体 | 关键发现 |
|------|----------|
| MLP 替代 δ-flow | 性能下降，δ-flow 更好捕获时空动态 |
| 去除 3D 初始质心 | 性能下降，3D 空间信息对学习 6D 动作有益 |
| 64 关键点 vs 128 | 性能相近，δ-flow 对关键点数量不敏感 |

### 噪声鲁棒性分析（Table 4）

| 噪声条件 | PickNP. | Pour | Open | Fold | Pivot |
|----------|---------|------|------|------|-------|
| 无噪声 | 95 | 95 | 95 | 80 | 85 |
| 大高斯(4×) | 95 | 90 | 90 | 75 | 80 |
| 大漂移(2×) | 85 | 75 | 85 | 65 | 75 |

即使在大噪声条件下仍保持较高性能，证明了 δ-flow 表示的鲁棒性。

### 真实机器人验证

在 XArm7 上验证了 4 个任务的人-机器人跨具身流匹配，奖励信号呈单调递增趋势，表明部署可行性。

## 亮点与洞察

1. **表示选择的深刻洞见**：系统分析了各种操控中心表示的优劣，证明物体中心光流在低维性、跨具身性、可奖励性及几何复杂度支持方面具有综合最优性
2. **δ-flow 的蒙特卡洛本质**：将多关键点轨迹压缩为统计特征本质上是蒙特卡洛估计，自然具备噪声鲁棒性
3. **训练-推理一致性**：训练和推理都使用生成流，避免了依赖专家流带来的分布偏移
4. **混合奖励的巧妙设计**：稀疏状态感知奖励提供任务信息，稠密 δ-flow 奖励提供运动先验，二者互补

## 局限性

1. 仅使用 2D 光流，对涉及平面外旋转的任务（如拧开瓶盖）可能受限
2. 跨具身数据集规模（12K 轨迹）相对较小
3. 真实世界实验仅验证了奖励匹配性，未进行完整的端到端部署

## 相关工作与启发

- 与 HuDor (Guzey et al., 2024) 不同，本文利用生成流（而非单条专家流）进行跨具身运动先验的稠密奖励塑形
- 将流生成与 RL 结合的思路可扩展到更多具身智能任务（如导航、工具使用）
- δ-flow 表示可作为通用的操控先验，应用于其他机器人学习范式

## 评分 ⭐⭐⭐⭐

创新性 ★★★★☆：δ-flow 表示和混合奖励设计新颖且有理论支撑
实验 ★★★★☆：10 个任务覆盖广，消融充分，但真实世界评估有限
写作 ★★★★☆：结构清晰，比较表格系统全面
实用性 ★★★☆☆：需要训练流生成模型和 RL 训练，部署复杂度较高

<!-- RELATED:START -->

## 相关论文

- [CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](../../CVPR2025/image_generation/ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](../../CVPR2025/image_generation/glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers](genhancer_imperfect_generative_models_are_secretly_strong_vision-centric_enhance.md)
- [Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)

<!-- RELATED:END -->
