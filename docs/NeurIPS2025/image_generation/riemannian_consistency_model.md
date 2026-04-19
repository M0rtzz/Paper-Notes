---
title: >-
  [论文解读] Riemannian Consistency Model
description: >-
  [NeurIPS 2025][图像生成][一致性模型] 首次将一致性模型（Consistency Model）扩展到黎曼流形上，利用指数映射参数化和协变导数推导出离散和连续时间 RCM 目标函数，实现在球面、平坦环面和 SO(3) 等非欧几何上的高质量少步生成。
tags:
  - NeurIPS 2025
  - 图像生成
  - 一致性模型
  - 黎曼流形
  - 流匹配
  - 少步生成
  - 协变导数
---

# Riemannian Consistency Model

**会议**: NeurIPS 2025  
**arXiv**: [2510.00983](https://arxiv.org/abs/2510.00983)  
**代码**: [GitHub](https://github.com/ccr-cheng/riemannian-consistency-model)  
**领域**: 扩散模型 / 生成模型 / 流形学习  
**关键词**: 一致性模型, 黎曼流形, 流匹配, 少步生成, 协变导数

## 一句话总结

首次将一致性模型（Consistency Model）扩展到黎曼流形上，利用指数映射参数化和协变导数推导出离散和连续时间 RCM 目标函数，实现在球面、平坦环面和 SO(3) 等非欧几何上的高质量少步生成。

## 研究背景与动机

扩散模型和流匹配模型在图像生成、蛋白质设计等领域取得了显著成功，但推理阶段需要数百步迭代采样，计算代价高昂。一致性模型（CM）通过"短路"概率流 ODE，使模型在1-2步内就能生成高质量样本，在欧几里得空间（如图像）上已展现出优越性能。

然而，许多重要的科学应用涉及非欧几里得空间的生成建模。例如蛋白质生成需要描述氨基酸的三维朝向（SO(3) 旋转群）和扭转角（平坦环面），现有方法通常需要200-1000步采样。如果能在黎曼流形上实现少步生成，将显著加速药物发现和酶设计流程。

将 CM 扩展到黎曼流形面临两大挑战：（1）弯曲流形要求一致性参数化必须落在流形上，简单的线性插值不再可行；（2）流形约束要求不同点处的向量场必须位于各自的切空间中，在计算时间导数时需要额外的几何修正（协变导数）。

## 方法详解

### 整体框架

RCM 框架包含：（1）基于指数映射的一致性参数化确保流形约束；（2）离散和连续时间训练目标的闭式推导；（3）理论证明黎曼一致性蒸馏（RCD）和黎曼一致性训练（RCT）的等价性；（4）简化训练目标消除指数映射微分的复杂计算。

### 关键设计

1. **黎曼一致性参数化**：直接在向量场 $v_\theta(x_t, t)$ 上学习，并通过指数映射构造一致性函数：$f_\theta(x_t, t) := \exp_{x_t} \kappa_t v_\theta(x_t, t)$。由于 $\kappa_1 = 0$，这自然满足一致性约束 $f_\theta(x_1, 1) = x_1$。损失函数使用测地距离度量一致性：$\mathcal{L}^N_{\text{RCM}} = N^2 \mathbb{E}_{t,x_t}[w_t d^2_g(f_\theta(x_t,t), f_{\theta^-}(x_{t+\Delta t}, t+\Delta t))]$。区别于欧几里得 CM 直接用 L2 范数，这里用测地距离保证了几何一致性。

2. **连续时间极限与协变导数**：当 $N \to \infty$ 时，连续时间损失为 $\mathcal{L}^{\infty}_{\text{RCM}} = \mathbb{E}_{t,x_t}[w\|d(\exp_x)_u(\dot{\kappa}v + \kappa\nabla_{\dot{x}}v) + d(\exp u)_x(\dot{x})\|^2_g]$，其中 $\nabla_{\dot{x}}$ 是沿 PF-ODE 的协变导数（Levi-Civita 联络）。协变导数的引入是 RCM 区别于欧几里得 CM 的核心——它捕获了弯曲几何导致的切空间变化，是流形上正确微分向量场的必要条件。

3. **RCD 到 RCT 的等价性证明**：关键利用 $\dot{f}$ 对 $\dot{x}$ 的线性性（来自协变导数的线性性和指数映射微分的线性性），以及黎曼流形上条件向量场到边际向量场的贝叶斯规则推广 $\dot{x} = \mathbb{E}[(\dot{x}|x_1)|x_t]$。通过将期望移出梯度操作，证明使用条件向量场训练可达到与蒸馏相同的优化效果。

4. **简化损失函数（sRCM）**：核心近似为 $d(\exp_x)_u \approx d(\exp u)_x$，消除了两个指数映射微分的区分需求，得到简化损失 $\mathcal{L}^{\infty}_{\text{sRCM}} = \mathbb{E}_{t,x_t}[w\|\dot{x} + \dot{\kappa}v + \kappa\nabla_{\dot{x}}v\|^2_g]$。对平坦环面这一近似精确成立；对一般流形，当预训练模型质量好时近似也较准确。

5. **运动学视角解读**：RCM 目标可分解为三个物理分量：（a）预测和边际向量场的差异；（b）向量场的内在变化（时间导数）；（c）由几何约束引起的外在变化（协变导数项）。这为理解曲率如何影响一致性目标提供了直观的物理图像，例如球面上的加速度公式对应匀速圆周运动。

### 损失函数 / 训练策略

- 蒸馏模式（RCD/sRCD）：使用预训练 RFM 模型近似边际向量场
- 训练模式（RCT/sRCT）：直接使用条件向量场，无需预训练教师
- 均选用线性调度 $\kappa_t = 1-t$，权重函数 $w_t = t^2/(1-t)^2$
- 采用 magnitude-preserving 全连接层和 force weight normalization 确保 JVP 稳定

## 实验关键数据

### 主实验

在2-球面地理数据集（KL散度↓）上的2步生成结果：

| 数据集 | RFM-100 | RFM-2 | sRCD | RCD | CDnaive | sRCT | RCT |
|--------|---------|-------|------|-----|---------|------|-----|
| Earthquake (6124) | 1.51 | 10.99 | **2.13** | 2.22 | 6.20 | 3.66 | 2.38 |
| Volcano (829) | 1.77 | 35.40 | **3.36** | 3.84 | 17.19 | 5.44 | 4.47 |
| Fire (4877) | 0.53 | 9.79 | **1.65** | 1.71 | 8.01 | 3.39 | 1.74 |
| Flood (12810) | 1.33 | 8.17 | **2.27** | 2.41 | 6.21 | 2.81 | 2.39 |

SO(3) 数据集 MMD↓（×10⁻²）：

| 数据集 | RFM-2 | sRCD | CDnaive | sRCT |
|--------|-------|------|---------|------|
| Swiss Roll | 19.64 | **1.51** | 2.75 | 4.17 |
| Cone | 19.96 | **5.47** | 21.46 | 7.53 |
| Line | 15.50 | **3.06** | 9.36 | 3.75 |

### 消融实验

高维环面上的 Fréchet 距离（维度扩展性）：

| 环面维度 | 2 | 8 | 32 | 64 | 128 |
|---------|---|---|----|----|-----|
| RFM (2步) | 0.52 | 1.01 | 1.95 | 1.47 | 1.83 |
| RCT | **0.22** | **0.54** | **0.46** | **0.58** | **0.62** |
| CTnaive | 0.73 | 1.58 | 2.41 | 24.80 | 35.16 |

### 关键发现

- 朴素欧几里得 CM 在所有流形上性能最差，证明了协变导数公式的必要性
- 简化损失 sRCD/sRCT 性能与精确版本相当甚至更好，同时消除了复杂的指数映射微分计算
- RCT 在预训练模型质量差时（如 Cone 数据集）可能优于 RCD
- CTnaive 在高维流形上性能急剧退化，而 RCT 保持稳定，体现了流形约束的重要性
- 离散时间 RCD 在弯曲流形（如 SO(3)）上表现较差，可能因离散化误差更大

## 亮点与洞察

- 运动学视角的解读非常优雅：将一致性损失分解为向量场差异、内在变化和几何外在变化三个物理意义明确的分量
- 理论严谨性高：完整推导了离散→连续极限、RCD↔RCT 等价性、简化损失的合理性
- 简化损失的设计极具工程价值：消除了不同流形上指数映射微分的符号计算需求，大大降低了实现难度

## 局限与展望

- 实验仅在相对低维（最高128维）的简单流形上验证，未涉及实际蛋白质设计等高维复杂应用
- 2步采样的质量虽远超 RFM-2 但通常仍不及 RFM-100，单步生成质量有待提升
- 当前框架假设指数映射和对数映射可以闭式计算，限制了对更一般流形的适用性
- 缺乏与其他加速采样方法（如蒸馏、rectified flow）的系统对比

## 相关工作与启发

- 延续了 Song et al. 的 Consistency Model 思路，核心创新在于引入黎曼几何工具
- 与 Riemannian Flow Matching (Chen & Lipman, 2024) 的关系：RFM 提供教师模型，RCM 实现少步生成
- 可启发将一致性模型扩展到其他非标准空间（如 Lorentz 空间、Grassmann 流形）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个黎曼一致性模型，理论贡献突出  
- **实验充分度**: ⭐⭐⭐⭐ 覆盖多种流形和数据集，但缺少大规模实际应用验证  
- **写作质量**: ⭐⭐⭐⭐⭐ 数学推导严谨，运动学解读直观  
- **价值**: ⭐⭐⭐⭐ 为黎曼流形上的高效生成奠定理论基础，对药物设计等领域有潜在影响

<!-- RELATED:START -->

## 相关论文

- [Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)
- [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](../../ECCV2024/image_generation/motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)
- [FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)

<!-- RELATED:END -->
