---
title: >-
  [论文解读] Rectified-CFG++ for Flow Based Models
description: >-
  [NeurIPS 2025][目标检测][Classifier-Free Guidance] 针对Rectified Flow模型中标准CFG导致的离流形漂移问题，提出Rectified-CFG++——一种自适应预测-校正引导策略，通过条件流预测+时间调度插值校正替代外推式引导，在Flux/SD3/SD3.5/Lumina等大规模模型上全面超越标准CFG。
tags:
  - NeurIPS 2025
  - 目标检测
  - Classifier-Free Guidance
  - Rectified Flow
  - 文本到图像生成
  - 预测-校正采样
  - 流模型
---

# Rectified-CFG++ for Flow Based Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.07631](https://arxiv.org/abs/2510.07631)  
**作者**: Shreshth Saini, Shashank Gupta, Alan C. Bovik (UT Austin)  
**代码**: [rectified-cfgpp.github.io](https://rectified-cfgpp.github.io/)  
**领域**: object_detection  
**关键词**: Classifier-Free Guidance, Rectified Flow, 文本到图像生成, 预测-校正采样, 流模型  

## 一句话总结

针对Rectified Flow模型中标准CFG导致的离流形漂移问题，提出Rectified-CFG++——一种自适应预测-校正引导策略，通过条件流预测+时间调度插值校正替代外推式引导，在Flux/SD3/SD3.5/Lumina等大规模模型上全面超越标准CFG。

## 研究背景与动机

### 问题背景
Classifier-Free Guidance (CFG) 是当前扩散模型中控制条件生成质量的核心技术，通过线性外推条件与无条件速度场来增强文本对齐。然而，Rectified Flow (RF) 模型因采用确定性ODE积分（无随机正则化），CFG的外推特性会导致采样轨迹偏离学习到的数据流形，产生过饱和、结构扭曲、文本错误等视觉伪影。

### 已有工作的不足
- **标准CFG**: 在RF模型中直接使用外推式组合 $\hat{v}_\omega = (1-\omega)v^u + \omega v^c$（$\omega \geq 1$），将轨迹推离流形 $\mathcal{M}_t$
- **CFG++**: 为扩散模型设计的流形约束引导，未针对RF几何结构优化
- **APG (Analytical Posterior Guidance)**: 部分缓解伪影，但在细节或几何精度上有妥协
- **CFG-Zero★**: 提供有限改进，仍受外推本质限制
- 上述方法均缺乏**流模型专用**的理论保证和几何感知设计

### 核心动机
RF模型的几何结构天然适合**插值**而非外推。设计一种利用条件流的确定性传输路径、以插值方式融入引导信号的采样策略，可以在保持流形一致性的同时实现高质量条件生成。

## 方法详解

### 核心思想：预测-校正替代外推

标准CFG的更新为外推式：$x_{t-\Delta t} = x_t + \Delta t(v^u_t + \omega \Delta v^\theta_t)$，其中 $\Delta v^\theta_t = v^c_t - v^u_t$。这种外推在确定性ODE中缺乏随机噪声的正则化，容易发散。

Rectified-CFG++ 用**三步**替代：

### Step 1: 条件流预测（Predictor）

使用纯条件速度场 $v^c_t$ 进行半步预测，将样本沿条件流形推进：

$$\tilde{x}_{t-\Delta t/2} \leftarrow x_t + \frac{\Delta t}{2} v^c_t$$

关键在于：使用 $v^c_t$ 而非 $v^u_t$ 或CFG混合速度，确保轨迹从一开始就锚定在目标条件子空间流形上，避免早期偏离。

### Step 2: 引导差异校正（Corrector via Guidance Difference）

在预测的中间点 $\tilde{x}_{t-\Delta t/2}$ 处，分别计算条件和无条件速度场：

$$v^c_{t-\Delta t/2} \leftarrow v_\theta(\tilde{x}_{t-\Delta t/2}, t-\Delta t/2, y)$$
$$v^u_{t-\Delta t/2} \leftarrow v_\theta(\tilde{x}_{t-\Delta t/2}, t-\Delta t/2, \varnothing)$$

在中间预测点评估引导差异 $\Delta v^\theta_{t-\Delta t/2}$，比在当前点 $x_t$ 处评估更准确——尤其当速度场快速变化时。

### Step 3: 插值式更新（Interpolative Update）

最终有效速度以条件方向为锚点，加上时间调度的引导校正：

$$\hat{v}_{\lambda t} \leftarrow v^c_t + \alpha(t)(v^c_{t-\Delta t/2} - v^u_{t-\Delta t/2})$$

其中调度函数 $\alpha(t) = \lambda_{\max}(1-t)^\gamma$，$\lambda_{\max} > 0, \gamma \geq 0$。随后用 $\hat{v}_{\lambda t}$ 执行标准ODE更新。

### 理论保证

**Lemma 3.1 (引导方向稳定性)**: 在Lipschitz连续假设下，中间点的引导差异与当前点的差异之差为 $O(\Delta t)$：
$$\|\Delta v^\theta_{t-\Delta t/2} - \Delta v^\theta_t(x_t)\| \leq L V_{\max} \Delta t$$

**Proposition 1 (单步扰动有界)**: Rectified-CFG++单步偏离纯条件流的距离严格受控：
$$\|\hat{x}_{t-1} - \tilde{x}_{t-1}\| \leq \alpha(t) B \Delta t$$

这保证了轨迹始终位于数据流形 $\mathcal{M}_t$ 的有界管状邻域内，邻域大小由 $\alpha(t)$ 和引导场界 $B$ 控制。

### 与CFG的关键区别
| 特性 | 标准CFG | Rectified-CFG++ |
|------|---------|-----------------|
| 引导方式 | 外推（extrapolation） | 插值（interpolation） |
| 基准速度 | 无条件 $v^u_t$ | 条件 $v^c_t$ |
| 引导评估点 | 当前点 $x_t$ | 中间预测点 $\tilde{x}_{t-\Delta t/2}$ |
| 流形保持 | 无保证，易漂移 | 理论保证有界邻域 |
| 额外网络/训练 | 否 | 否 |

## 实验关键数据

### 实验1：MS-COCO 10K 多模型综合评测

在四个主流RF模型上，Rectified-CFG++ 全面对比标准CFG：

| 模型 | 引导方法 | FID↓ | CLIP↑ | Aesthetic↑ | ImageReward↑ | PickScore↑ | HPSv2↑ |
|------|---------|------|-------|-----------|-------------|-----------|--------|
| Lumina | CFG | 26.93 | **0.3511** | **5.8226** | **1.0924** | 0.5867 | 0.2797 |
| Lumina | Rect-CFG++ | **22.49** | 0.3464 | 5.7755 | 0.9611 | **0.6133** | **0.3004** |
| SD3 | CFG | 23.89 | 0.3439 | 5.5465 | 0.9812 | 0.4408 | 0.2751 |
| SD3 | Rect-CFG++ | **23.39** | **0.3471** | **5.6529** | **1.0009** | **0.5591** | **0.2897** |
| SD3.5 | CFG | 20.29 | **0.3506** | 6.155 | 1.0487 | 0.4923 | 0.2933 |
| SD3.5 | Rect-CFG++ | **20.22** | 0.3497 | **6.1651** | **1.0796** | **0.5077** | **0.2946** |
| Flux-dev | CFG | 37.86 | 0.3351 | 4.721 | **1.0528** | 0.3248 | 0.2621 |
| Flux-dev | Rect-CFG++ | **32.23** | **0.3493** | **5.3251** | 0.948 | **0.6752** | **0.2996** |

Flux-dev上FID从37.86降至32.23（降幅14.9%），PickScore从0.3248提升至0.6752（翻倍），说明标准CFG在Flux上伪影特别严重；Rectified-CFG++的改善幅度最大。

### 实验2：引导策略对比（MS-COCO 1K, SD3.5）

| 引导方法 | FID↓ | ImageReward↑ | CLIP↑ | HPSv2↑ |
|---------|------|-------------|-------|--------|
| 无引导 | 77.30 | 0.3852 | 0.3260 | 0.2421 |
| CFG | 67.71 | 1.0530 | **0.3515** | 0.2941 |
| CFG-Zero★ | 68.39 | 0.9947 | 0.3458 | 0.2879 |
| APG | 67.23 | 1.0748 | 0.3513 | 0.2935 |
| **Rect-CFG++** | **67.15** | **1.0845** | 0.3506 | **0.2959** |

Rectified-CFG++在FID、ImageReward、HPSv2三项上均最优，CLIP仅略低于CFG。

### T2I-CompBench 组合生成评测

| 模型 | Color↑ | Shape↑ | Texture↑ | Spatial↑ |
|------|--------|--------|----------|----------|
| Flux CFG | 0.6132 | 0.4152 | 0.5928 | 0.2488 |
| Flux Rect-CFG++ | **0.7728** | **0.5018** | **0.6705** | **0.2790** |
| SD3 CFG | 0.7658 | 0.5698 | 0.7270 | 0.3199 |
| SD3 Rect-CFG++ | **0.8041** | **0.5778** | **0.7362** | **0.3306** |

Flux上Color属性从0.6132提升至0.7728（+26%），说明CFG在Flux上的颜色偏移问题被有效修复。

### 消融实验：组件贡献（MS-COCO 1K, SD3.5）

| 配置 | FID↓ | CLIP↑ | HPSv2↑ | Aesthetic↑ |
|------|------|-------|--------|-----------|
| 用无条件速度做预测 | 91.12 | 0.1439 | 0.1870 | 6.1049 |
| 无Predictor | 73.70 | 0.3410 | 0.2969 | 6.1064 |
| 无Corrector | 74.65 | 0.3414 | 0.2975 | 6.1047 |
| **完整Rect-CFG++** | **72.97** | **0.3446** | **0.2995** | **6.1587** |

用无条件速度做预测时CLIP骤降至0.14，证明条件预测步是方法的核心。

### 计算效率

在相似运行时间下（SD3.5, 512x512），Rectified-CFG++ 用20步NFE达到FID 74.47，标准CFG用28步NFE仅达到85.82。实际FLOPs几乎相同，运行时间差异约0.04秒。

## 亮点

- **原理简洁优雅**：用插值替代外推这一核心直觉清晰，预测-校正框架自然地将条件锚定和引导校正解耦，无需额外网络或训练
- **理论完备**：提供了流形一致性和轨迹有界性的严格数学证明，是少有的兼具理论和实验的引导方法
- **即插即用（Drop-in）**：无需训练、无需修改模型权重、几乎无额外计算，可直接替换现有RF模型中的CFG模块
- **文本渲染显著改善**：在图像内文字生成任务上表现突出，这是扩散模型的已知难点
- **全面验证**：覆盖4个大型模型、5个数据集、6+指标、用户研究，实验设计严谨

## 局限与展望

- **每步多一次前向传播**：预测步需要额外的条件速度场评估，虽然总步数可减少，但单步成本增加约1倍
- **超参数引入**：论文声称"parameter-free beyond guidance scale"，但实际上调度函数 $\alpha(t) = \lambda_{\max}(1-t)^\gamma$ 引入了两个额外超参
- **仅验证文本到图像**：未在视频生成、3D生成等流模型的其他应用中验证
- **Lumina上部分指标不如CFG**：CLIP、Aesthetic、ImageReward在Lumina上反而下降，说明方法不总是全面占优
- **领域分类不准**：本文是生成模型/采样方法论文，与object_detection领域无关

## 与相关工作的对比

- **CFG (Ho & Salimans 2022)**：外推式引导的原始方法，在RF模型中产生严重伪影；Rectified-CFG++用插值替代外推根本性解决
- **CFG++ (Chung et al. 2024)**：为扩散SDE设计的流形约束方法，依赖随机正则化，不适用于确定性RF
- **APG (Sadat et al. 2024)**：分析后验引导，部分缓解伪影但在细节上妥协；Table 3显示Rectified-CFG++在FID/ImageReward/HPSv2上均优于APG
- **CFG-Zero★**：通过零化初始引导减少早期漂移，但后期仍受外推影响；性能整体不如Rectified-CFG++
- **ReCFG**：使用引导差异的相关工作，但Rectified-CFG++的预测-校正框架和中间点评估是关键差异

## 评分

- 新颖性: ⭐⭐⭐⭐ — 插值替代外推的核心思想简单但有效，预测-校正框架设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ — 4个模型、多个数据集、6+指标、消融、用户研究，非常全面
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，图表丰富，结构完整
- 价值: ⭐⭐⭐⭐ — 即插即用的实用性强，对RF模型生态有直接贡献

<!-- RELATED:START -->

## 相关论文

- [Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/object_detection/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)
- [XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)
- [Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](../../CVPR2025/object_detection/enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)

<!-- RELATED:END -->
