---
title: >-
  [论文解读] Straighten Viscous Rectified Flow via Noise Optimization
description: >-
  [ICCV 2025][图像生成][Rectified Flow] 本文提出 VRFNO（Viscous Rectified Flow via Noise Optimization），通过引入历史速度项增强轨迹区分度、并用编码器联合训练来优化噪声构建最优耦合，有效拉直 Rectified Flow 的推理轨迹，在 CIFAR-10 和 AFHQ 上取得单步/少步生成的 SOTA 性能（单步 FID 4.50，无需蒸馏）。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Rectified Flow"
  - "噪声优化"
  - "少步生成"
  - "轨迹拉直"
  - "速度场"
---

# Straighten Viscous Rectified Flow via Noise Optimization

**会议**: ICCV 2025  
**arXiv**: [2507.10218](https://arxiv.org/abs/2507.10218)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: Rectified Flow, 噪声优化, 少步生成, 轨迹拉直, 速度场

## 一句话总结

本文提出 VRFNO（Viscous Rectified Flow via Noise Optimization），通过引入历史速度项增强轨迹区分度、并用编码器联合训练来优化噪声构建最优耦合，有效拉直 Rectified Flow 的推理轨迹，在 CIFAR-10 和 AFHQ 上取得单步/少步生成的 SOTA 性能（单步 FID 4.50，无需蒸馏）。

## 研究背景与动机

### Rectified Flow 的理想与现实

Rectified Flow（RF）是一种基于 ODE 的生成模型，核心思想是在噪声和图像之间构建直线插值轨迹，学习一个常速度场。如果模型能完美学到常速度场，理论上一步即可从高斯噪声生成图像。然而，实际训练中 RF 学不到真正的常速度场，推理轨迹仍然弯曲，仍需多步采样。

### Reflow 的成功与局限

RF 将轨迹弯曲归因于参考轨迹的交叉，并提出 Reflow 操作来解决：先用预训练模型生成图像，构建确定性耦合（deterministic coupling），然后用这些耦合训练新模型。作者通过深入分析发现 Reflow 之所以有效有两个被忽视的因素：

**确定性耦合的优势**：确定性耦合意味着噪声和图像之间存在可学习的确定性轨迹，训练相当于逐步将该轨迹拉直。而随机匹配的噪声-图像对没有显式轨迹关系，学习更困难。

**数据复用效应**：Reflow 在同一轨迹的不同时间步采样中间状态来训练，类似蒸馏中的多时间尺度优化。

然而 Reflow 的缺陷也很明显：

- **分布偏差**：生成图像与真实图像存在分布差距，每轮迭代都会积累误差。通常只能做 2-3 轮。
- **存储压力**：需要存储大量预生成的数据对，计算资源消耗大。

### 关键理论洞察

作者通过定理证明了一个重要事实：在高维空间中，随机匹配的直线插值轨迹交叉的概率为 $P \sim O(e^{-c(n \times n)})$，极其稀少。因此轨迹弯曲的真正原因不是"交叉"，而是**近似交叉**——不同轨迹的中间状态在统计性质上高度相似（特别是早期阶段噪声主导时），导致模型难以区分，预测模糊。

## 方法详解

### 整体框架

VRFNO 包含两个核心创新，集成在一个编码器-速度场联合训练框架中：(1) 引入历史速度项（HVT）作为辅助输入，帮助模型区分相似的中间状态；(2) 通过编码器对噪声做重参数化优化，构建"优化耦合"（optimized coupling）替代 Reflow 的确定性耦合，直接使用真实图像训练。

### 关键设计

1. **历史速度项（Historical Velocity Term, HVT）**:

    - 功能：将前一时间步的模型预测速度作为辅助信息输入当前时间步的速度场模型
    - 核心思路：基于定理 2，任意两条轨迹间的速度差异大于状态差异：
    $\Delta(v_{ref}^{(i)}, v_{ref}^{(j)}) \geq \Delta(X_t^{(i)}, X_t^{(j)})$
      因此在模型输入中加入速度信息可以更好地区分轨迹。改进后的 Viscous Rectified Flow 形式为：
    $dX_t = v(X_t, t, v_{history}) dt, \quad t \in [\Delta t, 1]$
      训练时 $v_{history} = \text{stopgrad}(v_\theta(X_{t-\Delta t}, t-\Delta t, 0))$，推理时第一步 HVT 设为 0，后续每步用前一步的预测速度
    - 设计动机：不同轨迹的中间状态可能非常相似（尤其在早期），但它们的运动方向差异更大。利用这种更显著的差异帮助模型做出更准确的预测

2. **噪声优化与优化耦合**:

    - 功能：用编码器将随机噪声转换为"优化噪声"，使噪声-图像对满足优化耦合条件
    - 核心思路：编码器 $E_\phi$ 以图像 $X_1$ 为输入，输出均值 $\mu$ 和方差 $\sigma^2$，通过重参数化技术生成优化噪声：
    $X_0 = \epsilon \cdot \sigma^2 + \mu$
      优化耦合的定义：当 $\|v_\theta(tX_1 + (1-t)X_0) - (X_1 - X_0)\| \leq \varepsilon$ 时，$(X_0, X_1)$ 称为优化耦合
    - 设计动机：避免依赖 Reflow 的预生成图像（存在分布偏差），直接使用真实数据集中的图像。编码器为每张图像找到一个更适合直线轨迹学习的噪声子空间，类似 Reflow 的数据复用效果但不受数据量限制

3. **编码器中的随机扰动**:

    - 功能：在编码器中间层注入随机噪声 $\tau \sim N(0, I)$
    - 核心思路：即使多个高斯噪声匹配到同一图像，重参数化后的均值和方差也会有差异，保证生成多样性
    - 设计动机：防止编码器产生记忆效应，确保同一图像可以匹配到同一子空间中的不同噪声

### 损失函数 / 训练策略

总损失为速度匹配损失加 KL 正则化：

$$L(\theta, \phi) = \underbrace{\mathbb{E}_{t \in p(t)}[d(v_{ref}, v_\theta(X_t, t, v_{history}))]}_{\text{VCL}} + \alpha \underbrace{\frac{1}{2}(\sigma^2 + \mu^2 - 1 - \log(\sigma^2))}_{\text{KLL}}$$

KL 正则化约束编码器输出的均值和方差接近标准高斯，防止过拟合。

训练分两阶段：第一阶段用 MSE 损失收敛后，第二阶段加入 LPIPS 损失联合训练至再次收敛。**无需蒸馏或对抗训练**。

采样时，先通过编码器对随机噪声做重参数化（需要一张数据集中的图像），然后按 Euler 方法沿 VRF ODE 求解。

## 实验关键数据

### 主实验

CIFAR-10 上的单步和少步生成（无蒸馏）：

| 方法 | NFE | IS↑ | FID↓ | KID(×10⁻³)↓ |
|------|-----|-----|------|------------|
| 1-RF | 1 | 1.13 | 379 | 428 |
| 2-RF | 1 | 8.15 | 11.97 | 8.66 |
| CAF | 1 | 8.32 | 4.81 | - |
| TraFlow | 1 | - | 4.50 | - |
| **VRFNO** | **1** | **9.59** | **4.50** | **2.73** |
| 2-RF | 10 | 9.13 | 3.83 | 1.63 |
| CAF | 10 | 9.12 | 3.77 | - |
| **VRFNO** | **10** | **9.51** | **3.36** | **1.31** |

AFHQ 单步生成（不同分辨率）：

| 方法 | 数据集 | 64×64 | 128×128 | 256×256 |
|------|--------|-------|---------|---------|
| 2-RF | AFHQ-CAT | 181.93 | 172.66 | 171.84 |
| **VRFNO** | **AFHQ-CAT** | **28.69** | **27.56** | **27.04** |
| 2-RF | AFHQ-DOG | 200.77 | 192.30 | 189.82 |
| **VRFNO** | **AFHQ-DOG** | **44.64** | **27.21** | **27.37** |

### 消融实验

CIFAR-10 上各组件贡献（FID↓）：

| 配置 | HVT | 噪声优化 | 1步FID | 5步FID | 10步FID |
|------|-----|---------|--------|--------|---------|
| A | ✗ | ✗ | 379 | 34.81 | 12.70 |
| B | ✓ | ✗ | 332 | 32.50 | 9.34 |
| C | ✗ | ✓ | 4.72 | 4.28 | 4.75 |
| D | ✓ | ✓ | **4.53** | **4.03** | **3.40** |

轨迹直线度（NFSS↓，越小越直）：

| 数据集 | 2-RF | 3-RF | CAF | VRFNO |
|--------|------|------|-----|-------|
| 2D | 0.067 | 0.053 | 0.058 | **0.054** |
| CIFAR-10 | 0.058 | 0.056 | 0.035 | **0.026** |

### 关键发现

- 噪声优化是性能提升的主要来源：单独使用将 1 步 FID 从 379 降至 4.72
- HVT 在各步数下都有一致的改善，特别是在 10 步时（12.70→9.34 无噪声优化，4.75→3.40 有噪声优化）
- VRFNO 的推理轨迹在 NFSS 评估中最直（0.026 vs CAF 的 0.035）
- 在 AFHQ 上 VRFNO 对比 2-RF 有数量级的提升（如 CAT 256×256：27.04 vs 171.84）
- 编码器参数量极小（不到速度场模型的 1/20），时间开销略高于 RF 但低于 CAF

## 亮点与洞察

- 对 Reflow 成功原因的深入分析非常有价值：将"轨迹交叉"的直觉解释纠正为"近似交叉导致的统计相似性混淆"，并提供了理论支撑
- 优化耦合的概念比确定性耦合更灵活：不需要预训练模型生成图像，不会积累误差，编码器的噪声子空间聚焦效应自然实现了数据复用
- 两阶段训练（MSE→MSE+LPIPS）的策略简单有效，避免了蒸馏和对抗训练的复杂性
- 历史速度项的引入虽然增加了少量计算，但从信息论的角度看是合理的：提供了轨迹方向的先验

## 局限与展望

- 采样时需要从数据集中采样一张图像作为编码器输入，限制了完全无条件的生成场景
- 目前仅在 CIFAR-10 和 AFHQ 上验证，尚未扩展到 ImageNet 等大规模数据集和高分辨率
- 编码器的额外推理时间（单步生成时耗时 0.305s vs RF 的 0.172s）在追求极致速度时可能成为瓶颈
- 第二阶段引入 LPIPS 的时机依赖经验判断（收敛后切换），缺乏自动化机制

## 相关工作与启发

- 与 CAF（Constant Acceleration Flow）最相似，但 CAF 需要两个速度场模型导致推理更慢，而 VRFNO 只需一个速度场加一个轻量编码器
- 噪声优化的思路与之前逐噪声迭代优化的方法不同：用编码器一次性做线性变换，大幅减少迭代次数
- 联合训练框架保持了边际分布不变性（定理 3），理论上保证了生成质量
- 历史速度项的思路可以推广到其他基于 ODE 的生成模型中

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](../../ICML2026/image_generation/offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[ICCV 2025\] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)
- [\[ICCV 2025\] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/image_generation/rectified-cfg_for_flow_based_models.md)

</div>

<!-- RELATED:END -->
