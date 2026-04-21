---
title: >-
  [论文解读] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes
description: >-
  [AAAI 2026][图像生成][扩散模型] 提出 Constrained Particle Seeking (CPS)，一种无梯度的扩散模型反问题求解方法，通过利用所有候选粒子信息构建前向过程的局部线性代理模型，并在转移核高密度区域的超球面约束下寻找最优粒子，性能可与梯度方法媲美。
tags:
  - AAAI 2026
  - 图像生成
  - 扩散模型
  - 反问题
  - 无梯度方法
  - 粒子优化
  - 约束优化
---

# Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes

**会议**: AAAI 2026  
**arXiv**: [2603.01837](https://arxiv.org/abs/2603.01837)  
**代码**: [GitHub](https://github.com/deng-ai-lab/CPS)  
**领域**: 扩散模型 / 反问题求解  
**关键词**: 扩散模型, 反问题, 无梯度方法, 粒子优化, 约束优化

## 一句话总结

提出 Constrained Particle Seeking (CPS)，一种无梯度的扩散模型反问题求解方法，通过利用所有候选粒子信息构建前向过程的局部线性代理模型，并在转移核高密度区域的超球面约束下寻找最优粒子，性能可与梯度方法媲美。

## 研究背景与动机

反问题（从间接噪声观测 $y = \mathcal{H}(x) + \eta$ 恢复原信号 $x$）是计算成像、医学、地球物理等领域的基础挑战。扩散模型作为强大先验在反问题求解中展现出优势，但存在关键限制：

**梯度依赖**：主流方法（DPS、ΠiGDM、DAPS 等）需要前向观测过程的梯度信息来引导采样，但许多实际场景中前向模型涉及高度非线性或计算昂贵的数值模拟（如黑洞成像、流体模拟），梯度计算困难或不可行。

**现有无梯度方法的不足**：
   - **SCG**（类拒绝采样）：每步采样多个候选粒子但只保留最佳的一个，丢弃了大量有价值信息，效率低下。
   - **EnKG**（集成卡尔曼滤波）：在高维空间维护数千粒子，计算开销大；全局线性化在早期阶段误差大。
   - **DPG**（策略梯度）：需对干净样本加噪扰动，可能偏离数据流形。
   - 这些方法性能通常不如梯度方法。

核心观察：SCG 丢弃的粒子并非无用——实验发现 top-2/3 粒子甚至取反符号后的 bottom 粒子都包含有用的引导信息。信息分布在整个粒子集合中，应该被充分利用。

## 方法详解

### 整体框架

CPS 将扩散反问题重新表述为约束优化问题：在每个时间步寻找一个最优粒子，使其在观测对齐目标上最优，同时保持在无条件先验的高密度区域内。包含三步：

1. 从反向转移核采样多个候选粒子
2. 利用所有粒子拟合前向过程的局部线性代理
3. 在超球面约束下求解最优粒子作为下一状态

### 关键设计

#### 1. 前向过程代理模型

优化终端代价需要估计 $\nabla_{x_t} \Phi(\hat{x}_{0|t})$，但前向模型 $\mathcal{H}(\cdot)$ 的梯度不可用。CPS 采用**统计线性化**在 $p(x_t|x_{t+1})$ 定义的局部区域构建线性代理 $\mathcal{H}(\hat{x}_{0|t}) \approx \mathbf{A}x_t + \mathbf{b}$。

采样 n 个粒子后，计算代理参数的闭合解：

$$\mathbf{A} = \frac{1}{n\sigma_t^2} \sum_{i=1}^{n} (\mathcal{H}(\hat{x}_{0|t}^i) - \bar{\mathcal{H}})(x_t^i - \mu_t)^\top$$

$$\mathbf{b} = \bar{\mathcal{H}} - \mathbf{A}\mu_t$$

关键优势：$\mathbf{A}$ 是所有采样粒子的加权线性组合，整合了全部候选信息，使得可以找到优于任何单个样本的粒子。

与 EnKG 的区别：EnKG 在全局空间线性化导致早期阶段大误差，CPS 仅在 $p(x_t|x_{t+1})$ 的局部区域线性化，Jensen 间隙分析显示 CPS 误差始终保持较小。

#### 2. 转移核约束（超球面约束）

高维高斯采样的范数集中在超球面 $S^{d-1}(\mu_t, \sigma_t\sqrt{d})$ 上（$\chi^d$ 分布的质量集中现象）。扩散模型偏好输入在该超球面上而非均值附近。因此约束优化问题为：

$$\min_{x_t} \|y - (\mathbf{A}x_t + \mathbf{b})\|^2 \quad \text{s.t.} \; x_t \in S^{d-1}(\mu_t, \sigma_t\sqrt{d})$$

通过 Lagrange 乘子法求解，由于 $\sigma_t$ 通常较小，约束占主导，可得渐近形式的解析解：

$$x_t^* \approx \mu_t + \sigma_t\sqrt{d} \frac{\mathbf{A}^\top(y - \bar{\mathcal{H}})}{\|\mathbf{A}^\top(y - \bar{\mathcal{H}})\|}$$

这个解可理解为在无条件均值 $\mu_t$ 基础上，沿着代理模型指示的最优方向偏移恰好一个"标准半径"的距离，确保粒子停留在转移核的高概率密度区域。

#### 3. Restart 策略增强鲁棒性

实际中不同初始噪声可能导致不同结果。为缓解累积误差，引入 Restart 策略：在时间步 t，将样本沿 Eq.2 重新加噪回 t+1，再用 CPS 重新引导去噪，逐步修正累积误差。外层循环遍历时间步 T-1 到 0，内层循环执行 $N_r$ 次 restart。

### 损失函数 / 训练策略

CPS 是**免训练**方法，直接利用预训练的扩散模型：
- 无需微调或额外训练
- 仅需对前向模型进行黑盒访问（只用前向过程的输出，不需要梯度）
- 每步采样 n 个候选粒子（实验中 n=32 通常足够）
- Restart 次数 $N_r$ 控制精度-效率权衡

## 实验关键数据

### 主实验

**表1：FFHQ 图像反问题（256×256）**

| 任务 | 方法类型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| 修复(95%) | 梯度 | ΠiGDM | 23.13 | 0.774 | 0.231 |
| 修复(95%) | 梯度 | DAPS | 22.71 | 0.754 | 0.226 |
| 修复(95%) | 无梯度 | SCG | 6.15 | 0.321 | 0.807 |
| 修复(95%) | 无梯度 | EnKG | 22.64 | 0.767 | 0.286 |
| 修复(95%) | **无梯度** | **CPS** | **24.90** | **0.794** | **0.161** |
| 超分(×4) | 梯度 | DPS | 27.65 | 0.822 | 0.120 |
| 超分(×4) | 无梯度 | **CPS** | **27.70** | **0.834** | **0.115** |
| JPEG(QF=5) | 梯度 | ΠiGDM | 18.25 | 0.593 | 0.304 |
| JPEG(QF=5) | 无梯度 | **CPS** | **23.23** | **0.784** | **0.223** |

CPS 在所有无梯度方法中大幅领先，在多个任务上与梯度方法持平甚至超越。特别是 JPEG 恢复这种不可微场景中，CPS 远超 ΠiGDM。

**表2：黑洞成像**

| 观测比例 | 方法 | PSNR↑ | BPSNR↑ |
|---|---|---|---|
| 100% | SCG | 25.45 | 30.56 |
| 100% | **CPS** | **25.74** | **31.84** |
| 3% | SCG | 24.65 | 31.28 |
| 3% | **CPS** | **25.03** | **31.81** |

**表3：流体数据同化**

| 稀疏程度 | 方法 | Relative L2↓ |
|---|---|---|
| ×2 | EnKG | 0.320 |
| ×2 | **CPS** | **0.295** |
| ×8 | EnKG | 0.821 |
| ×8 | **CPS** | **0.684** |

在科学反问题（黑洞成像、流体同化）中也展现了显著优势。

### 消融实验

- **粒子效率**：CPS 用 8 个粒子即可获得良好表现，EnKG 需要大量粒子才能工作，SCG 在图像问题上完全失败。
- **Restart 策略**：不仅提升整体性能，还显著增强算法稳定性，有效避免因糟糕初始噪声导致的失败。

### 关键发现

1. SCG 丢弃粒子是极大浪费——即使最差粒子取反方向后也包含有用引导信息。
2. 局部线性化的 Jensen 间隙远小于全局线性化（EnKG），验证了局部代理的优越性。
3. 超球面约束至关重要——它确保优化后的粒子仍在扩散先验的高密度区域。
4. 在 JPEG 恢复等不可微场景中，无梯度方法反而有天然优势。

## 亮点与洞察

- **信息利用效率高**：从"选最佳粒子"到"综合所有粒子信息合成最优粒子"，理念转变带来性能飞跃。
- **优雅的数学形式**：最终解析解简洁优美——均值 + 归一化方向 × 标准半径。
- **即插即用**：免训练、仅需黑盒前向访问，可直接用于各种预训练扩散模型。
- **跨域验证**：同时在图像恢复和科学问题上验证，说明方法的通用性。

## 局限与展望

1. 每步需要 n 次前向模型评估（采样+代理拟合），对计算昂贵的前向模型仍有开销。
2. 线性代理在高度非线性观测模型下可能精度不足（虽然局部化缓解了此问题）。
3. 仅在 FFHQ 和特定科学数据上测试，未在自然图像大规模数据集上验证。
4. Restart 策略增加了总采样步数，推理时间相应增长。

## 相关工作与启发

- 与 DPS 类梯度方法相比，CPS 不需要对 $\mathcal{H}$ 求导，极大拓宽了适用场景。
- 与 EnKG 相比，局部线性化+单粒子寻优避免了维护大粒子集的开销和全局线性化的误差。
- 约束优化视角将反问题求解从"被动选择"转为"主动搜寻"，可启发其他粒子方法的改进。
- Restart 策略的思想来自 RePaint/Restart 系列工作，在此框架下效果显著。

## 评分

- **新颖性**: ★★★★☆ — 约束超球面粒子寻优是新颖的无梯度策略
- **技术深度**: ★★★★★ — 随机控制+统计线性化+Lagrange乘子法+渐近分析，数学扎实
- **实验**: ★★★★☆ — 图像+黑洞+流体三类验证，与梯度方法对等比较
- **写作**: ★★★★☆ — 条理清晰，实验驱动的方法设计解释到位
- **实用性**: ★★★★★ — 免训练即插即用，黑盒前向访问，开源代码

<!-- RELATED:START -->

## 相关论文

- [A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models](../../ICML2025/image_generation/integrating_intermediate_layer_optimization_and_projected_gradient_descent_for_s.md)
- [Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](../../CVPR2025/image_generation/improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/image_generation/composition_and_alignment_of_diffusion_models_using_constrai.md)

<!-- RELATED:END -->
