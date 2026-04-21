---
title: >-
  [论文解读] CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers
description: >-
  [ICCV 2025][图像生成][扩散模型加速] 提出 Chords，一种基于多核层次 ODE 求解器的扩散采样加速框架，通过慢到快的核间纠正机制（inter-core rectification），在 4-8 个 GPU 上实现 2.1×~2.9× 加速，且不牺牲生成质量。
tags:
  - ICCV 2025
  - 图像生成
  - 扩散模型加速
  - 多核并行
  - ODE求解器
  - 训练无关
  - 视频生成
---

# CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers

**会议**: ICCV 2025  
**arXiv**: [2507.15260](https://arxiv.org/abs/2507.15260)  
**代码**: [项目页面](https://hanjq17.github.io/CHORDS)  
**领域**: image_generation  
**关键词**: 扩散模型加速, 多核并行, ODE求解器, 训练无关, 视频生成

## 一句话总结

提出 Chords，一种基于多核层次 ODE 求解器的扩散采样加速框架，通过慢到快的核间纠正机制（inter-core rectification），在 4-8 个 GPU 上实现 2.1×~2.9× 加速，且不牺牲生成质量。

## 研究背景与动机

扩散模型已成为高质量图像和视频生成的主导方法，但推理过程本质上是迭代的，计算代价极高。现有加速手段主要分两类：

**蒸馏方法**（progressive/consistency distillation）：需要额外训练，通用性差

**快速 ODE 求解器**（DDIM、DPM-Solver 等）：单核方法在步数减少时质量下降严重

**为什么并行化是一个自然的方向？** 传统数值计算中，多重网格（multigrid）方法通过多层级并行加速 ODE 求解由来已久。然而，将此思想迁移到扩散模型上面临挑战：

- 现有多核方法（如 ParaDIGMS）基于滑动窗口 Picard 迭代，收敛慢、对架构有限制
- SRDS 固定使用 √N 个核，灵活性不足，且需要手工设计流水线
- 没有一个统一的理论框架来指导最优核分配策略

**核心动机**：能否设计一种（1）训练无关，（2）模型无关，（3）灵活适应不同核数 的通用加速框架？

## 方法详解

### 整体框架

Chords 的核心思想是将多核扩散采样视为一个 ODE 求解器流水线：慢但准确的求解器通过理论保证的核间通信机制，逐步纠正快但不准确的求解器。

框架由四个组件构成：

1. **参数化**：通过初始化序列 **I** = [t⁽¹⁾, ..., t⁽ᴷ⁾] 确定每个核的起始时间点
2. **初始化**：核 k 在 t⁽ᵏ⁾ 时刻初始化，通过从 t=0 跳步到达
3. **终止**：核 k 到达 t=1 后输出结果，用户可按质量需求选择任意核的输出
4. **通信**：每隔 δ⁽ᵏ⁾ = t⁽ᵏ⁾ - t⁽ᵏ⁻¹⁾ 时间，慢核纠正快核

### 关键设计

#### 1. 多核纠正（Multi-core Rectification）

这是 Chords 的基础操作。考虑两个核：慢核 1 从 t 开始、快核 2 从 t' 开始（t < t'）。经过 δₜ = t' - t 时间后：

- 慢核 1 得到更准确的 x₁ₜ'
- 快核 2 已经前进到 x²ₜ'+δₜ

纠正规则：将慢核的准确信息传递给快核：

$$x^2_{t'+\delta_t} \leftarrow x^2_{t'+\delta_t} + r_\theta(x^1_{t'}, x^2_{t'}, t', \delta_t)$$

其中纠正项 r_θ 通过在 t' 时刻执行两次"单步跳跃"估算连续求解，计算差值来修正误差。

**理论保证（Proposition 2.1）**：在 f_θ 充分光滑的条件下，纠正后的误差是纠正前误差的高阶无穷小：

$$\|\tilde{x}_{t'} + r_\theta - x_{t'}\|_2 = o(\|\tilde{x}_{t'} - x_{t'}\|_2)$$

**为什么这个纠正有效？** 直觉上，慢核因为走得更细、更准确，它到达同一时间点的 latent 更接近真实 ODE 轨迹。纠正项利用了 f_θ 的局部线性性质，通过比较从不同起点出发的一步估计差异，来消除累积误差。

#### 2. 最优初始化序列选择

框架性能取决于初始化序列 **I** 的选择。作者定义了：

- **加速比** S(I) = 1/(1 - t⁽ᴷ⁾)：最快核提供多少倍加速
- **奖励函数** R(I)：衡量输出质量的代理指标

**为什么需要奖励函数而非直接优化？** 由于 f_θ 是高维神经网络，直接预计算误差不可行。奖励函数通过三个性质（最优性、单调性、权衡性）捕捉并行 ODE 求解的本质特征。

**Theorem 2.5**（三核最优初始化）：对于 K=3 和加速比 s，最优序列为：
- s ≤ 3 时：t⁽²⁾ = t⁽³⁾/2
- s > 3 时：t⁽²⁾ = 2t⁽³⁾ - 1

一般 K 个核采用从快到慢的贪心递推策略确定。

#### 3. 离散化实例化

将连续框架转化为实际可用的算法：

- **初始化**：核 k 通过从 t(i₁) 跳步 k-1 次到达 t(iₖ)
- **调度器 Scheduler(N, step, k)**：确定每步每核处理的时间区间
- **通信条件 Communicate(k, prev, cur)**：当 (cur-prev) 能被 iₖ-iₖ₋₁ 整除时触发纠正
- **流式输出**：各核按到达 t=1 的先后顺序输出，质量递增

### 损失函数 / 训练策略

Chords 是**训练无关**的方法，不涉及任何训练过程。它直接用现有的扩散采样器（DDIM、Euler 等）作为子例程，通过核间通信实现加速。核心超参数仅有初始化序列 **Î**。

对于 N=50 步、K=4/6/8 核，实际使用的初始化序列分别为：
- K=4: [0, 8, 16, 32]
- K=6: [0, 3, 6, 12, 24, 36]
- K=8: [0, 2, 4, 8, 16, 24, 32, 40]

## 实验关键数据

### 主实验

在三个视频扩散模型和两个图像扩散模型上全面评估。

| 模型 | 方法 | 4核加速比 | 8核加速比 | VBench/CLIP | Latent RMSE |
|------|------|-----------|-----------|-------------|-------------|
| HunyuanVideo | Sequential | 1.0× | 1.0× | 84.4% | - |
| HunyuanVideo | ParaDIGMS | 1.3× | 1.4× | 84.2% | 0.202 |
| HunyuanVideo | SRDS | 1.4× | 2.6× | 84.2% | 0.068 |
| **HunyuanVideo** | **Chords** | **2.1×** | **2.9×** | **84.1%** | **0.068** |
| Wan2.1 | Chords | 1.8× | 2.7× | 85.1% | 0.043 |
| CogVideoX1.5 | Chords | 2.0× | 2.4× | 81.8% | 0.055 |
| SD-3.5-Large | Chords | 2.0× | 2.4× | 32.5 | 0.211 |
| Flux | Chords | 2.0× | 2.6× | 31.0 | 0.179 |

关键发现：4核下 Chords 比 SRDS 平均快 67%，8核下快 50%。

### 消融实验

| 实验内容 | 设置 | HunyuanVideo 加速比 | Flux 加速比 |
|----------|------|---------------------|-------------|
| 初始化序列 | Chords（理论最优） | 2.9× (8核) | 2.4× (8核) |
| 初始化序列 | 均匀分布 | 2.6× (8核) | 2.2× (8核) |
| 初始化序列 | Chords（理论最优） | 2.1× (4核) | 2.0× (4核) |
| 初始化序列 | 均匀分布 | 1.8× (4核) | 1.8× (4核) |

不同步数 N 的影响（HunyuanVideo，8核）：

| N | 加速比 | VBench | Latent RMSE |
|---|--------|--------|-------------|
| 50 | 2.9× | 84.1% | 0.068 |
| 75 | 3.4× | 84.4% | 0.073 |
| 100 | 3.6× | 84.6% | 0.076 |

### 关键发现

1. **理论指导的初始化优于均匀分布**：在所有设置下，理论最优序列比均匀分布一致性地提升 10-15% 加速比
2. **加速比随核数和步数增长**：步数增大时加速比进一步提升（N=100 时达 3.6×）
3. **质量几乎无损**：所有设置下 VBench/CLIP Score 波动极小，Latent RMSE 显著低于 ParaDIGMS
4. **统一框架**：ParaDIGMS 和 SRDS 均可视为 Chords 框架的特殊实例化

## 亮点与洞察

1. **理论与实践的完美结合**：从连续 ODE 的多重网格理论出发，推导最优初始化，再离散化到实际算法——这种 top-down 方法论值得借鉴
2. **流式生成（Diffusion Streaming）能力**：层次化多核结构天然支持"先快后好"的渐进式输出，适合交互式场景
3. **与其他加速方法正交**：可与模型并行（DiT 中的注意力分割）、蒸馏等方法叠加使用
4. **框架的统一性**：将多个已有方法纳入同一框架，指明改进方向

## 局限与展望

1. 核间通信依赖同步点（synchronize barrier），在异构计算环境中可能产生等待开销
2. 理论推导基于简化的奖励函数（f_θ(x,t)=x），与实际神经网络行为有差距
3. 纠正项需要额外的网络前向传播（虽然与正常步骤并行），存在计算开销
4. 框架主要关注 ODE 采样，对 SDE 采样（随机采样）的扩展有待研究

## 相关工作与启发

- **ParaDIGMS** [Shih et al.]：滑动窗口 Picard 迭代，是 Chords 的特殊情况
- **SRDS** [Liu et al.]：多重网格并行扩散求解器，固定核数为 √N
- **传统多重网格方法** [Brandt 1977]：Chords 的理论根基
- **启发**：该框架可能扩展到其他迭代式生成模型的加速（如自回归模型的推测解码有类似思想）

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ (统一的理论框架 + 实用的算法设计)
- **实验充分度**: ⭐⭐⭐⭐⭐ (5个模型，3种核数配置)
- **写作质量**: ⭐⭐⭐⭐⭐ (理论推导清晰，层层递进)
- **实用价值**: ⭐⭐⭐⭐⭐ (即插即用，适用于多种模型)
- **总评**: ⭐⭐⭐⭐⭐ (扩散模型多核加速领域的重要工作)

<!-- RELATED:START -->

## 相关论文

- [Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)
- [Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)
- [REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents](reducio_generating_1k_video_within_16_seconds_using_extremely_compressed_motion_.md)
- [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)
- [HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)

<!-- RELATED:END -->
