---
title: >-
  [论文解读] Progressive Tempering Sampler with Diffusion
description: >-
  [ICML 2025][图像生成][扩散模型] 提出 Progressive Tempering Sampler with Diffusion (PTSD)，通过将 Parallel Tempering 的温度交换机制与扩散模型的神经采样器相结合，利用"温度引导"从高温扩散模型外推生成低温近似样本，在目标密度评估效率上实现数量级提升。
tags:
  - ICML 2025
  - 图像生成
  - 扩散模型
  - sampling
  - MCMC
  - parallel-tempering
  - neural-sampler
---

# Progressive Tempering Sampler with Diffusion

**会议**: ICML 2025  
**arXiv**: [2506.05231](https://arxiv.org/abs/2506.05231)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: diffusion-model, sampling, MCMC, parallel-tempering, neural-sampler  

## 一句话总结

提出 Progressive Tempering Sampler with Diffusion (PTSD)，通过将 Parallel Tempering 的温度交换机制与扩散模型的神经采样器相结合，利用"温度引导"从高温扩散模型外推生成低温近似样本，在目标密度评估效率上实现数量级提升。

## 研究背景与动机

从未归一化密度函数中采样是贝叶斯推断、统计物理和分子模拟中的基础问题。当前两类主流方法各有不足：

- **Parallel Tempering (PT)**：MCMC 领域的 SOTA，通过在多个温度层运行并行马尔可夫链并交换样本来实现高效混合。但每次需要新的独立样本时都必须重新运行完整的传播过程，计算代价高昂，且只能生成相关样本。
- **基于扩散的神经采样器**（DDS、iDEM、BNEM、CMCD 等）：可以摊销采样过程生成不相关样本，但在目标密度评估效率上远不如 PT。低效源于需要通过重要性采样估计目标函数或在轨迹模拟中大量调用目标密度。

关键洞察：这两类方法处于方法论频谱的两端——一端完全不使用数据训练，另一端完全依赖 PT 生成的数据事后拟合。PTSD 则定位于中间地带，融合两者优势。

## 方法详解

### 整体框架

PTSD 定义递减温度序列 [T_K, T_{K-1}, ..., T_1]，其中 T_1 为目标温度。算法分四步：

1. **高温初始化**：在最高两个温度 T_K, T_{K-1} 运行 PT 收集样本到缓冲区
2. **拟合初始扩散模型**：分别在两个高温上训练扩散模型 theta_K, theta_{K-1}
3. **温度引导外推**：利用温度引导机制从高温模型生成 T_{K-2} 的近似样本
4. **微调与迭代**：初始化 theta_{K-2} <- theta_{K-1}，微调模型；重复步骤 3-4 直到获得目标温度模型 theta_1

### 关键设计一：温度引导 (Temperature Guidance)

核心思想是利用已训练好的两个温度 T_1, T_2 的扩散模型，通过 Taylor 展开和有限差分近似来估计更低温度 T_0 的 score 函数。

对 score 函数在 T_1 处做一阶 Taylor 展开，并用有限差分近似导数，最终得到：

> score(x_t, T) ≈ (1+w) * score(x_t, T_1) - w * score(x_t, T_2)

其中权重 w = (T_1 - T) / (T_2 - T_1)。这与 classifier-free guidance 形式高度相似：通过对比"更好的"低温模型和"更差的"高温模型来实现引导。当 t -> 0 时近似不精确，但扩散模型在小时间步的 score 精度对生成质量影响有限。

### 关键设计二：截断重要性重采样 (Truncated Importance Resampling)

温度引导生成的是近似样本，误差会随温度层数累积。利用 PF-ODE 采样同时获得样本及其密度，计算自归一化重要性权重：

> w_n = [p_tilde(x_n)^{1/T_k} / q(x_n)] / sum_{n'} [p_tilde(x_{n'})^{1/T_k} / q(x_{n'})]

为防止 Hutchinson trace 估计器引入的方差导致不稳定，采用截断重要性采样——将权重裁剪到预设分位数 tau 的最大值，然后按权重做分类重采样填充缓冲区。

### 关键设计三：局部 PT 精炼 (Local PT Refinement)

在重要性重采样之后，对缓冲区样本执行若干步 MCMC 精炼。在相邻温度的缓冲区之间运行并行 PT 链，通过样本交换提升两个缓冲区的样本质量。可选地，仅对 IS 样本的子集运行 PT 链以优化能量评估的使用。这体现了"扩散模型做主力、MCMC 做精修"的设计哲学。

## 实验关键数据

### 样本质量对比 (Table 1)

在三个基准任务上与多种神经采样器比较 W2 距离（越低越好）：

| 方法 | GMM-40 W2 | MW-32 W2 | LJ-55 W2 |
|------|----------|----------|----------|
| iDEM | - | - | - |
| BNEM | 2.16 | - | 1.76 |
| PT+DM | - | - | - |
| **PTSD** | **1.93** | **4.99** | 1.81 |

PTSD 在 GMM-40 和 MW-32 上达到 SOTA 样本质量。在 LJ-55 上略逊于 BNEM（因 BNEM 显式回归目标能量以处理抑制区域），但目标评估次数远低于 BNEM。

### 目标密度评估效率 (Table 2)

PTSD 相比 DDS/CMCD 实现 2-3 个数量级的目标评估效率提升：

| 方法 | GMM-40 评估次数 | MW-32 评估次数 |
|------|--------|-------|
| CMCD | ~4.4e9 | ~1.6e9 |
| DDS | ~2.6e9 | ~8.2e8 |
| iDEM | ~5e8 | ~1.8e7 |
| BNEM | ~7.5e7 | ~1.8e7 |
| PT+DM | ~1e6 | ~1e6 |
| **PTSD** | **~1e6** | **~1e6** |

PTSD 与 PT+DM 效率相当，但在 Fig. 6 的详细对比中，相同评估预算下 PTSD 样本质量更优。温度引导产生了更具信息量的"交换"机制。

### 消融实验 (Table 4, MW-32)

| 配置 | TVD | W2 |
|------|------|---------|
| PTSD w/o temp-guide | 0.34 | 24.59 |
| PTSD w/o IS | 0.23 | 5.84 |
| **PTSD (完整)** | **0.14** | **4.99** |

温度引导和截断 IS 都是关键组件。即使移除 IS，PTSD 仍优于 Table 1 大多数基线。

### Alanine Dipeptide 验证 (Table 3)

| 指标 | PT+DM | PTSD |
|------|-------|------|
| Mean log-likelihood | 较低 | 213.32 |
| KL divergence | 6.9e-2 | 3.2e-2 |

在 2.6e7 能量评估预算下，PTSD 获得更高对数似然和更低 KL 散度。

## 关键发现

1. **温度引导有效外推**：在 LJ-55 上，从温度 2.0 和 1.5 的模型外推到 1.0，温度引导产生的分布与真实分布高度重叠，优于模型外推 (ME)、自动微分外推 (AD) 和 score rescaling (RS) 等替代方案
2. **跨温度的信息传递优于样本交换**：扩散模型作为目标密度的"功能表示"，通过权重共享实现比传统样本交换更高效的跨温度信息传递
3. **PTSD 在 Pareto 前沿上一致领先**：在 log-likelihood vs. 能量评估次数的帕累托前沿上（Fig. 10），PTSD 在所有数据集上均处于最优位置

## 亮点与洞察

- **方法论定位精准**：巧妙地将 PT 和神经采样器的融合定位于方法论频谱的中间地带，而非追求极端
- **温度引导与 CFG 的类比**：温度引导的数学形式 (1+w)*D1 - w*D2 与 classifier-free guidance 完全一致——"用差的版本对比好的版本"。这种跨领域的概念迁移优雅且启发性强
- **渐进式 bootstrapping**：不同于传统神经采样器的自我 bootstrapping（可能陷入低效循环），PTSD 的跨温度 bootstrapping 天然提供从易到难的课程学习效果
- **PF-ODE 的双重利用**：同时用于采样和密度估计（通过瞬时变量变换公式），使重要性权重计算不需要额外的目标评估

## 局限性

1. **Wall-clock 时间劣势**：虽然目标评估次数大幅减少，但需要为每个温度层微调扩散模型，当前实验中实际墙钟时间仍慢于纯 PT
2. **不可并行化**：温度必须逐步递减处理，无法像 PT 那样在多设备上并行运行不同温度链
3. **对温度调度和超参数敏感**：温度层差距过大或目标分布过于复杂时性能会变得脆弱；网络选择、学习率、截断阈值等均影响最终结果
4. **评估局限**：当前仅在合成多模态分布和小分子系统上验证，尚未在高维实际问题上测试

## 相关工作与启发

- **FAB (Midgley et al., 2023)**：用 alpha-2 散度和 AIS 训练 normalizing flow，引入 replay buffer
- **iDEM / BNEM**：直接估计 score 函数但依赖大量重要性采样，目标评估开销大
- **DDS / CMCD**：匹配采样过程与目标过程的路径测度，Langevin 预条件导致大量目标评估
- **Generalized PT (Zhang et al., 2025)**：用神经网络在相邻温度间传输样本以提升 PT 交换率——与 PTSD 形成互补视角
- **CFG (Ho & Salimans, 2021; Karras et al., 2024)**：PTSD 的温度引导直接类比 CFG 的"好/差模型对比"范式

**启发**：扩散模型在采样问题中的角色可以从"替代 MCMC"转变为"增强 MCMC"——两者融合可能是更务实的方向。

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ★★★★☆ |
| 理论深度 | ★★★★☆ |
| 实验充分度 | ★★★★☆ |
| 实用价值 | ★★★☆☆ |
| 写作质量 | ★★★★★ |
| 总体评价 | ★★★★☆ |

温度引导的推导虽基于简单的 Taylor 展开 + 有限差分，但与 CFG 的联系深刻且优美。实验在合成和分子任务上充分验证了效率提升。主要扣分在实用价值——wall-clock 时间仍不占优、温度调度敏感、尚未扩展到真正大规模问题。

<!-- RELATED:START -->

## 相关论文

- [Learnable Sampler Distillation for Discrete Diffusion Models](../../NeurIPS2025/image_generation/learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](../../NeurIPS2025/image_generation/diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [AMO Sampler: Enhancing Text Rendering with Overshooting](../../CVPR2025/image_generation/amo_sampler_enhancing_text_rendering_with_overshooting.md)
- [ProReflow: Progressive Reflow with Decomposed Velocity](../../CVPR2025/image_generation/proreflow_progressive_reflow_with_decomposed_velocity.md)

<!-- RELATED:END -->
