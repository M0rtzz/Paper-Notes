---
title: >-
  [论文解读] Adaptive Discretization for Consistency Models
description: >-
  [NeurIPS 2025][图像恢复][Consistency Model] 提出ADCM——通过将一致性模型的离散化步长形式化为局部一致性（可训练性）与全局一致性（稳定性）的约束优化问题，并用Gauss-Newton法求闭式解，实现自适应离散化，在CIFAR-10上用不到25%训练预算超越所有先前CM。
tags:
  - NeurIPS 2025
  - 图像恢复
  - Consistency Model
  - 自适应离散化
  - 训练效率
  - 单步生成
  - Lagrange乘子法
---

# Adaptive Discretization for Consistency Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.17266](https://arxiv.org/abs/2510.17266)  
**代码**: [GitHub](https://github.com/rainstonee/ADCM)  
**领域**: 图像生成 / 扩散模型  
**关键词**: Consistency Model, 自适应离散化, 训练效率, 单步生成, Lagrange乘子法

## 一句话总结

提出ADCM——通过将一致性模型的离散化步长形式化为局部一致性（可训练性）与全局一致性（稳定性）的约束优化问题，并用Gauss-Newton法求闭式解，实现自适应离散化，在CIFAR-10上用不到25%训练预算超越所有先前CM。

## 研究背景与动机

- **核心问题**: 一致性模型（CM）通过将PF-ODE轨迹上的点映射到端点实现单步生成，但其训练严重依赖相邻轨迹点的离散化策略选择
- **现有方案缺陷**: （1）离散CM（iCT, ECM）依赖手动设计的离散化调度，需要针对不同噪声策略和数据集反复调整；（2）连续CM（sCM）让Δt→0避免离散化，但面临严重的训练不稳定问题；（3）CCM通过PSNR阈值迭代求解，计算量大
- **根本矛盾**: Δt太小→局部一致性好但全局去噪误差大→不稳定；Δt太大→稳定但局部一致性差→难以训练
- **切入点**: 将离散化步长选择形式化为带约束的优化问题，自适应平衡可训练性和稳定性

## 方法详解

### 局部一致性与全局一致性

- **局部一致性**（优化目标）: $\mathcal{L}_\text{local} = \mathbb{E}[\|f_{\theta^-}(\mathbf{x}_t) - f_{\theta^-}(\mathbf{x}_{t-\Delta t})\|_2^2]$，希望最小化→需要小Δt
- **全局一致性**（约束条件）: $\mathcal{L}_\text{global} = \mathbb{E}[\|f_{\theta^-}(\mathbf{x}_{t-\Delta t}) - \mathbf{x}_0\|_2^2] \leq \delta$，控制去噪误差→需要大Δt

两者对Δt施加相反方向的约束。

### 约束优化与Lagrangian松弛

将两个目标统一为：

$$\Delta t^* = \arg\min_{\Delta t} \mathbb{E}[\mathcal{L}_\text{local}(t, \Delta t) + \lambda \mathcal{L}_\text{global}(t, \Delta t)]$$

Lagrange乘子 $\lambda$ 平衡可训练性和稳定性，通常 $\lambda \ll 1$（优先保证可训练性）。

### 统一框架：先前方法是特例

| 方法 | 对应的λ值 |
|------|---------|
| DM（如EDM）| λ→∞（最大步长Δt=t-ε）|
| 连续CM（sCM）| λ=0（最小步长Δt→0）|
| 离散CM（iCT, ECM）| 经验估计 |
| CCM | $\mathcal{L}_\text{local}$设为常数 |

### Gauss-Newton法求闭式解

用一阶Taylor展开近似 $f_{\theta^-}(\mathbf{x}_{t-\Delta t})$，通过JVP高效计算Jacobian方向向量 $\mathbf{v}$，得到闭式解：

$$\Delta t^* = \frac{\lambda}{1+\lambda} \frac{\mathbb{E}[\mathbf{v}^\top(f_{\theta^-}(\mathbf{x}_t) - \mathbf{x}_0)]}{\mathbb{E}[\mathbf{v}^\top \mathbf{v}]}$$

三个直观解释：（1）Jacobian越大→步长越小（输出变化剧烈时需谨慎）；（2）去噪误差越大→步长越大（保证稳定性）；（3）局部与全局优化方向越一致→步长越大。

### 自适应权重函数与损失

权重函数 $w(t) = 1/\mathcal{L}_\text{global}$：全局误差大时降权（避免不稳定），小时增权。最终损失使用Pseudo-Huber度量替代L2以降低方差：

$$\min_\theta \mathbb{E}\left[\frac{\sqrt{\|f_\theta(\mathbf{x}_t) - f_{\theta^-}(\mathbf{x}_{t-\Delta t^*})\|_2^2 + c^2} - c}{\sqrt{\|f_{\theta^-}(\mathbf{x}_{t-\Delta t^*}) - \mathbf{x}_0\|_2^2 + c^2} - c}\right]$$

### 训练流程

交替优化时间分割 $\mathbb{T}$ 和网络参数 $\theta$：每25000步更新一次 $\mathbb{T}$（从t=T出发通过Eq.10迭代到t=ε），中间用单个mini-batch估计期望即可。

## 实验关键数据

### CIFAR-10无条件生成（1-step FID↓）

| 方法 | 训练预算(Mimgs) | FID↓ |
|------|----------------|------|
| ECM | 12.8 | 4.54 |
| ECM | 51.2 | 3.60 |
| iCT | 409.6 | 2.83 |
| sCT (TrigFlow) | 204.8 | 2.85 |
| **ADCM** | **12.8** | **3.16** |
| **ADCM** | **76.8** | **2.80** |

ADCM用12.8M图即优于ECM的51.2M，用76.8M即超越iCT的409.6M（约19%训练预算）。

### ImageNet 64×64类条件生成

| 方法 | 模型大小 | 训练预算 | FID↓ |
|------|---------|---------|------|
| iCT-deep | 2× | 1638.4M | 3.25 |
| ECM | 2× | 12.8M | 3.67 |
| **ADCM** | **2×** | **12.8M** | **3.49** |
| **ADCM** | **2×** | **51.2M** | **3.04** |

ADCM(2×, 12.8M)已超越ECM(2×, 12.8M)并接近iCT-deep(2×, 1638.4M)。

### 训练效率

- 额外计算开销仅约4%（JVP计算+周期性更新$\mathbb{T}$）
- 收敛速度显著快于iCT、ECM、sCT
- 适配Flow Matching无需手动调整：FID 5.14 vs ECM 5.82（12.8M预算）

### λ的影响

- λ过小→过度关注全局一致性→收敛快但最终质量差
- λ过大→过度关注局部一致性→不稳定难收敛
- 最优λ实现训练稳定性和最终性能的平衡

## 亮点与洞察

1. **统一框架的理论优雅**: 将先前所有CM离散化方法（iCT/ECM/sCM/CCM/DM）统一为λ的不同特例，提供了清晰的理论视角
2. **自适应步长的直观解释**: 闭式解揭示了Jacobian、去噪误差、优化方向一致性三个因素如何共同决定最优步长
3. **极高的训练效率**: CIFAR-10上用不到25%预算超越所有先前CM，ImageNet上用3%预算接近iCT-deep，且额外开销仅4%
4. **无需手动调整**: 无论VE SDE还是Flow Matching，同一框架自动适配，不需要针对噪声策略重新设计离散化

## 局限性

1. **依赖预训练DM初始化**: 实验中CM均从预训练EDM初始化（ECM范式），从头训练的效果未验证
2. **高分辨率未充分验证**: ImageNet 512×512实验少且FID较高（10.53/2×/6.4M），与sCT差距较大
3. **Taylor一阶近似的适用性**: 当网络输出变化剧烈时一阶近似可能不准确
4. **λ仍需手动选择**: 虽然论文声称自适应，但λ本身是需要调整的超参数
5. **仅验证图像生成**: 未推广到视频、音频、3D等其他扩散模型应用场景

## 相关工作与启发

- **CM离散化谱系**: iCT（指数递减步长）→ ECM（解耦步长幅度和分布）→ CCM（PSNR阈值迭代求解）→ sCM（Δt→0的连续化）→ ADCM（自适应闭式解）
- **与DM优化的联系**: DM的训练本质上只优化全局一致性（λ→∞），ADCM揭示了CM需要同时考虑局部一致性
- **启发**: 自适应离散化思想可推广到flow matching的ODE求解器步长选择，以及diffusion distillation中的teacher-student步长匹配

## 评分

⭐⭐⭐⭐ — 理论框架优雅（统一先前方法+闭式解），训练效率提升显著（4-20×预算节省），适配性强。不足是高分辨率验证不足，且λ仍需手动调。

<!-- RELATED:START -->

## 相关论文

- [Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)
- [Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [Adaptive Estimation and Learning under Temporal Distribution Shift](../../ICML2025/image_restoration/adaptive_estimation_and_learning_under_temporal_distribution_shift.md)

<!-- RELATED:END -->
