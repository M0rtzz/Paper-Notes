---
title: >-
  [论文解读] SADA: Stability-guided Adaptive Diffusion Acceleration
description: >-
  [ICML2025][图像生成][扩散模型加速] 提出基于ODE轨迹二阶差分的稳定性准则（Stability Criterion），统一调控步级（step-wise）和token级（token-wise）稀疏决策，在SD-2/SDXL/Flux上实现≥1.8×加速且LPIPS≤0.10、FID≤4.5，显著优于DeepCache和AdaptiveDiffusion。
tags:
  - ICML2025
  - 图像生成
  - 扩散模型加速
  - ODE求解器
  - 稳定性准则
  - 自适应稀疏
  - 无训练加速
  - Token剪枝
  - 特征缓存
---

# SADA: Stability-guided Adaptive Diffusion Acceleration

**会议**: ICML2025  
**arXiv**: [2507.17135](https://arxiv.org/abs/2507.17135)  
**代码**: [GitHub](https://github.com/Ting-Justin-Jiang/sada-icml)  
**领域**: 扩散加速 / Diffusion Acceleration  
**关键词**: 扩散模型加速, ODE求解器, 稳定性准则, 自适应稀疏, 无训练加速, Token剪枝, 特征缓存

## 一句话总结

提出基于ODE轨迹二阶差分的稳定性准则（Stability Criterion），统一调控步级（step-wise）和token级（token-wise）稀疏决策，在SD-2/SDXL/Flux上实现≥1.8×加速且LPIPS≤0.10、FID≤4.5，显著优于DeepCache和AdaptiveDiffusion。

## 研究背景与动机

扩散模型在图像/视频/音频生成中取得显著成果，但推理效率受两大瓶颈制约：

**迭代去噪过程**：需要数十步采样，每步都需完整前向传播

**注意力计算的二次复杂度**：高分辨率下自注意力计算代价巨大

现有无训练加速方法主要分两类：

- **减少推理步数**：DDIM、DPM-Solver等高阶ODE求解器
- **减少单步计算量**：DeepCache（步级缓存）、Token Merging/Pruning（token级）

但这两类方法存在明显的**保真度差异（fidelity gap）**，原因在于：

- **(a)** 固定或预搜索的稀疏模式无法适应每个prompt不同的去噪轨迹
- **(b)** 这些方法未利用底层ODE公式及其数值求解器的信息

SADA 正是为解决这两个问题而提出的。

## 方法详解

### 核心思想：稳定性准则统一调控

SADA 将扩散加速问题建模为**稳定性预测问题**，核心是利用ODE轨迹上的精确梯度信息 $y_t = \frac{dx_t}{dt}$ 来度量去噪过程的局部动态稳定性。

### 1. 稳定性准则（Stability Criterion）

定义ODE轨迹的二阶差分 $\Delta^{(2)} y_t$ 作为稳定性指标。在时间步 $t$，若满足：

$$
(x_{t-1} - \hat{x}_{t-1}) \cdot \Delta^{(2)} y_t < 0
$$

则该步被判定为**稳定的**，可以进行加速。其中 $\hat{x}_{t-1}$ 是三阶外推估计值。

- 当准则返回 **True**（稳定）→ 执行**步级缓存辅助剪枝**（step-wise / multistep-wise）
- 当准则返回 **False**（不稳定）→ 执行**token级缓存辅助剪枝**（token-wise）

### 2. 步级缓存辅助剪枝（Step-wise Cache-Assisted Pruning）

提供两种近似方案：

**（a）Step-wise 近似（Adams-Moulton法）**：利用三阶Adams-Moulton方法沿ODE轨迹外推：

$$
\hat{x}_{t-1} = x_t - \frac{5\Delta t}{6} y_t - \frac{5\Delta t}{6} y_{t+1} + \frac{2\Delta t}{3} y_{t+2}
$$

局部截断误差为 $\mathcal{O}(\Delta t^2)$，与简单的有限差分相比具有更低的均值误差和更小的标准差。

**（b）Multistep-wise 近似（Lagrange插值）**：当轨迹进入稳定区域后，采用均匀步级剪枝+Lagrange插值。缓存每隔 $k$ 步的 $x_0^t$，对跳过的步用插值重建：

$$
\hat{x}_0^t = \sum_{i \in I} \left( \prod_{j \in I \setminus \{i\}} \frac{t - t_j}{t_i - t_j} \right) x_0^{t_i}
$$

插值误差为 $\mathcal{O}(h^{k+1})$。

### 3. Token级缓存辅助剪枝（Token-wise Cache-Assisted Pruning）

当步级稳定性准则返回False时，在更细粒度的token级别评估稳定性：

- 将token分为**不稳定集** $\mathcal{I}_{\text{fix}}$（需完整计算）和**稳定集** $\mathcal{I}_{\text{reduce}}$（用缓存近似）
- 仅对 $\mathcal{I}_{\text{fix}}$ 中的token执行注意力计算
- $\mathcal{I}_{\text{reduce}}$ 中的token用上一步的缓存表示 $\mathcal{C}_l$ 替代
- 每次计算后增量更新缓存

### 4. 理论保证

- **Theorem 3.1**：Lagrange外推误差 $R_k(h) = \mathcal{O}(h^k)$
- **Theorem 3.6**：重建误差 $\|\hat{x}_0^t - x_0^t\| = \mathcal{O}(\Delta t) + \mathcal{O}(\Delta x_t)$
- **Theorem 3.2-3.3**：证明采样轨迹的连续性和去噪器的一致性

## 实验关键数据

### 主实验：MS-COCO 2017 定量结果

| 模型 | 求解器 | 方法 | PSNR↑ | LPIPS↓ | FID↓ | 加速比 |
|------|--------|------|-------|--------|------|--------|
| SD-2 | DPM++ | DeepCache | 17.70 | 0.271 | 7.83 | 1.43× |
| SD-2 | DPM++ | AdaptiveDiffusion | 24.30 | 0.100 | 4.35 | 1.45× |
| SD-2 | DPM++ | **SADA** | **26.34** | **0.094** | **4.02** | **1.80×** |
| SDXL | DPM++ | DeepCache | 21.30 | 0.255 | 8.48 | 1.74× |
| SDXL | DPM++ | AdaptiveDiffusion | 26.10 | 0.125 | 4.59 | 1.65× |
| SDXL | DPM++ | **SADA** | **29.36** | **0.084** | **3.51** | **1.86×** |
| Flux | Flow | TeaCache | 19.14 | 0.216 | 4.89 | 2.00× |
| Flux | Flow | **SADA** | **29.44** | **0.060** | **1.95** | **2.02×** |

### 少步采样消融

| 模型 | 求解器 | 步数 | PSNR↑ | LPIPS↓ | FID↓ | 加速比 |
|------|--------|------|-------|--------|------|--------|
| SD-2 | DPM++ | 50 | 26.34 | 0.094 | 4.02 | 1.80× |
| SD-2 | DPM++ | 25 | 28.15 | 0.073 | 3.13 | 1.48× |
| SD-2 | DPM++ | 15 | 29.84 | 0.072 | 3.05 | 1.24× |
| SDXL | DPM++ | 50 | 29.36 | 0.084 | 3.51 | 1.86× |
| SDXL | DPM++ | 25 | 30.84 | 0.073 | 2.80 | 1.52× |

步数减少时保真度反而提升（误差累积更少），同时仍能提供~1.25-1.5×额外加速。

### 跨模态/跨任务

- **MusicLDM 音频生成**：1.81×加速，频谱LPIPS仅~0.01
- **ControlNet 可控生成**：1.41×加速，无需任何修改即可即插即用

## 亮点与洞察

1. **理论创新**：首次将ODE数值求解器与稀疏感知架构优化直接桥接，用稳定性准则统一步级和token级加速决策
2. **自适应分配**：不同prompt自动获得不同的稀疏模式，无需手动调参或预搜索
3. **有原则的近似**：利用Adams-Moulton法和Lagrange插值提供误差有界的近似方案，而非简单地复用噪声
4. **广泛兼容**：跨架构（UNet/DiT）、跨求解器（Euler/DPM++）、跨模态（图像/音频）、跨任务（ControlNet）
5. **即插即用**：无训练、无额外超参数调优，直接作为采样过程的插件

## 局限与展望

1. **视频生成未验证**：论文未在视频扩散模型（如Sora架构）上验证效果
2. **极少步场景加速有限**：15步时加速比降至~1.25×，边际收益递减
3. **稳定性准则依赖历史缓存**：前几步需完整计算来积累梯度历史，存在冷启动开销
4. **Token剪枝 vs Token Merging**：论文选择了token pruning而非merging（附录分析merging是低通滤波器），但在某些场景merging可能更优
5. **单GPU评测**：未报告多GPU分布式场景下的加速效果

## 相关工作与启发

- **DeepCache** (Ma et al., 2024)：缓存UNet中间层特征，固定间隔复用，SADA的自适应策略显著优于其固定模式
- **AdaptiveDiffusion** (Ye et al., 2024)：用三阶差分判断是否跳步，但直接复用噪声无纠正，SADA引入ODE梯度做原则性修正
- **TeaCache** (Liu et al., 2025)：引入误差累积阈值做缓存决策，SADA在Flux上将其FID从4.89降至1.95
- **DPM-Solver** 系列 (Lu et al., 2022)：高阶ODE求解器，SADA与之正交互补
- **Token Merging** (Bolya & Hoffman, 2023)：合并相似token减少注意力计算，SADA选择pruning+cache的方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 稳定性准则统一步级/token级决策是全新范式，ODE求解器与架构稀疏的桥接具有理论深度
- 实验充分度: ⭐⭐⭐⭐ — 三个主流模型×两种求解器×多步数消融+跨模态验证，较为全面
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，符号体系清晰，图表直观
- 价值: ⭐⭐⭐⭐⭐ — 实用的即插即用加速方案，1.8-2×加速+高保真度，对工业部署有直接价值

<!-- RELATED:START -->

## 相关论文

- [RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](../../CVPR2025/image_generation/rayflow_instance-aware_diffusion_acceleration_via_adaptive_flow_trajectories.md)
- [Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models](morse_dual-sampling_for_lossless_acceleration_of_diffusion_models.md)
- [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [Multidimensional Adaptive Coefficient for Inference Trajectory Optimization in Flow and Diffusion](multidimensional_adaptive_coefficient_for_inference_trajectory_optimization_in_f.md)

<!-- RELATED:END -->
