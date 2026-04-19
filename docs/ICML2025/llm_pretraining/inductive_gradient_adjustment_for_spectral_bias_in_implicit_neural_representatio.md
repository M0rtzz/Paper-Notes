---
title: >-
  [论文解读] Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations
description: >-
  [ICML2025][Implicit Neural Representations] 本文从 NTK 线性动力学模型出发，提出 Inductive Gradient Adjustment (IGA) 方法，通过归纳泛化 eNTK 梯度变换矩阵，**有目的性**地缓解 MLP 的频谱偏差，使 INR 在百万级数据点上也能高效学习高频细节。
tags:
  - ICML2025
  - Implicit Neural Representations
  - Spectral Bias
  - Neural Tangent Kernel
  - Gradient Adjustment
  - Training Dynamics
---

# Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations

**会议**: ICML2025  
**arXiv**: [2410.13271](https://arxiv.org/abs/2410.13271)  
**代码**: [LabShuHangGU/IGA-INR](https://github.com/LabShuHangGU/IGA-INR)  
**领域**: 隐式神经表示 (INR) / 频谱偏差 (Spectral Bias)  
**关键词**: Implicit Neural Representations, Spectral Bias, Neural Tangent Kernel, Gradient Adjustment, Training Dynamics

## 一句话总结

本文从 NTK 线性动力学模型出发，提出 Inductive Gradient Adjustment (IGA) 方法，通过归纳泛化 eNTK 梯度变换矩阵，**有目的性**地缓解 MLP 的频谱偏差，使 INR 在百万级数据点上也能高效学习高频细节。

## 研究背景与动机

- **隐式神经表示 (INR)** 用 MLP 将离散信号参数化为连续函数，广泛用于图像拟合、3D 重建、新视角合成等任务。
- **频谱偏差 (Spectral Bias)**：vanilla ReLU-MLP 倾向于先学低频分量，对高频细节收敛极慢，导致纹理模糊、边缘丢失。
- 现有缓解方案分两类：
    - **结构改进**：位置编码 (PE)、周期激活 (SIREN)、Gabor 小波激活等——需引入复杂推理结构。
    - **训练动态调整**：Fourier Reparameterized Training (FR)、Batch Normalization (BN)——不改变推理结构但缺乏理论指导，效果不稳定。
- **核心问题**：如何在理论指导下，**有目的地**调节训练动态以克服频谱偏差？NTK 矩阵是联系训练动态与频谱偏差的关键桥梁，但存在两大障碍：
    1. NTK 矩阵在深层网络中**无解析表达式**。
    2. NTK 矩阵大小随数据量 $N$ **二次增长**（如 Kodak 图像需 >8192 GiB 内存）。

## 方法详解

### 3.1 频谱偏差与训练动态的连接

标量信号 $\bm{y} \in \mathbb{R}^N$，MLP 参数 $\Theta$，残差 $\bm{r}_t = f(\bm{X};\Theta_t) - \bm{y}$。在宽网络 + 小学习率条件下，训练动态近似为线性模型：

$$\bm{r}_t = (\bm{I} - \eta \bm{K}) \bm{r}_{t-1}$$

对 NTK 矩阵 $\bm{K}$ 做特征分解 $\bm{K} = \sum_i \lambda_i \bm{v}_i \bm{v}_i^\top$，则：

$$\|\bm{r}_t\|_2 = \sqrt{\sum_{i=1}^{N} (1-\eta\lambda_i)^{2t} (\bm{v}_i^\top \bm{y})^2}$$

**关键洞察**：特征值 $\lambda_i$ 大则对应方向收敛快；vanilla MLP 中高频方向对应小特征值 → 高频收敛极慢 → 频谱偏差。使 $\bm{K}$ 谱更均匀即可缓解偏差。

### 3.2 Inductive Gradient Adjustment (IGA)

**Step 1 — NTK 梯度调整框架**：引入变换矩阵 $\bm{S}$ 调整梯度：

$$\Theta_{t+1} = \Theta_t - \eta \nabla_\Theta f(\bm{X};\Theta) \bm{S} \bm{r}_t$$

其中 $\bm{S} = \sum_i (g_i(\lambda_i)/\lambda_i) \bm{v}_i \bm{v}_i^\top$，修改后的收敛速率为 $\{g_i(\lambda_i)\}$。

**Step 2 — eNTK 替代 NTK**（解决无解析式问题）：用经验 NTK $\tilde{\bm{K}} = \nabla_{\Theta_t} f^\top \nabla_{\Theta_t} f$ 代替理论 NTK。Theorem 3.1 证明：随网络宽度 $m$ 增大，eNTK 特征值/特征向量一一收敛到 NTK，因此 eNTK-based 调整与 NTK-based 效果渐近等价。

**Step 3 — 归纳泛化**（解决维度灾难）：将 $N$ 个数据分为 $n$ 组（每组 $p$ 个，$N=np$），从每组采样 1 个点构成 $\bm{X}_e$（$|\bm{X}_e|=n \ll N$），在 $\bm{X}_e$ 上计算小尺寸 eNTK $\tilde{\bm{K}}_e \in \mathbb{R}^{n \times n}$，构造变换矩阵 $\tilde{\bm{S}}_e$，然后将其**归纳泛化**到全量数据的梯度：

$$\Theta_{t+1} = \Theta_t - \eta \sum_{i=1}^{p} \nabla_{\Theta_t} f(\bm{X}_i, \Theta_t) \tilde{\bm{S}}_e \bm{r}_t^i$$

Theorem 3.2 保证：泛化误差 $\epsilon_1 + \epsilon_2$ 随宽度 $m$ 增大而减小。

### 变换矩阵构造

对 $\tilde{\bm{K}}_e$ 特征分解后，将前 $\text{end}$ 个特征值均衡为 $\tilde{\tilde{\lambda}}_{\text{start}}$：

$$\tilde{\bm{S}_e} = \sum_{i=\text{start}}^{\text{end}} \frac{\tilde{\tilde{\lambda}}_{\text{start}}}{\tilde{\tilde{\lambda}}_i} \tilde{\tilde{\bm{v}}}_i \tilde{\tilde{\bm{v}}}_i^\top + \sum_{i \notin [\text{start},\text{end}]} \tilde{\tilde{\bm{v}}}_i \tilde{\tilde{\bm{v}}}_i^\top$$

- $\text{end}$ 越大 → 频谱越均匀 → 对高频的提升越强（可控调节）。
- Adam 优化器下用 $\tilde{\tilde{\lambda}}_{\text{end}+1}$ 替代 $\tilde{\tilde{\lambda}}_{\text{start}}$ 以保证收敛稳定性。

### 采样策略

- 1D/2D 信号：按相邻坐标分组（不重叠区间/patch）。
- 高维信号：展平后沿第一维分组。
- 每组选残差最大的点进入 $\bm{X}_e$。

## 实验关键数据

### 2D 彩色图像拟合（Kodak 数据集）

| 方法 | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS ↓ |
|------|--------|--------|-----------|---------|
| ReLU (Vanilla) | 21.78 | 0.4833 | 0.6521 | 0.6302 |
| ReLU + FR | 22.14 | 0.4919 | 0.6800 | 0.6315 |
| ReLU + BN | 22.55 | 0.5004 | 0.7090 | 0.6182 |
| **ReLU + IGA** | **23.00** | **0.5126** | **0.7383** | **0.5549** |
| PE (Vanilla) | 28.64 | 0.7832 | 0.9466 | 0.2223 |
| PE + FR | 31.65 | 0.8167 | 0.9564 | 0.1869 |
| PE + BN | 28.78 | 0.8030 | 0.9554 | 0.2346 |
| **PE + IGA** | **32.46** | **0.8822** | **0.9752** | **0.0938** |
| SIREN (Vanilla) | 32.65 | 0.8975 | 0.9818 | 0.0807 |
| SIREN + FR | 32.61 | 0.8991 | 0.9820 | 0.0813 |
| **SIREN + IGA** | **33.48** | **0.9121** | **0.9847** | **0.0668** |

### 3D 形状表示

| 方法 | IOU ↑ | Chamfer Distance ↓ |
|------|-------|---------------------|
| ReLU (Vanilla) | 9.647e-1 | 5.936e-6 |
| **ReLU + IGA** | **9.733e-1** | **5.487e-6** |
| PE (Vanilla) | 9.942e-1 | 5.123e-6 |
| **PE + IGA** | **9.970e-1** | **5.108e-6** |
| SIREN (Vanilla) | 9.889e-1 | 5.688e-6 |
| **SIREN + IGA** | **9.897e-1** | **5.157e-6** |

### 关键实验发现

- IGA 在 **所有架构**（ReLU / PE / SIREN）和 **所有指标** 上均一致优于 FR、BN。
- 增加均衡特征值数量 → 对高频的提升单调增强（可控性验证）。
- 组大小 $p$ 增大时，IGA 性能轻微下降但始终优于 baseline，验证归纳泛化的鲁棒性。

## 亮点与洞察

1. **理论驱动**：首次从 NTK 线性动力学出发，给出调节频谱偏差的**定量策略**，而非 FR/BN 那样的经验发现。
2. **可控调节**：通过 $\text{end}$ 参数精确控制对高频的提升力度，避免过度矫正。
3. **架构无关**：IGA 是训练时的梯度变换，不改变推理结构，可与任意 INR 架构（ReLU / PE / SIREN）自由组合。
4. **高效采样**：通过归纳泛化将 $O(N^2)$ 的 eNTK 计算降到 $O(n^2)$（$n \ll N$），使百万级数据点可行。
5. **两个定理**提供严格的渐近保证：eNTK → NTK 等价性（Thm 3.1）；归纳泛化误差界（Thm 3.2）。

## 局限与展望

1. **额外计算开销**：每步需在采样子集上计算 eNTK 及其特征分解，虽然 $n \ll N$ 但仍有非零 overhead。
2. **理论分析局限于两层网络**：Theorem 3.1/3.2 基于两层网络分析，深层网络的严格保证尚未给出。
3. **分组策略简单**：目前按相邻坐标分组，对非规则采样（如 NeRF 光线）的最优分组策略未探讨。
4. **超参数选择**：$n$、$p$、$\text{end}$ 的最优值需任务相关调优，缺乏全自动选择机制。
5. **SIREN + BN 不兼容**问题被指出但未解决。

## 相关工作与启发

- **Fourier Reparameterized Training (FR)** [Shi et al., 2024a]：在 Fourier 域学参数，可改善 INR 但缺乏理论指导。
- **BN for INR** [Cai et al., 2024]：经典 BN 也能提升 INR 性能，但与 SIREN 不兼容。
- **NTK 谱分析** [Ronen et al., 2019; Tancik et al., 2020]：用 NTK 谱解释频谱偏差，本文更进一步将其转化为可操作的训练策略。
- **Geifman et al., 2023**：首次尝试修改 NTK 谱加速收敛，但仅限 toy 设置；本文通过归纳泛化将其扩展到实际规模。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将 eNTK 谱调控转化为实用的训练时梯度变换，理论链条完整
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 2D 图像、3D 形状、合成数据，与 FR/BN 全面对比
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，公式到算法的转化流畅
- 价值: ⭐⭐⭐⭐ — INR 领域有实际应用价值，可控调节频谱偏差是重要贡献

<!-- RELATED:START -->

## 相关论文

- [Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)
- [Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](../../NeurIPS2025/llm_pretraining/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)
- [Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](../../NeurIPS2025/llm_pretraining/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)

<!-- RELATED:END -->
