---
title: >-
  [论文解读] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][科学计算][PINNs] 提出 Spectral PINNsformer (S-Pformer)，用 Fourier 特征嵌入替换 PINNsformer 的编码器，结合仅解码器 Transformer 架构，在减少 18.6% 参数量的同时在多个 PDE benchmark 上取得更优性能，有效缓解了频谱偏置问题。
tags:
  - NeurIPS 2025 (AI for Science Workshop)
  - 科学计算
  - PINNs
  - Transformer
  - Fourier Features
  - Spectral Bias
  - 偏微分方程
---

# Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding

**会议**: NeurIPS 2025 (AI for Science Workshop)  
**arXiv**: [2510.05385](https://arxiv.org/abs/2510.05385)  
**代码**: 已开源（论文中提供链接）  
**领域**: scientific_computing  
**关键词**: PINNs, Transformer, Fourier Features, Spectral Bias, PDE Solving  

## 一句话总结

提出 Spectral PINNsformer (S-Pformer)，用 Fourier 特征嵌入替换 PINNsformer 的编码器，结合仅解码器 Transformer 架构，在减少 18.6% 参数量的同时在多个 PDE benchmark 上取得更优性能，有效缓解了频谱偏置问题。

## 研究背景与动机

**领域背景**: 偏微分方程（PDE）数值求解是科学工程的核心问题，传统方法（有限差分、谱方法）依赖精细网格离散化，计算开销大且难以适应复杂几何。Physics-Informed Neural Networks (PINNs) 将物理约束嵌入损失函数，实现无网格求解。

**现有不足 — 频谱偏置**: 主流 MLP-based PINNs 存在严重的 spectral bias，难以学习 PDE 解中的高频分量，在含丰富多尺度行为的问题上表现不佳。

**现有不足 — 时序依赖**: MLP-PINNs 做逐点预测，无法捕捉 PDE 解的时空关联性，对抛物型和双曲型 PDE（含时间导数）效果受限。

**PINNsformer 的尝试**: Zhao et al. (2024) 提出 PINNsformer，使用编码器-解码器 Transformer 架构通过 self-attention 捕捉时空关系，大幅提升性能。

**PINNsformer 的冗余**: 该编码器-解码器设计源自序列到序列任务（如翻译），但 PINN 场景中输入输出结构相同，编码器引入了不必要的参数冗余和计算开销。

**本文动机**: 用 Fourier 特征嵌入替代编码器来同时解决频谱偏置和参数冗余两大问题，实现更轻量且更强的 Transformer PINN。

## 方法详解

### 整体框架

S-Pformer 架构由三部分组成：(1) 带 Fourier 特征的输入嵌入模块，(2) 仅解码器的多头注意力模块，(3) 线性输出网络。输入先经伪序列生成器展开为时间序列，再通过 Fourier 嵌入和位置嵌入编码多尺度频率信息，最后由解码器的 self-attention 捕捉时空依赖关系。

### 关键设计

#### 模块一：Fourier 特征嵌入 (Fourier Feature Embedding)

- 将归一化后的输入坐标 $\tilde{\mathbf{z}} = (\tilde{\mathbf{x}}, \tilde{t}) \in [0,1]^{d_{in}}$ 通过随机投影矩阵 $\mathbf{B} \sim \mathcal{N}(0, \mathbf{I})$ 映射到高维频率空间
- Fourier 嵌入: $E_f(\tilde{\mathbf{z}}) = \theta_f([\sin(2\pi\mathbf{B}\tilde{\mathbf{z}}), \cos(2\pi\mathbf{B}\tilde{\mathbf{z}})])$，其中 $d_{\text{mapping}}$ 控制频率带数量（默认 64）
- 位置嵌入: $E_p(\tilde{\mathbf{z}}) = \theta_p(\tilde{\mathbf{z}})$，线性变换保留空间-时间局部性
- 最终嵌入: $E(\tilde{\mathbf{z}}) = E_f(\tilde{\mathbf{z}}) + E_p(\tilde{\mathbf{z}})$
- **核心思想**: Fourier 特征编码全局周期性模式以捕捉多尺度 PDE 解的振荡行为，位置嵌入保留局部时空关系；两者互补地替代了原编码器和时空混合器

#### 模块二：仅解码器 Transformer (Decoder-Only Transformer)

- 每层包含: Wavelet 激活 → 多头自注意力 → 残差连接 → Wavelet 激活 → 前馈网络 → 残差连接
- Wavelet 激活函数: $\text{Wavelet}(z) = \omega_1 \sin(z) + \omega_2 \cos(z)$，$\omega_1, \omega_2$ 可学习，替代 ReLU/LayerNorm（后者对 PINNs 不友好，导数不连续）
- 自注意力直接应用于嵌入坐标，在无编码器的情况下维持对时间依赖 PDE 的建模能力
- 默认配置: $N=1$ 层, $n_{\text{heads}}=2$, $d_{\text{ff}}=512$, $d_{\text{emb}}=32$

#### 模块三：NTK 损失权重自适应 (NTK Learning Scheme)

- 计算各损失分量的 Jacobian $J_i$ 和 NTK trace: $K_i = \text{Tr}(J_i J_i^\top)$
- 损失权重反比于 NTK trace: $\lambda_i = \frac{\sum K}{K_i}$，高灵敏度分量获得更小权重
- 每 50 次迭代更新一次权重，确保 PDE 残差、初始条件、边界条件三项损失均衡收敛

### 损失函数

标准 PINN 三项损失:

$$\mathcal{L}(u_\theta) = \frac{\lambda_1}{N_\mathcal{F}} \sum \|\mathcal{F}(u_\theta)\|^2 + \frac{\lambda_2}{N_\mathcal{I}} \sum \|\mathcal{I}(u_\theta)\|^2 + \frac{\lambda_3}{N_\mathcal{B}} \sum \|\mathcal{B}(u_\theta)\|^2$$

其中 $\lambda_1, \lambda_2, \lambda_3$ 由 NTK 方法动态调节。优化器使用 L-BFGS (Strong-Wolfe line search)，训练 1000 次迭代。

## 实验关键数据

### 主实验：Transformer 架构对比 (Table 2)

| 模型 | PDE 类型 | rMAE | rMSE | 训练时间 |
|------|----------|------|------|----------|
| Pformer | Convection | 0.018 | 0.020 | 0:17:53 |
| Pformer | 1D-Reaction | 7.38e-3 | 0.163 | 0:03:59 |
| Pformer | 1D-Wave | 0.083 | 0.091 | 1:11:45 |
| Pformer | Navier-Stokes | 0.091 | 0.085 | 2:17:09 |
| DO-Pformer | Convection | 0.025 | 0.029 | 0:11:41 |
| DO-Pformer | 1D-Wave | 0.015 | 0.017 | 0:37:48 |
| **S-Pformer** | **Convection** | **0.016** | **0.018** | 0:14:29 |
| **S-Pformer** | **1D-Reaction** | **1.15e-3** | **2.98e-3** | 0:03:48 |
| **S-Pformer** | **1D-Wave** | **6.94e-3** | **7.01e-3** | 0:42:40 |
| **S-Pformer** | **Navier-Stokes** | **0.079** | **0.071** | 1:03:55 |

S-Pformer 在全部 4 个 benchmark 上 rMAE/rMSE 均优于 Pformer，且 Navier-Stokes 训练时间缩短 54%。

### 消融实验与频谱分析 (Table 3 + Table 1)

| 频率带 | S-Pformer MAE | DO-Pformer MAE | Pformer MAE |
|--------|---------------|----------------|-------------|
| Very Low ($f < 0.3f_n$) | 0.1401 | 0.1940 | 0.1400 |
| Low ($0.3f_n \leq f < 0.5f_n$) | **0.0904** | 0.1683 | 0.1764 |
| Mid ($0.5f_n \leq f < 0.7f_n$) | **0.0302** | 0.0354 | 0.0363 |
| High ($0.7f_n \leq f < 0.9f_n$) | **0.0110** | 0.0157 | 0.0155 |
| Very High ($f \geq 0.9f_n$) | **0.0093** | 0.0136 | 0.0133 |

参数量对比: Pformer 453,561 → S-Pformer 369,039 (**减少 18.6%**)。DO-Pformer 仅去掉编码器换线性层，缺乏 Fourier 特征，在低频和中频段误差明显大于 S-Pformer，证实了 Fourier 特征对频谱偏置的缓解作用。

### 优化后 S-Pformer vs 优化后 MLP-PINN (Table 4)

| 问题 | 模型 | rMAE | rMSE | 参数量 |
|------|------|------|------|--------|
| Convection | MLP-PINN | 0.663 | 0.745 | 66,561 |
| Convection | **S-Pformer** | **0.015** | **0.018** | 305,551 |
| 1D-Reaction | MLP-PINN | 0.014 | 0.028 | 1,052,673 |
| 1D-Reaction | **S-Pformer** | **1.09e-3** | **2.15e-3** | 167,471 |
| 1D-Wave | MLP-PINN | 0.023 | 0.023 | 2,365,441 |
| 1D-Wave | **S-Pformer** | **2.89e-3** | **2.94e-3** | 247,823 |
| Navier-Stokes | MLP-PINN | **0.045** | **0.046** | 264,706 |
| Navier-Stokes | S-Pformer | 0.057 | 0.062 | 149,680 |

### 关键发现

1. S-Pformer 在 Convection 上 rMAE 仅为 MLP-PINN 的 1/44，且参数量更少（305K vs 不适用相同量级）
2. 1D-Reaction 上 S-Pformer 以 1/6 参数量（167K vs 1.05M）取得 13 倍精度提升
3. 高频段误差降低约 30%（相比 DO-Pformer），直接证明 Fourier 特征对抗频谱偏置
4. Navier-Stokes 是唯一 MLP 略优的问题，因其含数据驱动分量而非纯物理约束

## 亮点与洞察

- **「减法」设计哲学**: 不是叠加更多模块，而是移除冗余编码器并用更有针对性的 Fourier 嵌入替代，挑战了 "bigger is better" 范式
- **频谱偏置的显式解决**: 通过随机 Fourier 特征将低维输入投射到多频率空间，从根本上赋予网络表达高频函数的能力
- **Wavelet 激活函数**: 用可学习的 sin/cos 加权组合替代 ReLU，物理上更契合 PDE 解的周期性特征
- **NTK 自适应权重**: 从梯度灵敏度角度动态平衡多任务损失，比手动调参更原则性

## 局限性

1. **Navier-Stokes 略逊 MLP**: 在含数据驱动分量的问题上，纯 Transformer 架构尚未充分发挥优势
2. **超参敏感性**: 优化后性能提升显著（如 1D-Wave rMAE 从 6.94e-3 降到 2.89e-3），说明默认超参并非最优，需要 Bayesian 搜索
3. **问题规模有限**: 仅在 1D/2D 经典 PDE 上评测，未涉及高维、多物理场耦合或复杂几何
4. **训练效率**: 仍依赖 L-BFGS 优化器和 NTK 权重计算，后者引入额外的 Jacobian 计算开销

## 相关工作与启发

- **PINNsformer (Zhao et al., 2024)**: 本文的直接基础，编码器-解码器 Transformer PINN
- **Fourier Features (Tancik et al., 2020)**: 证明随机 Fourier 特征可帮助网络学习高频函数
- **NTK for PINNs (Wang et al., 2020/2021)**: 从神经切线核视角分析 PINN 训练失败原因及频谱偏置
- **启发**: Fourier 嵌入 + 仅解码器 Transformer 的范式可推广到其他科学计算场景（分子动力学、天气预测等）

## 评分

- 新颖性: ⭐⭐⭐ — 核心思路是去编码器 + 加 Fourier 特征，各组件已有先例但组合有意义
- 实验充分度: ⭐⭐⭐ — 4 个 benchmark + 频谱分析 + 消融 + 超参优化，较充分但规模偏小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，消融设计合理
- 价值: ⭐⭐⭐ — 对 PINN 社区有实用参考价值，但应用范围和影响力有限

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](../../ICML2025/scientific_computing/differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)
- [\[ICML 2025\] Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators](../../ICML2025/scientific_computing/maximal_update_parametrization_and_zero-shot_hyperparameter_transfer_for_fourier.md)

<!-- RELATED:END -->
