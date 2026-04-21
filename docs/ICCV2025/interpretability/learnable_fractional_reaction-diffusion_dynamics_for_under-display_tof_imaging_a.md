---
title: >-
  [论文解读] Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond
description: >-
  [ICCV 2025][屏下ToF成像] LFRD² 提出一种混合框架，将可学习的时间分数阶反应-扩散方程与神经网络结合，用于屏下 ToF（UD-ToF）深度图恢复。通过分数阶微积分捕获迭代过程中的长期记忆依赖，并引入高效的连续卷积算子替代离散卷积，在 UD-ToF 深度恢复、ToF 去噪和深度超分辨率任务上均取得最优性能。
tags:
  - ICCV 2025
  - 屏下ToF成像
  - 分数阶反应-扩散
  - 连续卷积
  - 深度恢复
  - 物理驱动
---

# Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond

**会议**: ICCV 2025  
**arXiv**: [2511.01704](https://arxiv.org/abs/2511.01704)  
**代码**: [https://github.com/wudiqx106/LFRD2](https://github.com/wudiqx106/LFRD2)  
**领域**: Depth Restoration / Computational Imaging  
**关键词**: 屏下ToF成像, 分数阶反应-扩散, 连续卷积, 深度恢复, 物理驱动

## 一句话总结

LFRD² 提出一种混合框架，将可学习的时间分数阶反应-扩散方程与神经网络结合，用于屏下 ToF（UD-ToF）深度图恢复。通过分数阶微积分捕获迭代过程中的长期记忆依赖，并引入高效的连续卷积算子替代离散卷积，在 UD-ToF 深度恢复、ToF 去噪和深度超分辨率任务上均取得最优性能。

## 研究背景与动机

- **全面屏趋势**推动了屏下传感器的发展：屏下RGB相机已商用，屏下 ToF（UD-ToF）深度相机是下一步
- **TOLED透明面板**对 ToF 相机造成严重退化：信号衰减、多路径干扰（MPI）、时间噪声等，导致深度质量大幅下降
- **传统扩散方法**（P-M 扩散等）利用物理先验进行深度优化，具有鲁棒的适应性和泛化性，但参数建模复杂、计算量大
- **深度学习方法**在高层图像理解和上下文推理方面有优势，但依赖网络架构设计和数据质量，缺乏物理可解释性
- **整数阶算法展开**（algorithm unrolling）将迭代算法映射到深度网络，但使用的是**整数阶微分方程**（IDE），预测状态仅依赖当前状态，忽略了历史信息
- **分数阶微分方程**（FDE）具有**记忆性**——当前状态依赖所有历史状态，更符合真实物理过程
- **核心动机**：能否利用神经网络学习分数阶反应-扩散方程的解，在保持物理可解释性的同时提升深度恢复质量？

## 方法详解

### 整体框架

LFRD² 包含两个阶段：
1. **Deep Initial State Builder (DISB)**：使用现有网络（如 UD-ToFnet）生成初始深度图 $u_0$
2. **深度分数阶反应-扩散模块**：基于 Caputo 分数阶导数进行迭代深度细化

### 关键设计

1. **分数阶反应-扩散动力学**：

    - 采用 Caputo 分数阶导数（阶数 $0 < \alpha < 1$）：
    ${}^C_0 D_t^\alpha u(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t (t-\tau)^{-\alpha} u'(\tau) d\tau$
    - 使用 L1 近似离散化，得到迭代格式：
    $u_{n+1} = u_n + S[\text{div}(g|\nabla u_n|\nabla u_n) + \lambda(u_0 - u_n)] - \sum_{k=1}^n a_k^{(\alpha)}(u_{n+1-k} - u_{n-k})$
    - 其中 $S = \Gamma(2-\alpha)/a_0^\alpha$，$a_k^{(\alpha)} = (k+1)^{1-\alpha} - k^{1-\alpha}$
    - **记忆性**：当前状态 $u_{n+1}$ 依赖所有历史状态 $u_0, ..., u_n$，而非仅依赖 $u_n$
    - **分数阶阶数 $\alpha$ 由神经网络动态生成**，而非预设固定值
    - 扩散项 $\text{div}(g|\nabla u|\nabla u)$ 中 $g(\cdot)$ 由神经网络学习而非使用传统导电率函数
    - 反应项 $\lambda(u_0 - u_n)$ 驱动深度演化向目标状态靠近

2. **高效连续卷积算子**：

    - 传统离散卷积忽略自然场景的连续性
    - 现有连续卷积实现（基于 MLP 的 Neural Field）计算成本高、超参复杂
    - 本文基于重复微分/积分性质：$u * \mathcal{K} = u^{(-n)} * \mathcal{K}^{(n)}$
    - 当 $n=2$ 时，估计核 $\hat{\mathcal{K}}^{(2)}$ 退化为稀疏 Dirac delta
    - **创新点**：不预定义高斯核和控制点，而是由 DISB 直接生成 Dirac delta
    - 信号反导数的高效近似：$u^{(-2)} \approx A \cdot u(x_0, y_0) + B$，系数 $A, B$ 通过三层卷积预测
    - 相比 NFC（Neural Field Convolution），**FLOPs 减少 62%**（7.69G vs 20.5G），速度更快（22.75ms vs 28.42ms）

3. **物理可解释性**：

    - 整个迭代过程编码了时间分数阶反应-扩散方程
    - 分数阶的非局部性质为描述具有记忆效应的动态过程提供了合适框架
    - 神经网络作为分数阶阶数的估计器，可视为物理信息神经网络（PINNs）的一种形式

### 损失函数 / 训练策略

- 使用 Adam 优化器，初始学习率 $1 \times 10^{-4}$，batch size 16
- SUD-ToF 训练 250 epoch，RUD-ToF 训练 1000 epoch
- 原始 $180 \times 240$ 图像裁剪为 $176 \times 240$ patch
- DISB 基于 UD-ToFnet，保持原始设置
- 反应项系数 $\lambda = 0.01$
- 实验在 NVIDIA RTX 3090 上进行

## 实验关键数据

### 主实验 (表格)

**SUD-ToF / RUD-ToF 数据集**：

| 方法 | SUD-ToF MAE↓ | SUD-ToF RMSE↓ | RUD-ToF MAE↓ | RUD-ToF RMSE↓ |
|------|-------------|--------------|-------------|--------------|
| PE-ToF | 9.77 | 15.92 | 21.22 | 48.76 |
| NAFNet | 11.08 | 18.24 | 20.41 | 33.83 |
| Restormer | 9.75 | 14.76 | 18.94 | 31.78 |
| UD-ToFnet | 8.88 | 11.50 | 17.29 | 31.11 |
| **LFRD² (Ours)** | **8.41** | **10.99** | **16.73** | **30.94** |

**FLAT 数据集（ToF 去噪）**：

| 方法 | MAE↓ | RMSE↓ |
|------|------|-------|
| SHARPnet | 4.62 | 10.26 |
| UD-ToFnet | 4.41 | 8.23 |
| **LFRD²** | **4.13** | **7.35** |

**NYUv2 数据集（深度超分辨率，MSE/MAE）**：

| 方法 | 4× | 8× | 16× |
|------|----|----|------|
| DSR-EI | 2.94/0.49 | 13.3/1.19 | 57.0/2.70 |
| **LFRD²** | **2.85/0.47** | **12.8/1.16** | **52.3/2.58** |

### 消融实验 (表格)

**核心组件消融（RUD-ToF）**：

| 配置 | Params/M | FLOPs/G | Speed/ms | MAE↓ | RMSE↓ |
|------|----------|---------|----------|------|-------|
| Baseline (UD-ToFnet) | 2.17 | 8.65 | 15.20 | 17.29 | 31.11 |
| + GRU | +0.18 | +7.62 | 19.89 | 17.02 | 31.09 |
| + LSTM | +0.24 | +10.4 | 22.15 | 16.96 | 31.22 |
| **+ LFRD²** | **+0.18** | **+7.69** | **22.75** | **16.73** | **30.94** |
| w/o FC (整数阶) | +0.01 | +0.41 | 20.67 | 17.00 | 30.99 |
| w/o CC (无连续卷积) | +0.17 | +7.28 | 22.11 | 16.88 | 31.03 |
| NFC | +0.13 | +20.5 | 28.42 | 16.97 | 31.00 |

**分数阶阶数消融**：

| 阶数 $\alpha$ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 可学习(Ours) |
|--------|------|------|------|------|------|------------|
| MAE/mm | 18.12 | 18.38 | 18.86 | 19.01 | 18.29 | **17.62** |
| $\rho_{1.02}$/% | 66.79 | 66.80 | 66.16 | 65.73 | 66.04 | **67.43** |

### 关键发现

- **分数阶 vs 整数阶**：去掉分数阶微积分（w/o FC）后 MAE 从 16.73 升至 17.00，证实记忆性的重要性
- **连续卷积 vs 离散**：去掉连续卷积（w/o CC）后 MAE 从 16.73 升至 16.88
- **vs RNN 变体**：LFRD² 在 MAE 和 RMSE 上均优于 GRU/LSTM，且参数量与 GRU 相当
- **vs NFC**：精度相当但 FLOPs 减少 62%，速度提升 25%
- **可学习阶数优于所有固定阶数**，动态适应不同样本的最优阶数
- **跨任务泛化**：同一框架在 UD-ToF 恢复、ToF 去噪、深度超分辨率三个不同任务上均取得 SOTA
- **即插即用**：DISB 可替换为 PE-ToF、NAFNet、Restormer 等不同基线，都能获得提升

## 亮点与洞察

- **物理驱动+数据驱动的混合设计**：将分数阶PDE嵌入神经网络迭代，既有物理可解释性又有学习灵活性
- **分数阶的记忆性**是关键创新：利用历史迭代状态的加权组合，比仅依赖当前状态的整数阶方法更鲁棒
- **动态分数阶阶数**由网络学习，解决了传统分数阶方法参数选择困难的痛点
- 连续卷积的高效实现（系数预测 + 重复微分）比 MLP-based Neural Field 更实用
- **跨任务适用性好**：同一框架不修改即可用于不同深度恢复任务

## 局限与展望

- 训练策略需要仔细设置以避免数值不稳定（NaN 值），鲁棒性有待提升
- 当前使用显式数值格式，隐式格式可能更稳定高效
- 迭代步数固定，自适应步数控制可能进一步提升效率
- DISB 的选择对最终性能有影响，但如何选择最优初始化器未做深入探讨
- 连续卷积虽然比 NFC 高效，但仍增加了约 50% 的推理时间

## 相关工作与启发

- **Perona-Malik 扩散**：经典的图像增强扩散模型，本文的整数阶基线
- **TNRD (Chen & Pock 2016)**：可训练非线性反应-扩散模型，将时变滤波器参数从数据中学习
- **UD-ToFnet (Qiao et al.)**：UD-ToF 深度恢复的先驱工作，本文的 DISB 基线
- **NFC (Nsampi et al.)**：基于重复微分的连续卷积实现，本文连续卷积的参照方法
- **Algorithm Unrolling**：将迭代算法展开为网络层，LFRD² 的方法论基础
- 启发：分数阶微积分在图像处理中的潜力远未被充分挖掘

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将分数阶反应-扩散与深度学习结合，可学习阶数设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 四个数据集、三种任务、详细消融（核心组件、连续卷积输入、阶数选择）
- **写作质量**: ⭐⭐⭐⭐ — 数学推导严谨，图示清晰，物理解释充分
- **价值**: ⭐⭐⭐⭐ — 对物理驱动深度恢复有方法论贡献，跨任务泛化性好

<!-- RELATED:START -->

## 相关论文

- [Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [On the Effect of Uncertainty on Layer-wise Inference Dynamics](../../ICML2025/interpretability/on_the_effect_of_uncertainty_on_layer-wise_inference_dynamics.md)
- [Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)
- [Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts](../../ICML2025/interpretability/avoiding_leakage_poisoning_concept_interventions_under_distribution_shifts.md)
- [Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](../../NeurIPS2025/interpretability/dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)

<!-- RELATED:END -->
