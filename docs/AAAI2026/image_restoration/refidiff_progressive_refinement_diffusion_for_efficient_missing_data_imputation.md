---
title: >-
  [论文解读] RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation
description: >-
  [AAAI 2026][图像恢复][missing data imputation] 提出 RefiDiff 框架，通过渐进式 refinement 策略统一 predictive 和 generative 两种缺失值填补范式，结合 Mamba-based denoising network 实现高维混合类型表格数据的高效高精度填补，在 MNAR 场景下尤其突出。
tags:
  - AAAI 2026
  - 图像恢复
  - missing data imputation
  - 扩散模型
  - Mamba
  - tabular data
  - MNAR
---

# RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation

**会议**: AAAI 2026  
**arXiv**: [2505.14451](https://arxiv.org/abs/2505.14451)  
**代码**: [GitHub](https://github.com/Atik-Ahamed/RefiDiff)  
**领域**: Data Imputation / Tabular Data  
**关键词**: missing data imputation, diffusion model, Mamba, tabular data, MNAR

## 一句话总结

提出 RefiDiff 框架，通过渐进式 refinement 策略统一 predictive 和 generative 两种缺失值填补范式，结合 Mamba-based denoising network 实现高维混合类型表格数据的高效高精度填补，在 MNAR 场景下尤其突出。

## 背景与动机

- 缺失值在高维混合类型数据集中普遍存在，涉及 MCAR/MAR/MNAR 三种机制，其中 MNAR 最具挑战性
- **Predictive 方法**（如回归/分类器）：高效、确定性，但缺乏不确定性建模，偏向 local view
- **Generative 方法**（如 diffusion）：不确定性感知，但计算密集，偏向 global view
- 现有方法很少将两种范式有效统一，且在 MNAR、高维混合类型数据上表现受限

## 核心问题

如何将 predictive 方法的局部精确性与 generative 方法的全局分布建模能力结合，同时保持计算效率和无需超参调优？

## 方法详解

### 整体框架

四阶段流水线：Pre-processing → Warm-up Refinement → Diffusion Imputation → Post-processing & Polishing

### 关键设计

**1. Pre-processing**: 分类特征 binary encoding，数值特征标准化（仅用 observed 计算 $\mu, \sigma$），缺失位置填零

**2. Warm-up Refinement（Pre-refinement）**: 对每个特征列 $f_j$ 训练轻量模型 $\theta_1^{(j)}$（XGBoost），单遍扫描填补所有缺失值：
$$\hat{Z}_{i,j} = \begin{cases} Z_{i,j}, & M_{i,j}=0 \\ \theta_1^{(j)}(Z_{i,\setminus j}), & M_{i,j}=1 \end{cases}$$

关键性质：Non-overwriting、Well-defined mapping、One-pass completion

**3. Diffusion Module**: 采用 VE SDE 连续时间扩散，训练 Mamba-based denoising network $\theta_2$，diamond 结构（2个 up-sampling + 2个 down-sampling 残差块），每个块含 Mamba 层 + FC + PE + LayerNorm + Dropout。损失函数（EDM loss）：
$$\mathcal{L}_{\text{SM}}(\theta_2) = \mathbb{E}_{X_0,\varepsilon,t}\left[\|\theta_2(X_t,t,M) - \nabla_{X_t}\log p(X_t|X_0)\|_2^2\right]$$

推理时进行 $N$ 次 reverse diffusion 取平均，observed 位置始终 clamp 不变。

**理论保证**: KL 散度上界为 $\text{KL} \leq C_1 T \varepsilon_\theta^2 + C_2 \delta t + C_3 / N$

**4. Post-refinement（Polishing）**: 对 diffusion 输出再做一遍 column-wise regression 修正残余噪声

## 实验关键数据

| 方法 | MNAR MAE/RMSE (IS) | MCAR MAE/RMSE (IS) | MAR MAE/RMSE (IS) | Rank |
|------|-------------------|-------------------|------------------|------|
| DIFFPUTER | 37.27/86.86 | 31.72/63.49 | 39.15/90.95 | 2.67 |
| ReMasker | 39.66/80.23 | 35.84/65.19 | 38.39/78.82 | 3.00 |
| **RefiDiff** | **34.49/78.83** | **31.41/63.16** | **34.52/78.22** | **1.17** |

- 9个 real-world 数据集，数值+分类特征综合评估
- 分类精度同样排名第一（平均 rank 1.17）
- 比 DIFFPUTER **快 4 倍**，参数量更少
- 去掉 diffusion 后 OOS RMSE 从 73.82 升至 91.80（MAR），70.12 升至 81.07（MNAR）

## 亮点

- **范式统一**: 首次将 predictive + generative 通过渐进式 refinement 无缝融合
- **Mamba 替代 Transformer**: 线性复杂度 + Transformer 级表达力，4x 提速
- **Plug-and-play**: 无需针对不同数据集调整架构或超参
- **理论保障**: 提供 KL 散度量化上界，证明 conditional sampling 的正确性
- **MNAR 显著优势**: 在最具挑战性的 MNAR 场景下提升最为明显

## 局限与展望

- 分类特征采用 binary encoding 再做连续扩散，可能损失语义保真度
- 仅在 UCI 等中等规模数据集验证，百万级超大规模场景待探索
- Warm-up 仅做单遍扫描，对极端高缺失率可能不够充分
- 未探索 streaming/在线数据场景

## 对比

与 DIFFPUTER 相比：去除迭代 EM，warm-up 单遍 + 单次 diffusion 即可达到更好效果；Mamba 替代 TabDDPM 的 Transformer denoiser，参数量和计算量大幅减少。与 ReMasker 相比：ReMasker 用 masked autoencoding，RefiDiff 用扩散 + 渐进 refinement，OOS 泛化更好。

## 启发

- Mamba 在非序列数据（表格）上也能发挥长程依赖捕获优势，值得在其他非图像非文本任务中探索
- "Warm-up + Diffusion + Polishing" 三阶段的渐进式设计思路可推广到其他生成式填补/修复任务
- 单遍 local prediction 作为 diffusion 的初始化，是一种实用的降低生成模型负担的策略

## 评分

⭐⭐⭐⭐ — 方法设计优雅，理论与实验完整，MNAR 场景贡献突出，但数据规模和特征类型覆盖仍有扩展空间
