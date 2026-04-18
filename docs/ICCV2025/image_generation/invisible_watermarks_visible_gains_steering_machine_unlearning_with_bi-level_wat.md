---
title: >-
  [论文解读] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design
description: >-
  [ICCV 2025][图像生成][机器遗忘] 提出 Water4MU，通过双层优化（BLO）框架将数字水印机制与机器遗忘（MU）相结合，在上层优化水印网络使其有利于遗忘，在下层执行遗忘优化，从而在不显著损害模型效用的前提下显著提升遗忘效果。
tags:
  - ICCV 2025
  - 图像生成
  - 机器遗忘
  - 数字水印
  - 双层优化
  - 隐式梯度
  - 扩散模型概念擦除
---

# Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design

**会议**: ICCV 2025  
**arXiv**: [2508.10065](https://arxiv.org/abs/2508.10065)  
**代码**: 无  
**领域**: 机器遗忘/数据水印  
**关键词**: 机器遗忘, 数字水印, 双层优化, 隐式梯度, 扩散模型概念擦除

## 一句话总结

提出 Water4MU，通过双层优化（BLO）框架将数字水印机制与机器遗忘（MU）相结合，在上层优化水印网络使其有利于遗忘，在下层执行遗忘优化，从而在不显著损害模型效用的前提下显著提升遗忘效果。

## 研究背景与动机

机器遗忘（Machine Unlearning, MU）旨在从已训练模型中移除特定数据的影响，以满足"被遗忘权"等隐私法规要求。目前主流 MU 方法集中在模型权重层面的调整（如梯度上升 GA、微调 FT 等），对数据层面的修改（如水印）如何影响遗忘过程几乎没有探索。

本文的核心动机源自两个关键观察：

**水印与遗忘的正交性**：在精确遗忘方法 Retrain 上的实验表明，无论是在水印数据上训练并在原始数据上评估（$\mathcal{S}_2$），还是在原始数据上训练并在水印数据上评估（$\mathcal{S}_1$），遗忘效果与基线场景（$\mathcal{S}_0$）相比差异极小。这说明水印操作不会破坏遗忘效果。

**水印可以被主动设计以促进遗忘**：既然水印不会干扰遗忘，那么是否可以反过来利用水印机制，通过优化水印网络参数和水印消息，使得遗忘过程更加高效？这就是 Water4MU 的出发点。

此外，在"challenging forgets"（难以遗忘的数据子集）场景下，现有 MU 方法表现不佳，而 Water4MU 在这些难场景中展现出更大优势。

## 方法详解

### 整体框架

Water4MU 采用双层优化（Bi-Level Optimization, BLO）框架。核心思想是将水印过程视为"领导者"（上层），遗忘过程视为"追随者"（下层），通过优化水印网络参数 $\boldsymbol{\psi}$ 来使下层遗忘更加高效。

整体流程：
1. 上层：优化水印编码器参数 $\boldsymbol{\psi}$ 和解码器参数 $\boldsymbol{\phi}$，使得遗忘后的模型在原始（非水印）数据上表现良好，同时维持水印网络自身功能
2. 下层：在水印化的忘记集 $\hat{\mathcal{D}_f}$ 和保留集 $\hat{\mathcal{D}_r}$ 上执行标准遗忘优化

### 关键设计

1. **下层遗忘优化**：给定水印编码器 $f_{\boldsymbol{\psi}}$ 产生的水印数据集，下层目标函数采用 GradDiff 方法（梯度差分），即：

$$\boldsymbol{\theta}_u(\boldsymbol{\psi}) = \arg\min_{\boldsymbol{\theta}} -\ell_{CE}(\boldsymbol{\theta}; \hat{\mathcal{D}_f}) + \ell_{CE}(\boldsymbol{\theta}; \hat{\mathcal{D}_r})$$

其中 $\hat{\mathcal{D}_f} = \{f_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{m})\}_{\mathbf{x} \in \mathcal{D}_f}$。通过在忘记集上做梯度上升（负交叉熵）、在保留集上做标准训练来实现遗忘。关键在于遗忘后模型 $\boldsymbol{\theta}_u$ 是水印参数 $\boldsymbol{\psi}$ 的函数。

2. **上层水印优化**：上层目标包含两部分——(a) 验证遗忘模型在原始无水印数据上的 MU 效果，(b) 保持水印网络的编解码能力：

$$\hat{\mathcal{L}}(\boldsymbol{\psi}, \boldsymbol{\phi}, \boldsymbol{\theta}_u(\boldsymbol{\psi})) = \mathcal{L}_{mu}(\boldsymbol{\theta}_u(\boldsymbol{\psi}); \mathcal{D}_f, \mathcal{D}_r) + \mathcal{L}_{wm}(\boldsymbol{\psi}, \boldsymbol{\phi}; \mathbf{m}, \mathcal{D}_f \cup \mathcal{D}_r)$$

这确保水印不仅服务于遗忘，还保留其数据溯源功能。

3. **隐式梯度求解**：BLO 的关键难点在于上层梯度依赖下层最优解。本文采用隐式函数定理，将 Hessian 矩阵近似为对角阵 $\nabla^2_{\boldsymbol{\theta}\boldsymbol{\theta}} \ell_{mu} \approx \lambda \mathbf{I}$，从而将上层梯度简化为：

$$\frac{d\hat{\mathcal{L}}}{d\boldsymbol{\psi}} = \nabla_{\boldsymbol{\psi}} \hat{\mathcal{L}} - \frac{\partial}{\partial \boldsymbol{\psi}}[\nabla_{\boldsymbol{\theta}} \ell_{mu}^\top \nabla_{\boldsymbol{\theta}} \hat{\mathcal{L}}]$$

如此只需要一阶导数即可实现整个 BLO 的优化，避免了高阶导数的计算开销。

4. **水印消息选择**：在固定水印网络的情况下，还可以通过优化水印消息 $\mathbf{m} \in \{0,1\}^L$ 来进一步提升遗忘效果，思路类似但将上层变量替换为 $\mathbf{m}$。

5. **扩展到图像生成**：Water4MU 可扩展到扩散模型的 prompt 级概念擦除场景，在忘记特定概念的 prompt 时对相应水印化数据进行遗忘优化。

### 损失函数 / 训练策略

- **下层**：GradDiff 目标，$\lambda_f = \lambda_r = 1$，忘记集用负交叉熵，保留集用正交叉熵
- **上层**：遗忘验证损失 + 水印网络训练损失
- **水印基础**：采用 HiDDeN 框架，包含图像重建损失 $\ell_{rec}$ 和消息解码损失 $\ell_{dec}$
- 上层学习率 $10^{-4}$（10 epochs），下层学习率 $10^{-2}$（3 epochs），Hessian 参数 $\lambda = 10^{-2}$

## 实验关键数据

### 主实验

在 CIFAR-10/ResNet-18 上对比多种 MU 方法在有无 Water4MU 时的表现（10% 随机数据遗忘）：

| 方法 | UA↑ (Original→Water4MU) | MIA↑ (Original→Water4MU) | RA↑ (Original→Water4MU) |
|------|------------------------|--------------------------|------------------------|
| Retrain | 6.78→10.01 (+3.23) | 16.06→19.33 (+3.27) | 100.00→99.93 (-0.07) |
| GA | 0.80→1.92 (+1.12) | 1.89→5.67 (+3.78) | 99.42→99.18 (-0.24) |
| FT | 1.85→4.93 (+3.08) | 5.60→8.26 (+2.66) | 99.66→98.75 (-0.91) |
| Sparse | 6.11→7.50 (+1.39) | 13.08→14.70 (+1.62) | 97.76→97.22 (-0.54) |
| IU | 0.64→2.62 (+1.98) | 1.53→3.67 (+2.14) | 99.43→98.98 (-0.45) |

类别遗忘场景提升更显著：FT 的 UA 从 37.29 提升至 53.25（+15.96），MIA 从 55.96 提升至 69.24（+13.28）。

### 消融实验

| 配置 | 说明 |
|------|------|
| CIFAR-100 随机遗忘 | UA 提升 2.13~4.19，MIA 提升 0.13~7.08，RA 下降 ≤0.75 |
| SVHN 随机遗忘 | UA 提升 1.30~2.65，MIA 提升 1.73~6.34 |
| ImageNet 类别遗忘 | UA 提升 2.12~8.19，MIA 提升 0.10~3.21 |
| 水印消息选择 | 在固定水印网络下进一步提升 MU 效果 |
| Challenging forgets | Water4MU 在难遗忘样本上优势尤为突出 |

### 关键发现

- Water4MU 能一致性地提升所有 MU 方法的遗忘效果（UA 和 MIA 指标），同时模型效用（RA/TA）仅有轻微下降
- 在类别遗忘场景下提升幅度更大，尤其是 FT 方法受益最多
- 在"challenging forgets"等难遗忘场景中优势最为明显
- 扩展至扩散模型的 prompt 遗忘同样有效

## 亮点与洞察

- **跨领域创新**：首次将数字水印与机器遗忘结合，建立了两者之间的正式联系
- **即插即用**：Water4MU 可与任意现有 MU 方法结合使用，不改变遗忘算法本身
- **理论优雅**：BLO + 隐式梯度的方法论有坚实的优化理论基础，同时实现简洁（只需一阶导数）
- **实际意义**：水印同时服务于数据溯源和遗忘增强，一举两得

## 局限与展望

- 需要预训练水印网络（HiDDeN），引入额外计算开销
- Hessian 对角近似可能在某些情况下不够精确
- 在大规模模型（如 LLM）上的扩展性有待验证
- RA/TA 的轻微下降在安全关键场景中可能需要权衡
- 目前仅限于图像领域，文本/音频等模态的扩展尚未探索

## 相关工作与启发

- **HiDDeN** [Zhu et al., 2018]：本文水印基础框架
- **GradDiff** [Liu et al., 2022]：梯度差分遗忘方法
- **Visual Prompting** [Bahng et al., 2022]：数据层面的修改影响模型行为
- 启发：数据预处理（水印/扰动/augmentation）可以被主动设计以服务于下游任务目标

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次探索水印与MU的交互，BLO框架新颖
- **实验充分度**: ⭐⭐⭐⭐ 多数据集、多方法、多场景（分类+生成）
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，从观察到方法的推导自然流畅
- **价值**: ⭐⭐⭐⭐ 开辟了水印辅助遗忘的新方向，有实际应用潜力
