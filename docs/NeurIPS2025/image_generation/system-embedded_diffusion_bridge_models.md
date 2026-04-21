---
title: >-
  [论文解读] System-Embedded Diffusion Bridge Models
description: >-
  [NeurIPS 2025][图像生成][扩散桥模型] 提出System-embedded Diffusion Bridge Models（SDB），将已知的线性测量系统直接嵌入矩阵值SDE的系数中，实现了对值域空间去噪和零空间信息合成的分离控制，在多种逆问题上取得一致性提升并展现出强大的系统失配鲁棒性。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散桥模型
  - 逆问题
  - 矩阵值SDE
  - 测量系统嵌入
  - 伪逆重建
---

# System-Embedded Diffusion Bridge Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.23726](https://arxiv.org/abs/2506.23726)  
**代码**: https://github.com/sobieskibj/sdb (有)  
**领域**: 图像生成 / 逆问题求解  
**关键词**: 扩散桥模型, 逆问题, 矩阵值SDE, 测量系统嵌入, 伪逆重建

## 一句话总结

提出System-embedded Diffusion Bridge Models（SDB），将已知的线性测量系统直接嵌入矩阵值SDE的系数中，实现了对值域空间去噪和零空间信息合成的分离控制，在多种逆问题上取得一致性提升并展现出强大的系统失配鲁棒性。

## 研究背景与动机

逆问题（从不完整或有噪声的测量中恢复信号）是科学和工程中的基本任务。基于扩散模型的求解方法形成了两大范式：**无监督方法**利用预训练的生成模型，通过测量系统引导条件生成；**有监督桥方法**在配对数据上训练随机过程，建立从退化数据到干净数据的映射。

核心矛盾在于：无监督方法通常假设已知测量系统并利用其结构信息，但桥方法却忽视了这一先验知识，仅关注任意分布之间的通用映射。然而在CT、MRI等实际应用中，线性测量过程是已知的，且数据集包含配对样本。例如，在图像修复中，现有桥方法无法区分已知的值域部分（未遮挡区域）和缺失的零空间部分（遮挡区域），导致在值域空间引入不必要的噪声。

SDB的核心idea是：**将测量系统的响应矩阵和噪声协方差直接嵌入扩散过程的矩阵值SDE系数中**，使生成过程能够区分值域空间和零空间，分别进行去噪和信息合成。

## 方法详解

### 整体框架

SDB构建了一个从伪逆重建（pseudoinverse reconstruction, PR）到干净信号的扩散桥。给定线性测量系统 $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\Sigma}^{1/2}\boldsymbol{\epsilon}$，SDB利用伪逆 $\mathbf{A}^+$ 将测量 $\mathbf{y}$ 映射回信号空间 $\hat{\mathbf{x}} = \mathbf{A}^+\mathbf{y}$，然后学习从PR到干净信号的扩散桥。

### 关键设计

1. **测量系统嵌入**: SDB的核心贡献是设计了特定的均值矩阵 $\mathbf{H}_t$ 和协方差矩阵 $\boldsymbol{\Sigma}_t$：

    $\mathbf{H}_t = \mathbf{A}^+\mathbf{A} + \alpha_t(\mathbf{I} - \mathbf{A}^+\mathbf{A})$
    $\boldsymbol{\Sigma}_t = \gamma_t\mathbf{A}^+\boldsymbol{\Sigma}\mathbf{A}^{+\top} + \beta_t(\mathbf{I} - \mathbf{A}^+\mathbf{A})$

   其中 $\alpha_t, \beta_t, \gamma_t$ 分别控制零空间漂移、零空间扩散和值域空间扩散。这一设计使得中间状态 $\mathbf{x}_t$ 的值域和零空间分量独立演化：值域部分直接建模测量噪声（无噪声时完美保持），零空间部分进行信息合成。

2. **SDE视角与理论保证**: 利用Theorem 1（Tivnan et al., 2025）的映射关系，从 $\mathbf{H}_t, \boldsymbol{\Sigma}_t$ 推导出对应的漂移和扩散系数 $\mathbf{F}_t, \mathbf{G}_t$。进一步证明（Theorem 2）当 $\alpha_t, \beta_t$ 取特定形式时，零空间部分退化为最优传输ODE，建立了与Schrödinger Bridge的联系。Theorem 3证明在满足 $\lim_{t\to 1}\gamma_t=1, \lim_{t\to 1}\alpha_t^2/\beta_t=0$ 的条件下，SDB能生成渐近精确的贝叶斯后验样本。

3. **三种变体**: 

    - **SDB (SB)**: 基于Schrödinger Bridge，$\alpha_t = \bar{\sigma}_t^2 / (\bar{\sigma}_t^2 + \sigma_t^2)$，具有最优传输性质
    - **SDB (VP)**: 重新诠释VP扩散，$\alpha_t = 1-t$，零空间中进行凸插值
    - **SDB (VE)**: 重新诠释VE扩散，$\alpha_t = 1$，零空间收敛到各向同性高斯

### 损失函数 / 训练策略

训练采用去噪目标，使用L1重建损失：$L_\theta = \|\mathcal{D}_\theta(\mathbf{x}_t, t) - \mathbf{x}\|_1$。网络 $\mathcal{D}_\theta$ 直接预测干净信号。训练时同时采样值域空间噪声 $\boldsymbol{\epsilon} \in \mathbb{R}^m$ 和零空间噪声 $\boldsymbol{\epsilon}' \in \mathbb{R}^d$，体现了两个空间的独立建模。采样使用Euler-Maruyama求解器，100步离散化。

## 实验关键数据

### 主实验

| 任务-数据集 | 指标 | SDB (SB) | 之前SOTA桥方法 | 提升 |
|------------|------|----------|--------------|------|
| Inpainting-CelebA | FID↓ | **4.63** | 4.68 (IR-SDE) | -0.05 |
| Inpainting-CelebA | PSNR↑ | **30.40** | 29.92 (IR-SDE) | +0.48 |
| SuperRes-DIV2K | FID↓ | **81.56** | 83.73 (I2SB) | -2.17 |
| SuperRes-DIV2K | PSNR↑ | **26.10** | 25.79 (DDBM) | +0.31 |
| CT重建-RSNA | FID↓ | **15.02** | 18.88 (IR-SDE) | -3.86 |
| CT重建-RSNA | PSNR↑ | **46.672** | 44.415 (DDBM) | +2.26 |
| MRI重建-Br35H | FID↓ | **29.85** | 30.14 (IR-SDE) | -0.29 |
| MRI重建-Br35H | PSNR↑ | **29.812** | 28.971 (DDBM) | +0.84 |

### 消融实验（系统失配鲁棒性）

| MRI模型失配设置 | SDB (SB) PSNR | DDBM PSNR | 差距 |
|---------------|-------------|-----------|------|
| $\lambda_1=16$（训练值） | 29.81 | 28.97 | +0.84 |
| $\lambda_1=14$ | 稳定保持优势 | 显著下降 | 差距扩大 |
| $\lambda_1=12$ | 仍保持领先 | 进一步恶化 | 差距更大 |
| $\sigma^2$增大至2倍 | 鲁棒 | 下降明显 | SDB优势显著 |

### 关键发现

- SDB (SB) 在所有四个逆问题上一致超越所有基线桥方法，且性能排名最为稳定
- 在医学成像任务（CT/MRI）上，SDB相比基线桥方法的PSNR提升超过2dB，差异显著
- 系统失配实验中，SDB的性能优势随失配程度增大而**扩大**，展现出优越的泛化能力
- 无监督方法在统一设置下性能明显低于桥方法

## 亮点与洞察

- 优雅地将测量系统的数学结构嵌入扩散过程，将"领域知识注入生成模型"的理念做到了极致
- 通过值域-零空间分解，实现了对两个空间的独立建模和独立噪声控制
- 提供了三个定理的理论支撑，包括OT联系和后验采样的渐近精确性
- 系统失配鲁棒性实验对实际部署具有重要参考价值

## 局限与展望

- 仅适用于线性测量系统，非线性扩展仅有初步概念验证
- 方差调度采用简单的线性设计，值域和零空间的调度交互关系有待深入研究
- CT/MRI实验在2D切片上进行，实际临床应用需要3D重建
- 需要已知或可计算测量系统的伪逆 $\mathbf{A}^+$

## 相关工作与启发

- **vs I2SB**: 共享最相似的随机过程结构，但SDB通过嵌入测量系统信息获得了显著的性能提升
- **vs IR-SDE/GOUB**: 这些方法使用标量SDE系数，SDB推广到矩阵值系数实现了更精细的控制
- **vs DPS/ΠGD等无监督方法**: 无监督方法利用已知系统但不用配对数据；SDB同时利用两者
- **vs DDBM**: DDBM对称化方差调度，SDB通过系统嵌入实现了更有原则的设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将测量系统嵌入SDE系数的思路非常原创且数学上优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 四个逆问题、三个变体、系统失配分析，统一框架下公平比较
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，动机明确，理论与实验相互印证
- 价值: ⭐⭐⭐⭐ 对逆问题领域贡献突出，但应用范围受限于线性系统

<!-- RELATED:START -->

## 相关论文

- [Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [Aligning Compound AI Systems via System-level DPO](aligning_compound_ai_systems_via_system-level_dpo.md)
- [Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)
- [Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [MagicQuill: An Intelligent Interactive Image Editing System](../../CVPR2025/image_generation/magicquill_an_intelligent_interactive_image_editing_system.md)

<!-- RELATED:END -->
