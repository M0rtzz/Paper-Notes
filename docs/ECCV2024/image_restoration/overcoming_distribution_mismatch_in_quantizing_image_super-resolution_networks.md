---
title: >-
  [论文解读] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks
description: >-
  [ECCV 2024][图像恢复][超分辨率] 本文提出 ODM 框架，通过协同失配正则化（cooperative mismatch regularization）和逐层权重裁剪校正（weight clipping correction）两个简单策略，在不引入推理时动态模块的前提下解决 SR 网络量化中的分布失配问题，以极小的额外开销达到 SOTA。
tags:
  - ECCV 2024
  - 图像恢复
  - 超分辨率
  - 量化
  - distribution mismatch
  - gradient conflict
---

# Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks

**会议**: ECCV 2024  
**arXiv**: [2307.13337](https://arxiv.org/abs/2307.13337)  
**代码**: [https://github.com/Cheeun/ODM](https://github.com/Cheeun/ODM)  
**领域**: 图像修复  
**关键词**: image super-resolution, network quantization, distribution mismatch, quantization-aware training, gradient conflict

## 一句话总结

本文提出 ODM 框架，通过协同失配正则化（cooperative mismatch regularization）和逐层权重裁剪校正（weight clipping correction）两个简单策略，在不引入推理时动态模块的前提下解决 SR 网络量化中的分布失配问题，以极小的额外开销达到 SOTA。

## 研究背景与动机

**领域现状**：图像超分辨率（SR）网络（EDSR, RDN, SwinIR 等）虽然性能卓越，但计算量巨大，限制了在移动端等资源受限场景的部署。网络量化通过将 32-bit 浮点运算映射到低比特整数运算来大幅降低计算成本，在高级视觉任务（分类等）中已被证明有效。

**现有痛点**：SR 网络量化到低比特时性能急剧下降。根本原因是 SR 网络通常不使用 BatchNorm，导致不同通道和不同输入图像之间的特征（激活）分布差异极大（distribution mismatch），使得单一的量化范围 $[l, u]$ 无法同时适应所有分布。

**核心矛盾**：现有方法（DAQ, DDTB）通过动态模块在推理时自适应调整量化范围来适配不同的分布，但动态模块本身引入了额外的计算开销（额外的 bitOPs 和存储），削弱了量化带来的计算效率优势。本质上是「量化精度 vs 推理效率」的矛盾。

**本文目标**：在不使用任何推理时动态模块的前提下，通过训练阶段的正则化策略预先调整特征分布使之对量化友好，同时确定更精确的逐层权重量化范围。

**切入角度**：从多任务学习中的梯度冲突视角出发——直接添加失配正则化会与重建损失产生梯度冲突，导致约一半的参数被推往相反方向。因此引入基于梯度余弦相似度的协同策略，仅在两个损失方向一致时才施加正则化。

**核心 idea**：用训练时的协同正则化预先让 SR 特征分布变得对量化友好，而非推理时用动态模块去适应不友好的分布。

## 方法详解

### 整体框架

ODM 是一个量化感知训练（QAT）框架，适用于任意 SR 网络（EDSR、RDN、SwinIR 等）：
1. 从预训练的 32-bit SR 网络出发
2. 每个卷积/线性层的激活和权重按均匀量化函数 $q(\mathbf{X}_i; l, u) = \text{Int}\left(\frac{\text{clip}(\mathbf{X}_i, l, u) - l}{s}\right) \cdot s + l$ 量化，其中 $s = \frac{u - l}{2^b - 1}$
3. 训练过程中同时优化重建损失 $\mathcal{L}_R$ 和失配正则化损失 $\mathcal{L}_M$，通过协同策略避免梯度冲突
4. 权重量化范围通过逐层校正参数 $\gamma_w$ 动态调整
5. 推理时无任何额外模块，量化范围和校正参数均可预计算

### 关键设计

1. **Cooperative Mismatch Regularization（协同失配正则化）**：

    - **功能**：在训练时正则化每层特征与其量化网格之间的距离（mismatch），使特征分布预先对量化友好
    - **核心思路**：定义失配为 $M(\mathbf{X}_i) = \|\mathbf{X}_i - q(\mathbf{X}_i; l_a, u_a)\|_2$，总失配损失 $\mathcal{L}_M = \sum_i^{\#\text{layers}} M(\mathbf{X}_i)$。但直接联合优化 $\mathcal{L}_R + \mathcal{L}_M$ 会产生严重梯度冲突（约 50% 参数方向相反）。因此采用协同策略：
    $\theta^{t+1} = \theta^t - \beta^t \cdot \left(\lambda_R \nabla_\theta \mathcal{L}_R + \lambda_M \cdot \text{sim}(\nabla_\theta \mathcal{L}_R, \nabla_\theta \mathcal{L}_M) \cdot \nabla_\theta \mathcal{L}_M\right)$
   其中 $\text{sim}(\mathbf{v}_a, \mathbf{v}_b) = \frac{\cos(\mathbf{v}_a, \mathbf{v}_b) + 1}{2} \in [0, 1]$。当两个梯度方向一致时 sim 接近 1（充分利用正则化），方向相反时 sim 接近 0（忽略正则化，跟随重建损失）
    - **设计动机**：实验观察到（Fig.2）简单联合优化时梯度余弦相似度频繁为负，约 50% 参数发生梯度冲突。直接添加 $\mathcal{L}_M$ 虽使分布对量化友好但偏离 32-bit 原始分布的高密度区域（near 0），导致重建精度下降（消融实验降 0.13dB）。协同策略可兼顾量化友好性和重建精度

2. **Weight Clipping Correction（权重裁剪校正）**：

    - **功能**：为每层权重确定更精确的对称量化范围 $[-u_w, u_w]$，替代全局统一策略
    - **核心思路**：现有方法用固定全局策略 $u_w^t = f(\mathbf{W}^t)$（如 max）确定所有层的量化范围。本文引入逐层可学习校正参数 $\gamma_w$：
    $u_w^{t+1} = f(\mathbf{W}^{t+1}) \cdot \left(\gamma_w^t - \beta^t \cdot \nabla_{\gamma_w} \mathcal{L}_R(\gamma_w^t)\right)$
   $\gamma_w$ 初始化为 1，$f(\cdot)$ 用 99th 百分位函数代替 max 来避免离群值主导范围
    - **设计动机**：(a) 各层权重分布差异大，统一的 max 策略使某些层的量化网格被离群值拉到低密度区（Fig.3），低比特时问题更严重；(b) 纯可学习 $u_w$（Eq.8）的问题是基于上一步权重 $\mathbf{W}^{t-1}$ 优化但要量化当前权重 $\mathbf{W}^t$，存在滞后。校正因子 $\gamma_w$ 分离了「基于当前权重的全局策略」和「逐层调整」两个关注点。额外开销仅每层 1 个标量参数（EDSR 0.06K，RDN 0.15K），推理时可预计算到量化范围中，零额外 bitOPs

3. **量化范围初始化与更新**：

    - **功能**：激活量化范围 $[l_a, u_a]$ 的合理初始化和可学习更新
    - **核心思路**：用训练数据的 99th 和 1st 百分位均值初始化 $l_a, u_a$（避免离群值），然后通过 STE（直通估计器）端到端学习。裁剪参数用 $10 \times \beta^0$ 的较大学习率更新
    - **设计动机**：百分位初始化比 min/max 更鲁棒，较大学习率让裁剪参数快速适应分布变化

### 损失函数 / 训练策略

- **损失函数**：重建损失 $\mathcal{L}_R = \mathcal{L}_1(\mathcal{Q}(\mathbf{I}_{LR}), \mathbf{I}_{HR})$ + 协同失配正则化 $\mathcal{L}_M$
- **超参数**：$\lambda_R = 1$，$\lambda_M = 10^{-5}$（RDN 用 $10^{-6}$ 因其整体失配较大），百分位 $j = 99$
- **训练配置**：DIV2K 数据集，60K iterations，batch size 8，初始学习率 $10^{-4}$，每 15K iterations 减半，RTX 2080Ti GPU

## 实验关键数据

### 主实验（EDSR ×4 量化）

| 方法 | Bit | Set5 PSNR↑ | Set5 SSIM↑ | Set14 PSNR↑ | Urban100 PSNR↑ | Urban100 SSIM↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| EDSR（FP32） | 32 | 32.10 | 0.894 | 28.58 | 26.04 | 0.785 |
| EDSR-PAMS | 4 | 31.59 | 0.885 | 28.20 | 25.32 | 0.762 |
| EDSR-DAQ | 4 | 31.85 | 0.887 | 28.38 | 25.73 | 0.772 |
| EDSR-DDTB | 4 | 31.85 | 0.889 | 28.39 | 25.69 | 0.774 |
| **EDSR-ODM** | **4** | **32.00** | **0.891** | **28.47** | **25.80** | **0.778** |
| EDSR-PAMS | 2 | 29.51 | 0.835 | 26.79 | 23.72 | 0.688 |
| EDSR-DAQ | 2 | 31.01 | 0.871 | 27.89 | 24.88 | 0.740 |
| EDSR-DDTB | 2 | 30.97 | 0.876 | 27.87 | 24.82 | 0.742 |
| **EDSR-ODM** | **2** | **31.50** | **0.882** | **28.14** | **25.17** | **0.755** |

### 消融实验（EDSR 2-bit ×4）

| 模型 | WCC | Cooperative | MR | Set5 PSNR / SSIM | Urban100 PSNR / SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|
| (a) Baseline | - | - | - | 29.94 / 0.848 | 23.99 / 0.703 |
| (b) +协同MR | - | ✓ | ✓ | 30.34 / 0.859 | 24.27 / 0.715 |
| (c) +WCC | ✓ | - | - | 31.12 / 0.876 | 24.91 / 0.746 |
| (d) +WCC+朴素MR | ✓ | - | ✓ | 30.99 / 0.871 | 24.79 / 0.735 |
| **(e) ODM** | **✓** | **✓** | **✓** | **31.50 / 0.882** | **25.17 / 0.755** |

### 关键发现

- **2-bit 场景增益最显著**：EDSR 2-bit 上比次优方法（DAQ）高 0.49dB（Set5），RDN 2-bit 上高 0.66dB，SwinIR 2-bit 上高 0.43dB，低比特下分布失配问题更严重，ODM 的收益也越大
- **4-bit 接近 FP32**：EDSR-ODM 4-bit 在 Set5 上与 FP32 仅差 0.1dB，说明 ODM 有效弥合了量化网络和全精度网络的差距
- **朴素 MR 反而有害**：直接联合 $\mathcal{L}_R + \mathcal{L}_M$（Model d）比仅 WCC（Model c）低 0.13dB，验证了梯度冲突确实阻碍重建
- **协同策略扭转局势**：Model (e) 比 Model (d) 高 0.51dB，证明梯度对齐筛选的有效性
- **WCC 贡献最大**：Model (c) vs (a) 提升 1.18dB，说明逐层权重量化范围的精确选择至关重要
- **计算开销极小**：EDSR 上额外存储仅 0.06K（vs DDTB 1.9K），零额外 bitOPs（vs DAQ 0.5T），是唯一不需要通道级量化或动态模块的方法
- **跨架构通用**：CNN（EDSR, RDN）和 Transformer（SwinIR）均有效，×2 和 ×4 均有效

## 亮点与洞察

1. **训练时解决 vs 推理时补救**：范式转换——不是在推理时用更复杂的量化函数去适配不友好的分布，而是在训练阶段预先把分布调整到对量化友好。这保证了推理效率不受损
2. **梯度冲突的发现和解决**：借鉴多任务学习中的梯度冲突理论，用余弦相似度加权实现两个目标的协同优化。这个思路可推广到其他需要正则化的场景
3. **分布可视化直观说明问题**：Fig.5 清楚展示了三种训练策略下的激活分布差异——朴素 MR 导致多模态分布偏离原始高密度区，协同 MR 在保持原始分布结构的同时减小离群值
4. **解决方案的极简性**：整个框架无需修改网络架构，只需在训练时加一个损失项和逐层一个标量参数，易于集成到现有 QAT 流程

## 局限与展望

1. **仅验证了部分 SR 网络**：EDSR/RDN/SwinIR 均为经典网络，对更新的大型 SR 模型（如 HAT、SRFormer）的效果未知
2. **梯度相似度计算有额外训练开销**：需要分别计算两个损失的梯度再算余弦相似度，训练时间增加（论文未报告训练时间对比）
3. **超参数 $\lambda_M$ 需要针对不同网络调整**：RDN 和 EDSR 用不同的 $\lambda_M$，自动化调参策略可进一步探索
4. **逐层校正参数的初始化**：$\gamma_w$ 固定初始化为 1 可能不是最优，可考虑基于每层分布特征的自适应初始化
5. **未探索混合精度**：所有层使用相同比特宽度，结合 ODM 和混合精度分配策略可能进一步提升效率-精度权衡
6. **仅使用 L1 重建损失**：未探索与感知损失或 GAN 损失结合时的效果

## 相关工作与启发

- **PAMS [Li et al., ECCV 2020]**：首个为 SR 量化学习裁剪参数的工作，是本文的 baseline 之一，但不能处理通道/图像维度的分布差异
- **DAQ [Hong et al., WACV 2022]**：引入通道级动态量化函数，效果好但增加 bitOPs。本文证明无需通道级量化也能超越 DAQ
- **DDTB [Zhong et al., ECCV 2022]**：用输入自适应动态模块，存储开销大（比 ODM 大 31.7×-304.7×）
- **梯度冲突 [Du et al., 2018]**：多任务学习中辅助损失的梯度相似度加权策略，本文将此思路创造性地应用于量化训练
- **QuantSR [Qin et al., NeurIPS 2024]**：并行工作，在前向和反向过程中引入额外变换函数，ODM 在 2-bit 上优势明显（+0.51dB on SRResNet）
- **启发**：「分布友好性」的思路可推广到其他低级视觉任务的量化（去噪、去模糊等），梯度冲突感知的正则化策略有通用价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从梯度冲突角度审视量化正则化是新颖的视角，训练时解决推理时问题的范式转换有价值，但核心技术（梯度余弦加权）借鉴自多任务学习
- **实验充分度**: ⭐⭐⭐⭐⭐ 三种架构（CNN×2+Transformer）、三个比特（4/3/2）、四个数据集、完整消融、复杂度分析、分布可视化，非常全面
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，观察→动机→方案的逻辑链条紧密，分布可视化和梯度冲突图很好地支撑了叙事
- **价值**: ⭐⭐⭐⭐ 实用价值高——无推理开销的 SR 量化方案对部署有直接意义，2-bit 场景的大幅提升对边缘设备部署重要

<!-- RELATED:START -->

## 相关论文

- [Accelerating Image Super-Resolution Networks with Pixel-Level Classification](accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)
- [Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)

<!-- RELATED:END -->
