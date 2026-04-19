---
title: >-
  [论文解读] ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion
description: >-
  [ECCV 2024][3D视觉][自监督深度估计] 提出一种概率融合框架 ProDepth，通过辅助解码器推断动态区域不确定性，以加权几何均值自适应融合单帧和多帧深度概率分布来修正代价体中的错误匹配代价，并配合不确定性感知的损失重加权策略，在自监督多帧单目深度估计中取得 SOTA。
tags:
  - ECCV 2024
  - 3D视觉
  - 自监督深度估计
  - 多帧单目深度
  - 概率融合
  - 代价体调制
  - 动态物体处理
---

# ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion

**会议**: ECCV 2024  
**arXiv**: [2407.09303](https://arxiv.org/abs/2407.09303)  
**代码**: [sungmin-woo.github.io/prodepth](https://sungmin-woo.github.io/prodepth/) (有)  
**领域**: 3D视觉  
**关键词**: 自监督深度估计, 多帧单目深度, 概率融合, 代价体调制, 动态物体处理  

## 一句话总结

提出一种概率融合框架 ProDepth，通过辅助解码器推断动态区域不确定性，以加权几何均值自适应融合单帧和多帧深度概率分布来修正代价体中的错误匹配代价，并配合不确定性感知的损失重加权策略，在自监督多帧单目深度估计中取得 SOTA。

## 研究背景与动机

- **领域现状**: 自监督多帧单目深度估计依赖连续帧之间在静态场景假设下的几何一致性，利用代价体（cost volume）评估各深度候选的概率，在整体性能上优于单帧方法
- **现有痛点**: 动态场景中运动物体破坏了静态场景假设，导致代价体中特征匹配错位（misaligned feature matching），产生错误的深度概率分布；同时基于重投影的光度损失在动态区域提供误导性的训练监督
- **核心矛盾**: 多帧方法在静态区域预测更准确，而单帧方法因不依赖代价体可更好地处理运动物体——两者优势互补，但如何在像素级别自适应地选择信任哪个线索仍是难题
- **本文解决什么**: (1) 如何准确识别动态物体（不依赖额外的语义分割网络）；(2) 如何直接修正代价体中被动态物体污染的匹配代价分布；(3) 如何减轻动态区域错误监督对训练的影响
- **切入角度**: 现有方法要么在损失层面用单帧深度监督多帧深度（ManyDepth），要么在输入层面用单帧深度调整动态物体位置（DynamicDepth），均未直接修复代价体本身的错误
- **核心idea**: 将单帧深度和多帧代价体均表示为深度候选的概率分布，基于推断的不确定性用加权几何均值自适应融合两个分布，直接调制代价体

## 方法详解

### 整体框架

ProDepth 包含三大组件，如图 2 所示：
1. **辅助深度估计与不确定性推理**: 单帧解码器估计深度的高斯分布；辅助代价体解码器估计受污染的深度，与单帧深度对比推断不确定性
2. **概率代价体调制 (PCVM)**: 基于不确定性自适应融合单帧和多帧的深度概率分布
3. **不确定性感知损失重加权**: 在训练时根据不确定性减少动态区域的错误监督

### 关键设计

**模块一：辅助深度估计与不确定性推理**

单帧深度估计网络 $\theta_{\text{single}}$ 输出高斯分布的均值 $D_{\text{single}}$ 和方差 $\sigma_p^2$，通过最大化对数似然训练：

$$\mathcal{L}_p^{\log}(D_{\text{single}}) = \frac{(\mathcal{L}_p(D_{\text{single}}))^2}{\sigma_p^2} + \log \sigma_p^2$$

辅助解码器 $\psi_{\text{dec}}$ 从被污染的代价体特征估计深度 $D_{\text{cv}}$，利用深度估计的结构感知性（同一物体内像素深度一致），将像素级不一致放大为物体级的清晰错误。不确定性通过两个深度的差异计算：

$$U = 1 - e^{-\beta |D_{\text{single}} - D_{\text{cv}}|}, \quad \beta = 0.6$$

$U \in [0,1]$ 表示每个像素为动态区域的概率。

**模块二：概率代价体调制 (PCVM)**

将单帧深度转换为深度候选的概率分布（高斯 PDF）：

$$p_{\text{single}}(d_i|x) = \frac{1}{\sqrt{2\pi\sigma_p^2(x)}} \exp\left(-\frac{(d_i - D_{\text{single}}(x))^2}{2\sigma_p^2(x)}\right)$$

将多帧代价体匹配代价通过 softmax 转换为概率分布：

$$p_{\text{cv}}(d_i|x) = \frac{\exp(-\mathcal{C}(x,i))}{\sum_{j=1}^k \exp(-\mathcal{C}(x,j))}$$

用加权几何均值融合两个分布：

$$P(d|x) = p_{\text{single}}(d|x)^{U(x)} \cdot p_{\text{cv}}(d|x)^{1-U(x)}$$

高不确定性（动态像素）→ 融合结果偏向单帧分布；低不确定性（静态像素）→ 偏向多帧分布。融合后通过 min-max 归一化恢复到原始代价体尺度。

**模块三：不确定性感知损失重加权**

$$\mathcal{L}_{up} = M \odot (1 - U) \odot \mathcal{L}_p, \quad M = [U < \gamma]$$

结合二值掩码 $M$（排除高不确定性区域）和连续权重 $(1-U)$（根据概率降低可能动态区域的损失权重），比纯二值掩码更精细。

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L} = \sum_x \left[\mathcal{L}_{up,s}(D_{\text{multi}}) + \lambda_1 \mathcal{L}_{up,s}^{log}(D_{\text{single}}) + \lambda_2 \mathcal{L}_p(D_{\text{cv}}) + \lambda_3 \mathcal{L}_c\right]$$

- 多帧和单帧深度用不确定性感知损失 $\mathcal{L}_{up}$ 防止动态区域过拟合
- 代价体深度用标准 $\mathcal{L}_p$（**有意允许**动态区域的错误监督），使辅助解码器学会产生错误深度，从而增强不确定性推理
- $\mathcal{L}_p(D_{\text{cv}})$ 的梯度仅回传到代价体解码器参数，不影响代价体本身

## 实验关键数据

### 主实验

**Cityscapes 数据集**（动态物体较多）：

| 方法 | 额外语义 | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | $\delta < 1.25$ ↑ |
|------|---------|-----------|----------|--------|-------------------|
| ManyDepth | 无 | 0.114 | 1.193 | 6.223 | 0.875 |
| DynamicDepth | ✓ | 0.103 | 1.000 | 5.867 | 0.895 |
| **ProDepth** | **无** | **0.095** | **0.876** | **5.531** | **0.908** |

**KITTI 数据集**：

| 方法 | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | $\delta < 1.25$ ↑ |
|------|-----------|----------|--------|-------------------|
| DepthFormer | 0.090 | 0.661 | 4.149 | 0.905 |
| DualRefine | 0.090 | 0.658 | 4.237 | 0.912 |
| **ProDepth** | **0.086** | **0.629** | **4.139** | **0.918** |

### 消融实验

Cityscapes 上的组件消融（Abs Rel ↓）：

| # | 不确定性推理 | PCVM | 损失策略 | Abs Rel |
|---|-----------|------|---------|---------|
| 1 | consistency mask | 无 | masking | 0.107 |
| 3 | segmentation mask | 无 | masking | 0.100 |
| 7 | 加权不确定性 $U$ | 无 | masking | 0.100 |
| 8 | 加权不确定性 $U$ | ✓ | masking | 0.098 |
| 9 | 加权不确定性 $U$ | 无 | reweighting | 0.097 |
| **10** | **加权不确定性 $U$** | **✓** | **masking+reweighting** | **0.095** |

### 关键发现

1. PCVM 与分割掩码结合反而性能下降（行 #4），因为分割掩码包含静态物体，导致多帧线索被丢弃
2. 加权不确定性 + PCVM 效果最好，因为概率融合比基于二值准则选择更有效
3. 损失重加权比纯二值掩码更优，因为能处理不确定性模糊的边界区域
4. Waymo 泛化实验表明 ProDepth 跨数据集迁移能力优于需要语义分割的 DynamicDepth

## 亮点与洞察

- **加权几何均值融合**: 相比加权算术均值（加法），乘法性质能更好保留各分布的峰值位置
- **自举式不确定性推理**: 利用代价体自身的错误来反推动态区域，无需外部网络
- **概率连续性**: 不确定性是连续的 $[0,1]$，比二值掩码更精细
- **端到端统一**: 不确定性推理、代价体修正、损失调整三者紧密耦合

## 局限与展望

- 不确定性推理依赖单帧深度质量，若单帧深度本身不准则不确定性可能有偏
- PCVM 在深度候选离散化较稀疏时可能丢失精度
- 实验仅在自动驾驶场景（Cityscapes/KITTI/Waymo）验证，对室内场景适用性未知
- 可结合光流信息进一步提升动态物体识别精度

## 相关工作与启发

- **ManyDepth**: 首次引入自适应代价体解决尺度模糊，提出二值一致性掩码 → 本文指出其像素级独立、缺乏结构感知
- **DynamicDepth**: 用预训练分割网络识别可移动物体，调整输入图像 → 本文指出分割掩码包含不必要的静态物体
- **DepthFormer**: 用注意力机制替代传统相似度度量提升特征匹配 → 仍未解决代价体本身的错误

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 概率代价体调制是新颖的直接修正策略
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个数据集、详细消融、泛化实验完备
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，问题分析到位
- **实用价值**: ⭐⭐⭐⭐ — 不依赖额外语义网络，部署友好

<!-- RELATED:START -->

## 相关论文

- [High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion](pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)
- [DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)

<!-- RELATED:END -->
