---
title: >-
  [论文解读] Learning a Distance Measure from the Information-Estimation Geometry of Data
description: >-
  [ICLR 2026][图像生成][信息估计度量] 提出 Information-Estimation Metric (IEM)，一种由数据概率密度几何诱导的新型距离函数，通过比较不同噪声水平下的 score 向量场来度量信号间距离，无监督训练的 IEM 在预测人类感知判断上可媲美有监督方法。
tags:
  - ICLR 2026
  - 图像生成
  - 信息估计度量
  - 去噪误差
  - 概率密度几何
  - 感知距离
  - 扩散模型
---

# Learning a Distance Measure from the Information-Estimation Geometry of Data

**会议**: ICLR 2026  
**arXiv**: [2510.02514](https://arxiv.org/abs/2510.02514)  
**代码**: [GitHub](https://github.com/ohayonguy/information-estimation-metric)  
**领域**: 度量学习 / 感知质量评估  
**关键词**: 信息估计度量, 去噪误差, 概率密度几何, 感知距离, 扩散模型

## 一句话总结
提出 Information-Estimation Metric (IEM)，一种由数据概率密度几何诱导的新型距离函数，通过比较不同噪声水平下的 score 向量场来度量信号间距离，无监督训练的 IEM 在预测人类感知判断上可媲美有监督方法。

## 研究背景与动机
- 距离函数是科学和工程的核心工具，但自然信号（如图像）的感知距离缺乏精确数学定义
- 现有最佳感知度量（LPIPS、DISTS）依赖人类标注数据训练，成本高且可解释性差
- 信息论量（如互信息）对密度的全局几何不敏感，而估计论量（如去噪误差）直接与密度几何相关
- 去噪误差与 score 函数的关系（Tweedie-Miyasawa 公式）是扩散模型的基础——能否利用这一关系构建感知度量？

## 方法详解

### 整体框架
IEM 建立在 pointwise I-MMSE 公式之上：信号的对数概率可分解为最优去噪器在不同 SNR 水平上的去噪误差。通过比较两个信号周围的 score 向量场来定义距离。

### 关键设计

1. **IEM 定义**: 比较两点 $\boldsymbol{x}_1, \boldsymbol{x}_2$ 周围模糊密度的 score 向量场差异：
$$\text{IEM}(\boldsymbol{x}_1, \boldsymbol{x}_2, \Gamma) = \left(\int_0^\Gamma \mathbb{E}\left[\|\nabla \log p_{\mathbf{y}_\gamma}(\gamma \boldsymbol{x}_1 + \mathbf{w}_\gamma) - \nabla \log p_{\mathbf{y}_\gamma}(\gamma \boldsymbol{x}_2 + \mathbf{w}_\gamma)\|^2\right] d\gamma\right)^{1/2}$$
   其中 $\gamma$ 为信噪比，$\mathbf{w}_\gamma$ 为维纳过程噪声。IEM 可用训练好的去噪器（类似扩散模型）近似计算。

2. **度量性质**: 证明 IEM 是合法的距离度量（对称性、非负性、正定性、三角不等式）。对高斯分布，IEM 退化为 Mahalanobis 距离：$\text{IEM} = \sqrt{(\boldsymbol{x}_1 - \boldsymbol{x}_2)^\top \Sigma^{-1} (\boldsymbol{x}_1 - \boldsymbol{x}_2)}$

3. **局部黎曼度量**: 二阶展开给出黎曼度量 $\boldsymbol{G}(\boldsymbol{x}, \Gamma)$：
$$\boldsymbol{G}(\boldsymbol{x}, \Gamma) = \int_0^\Gamma \gamma^2 \mathbb{E}\left[(\nabla^2 \log p_{\mathbf{y}_\gamma}(\gamma \boldsymbol{x} + \mathbf{w}_\gamma))^2\right] d\gamma$$
   直觉：在 log 密度曲率高的区域和导致概率变化大的扰动方向上更敏感。

4. **广义 IEM**: 引入可学习函数 $f$ 调制 score 差异的权重，使 IEM 可适应不同感知任务（如纹理相似性 vs 失真评估）。

### 损失函数 / 训练策略
- 训练 Hourglass Diffusion Transformer (HDiT) 作为去噪器，在 ImageNet-1k 256×256 上训练
- 使用 MSE 损失和 log-uniform 噪声水平调度
- 计算 IEM 时，将去噪器插入定义式并数值求解一维积分

## 实验关键数据

### 主实验（SRCC 与人类 MOS 的相关性）
| 方法 | 是否监督 | TID2013 | LIVE | CSIQ | TQD(纹理) |
|------|---------|---------|------|------|-----------|
| PSNR | 否 | 0.69 | 0.87 | 0.81 | 0.34 |
| SSIM | 否 | 0.64 | 0.91 | 0.82 | 0.51 |
| LPIPS | 是 | 0.71 | 0.94 | 0.88 | 0.48 |
| DISTS | 是 | 0.83 | 0.95 | 0.93 | **0.83** |
| TOPIQ | 是 | **0.86** | **0.97** | **0.95** | 0.67 |
| **IEM (无监督)** | **否** | **0.83** | **0.96** | **0.94** | 0.51 |
| **IEM_sq (无监督)** | **否** | 0.66 | 0.82 | 0.79 | **0.79** |
| **IEM_fω (有监督f)** | **部分** | **0.84** | **0.96** | **0.94** | **0.77** |

### 消融实验（Max Differentiation Competition）
| 操作 | IEM 结果 | DISTS 结果 | 说明 |
|------|---------|-----------|------|
| 最小化度量(PSNR=10dB) | 无伪影高质量 | 明显伪影 | IEM作为优化目标更鲁棒 |
| 最大化度量(PSNR=10dB) | 非结构化噪声 | 模式化伪影 | IEM对偏离数据支撑的扰动最敏感 |

### 关键发现
- 无监督 IEM 在 TID2013/LIVE/CSIQ 上与最佳有监督方法竞争（SRCC 0.83-0.96）
- $\text{IEM}_{sq}$ 在纹理相似性（TQD）上表现优异，$f$ 的选择控制全局 vs 局部失真敏感度
- 学习 $f_\omega$ 后可同时在所有数据集上取得强结果
- IEM 最小化生成的图像无伪影，说明可作为独立优化目标

## 亮点与洞察
- 信息论与估计论的深刻联系为构建感知度量提供了原理性基础
- 高斯情况下退化为 Mahalanobis 距离提供了优美的理论锚点
- 等距线可以不连通（高斯混合情况），反映了度量对密度全局几何的适应性
- 为从无标注数据推导感知度量这一基本问题提供了突破性方案

## 局限与展望
- 计算成本高：需在多个 SNR 水平上运行去噪器并求积分，比 LPIPS 慢得多
- 超参 $\Gamma$ 的选择缺乏系统性原则
- 目前仅在 256×256 图像上验证
- 作为优化目标的应用（如图像恢复、压缩）有待探索

## 相关工作与启发
- 建立在 I-MMSE 公式（Guo et al. 2005）和 Tweedie-Miyasawa 公式之上
- 与扩散模型共享理论基础（score function = 去噪误差），但目的不同
- 为无监督特征学习和度量学习提供了新视角
- 可扩展到音频等其他连续信号域

## 技术细节补充
- 去噪器使用 Hourglass Diffusion Transformer (HDiT)，对图像分辨率线性缩放
- 在 ImageNet-1k 256×256 上训练，使用 log-uniform 噪声级别调度
- 高斯分布先验下 IEM = Mahalanobis 距离，有闭形解
- 拉普拉斯先验的例子展示 IEM 沿概率密度山脊分布差异化的局部敏感度
- $\Gamma=1/4$ 在标准 IQA 基准上效果最佳，$\Gamma=10^6$ 在纹理数据集上效果最佳
- IEM 最小化作为优化目标时无伪影，超越 DISTS 等有监督方法
- Mismatched IEM 可用于评估不同生成模型的差异
- 代码已开源，包含去噪器训练、IEM 计算和实验复现的全部细节

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从信息估计理论推导感知度量，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ 全面的基准比较和可视化分析，但缺少大尺度应用
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨优美，插图直观有力
- 价值: ⭐⭐⭐⭐⭐ 为感知度量学习提供了全新的理论框架

<!-- RELATED:START -->

## 相关论文

- [The Spacetime of Diffusion Models: An Information Geometry Perspective](the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](../../NeurIPS2025/image_generation/mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)
- [Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [Monocular Normal Estimation via Shading Sequence Estimation](monocular_normal_estimation_via_shading_sequence_estimation.md)

<!-- RELATED:END -->
