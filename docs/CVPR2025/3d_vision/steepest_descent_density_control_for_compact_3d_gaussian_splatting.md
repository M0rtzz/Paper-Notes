---
title: >-
  [论文解读] Steepest Descent Density Control for Compact 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] SteepGS 从非凸优化理论出发，揭示了 3DGS 中密度控制的本质是帮助高斯基元逃离鞍点，并推导出最优分裂策略——分裂为两个后代、透明度减半、沿分裂矩阵最小特征向量方向位移——在保持渲染质量的同时将高斯点数减少约 50%。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯泼溅
  - 密度控制
  - 鞍点逃逸
  - 分裂矩阵
  - 紧凑表示
---

# Steepest Descent Density Control for Compact 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2505.05587](https://arxiv.org/abs/2505.05587)  
**代码**: [https://vita-group.github.io/SteepGS](https://vita-group.github.io/SteepGS)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 密度控制, 鞍点逃逸, 分裂矩阵, 紧凑表示

## 一句话总结

SteepGS 从非凸优化理论出发，揭示了 3DGS 中密度控制的本质是帮助高斯基元逃离鞍点，并推导出最优分裂策略——分裂为两个后代、透明度减半、沿分裂矩阵最小特征向量方向位移——在保持渲染质量的同时将高斯点数减少约 50%。

## 研究背景与动机

**领域现状**：3D Gaussian Splatting（3DGS）通过高斯基元混合来表示辐射场，实现了实时高分辨率新视角合成。其核心训练流程交替进行梯度优化和自适应密度控制（ADC）——ADC 通过 clone 和 split 操作动态调整高斯点数量，以覆盖场景细节。

**现有痛点**：原始 ADC 基于启发式规则（视空间梯度范数阈值 + 尺度阈值）来决定分裂，经常产生大量冗余高斯点——典型场景可达 300 万以上点，导致：(1) 内存占用过高；(2) 渲染速度下降；(3) 存储开销大，难以部署到手机/VR 头显等资源受限设备。

**核心矛盾**：密度控制的目标是用最少的点达到最好的渲染质量，但现有方法缺乏理论指导——不清楚什么时候应该分裂、分裂成几个、新点放在哪里、透明度怎么调。已有改进如 3DGS-MCMC 和 Revising-3DGS 仍依赖启发式，改进有限。

**本文目标**：从优化理论角度揭示密度控制的底层机制，推导出有理论保证的最优分裂策略，实现紧凑点云。

**切入角度**：作者观察到训练过程中很多高斯基元位于"鞍点"区域——在当前参数下梯度不足以进一步降低损失，但这些区域确实还有待改善。分裂操作实际上是在逃离鞍点：将一个点变成两个方向不同的点，打破了梯度停滞的困局。这与神经架构分裂（S2D、Firefly）中的思路有理论关联。

**核心 idea**：引入"分裂矩阵" $\mathbf{S}^{(i)}$ 来刻画每个高斯点的分裂行为，证明分裂能降低损失 iff 分裂矩阵有负特征值；最优分裂是将后代沿最小特征向量方向对称位移、透明度各取 1/2。

## 方法详解

### 整体框架

SteepGS 的整体流程与标准 3DGS 相同——从 SfM 点云初始化，交替进行光度误差优化和密度控制。核心区别在于将原始 ADC 替换为基于分裂矩阵的 Steepest Density Control (SDC)。每 100 步执行一次密度控制：计算每个高斯的分裂矩阵 → 求最小特征值 → 负特征值的点执行分裂 → 后代沿特征向量方向对称放置。分裂矩阵的计算被嵌入到 CUDA 内核中并行执行。

### 关键设计

1. **分裂矩阵 (Splitting Matrix)**:

    - 功能：完全刻画分裂操作对损失函数的影响，决定一个高斯点是否应该被分裂以及如何分裂
    - 核心思路：通过 Theorem 1 对分裂后的损失做二阶泰勒展开，将损失变化分解为"均值位移项"（等价于标准梯度下降的效果）和"分裂特征函数" $\Delta^{(i)}$ 之和。分裂特征函数取二次型形式 $\frac{1}{2}\sum_j w_j^{(i)} \delta_j^{(i)\top} \mathbf{S}^{(i)} \delta_j^{(i)}$，其中分裂矩阵 $\mathbf{S}^{(i)} = \mathbb{E}[\frac{\partial \ell}{\partial \sigma_\Pi} \nabla^2_{\theta^{(i)}} \sigma_\Pi]$ 结合了损失梯度和高斯基元的 Hessian。当且仅当 $\lambda_{\min}(\mathbf{S}^{(i)}) < 0$ 时分裂能降低损失
    - 设计动机：原始 ADC 使用视空间位置梯度范数作为分裂条件，但这只是一个启发式代理指标。分裂矩阵从损失函数的二阶信息出发，精确判断哪些点真正受困于鞍点，避免对已经"安居"的点做无效分裂

2. **最陡密度控制 (Steepest Density Control, SDC)**:

    - 功能：给出理论最优的分裂方案，以最少的后代数实现最大的损失下降
    - 核心思路：Theorem 2 证明了最优解的三个结论：(a) 分裂为 $m_i^* = 2$ 个后代即为最优,更多后代不会带来额外收益；(b) 权重 $w_1 = w_2 = 1/2$，即每个后代的透明度减半以保持局部密度守恒；(c) 位移方向 $\delta_1 = \mathbf{v}_{\min}(\mathbf{S}^{(i)}), \delta_2 = -\mathbf{v}_{\min}(\mathbf{S}^{(i)})$，即沿分裂矩阵最小特征向量的正负方向对称放置。这保证了在约束条件（位移范数有界、权重和为1）下的最陡下降
    - 设计动机：原始 ADC 的 clone 操作沿梯度方向放置后代、split 操作从父分布随机采样并缩放尺度 0.8 倍，都缺乏理论最优性保证。SDC 给出了解析解，避免了随机性带来的不确定性

3. **高效 CUDA 实现**:

    - 功能：使分裂矩阵的计算可以高效地集成到现有 3DGS 的训练流程中
    - 核心思路：分裂矩阵的两个成分中，损失梯度 $\partial\ell/\partial\sigma_\Pi$ 在反向传播时已经计算过可以复用；高斯基元的 Hessian $\nabla^2_\theta \sigma_\Pi$ 有解析形式 $\sigma^{(i)}\mathbf{\Upsilon}\mathbf{\Upsilon}^\top - \sigma^{(i)}\mathbf{P}^\top\Pi(\Sigma^{(i)})^{-1}\mathbf{P}$，主要依赖前向计算中已有的投影和协方差信息。对于 3×3 的分裂矩阵，特征值分解可以用 Smith (1961) 的解析公式直接计算，无需迭代
    - 设计动机：如果分裂矩阵的计算过于昂贵就失去了实用价值。通过复用已有中间计算结果和利用小矩阵的解析特征分解，将额外计算开销降到最低

### 损失函数 / 训练策略

训练损失与标准 3DGS 相同：$\ell_1$ 像素距离 + SSIM。SDC 每 100 步从第 500 步开始执行。分裂阈值设为 $\lambda_{\min} < -1e{-6}$。其余超参数保持 3DGS 默认值。每个场景在单张 V100 GPU 上训练。

## 实验关键数据

### 主实验

| 方法 | MipNeRF360 #Points↓ | PSNR↑ | SSIM↑ | T&T PSNR↑ | DeepBlending PSNR↑ |
|------|---------------------|-------|-------|-----------|-------------------|
| 3DGS (原始) | 3.339M | 29.037 | 0.872 | 23.743 | 29.690 |
| 3DGS + Thres. | 1.632M | 27.851 | 0.848 | 22.415 | 29.374 |
| 3DGS-MCMC | 1.606M | 28.149 | 0.853 | 22.545 | 29.439 |
| Revising 3DGS | 1.606M | 28.085 | 0.850 | 22.339 | 29.439 |
| **SteepGS** | **1.606M** | **28.734** | **0.857** | **23.684** | **29.963** |

### 消融实验（隐含在主实验中）

| 配置 | MipNeRF360 PSNR↑ | vs 同点数基线 | 说明 |
|------|-----------------|-------------|------|
| 3DGS (3.339M) | 29.037 | - | 原始 full model |
| SteepGS (1.606M) | 28.734 | 最优 | 点数减半仅降 0.3 dB |
| 3DGS-MCMC (1.606M) | 28.149 | +0.585 | SteepGS 同等点数下显著优于 |
| Revising 3DGS (1.606M) | 28.085 | +0.649 | 启发式方法改进有限 |
| 3DGS + Thres. (1.632M) | 27.851 | +0.883 | 简单截断效果最差 |

### 关键发现

- SteepGS 在 MipNeRF360 上用约 48% 的点数（1.606M vs 3.339M）仅损失 0.3 dB PSNR，在 Tank&Temple 上甚至与原始 3DGS 持平（23.684 vs 23.743），在 DeepBlending 上更好（29.963 vs 29.690）
- 在相同点数预算下，SteepGS 在所有数据集和指标上一致优于 3DGS-MCMC 和 Revising-3DGS
- 可视化显示 SteepGS 的分裂策略更聚焦于真正需要细化的区域（如椅子座面），原始 ADC 则在已经训练好的区域（如靠背）也大量分裂，导致冗余

## 亮点与洞察

- **从理论上揭示了"分裂 = 逃离鞍点"**这一深刻洞察——将 3DGS 的密度控制与非凸优化理论（鞍点逃逸）建立了优雅的联系，这个视角不仅解释了为什么分裂有效，还告诉我们什么时候分裂无效
- **"分裂为 2 个后代是最优的"这一结论**为原始 ADC 的设计选择提供了理论依据，同时否定了更多后代可能更好的直觉
- **透明度减半的结论**纠正了 3DGS-MCMC 和 Revising-GS 中基于渲染启发式的不精确透明度调整方案
- 分裂矩阵只需要"部分 Hessian 信息"（逐点的而非跨点的），计算量可控——这是理论优雅性和实用性的完美结合

## 局限与展望

- 当前只对位置参数 $\mathbf{p}$ 构建分裂矩阵（$\dim\Theta=3$），未涉及协方差/颜色等其他参数，理论上可以扩展
- 分裂阈值 $-1e{-6}$ 是固定的，理论上可以设计自适应阈值策略
- 论文没有报告训练时间对比——虽然分裂矩阵计算利用了已有信息，但特征分解和额外的 Hessian 计算仍有开销
- 与后处理剪枝方法（如 LightGaussian）正交，可以组合使用进一步压缩

## 相关工作与启发

- **vs 原始 ADC**: ADC 基于视空间梯度启发式，无法保证分裂降低损失；SDC 有理论保证且更紧凑
- **vs 3DGS-MCMC**: MCMC 将密度控制重写为确定性状态转移，仍基于启发式的 opacity 采样；SDC 从损失函数出发有更强理论基础
- **vs Revising-GS**: 基于像素误差驱动密度控制，方向正确但分裂方案仍启发式
- **vs S2D/Firefly（神经架构分裂）**: SteepGS 将神经元分裂的理论框架迁移到 3DGS 场景中，但需要处理高斯基元的特殊结构（投影、α-blending 等）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从非凸优化视角理解 3DGS 密度控制是一个深刻且全面的理论贡献
- 实验充分度: ⭐⭐⭐⭐ 三个标准数据集验证充分，但缺少训练时间和消融实验的详细数据
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，物理直觉和数学形式的结合非常好
- 价值: ⭐⭐⭐⭐⭐ 50% 点数减少对 3DGS 实际部署有重大意义，理论框架可指导未来研究

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](../../ECCV2024/3d_vision/pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] 3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations](3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)

<!-- RELATED:END -->
