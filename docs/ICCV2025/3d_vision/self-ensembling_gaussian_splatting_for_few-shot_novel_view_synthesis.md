---
title: >-
  [论文解读] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis
description: >-
  [3D视觉] SE-GS 通过不确定性感知扰动策略在训练过程中动态生成多样化的 3DGS 模型，并利用自集成机制使 Σ-model 聚合扰动模型的信息，有效缓解稀疏视角下的过拟合问题，在多个数据集上实现 SOTA 的少样本新视角合成性能。
tags:
  - 3D视觉
---

# Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2411.00144](https://arxiv.org/abs/2411.00144)
- **代码**: [项目页面](https://chen-zhao.com/SE-GS)
- **领域**: 3D视觉
- **关键词**: 3D Gaussian Splatting, 少样本新视角合成, 自集成学习, 不确定性感知扰动

## 一句话总结
SE-GS 通过不确定性感知扰动策略在训练过程中动态生成多样化的 3DGS 模型，并利用自集成机制使 Σ-model 聚合扰动模型的信息，有效缓解稀疏视角下的过拟合问题，在多个数据集上实现 SOTA 的少样本新视角合成性能。

## 研究背景与动机

3D Gaussian Splatting（3DGS）在新视角合成中表现优异，但在**稀疏视角训练**时容易过拟合：

**过拟合现象严重**：实验显示（Fig.2），训练集性能随迭代持续提升，但测试集在约 2000 次迭代后开始下降；3 视角时过拟合更为显著

**现有方法的局限**：
   - 深度先验方法（DNGaussian, FSGS）引入噪声深度估计，在训练视角增多时可能反而损害性能
   - 多模型正则化（CoR-GS）训练多个 3DGS 模型计算代价高，且模型间缺乏足够多样性

**集成学习的潜力**：集成学习在检测和分割中已被证明有效缓解过拟合，但如何在 3DGS 中实现高效集成尚未被探索

## 方法详解

### 整体框架

SE-GS 联合训练两个模型：
- **Δ-model**：在可用训练图像上正常训练，并被动态扰动以生成多样化的模型
- **Σ-model**：通过最小化与扰动模型的差异实现自集成，推理时使用此模型

### 关键设计一：不确定性感知扰动

朴素的全局随机扰动会使模型偏离过远，导致不稳定。SE-GS 通过以下步骤精确扰动：

**1. 创建伪视角**：通过训练视角之间的球面线性插值（SLERP）生成 $M$ 个伪视角：

$$\hat{\mathbf{R}} = \text{SLERP}(\mathbf{R}_1, \mathbf{R}_2, \beta)$$

**2. 计算不确定性图**：在缓冲区中存储不同训练步数渲染的伪视角图像，计算像素级不确定性：

$$\mathbf{U} = \sqrt{\frac{1}{S}\sum_{i=1}^S (\mathbf{I}_i - \bar{\mathbf{I}})^2}$$

并进行 $k=5$ 的局部平滑 $\hat{\mathbf{U}}$。

**3. 选择性扰动**：仅扰动与高不确定性像素重叠的 Gaussian：

$$\hat{G}_\Delta^t = G_\Delta^t + \delta_t \cdot h(G_\Delta^t, \hat{\mathcal{U}}^t)$$

其中指示函数 $h$ 判断 Gaussian 投影区域的最大不确定性是否超过阈值 $\tau$。对旋转使用 6D 连续表示进行扰动以保持连续性。

### 关键设计二：自集成正则化

Σ-model 在训练视角上正常训练，同时通过额外正则化约束其与扰动模型的一致性：

$$\mathcal{L}_r = (1-\lambda)\|\mathbf{I}_\Sigma^t - \mathbf{I}_\Delta^t\|_1 + \lambda\mathcal{L}_{\text{D-SSIM}}(\mathbf{I}_\Sigma^t, \mathbf{I}_\Delta^t)$$

其中 $\lambda=0.2$，正则化在伪视角上以自监督方式执行，不依赖额外 GT 信号。

### 总损失函数

$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \gamma\mathcal{L}_r$$

其中 $\gamma=1$，$\mathcal{L}_{\text{RGB}}$ 是训练视角上的光度损失。

### 关键优势
- 扰动模型从 Δ-model 派生而非从头训练，计算开销可忽略
- 不确定性在 2D 渲染空间计算，自然处理训练中 Gaussian 数量变化的问题
- 正则化在伪视角执行，独立于外部信息（如深度）

## 实验

### 主实验：LLFF 数据集

| 方法 | 3-view PSNR | 6-view PSNR | 9-view PSNR |
|------|-----------|-----------|-----------|
| 3DGS | 19.22 | 23.80 | 25.44 |
| DNGaussian | 19.12 | 22.01 | 22.62 |
| FSGS | 20.43 | 24.09 | 25.31 |
| CoR-GS | 20.45 | 24.49 | 26.06 |
| **SE-GS** | **20.79** | **24.78** | **26.36** |

SE-GS 在 3/6/9 视角设置下均取得最佳 PSNR，同时在 SSIM 和 LPIPS 上也全面领先。

### 消融实验：扰动策略对比

| 扰动方式 | PSNR | SSIM | LPIPS |
|---------|------|------|-------|
| 无扰动（仅 cross-model）| 基准 | - | - |
| 全局随机扰动 | 下降 | - | - |
| **不确定性感知扰动** | **最高** | **最高** | **最低** |

- 朴素全局扰动因扰动过大反而损害性能
- 不确定性感知的选择性扰动显著优于其他策略

### 关键发现
- 缓冲区大小 $S=5$ 和伪视角数 $M=10$ 为最优配置
- SE-GS 比训练 $k$ 个独立模型的显式集成更高效更有效
- 随着训练视角增多（如 9 视角），SE-GS 相对 vanilla 3DGS 的提升依然显著
- 不同于深度先验方法，SE-GS 不会在视角增多时性能退化

## 亮点与洞察

1. **首次将自集成机制引入 3DGS**：巧妙地利用训练动态中的不确定性信号
2. **计算高效**：相比显式多模型集成，几乎不增加训练成本
3. **即插即用**：与深度先验等正交方法可结合使用
4. **自监督正则化**：不依赖外部 GT 深度或生成的新视角图像

## 局限性
- 仍需要一定数量的初始 SfM 点作为 3DGS 初始化
- 伪视角必须位于训练视角的插值范围内，对外推场景效果有限
- 对极少（如 1-2 张）视角的场景，不确定性估计可能不够可靠

## 相关工作
- 3DGS：显式点基表示，支持实时渲染
- DNGaussian / FSGS：利用单目深度先验缓解稀疏视角问题
- CoR-GS：训练多个 3DGS 模型进行交叉正则化
- 集成学习：通过聚合多模型预测提高鲁棒性，temporal ensemble 和 consistency regularization

## 评分
- **创新性**: ⭐⭐⭐⭐ — 不确定性感知扰动+自集成的思路新颖
- **实用性**: ⭐⭐⭐⭐ — 低开销、无需外部数据、多数据集全面提升
- **实验完整度**: ⭐⭐⭐⭐⭐ — LLFF/DTU/Mip-NeRF360/MVImgNet 全面验证+充分消融
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法推导严谨

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)

<!-- RELATED:END -->
