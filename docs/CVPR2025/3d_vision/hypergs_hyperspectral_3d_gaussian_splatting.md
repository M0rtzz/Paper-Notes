---
title: >-
  [论文解读] HyperGS: Hyperspectral 3D Gaussian Splatting
description: >-
  [CVPR 2025][hyperspectral][3DGS] 提出 HyperGS，首个将 3D Gaussian Splatting 扩展到高光谱新视角合成的框架，通过潜在空间建模、自适应密度控制和光谱感知剪枝，在高光谱场景重建中实现 14dB 的精度提升。
tags:
  - CVPR 2025
  - 3D Gaussian Splatting
  - hyperspectral imaging
  - novel view synthesis
  - latent space
  - autoencoder
---

# HyperGS: Hyperspectral 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2412.12849](https://arxiv.org/abs/2412.12849)  
**代码**: 未公开  
**领域**: 3d_vision  
**关键词**: hyperspectral imaging, 3D Gaussian Splatting, novel view synthesis, latent space, spectral reconstruction

## 一句话总结

首次将 3DGS 成功扩展到高光谱新视角合成（HNVS），通过在学习的潜在空间中进行高光谱渲染，配合自适应密度控制与像素级光谱剪枝，实现高维光谱数据的高效准确重建。

## 研究背景与动机

**领域现状**: 3DGS 在 RGB 新视角合成中取得了巨大成功，实现了高效高质量的实时渲染。然而，RGB 成像仅捕获三通道信息，无法表达材料的光谱特性。高光谱成像（HSI）通过 128-141 个窄波段通道捕获连续光谱，对遥感、医学诊断和机器人等领域至关重要。

**现有痛点**: (1) HS-NeRF 等基于 NeRF 的 HNVS 方法训练不稳定、渲染速度慢；(2) 直接将 3DGS 扩展到高维光谱数据时，朴素方案无法有效处理高维度带来的优化挑战和阈值设置困难；(3) 缺乏系统性的 HNVS 基准测试。

**核心矛盾**: 高光谱数据维度极高（128-141 通道），直接优化计算开销大，且各通道信噪比差异大，传统 L1+SSIM 损失函数难以稳定训练。

**本文切入角度**: 在学习的潜在空间中进行 3DGS 建模，利用自编码器压缩高维光谱信息，配合深度感知的密度控制和像素级光谱剪枝策略。

## 方法详解

### 整体框架

1. **预处理**：卷积自编码器（AE）将高维光谱图像压缩到低维潜在空间
2. **SfM 初始化**：从灰度通道切片估计 SfM 点云，将 3D 点重投影到潜在高光谱空间获得初始光谱签名
3. **潜在空间 3DGS**：在潜在空间中进行高斯渲染 + MLP 预测视角相关效果
4. **解码与训练**：通过冻结解码器将潜在预测解码为全光谱图像，计算光谱损失

### 关键设计

**1. 高光谱压缩自编码器**
- **功能**: 使用 1D 卷积 + Squeeze-Excitation 块构建对称 AE，将高维光谱数据压缩到低维潜在表示。
- **核心思路**: 编码器通过 max-pooling 压缩光谱维度，解码器通过上采样还原；不使用 skip connection，确保解码器可独立工作。
- **损失函数**: Huber Loss $L_{ae} = L_{Huber}(C^*(p), Dec(Enc(C^*(p))))$，对异常值鲁棒，处理不同相机的信噪比差异。
- **设计动机**: 潜在空间既降低 3DGS 优化的计算开销，又封装相机的光谱灵敏度，为误差提供有界上限。

**2. 深度感知自适应密度控制**
- **功能**: 改进 3DGS 的 split/clone 标准，用深度缩放函数 $h(d,i) = (|\mathbf{E}_d \mathbf{X}_i| / (\beta_{field} \times R))^2$ 调节梯度影响。
- **核心思路**: 通过深度平方缩放 NDC 梯度，减少近相机 Gaussians 的高梯度假信号，使密度控制在高光谱的宽动态范围下更稳定。
- **设计动机**: 高光谱数据通道多、值范围大，传统固定阈值的 3DGS 密度控制无法有效工作，近距离 Gaussians 在多视角下易产生不一致分裂。

**3. 像素级光谱高斯剪枝**
- **功能**: 对每个 Gaussian 计算像素级光谱重要性分数 $\mathcal{I}[g_i, p, d] = (1 - |C^*_d(p) - Dec(f_i)|) \alpha_i T_i$，保留在任意像素 Top-K 内的 Gaussians。
- **核心思路**: 不按平均分数剪枝（会过度去除），而按"是否在某像素的 Top-K 重要度内"决定保留。
- **设计动机**: 跨视角剪枝会导致过度修剪，丢失光谱细节；像素级方法保证每个像素都有足够的光谱表达能力。

### 损失函数

$$L_d(p) = (1-\lambda)(\beta L_{CB}(p) + L_{CS}(p)) + \lambda L_{SSIM}(p)$$

- $L_{CB}$: Charbonnier Loss（比 L1 更平滑，避免敏感波段的极端误差）
- $L_{CS}$: 余弦相似度损失（衡量光谱向量的角度距离，适合光谱比较）
- $L_{SSIM}$: 保持空间和几何一致性
- $\beta$ 权衡光谱损失中的 Charbonnier 和余弦项，$\lambda$ 控制 SSIM 权重

## 实验关键数据

### 主实验

**BaySpec 数据集**（141 通道高噪声，~360 images/scene）:

| 方法 | PSNR↑ | SSIM↑ | SAM↓ | RMSE↓ |
|---|---|---|---|---|
| MipNeRF360 | 26.53 | 0.7442 | 0.0280 | 0.0476 |
| HS-NeRF | 19.82 | 0.6714 | 0.0534 | 0.1071 |
| 3DGS | 22.91 | 0.6321 | 0.1335 | 0.0810 |
| **HyperGS** | **27.11** | **0.7804** | **0.0254** | **0.0440** |

**SOP 数据集**（128 通道低噪声，~40 images/scene）:

| 方法 | PSNR↑ | SSIM↑ | SAM↓ | RMSE↓ |
|---|---|---|---|---|
| MipNeRF360 | 12.28 | 0.6824 | 0.1369 | 0.2658 |
| 3DGS | 28.58 | 0.9627 | 0.0301 | 0.0478 |
| **HyperGS** | **30.51** | **0.9756** | **0.00415** | **0.0354** |

### 消融实验

| 消融步骤 | PSNR↑ | SSIM↑ | SAM↓ | RMSE↓ | N.Prim↓ |
|---|---|---|---|---|---|
| Base 3DGS | 22.91 | 0.6320 | 0.1335 | 0.0810 | 440k |
| + Spec. SFM | 23.05 | 0.6331 | 0.1310 | 0.0799 | 421k |
| + Latent AE | 24.87 | 0.7101 | 0.0548 | 0.0602 | 500k |
| + Densification | 25.25 | 0.7356 | 0.0365 | 0.0548 | 1.3M |
| + Pruning | 25.17 | 0.7199 | 0.0374 | 0.0555 | 412k |
| + View MLP | 27.05 | 0.7792 | 0.0253 | 0.0443 | 309k |
| + Custom Loss | **27.11** | **0.7804** | **0.0254** | **0.0440** | **309k** |

### 关键发现

1. **潜在空间建模是核心贡献**: 从 Base 3DGS 到 +Latent AE，PSNR 提升近 2dB，SAM 从 0.1335 降至 0.0548，光谱精度提升最为显著。
2. **剪枝+密度控制的模型压缩效果**: 密度增强后 Gaussians 从 500k 膨胀到 1.3M，但剪枝后降至 412k，模型更小且光谱更干净。
3. **不同相机场景差异**: 高帧率高噪声的 BaySpec 数据集中 NeRF 方法表现尚可，但低帧率低噪声的 SOP 数据集中 3DGS 方法远超 NeRF，因为 3DGS 的显式表示更适合少视角插值。
4. **全方位超越**: HyperGS 在所有场景和所有指标上均取得最佳结果。

## 亮点与洞察

- 首次系统性地将 3DGS 适配到高光谱领域，建立了完整的 HNVS 基准
- 潜在空间 + AE 的策略具有通用性，可扩展到其他高维信号（如多光谱、红外）的 3DGS 建模
- 视角依赖 MLP 同时预测光谱效果和各向异性不透明度，巧妙处理了光谱的视角变化
- Huber Loss 训练 AE + Charbonnier+Cosine 训练 3DGS 的损失函数设计针对光谱特性优化
- 像素级 Top-K 剪枝策略比传统平均/总量剪枝更适合高维数据

## 局限与展望

- 需要预训练 AE，增加了额外步骤
- 依赖 COLMAP 进行灰度 SfM，可能在纹理不足场景失败
- 实验仅在 HS-NeRF 提供的两个小型数据集上验证，缺乏大规模场景评估
- 合成数据集基于语义标签替换光谱签名，与真实高光谱场景有差距
- 未讨论在线推理速度和实时渲染能力

## 相关工作与启发

- **HS-NeRF**: 首个 HNVS 方法，但非端到端训练且不稳定，为本文提供了数据集和动机
- **Scaffold-GS / Mip-Splatting**: 3DGS 的改进方向（压缩、抗锯齿），但未考虑高维扩展
- **VDGS**: 用混合 NeRF-MLP 预测颜色/不透明度做多光谱重建，但仍依赖 3DGS 基底
- **启发**: 潜在空间压缩 + 显式 3D 表示的组合范式可推广到辐射场景的多模态扩展

## 评分

⭐⭐⭐⭐ — 首次在 3DGS 框架下实现高光谱新视角合成，方法设计合理且消融完整，但实验规模较小、缺乏实时性讨论。

<!-- RELATED:START -->

## 相关论文

- [3D-HGS: 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)
- [GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)
- [VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)

<!-- RELATED:END -->
