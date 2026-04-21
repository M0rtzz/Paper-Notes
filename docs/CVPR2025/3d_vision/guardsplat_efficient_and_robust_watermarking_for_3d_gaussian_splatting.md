---
title: >-
  [论文解读] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3DGS] 提出 GuardSplat，通过 CLIP 引导的消息解耦优化（仅训练解码器 5 分钟）和 SH-aware 水印嵌入（仅修改球谐偏移量），实现对 3DGS 资产的高容量、高保真、鲁棒版权保护，总优化时间仅 15 分钟。
tags:
  - CVPR 2025
  - 3D视觉
  - 3DGS
  - watermarking
  - CLIP
  - spherical harmonics
  - copyright protection
  - anti-distortion
---

# GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2411.19895](https://arxiv.org/abs/2411.19895)  
**代码**: [GitHub](https://github.com/NarcissusEx/GuardSplat)  
**领域**: 3d_vision  
**关键词**: 3DGS, watermarking, CLIP, spherical harmonics, copyright protection, anti-distortion

## 一句话总结

提出 GuardSplat，通过 CLIP 引导的消息解耦优化（仅训练解码器 5 分钟）和 SH-aware 水印嵌入（仅修改球谐偏移量），实现对 3DGS 资产的高容量、高保真、鲁棒版权保护，总优化时间仅 15 分钟。

## 研究背景与动机

**领域现状**: 3DGS 凭借高保真度和实时渲染速度被广泛应用于影视、游戏、VR 等领域。然而 3DGS 资产的版权保护尚未有效解决。

**现有痛点**:
- **方案 (a) 直接在水印图像上训 3DGS**: 新视角不能保证一致水印，提取精度低
- **方案 (b) 逐场景训练解码器 (CopyRNeRF等)**: 每个场景都要从头训解码器，耗时严重
- **方案 (c) 使用 2D 预训练解码器 (WateRF等)**: 2D 水印网络有保真度-容量权衡，编码器-解码器联合训练耗时；直接用其解码器优化 3D 模型结果次优
- **通用问题**: 修改 3DGS 所有属性（位置、协方差、不透明度等）会破坏 3D 结构导致低保真度

**核心矛盾**: 如何在保持 3DGS 渲染质量的前提下，高效嵌入大容量水印且抵抗各种失真攻击？

**本文切入角度**: 利用 CLIP 的文本-图像对齐能力构建从文本域到图像域的桥梁：在文本域训练解码器（无需任何图像），然后直接应用于图像域提取水印。

## 方法详解

### 整体框架

三个阶段：
1. **消息解码器训练** (5 min): 将二进制消息编码为 CLIP 文本 token → CLIP 文本编码器 → MLP 解码器提取消息
2. **SH-aware 水印嵌入** (10 min): 冻结 3DGS 所有属性，仅学习球谐偏移量 $\mathbf{h}_i^o$
3. **消息提取**: 渲染视图 → CLIP 视觉编码器 → 预训练解码器 → 提取消息

### 关键设计

**1. CLIP 引导的消息解耦优化（Message Decoupling Optimization）**
- **功能**: 仅训练消息解码器 $\mathcal{D}_M$（3 层 MLP，512→L），不需要训练编码器或任何图像
- **核心思路**:
    - 将二进制消息 $M \in \{0,1\}^L$ 通过映射函数 $\Phi$ 转为 CLIP token 序列 $T$
    - CLIP 文本编码器 $\mathcal{E}_T$ 提取文本特征 $F_T \in \mathbb{R}^{512}$
    - MLP 解码器从 $F_T$ 恢复消息 $\hat{M}$，用 BCE 损失优化
    - 推理时利用 CLIP 文本-图像对齐特性：解码器可直接从 CLIP 视觉特征 $F_V$ 提取消息
- **设计动机**: 解耦编码器和解码器训练，解码器训练不受保真度约束，仅需 5 分钟；CLIP 的 400M 对训练数据提供了丰富的跨模态表示

**2. SH-aware 消息嵌入（SH-aware Message Embedding）**
- **功能**: 为每个 3D Gaussian 创建可学习 SH 偏移量 $\mathbf{h}_i^o \in \mathbb{R}^{48}$，冻结其余所有属性
- **核心思路**: 
    - 只修改球谐系数（控制视角相关的颜色效果），不改变位置 $\mu$、协方差 $\Sigma$、不透明度 $\alpha$
    - SH 参数控制高光/镜面反射等效果，仅少量区域敏感，因此小幅偏移对整体保真度影响极小
    - 偏移量正则: $\mathcal{L}_{off} = -\frac{1}{N}\sum_{i=1}^{N}\|\mathbf{h}_i^o\|_2^2$
- **设计动机**: 仅修改颜色表示保持 3D 结构完整，防止恶意用户通过修改模型文件去除水印

**3. 抗失真消息提取（Anti-distortion Message Extraction）**
- **功能**: 在优化过程中引入可微失真层，随机模拟裁剪、缩放、旋转、JPEG 压缩、亮度抖动
- **核心思路**: 让 SH 偏移量在训练时就学会对抗各种失真
- **设计动机**: CLIP 自身对高斯模糊和噪声有天然鲁棒性，但对旋转和 JPEG 压缩脆弱，需要显式增强

### 损失函数

$$\mathcal{L} = \lambda_{recon}(\mathcal{L}_{rgb} + \mathcal{L}_{lpips}) + \lambda_{msg}\mathcal{L}_{msg} + \lambda_{off}\mathcal{L}_{off}$$

- $\lambda_{recon}=1$, $\lambda_{msg}=0.03$, $\lambda_{off}=10$
- $\mathcal{L}_{rgb}$: SSIM + L1 重建损失
- $\mathcal{L}_{lpips}$: LPIPS 感知损失
- $\mathcal{L}_{msg}$: 消息提取 BCE 损失

## 实验关键数据

### 主实验（Blender + LLFF, 32-bit）

| 方法 | Bit Acc | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| CopyRNeRF | 78.08 | 26.13 | 0.896 | 0.041 |
| WateRF | 88.58 | 31.19 | 0.936 | 0.040 |
| GaussianMarker | 98.85 | 33.98 | 0.979 | 0.016 |
| **GuardSplat (Ours)** | **99.04** | **39.40** | **0.994** | **0.002** |

PSNR 比 GaussianMarker 高 5.4 dB，LPIPS 低 87%。

### 鲁棒性（16-bit, 多种失真）

| 失真 | GuardSplat | GaussianMarker | WateRF |
|---|---|---|---|
| None | 99.64 | 99.36 | 95.67 |
| Rotation (±π/6) | **94.56** | 70.84 | 93.13 |
| JPEG (10%) | **94.70** | 86.22 | 86.99 |
| VAE Attack | **82.35** | 52.00 | 51.73 |
| Combined | **93.38** | 83.49 | 84.12 |

### 效率对比

| 方法 | 解码器训练 | 水印嵌入 | 总时间 |
|---|---|---|---|
| CopyRNeRF | - | ~hours | hours |
| WateRF | ~hours | ~30min | hours |
| GaussianMarker | - | ~30min | ~30min |
| **GuardSplat** | **5min** | **10min** | **15min** |

### 关键发现

1. **CLIP 桥梁有效**: 文本域训练的解码器可直接迁移到视觉域，比提取精度高于预训练 2D 解码器
2. **SH-only 修改关键**: 对比修改所有属性（Offset_all），仅修改 SH 偏移量保真度大幅提升（PSNR +5 dB），且水印更难被去除
3. **StegExpose 安全检测**: ROC 曲线接近 Reference 线，水印不可被隐写分析检测
4. **48-bit 大容量**: 即使嵌入 48 位消息，Bit Acc 仍达 98.29%，PSNR 38.90

## 亮点与洞察

- CLIP 文本-图像对齐的创造性利用：在文本域训练解码器是一个优雅的 zero-shot 迁移方案
- SH-aware 嵌入对 3DGS 的 domain knowledge 充分利用：SH 控制视角相关效果，是最佳嵌入位点
- 15 分钟总优化时间使得实际商业部署成为可能
- 抗失真模块的设计使水印在 VAE 攻击下仍可提取（82.35%），远超竞品

## 局限与展望

- SH 偏移量存储在模型文件中，理论上可被针对性攻击（如 SH 重置+微调）
- CLIP ViT-B/32 的视觉编码器分辨率有限，可能影响高频细节的水印嵌入
- 未验证在 2DGS、3DGS++ 等变体上的兼容性
- 标准 CLIP 可能被微调版本替代导致解码失败
- 仅在合成（Blender）和简单真实（LLFF）场景测试

## 相关工作与启发

- CopyRNeRF 开创了 NeRF 水印，但逐场景训练不实用；本文解耦优化解决了效率瓶颈
- GaussianMarker 修改所有属性导致保真度下降；本文 SH-only 策略是更优方案
- 启发：CLIP 的对齐特性可作为跨模态"翻译器"用于更多 3D 安全/隐私任务

## 评分

⭐⭐⭐⭐ — 方法设计巧妙，CLIP 解耦和 SH-only 嵌入两个核心设计都有洞察力支撑；实验全面，覆盖容量/保真度/鲁棒性/安全性/效率五个维度。

<!-- RELATED:START -->

## 相关论文

- [3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)
- [Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](../../ICCV2025/3d_vision/robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)
- [Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)

<!-- RELATED:END -->
