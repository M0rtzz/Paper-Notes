---
title: >-
  [论文解读] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 首次将事件相机与可变形 3D 高斯溅射（3D-GS）结合用于动态场景重建，提出 GS-阈值联合建模策略和动静分解策略，在新构建的事件-4D 基准上实现了 SOTA 的渲染质量和速度（合成数据平均 PSNR 提升 2.73dB，渲染速度达 4D-GS 的 1.71 倍）。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯溅射
  - 事件相机
  - 动态场景重建
  - 阈值建模
  - 动静分解
---

# Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2411.16180](https://arxiv.org/abs/2411.16180)  
**代码**: 即将公开  
**领域**: 3d_vision  
**关键词**: 3D高斯溅射, 事件相机, 动态场景重建, 阈值建模, 动静分解

## 一句话总结

首次将事件相机与可变形 3D 高斯溅射（3D-GS）结合用于动态场景重建，提出 GS-阈值联合建模策略和动静分解策略，在新构建的事件-4D 基准上实现了 SOTA 的渲染质量和速度（合成数据平均 PSNR 提升 2.73dB，渲染速度达 4D-GS 的 1.71 倍）。

## 研究背景与动机

动态场景重建和新视角合成是 VR/AR 等沉浸式应用的基础。虽然 3D-GS 通过高效可微光栅化实现了实时渲染，但其动态扩展方法（如 4D-GS、Deformable-3DGS）受限于 RGB 相机的固有局限：

**低帧率**：RGB 相机帧间缺失中间运动信息，导致快速运动重建质量下降

**运动模糊**：高速运动场景进一步恶化重建质量

**事件相机的优势**：微秒级时间分辨率，可捕获帧间连续运动和近无限视点的监督信号。但将事件引入 3D-GS 面临核心挑战：**阈值变化建模**。事件触发依赖亮度变化阈值 $C$，该阈值在极性、空间和时间上均存在复杂变化，现有方法假设常数阈值会显著降低事件监督质量（参见 Fig. 3a）。

## 方法详解

### 整体框架

方法包含两个核心策略：
1. **GS-阈值联合建模（GTJM）**：解决事件阈值变化问题
2. **动静分解（DSD）**：分离动态和静态高斯，提升效率和质量

### GS-阈值联合建模（GTJM）

事件相机的亮度变化模型为：

$$E(t, t+\Delta t) = \int_{t}^{t+\Delta t} C \cdot e(\tau) d\tau$$

渲染估计的亮度变化为：

$$\hat{E}(t, t+\Delta t) = \log(\hat{I}(t+\Delta t)) - \log(I(t))$$

**第一阶段：RGB 辅助阈值估计**

利用 RGB 帧之间的真实亮度变化来监督阈值优化。将事件累积为事件计数图 $ECM_{t,f} \in \mathbb{R}^{B \times P \times H \times W}$，使用可学习阈值参数 $\hat{C}_{t,f}$：

$$\hat{E}_{thres}(t,f) = \sum_{b=1}^{B}\sum_{p=1}^{P}(ECM_{t,f} \odot \hat{C}_{t,f})_{b,p,:,:}$$

阈值建模损失：$\mathcal{L}_{thres} = \|E_{thres}(t,f) - \hat{E}_{thres}(t,f)\|_2^2$

**第二阶段：GS 增强的阈值精炼**

RGB 帧稀疏导致监督不足。关键洞见：训练好的 3D-GS 可以渲染中间帧作为伪监督，增强阈值优化。冻结 GS，联合使用 $\mathcal{L}_{thres}$ 和 $\mathcal{L}_{event}$ 优化阈值。

**联合优化**：最终同时优化阈值和 3D-GS：

$$\hat{C}^*, GS^* = \arg\min_{\hat{C}, GS}(\mathcal{L}_{thres} + \mathcal{L}_{event} + \mathcal{L}_{rgb})$$

形成相互增强的正循环：优化的阈值改善事件监督→更好的 3D-GS→更准确的伪帧→更精确的阈值估计。

### 动静分解（DSD）

**问题**：现有方法统一使用动态高斯建模整个场景，浪费了变形场容量，降低了渲染速度。

**2D 分解**：利用静态高斯"天然无法表示运动"的特性。前 3k 次迭代仅用静态高斯训练，动态区域自然重建质量差。用预训练 VGG19 提取多尺度特征，计算渲染图和真值图的余弦相似度图，直方图呈双峰分布，通过 Otsu 方法生成动态区域掩码。

**2D→3D 对应**：将多视角动态区域像素反投影到 3D 空间，基于空间邻近性将对齐的 3D 点映射到高斯。

**缓冲区软分解**：使用双半径 $r_1, r_2$：$r_1$ 内为动态，$r_2$ 外为静态，中间区域裁剪作为缓冲区，允许自适应密度控制优化分解边界。

**联合渲染**：静态高斯跳过变形场，与变形后的动态高斯合并送入光栅化器。变形场使用 MLP 输出位置、旋转、缩放位移。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{thres} + \mathcal{L}_{event} + \mathcal{L}_{rgb}$$

其中 $\mathcal{L}_{rgb} = (1-\lambda_s)\|\hat{I}(t) - I(t)\|_1 + \lambda_s \mathcal{L}_{D-SSIM}(\hat{I}(t), I(t))$

## 实验关键数据

### 主实验：合成数据集定量结果（Table 2，8 个场景平均）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| 3D-GS（静态基线） | ~22.7 | ~0.913 | ~0.098 | ~233 |
| K-Planes | ~23.2 | ~0.913 | ~0.044 | ~2.37 |
| 4D-GS | ~25.6 | ~0.944 | ~0.069 | ~89 |
| Deformable-3DGS | ~25.5 | ~0.938 | ~0.033 | ~70 |
| Event-4DGS | ~28.8 | ~0.950 | ~0.039 | ~55 |
| **Ours** | **~31.6** | **~0.966** | **~0.022** | **~156** |

关键发现：
- 事件引入（Event-4DGS vs Deformable-3DGS）：平均 +3.28 dB
- 阈值建模（Ours vs Event-4DGS）：平均 +2.73 dB
- 渲染速度：平均 1.71× 快于 4D-GS

### 真实世界数据集（Table 3）

| 方法 | Excavator PSNR | Jeep PSNR | Flowers PSNR | Eagle PSNR |
|------|---------------|-----------|-------------|-----------|
| 4D-GS | 28.35 | 28.34 | 26.82 | 27.59 |
| Event-4DGS | 29.67 | 29.64 | 27.53 | 29.08 |
| **Ours** | **31.28** | **30.41** | **28.57** | **31.29** |

FPS 同样大幅领先：Ours 179/89/149/192 vs Event-4DGS 57/47/40/63。

### 消融实验（Table 4，合成数据集平均）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| w/o GTJM | 29.39 | 0.956 | 0.034 | 153 |
| w/o Joint Opt. in GTJM | 30.87 | 0.963 | 0.026 | 152 |
| w/o DSD | 30.78 | 0.961 | 0.026 | **57** |
| w/o Buffer-based Soft Dec. | 31.02 | 0.963 | 0.025 | 138 |
| **Full** | **31.56** | **0.966** | **0.022** | 156 |

关键发现：
- GTJM 贡献 +2.17 dB PSNR 提升
- DSD 在保持质量的同时将 FPS 从 57 提升至 156（2.74×）
- 缓冲区软分解额外贡献 +0.54 dB

### 阈值建模的互增强验证（Table 1）

| 方向 | 阶段 | 效果 |
|------|------|------|
| TM→3D Rec. | RGB辅助初始化 → 联合优化 | PSNR: 24.46 → 26.63 |
| 3D Rec.→TM | 冻结GS辅助 | MSE: 8.317 → 7.077 (×10⁻⁴) |
| 联合优化 | 同时优化两者 | PSNR: 28.01, MSE: 6.322 |

## 亮点与洞察

1. **相互增强的范式**：GS-阈值联合建模创造了一个正循环——更好的阈值产生更准确的事件监督，更好的 3D-GS 提供更精确的伪帧用于阈值精炼
2. **"静态高斯不能表示运动"的巧妙利用**：无需额外语义或运动先验，仅通过前 3k 迭代的重建误差就能自动识别动态区域
3. **缓冲区软分解的鲁棒性**：当缓冲区大小超过约 12 个基本单位时，重建质量趋于稳定，降低了超参数敏感度
4. **首个事件-4D 基准**：8 个合成 + 4 个真实世界场景，为后续研究提供标准化评估平台

## 局限性

1. 真实世界数据采集系统（分束器+事件相机+帧相机+STM32）复杂，部署成本较高
2. 单目设置限制了 3D 重建精度
3. DSD 仅执行一次，对于动态区域随时间显著变化的场景可能不够灵活

## 相关工作与启发

- 与 DE-NeRF（Ma et al., 2023）首次结合事件的动态 NeRF 不同，本文在 3D-GS 框架下实现了实时渲染
- 阈值建模思路可推广到其他事件相机应用（如 SLAM、光流估计）
- 动静分解策略与场景流估计结合可进一步提升动态区域重建质量

## 评分 ⭐⭐⭐⭐

创新性 ★★★★☆：首次将事件相机引入可变形 3D-GS，阈值联合建模和动静分解都有新意
实验 ★★★★★：自建基准全面，合成和真实世界均有充分评估，消融实验详细
写作 ★★★★☆：方法描述清晰，图表丰富
实用性 ★★★☆☆：依赖事件相机硬件，应用场景相对小众

<!-- RELATED:START -->

## 相关论文

- [TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction](timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)
- [BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)
- [BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)

<!-- RELATED:END -->
