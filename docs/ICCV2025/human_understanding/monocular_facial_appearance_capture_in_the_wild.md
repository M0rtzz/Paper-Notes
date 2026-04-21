---
title: >-
  [论文解读] Monocular Facial Appearance Capture in the Wild
description: >-
  [ICCV 2025][人体理解][面部外观捕捉] 提出一种从单目头部旋转视频重建面部外观属性（漫反射反照率、高光强度、高光粗糙度）的方法，通过提出遮挡感知的 split-sum 近似着色模型，在不对光照环境做任何简化假设的情况下实现了逼近工作室级别的面部外观捕捉质量。
tags:
  - ICCV 2025
  - 人体理解
  - 面部外观捕捉
  - 逆渲染
  - 遮挡感知
  - 单目视频
  - split-sum近似
---

# Monocular Facial Appearance Capture in the Wild

**会议**: ICCV 2025  
**arXiv**: [2412.12765](https://arxiv.org/abs/2412.12765)  
**代码**: 无  
**领域**: human_understanding  
**关键词**: 面部外观捕捉, 逆渲染, 遮挡感知, 单目视频, split-sum近似

## 一句话总结

提出一种从单目头部旋转视频重建面部外观属性（漫反射反照率、高光强度、高光粗糙度）的方法，通过提出遮挡感知的 split-sum 近似着色模型，在不对光照环境做任何简化假设的情况下实现了逼近工作室级别的面部外观捕捉质量。

## 研究背景与动机

高质量3D面部扫描对电影、游戏、通信等领域至关重要。传统方法依赖多相机、标定光照的工作室环境，虽然能获得精确的外观贴图（漫反射反照率、高光强度、高光粗糙度），但成本高昂。近年来轻量级面部重建已取得进展，但现有 in-the-wild 方法通常存在以下限制：

- **SunStage** 假设场景中有单一点光源（太阳），限制了适用场景
- **CoRA** 需要在暗室中使用手机闪光灯
- **FLARE** 使用标准 split-sum 近似，忽略自遮挡导致光照烘焙进反照率
- **NextFace** 依赖统计先验，表达能力有限

核心问题在于：现有方法要么假设特定光照条件，要么忽视面部的自遮挡效应（如鼻子对面颊的阴影），导致外观分解不准确。

## 方法详解

### 整体框架

输入为单目头部旋转视频。预处理阶段通过基于关键点的单目追踪获得初始3DMM网格、固定相机位姿和逐帧头部姿态。然后通过可微分渲染的逆渲染优化，同时求解几何形状、外观参数（漫反射反照率 $\rho$、高光强度、高光粗糙度）和环境光照。

### 关键设计

1. **几何优化（Laplacian预条件化）**: 直接优化顶点位置而非 3DMM 混合权重。采用类似 Nicolet 的预条件化框架，通过 $(I + \lambda_{geo} L)^2$ 矩阵偏置梯度步向平滑解：$v \leftarrow v - \eta(I + \lambda_{geo} L)^2 \frac{\partial \mathcal{L}}{\partial v}$。设置 $\lambda_{geo}=19$，可以在使用大学习率的同时保持网格平滑且无自交叉。设计动机：使得几何和纹理可以同时优化，无需传统的两阶段方法。

2. **遮挡感知着色模型（Visibility-Modulated Split-Sum）**: 标准 split-sum 近似将渲染方程分为 BRDF 积分和预滤波环境光贴图两项，但忽略了自遮挡。本文引入可见性项 $V(\mathbf{x}, \omega_i)$，在第二项积分中加入逐点不同的光线可见性调制。对于高光（低粗糙度）分量，利用近似 $\tilde{V}(\mathbf{x}, \omega_r) \approx \frac{1}{K}\sum_{k=1}^{K} \frac{V(\mathbf{x}, \omega_k)}{D(\mathbf{n}, \omega_k, \omega_r, r)}$ 通过蒙特卡洛采样软化可见性。对于漫反射分量则使用 OptiX 光线追踪和多重重要性采样。设计动机：正确建模自遮挡是防止阴影烘焙进反照率的关键。

3. **漫反射正则化**: 添加弱正则项 $\mathcal{L}_{diffuse} = \|I_{diffuse}\|_2^2$ 鼓励漫反射渲染尽可能小。设计动机：防止高光信号被过度烘焙到漫反射分量中，实现更好的漫反射/高光分离。

### 损失函数 / 训练策略

总损失：
$$\mathcal{L} = \mathcal{L}_{img} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{Lap}\mathcal{L}_{Lap} + \lambda_{light}\mathcal{L}_{light} + \lambda_{rough}\mathcal{L}_{rough} + \lambda_{diffuse}\mathcal{L}_{diffuse}$$

- $\mathcal{L}_{img}$：L1 图像重建损失
- $\mathcal{L}_{mask}$：L1 掩码损失（使用MODNet分割）
- $\mathcal{L}_{Lap}$：Laplacian正则化，保持优化网格接近初始3DMM
- $\mathcal{L}_{light}$：白光正则化
- $\mathcal{L}_{rough}$：roughness纹理的全变差正则化
- 几何和纹理同时优化（非两阶段）

## 实验关键数据

### 主实验（重建误差，在皮肤区域上计算）

| 方法 | PSNR ↑ | MAE ↓ | SSIM ↑ | LPIPS ↓ |
|------|--------|-------|--------|---------|
| NextFace | 25.30 | 10.63 | 0.78 | 0.31 |
| SunStage | 29.47 | 5.28 | 0.88 | 0.14 |
| FLARE | 30.40 | 2.01 | 0.94 | 0.15 |
| Ours (w/o vis) | 34.55 | 1.79 | 0.96 | 0.10 |
| **Ours** | **38.09** | **1.18** | **0.97** | **0.10** |

本方法在所有指标上全面领先，PSNR 相比 FLARE 提升 7.69 dB，MAE 降低 41%。

### 消融实验（光线可见性 & 组件分析）

| 设置 | 效果 |
|------|------|
| w/o visibility (标准 split-sum) | 阴影烘焙进反照率；重照明时鼻翼区域阴影错误；PSNR 34.55 vs 38.09 |
| w/o $\mathcal{L}_{diffuse}$ 正则 | 高光信号被过多烘焙到漫反射分量 |
| 优化 3DMM 混合权重 vs 直接优化顶点 | 直接优化顶点可从着色恢复正确鼻型，重建误差更小 |
| 光线追踪 vs 修改的 split-sum（高光） | 光线追踪在逆渲染中产生闪烁和高频伪影，修改的 split-sum 更稳定 |

在合成数据集上的评估进一步证实了遮挡感知的显著优势：漫反射反照率误差大幅降低，形状恢复在自遮挡区域（如嘴唇）也更准确。

### 关键发现

- 不对光照做任何假设（不需要太阳/闪光灯），即可在室内外各类场景工作
- 正确的可见性建模是实现高质量漫反射/高光分离的核心
- 直接优化顶点位置（配合预条件化）比通过 3DMM 参数化更灵活
- 对高光分量使用改进的 split-sum 比直接光线追踪更稳定

## 亮点与洞察

- 技术贡献扎实：修改 split-sum 近似加入可见性项的方案优雅且高效，对高光和漫反射分别采用不同策略（近似可见性 vs 光线追踪）体现了工程智慧
- 端到端框架无需两阶段训练，几何和纹理同时优化的能力来源于 Laplacian 预条件化技巧
- 结果令人信服：定性比较中漫反射/高光分离明显优于 FLARE，重照明效果接近工作室级别
- 实际应用价值高：只需一段简单的头部旋转视频即可获得可用于VFX管线的面部资产

## 局限与展望

- 依赖头部姿态估计的准确性，不精确的姿态会严重影响重建质量
- 当面部在所有帧中都处于极端阴影时无法恢复外观
- 当前面部模板不包含眼睛模型
- 无法保证正确的肤色恢复（光照-外观歧义性）
- 假设表情不变，限制了对说话/表情丰富视频的适用性

## 相关工作与启发

- 与 FLARE 的核心差异在于可见性建模：FLARE 的标准 split-sum 忽略自遮挡导致大量烘焙，本文通过修改公式从根源解决
- 与 SunStage 的差异在光照假设：SunStage 假设已知太阳位置作为单一点光源
- Munkberg et al. 的可微分 split-sum 是本文着色模型的基础
- 对面部扫描领域有重要启发：轻量级捕捉与工作室级质量之间的差距正在缩小

## 评分

- **新颖性**: ⭐⭐⭐⭐ 遮挡感知split-sum近似是核心创新点
- **实验充分度**: ⭐⭐⭐⭐ 合成数据和真实数据均有充分验证
- **写作质量**: ⭐⭐⭐⭐⭐ 公式推导清晰，图示质量高
- **价值**: ⭐⭐⭐⭐ 显著缩小了轻量级捕捉与工作室质量之间的差距

<!-- RELATED:START -->

## 相关论文

- [NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction](ngd_neural_gradient_based_deformation_for_monocular_garment_reconstruction.md)
- [Understanding Co-speech Gestures in-the-wild](understanding_co-speech_gestures_in-the-wild.md)
- [MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances](magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)
- [Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)

<!-- RELATED:END -->
