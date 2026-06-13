---
title: >-
  [论文解读] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization
description: >-
  [ICCV 2025][3D视觉][Gaussian Splatting] 提出 CL-Splats，一种基于 3D Gaussian Splatting 的持续学习框架，通过 DINOv2 变化检测、2D→3D 掩码提升和球体约束的局部优化，从稀疏新视图高效增量更新场景重建，在合成和真实场景上大幅超越 CL-NeRF 等方法（PSNR：40.1 vs 30.1 dB），并支持历史恢复和并发更新等应用。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "Gaussian Splatting"
  - "持续学习"
  - "局部优化"
  - "场景更新"
  - "变化检测"
---

# CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization

**会议**: ICCV 2025  
**arXiv**: [2506.21117](https://arxiv.org/abs/2506.21117)  
**代码**: [https://cl-splats.github.io/](https://cl-splats.github.io/)  
**领域**: 3D Vision / Scene Reconstruction  
**关键词**: Gaussian Splatting, 持续学习, 局部优化, 场景更新, 变化检测

## 一句话总结

提出 CL-Splats，一种基于 3D Gaussian Splatting 的持续学习框架，通过 DINOv2 变化检测、2D→3D 掩码提升和球体约束的局部优化，从稀疏新视图高效增量更新场景重建，在合成和真实场景上大幅超越 CL-NeRF 等方法（PSNR：40.1 vs 30.1 dB），并支持历史恢复和并发更新等应用。

## 研究背景与动机

在机器人、混合现实和具身 AI 等应用中，场景随时间不断变化（物体被移动、添加或移除），需要高效更新 3D 场景表示。朴素的方法是重新从头运行 3DGS/NeRF，但这不仅浪费已有重建信息，还要求重新捕获整个场景。

现有持续学习方法存在明显不足：
- **CL-NeRF**: 基于 NeRF 的隐式表示，存在灾难性遗忘问题，无法精确恢复历史状态，且需要知道未变化区域的相机位姿。
- **CLNeRF**: 需要变化区域之外的帧，不够高效，渲染速度 <<1 FPS。
- **3DGS 直接重训**: 稀疏视角下会破坏未观测区域的已有重建。

作者的核心目标是：仅用捕获**变化局部区域**的少量新图像，高效且精确地更新已有 3DGS 重建，同时保持未变化区域的完整性。

## 方法详解

### 整体框架

CL-Splats 的流程分为三个阶段：
1. **2D 变化检测**: 使用 DINOv2 特征比较新图像与已有重建的渲染图像，生成 2D 变化掩码。
2. **3D 掩码提升**: 通过多数投票将 2D 掩码投射到 3D 空间，确定哪些 Gaussian 属于变化区域，并为新物体采样新的点。
3. **局部约束优化**: 仅在变化区域内优化 Gaussian，使用球体约束防止 Gaussian 逃逸，设计高效渲染核心避免全场景计算。

### 关键设计

1. **基于 DINOv2 的变化检测**: 给定新视角图像 $I_i^t$ 和从已有重建 $\mathcal{G}^{t-1}$ 渲染的对应视角图像 $\hat{I}_i^{t-1}$，使用 DINOv2 提取 per-patch 特征图，计算余弦相似度，低于阈值 $\tau_1$ 的区域标记为变化。然后膨胀掩码填充噪声孔洞。选择 DINOv2 而非像素级 L2 或 SSIM 是因 DINOv2 对光照和渲染误差更鲁棒，recall 从 0.745/0.761 提升到 **0.961**。

2. **多数投票 3D 掩码 + 新点采样**: 将已有 Gaussian 投影到每个 2D 掩码上统计，出现在超过 K 个视角掩码中的 Gaussian 被标记为变化区域 $\mathcal{O}^t$。对于新出现的物体（无已有 Gaussian 对应），设计递归采样算法（Algorithm 1）：先随机采样，再在 3D 掩码区域附近采样，确保新区域有足够的初始点。

3. **球体约束局部优化**: 对变化区域的 Gaussian 使用 HDBSCAN 聚类，每个聚类拟合一个包围球。优化过程中动态检查 Gaussian 中心是否在球体并集内，逃逸的 Gaussian 被裁剪。这确保优化严格局部化，不影响未变化区域。球体选择优于矩形包围盒：成员检查所需 FLOPS 只需 1/3。

### 损失函数 / 训练策略

- 使用标准 3DGS 的光度损失，但**仅在动态生成的 2D 渲染掩码内计算**。
- 每步优化时：(1) 将 $\mathcal{O}^t$ 投影到图像平面生成动态渲染掩码；(2) 仅在掩码像素内渲染和计算梯度；(3) 反向传播只更新贡献于掩码像素的 Gaussian。
- 关键性质：局部优化核心产生的梯度与全场景优化对 $\mathcal{O}^t$ 完全一致，但计算量大幅减少。
- 冻结变化区域外所有 Gaussian 的参数是最关键因素（不冻结 PSNR 从 40.8 降到 20.8）。

## 实验关键数据

### 主实验 (表格)

**CL-Splats 数据集 (合成 + 真实)**

| 方法 | 合成 PSNR↑ | 合成 LPIPS↓ | 合成 SSIM↑ | 合成 FPS↑ | 真实 PSNR↑ | 真实 LPIPS↓ | 真实 SSIM↑ | 真实 FPS↑ |
|------|-----------|------------|-----------|----------|-----------|------------|-----------|----------|
| 3DGS | 21.993 | 0.189 | 0.838 | 221 | 11.764 | 0.376 | 0.399 | 125 |
| 3DGS+M | 15.127 | 0.303 | 0.737 | 254 | 8.585 | 0.461 | 0.271 | 151 |
| GaussianEditor | 19.801 | 0.197 | 0.871 | 227 | 24.133 | 0.143 | 0.867 | 137 |
| CLNeRF | 26.758 | 0.322 | 0.738 | <1 | 24.541 | 0.373 | 0.658 | <1 |
| CL-NeRF | 30.063 | 0.058 | 0.939 | <1 | 23.268 | 0.290 | 0.725 | <1 |
| **CL-Splats** | **40.125** | **0.015** | **0.985** | **223** | **28.249** | **0.065** | **0.930** | **135** |

CL-Splats 在合成数据上 PSNR 超第二名 10 dB（40.1 vs 30.1），接近密集重采全场景的 3DGS 上限（~42 dB），同时保持实时渲染（>120 FPS）。

**CL-NeRF 数据集**

| 方法 | PSNR↑ | LPIPS↓ | SSIM↑ |
|------|-------|--------|-------|
| 3DGS | 11.072 | 0.356 | 0.537 |
| CL-NeRF | 27.302 | 0.177 | 0.829 |
| **CL-Splats** | **29.984** | **0.156** | **0.839** |

### 消融实验 (表格)

**优化组件消融 (CL-Splats 数据集 Level 2)**

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 时间 |
|------|-------|-------|--------|------|
| (a) 不冻结背景 | 20.773 | 0.811 | 0.176 | 8 min |
| (b) 全视角投票 | 35.611 | 0.881 | 0.102 | 5 min |
| (c) 无局部核心 | 40.812 | 0.978 | 0.018 | **8 min** |
| (d) 矩形包围盒 | 40.717 | 0.979 | 0.018 | 5 min |
| **(e) 完整方法** | **40.833** | **0.980** | **0.018** | **5 min** |

冻结背景是最关键因素（+20 dB）；局部优化核心将时间从 8 min 减至 5 min（-60%）且精度不变。

**掩码质量对比**

| 方法 | Recall↑ | Precision↑ |
|------|---------|-----------|
| Color L2 | 0.761 | 0.281 |
| SSIM | 0.745 | 0.332 |
| DINOv2 掩码 | **0.961** | 0.370 |
| 完整方法（3D 投影后） | 0.942 | **0.609** |

### 关键发现

- **极快收敛**: CL-Splats 在 5K 迭代（40 秒，RTX Quadro 6000）达到高质量重建，CL-NeRF 需 25K 迭代和 50 分钟（慢 75×）。
- **3DGS+M 反倒更差**: 直接用 2D 掩码约束 3DGS 光度损失（不做 3D 提升）效果比不约束更差（15.1 vs 22.0 dB），验证了 2D 掩码缺乏 3D 空间意识的问题。
- **物体移除最简单，多物体变化最难**: 移除只需删除 Gaussian，而多物体变化涉及多个聚类的同步优化。
- **历史恢复仅需 36MB/步**: 相比朴素存储每步 1173MB，利用局部性只存变化区域和索引，存储效率提升 32×。

## 亮点与洞察

- **显式表示的优势**: 3DGS 的显式 Gaussian 表示天然支持局部编辑、历史恢复和并发更新，这些是 NeRF 隐式表示难以实现的。
- **局部优化核心设计精巧**: 动态投影 + 渲染掩码 + 局部反向传播，保证梯度精确等价于全场景优化，同时计算量大幅降低。
- **HDBSCAN + 球体约束**: 自动聚类变化区域并拟合包围球，优雅地解决了"如何定义 3D 优化边界"的问题。
- **应用前景广**: 并发更新（多个独立变化可并行优化后合并）和历史恢复（高效存储场景演变）对机器人和混合现实意义重大。

## 局限与展望

- 假设场景变化是局部的，无法处理全局光照变化（如白天变夜晚）。
- 依赖 COLMAP 估计新视角位姿，在大幅变化场景下位姿估计可能失败。
- 3DGS 本身在极稀疏视角（2-3 张）下的重建质量受限，方法的性能受基础表示能力约束。
- 大规模户外场景（如自动驾驶数据）的扩展性尚不明确。
- 球体约束可能不适合非凸形状的变化区域，虽然可用多球覆盖但增加复杂度。

## 相关工作与启发

- 与 GaussianEditor 等 3D 编辑方法不同，CL-Splats 基于真实观测而非用户指令驱动更新，更适合自主系统。
- CL-NeRF 和 CLNeRF 是最直接的竞争对手，但它们的 NeRF 表示限制了速度和灵活性。
- DINOv2 用于变化检测是一个轻巧而高效的选择，远优于像素级比较。
- 对机器人领域的启示：家用机器人可通过定期稀疏拍摄来维护不断更新的家庭 3D 地图，追踪物体位置变化。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将持续学习引入 3DGS 的局部优化框架设计完整且有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ 合成+真实数据集，多个 baseline，消融全面，还提供了新数据集
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，图文并茂，流程完整
- **价值**: ⭐⭐⭐⭐⭐ 实际应用价值高，对动态环境中的 3D 重建有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments](gaussianupdate_continual_3d_gaussian_splatting_update_for_changing_environments.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)
- [\[AAAI 2026\] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting](../../AAAI2026/3d_vision/splats_in_splats_robust_and_effective_3d_steganography_towards_gaussian_splattin.md)
- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)

</div>

<!-- RELATED:END -->
