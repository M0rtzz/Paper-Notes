---
title: >-
  [论文解读] Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning
description: >-
  [AAAI 2026 Oral][3D视觉][雨天3D重建] 提出 OmniRain3D 数据集（首次同时建模视角异质性和亮度动态性的雨天 3D 场景数据集）以及 REVR-GSNet 端到端框架（联合递归亮度增强 + 高斯基元优化 + GS引导去雨），实现从雨天退化图像到高保真干净 3D 场景的重建。
tags:
  - "AAAI 2026 Oral"
  - "3D视觉"
  - "雨天3D重建"
  - "3D高斯溅射"
  - "去雨"
  - "亮度增强"
  - "端到端框架"
---

# Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.06734](https://arxiv.org/abs/2511.06734)  
**代码**: [https://github.com/ncfjd/REVR-GSNet](https://github.com/ncfjd/REVR-GSNet)  
**领域**: 3D视觉  
**关键词**: 雨天3D重建, 3D高斯溅射, 去雨, 亮度增强, 端到端框架

## 一句话总结

提出 OmniRain3D 数据集（首次同时建模视角异质性和亮度动态性的雨天 3D 场景数据集）以及 REVR-GSNet 端到端框架（联合递归亮度增强 + 高斯基元优化 + GS引导去雨），实现从雨天退化图像到高保真干净 3D 场景的重建。

## 研究背景与动机

### 领域现状

3D 场景重建（如 NeRF、3DGS）在干净场景下已取得优异效果，但在雨天等恶劣天气下，多视角图像受到雨滴/雨线遮挡和可见度下降的影响，导致多视角一致性被破坏，重建质量严重退化。这对自动驾驶和机器人导航等需要全天候工作的系统构成重大挑战。

### 现有痛点

作者识别了两个被现有工作忽视的关键特征：

**视角异质性（Perspective Heterogeneity）**：3D 空间中的雨滴投影到不同视角的 2D 图像上，外观会发生变化（仰视呈 Λ 型扩散、水平呈平行、俯视呈 v 型收敛；相机偏离降雨方向时雨线会倾斜）。现有数据集（如 HydroViews）直接线性叠加 2D 雨层，缺乏 3D 物理一致性。

**亮度动态性（Brightness Dynamicity）**：真实雨天中云层覆盖会导致环境亮度降低，雨越大亮度越暗。现有数据集（如 RainyScape）虽然在 3D 空间模拟雨效果，但忽略了降雨对亮度的影响，与真实场景存在显著域差距。

### 核心矛盾与切入角度

现有方法（DerainNeRF、DerainGS）通常采用**两阶段流水线**：先用预训练去雨网络去除退化，再做 3D 重建。这种分离范式存在两个问题：(1) 预训练模型可能过拟合特定雨模式；(2) 缺少亮度调整机制。本文提出端到端统一框架，同时处理去雨与亮度恢复。

## 方法详解

### 整体框架

REVR-GSNet 采用**联合交替优化**策略，包含三个模块的协同工作：

- **Phase 1**：RBE（递归亮度增强）+ GPO（高斯基元优化）联合优化——逐步提升亮度并嵌入到 3DGS 中
- **Phase 2**：GPO + GRE（GS引导去雨）联合优化——利用 3DGS 渲染图引导去雨，去雨结果反馈优化 3DGS
- **Phase 3**：仅 GPO——生成最终干净辐射场 $V^M$

### 关键设计

#### 1. OmniRain3D 数据集构建

**功能**：构建首个同时包含视角异质性和亮度动态性的雨天 3D 场景数据集。

**核心流程**：
- **视角提取**：用 COLMAP 从干净背景图提取所有相机外参，获取仰角 $\theta$ 和方位角 $\phi$
- **动态雨线渲染**：在 Blender 中建立 3D 雨模型，包含六维气象参数 $S = \{\omega_{den}, \omega_{dep}, \omega_{str}, \omega_{dir}, \omega_{qty}, \omega_{scl}\}$（密度、深度、风力、风向、雨量、尺度），对每个相机姿态同步渲染雨线
- **自适应亮度调节**：基于 Beer-Lambert 定律的指数衰减模型：

$$L = L_0 e^{-\gamma \omega_{den}}$$

设置三级雨密度（小雨、中雨、大雨），计算对应亮度，最终**亮度调整的背景 + 对应密度的雨线 = 合成雨天图像**。

整体成像模型：$O_t(\theta_i, \phi_j) = L \odot (B_t(\theta_i, \phi_j) + R_t(\theta_i, \phi_j))$

**设计动机**：弥补 HydroViews（仅 2D 叠加）和 RainyScape（忽略亮度）的不足，提供更接近真实场景的训练/评估数据。

#### 2. Recursive Brightness Enhancement (RBE，递归亮度增强)

**功能**：逐步校正低亮度雨天图像的亮度。

**核心思路**：采用轻量 CNN（CPEN，7 层卷积 + 对称跳连）估计亮度调整参数，然后递归应用二次亮度增强曲线：

$$\mathbf{BE}(I_t, A_1) = I_t + A_1 I_t (1 - I_t)$$

递归 $n=4$ 步，每步用不同参数 $A_a$ 逐步提升亮度。

**设计动机**：单步增强难以处理严重暗化，递归方式逐步逼近目标亮度，且参数化曲线保证增强的可控性。

#### 3. Gaussian Primitives Optimization (GPO，高斯基元优化)

**功能**：利用增强后的多视角图像构建和优化 3D 高斯场景表征。

**核心流程**：
- 从增强图像 $\{E_t^i\}$ 用 COLMAP 估计相机姿态
- 构建 3DGS 表征 $V = \{\mu_z, \Sigma_z, \sigma_z, h_z\}$（位置、协方差、不透明度、球谐系数）
- 通过可微分光栅化优化 3DGS 属性

**关键洞察**：虽然增强图像仍含雨线，但利用**跨视角一致性**和空间相关性，辐射场优化过程能有效抑制这些伪影。

#### 4. GS-guided Rain Elimination (GRE，GS引导去雨)

**功能**：利用当前 3DGS 渲染的参考图引导去雨过程。

**核心思路**：渲染图 $R_t$ 相比增强图 $E_t$ 具有更少的雨伪影和更清晰的结构（因为 3DGS 已做了多视角融合）。采用递归雨估计网络 (RREN)：
- 将 $R_t$ 和 $E_t$ 拼接作为输入
- 使用带 LSTM 单元的循环 U-Net 架构，嵌入残差循环块 (RRB)
- 循环 $l=6$ 步，每步估计雨线图 $M_o$，通过残差减法得到去雨图 $D_t$
- 去雨图反馈到 GPO 继续优化 3DGS

$$D_t = \text{Cat}(R_t, E_t) - E_\phi(\text{Cat}(R_t, E_t))$$

**设计动机**：3DGS 渲染图已经"隐式去雨"了部分，将其作为引导信号帮助去雨网络更好地区分雨线和场景内容。

### 损失函数 / 训练策略

- 整体采用端到端训练，PyTorch + RTX 3090
- RBE 和 GRE 使用 Adam 优化器，初始学习率 $10^{-3}$
- GPO 的 3DGS 各属性使用不同学习率（means: $1.6 \times 10^{-4}$, scaling: $5 \times 10^{-4}$, SH: $2.5 \times 10^{-3}$）
- 所有方法统一训练 30,000 步做公平对比

## 实验关键数据

### 主实验

**OmniRain3D 雨线场景**（正常亮度，4 个场景）：

| 场景 | 指标 | REVR-GSNet | DerainGS | RainyScape | DerainNeRF |
|------|------|-----------|---------|-----------|-----------|
| Francis | PSNR↑ | **24.56** | 23.40 | 22.99 | 16.17 |
| Garden | PSNR↑ | **25.35** | 25.30 | 22.58 | 21.74 |
| Garden | LPIPS↓ | **0.184** | 0.200 | 0.241 | 0.320 |
| Caterpillar | PSNR↑ | **21.48** | 20.26 | 19.90 | 13.99 |

**OmniRain3D 暗光雨天场景**（4 个场景，其他方法均需亮度预处理）：

| 场景 | 指标 | REVR-GSNet | DerainGS† | RainyScape† | DerainNeRF† |
|------|------|-----------|----------|------------|------------|
| Bicycle | PSNR↑ | **19.06** | 18.88 | 18.63 | 18.13 |
| Family | PSNR↑ | **17.83** | 17.78 | 16.92 | 17.05 |
| Family | LPIPS↓ | **0.440** | 0.461 | 0.497 | 0.595 |

**HydroViews 雨滴场景**（3 个场景平均）：

| 场景 | REVR-GSNet | RainyScape | DRSformer* | NeRD-Rain* |
|------|-----------|-----------|-----------|-----------|
| Stump (PSNR) | **22.61** | 22.59 | 18.23 | 19.79 |
| Stump (LPIPS) | **0.258** | 0.284 | 0.303 | 0.336 |

### 消融实验

在 HydroViews 数据集上的组件消融：

| 配置 | GPO | RBE | GRE | PSNR↑ | SSIM↑ |
|------|-----|-----|-----|-------|-------|
| 仅 GPO | ✓ | | | 19.03 | 0.514 |
| GPO + RBE | ✓ | ✓ | | 22.71 | 0.615 |
| GPO + GRE | ✓ | | ✓ | 21.64 | 0.535 |
| **完整模型** | **✓** | **✓** | **✓** | **23.88** | **0.687** |

RBE 贡献最大（PSNR +3.68），说明亮度恢复是暗光雨天重建的关键。GRE 也有显著贡献（+2.61），三者联合效果最佳。

### 关键发现

1. 在暗光雨天场景中 REVR-GSNet 优势更明显——其他方法需要亮度预处理但仍不如端到端方案
2. OmniRain3D 数据集上所有基线方法（3DGS、NeRF、RainyScape）的表现都优于 HydroViews 数据集训练的版本，验证了数据集更高的真实感
3. 亮度直方图分析显示 OmniRain3D 的亮度分布与真实雨天图像更接近
4. 在真实雨天场景上也展现了良好的泛化能力

## 亮点与洞察

1. **问题定义精准**：首次明确提出雨天 3D 重建中"视角异质性"和"亮度动态性"两个被忽视的关键特征，并为此构建专门数据集
2. **闭环设计**：RBE → GPO → GRE 的交替优化形成闭环，去雨改善重建、重建反过来引导去雨，互相增强
3. **物理建模**：OmniRain3D 的构建基于 Beer-Lambert 定律等物理模型，而非简单的图像处理
4. **实用性强**：端到端框架避免了多阶段方法的错误累积，对真实场景的泛化更好

## 局限与展望

1. 数据集虽然比 HydroViews 更真实，但仍是合成数据，与真实世界雨天仍有域差距（如雾气、积水反射等未建模）
2. REVR-GSNet 在部分场景上的提升幅度有限（如 Lgnatius 场景仅比 DerainGS 高 0.46 PSNR），在异常复杂场景下可能需要更强的去雨模块
3. 亮度衰减模型假设了雨密度-亮度的简单指数关系，实际场景中还受时间、光源方向等因素影响
4. 未讨论计算效率——交替优化策略可能增加训练时间
5. 仅与较少（5个）baseline 对比，缺少更多最新去雨方法的比较

## 相关工作与启发

- **3DGS 在恶劣天气下的应用**是一个新兴方向，本文为雨天场景提供了范式参考
- 类似的思路可扩展到雾天（亮度衰减 → 雾浓度衰减）、雪天等其他天气条件
- RBE 的递归曲线增强思想源自 Zero-DCE (CVPR 2020)，在 3D 重建场景中的应用是新颖的
- 端到端"渲染引导恢复"的范式（GRE）可推广到其他退化场景（如运动模糊、灰尘等

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题视角新颖（两个被忽视的特征），数据集和方法均有贡献
- 实验充分度: ⭐⭐⭐⭐ — 多数据集评估+消融+真实场景+数据集对比，但 baseline 偏少
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、框架清晰，但部分细节（如训练策略的切换时机）可更详细
- 价值: ⭐⭐⭐⭐ — 数据集和方法均有实用价值，为恶劣天气 3D 重建提供基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[AAAI 2026\] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)
- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)

</div>

<!-- RELATED:END -->
