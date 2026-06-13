---
title: >-
  [论文解读] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 本文提出 AAA-Gaussians，通过自适应 3D 平滑滤波器、视空间透视正确包围盒、基于视锥体的 3D 裁剪三项技术，在统一框架内系统解决了 3DGS 的锯齿、投影失真、弹出伪影等问题，在 in-distribution 和 out-of-distribution 视角下均实现了 SOTA 的无伪影实时渲染。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "抗锯齿"
  - "伪影消除"
  - "3D评估"
  - "实时渲染"
---

# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering

**会议**: ICCV 2025  
**arXiv**: 无 (CVF Open Access)  
**代码**: [https://github.com/DerThomy/AAA-Gaussians](https://github.com/DerThomy/AAA-Gaussians)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 抗锯齿, 伪影消除, 3D评估, 实时渲染

## 一句话总结

本文提出 AAA-Gaussians，通过自适应 3D 平滑滤波器、视空间透视正确包围盒、基于视锥体的 3D 裁剪三项技术，在统一框架内系统解决了 3DGS 的锯齿、投影失真、弹出伪影等问题，在 in-distribution 和 out-of-distribution 视角下均实现了 SOTA 的无伪影实时渲染。

## 研究背景与动机

1. **领域现状**：3D Gaussian Splatting (3DGS) 因实时渲染能力和高质量图像合成而广受关注。然而原始 3DGS 将 3D 高斯体近似投影为 2D splat 进行渲染，这一简化在非标准相机设置（如大 FOV、VR 场景）下会引入多种伪影。

2. **现有痛点**：3DGS 存在四类伪影：(1) 仿射投影近似导致的变形伪影，尤其在图像边缘和大 FOV 下；(2) 缩放距离变化时的锯齿伪影；(3) 高斯体延伸到视锥体外时的弹出伪影；(4) 全局深度排序简化导致的 popping 伪影。现有工作虽然分别尝试解决这些问题，但要么引入显著性能损失，要么只解决部分问题。

3. **核心矛盾**：2D splat 评估快但近似不准确，3D 光线追踪准确但太慢且需要额外加速结构。最近的混合 2D/3D 方法虽然在 3D 中评估高斯体，但仍依赖屏幕空间计算（如 2D 包围盒、2D 抗锯齿滤波），在分布外视角下仍然容易产生伪影。

4. **本文目标**：在 3DGS 渲染管线的所有步骤中贯彻 3D 高斯体的 3D 本质，从根本上消除各类伪影，同时保持光栅化的效率优势。

5. **切入角度**：作者观察到现有方法的根本问题在于渲染管线中的某些步骤仍在 2D 屏幕空间操作，而完全在 3D 空间中处理（包围、裁剪、抗锯齿、评估）可以统一解决所有伪影。

6. **核心 idea**：提出自适应 3D 平滑滤波器替代 2D Mip 滤波器、视空间包围替代屏幕空间包围、视锥体裁剪替代 2D tile 裁剪，构建首个完全无伪影的 3DGS 渲染框架。

## 方法详解

### 整体框架

AAA-Gaussians 建立在分层光栅化器之上，保留 StopThePop 的逐光线排序和 Hahlbohm 等人的 3D 高斯评估方式，但替换了包围、裁剪、深度评估、贡献估计和抗锯齿组件为全 3D 实现。致密化采用 MCMC 方案。输入为训练图像集，输出为高质量且视角一致的 3DGS 场景表示。

### 关键设计

1. **自适应 3D 抗锯齿滤波器**:

    - 功能：消除因相机距离变化导致的欠采样和过采样锯齿伪影。
    - 核心思路：现有的 Yu 等人 3D 平滑滤波器在 3D 评估模式下会导致高斯体过度透明，因为其振幅归一化因子 $\sqrt{|\Sigma|/|\hat{\Sigma}|}$ 按体积变化缩放，但 3D 评估只取光线最大贡献点而非积分。本文提出仅按垂直于视线方向 $d$ 的面积变化来调整振幅：$\hat{G}_\perp(x) = \sqrt{|\Sigma_\perp|/|\hat{\Sigma}_\perp|} \cdot \exp(-\frac{1}{2}(x-\mu)^\top \hat{\Sigma}^{-1}(x-\mu))$，其中 $\Sigma_\perp$ 是投影到 $d$ 正交子空间的 2×2 协方差矩阵。同时结合训练时的最大采样频率 $\hat{v}_{train}$，当相机靠近时不过度收缩高斯体。
    - 设计动机：朴素的按体积变化调整会对高度各向异性的高斯体过度缩小振幅，导致透明化。按垂直方向面积变化调整更符合光栅化评估的物理语义。

2. **视空间透视正确包围**:

    - 功能：消除高斯体延伸到图像平面后方时的弹出（pop-in）伪影。
    - 核心思路：Hahlbohm 等人在屏幕空间拟合平面来包围高斯体，当高斯体的 z 范围超出近远平面时直接丢弃，导致弹出伪影。本文改为在视空间中用角度 $\theta$ 和 $\phi$ 拟合包围平面，通过求解 $\theta_{1,2} = \tan^{-1}(\frac{s_{1,3} \pm \sqrt{s_{1,3}^2 - s_{1,1}s_{3,3}}}{s_{3,3}})$ 得到视空间角度范围，再限制到 $[-\pi/2+\epsilon, \pi/2-\epsilon]$ 并映射到屏幕。当相机在高斯体内部时丢弃该高斯。
    - 设计动机：屏幕空间包围在高斯体跨越图像平面时完全失效，视空间包围通过角度表示天然避免了这个问题。

3. **基于视锥体的 3D 裁剪**:

    - 功能：加速渲染并减少排序开销，同时消除不正确裁剪导致的伪影。
    - 核心思路：将 2D tile 裁剪提升为 3D 视锥体裁剪。对每个 tile 构建由 4 个屏幕空间平面定义的 3D 视锥体 $\mathcal{F}$，然后在高斯归一化空间中找到视锥体内距原点最近的点（即最大贡献点），若 $\rho(x)^2 > \tau_\rho$ 则裁剪。通过只投影到离高斯中心最近的 x/y 平面及其对应边来优化计算，将评估从朴素的 4 面+4 边减少到 2 面+3 边。
    - 设计动机：2D 轴对齐包围盒 (AABB) 对非轴对齐椭球体包围不紧，导致大量无效计算。3D 视锥体裁剪能精确判断高斯体是否对特定 tile 有贡献，显著减少排序和渲染开销。

### 损失函数 / 训练策略

- 使用 MCMC 致密化方案，与其他方法保持相同的高斯数量
- 3D 滤波器核大小 $k = 0.3$，与 Mip-Splatting 一致
- 分层 k-buffer 排序用于正确的逐像素混合顺序

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 伪影 |
|--------|------|-------|-------|--------|------|
| Mip-NeRF 360 | 3DGS | 27.44 | 0.814 | 0.215 | 失真+锯齿+弹出 |
| Mip-NeRF 360 | MCMC | 28.03 | 0.836 | 0.187 | 失真+锯齿+弹出 |
| Mip-NeRF 360 | **Ours** | **27.84** | **0.836** | **0.188** | **无** |
| Deep Blending | 3DGS | 29.51 | 0.902 | 0.237 | 失真+锯齿+弹出 |
| Deep Blending | **Ours** | **30.49** | **0.913** | **0.222** | **无** |

**大 FOV 评估（3× FOV 扩大）**:

| 数据集 | MCMC | Taming-3DGS | **Ours** |
|--------|------|-------------|----------|
| Mip-NeRF 360 PSNR | 23.35 | 23.30 | **27.84** |
| T&T PSNR | 14.37 | 11.55 | **23.58** |
| Deep Blending PSNR | 18.32 | 20.33 | **30.49** |

### 消融实验

| 配置 | PSNR↑(MipNeRF360) | 伪影 |
|------|-------------------|------|
| Full model | 27.84 | 无 |
| w/o hier. sort | 27.90 (+0.06) | 有popping |
| w/o AA | 27.81 (-0.03) | 有锯齿 |
| w/o 3D eval | 27.87 (+0.03) | 有失真 |

### 关键发现

- 在 in-distribution 视角下，去掉各组件可能反而提高指标（因为可以利用 per-view 不一致性作弊），但在 out-of-distribution 下差距极大
- 大 FOV 场景下，MCMC/Taming-3DGS 等方法 PSNR 暴跌 10+ dB，本文方法几乎不受影响
- 视锥体裁剪将渲染时间减半（14.40ms → 7.72ms），是关键的性能优化
- 3D 评估的额外开销可忽略，甚至可能更快（7.64ms vs 7.72ms）

## 亮点与洞察

- **统一框架解决所有伪影**：首次在单一方法中同时消除锯齿、投影失真、弹出和排序伪影，而非像之前工作那样各自解决一个问题。对 VR/AR 等需要大 FOV 的应用有重要意义。
- **自适应 3D 滤波器的理论洞察**：发现按体积缩放振幅在 3D 评估模式下不正确，改为按垂直方向面积缩放，这个物理直觉简洁而有效。
- **in-distribution 指标的欺骗性**：证明了去掉某些组件可能在标准测试集上反而提分（通过 per-view 作弊），凸显了 out-of-distribution 评估的重要性。

## 局限与展望

- 分层排序引入了约 2× 的性能开销（7.72ms vs 4.11ms w/o sort）
- 对非重叠假设导致的混合伪影无法解决（需昂贵的体积积分）
- 当前实验主要在静态场景上，对动态场景的泛化需要进一步验证

## 相关工作与启发

- **vs Mip-Splatting**：Mip-Splatting 的 2D 屏幕空间 Mip 滤波器无法与 3D 评估结合，本文的 3D 自适应滤波器完全在 3D 空间操作
- **vs StopThePop**：StopThePop 解决了排序导致的 popping，但仍有投影和锯齿伪影；本文在其基础上补全了完整的解决方案
- **vs Hybrid Transparency**：Hybrid Transparency 在图像边缘仍有 pop-in，且抗锯齿能力不足

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个统一解决 3DGS 所有伪影的框架
- 实验充分度: ⭐⭐⭐⭐⭐ in/out-of-distribution + 多分辨率 + 详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐⭐ 对 3DGS 在 VR/AR 等实际应用有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Anti-Aliased 2D Gaussian Splatting](../../NeurIPS2025/3d_vision/anti-aliased_2d_gaussian_splatting.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [\[ICCV 2025\] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
