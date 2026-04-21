---
title: >-
  [论文解读] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][高斯泼溅] 将3DGS扩展到7维（空间3D+时间1D+方向3D），通过条件切片机制将7D高斯投影为与3DGS管线兼容的3D高斯，在具有视角依赖效果的动态场景上PSNR提升最高7.36dB，同时维持401 FPS实时渲染。
tags:
  - ICCV 2025
  - 3D视觉
  - 高斯泼溅
  - 动态场景渲染
  - 视角依赖效果
  - 实时渲染
  - 新视图合成
---

# 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2503.07946](https://arxiv.org/abs/2503.07946)  
**代码**: https://gaozhongpai.github.io/7dgs/ (项目页)  
**领域**: 3D视觉  
**关键词**: 高斯泼溅, 动态场景渲染, 视角依赖效果, 实时渲染, 新视图合成

## 一句话总结
将3DGS扩展到7维（空间3D+时间1D+方向3D），通过条件切片机制将7D高斯投影为与3DGS管线兼容的3D高斯，在具有视角依赖效果的动态场景上PSNR提升最高7.36dB，同时维持401 FPS实时渲染。

## 研究背景与动机
动态场景的真实感渲染需要同时建模三个维度：**空间几何**、**时间动态**和**视角依赖外观**。这三者存在复杂的相互依赖——例如，运动物体上的镜面高光随观察方向和位置同时变化。

现有方法分别解决了其中的子问题：
- **4DGS**（空间+时间）：处理动态场景但忽略视角依赖效果
- **6DGS**（空间+方向）：捕获视角依赖但仅限静态场景
- 没有方法能在**统一框架**中同时处理这三个维度并保持实时性能

**核心矛盾**：高维表示带来更强的建模能力，但也带来巨大的计算开销。直接在7D空间操作无法保持实时渲染。

**核心 idea**：将场景元素表示为7D高斯分布，利用多元高斯分布的条件分布性质，通过数学上严格的条件切片机制，将7D高斯"切片"为时间和视角条件下的3D高斯，无缝接入现有3DGS渲染管线。

## 方法详解

### 整体框架
7DGS 每个场景元素建模为7D高斯 $X = (X_p, X_t, X_d) \sim \mathcal{N}(\mu, \Sigma)$，其中 $X_p \in \mathbb{R}^3$ 为空间坐标，$X_t \in \mathbb{R}$ 为时间，$X_d \in \mathbb{R}^3$ 为观察方向。7×7协方差矩阵的交叉项 $\Sigma_{pt}, \Sigma_{pd}, \Sigma_{td}$ 编码三个维度之间的相关性。渲染时通过条件切片将7D降为3D，送入标准3DGS光栅化管线。

### 关键设计

1. **7D高斯表示**：每个场景元素的完整7D协方差矩阵通过 Cholesky 分解 $\Sigma = LL^\top$ 参数化，保证正定性。交叉协方差块 $\Sigma_{pt}$（空间-时间）、$\Sigma_{pd}$（空间-方向）、$\Sigma_{td}$（时间-方向）自然建模三维度的耦合关系。

    - **设计动机**：移动镜面高光需要同时感知位置、时间和方向的变化，单独建模无法捕捉这种耦合。

2. **条件切片机制**：给定观测时间 $t$ 和视角 $d$，利用多元高斯条件分布公式得到空间分量的条件分布：
    $\mu_{cond} = \mu_p + \Sigma_{p,(t,d)} \Sigma_{(t,d)}^{-1} \begin{pmatrix} t - \mu_t \\ d - \mu_d \end{pmatrix}$
    $\Sigma_{cond} = \Sigma_p - \Sigma_{p,(t,d)} \Sigma_{(t,d)}^{-1} \Sigma_{p,(t,d)}^\top$
   同时通过时间和方向调制因子修正不透明度 $\alpha_{cond} = \alpha \cdot f_{temp} \cdot f_{dir}$。
    - **设计动机**：该操作数学上精确，无近似误差，且产出的3D高斯可直接复用3DGS的高效光栅化。

3. **自适应高斯精化（AGR）**：用轻量MLP（$C_{in} \times 64 \times C_{out}$）从特征向量 $f = \mu_p \oplus \mu_t \oplus \mu_d \oplus \gamma(t)$ 预测残差修正 $\Delta\mu_p, \Delta\mu_t, \Delta\mu_d, \Delta l$，在条件切片前动态调整高斯参数，以建模非刚性变形等复杂运动。

    - **设计动机**：条件切片保持高斯形状不随时间变化，AGR弥补了这一局限，使得同一高斯在不同时刻可以表现出不同的空间形状。

### 损失函数 / 训练策略
- 使用与3DGS相同的损失函数（L1 + SSIM）、优化器和超参数
- 仅修改最小不透明度阈值为 $\tau_{min}=0.01$，补偿条件不透明度调制
- 高斯分裂策略新增时间维度判据：当 $\|\Sigma_{pt}\|$ 超过阈值且时间尺度 $\Sigma_t$ 较大时触发分裂
- AGR 网络在3000迭代后开始训练，调制参数 $\lambda_t, \lambda_d$ 在15000迭代后变为可学习

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | 点数↓ |
|--------|------|-------|-------|--------|------|-------|
| 7DGS-PBR (avg) | 4DGS | 27.79 | 0.934 | 0.079 | 192.6 | 641,960 |
| 7DGS-PBR (avg) | **7DGS** | **32.50** | **0.958** | **0.051** | 174.7 | **98,440** |
| D-NeRF (avg) | 4DGS | 33.21 | 0.969 | 0.036 | 296.4 | 255,319 |
| D-NeRF (avg) | **7DGS** | **34.34** | **0.972** | **0.032** | 194.2 | **47,378** |
| Technicolor (avg) | 4DGS | 33.25 | 0.905 | 0.216 | 84.2 | 838,892 |
| Technicolor (avg) | **7DGS** | **33.58** | **0.912** | **0.198** | 79.2 | **416,390** |

在 heart1 场景上提升最大：PSNR 从 27.30 → 35.48（+8.18 dB），点数从 694K 降至 83K（仅需11.9%）。

### 消融实验

| 变体 | PSNR (7DGS-PBR) | FPS | 点数 |
|------|-----------------|-----|------|
| 4DGS (baseline) | 27.79 | 192.6 | 641,960 |
| 7DGS w/o AGR | 31.77 | **376.0** | 88,393 |
| 7DGS (full) | **32.50** | 174.7 | 98,440 |

去掉 AGR 后 PSNR 仍比 4DGS 高 +3.98 dB，且 FPS 翻倍达 376 FPS（D-NeRF 上达 377.8 FPS），证明核心7D表示本身的有效性。AGR 额外带来 +0.73 dB 提升但降低了渲染速度。

### 关键发现
- 7DGS 在视角依赖效果明显的场景（heart、cloud、suzanne）上优势最大，PSNR 提升 4-8 dB
- 7DGS 所需点数仅为 4DGS 的 15-50%，因为统一表示避免了冗余高斯
- 去掉 AGR 的 7DGS w/o AGR 是速度-质量折中的好选择（401 FPS + 远超4DGS的质量）

## 亮点与洞察
- **数学优雅的统一表示**：利用多元高斯条件分布的闭式解将7D降维到3D，无需额外网络，理论上干净
- **向后兼容3DGS生态**：条件切片后可直接复用3DGS的光栅化器、密度控制等，工程落地友好
- **高效**：在更好质量的同时用更少的点，渲染速度可达401 FPS
- 自定义了 7DGS-PBR 数据集（真实CT心脏 + 体积云 + 体积火焰），填补了动态+视角依赖的评测空白

## 局限与展望
- 7D协方差矩阵参数量（28个独立元素 per Gaussian）比3DGS（6个）和4DGS多得多，内存开销增加
- Technicolor 数据集上提升较小（+0.33 dB），说明对真实复杂场景的泛化可能有限
- AGR 的 MLP 引入了额外计算，使完整版 7DGS 的 FPS 低于 4DGS
- 颜色仍用球谐函数表示，未引入时间依赖的颜色建模
- 条件切片需要对 $\Sigma_{(t,d)}$（4×4矩阵）求逆，虽然不大但对大量高斯仍有开销

## 相关工作与启发
- **3DGS / 4DGS / 6DGS**：本文是在GS家族维度扩展的自然延伸，统一了时间和方向
- **D-NeRF / HexPlane**：NeRF路线的动态场景方法，7DGS在速度和质量上全面超越
- **启发**：条件切片的思路可推广到更高维高斯（如加入光照维度、材质属性），或用于其他需要维度约简的场景表示
- 与 Ex4DGS（关键帧插值建模运动）相比，7DGS 通过协方差矩阵隐式编码运动，无需显式关键帧

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次统一空间-时间-方向的7D高斯，条件切片机制设计优雅
- 实验充分度: ⭐⭐⭐⭐ 三个数据集全面评估，含自定义数据集；缺少更多真实场景验证
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，算法伪代码清晰
- 价值: ⭐⭐⭐⭐⭐ 统一框架+实时性能，对动态渲染有重要推进

<!-- RELATED:START -->

## 相关论文

- [EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)
- [Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)

<!-- RELATED:END -->
