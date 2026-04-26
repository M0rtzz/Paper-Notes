---
title: >-
  [论文解读] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution
description: >-
  [CVPR 2025][超分辨率] 提出 BF-STVSR，结合 B 样条映射器（时间平滑插值）和傅里叶映射器（空间高频捕获）实现连续时空视频超分辨率，完全无需预训练光流网络（RAFT），在 GoPro 数据集上 PSNR 达 30.22dB，FLOPs 在所有方法中最低。
tags:
  - CVPR 2025
  - 超分辨率
  - B-spline
  - Fourier
  - continuous STVSR
  - temporal interpolation
---

# BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2501.11043](https://arxiv.org/abs/2501.11043)  
**代码**: 有  
**领域**: 视频理解  
**关键词**: video super-resolution, B-spline, Fourier, continuous STVSR, temporal interpolation

## 一句话总结
提出 BF-STVSR，结合 B 样条映射器（时间平滑插值）和傅里叶映射器（空间高频捕获）实现连续时空视频超分辨率，完全无需预训练光流网络（RAFT），在 GoPro 数据集上 PSNR 达 30.22dB，FLOPs 在所有方法中最低。

## 研究背景与动机

**领域现状**：连续时空视频超分辨率（C-STVSR）需要同时实现任意时间插值和任意空间放大。现有方法如 MoTIF、VideoINR 依赖预训练光流网络（RAFT）来建模帧间运动。

**现有痛点**：(1) 依赖 RAFT 增加计算成本和推理延迟，RAFT 本身需要额外的预训练和推理开销。(2) 位置编码（PE）在 C-STVSR 中反而降低性能——这与其在 NeRF 等领域的表现完全相反，是一个值得关注的发现。(3) 现有方法在处理大运动和复杂纹理时，空间细节保持不足。

**核心矛盾**：时间维度需要平滑的运动建模（避免抖动和不连续），空间维度需要高频细节恢复（捕获纹理和边缘），两者对表示方式的需求截然不同，用统一的 MLP 难以同时满足。

**本文目标** 设计无需外部光流的高效 C-STVSR 框架，同时改善时间一致性和空间细节质量。

**切入角度**：B 样条基函数天然具备局部平滑性和可控性，适合运动轨迹建模（时间）；傅里叶基函数擅长捕获信号的频率成分，适合空间高频细节恢复。

**核心 idea**：用 B 样条预测运动向量实现时间平滑插值，用傅里叶基捕获空间高频细节，两者互补，彻底摆脱对 RAFT 光流的依赖。

## 方法详解

### 整体框架
输入两帧低分辨率帧 $I_0^L, I_1^L$，编码器 E 产生三个潜在特征：$F_0^L, F_{(0,1)}^L, F_1^L$。B-spline Mapper 基于时间坐标 $t$ 预测高分辨率运动向量和可靠性图；Fourier Mapper 估计主频率和振幅增强空间特征。前向 warp 后通过最近邻上采样，拼接时间和模板特征送入解码器生成最终帧。

### 关键设计

1. **B-spline Mapper（时间插值）**

    - 功能：预测高分辨率运动向量 $M_{0 \to t}^H$, $M_{1 \to t}^H$ 和可靠性图
    - 核心思路：$p_\psi(z_r, \delta_r, \hat{t}) = c_r \odot \beta^n((\hat{t} - k_r)/d)$，其中 $c_r$ 为预测系数，$k_r$ 为预测节点，$d$ 为帧间隔相关的膨胀参数
    - 设计动机：B 样条基函数具有局部支撑和 $C^{n-1}$ 连续性，天然保证运动轨迹在时间方向的平滑性，比 MLP 的全局非线性更适合运动建模

2. **Fourier Mapper（空间增强）**

    - 功能：捕获空间域的主频率和振幅，增强高频细节
    - 核心思路：$g_\phi(z_r, \delta_r) = A_r \odot [\cos(\pi F_r \delta_r); \sin(\pi F_r \delta_r)]$，$A_r$ 为振幅估计，$F_r$ 为频率估计
    - 设计动机：解决 MLP 的频谱偏置（spectral bias）问题——MLP 倾向学习低频信号，而视频超分辨率恰恰需要恢复高频细节

### 损失函数 / 训练策略
- 简化损失：$\mathcal{L} = \mathcal{L}_{char}(\hat{I}_t^H, I_t^H)$，仅 Charbonnier 像素损失，移除了对 RAFT 的运动监督
- 两阶段训练：Stage 1 固定 ×4 放大 450K 迭代 → Stage 2 均匀采样放大倍数 [2,4] 训练 150K 迭代
- Adam 优化器，cosine annealing 学习率 $10^{-4}$ → $10^{-7}$
- 批量大小 32，随机旋转和翻转增强

## 实验关键数据

### 主实验

| 数据集 | 设置 | BF-STVSR PSNR | BF-STVSR SSIM |
|--------|------|--------------|--------------|
| Vid4 | ×4空间 ×2时间 | 25.85 | 0.7772 |
| GoPro-Center | ×4空间 ×8时间 | 31.17 | 0.8898 |
| GoPro-Average | ×4空间 ×8时间 | 30.22 | 0.8802 |
| Adobe-Average | ×4空间 ×8时间 | 30.12 | 0.8808 |

参数量 13.47M（vs MoTIF 12.55M），FLOPs 在所有方法中最低。

### 消融实验

| 配置 | GoPro PSNR | SSIM |
|------|-----------|------|
| MoTIF+RAFT baseline | 30.04 | 0.8773 |
| +B-spline only | 30.12 | 0.8783 |
| +Fourier only | 30.16 | 0.8792 |
| +B-spline+Fourier | **30.22** | **0.8802** |
| 加入位置编码 | 性能下降 | — |

### 关键发现
- 去掉 RAFT 监督不影响甚至略微提升性能，B 样条完全替代外部光流
- **位置编码在 C-STVSR 中有害**——与 NeRF 领域完全相反，值得深入研究
- FLOPs 最低且推理约 70-100ms（1280×720），fast CUDA kernel 加速有效
- 在分布外放大倍数（如 ×6 时间 ×6 空间）上也保持稳定性能

## 亮点与洞察
- **去 RAFT 化**是重要的工程价值——消除外部依赖简化部署流水线，降低推理延迟
- **B 样条+傅里叶互补设计**非常优雅：一个管时间平滑，一个管空间细节
- 位置编码在 C-STVSR 中无效的发现值得关注，暗示不同视觉任务对位置信息的需求不同
- 可迁移到其他需要时空连续表示的任务（如视频生成、动态 NeRF）

## 局限与展望
- 超大运动场景（×12 时间放大）PSNR 显著下降，B 样条的局部性可能限制了大位移建模
- 仅在固定和连续放大两种设置上验证，未在感知质量指标（LPIPS）上全面评估
- 未探索与扩散模型的结合——扩散模型在超分辨率中越来越强

## 相关工作与启发
- **vs MoTIF**: MoTIF 需要 RAFT 光流，BF-STVSR 完全摆脱该依赖且性能更好
- **vs VideoINR**: VideoINR 用 MLP 隐式表示，BF-STVSR 的 B 样条+傅里叶显式基更高效

## 评分
- 新颖性: ⭐⭐⭐⭐ B 样条+傅里叶组合设计新颖，去 RAFT 化有实际意义
- 实验充分度: ⭐⭐⭐⭐ 多数据集多设置+消融+感知指标
- 写作质量: ⭐⭐⭐⭐ 动机清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ 去 RAFT 化对视频超分辨率实际部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)
- [\[CVPR 2025\] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency](geometry-guided_online_3d_video_synthesis_with_multi-view_temporal_consistency.md)
- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)

<!-- RELATED:END -->
