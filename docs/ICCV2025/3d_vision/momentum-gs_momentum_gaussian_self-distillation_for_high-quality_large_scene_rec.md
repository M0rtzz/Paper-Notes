---
title: >-
  [论文解读] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] Momentum-GS 提出基于动量的自蒸馏机制来解决大规模场景3D高斯溅射中分块并行训练的一致性问题，通过动量教师高斯解码器提供全局引导并解耦分块数量与GPU数量的限制，在多个大规模场景数据集上取得SOTA，LPIPS较CityGaussian提升18.7%。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯溅射
  - 大规模场景重建
  - 动量自蒸馏
  - 混合表示
  - 分块并行训练
---

# Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2412.04887](https://arxiv.org/abs/2412.04887)  
**代码**: https://jixuan-fan.github.io/Momentum-GS_Page/  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 大规模场景重建, 动量自蒸馏, 混合表示, 分块并行训练

## 一句话总结

Momentum-GS 提出基于动量的自蒸馏机制来解决大规模场景3D高斯溅射中分块并行训练的一致性问题，通过动量教师高斯解码器提供全局引导并解耦分块数量与GPU数量的限制，在多个大规模场景数据集上取得SOTA，LPIPS较CityGaussian提升18.7%。

## 研究背景与动机

**领域现状**：3D高斯溅射（3DGS）在场景重建中表现出色，大规模场景重建通常采用分治策略（divide-and-conquer），将场景分成多个独立块并行训练。

**现有痛点**：
   - **内存和存储问题**：数百万高斯的显式表示导致巨大内存和存储开销。混合表示（implicit+explicit）如 Scaffold-GS 可以缓解，但引入新的问题。
   - **跨块不一致**：各块独立训练忽略了块间关系，导致块边界处出现明显的光照不连续和过渡不自然（如 CityGaussian 中可见的突变光照）。
   - **GPU数量限制**：混合表示使用共享的高斯解码器（MLP），并行训练时分块数量受限于可用GPU数量。

**核心矛盾**：独立训练各块→数据多样性不足→解码器质量差→重建精度下降；共享解码器并行训练→分块数受GPU限制→可扩展性差。

**本文目标** 如何在大规模场景混合表示分块训练中，既保证跨块一致性，又不受GPU数量限制？

**切入角度**：借鉴自监督学习中的动量教师模型思想，用动量更新的教师解码器为各块提供稳定的全局一致性引导。

**核心 idea**：用动量更新的教师高斯解码器对学生解码器进行自蒸馏，解耦分块数与 GPU 数的约束，同时通过重建质量引导的块权重实现全局一致性优化。

## 方法详解

### 整体框架

输入：大规模场景的多视角图像 + SfM稀疏点云。输出：高质量3D高斯场景表示。

Pipeline：将大场景分为 $n$ 块，每次从中采样 $k$ 块分配到 $k$ 个 GPU 上并行训练，所有块共享一个学生高斯解码器 $D_s$，同时维护一个动量更新的教师解码器 $D_t$ 提供一致性引导。

### 关键设计

1. **场景动量自蒸馏（Scene Momentum Self-Distillation）**

    - 功能：解耦分块数量与GPU数量的约束，同时增强跨块一致性
    - 核心思路：维护教师高斯解码器 $D_t$（参数 $\theta_t$）和学生高斯解码器 $D_s$（参数 $\theta_s$）。教师参数通过动量更新：$\theta_t \leftarrow m \cdot \theta_t + (1-m) \cdot \theta_s$，其中动量系数 $m=0.9$。教师解码器不参与梯度反传，而是提供稳定的全局参考，通过一致性损失引导学生解码器：$\mathcal{L}_{\text{consistency}} = \|D_m(f_b, v_b; \theta_t) - D_o(f_b, v_b; \theta_s)\|_2$
    - 设计动机：顺序训练策略让每个 GPU 轮流处理不同块，突破了 GPU 数量对分块数的限制。动量教师平滑演化，为交错训练的各块提供一致的全局引导，避免因块间数据差异导致的解码器震荡。
    - 与之前方法的区别：CityGaussian 等方法要么独立训练各块（无法合并解码器），要么直接共享解码器并行训练（分块数=GPU数）。本文通过动量蒸馏实现了解耦。

2. **重建引导的块权重（Reconstruction-guided Block Weighting）**

    - 功能：动态调整各块的训练权重，优先关注重建质量差的块
    - 核心思路：用动量平滑的 PSNR 和 SSIM 追踪各块重建质量，计算每块与最佳块的偏差 $\delta_p, \delta_s$，然后用类高斯分布赋予权重：$w_i = 2 - \exp\left(\frac{\delta_p^2 + \lambda \cdot \delta_s^2}{-2\sigma^2}\right)$
    - 设计动机：由于场景分块不均匀，有些块天然比其他块难重建。自适应权重让解码器优先关注弱块，避免收敛到局部最优，提升全局一致性。

### 损失函数 / 训练策略

总损失函数：$\mathcal{L} = \mathcal{L}_1 + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{consistency}}\mathcal{L}_{\text{consistency}}$

每次前向传播，每个块随机选择一个视角，用共享解码器预测高斯参数并渲染图像，与真值比较计算重建损失。各块梯度累积到共享解码器中。

## 实验关键数据

### 主实验（5个大规模场景）

| 方法 | Building PSNR↑ | Rubble PSNR↑ | Residence PSNR↑ | Sci-Art PSNR↑ | Rubble LPIPS↓ |
|------|----------------|--------------|-----------------|---------------|---------------|
| 3D-GS | 22.53 | 25.51 | 22.36 | 24.13 | 0.316 |
| VastGaussian | 21.80 | 25.20 | 21.01 | 22.64 | 0.264 |
| CityGaussian | 22.70 | 26.45 | 23.35 | 24.49 | 0.232 |
| DOGS | 22.73 | 25.78 | 21.94 | 24.42 | 0.257 |
| **Momentum-GS** | **23.65** | **26.66** | **23.37** | **25.06** | **0.200** |

MatrixCity 结果：Momentum-GS PSNR 29.11 / SSIM 0.881 / LPIPS 0.180，全面超越所有基线。

渲染效率：**59.91 FPS**，内存仅 **4.62 GB**（最佳），显著优于 CityGaussian（26.10 FPS, 14.68 GB）。

### 消融实验

| 训练策略 | #Block | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------|--------|-------|-------|--------|
| (a) baseline 单块 | 1 | 22.25 | 0.742 | 0.272 |
| (b) 并行训练 | 4 | 23.10 | 0.790 | 0.221 |
| (c) 独立训练 | 4 | 22.85 | 0.781 | 0.229 |
| (d) 独立训练 | 8 | 23.23 | 0.796 | 0.211 |
| (e) + 动量自蒸馏 | 8 | 23.56 | 0.806 | 0.205 |
| (f) Full (+ 块权重) | 8 | **23.65** | **0.813** | **0.194** |

### 关键发现

- 并行训练(b)优于独立训练(c)相同分块数：共享解码器带来的数据多样性是关键
- 动量自蒸馏(e)在8块上显著优于独立训练8块(d)：LPIPS 从 0.211 降至 0.205
- 块权重(f)进一步提升一致性：LPIPS 从 0.205 降至 0.194
- 块权重中同时使用 PSNR + SSIM 效果最好（vs 仅用 PSNR 或仅用 SSIM）

## 亮点与洞察

- **动量蒸馏解耦硬件约束**：将分块数与 GPU 数解耦是一个非常实用的创新，让大规模场景重建不再受硬件数量瓶颈限制。这种思路可以迁移到任何需要分布式分块训练但要求全局一致性的任务。
- **重建质量引导的自适应权重**：用类高斯分布根据重建偏差动态加权各块，既简单又有效。这个 trick 可以直接用于其他分块训练范式。
- **效率+质量双赢**：不仅重建质量最好，渲染 FPS 最高（59.91）且内存最低（4.62GB），说明混合表示 + 自蒸馏的方案在效率上也有优势。

## 局限与展望

- 动量系数 $m=0.9$ 是固定的，可能需要根据场景复杂度自适应调整
- 目前块权重仅使用 PSNR/SSIM 作为质量指标，感知级指标（如 LPIPS）可能更合适
- 场景分块策略沿用 CityGaussian，本文未探索更优的分块方式
- 未在室内场景或动态场景上验证

## 相关工作与启发

- **vs CityGaussian**：CityGaussian 采用分治策略但各块独立训练，缺乏跨块交互，导致边界不一致。Momentum-GS 通过动量蒸馏和块权重显著改善了一致性，LPIPS 提升 18.7%。
- **vs VastGaussian**：VastGaussian 也用分治策略，但同样缺少跨块一致性机制。
- **vs DOGS**：DOGS 用 ADMM 进行分布式训练加速，但不关注优化高斯表示本身。Momentum-GS 通过混合表示+自蒸馏同时解决效率和质量问题。

## 评分

- 新颖性: ⭐⭐⭐⭐ 动量蒸馏用于3DGS分块训练的想法新颖且巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 5个大规模场景+多个基线+详细消融，非常充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，Figure 2的对比图很生动
- 价值: ⭐⭐⭐⭐ 对大规模场景重建有很强的实际应用价值

<!-- RELATED:START -->

## 相关论文

- [S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [HRAvatar: High-Quality and Relightable Gaussian Head Avatar](../../CVPR2025/3d_vision/hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](../../ECCV2024/3d_vision/gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)

<!-- RELATED:END -->
