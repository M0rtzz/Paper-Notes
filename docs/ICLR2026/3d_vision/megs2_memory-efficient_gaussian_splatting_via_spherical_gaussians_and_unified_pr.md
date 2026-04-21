---
title: >-
  [论文解读] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning
description: >-
  [ICLR 2026][3D视觉][3D Gaussian Splatting] 提出MEGS2——从渲染VRAM角度出发压缩3DGS：用可裁剪的任意方向球面高斯(SG)完全替代球谐函数(SH)降低每个primitive的参数量 + 统一软剪枝框架将primitive数量和lobe数量的裁剪建模为单一内存约束优化问题 -> 实现8x静态VRAM压缩和6x渲染VRAM压缩，同时保持渲染质量，首次让3DGS在移动端实时运行。
tags:
  - ICLR 2026
  - 3D视觉
  - 3D Gaussian Splatting
  - 内存压缩
  - 球谐函数替代
  - Spherical Gaussians
  - 统一剪枝
---

# MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning

**会议**: ICLR 2026  
**arXiv**: [2509.07021](https://arxiv.org/abs/2509.07021)  
**代码**: 待发布  
**领域**: 3D视觉/渲染压缩  
**关键词**: 3D Gaussian Splatting, 内存压缩, 球谐函数替代, Spherical Gaussians, 统一剪枝

## 一句话总结
提出MEGS2——从渲染VRAM角度出发压缩3DGS：用可裁剪的任意方向球面高斯(SG)完全替代球谐函数(SH)降低每个primitive的参数量 + 统一软剪枝框架将primitive数量和lobe数量的裁剪建模为单一内存约束优化问题 -> 实现8x静态VRAM压缩和6x渲染VRAM压缩，同时保持渲染质量，首次让3DGS在移动端实时运行。

## 研究背景与动机

**领域现状**：3DGS已成为主流新视角合成技术，但高内存消耗严重限制了其在边缘设备上的部署。大量压缩方法被提出，但绝大多数只关注存储压缩(文件大小)而忽视了渲染内存压缩(VRAM)。

**现有痛点**：(1) 基于神经压缩/VQ/hash grid的方法(CompactGaussian/EAGLES/HAC++)虽然存储压缩率高，但渲染前必须解压全部参数，VRAM甚至超过原始3DGS；(2) primitive剪枝方法(GaussianSpa/Mini-Splatting)能减少VRAM但压缩率有限——过度剪枝会严重损害质量；(3) 球谐函数(SH)作为颜色表示参数效率低，高阶系数多但利用率差。

**核心矛盾**：渲染VRAM = primitive数量 x 每个primitive的参数量。现有方法只优化其中一个因素。需要同时减少两个因素才能突破VRAM瓶颈。

**切入角度**：SH是全局基函数，需要大量高阶系数才能表示局部高频细节(锐利高光)。球面高斯(SG)是局部基函数，用少量lobe就能高效建模视角依赖效果，且lobe数量可灵活调整——天然适合剪枝。

**核心 idea**：用可剪枝的SG替代SH降低每个primitive的参数成本，再通过统一约束优化同时裁剪primitive数量和lobe数量，实现VRAM最优分配。

## 方法详解

### 整体框架
输入：3DGS场景。输出：内存高效的3DGS表示(静态+渲染VRAM大幅降低)。

三个核心组件：(A) SG替代SH作为颜色表示 (B) 统一软剪枝框架 (C) 后处理(移除+颜色补偿+微调)

### 关键设计

1. **任意方向可剪枝球面高斯(SG)替代SH**：

    - 功能：用SG完全替代SH作为视角依赖颜色的表示
    - 核心思路：每个primitive的颜色 $c(\mathbf{v}) = c_0 + \sum_{i=1}^n G(\mathbf{v}; \mu_i, s_i, a_i)$，其中 $c_0$ 是漫反射分量，每个SG lobe由方向轴 $\mu_i$、锐度 $s_i$、RGB幅度 $a_i$ 定义。关键是lobe方向不约束为正交——任意方向提供更高自由度
    - 设计动机：3阶SH需48个参数(16系数x3通道)，3-lobe SG只需约一半参数且能更好捕捉局部高频细节。固定正交轴的SG(SG-Splatting)会导致0.6dB PSNR下降，任意方向SG避免了这个问题。SG的lobe数量可变——natural fit for pruning

2. **统一软剪枝框架(ADMM-inspired)**：

    - 功能：将primitive数量裁剪和每个primitive的lobe数量裁剪统一为单一内存约束优化问题
    - 核心思路：优化目标 $\min \mathcal{L}(\mathbf{o}, \mathbf{s}, \Theta)$，约束 $\rho_o \|\mathbf{o}\|_0 + \rho_s \|\mathbf{s}\|_0 \leq \kappa$，其中 $\rho_o=11$(每个primitive的基础参数), $\rho_s=7$(每个SG lobe的参数), $\kappa$ 是总参数预算。由于L0范数不可微，引入ADMM代理变量分解为可解的子问题：梯度步+近端投影步+对偶更新
    - 设计动机：顺序剪枝(先减primitive再减lobe)会陷入次优——因为两者的最优分配是耦合的。统一框架在总预算约束下自动找到primitive数量和lobe数量的最优权衡。实验证明统一优于顺序

3. **后处理：颜色补偿**：

    - 功能：移除低sharpness的lobe时补偿其对漫反射颜色的贡献
    - 核心思路：移除lobe $i$ 后，计算补偿项 $\Delta c_0 = a_i \cdot \frac{1 - e^{-2s_i}}{2s_i}$，更新漫反射颜色 $c_0' = c_0 + \Delta c_0$。这是通过最小化球面上颜色差异的积分推导出的解析解
    - 设计动机：直接移除低shaprness lobe会丢失其对平均颜色的贡献，导致整体色偏。解析补偿几乎无额外开销地恢复能量

### 损失函数 / 训练策略
- 基于3DGS标准训练流程(Kerbl et al., 2023)
- ADMM优化：交替执行梯度步(更新渲染loss) + 近端投影步(enforcing sparsity) + 对偶变量更新
- 后处理：移除near-zero opacity的primitive和near-zero sharpness的lobe -> 颜色补偿 -> 少量微调恢复质量

## 实验关键数据

### 主实验 (Mip-NeRF360)

| 方法 | PSNR | SSIM | LPIPS | 静态VRAM(MB) | 渲染VRAM(MB) |
|------|------|------|-------|-------------|-------------|
| 3DGS | 27.48 | 0.813 | 0.217 | 648 | 1717 |
| GaussianSpa | 27.56 | 0.824 | 0.215 | 115 | 448 |
| **MEGS2 (HQ)** | **27.54** | **0.824** | **0.209** | **55** | **265** |
| MEGS2 (LM) | 27.21 | 0.814 | 0.227 | 40 | 224 |

### 消融实验 (Mip-NeRF360)

| 配置 | PSNR | LPIPS | VRAM(MB) | 说明 |
|------|------|-------|---------|------|
| GaussianSpa + Reduced3DGS | 26.05 | 0.280 | 402 | naive组合严重掉质量 |
| GaussianSpa(SH->SG) | 27.01 | 0.230 | 339 | 简单替换不够 |
| soft->hard pruning | 27.23 | 0.228 | 288 | 硬剪枝不如软剪枝 |
| unified->sequential | 27.33 | 0.222 | 328 | 顺序不如统一 |
| w/o color comp. | 27.46 | 0.213 | 265 | 颜色补偿有帮助 |
| **Full model** | **27.54** | **0.209** | **265** | 所有组件协同最优 |

### 关键发现
- **VRAM压缩**: 相比3DGS 8x静态VRAM压缩(648->55MB)和6x渲染VRAM(1717->265MB)。相比SOTA的GaussianSpa还有2x静态和40%渲染VRAM降低
- **质量保持**: PSNR几乎无损(27.54 vs 27.56)，LPIPS甚至更好(0.209 vs 0.215)
- **SG优于SH**: SG能更好拟合局部高频信号(锐利反射/高光)，在Bicycle/Truck场景的镜面反射上明显优于SH
- **Lobe分布**: 多数primitive只需0-1个lobe(强漫反射)，少数需2-3个(镜面高光)，平均1.3-1.7个lobe/primitive

## 亮点与洞察
- **问题定义的精准性**：区分存储压缩和内存压缩是关键洞察。现有work大量关注前者但忽视后者，而后者才是边缘部署的真正瓶颈
- **SG替代SH的合理性**：场景中绝大多数表面是漫反射的(不需要lobe)，只有少量镜面/高光需要lobe -> SG的可变lobe数量完美匹配这个长尾分布
- **统一剪枝的ADMM求解**：把两个离散优化统一为一个连续优化通过ADMM分解，优雅且有理论支撑。这个框架可泛化到任何需要同时优化"实体数量"和"每实体复杂度"的场景
- **颜色补偿的解析解**：通过球面积分推导出closed-form解，无额外计算开销，简单有效

## 局限与展望
- 聚焦于静态VRAM压缩，动态VRAM(渲染器实现相关)的优化留给未来
- 在高度复杂的高光场景(如全镜面物体)上性能有待进一步验证
- 可以与神经压缩方法(如HAC++)组合，同时优化存储和VRAM
- SG lobe的最优初始化策略值得探索

## 相关工作与启发
- **vs GaussianSpa**: 只做primitive剪枝，每个primitive仍用全SH -> VRAM下限较高。MEGS2通过额外压缩每个primitive的参数突破了这个瓶颈
- **vs Reduced3DGS**: 也尝试剪SH系数，但SH的全局性使其不适合稀疏裁剪(去掉高阶会全局损失细节)。SG的局部性使lobe剪枝更安全
- **vs CompactGaussian/EAGLES**: 存储压缩率高但VRAM反而可能增加(需解压)。根本不解决渲染内存问题

## 评分
- 新颖性: ⭐⭐⭐⭐ SG完全替代SH + 统一剪枝框架，概念清晰创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集完整对比，消融详尽，WebGL实机验证
- 写作质量: ⭐⭐⭐⭐ VRAM分析清晰，问题分解明确
- 价值: ⭐⭐⭐⭐⭐ 首次系统解决3DGS的渲染内存瓶颈，直接推动边缘端部署

<!-- RELATED:START -->

## 相关论文

- [Learning Unified Representation of 3D Gaussian Splatting](learning_unified_representation_of_3d_gaussian_splatting.md)
- [MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](../../ICCV2025/3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](../../CVPR2026/3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](../../CVPR2026/3d_vision/prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)

<!-- RELATED:END -->
