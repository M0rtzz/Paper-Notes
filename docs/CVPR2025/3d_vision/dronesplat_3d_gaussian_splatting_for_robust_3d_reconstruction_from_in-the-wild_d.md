---
title: >-
  [论文解读] DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery
description: >-
  [CVPR 2025][3D视觉][无人机3D重建] DroneSplat 提出了一个面向野外无人机影像的鲁棒 3DGS 框架，通过自适应局部-全局掩膜策略消除动态干扰物，结合基于多视图立体的几何感知点采样和体素引导优化策略解决有限视角下的重建质量问题，并提供了 24 个场景的无人机 3D 重建数据集。
tags:
  - CVPR 2025
  - 3D视觉
  - 无人机3D重建
  - 3D高斯泼溅
  - 动态干扰物消除
  - 稀疏视角重建
  - 自适应掩膜
---

# DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery

**会议**: CVPR 2025  
**arXiv**: [2503.16964](https://arxiv.org/abs/2503.16964)  
**代码**: https://bityia.github.io/DroneSplat/ (有)  
**领域**: 3D视觉  
**关键词**: 无人机3D重建, 3D高斯泼溅, 动态干扰物消除, 稀疏视角重建, 自适应掩膜

## 一句话总结
DroneSplat 提出了一个面向野外无人机影像的鲁棒 3DGS 框架，通过自适应局部-全局掩膜策略消除动态干扰物，结合基于多视图立体的几何感知点采样和体素引导优化策略解决有限视角下的重建质量问题，并提供了 24 个场景的无人机 3D 重建数据集。

## 研究背景与动机

1. **领域现状**：无人机凭借出色的机动性已成为野外场景重建的重要工具。NeRF 和 3DGS 等辐射场方法在 3D 表示和新视角合成方面展现了卓越潜力。
2. **现有痛点**：将辐射场方法应用于野外无人机影像面临两大挑战——（a）**场景动态性**：无人机拍摄的野外图像常包含移动物体（汽车、行人），违反多视角一致性假设；（b）**视角稀疏性**：单次飞行中某些区域的视角覆盖有限，导致辐射场过拟合输入视角。
3. **核心矛盾**：现有动态干扰物消除方法要么依赖预定义类别（无法区分同类中静/动物体），要么使用硬阈值（无法适应不同场景和训练阶段的变化）；稀疏视角方法引入几何先验但缺乏配套的 3DGS 优化策略。
4. **本文目标** 在野外无人机影像中同时处理动态干扰物识别消除和有限视角下的高质量几何重建。
5. **切入角度**：作者观察到（a）动态物体的残差在不同训练阶段变化很大，固定阈值不合理；（b）DUSt3R 预测的稠密点云虽提供丰富先验，但需要配套的体素级优化约束才能充分利用。
6. **核心 idea**：结合自适应局部-全局掩膜实现鲁棒的动态干扰物消除，同时通过几何感知点采样和体素引导优化充分利用多视图立体先验实现稀疏视角下的高质量重建。

## 方法详解

### 整体框架
给定一组有位姿的无人机图像，DroneSplat 的处理流程为：（1）使用 DUSt3R 预测稠密点云并通过几何感知采样获取初始化点；（2）在 3DGS 训练过程中，通过自适应局部-全局掩膜（ALGM）识别并消除动态干扰物；（3）使用体素引导优化策略约束高斯优化过程，最终实现鲁棒的静态场景重建。

### 关键设计

1. **自适应局部掩膜（Adaptive Local Masking）**:

    - 功能：基于实时残差和统计方法自适应调整阈值，在每帧中识别动态干扰物。
    - 核心思路：首先用 SAM v2 对每张图像进行无类别分割得到物体级 mask，然后计算归一化残差的物体级均值 $\mathcal{R}_i^j(t)$。自适应阈值 $\mathcal{T}_i^L(t) = \mathbb{E}[R_i](t) + \text{Var}[R_i](t)(1 + \lambda_L \frac{T_{max}-t}{T_{max}})$ 基于当前帧的残差均值和方差动态调整，并在训练早期放松上限以适应不同物体收敛速度的差异。残差超过阈值的物体被标记为动态干扰物。
    - 设计动机：固定阈值在不同场景和训练阶段表现不同——过高会漏检大动态物体，过低会误检静态物体。自适应阈值根据实际残差分布自动调整，无需手动调参。

2. **互补全局掩膜（Complement Global Masking）**:

    - 功能：跨帧追踪动态干扰物，处理在某些帧中暂时静止但整体为动态的物体（如红灯前等候的车辆）。
    - 核心思路：设置更高的全局阈值 $\mathcal{T}_i^G = \mathbb{E}[R_i](t) + \lambda_G \text{Var}[R_i](t)$（$\lambda_G > 1 + \lambda_L$），超过阈值的物体标记为追踪候选。从候选 mask 选取 5 个 prompt 点输入 SAM v2 的视频分割功能，在所有帧中追踪该物体。全局掩膜集合随训练迭代累积：$\mathcal{M}_i^G(t) = \mathcal{M}_i^G(t-1) \cup \hat{m}_i^j$。最终掩膜为局部和全局掩膜的并集。
    - 设计动机：局部掩膜只能识别当前帧中残差高的物体，无法处理在某些帧中暂时静止的动态物体。全局掩膜通过跨帧追踪补充了这一盲区，两者协同达到全面的动态物体消除。

3. **体素引导的高斯优化（Voxel-guided Gaussian Splatting）**:

    - 功能：利用多视图立体先验约束 3DGS 优化，解决有限视角下的过拟合问题。
    - 核心思路：（a）**几何感知点采样**：使用 DUSt3R 预测稠密点云，提取 FPFH 描述子衡量几何特征，结合置信度计算综合得分 $\text{Score}(p) = \text{Conf}(p) \cdot \tilde{\text{FPFH}}(p)$，在自适应划分的体素中保留 top-k 高分点初始化高斯。（b）**体素引导优化**：对每个体素内的高斯施加边界约束（中心和缩放不超过 $\tau$ 倍体素长度），超出边界的高斯梯度指数衰减；若梯度累积达到阈值且指向空体素，则分裂/克隆到该空体素；低质量体素（高斯数不足或平均不透明度过低）被移除。
    - 设计动机：InstantSplat 虽利用 DUSt3R 初始化高斯，但后续优化无约束，导致性能不佳。体素引导策略将几何先验的约束从初始化延伸到整个优化过程，有效防止高斯偏离合理位置。

### 损失函数 / 训练策略
- 使用标准 3DGS 损失（L1 + D-SSIM），通过掩膜排除动态区域的损失贡献
- 自适应掩膜从 500 次迭代后启动，$\lambda_L = 0.4$，$\lambda_G = 2.8$
- 总训练迭代 7000 次，在 NVIDIA A100 上运行
- DUSt3R 使用 512 分辨率，体素参数 $N = 80$，$k = 3$

## 实验关键数据

### 主实验

| 数据集/场景 | 指标 | DroneSplat | 3DGS | WildGaussians | 最佳 baseline |
|------------|------|-----------|------|---------------|-------------|
| DroneSplat (Low Dynamic) | PSNR↑ | **24.56** | 22.43 | 22.41 | 23.29 (GS-W) |
| DroneSplat (Medium Dynamic) | PSNR↑ | **17.89** | 17.04 | 16.96 | 17.44 (GS-W) |
| DroneSplat (High Dynamic) | PSNR↑ | **19.51** | 17.11 | 17.15 | 17.09 (GS-W) |
| On-the-go 数据集 | PSNR↑ | 有竞争力 | baseline | 次优 | NeRF-HuGS |

### 消融实验

| 配置 | 说明 |
|------|------|
| DroneSplat (full) | 完整框架，最优性能 |
| Ours (COLMAP) | 使用 COLMAP 初始化替代 DUSt3R，性能接近但略低 |
| w/o Global Masking | 去掉全局掩膜，暂时静止的动态物体无法消除 |
| w/o Voxel-guided | 去掉体素引导，有限视角下重建质量显著下降 |

### 关键发现
- **动态等级越高，DroneSplat 优势越大**：在 High Dynamic 场景中，DroneSplat 相对 3DGS 的 PSNR 提升达 2.4 dB，说明 ALGM 策略在复杂动态场景中尤为有效。
- **局部+全局掩膜协同效果显著**：全局掩膜能够捕捉在某些帧中暂时静止但整体动态的物体（如等红灯的车），单独使用局部掩膜会遗漏这类干扰物。
- **体素引导对稀疏视角至关重要**：在仅 6 个输入视角的场景中，体素引导优化相比 vanilla 3DGS 提供了更稳定的几何约束。
- **自制数据集填补空白**：提供的 24 个无人机场景数据集涵盖了不同动态等级和静态场景，为该方向提供了标准评估平台。

## 亮点与洞察
- **自适应阈值设计精巧**：局部阈值随训练进程自动收紧（早期宽松允许不同收敛速度、后期严格精确识别），无需手动调参——这种训练感知的阈值策略可迁移到任何需要异常检测的辐射场任务。
- **局部-全局掩膜协同机制**：局部掩膜处理当前帧的显著动态物体，全局掩膜通过 SAM v2 视频追踪处理跨帧一致的动态物体——这种分层检测思路很适合处理复杂的非静态场景。
- **体素引导优化将先验约束从初始化延伸到全过程**：解决了 InstantSplat "初始化好但优化散掉"的问题，空体素的扩展机制也很巧妙。

## 局限与展望
- **SAM v2 的计算开销**：全局掩膜追踪依赖 SAM v2 的视频分割，对大规模场景（数千张图像）的效率可能成为瓶颈。
- **体素大小的自适应性有限**：当前体素大小由场景最短边等分确定，对高度不规则的场景可能不够灵活。
- **未处理光照变化**：野外无人机影像常有光照变化，当前方法未考虑曝光不一致问题。
- **仅针对 3DGS**：方法是否能扩展到 2DGS 或 Mip-Splatting 等变体未验证。

## 相关工作与启发
- **vs RobustNeRF**: RobustNeRF 使用固定阈值过滤高残差像素，在训练全程不调整。DroneSplat 的自适应阈值根据残差分布动态变化，更鲁棒。
- **vs NeRF-HuGS**: NeRF-HuGS 需要手动为每个场景调参，且阈值训练中不变。DroneSplat 完全自动化，阈值随训练自适应。
- **vs WildGaussians**: WildGaussians 用 DINOv2 特征预测像素级不确定性，但低分辨率特征导致边缘模糊和小目标遗漏。DroneSplat 通过物体级掩膜解决了这个问题。
- **vs InstantSplat**: InstantSplat 用 DUSt3R 初始化但不约束后续优化，DroneSplat 的体素引导策略补充了这一缺失。

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应局部-全局掩膜和体素引导优化都是针对具体问题的精巧设计
- 实验充分度: ⭐⭐⭐⭐ 自制数据集+两个公开数据集，消除和稀疏视角两个任务都有对比
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法阐述逻辑性强，图示质量高
- 价值: ⭐⭐⭐⭐ 针对无人机实际应用场景的实用框架，提供的数据集有价值

<!-- RELATED:START -->

## 相关论文

- [3D-GSW: 3D Gaussian Splatting for Robust Watermarking](3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)
- [GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)

<!-- RELATED:END -->
