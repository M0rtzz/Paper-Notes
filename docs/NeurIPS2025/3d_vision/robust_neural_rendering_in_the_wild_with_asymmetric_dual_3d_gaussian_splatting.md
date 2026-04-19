---
title: >-
  [论文解读] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][3DGS] AsymGS利用一个关键观察——野外训练数据引起的重建伪影具有随机性——提出非对称双3DGS框架，通过互补掩码策略和一致性约束抑制伪影，并引入Dynamic EMA Proxy实现高效训练，在多个野外数据集上显著超越现有方法。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3DGS
  - 野外场景重建
  - 双模型一致性
  - 瞬态干扰物
  - EMA代理
---

# Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2506.03538](https://arxiv.org/abs/2506.03538)  
**代码**: [GitHub](https://steveli88.github.io/AsymGS)  
**领域**: 3D视觉 / 神经渲染  
**关键词**: 3DGS, 野外场景重建, 双模型一致性, 瞬态干扰物, EMA代理

## 一句话总结

AsymGS利用一个关键观察——野外训练数据引起的重建伪影具有随机性——提出非对称双3DGS框架，通过互补掩码策略和一致性约束抑制伪影，并引入Dynamic EMA Proxy实现高效训练，在多个野外数据集上显著超越现有方法。

## 研究背景与动机

3D高斯溅射(3DGS)在理想条件下能实现高质量场景重建，但现实野外图像中普遍存在光照变化和瞬态干扰物（行人、车辆等），这些因素引入噪声监督信号，导致重建质量下降。

现有方法的核心问题：
- 逐图像外观嵌入仅通过光度损失弱监督，缺乏直接约束
- 离群值过滤依赖手工规则，稳定性和泛化性不足
- 单模型无法区分真实场景结构和数据噪声引起的伪影

**核心洞察**：野外数据引起的伪影是**随机**的——同一场景用不同训练顺序多次训练，真实结构保持一致，而伪影在不同运行间会发生变化。这意味着可以通过跨模型一致性来过滤伪影。

## 方法详解

### 整体框架

AsymGS维护两个并行训练的3DGS模型，通过一致性约束鼓励它们在可靠场景结构上收敛，同时用非对称掩码策略防止确认偏差(confirmation bias)。

### 关键设计

1. **双3DGS与互一致性约束**:

    - 维护两组高斯集合 $\mathbb{G}_1$ 和 $\mathbb{G}_2$，在每个训练迭代中分别从不同视图列表采样
    - 互一致性损失：要求两个模型从同一视角渲染的结果（使用view-independent渲染，不含外观变换）保持一致
    - 关键点：一致性约束在view-independent渲染 $\hat{\mathbf{I}}$ 上进行，而非view-dependent渲染 $\tilde{\mathbf{I}}$，因为前者反映场景固有外观
    - 采用渐进策略：先独立预热训练，再引入一致性约束

2. **非对称掩码策略**:

    - **Multi-Cue Adaptive Mask ($\mathbf{M}_h$)**：硬掩码，整合多种线索识别瞬态区域：
        - SAM分割出语义区域
        - COLMAP多视图立体匹配检测静态内容（具有足够匹配点的区域为静态）
        - 像素级残差（L1重建误差）和特征级残差（DINOv2余弦距离）联合识别干扰区域
    - **Self-Supervised Soft Mask ($\mathbf{M}_s$)**：可学习软掩码(0-1连续值)
        - 基于DINOv2特征的余弦相似度自监督学习
        - 初始化全1，训练中逐步精化，对模糊区域更敏感
    - 两个模型分别使用 $\mathbf{M}_h$ 和 $\mathbf{M}_s$，引入互补归纳偏置，防止收敛到相同错误

3. **Dynamic EMA Proxy**:

    - 用动态EMA副本替代第二个3DGS模型，大幅降低计算开销
    - **动态更新机制**：处理3DGS训练中的克隆、分裂、剪枝操作
        - 克隆：EMA属性一并克隆
        - 剪枝：EMA对应项同步移除
        - 分裂：位置和方差重新初始化，其余属性继承
    - **交替掩码策略**：单模型下交替使用 $\mathbf{M}_h$ 和 $\mathbf{M}_s$，保持训练多样性

### 损失函数 / 训练策略

总损失（GS-GS版本）：
$$\mathcal{L} = \mathcal{L}_{r1}^{\mathbf{M}_h} + \mathcal{L}_{r2}^{\mathbf{M}_s} + \lambda_m(\mathcal{L}_{m1} + \mathcal{L}_{m2}) + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$$

- 重建损失：DSSIM + L1，通过掩码过滤干扰区域
- 互一致性损失：仅使用L1（DSSIM会降低性能）
- 掩码损失：DINOv2特征余弦相似度的L1损失
- 外观建模：跟随WildGaussian，使用per-Gaussian和per-view嵌入+MLP预测仿射变换参数

## 实验关键数据

### 主实验

| 数据集 | 指标 | AsymGS (GS-GS) | AsymGS (EMA-GS) | HybridGS (之前SOTA) | 训练时间 |
|--------|------|-----------------|-----------------|---------------------|----------|
| NeRF On-the-go (High Occ.) | PSNR↑ | **24.34** | 24.12 | 23.05 | GS-GS: 0.28h |
| NeRF On-the-go (Med. Occ.) | PSNR↑ | **24.56** | 24.32 | 23.51 | EMA-GS: 0.18h |
| NeRF On-the-go (Low Occ.) | PSNR↑ | **21.91** | 21.77 | 21.42 | - |
| RobustNeRF (Yoda) | PSNR↑ | **37.18** | - | 35.32 | 0.31h |
| RobustNeRF (Crab) | PSNR↑ | **36.18** | - | 35.17 | - |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| 单3DGS (baseline) | ~19.0 | ~0.65 | ~0.34 | 无任何处理 |
| + 外观嵌入 (WildGaussian) | 23.03 | 0.771 | 0.172 | 外观建模有效 |
| + 双模型+一致性 | 提升 | 提升 | 降低 | 互一致性抑制伪影 |
| + 非对称掩码 | **24.34** | **0.825** | **0.150** | 互补掩码进一步优化 |

### 关键发现

- 非对称掩码策略对防止确认偏差至关重要，对称设计（两个模型用相同掩码）效果明显下降
- EMA-GS在训练时间仅增加约50%（vs单模型）的情况下接近GS-GS的效果，训练时间节省约1/3
- 一致性约束过早引入会影响收敛，渐进式策略是必要的
- 在view-independent渲染上施加一致性约束比view-dependent更有效

## 亮点与洞察

- **随机性观察驱动设计**：伪影的随机性这一观察简洁而深刻，直接启发了双模型一致性框架的设计
- **非对称训练避免确认偏差**：硬掩码确定但可能过于自信，软掩码灵活但可能不精确，互补使用是巧妙的设计
- **Dynamic EMA的3DGS适配**：标准EMA假设参数数量固定，而3DGS有克隆/分裂/剪枝操作，动态EMA机制是实用的工程创新
- **高效性**：EMA-GS在PhotoTourism上从之前SOTA的2.9h减少到同等级训练时间，同时质量更好

## 局限与展望

- 双模型版本(GS-GS)训练开销仍然较大，虽然EMA-GS显著缓解
- 多种线索融合的硬掩码依赖SAM和COLMAP等外部工具
- 未在动态场景（如自动驾驶连续帧）上验证

## 相关工作与启发

- **vs WildGaussian**: WildGaussian使用per-Gaussian外观嵌入处理光照变化，但未解决瞬态伪影；AsymGS在此基础上添加双模型一致性
- **vs HybridGS**: HybridGS用3DGS+2DGS分别建模静态和动态内容，但双表示设计复杂；AsymGS更原则性地通过统计一致性解决问题
- **vs SpotlessSplats**: SpotlessSplats基于阈值残差学习掩码，AsymGS的多线索自适应掩码更鲁棒

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 伪影随机性的观察和非对称双模型设计都很有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集全面评估，消融详细，训练效率也有对比
- 写作质量: ⭐⭐⭐⭐ 逻辑线清晰，从观察到方法到实现层层推进
- 价值: ⭐⭐⭐⭐ 为野外3DGS重建提供了有效且高效的解决方案，EMA-GS具有很强的实用性

<!-- RELATED:START -->

## 相关论文

- [DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery](../../CVPR2025/3d_vision/dronesplat_3d_gaussian_splatting_for_robust_3d_reconstruction_from_in-the-wild_d.md)
- [RPBG: Towards Robust Neural Point-based Graphics in the Wild](../../ECCV2024/3d_vision/rpbg_towards_robust_neural_point-based_graphics_in_the_wild.md)
- [LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)
- [Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)
- [Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](../../ICCV2025/3d_vision/robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)

<!-- RELATED:END -->
