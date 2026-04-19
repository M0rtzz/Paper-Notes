---
title: >-
  [论文解读] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 首次针对3D高斯泼溅(3DGS)提出密度引导的投毒攻击方法，通过在低密度区域注入幻觉物体的高斯点并引入自适应噪声破坏多视角一致性，实现从目标视角清晰可见而不干扰其余视角的隐蔽攻击。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D Gaussian Splatting
  - 数据投毒攻击
  - 对抗安全
  - 核密度估计
  - 多视角一致性
---

# StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions

**会议**: ICCV 2025  
**arXiv**: [2510.02314](https://arxiv.org/abs/2510.02314)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 数据投毒攻击, 对抗安全, 核密度估计, 多视角一致性

## 一句话总结

首次针对3D高斯泼溅(3DGS)提出密度引导的投毒攻击方法，通过在低密度区域注入幻觉物体的高斯点并引入自适应噪声破坏多视角一致性，实现从目标视角清晰可见而不干扰其余视角的隐蔽攻击。

## 研究背景与动机

随着3DGS在自动驾驶、AR/VR等安全关键应用中的广泛部署，其安全漏洞值得关注。此前IPA-NeRF针对NeRF实现了视角相关的幻觉注入，但将其迁移到3DGS上效果很差。作者深入分析发现原因有二：

**多视角一致性天然防御**：3DGS使用显式点云表示，内在的多视角一致性会将不一致的幻觉内容视为噪声并在训练中消除

**显式几何约束**：直接将幻觉物体反投影到高斯点云中面临深度确定困难，可能被已有几何遮挡

简单将IPA-NeRF适配到3DGS（IPA-Splat）或直接在训练图像上叠加幻觉内容都无法有效攻击，说明需要专门针对3DGS特性设计攻击策略。

## 方法详解

### 整体框架

方法由两个互补策略组成：(1) 密度引导的点云攻击，在低密度区域放置幻觉物体的高斯点；(2) 视角一致性扰乱攻击，通过自适应噪声调度破坏3DGS的多视角一致性。

攻击目标形式化为优化问题：
$$\min_{\tilde{G}} \| \tilde{I}_{\text{ILL}} - I_{\text{ILL}} \|_2^2 + \sum_{v_k \neq v_p} \| \mathfrak{R}(\tilde{G}, v_k) - \mathfrak{R}(G, v_k) \|_2^2$$
即最大化幻觉在目标视角的可见性，同时最小化对无辜视角的影响。

### 关键设计1：密度引导的点云攻击

**场景空间分析**：首先对已重建的高斯点云 $G$ 计算轴对齐包围盒(AABB)并体素化为均匀网格 $\mathcal{S}$。利用体渲染估计每个高斯点的不透明度 $\alpha(g)$，汇总得到各体素的密度 $\rho(s) = \sum_{g \in s} \alpha(g)$。

**连续密度估计**：使用高斯核密度估计(KDE)将离散密度平滑为连续函数：
$$f(x) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} K_h(x - c(s)) \cdot \rho(s)$$

**最优位置选择**：从目标视角的相机中心 $C$ 向幻觉物体每个像素射出射线，在射线上采样找到密度最小的点：
$$x_{\min} = \arg\min_{x \in C+t \cdot d, t \in [t_{\min}, t_{\max}]} f(x)$$
其中 $t_{\min}=0.3$ 避免近相机浮点伪影。在 $x_{\min}$ 处插入新高斯点，颜色取自幻觉物体对应像素。

核心思路：低密度区域要么不被其他视角覆盖，要么被已有几何遮挡，因此放置在这些区域的幻觉点对无辜视角干扰最小。

### 关键设计2：视角一致性扰乱攻击

对于视角高度重叠的场景，单靠点云注入仍不够。方法选择性地向无辜视角训练图像添加高斯噪声，保持目标视角图像干净：
$$I_k' = \mathbf{1}_{v_k = v_p} \cdot I_k + \mathbf{1}_{v_k \neq v_p} \cdot \text{clip}(I_k + \eta)$$

噪声强度 $\sigma_t$ 随训练迭代递减，三种衰减策略：
- **线性衰减**: $\sigma_{\text{linear}}(t) = \sigma_0 \cdot (1 - t/T)$
- **余弦衰减**: $\sigma_{\text{cosine}}(t) = \sigma_0 \cdot \cos(\pi t / 2T)$
- **平方根衰减**: $\sigma_{\text{sqrt}}(t) = \sigma_0 \cdot \sqrt{1 - t/T}$

早期强噪声破坏多视角一致性防御，后期减弱噪声保证无辜视角渲染质量。

### 损失函数

攻击的目标函数即双项优化：第一项驱动幻觉在目标视角的渲染接近预设外观，第二项约束其余视角不受影响。训练中3DGS本身使用标准的 $\ell_1 + \ell_{\text{ssim}}$ 重建损失。

## 实验关键数据

### 主实验：单视角攻击（Table 1，Mip-NeRF 360数据集）

| 方法 | V-illusory PSNR↑ | V-illusory SSIM↑ | V-test PSNR↑ |
|------|:-:|:-:|:-:|
| IPA-NeRF (Nerfacto) | 16.00 | 0.582 | 21.94 |
| IPA-NeRF (Instant-NGP) | 17.60 | 0.618 | 20.00 |
| IPA-Splat | 13.23 | 0.518 | 27.39 |
| **Ours** | **27.04** | **0.813** | **27.76** |

本文方法幻觉视角PSNR达到27.04（远超基线最高17.60），同时保持无辜视角质量（27.76 vs 原始29.45仅下降1.69）。

### 多视角攻击（Table 3，2/3/4个毒化视角）

同时攻击4个视角时仍达到26.95 PSNR（幻觉视角），无辜视角仅降至27.59。

### 消融实验（Table 6 - 攻击成分组合）

| 组合 | V-illusory PSNR | ASR |
|------|:-:|:-:|
| 仅毒化视角图像 | 13.22 | 0/7 |
| 图像 + 密度引导 | 26.01 | 6/7 |
| 仅一致性扰乱 | 13.31 | 0/7 |
| **全部组合** | **27.04** | **7/7** |

密度引导点云攻击和视角一致性扰乱缺一不可，完整组合实现100%攻击成功率。

### 关键发现

- KDE带宽 $h=7.5$ 取得最优平衡（Table 4）
- 初始噪声 $\sigma_0=100$ + 线性衰减效果最佳（Table 5）
- 攻击影响范围极小：相邻±5°内可见幻觉，超出范围几乎无影响（Figure 10）
- 基于KDE的评估协议证实场景密度与攻击难度负相关（Table 2）

## 亮点与洞察

1. **首创性工作**：第一篇针对3DGS的幻觉注入投毒攻击，揭示了显式3D表示的安全盲区
2. **KDE驱动的优雅设计**：利用核密度估计连接"几何可见性"和"攻击可行性"，将3D攻击问题转化为密度优化问题
3. **攻防对称性洞察**：3DGS的多视角一致性既是其优势（渲染质量保障），也是防御屏障；本文巧妙将噪声注入和低密度利用结合来突破这道防线
4. **系统化评估协议**：提出基于KDE的攻击难度评估方案，为未来安全研究提供可复现的基准

## 局限性

- 在视角高度重叠的场景中攻击效果下降（Hard模式PSNR降至17.53）
- 幻觉物体需要预先设计，无法自适应生成
- 防御方案未探讨——未来需要研究如何检测和抵御此类攻击

## 相关工作

- **IPA-NeRF**：NeRF上的投毒攻击先驱，通过双层优化注入视角相关幻觉，但不适用于3DGS
- **Poison-Splat**：针对3DGS的资源消耗攻击（增加内存开销），与本文关注视觉幻觉注入不同
- **GaussianMarker / 3D-GSW**：3DGS水印方法，嵌入二进制信息，与投毒攻击形成安全研究的两面
- **FGSM / PGD**：经典对抗攻击方法，本文将对抗概念拓展到3D场景表示领域

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首次攻击3DGS，问题定义新颖，KDE引导策略原创
- **技术深度**: ⭐⭐⭐⭐ — 方法设计精巧但各模块相对独立
- **实验充分性**: ⭐⭐⭐⭐⭐ — 三数据集、多视角、难度分级、完整消融
- **实用价值**: ⭐⭐⭐⭐ — 揭示安全隐患，对防御研究有重要启发
- **总体推荐**: ⭐⭐⭐⭐☆ — 开拓性安全研究，攻击场景的实际威胁需进一步论证

<!-- RELATED:START -->

## 相关论文

- [Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)
- [PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)
- [GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

<!-- RELATED:END -->
