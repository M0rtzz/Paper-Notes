---
title: >-
  [论文解读] MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference
description: >-
  [NeurIPS 2025][3D视觉][高斯splatting] 提出MaterialRefGS，通过多视角一致的材质推断约束和基于2DGS光线追踪的环境建模策略，实现反射表面的高保真新视角合成和精确光照分解。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 高斯splatting
  - 反射建模
  - 多视角一致性
  - PBR
  - 光照分解
---

# MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference

**会议**: NeurIPS 2025  
**arXiv**: [2510.11387](https://arxiv.org/abs/2510.11387)  
**代码**: [项目页面](https://wen-yuan-zhang.github.io/MaterialRefGS)  
**领域**: 3D视觉  
**关键词**: 高斯splatting, 反射建模, 多视角一致性, PBR, 光照分解

## 一句话总结

提出MaterialRefGS，通过多视角一致的材质推断约束和基于2DGS光线追踪的环境建模策略，实现反射表面的高保真新视角合成和精确光照分解。

## 研究背景与动机

在3D高斯splatting的新视角合成中，**反射表面**是一个核心难题。原始3DGS用球谐函数编码视角依赖的颜色，无法捕捉高频镜面反射。近期方法将反射相关的材质属性（金属度、粗糙度等）附加到高斯基元上，通过PBR延迟着色合成最终颜色。

**现有PBR方法的两大痛点**：

**材质推断的病态性**：所有材质参数仅通过光照传输后的光度损失优化，多种光照+材质组合可以解释同一像素——导致光照分解次优。更关键的是，**高斯表示的视角依赖性与学习视角无关材质属性的目标相矛盾**：同一物理属性在不同视角下通过alpha混合产生不一致的外观，BRDF无法从这种模糊观察中推断准确反射。

**间接光照缺失**：使用split-sum近似建模镜面光照时，如果入射方向被其他物体遮挡，从环境贴图查询到的辐射不正确——现有方法要么忽略遮挡要么用离线二值指示器粗糙处理。

**本文的核心insight**：多视角一致的材质推断是学习准确反射的关键。通过（1）约束不同视角的材质map保持几何一致，（2）利用多视角光度差异作为反射强度先验，（3）通过可微光线追踪建模间接光照，三管齐下解决上述问题。

## 方法详解

### 整体框架

以2DGS为基础表示，每个高斯附加材质属性：漫反射颜色 $c_d \in \mathbb{R}^3$、反照率 $a$、金属度 $m$、粗糙度 $r$。渲染过程采用延迟着色PBR：先将材质属性光栅化为屏幕空间的材质map，再用简化的Disney BRDF计算最终颜色 $c(\omega_o) = (1-m)c_d + L_s(\omega_o, a, m, r, n)$。漫反射由高斯直接预测，镜面反射由BRDF评估。

### 关键设计

1. **多视角材质一致性约束（Multi-view Material Consistency）**：对于在视角 $v_i$ 和 $v_j$ 中都可见的表面点 $p$，其在两个视角材质map上的投影应相同。具体实现：在参考视角 $v_i$ 采样 $7 \times 7$ 像素patch，通过单应矩阵 $H_{ij}$ warp到源视角 $v_j$：
    $H_{ij} = K_j(R_{ij} - \frac{T_{ij}n_i^T}{d_i})K_i^{-1}$
   然后在漫反射、粗糙度、金属度map上计算MSE损失：$\mathcal{L}_{mv} = \|\Psi_i[P(\pi_i(p))] - \Psi_j[P'(\pi_j(p))]\|_2$。**设计动机**：约束高斯在不同视角产生一致的材质输出，限制视角特定的过拟合，使BRDF能从一致的材质观察中推断全局物理反射效果。

2. **多视角反射强度先验（Reflection Strength Prior）**：高反射表面在不同视角下外观差异显著。基于这一观察，追踪相机轨迹上物体表面的光度变化来量化反射得分：

    - 对参考视角，采样 $3 \times 3$ patch warp到相邻 $M$ 个视角
    - 计算跨视角标准差作为反射得分 $\text{refscore}$（公式6）
    - 通过空间反射融合模块（ball query + top-K平均）聚合为多视角一致的反射强度先验 $w_{ref}$
    - 用 $w_{ref}$ 监督金属度map：高反射区域被鼓励拥有更高的金属度值
   
   损失为：当预测金属度 $\Psi^M(u,v) < M_0$ 时，$\mathcal{L}_{ref} = w_{ref} \cdot \Gamma(u,v) \cdot |M_0 - \Psi^M|$。使用SAM2分割前景为语义区域，按区域门控 $\Gamma$。

3. **基于2DGS光线追踪的环境建模**：将入射辐射分解为直接和间接分量：$L_i(\omega_i) = L_{\text{indirect}}(\omega_i) + (1-O(\omega_i)) \cdot L_{\text{direct}}(\omega_i)$。直接光从可学习环境cubemap查询。间接光通过BVH光线追踪获得——沿反射方向追踪光线，找到所有相交高斯，按深度排序后splatting计算间接颜色和遮挡概率。整个过程完全可微，高斯参与环境光照建模并被联合优化。相比Ref-Gaussian的离线二值可见性指标，这种方法更物理正确且支持端到端训练。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_c + \lambda_{n-d}\mathcal{L}_{n-d} + \lambda_n\mathcal{L}_n + \lambda_{mv}\mathcal{L}_{mv} + \lambda_{ref}\mathcal{L}_{ref}$$

- $\mathcal{L}_c = 0.8 \cdot \mathcal{L}_{rgb} + 0.2 \cdot \mathcal{L}_{D-SSIM}$：光度损失
- $\mathcal{L}_{n-d}$：2DGS的深度-法线一致性损失
- $\mathcal{L}_n = |1 - N^TN̂|$：法线先验损失（StableNormal提供监督）

**训练流程**（共30k iter）：
- 0-3k iter：仅训练2DGS + 法线先验
- 3k-10k iter：引入PBR和环境建模
- 10k+ iter：移除法线先验（避免不准确预测的偏差），引入多视角正则化

## 实验关键数据

### 主实验（新视角合成）

| 方法 | ShinyBlender PSNR↑ | ShinyBlender MAE↓° | GlossySynthetic PSNR↑ | Ref-Real PSNR↑ | MipNeRF360 LPIPS↓ |
|------|--------------------|--------------------|----------------------|----------------|-------------------|
| 3DGS | 30.37 | - | 26.01 | 23.85 | 0.214 |
| 3DGS-DR | 33.94 | 2.62 | 29.49 | 23.99 | 0.249 |
| Ref-Gaussian | 35.04 | 4.59 | 30.08 | 24.61 | 0.272 |
| EnvGS | 33.83 | 6.36 | 28.17 | 24.85 | 0.222 |
| **MaterialRefGS** | **35.57** | **2.04** | **30.83** | **25.04** | **0.181** |

在ShinyBlender上比3DGS-DR PSNR高1.6dB，法线MAE从2.62°降至2.04°；在复杂真实世界MipNeRF360上LPIPS比所有方法都低。

### 消融实验

| 配置 | ShinyBlender PSNR↑ | Ref-Real PSNR↑ | Ref-Real LPIPS↓ | 说明 |
|------|--------------------|---------|---------|----|
| w/o $\mathcal{L}_{mv}$, w/o $\mathcal{L}_{ref}$ | 34.87 | 24.24 | 0.260 | 无多视角约束 |
| w/ $\mathcal{L}_{mv}$, w/o $\mathcal{L}_{ref}$ | 35.21 | 24.47 | 0.242 | 仅一致性约束 |
| w/o $\mathcal{L}_n$ | 35.37 | 24.39 | 0.229 | 无法线先验 |
| w/o Environment | 34.69 | 24.76 | 0.199 | 无光线追踪环境建模 |
| **Full Model** | **35.57** | **25.04** | **0.185** | 全部组件 |

法线先验+材质一致性+环境建模三者对几何精度均有贡献（法线MAE表）：

| 配置 | MAE↓° | CD↓ |
|------|-------|-----|
| w/o $\mathcal{L}_n$, w/o Reg, w/o Env | 3.47 | 0.94 |
| w/o $\mathcal{L}_n$ | 2.59 | 0.68 |
| Full Model | 2.04 | 0.60 |

### 关键发现

- 多视角材质一致性约束是提升光照分解质量的关键：加入后材质map从视角间不连续变为均匀一致
- 反射强度先验通过光度变化量化反射性，为金属度提供了一种无需额外标签的自监督信号
- 环境建模的间接光照对有遮挡的互反射场景至关重要——去掉后反射区域明显模糊
- 在MipNeRF 360等复杂真实场景上，现有反射方法常性能退化，MaterialRefGS保持竞争力
- 法线先验在早期训练阶段引导几何初始化，但后期应移除以避免偏差——这一训练设计体现了对优化动态的理解

## 亮点与洞察

- **多视角一致性作为材质推断的正则化**的思路非常自然但此前被忽视——同一物理属性在不同视角下应一致，这是PBR的基本假设
- 用多视角光度变化量化反射强度并作为金属度先验，把常规观察转化为有效监督信号
- 将漫反射交给高斯、镜面反射交给BRDF的设计降低了优化难度，比全BRDF方法更适合复杂场景
- 可微光线追踪实现的间接光照建模，让高斯本身参与环境建模的联合优化

## 局限与展望

- 方法不等价于完整的逆渲染——不适合材质分解的定量评估和relighting
- 反射强度先验的SAM2分割和ball query增加了预处理复杂度
- 对简单漫反射场景引入的多视角约束可能带来不必要的计算开销
- 间接光照的光线追踪效率受高斯数量影响，超大场景可能成为瓶颈

## 相关工作与启发

- **2DGS**：本文基础表示，将3D高斯压扁为2D盘提供更好的表面法线
- **3DGS-DR / Ref-Gaussian**：先前的延迟着色PBR方法，但缺乏多视角一致性约束
- **EnvGS**：同样使用光线追踪建模环境，但未考虑多视角材质一致性
- 启发：在3DGS的材质/几何优化中引入跨视角约束是一个通用且有效的正则化思路

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多视角材质一致性约束和反射强度先验是重要创新点
- **实验充分度**: ⭐⭐⭐⭐⭐ 4个数据集全面评估（合成+真实），消融彻底，包括法线和几何精度
- **写作质量**: ⭐⭐⭐⭐ 方法动机阐述清晰，图示质量高
- **价值**: ⭐⭐⭐⭐ 对反射表面的3DGS渲染有实际提升，多视角一致性思路可推广

<!-- RELATED:START -->

## 相关论文

- [GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)
- [SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](../../ICCV2025/3d_vision/supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [PanSt3R: Multi-view Consistent Panoptic Segmentation](../../ICCV2025/3d_vision/panst3r_multi-view_consistent_panoptic_segmentation.md)
- [MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](../../ICCV2025/3d_vision/materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)

<!-- RELATED:END -->
