---
title: >-
  [论文解读] HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity
description: >-
  [ICCV 2025][人体理解][神经表面重建] 提出 HiNeuS，一个统一的神经表面重建框架，通过 SDF 引导的可见性验证、平面共形正则化和渲染优先的 Eikonal 松弛三项创新，同时解决反射歧义、低纹理退化和细节保留三大核心挑战。
tags:
  - ICCV 2025
  - 人体理解
  - 神经表面重建
  - SDF
  - 反射处理
  - 低纹理区域
  - Eikonal 约束
---

# HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity

**会议**: ICCV 2025  
**arXiv**: [2506.23854](https://arxiv.org/abs/2506.23854)  
**代码**: 即将开源  
**领域**: 人体理解  
**关键词**: 神经表面重建, SDF, 反射处理, 低纹理区域, Eikonal 约束

## 一句话总结

提出 HiNeuS，一个统一的神经表面重建框架，通过 SDF 引导的可见性验证、平面共形正则化和渲染优先的 Eikonal 松弛三项创新，同时解决反射歧义、低纹理退化和细节保留三大核心挑战。

## 研究背景与动机

神经表面重建是 3D 计算机视觉的核心技术。基于 SDF 的方法（如 NeuS、VolSDF）通过将体积渲染与几何先验结合，在表面恢复精度上超越了纯密度场方法（如 NeRF）。然而，现有方法在三个关键场景下仍表现不佳，且这些问题通常被**孤立地处理**：

**多视角不一致性（反射歧义）**：强反射和间接光照违反了多视角光度一致性假设。镜面反射使得同一表面点在不同视角下颜色差异巨大，导致几何伪影。现有方法如 Ref-NeuS 处理镜面但在低纹理区域引入噪声。

**低纹理表面退化**：缺乏纹理的区域（如白墙、路面）视觉线索稀疏，导致过度正则化，侵蚀有效的几何结构。表面正则化方法虽然改善平面区域，但会过度平滑细节。

**细节-几何冲突**：传统 Eikonal 约束 $\|\nabla f(\mathbf{x})\|_2 = 1$ 要求均匀平滑，这与保留高频几何细节（薄结构、锐边）矛盾。Neuralangelo 等方法恢复了精细结构，但对视角相关效果不鲁棒。

这三个问题的根本矛盾在于：**外观约束（光度损失）和几何约束（Eikonal）之间需要动态平衡**，而现有方法要么只处理其中一个方面，要么通过分阶段优化来回避冲突。HiNeuS 的核心目标是实现一个统一框架，让外观-几何约束在训练过程中**协同演化**。

## 方法详解

### 整体框架

HiNeuS 基于 SDF 体积渲染框架（类同 NeuS/VolSDF），使用 SDF 网络 + 颜色网络 + 提议网络。在标准渲染损失 $\mathcal{L}_{rgb}$ 的基础上，引入三个创新模块形成统一损失：$\mathcal{L}_{total} = \mathcal{L}_{rgb} + \mathcal{L}_{planar} + \mathcal{L}_{eikonal}$。三个模块之间通过共享的 SDF 值、渲染误差和外观特征形成双向耦合。

### 关键设计

1. **SDF 引导的多视角一致性验证**：通过连续 SDF 评估计算可见性因子 $\mathbb{V}_j$，解决反射歧义。对于表面点 $\mathbf{x}_0^i$ 从视角 $j$ 观察时，沿射线采样 $K$ 个点，通过 SDF 值判断遮挡：

$$\mathbb{V}_j = \prod_{k=1}^{K} \sigma(\beta f(\mathbf{x}_k^{(j)}))$$

当 $f(\mathbf{x}) > 0$（自由空间）时 $\sigma(\beta f) \approx 1$，当 $f(\mathbf{x}) < 0$（被占据）时 $\sigma(\beta f) \approx 0$，整个射线上的乘积接近 1 仅当所有点都在自由空间。然后定义**歧义因子** $\lambda_{ambiguity}$ 为可见视角间的 Mahalanobis 颜色距离的加权平均，用于降低高歧义射线的损失权重。还引入了**自反射补偿**：对有间接反射的射线，建模为 $\mathbf{C}' = (1-\mathbb{S})\mathbf{C} + \mathbb{S}g(\mathbf{x}_r)$，其中 $\mathbb{S}$ 是反射概率，$g$ 是反射 MLP。设计动机：SDF 的连续性避免了 mesh 离散化伪影，且能检测薄结构后的遮挡。

2. **局部几何约束正则化（平面共形）**：对低纹理区域强制局部平面性。在表面点 $\mathbf{x}_0$ 沿射线采样邻近点 $\mathbf{x}_k$，强制 SDF 值满足局部线性关系：

$$\mathcal{L}_{planar} = \frac{1}{K}\sum_{k=1}^{K} \lambda_{pla}^k \left|\frac{f(\mathbf{x}_k)}{\|\mathbf{x}_k - \mathbf{x}_0\|} - \mathbf{n}_0^\top \frac{\mathbf{x}_k - \mathbf{x}_0}{\|\mathbf{x}_k - \mathbf{x}_0\|}\right|$$

关键在自适应权重 $\lambda_{pla}^k = \frac{\epsilon}{\|\mathbf{c}_{feat}(\mathbf{x}_k) - \mathbf{c}_{feat}(\mathbf{x}_0)\|_2 + \epsilon}$：**纹理变化大的区域权重低**（保留锐边），**纹理均匀的区域权重高**（强制平面性）。这通过外观特征自动区分了需要平滑和需要保留细节的位置。

3. **渲染优先的 Eikonal 松弛**：动态调整 Eikonal 约束的强度，遵循"渲染误差大的地方放松几何约束"原则：

$$\omega(\mathbf{x}) = \lambda_{pla}(\mathbf{x}) \cdot \exp(-\gamma\|\mathbf{C}(\mathbf{x}) - \hat{\mathbf{C}}(\mathbf{x})\|_2)$$

$$\mathcal{L}_{eikonal} = \frac{1}{|\mathcal{S}|}\sum_{\mathbf{x}\in\mathcal{S}} \omega(\mathbf{x})(\|\nabla f(\mathbf{x})\|_2 - 1)^2$$

三个核心性质：(i) 渲染误差大 → $\omega$ 小 → Eikonal 约束松弛 → 允许几何调整以改善渲染；(ii) 平面性高 → $\lambda_{pla}$ 大 → 低纹理区域仍保持正则化；(iii) 指数衰减提供渐进收敛。这将渲染误差作为"学习到的注意力"，自动解决了几何精度与细节保留的冲突。

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{rgb} + \mathcal{L}_{planar} + \mathcal{L}_{eikonal}$
- SDF 网络：8层 MLP（256通道），几何初始化；颜色网络：4层 MLP（128通道）
- 渐进式哈希编码：分辨率从 $32^3$ 到 $2048^3$
- Adam 优化器，学习率从 $10^{-2}$ 衰减到 $10^{-5}$，共 500k 迭代
- $\beta = 100$，$\gamma = 5.0$，渲染误差裁剪到 $\leq 0.2$，$\lambda_{pla}$ 从 0.1 退火到 1.0（100k 步）

## 实验关键数据

### 主实验

**NeRF-Synthetic（PSNR dB ↑）**

| 方法 | Chair | Lego | Mic | Ship | Avg. |
|------|-------|------|-----|------|------|
| NeuS | 27.95 | 29.85 | 29.89 | 25.46 | 27.44 |
| 3DGS | 35.83 | 35.78 | 35.36 | 30.80 | 32.68 |
| **HiNeuS** | **37.29** | **36.11** | **38.17** | **34.39** | **35.00** |

相比 3DGS 提升 2.32 dB，相比 NeuS 提升 7.56 dB。

**GlossySynthetic（Chamfer Distance mm ↓）**

| 方法 | Angel | Horse | Avg. |
|------|-------|-------|------|
| Ref-NeuS | 0.0041 | 0.0062 | 0.0048 |
| NeRO | 0.0034 | 0.0049 | 0.0042 |
| **HiNeuS** | **0.0032** | **0.0045** | **0.0038** |

相比 Ref-NeuS 降低 21.4%。

### 消融实验

**Mip-NeRF 360 数据集（各模块贡献）**

| 配置 | Avg PSNR ↑ | 影响 |
|------|-----------|------|
| 完整方法 | 29.50 | 基准 |
| 去除歧义因子 $\lambda_{ambiguity}$ | 28.08 | -1.42 dB，反射场景退化严重 |
| 使用 mesh 可见性（替换 SDF） | 28.72 | -0.78 dB，薄结构出现遮挡伪影 |
| 去除平面约束 $\mathcal{L}_{planar}$ | 29.05 | -0.45 dB，低纹理区域不稳定 |
| Eikonal 无自适应权重 $\omega$ | 28.20 | -1.30 dB，细节过度平滑 |

### 关键发现

- 三个模块之间存在**非线性协同效应**：单独去除歧义因子（-1.42dB）比去除平面约束（-0.45dB）影响大得多
- SDF 连续可见性验证相比 mesh 离散化方法在薄结构上优势明显（TBell：0.0033 vs 0.0036mm）
- 在 UrbanScene3D 真实城市场景中，HiNeuS 能正确忽略行驶车辆等动态物体，恢复街灯杆等精细结构
- 反射补偿项 $\mathbb{S}g(\mathbf{x}_r)$ 贡献了 GlossySynthetic 总提升的 37%
- 方法可推广到逆渲染任务（材质分解、视角一致重光照）

## 亮点与洞察

- **"让渲染误差指导几何约束"**的设计哲学很有洞察力——用可观测量（渲染质量）来自动调整不可观测量（几何正则化）的强度
- SDF 的连续性被充分利用——可见性、反射概率、自适应权重都基于 SDF 值计算，形成了统一的数学框架
- 平面约束的自适应权重巧妙地利用外观特征差异来区分纹理边界和低纹理区域
- 三个模块看似独立，但通过 SDF 和渲染误差的双向耦合形成了协同优化

## 局限与展望

- 500k 迭代的训练时间较长，效率不如近期基于 3D 高斯的方法
- 反射 MLP 的容量有限，可能无法处理复杂的多次反弹光照
- 未与 NeuS2、Instant-NeuS 等高效 NeuS 变体在效率上做详细对比
- 自反射补偿的 $t_{max} = 0.1$ 设置较为启发式，可能需要场景自适应
- 在极端低纹理场景（全白墙面）中平面假设可能不够，需要引入深度先验

## 相关工作与启发

- **NeuS/VolSDF** 是基础——SDF-to-density 映射和体积渲染框架
- **Ref-NeuS** 是反射处理的直接对比——参数化 BRDF 方法不够灵活
- **Neuralangelo** 是高保真重建的代表——但未处理反射
- 启发：自适应松弛策略可以推广到其他存在约束冲突的 3D 重建任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 三个模块各有新意，统一框架的协同设计尤其出色
- **实验充分度**: ⭐⭐⭐⭐⭐ 合成+真实、室内+室外、多个基准、详细消融
- **写作质量**: ⭐⭐⭐⭐ 数学推导严谨，物理直觉解释清晰
- **价值**: ⭐⭐⭐⭐ 对神经表面重建中的实际痛点提供了系统性解决方案

<!-- RELATED:START -->

## 相关论文

- [4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](../../CVPR2026/human_understanding/textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)
- [Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](../../NeurIPS2025/human_understanding/reflective_translation_improving_low-resource_machine_translation_via_structured.md)

<!-- RELATED:END -->
