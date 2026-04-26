---
title: >-
  [论文解读] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields
description: >-
  [CVPR 2025][3D视觉][逆渲染] PBR-NeRF 在 NeILF++ 的基础上引入了两个基于物理的先验损失（能量守恒损失和 NDF 加权高光损失），有效约束了逆渲染中材质-光照的分解歧义，在不牺牲新视角合成质量的前提下实现了 SOTA 的材质估计。
tags:
  - CVPR 2025
  - 3D视觉
  - 逆渲染
  - 材质估计
  - 物理先验
  - NeRF
  - BRDF
---

# PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields

**会议**: CVPR 2025  
**arXiv**: [2412.09680](https://arxiv.org/abs/2412.09680)  
**代码**: https://github.com/s3anwu/pbrnerf  
**领域**: 3D视觉  
**关键词**: 逆渲染, 材质估计, 物理先验, NeRF, BRDF

## 一句话总结

PBR-NeRF 在 NeILF++ 的基础上引入了两个基于物理的先验损失（能量守恒损失和 NDF 加权高光损失），有效约束了逆渲染中材质-光照的分解歧义，在不牺牲新视角合成质量的前提下实现了 SOTA 的材质估计。

## 研究背景与动机

**领域现状**：NeRF 和 3DGS 在新视角合成方面取得了显著进展，但它们将场景视为"黑箱"，只建模视角相关的外观，而不区分场景的材质和光照。这意味着它们无法支持重光照、材质编辑等下游任务。逆渲染方法如 NeILF++ 尝试联合估计几何、材质和光照，但面临严重的歧义问题。

**现有痛点**：现有的逆渲染方法在分解材质和光照时缺乏有效的物理先验。NeILF++ 使用的 Lambertian 先验过于强烈，强制假设材质为完全粗糙、非金属的，这会抑制 BRDF 的高光 lobe，导致高光信息被错误地"烘焙"进漫反射 albedo 中。虽然这不影响 RGB 渲染质量，但严重降低了材质估计的准确性。

**核心矛盾**：逆渲染的本质困难在于材质-光照歧义——同一张图片可以由无穷多种材质、光照、几何的组合来解释。现有方法要么使用过强的先验限制了表达力，要么缺乏足够的约束导致材质估计不准确。

**本文目标**：设计更合理的物理先验来约束逆渲染，既要确保 BRDF 的物理有效性，又要促进漫反射和高光的正确分离。

**切入角度**：作者从基于物理的渲染（PBR）理论出发，观察到 Disney BRDF 模型本身不保证能量守恒，且漫反射 lobe 容易在镜面反射方向上过度补偿。这两个物理层面的缺陷可以用显式的损失函数来纠正。

**核心 idea**：通过两个直观的物理损失——能量守恒损失和 NDF 加权高光损失——直接约束 Disney BRDF 的行为，从而在不需要额外网络或数据的情况下大幅提升材质估计质量。

## 方法详解

### 整体框架

PBR-NeRF 的整体管线沿用 NeILF++ 的隐式微分渲染器（IDR），包含三个神经网络：(1) NeRF SDF 网络预测几何（密度）和颜色；(2) NeILF MLP 预测入射辐照度；(3) BRDF MLP 预测 Disney BRDF 参数（albedo $b$、roughness $r$、metallicness $m$）。渲染方程通过 Fibonacci 采样的固定入射方向集 $S_L$（256个方向）离散近似。PBR-NeRF 的核心创新在于在材质估计阶段引入两个新的物理损失，作用于 Disney BRDF 的漫反射和高光分量。

### 关键设计

1. **能量守恒损失（Conservation of Energy Loss）**:

    - 功能：确保 BRDF 不违反能量守恒定律，即反射的总能量不超过入射能量
    - 核心思路：对 BRDF 在所有入射方向上的加权积分施加上界约束。离散化形式为 $\mathcal{L}_{\text{cons}} = \max\{(\frac{2\pi}{|S_L|}\sum f_r(\omega_i \cdot \mathbf{n})) - 1, 0\}$，采用 ReLU 风格的惩罚——只在反射权重总和超过 1 时才产生损失
    - 设计动机：Disney BRDF 和其他微面元模型本身并不保证能量守恒。无约束时材质可能"创造"能量（反射比接收更多的光），这会导致光照估计偏暗来补偿，进而影响重光照等下游任务

2. **NDF 加权高光损失（NDF-weighted Specular Loss）**:

    - 功能：促进 BRDF 的漫反射 lobe 和高光 lobe 的正确分离，消除"烘焙"高光现象
    - 核心思路：用法线分布函数（NDF）作为权重，在镜面反射方向上惩罚漫反射分量的大小。对每个入射方向计算 half vector 处的 NDF 值，经 softmax 归一化后加权漫反射 lobe。NDF 项被 detach 以阻止梯度流过，确保只有漫反射参数被更新
    - 设计动机：在没有约束的情况下，漫反射 lobe 倾向于在镜面反射角度上过度补偿。通过在高光区域压制漫反射，间接迫使高光 lobe 扩展来解释高光效果

3. **三阶段联合优化**:

    - 功能：稳定地联合优化几何、材质和光照
    - 核心思路：分三个阶段训练——(1) 几何阶段：只训练 NeRF SDF 初始化几何；(2) 材质阶段：冻结 SDF，训练 NeILF 和 BRDF 网络；(3) 联合阶段：所有网络一起训练
    - 设计动机：分阶段训练避免了几何、材质和光照之间的耦合优化困难

### 损失函数 / 训练策略

总损失包含几何损失和材质损失。关键超参数为 $\lambda_{\text{cons}}=0.01$，$\lambda_{\text{spec}}$ 在 NeILF++ 数据集上为 0.5、DTU 上为 0.01。训练在单张 A6000 上进行约 3-7.5 小时。

## 实验关键数据

### 主实验

在 NeILF++ 数据集上的材质估计对比（6 个光照条件平均）：

| 评估量 | 方法 | PSNR | SSIM |
|--------|------|------|------|
| RGB | NeILF++ | 30.51 | 86.59 |
| RGB | PBR-NeRF | **31.27** | **87.05** |
| Albedo | NeILF++ | 17.29 | 75.87 |
| Albedo | PBR-NeRF | **20.08** | **86.82** |
| Roughness | NeILF++ | 21.83 | 91.56 |
| Roughness | PBR-NeRF | **22.47** | **92.31** |
| Metallicness | NeILF++ | 18.84 | 83.46 |
| Metallicness | PBR-NeRF | **21.62** | **74.08** |

### 消融实验

| 配置 | Albedo PSNR | RGB PSNR | 说明 |
|------|------------|----------|------|
| Full PBR-NeRF | 20.08 | 31.27 | 完整模型 |
| w/o Conservation Loss | ~18.5 | ~31.0 | 去掉能量守恒损失 |
| w/o Specular Loss | ~18.0 | ~30.8 | 去掉高光分离损失 |
| NeILF++ baseline | 17.29 | 30.51 | 两个损失都没有 |

### 关键发现

- Albedo 估计提升最为显著（+2.79 dB PSNR），说明两个物理损失有效解决了高光烘焙问题
- RGB 渲染质量不降反升（+0.76 dB），证明正确的材质-光照分解也有助于新视角合成
- 在混合光照（Mix）场景下改进尤为明显，因为近场光源和面光源使得材质-光照歧义更加严重
- 方法对超参数选择较为鲁棒，两个数据集只需微调高光损失权重即可

## 亮点与洞察

- **物理损失的简洁与通用性**：两个损失都直接作用于 BRDF，不依赖特定的渲染框架或网络架构。理论上可以即插即用到任何使用 Disney BRDF 的逆渲染系统中
- **detach 操作的巧妙应用**：在高光损失中 detach NDF 项，确保梯度只流向漫反射参数而不改变 NDF（roughness），避免了两个 lobe 同时被调整导致的不稳定
- **不牺牲 NVS 质量的材质改进**：很多逆渲染工作在追求正确材质分解的同时会损失 RGB 渲染质量，PBR-NeRF 做到了两者兼得

## 局限与展望

- 方法仍依赖 NeILF++ 的三阶段训练，训练时间较长（3-7.5 小时），远慢于 3DGS 类方法
- 只在合成数据上进行了定量材质评估，真实场景缺乏 ground truth 材质
- 能量守恒损失只防止了能量创造，但没有处理能量销毁的问题
- 未与基于 3DGS 的逆渲染方法进行对比

## 相关工作与启发

- **vs NeILF++**: 直接基础模型。仅添加两个损失项即实现了最小化修改带来最大化收益
- **vs PhySG/SG-ENV**: 早期逆渲染方法，各指标大幅落后
- **vs NeRFactor/TensoIR**: 完全分解材质-光照但计算代价更高，PBR-NeRF 更轻量

## 评分

- 新颖性: ⭐⭐⭐⭐ 两个物理损失设计简洁优雅，但整体框架是增量改进
- 实验充分度: ⭐⭐⭐⭐ 两个数据集详尽定量评价，但缺少真实场景材质 GT
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，物理动机阐述透彻
- 价值: ⭐⭐⭐⭐ 即插即用的物理先验设计具有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)
- [\[CVPR 2025\] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [\[CVPR 2025\] Preconditioners for the Stochastic Training of Neural Fields](preconditioners_for_the_stochastic_training_of_neural_fields.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

<!-- RELATED:END -->
