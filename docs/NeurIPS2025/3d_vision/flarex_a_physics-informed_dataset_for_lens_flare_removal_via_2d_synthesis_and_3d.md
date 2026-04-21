---
title: >-
  [论文解读] FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering
description: >-
  [NeurIPS 2025][3D视觉][镜头光晕去除] 提出 FlareX 数据集，通过参数化模板创建、基于光照定律的 2D 合成和基于物理引擎的 3D 渲染三个阶段生成物理真实的镜头光晕数据，训练的模型在真实世界测试集上显著超越此前所有数据集。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 镜头光晕去除
  - 数据集
  - 物理仿真
  - 2D合成
  - 3D渲染
---

# FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering

**会议**: NeurIPS 2025  
**arXiv**: [2510.09995](https://arxiv.org/abs/2510.09995)  
**代码**: https://github.com/qulishen/FlareX  
**领域**: 3D视觉 / 图像复原  
**关键词**: 镜头光晕去除, 数据集, 物理仿真, 2D合成, 3D渲染

## 一句话总结
提出 FlareX 数据集，通过参数化模板创建、基于光照定律的 2D 合成和基于物理引擎的 3D 渲染三个阶段生成物理真实的镜头光晕数据，训练的模型在真实世界测试集上显著超越此前所有数据集。

## 研究背景与动机

**领域现状**：镜头光晕（lens flare）在强光条件下拍摄时会导致颜色偏移和信息丢失，影响下游视觉任务。深度学习方法依赖大规模配对数据集来训练去光晕模型，但真实世界中获取配对数据极其困难。

**现有痛点**：现有数据集（Wu et al., Flare7K, Flare7K++）主要通过 2D 叠加人工光晕模板到背景图上来合成，存在两个核心问题：(a) 光晕模板种类有限（3-35 类），无法覆盖真实世界中多样的光晕形态；(b) 合成过程忽略物理定律，随机叠加导致光晕亮度和位置不自然。

**核心矛盾**：训练数据的物理不真实性和模式多样性不足，导致模型在真实场景中泛化能力有限，尤其对反射光晕和屏外光晕效果差。

**本文目标**：(a) 如何生成更多样的光晕模板？(b) 如何让合成图像亮度分布符合物理定律？(c) 如何获取可靠的真实世界测试集？

**切入角度**：利用 Blender 3D 物理引擎模拟光晕形成过程，引入光照定律（illumination law）量化光源与镜头的关系。

**核心 idea**：结合参数化模板 + 光照定律 2D 合成 + 3D 物理渲染，构建多样且物理真实的镜头光晕数据集。

## 方法详解

### 整体框架
FlareX 的生成流程分三个阶段：(1) 用 Blender 参数化创建 9,500 个光晕模板（95 类 × 100 参数变体）；(2) 将模板按光照定律合成到背景图像上（Flare-2D）；(3) 构建 60 个 3D 场景直接渲染 3,000 对光晕图像（Flare-3D）。最终混合 2D 合成和 3D 渲染数据。

### 关键设计

1. **参数化光晕模板创建**:

    - 功能：在 Blender 中手动创建包含光源、条纹、虹膜、炫光等组件的光晕，绑定物理约束后调参生成多样模板
    - 核心思路：利用 Blender 的 Flared 插件作为基础，手动调节相机焦距和光晕组件参数（位置、颜色、大小、形状），将光源绑定到空间点并应用组件间互约束。黑色背景渲染得到光晕模板，去除光晕组件后重渲染得到光源标注
    - 设计动机：此前数据集仅 3-35 种光晕类型，无法覆盖多反射和非球形光源场景。参数化方法从 95 种基础类型扩展出 9,500 个模板

2. **基于光照定律的亮度调整模块（BAM）**:

    - 功能：根据光源到相机的深度和入射角，按照光照定律自动调整叠加光晕的亮度
    - 核心思路：首先对光晕模板做随机仿射变换 $\mathbf{F_i'} = \mathcal{T}_i(\mathbf{F})$，用预训练单目深度估计模型得到背景图深度图 $\mathbf{D}$。然后计算光源深度 $d_i = \frac{1}{N}\sum_{j=1}^{N} \mathbf{D}_{(x_j,y_j)}$ 和入射角 $\theta_i = \arctan(\frac{2r_i}{W} \cdot \tan\frac{\varphi}{2})$，代入光照公式 $E = \frac{I \cdot \cos\theta}{d^2}$ 进行亮度调整：$\mathbf{F_i''} = \frac{\mathbf{F_i'} \cdot \bar{d}^2}{d_i^2 \cdot \sqrt{1+(\frac{2r_i}{W} \cdot \tan(\frac{\varphi}{2}))^2}}$
    - 设计动机：此前方法随机叠加导致亮度不自然（过亮或光晕过大），引入光照定律让光晕强度与空间位置关联，同时支持通过调整视场角 $\varphi$ 为特定相机定制数据集

3. **3D 场景渲染（Flare-3D）**:

    - 功能：在 Blender 中构建 60 个 3D 场景，将光晕放置在合理位置，从多视角渲染配对图像
    - 核心思路：在 3D 场景中定制光晕放置位置（光源附近、天空、窗户外），设定相机运动轨迹关键帧，沿轨迹渲染带/不带光晕的图像对，得到 3,000 对训练数据
    - 设计动机：2D 深度估计不总是准确，3D 渲染天然遵守物理定律，光晕外观与空间位置自动关联，同时避免了 2D 加法溢出导致的图像失真

4. **真实世界测试集构建（Masking 方法）**:

    - 功能：用遮挡器遮挡光源获取无光晕图像，排除遮挡区域后评估
    - 核心思路：拍摄光晕图后，用视力检查遮光板遮挡直接光源拍无光晕图，标注遮挡区域在评估时排除。收集 100 对高分辨率（3024×3024）测试图像
    - 设计动机：此前方法（擦拭镜头）无法消除反射光晕，导致 ground truth 中残留光晕，使评估结果失真

### 损失函数 / 训练策略
训练损失与此前工作一致：L1 损失 + VGG-19 感知损失 + 重建损失。模型在 512×512 裁剪图像上训练 30,000 次迭代。

## 实验关键数据

### 主实验

| 训练数据集 | 模型 | PSNR↑ | SSIM↑ | LPIPS↓ |
|-----------|------|-------|-------|--------|
| Wu et al. | Uformer | 22.252 | 0.645 | 0.143 |
| Flare7K | Uformer | 23.386 | 0.667 | 0.141 |
| Flare7K++ | Uformer | 23.615 | 0.667 | 0.142 |
| **FlareX (Ours)** | **Uformer** | **25.459** | **0.692** | **0.133** |
| Wu et al. | Restormer | 21.798 | 0.653 | 0.145 |
| Flare7K++ | Restormer | 24.495 | 0.671 | 0.138 |
| **FlareX (Ours)** | **Restormer** | **25.096** | **0.688** | **0.131** |

Uformer 在 FlareX 上训练比 Flare7K++ 提升 **+1.84 dB PSNR**，SSIM 提高 3.75%，LPIPS 降低 5%。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| Flare-2D only | 24.853 | 0.690 | 0.140 | 仅用2D合成 |
| Flare-3D only | 23.647 | 0.687 | 0.137 | 仅用3D渲染（数据量有限） |
| **FlareX (2D+3D)** | **25.459** | **0.692** | **0.133** | 混合数据最优 |
| Flare7K w/o 光照定律 | 23.386 | 0.667 | 0.141 | 原始合成 |
| Flare7K w/ 光照定律 | 23.711 | 0.675 | 0.138 | 加入光照定律提升 |
| FlareX w/o 光照定律 | 25.122 | 0.681 | 0.135 | — |
| FlareX w/ 光照定律 | 25.459 | 0.692 | 0.133 | 光照定律提升 +0.34 dB |

### 关键发现
- 2D 和 3D 数据互补，混合使用效果最佳；单独使用 Flare-3D 因数据量仅 3,000 对而表现较弱
- 光照定律对两个数据集都有提升效果，证明物理约束的通用价值
- 在屏外光晕去除任务中（无参考评估），FlareX 训练的模型在 NIQE(3.967) 和 BRISQUE(28.354) 上均为最优
- 用户研究显示 FlareX 测试集比现有测试集更好地反映模型间性能差异

## 亮点与洞察
- **光照定律融入数据合成**：将 $E = I\cos\theta / d^2$ 引入合成管线，用深度图和入射角自动调节光晕亮度。巧妙之处在于不需修改模型架构，仅改进数据就能提升性能
- **Masking 测试方法**：用遮光板物理遮挡光源获取真正的 flare-free 图像，解决了此前擦拭镜头无法消除反射光晕的根本问题
- **2D+3D 混合策略**：2D 合成提供数据量，3D 渲染保证物理一致性，两者互补。这种混合策略可迁移到其他需要合成数据的低级视觉任务

## 局限与展望
- 对极强光源和大面积光晕的去除仍有明显伪影
- 3D 场景仅 60 个，场景多样性有限
- 未探索将物理先验直接嵌入去光晕模型架构的可能性
- 光照定律假设光源为点光源，对面光源的适用性未验证

## 相关工作与启发
- **vs Flare7K/Flare7K++**：此前最大的光晕数据集，但模板种类少（35 类 vs 95 类）且合成无物理约束。FlareX 在所有模型上全面超越
- **vs SDFRD**：专注于 DSLR 相机的合成数据集，而 FlareX 同时覆盖白天和夜间场景
- 数据驱动思路可借鉴：在其他图像复原任务（如去雾、去雨）中同样可引入物理定律改进合成数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 物理引导的数据合成思路有新意，但核心方法（光照定律）相对经典
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个基线模型 × 5 个数据集对比 + 消融 + 用户研究 + 屏外光晕评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ 对光晕去除社区是重要基础贡献，数据集和代码已开源

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)
- [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](../../CVPR2025/3d_vision/pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](../../CVPR2026/3d_vision/dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)
- [DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](../../ICLR2026/3d_vision/diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)
- [Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](../../CVPR2025/3d_vision/sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)

<!-- RELATED:END -->
