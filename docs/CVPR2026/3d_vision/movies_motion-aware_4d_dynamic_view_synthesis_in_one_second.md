---
title: >-
  [论文解读] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second
description: >-
  [CVPR 2026][3D视觉][动态视图合成] 提出 MoVieS，一个前馈式 4D 动态场景重建框架，通过 **动态溅射像素 (Dynamic Splatter Pixel)** 表示将外观、几何和运动统一建模，从单目视频在约 1 秒内完成 4D 重建，并支持新视角合成、3D 点跟踪、场景流估计和运动物体分割等多种任务。
tags:
  - CVPR 2026
  - 3D视觉
  - 动态视图合成
  - 4D重建
  - 3D高斯泼溅
  - 点跟踪
  - 前馈式重建
---

# MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second

**会议**: CVPR 2026  
**arXiv**: [2507.10065](https://arxiv.org/abs/2507.10065)  
**代码**: [有](https://chenguolin.github.io/projects/MoVieS)  
**领域**: 3D视觉  
**关键词**: 动态视图合成, 4D重建, 3D高斯泼溅, 点跟踪, 前馈式重建  

## 一句话总结

提出 MoVieS，一个前馈式 4D 动态场景重建框架，通过 **动态溅射像素 (Dynamic Splatter Pixel)** 表示将外观、几何和运动统一建模，从单目视频在约 1 秒内完成 4D 重建，并支持新视角合成、3D 点跟踪、场景流估计和运动物体分割等多种任务。

## 研究背景与动机

现有 3D 视觉方法存在三个核心痛点：

**任务割裂**：深度估计、3D 重建、新视角合成、点跟踪等任务各自独立，缺乏统一建模。但这些任务共享底层 3D 先验，分离处理浪费了它们之间的互补信息。

**静态场景局限**：大多数前馈式重建方法（如 pixelSplat、GS-LRM、VGGT）仅处理静态场景，无法建模运动物体。

**优化式动态重建效率低下**：Shape-of-Motion、MoSca 等方法需要 10-45 分钟的逐场景优化，且依赖外部光流/点跟踪模型提供运动监督，流程复杂且难以泛化。

已有的前馈动态方法也各有缺陷：BTimer 逐帧独立预测缺乏时序一致性，需额外 enhancer 模块；NutWorld 缺少运动显式监督，使用正交相机导致投影失真。MoVieS 的核心动机是：**能否用一个统一的前馈模型，同时输出外观、几何和运动，并在 1 秒内完成 4D 重建？** 关键洞察在于新视角合成和运动估计可以相互促进——渲染损失为运动提供密集空间约束，而显式运动监督帮助模型学习时序一致的几何。

## 方法详解

### 整体框架

MoVieS 接收带有相机参数和时间戳的单目视频 $\mathcal{V} = \{\mathbf{I}_i, \mathbf{P}_i, \mathbf{K}_i, t_i\}_{i=1}^{N}$，通过三个阶段处理：

1. **特征提取**：预训练图像编码器（DINOv2）提取逐帧特征，融合相机嵌入和时间戳 token
2. **跨帧注意力**：VGGT 的几何预训练注意力块实现帧间信息交互，产生富含跨帧上下文的共享特征
3. **三头预测**：深度头、溅射头（外观）、运动头并行输出动态溅射像素的全部属性

最终通过可微 3DGS 渲染器将动态溅射像素渲染为目标视角和时间的图像。

### 关键设计

1. **动态溅射像素 (Dynamic Splatter Pixel)**：核心表示方法。将动态场景分解为静态高斯基元和时变形变场。每个像素对应一个溅射像素 $\mathbf{g} = \{\mathbf{x}, \mathbf{a}\}$，其中 $\mathbf{x} \in \mathbb{R}^3$ 为规范空间位置，$\mathbf{a} \in \mathbb{R}^{11}$ 包含旋转四元数、尺度、不透明度和颜色。对于动态内容，附加时变形变 $\mathbf{m}(t) = \{\Delta\mathbf{x}(t), \Delta\mathbf{a}(t)\}$，通过简单加法更新：
$$\mathbf{x} \leftarrow \mathbf{x} + \Delta\mathbf{x}(t), \quad \mathbf{a} \leftarrow \mathbf{a} + \Delta\mathbf{a}(t)$$
设计动机：与以往将运动嵌入隐式场不同，显式分离静态几何和动态形变使得运动可被直接监督和可视化，同时在静态场景中运动自然趋近于零。

2. **双重相机条件注入**：将相机参数编码为两种互补形式注入特征。(a) **Plücker 嵌入**：将 $\mathbf{P}_i$ 和 $\mathbf{K}_i$ 转换为像素对齐的 Plücker 射线表示，与图像特征空间相加，提供局部几何约束。(b) **Camera Token**：将相机参数通过线性层编码为全局 token，拼接到注意力序列中，提供全局视角信息。消融实验证明两者结合效果最佳（PSNR 27.60 vs 单独 Plücker 25.81 或 Camera Token 26.81）。

3. **运动头 (Motion Head)**：采用自适应层归一化 (AdaLN) 将查询时间 $t_q$ 的正弦编码注入特征 token，然后通过 DPT 卷积预测每像素 3D 位移 $\Delta\mathbf{x}$ 和属性形变 $\Delta\mathbf{a}$。支持对任意查询时间戳预测运动，在推理时可通过改变 $t_q$ 实现连续时间的 4D 重建。运动图的可视化将 XYZ 坐标归一化到 $[0,1]$ 并映射为 RGB 通道。

4. **解耦深度头与溅射头**：不同于以往用单个头预测所有高斯属性，MoVieS 使用独立的深度头（从 VGGT 初始化）负责几何，溅射头（从头训练）负责外观。溅射头还引入 RGB 直连捷径从输入图像到最终卷积层，保留高频细节和色彩保真度。此解耦设计更好地利用了 VGGT 的几何先验。

5. **运动监督设计**：结合点级 L1 损失和分布级损失：
$$\mathcal{L}_{\text{motion}} = \frac{\lambda_{\text{pt}}}{P}\sum_{i \in \Omega}\|\Delta\hat{\mathbf{x}}_i - \Delta\mathbf{x}_i\|_1 + \frac{\lambda_{\text{dist}}}{P^2}\sum_{(i,j) \in \Omega \times \Omega}\|\Delta\hat{\mathbf{x}}_i \cdot \Delta\hat{\mathbf{x}}_j^\top - \Delta\mathbf{x}_i \cdot \Delta\mathbf{x}_j^\top\|_1$$
点级损失提供绝对运动约束，分布级损失保持像素间的相对运动结构。消融显示两者互补：点级产生合理运动图，分布级增强运动边界锐利度。

### 损失函数/训练策略

总损失为三部分加权组合：$\mathcal{L} = \lambda_d \mathcal{L}_{\text{depth}} + \lambda_r \mathcal{L}_{\text{rendering}} + \lambda_m \mathcal{L}_{\text{motion}}$

- **深度损失**：预测深度图与 GT 的 MSE + 空间梯度 L1 损失，过滤无效值
- **渲染损失**：像素 MSE + LPIPS 感知损失（$\lambda_{\text{LPIPS}} = 0.5$），对 $M$ 个随机采样目标时间戳渲染图计算
- **权重设置**：$\lambda_d = 1, \lambda_r = 1, \lambda_m = 10, \lambda_{\text{pt}} = 1, \lambda_{\text{dist}} = 10$
- **课程训练**：分三阶段逐步增加复杂度——(1) 静态场景预训练 (2) 动态场景+多视角训练 (3) 高分辨率微调
- **数据集**：8 个异构数据集混合训练（RealEstate10K 70K 场景、TartanAir、MatrixCity、PointOdyssey、DynamicReplica、Spring、VKITTI2、Stereo4D 98K 场景）
- **工程优化**：gsplat 渲染后端、DeepSpeed、梯度检查点、梯度累积、bf16 混合精度，32×H20 GPU 约 5 天

## 实验关键数据

### 主实验：新视角合成

| 方法 | 类型 | 每场景耗时 | RE10K PSNR↑ | DyCheck mPSNR↑ | DyCheck mSSIM↑ | NVIDIA PSNR↑ |
|------|------|-----------|-------------|-----------------|----------------|--------------|
| DepthSplat | 前馈(静态) | 0.60s | 26.57 | 13.83 | 43.64 | 17.16 |
| GS-LRM† | 前馈(静态) | 0.57s | 26.94 | 14.60 | 45.35 | 17.83 |
| Ours (static) | 前馈(静态) | 0.84s | **27.60** | 15.24 | 47.84 | 18.73 |
| Splatter-a-Video | 优化式 | 37min | - | 13.61 | 31.31 | 14.39 |
| Shape-of-Motion | 优化式 | 10min | - | 17.96 | 56.62 | 15.30 |
| MoSca | 优化式 | 45min | - | 18.24 | 55.14 | 21.45 |
| **MoVieS** | **前馈(动态)** | **0.93s** | 26.98 | **18.46** | **58.87** | 19.16 |

### 主实验：3D 点跟踪 (TAPVid-3D)

| 方法 | ADT EPE3D↓ | ADT δ0.05↑ | ADT δ0.10↑ | DriveTrack EPE3D↓ | Panoptic δ0.05↑ |
|------|------------|------------|------------|-------------------|-----------------|
| BootsTAPIR† | 0.5539 | 17.73% | 32.97% | 0.0617 | 69.28% |
| CoTracker3† | 0.5614 | 19.88% | 35.82% | 0.0637 | 69.27% |
| SpatialTracker | 0.5413 | 18.08% | 38.23% | 0.0648 | 72.91% |
| **MoVieS** | **0.2153** | **52.05%** | **71.63%** | **0.0472** | **87.88%** |

### 消融实验

| 运动监督策略 | ADT EPE3D↓ | ADT δ0.05↑ | ADT δ0.10↑ |
|-------------|-----------|-----------|-----------|
| 无运动监督 | 0.7938 | 19.58% | 32.86% |
| + 逐点 L1 | 0.2262 | 48.74% | 69.93% |
| + 分布损失 | 0.2496 | 45.98% | 66.87% |
| 两者结合 (Ours) | **0.2153** | **52.05%** | **71.63%** |

| NVS 与运动的协同效应 | DyCheck mPSNR↑ | NVIDIA PSNR↑ | ADT EPE3D↓ | ADT δ0.05↑ |
|---------------------|---------------|-------------|-----------|-----------|
| NVS 无运动 | 15.82 | 18.38 | 0.7938 | 19.58% |
| 运动无 NVS | 16.26 | 18.98 | 0.3801 | 24.72% |
| **完整模型** | **18.46** | **19.16** | **0.2153** | **52.05%** |

### 关键发现

1. **速度优势惊人**：MoVieS 仅需 0.93 秒完成 4D 重建，比优化式方法快 600-2900 倍（Shape-of-Motion 10min, MoSca 45min），同时达到可比甚至更优的性能
2. **运动与视图合成强耦合**：消融实验清晰证明两个任务互相促进。仅靠 NVS 无法学到有意义的运动（EPE3D 0.79 vs 0.22）；缺少 NVS 的运动预测模糊且低质量。联合训练使两个任务都显著提升
3. **静态-动态无缝衔接**：处理静态输入时预测运动自然收敛到零（< 1e-3），模型隐式学会了区分静态和动态区域
4. **3D 点跟踪大幅领先**：在 ADT 上 EPE3D 从次优的 0.54 降至 0.22（提升 60%），δ0.05 从 19.88% 提升至 52.05%，因为直接在 3D 空间估计位移，避免了 2D 跟踪+深度反投影的误差累积
5. **零样本泛化能力**：运动图可直接用于场景流估计（运动向量从世界坐标转到相机坐标）和运动物体分割（对运动向量范数取阈值），无需任何任务特定微调

## 亮点与洞察

- **统一表示的优雅性**：动态溅射像素将静态 3DGS 自然扩展到 4D，通过简单加法形变实现动态建模，同时保持可微渲染完整性。比隐式形变场或 4D 原语更简洁高效
- **代理任务思想**：新视角合成作为运动学习的代理任务，提供了比稀疏点跟踪标注密集得多的空间约束。「用渲染监督运动」的思路值得广泛借鉴
- **异构数据大规模训练**：灵活的模型设计允许在 8 个标注不同的数据集上混合训练，课程学习策略有效缓解了异构数据带来的不稳定性
- **预训练+微调在 4D 的成功**：基于 VGGT 初始化将训练时间缩短约 3 倍，但从零训练也能达到类似效果

## 局限性

1. **依赖已知相机参数**：假设输入视频带有准确位姿和内参，未处理无位姿视频（作者明确留给未来工作）
2. **NVIDIA 数据集上不如 MoSca**：在多视角动态场景（NVIDIA PSNR 21.45 vs 19.16），优化式方法在细节拟合上仍有优势
3. **训练不稳定性**：三阶段课程训练 + 32 张 H20 GPU 的开销对复现造成门槛，训练中出现 loss 震荡和 None 梯度
4. **运动头时间复杂度**：每个查询时间戳需独立推理，密集时间采样时推理开销线性增长

## 评分

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 动态溅射像素和外观-几何-运动统一建模思想新颖，但基础组件（3DGS、VGGT、DPT）均为已有技术组合 |
| 实验 | ⭐⭐⭐⭐⭐ | 覆盖静态/动态 NVS、3D 点跟踪、零样本应用，消融设计精到（运动监督、NVS-运动协同、相机条件），公平比较（统一相机参数） |
| 写作 | ⭐⭐⭐⭐ | 结构清晰，图表质量高，动机论述充分，消融可视化有效支撑设计选择 |
| 价值 | ⭐⭐⭐⭐⭐ | 将 4D 动态重建从分钟级压缩到秒级同时保持竞争力，统一框架的实用性和零样本泛化为后续工作奠定强基础 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)
- [\[ICLR 2026\] Sharp Monocular View Synthesis in Less Than a Second](../../ICLR2026/3d_vision/sharp_monocular_view_synthesis_in_less_than_a_second.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)

</div>

<!-- RELATED:END -->
