---
title: >-
  [论文解读] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation
description: >-
  [CVPR 2025][自动驾驶][深度基础模型] Prompt Depth Anything 首次将"提示"范式引入深度基础模型，利用低成本 LiDAR（如 iPhone LiDAR）作为度量提示，通过简洁的多尺度提示融合架构指导 Depth Anything 模型输出精确的度量深度，实现了高达 4K 分辨率的高质量深度估计。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "深度基础模型"
  - "度量深度估计"
  - "LiDAR提示"
  - "深度补全"
  - "4K分辨率"
---

# Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation

**会议**: CVPR 2025  
**arXiv**: [2412.14015](https://arxiv.org/abs/2412.14015)  
**代码**: [https://PromptDA.github.io/](https://PromptDA.github.io/)  
**领域**: 自动驾驶  
**关键词**: 深度基础模型, 度量深度估计, LiDAR提示, 深度补全, 4K分辨率

## 一句话总结

Prompt Depth Anything 首次将"提示"范式引入深度基础模型，利用低成本 LiDAR（如 iPhone LiDAR）作为度量提示，通过简洁的多尺度提示融合架构指导 Depth Anything 模型输出精确的度量深度，实现了高达 4K 分辨率的高质量深度估计。

## 研究背景与动机

**领域现状**：深度基础模型（如 Depth Anything v2、DepthPro、Metric3D v2）通过大规模数据训练获得了强大的泛化能力，能生成高质量的相对深度图。然而，这些模型在度量深度（metric depth）估计上仍存在尺度歧义问题，即无法给出准确的绝对距离。

**现有痛点**：现有解决度量深度的方案主要有两条路：（1）在度量数据集上微调深度基础模型，但泛化能力会下降；（2）将相机内参作为额外输入训练度量深度模型，但效果仍不理想。如图 1(b) 所示，Metric3D v2 存在不准确的尺度和帧间不一致性。

**核心矛盾**：深度基础模型具有强大的局部形状理解能力（学到了丰富的几何先验），但缺乏绝对尺度信息。单目图像本质上无法提供度量信息，而现有的度量提示方式（如相机内参）提供的约束不够强。

**本文目标**：设计一种方式将度量信息"注入"到深度基础模型中，使其输出精确的绝对深度，同时保持高分辨率和泛化能力。

**切入角度**：作者类比 NLP 和视觉基础模型中的"预训练 + 提示微调"范式——一个设计合理的提示（prompt）可以释放基础模型在下游任务上的潜力。低成本 LiDAR（如 iPhone 内置的 dToF 传感器）提供了精确但低分辨率（240×320）、有噪声的深度，正好可以作为"度量提示"。

**核心 idea**：将度量深度估计重构为"用度量提示引导深度基础模型"的条件生成任务，低成本 LiDAR 提供绝对尺度信息，基础模型负责高分辨率的局部几何细节，两者互补。

## 方法详解

### 整体框架

Prompt Depth Anything 基于 Depth Anything v2 的 ViT-Large + DPT 解码器架构。输入是一张 RGB 图像和一张低分辨率 LiDAR 深度图，输出是高分辨率（最高 4K）的精确度量深度图。核心改动是在 DPT 解码器的每个尺度上添加一个提示融合模块，将 LiDAR 深度信息多尺度地注入到解码过程中。训练数据通过合成数据 LiDAR 仿真和真实数据伪 GT 深度生成的混合 pipeline 构建。

### 关键设计

1. **多尺度提示融合架构（Prompt Fusion Architecture）**:

    - 功能：将低分辨率 LiDAR 深度作为条件信息融入深度基础模型的解码过程
    - 核心思路：在 DPT 解码器的每个尺度 $S_i$ 上，先将 LiDAR 深度图双线性插值到当前尺度的空间分辨率，然后通过一个浅层卷积网络（两层 3×3 卷积）提取深度特征，再用零初始化的卷积层投影到与图像特征相同的维度，最后加到 DPT 的中间特征上参与深度解码。整个融合模块仅增加 5.7% 的计算开销（1.789 vs 1.691 TFLOPs）。
    - 设计动机：零初始化设计保证初始输出与原始基础模型完全一致，完整继承了预训练模型的能力。多尺度融合充分利用了 LiDAR 提供的不同粒度的空间距离信息。相比 ControlNet、CrossAttention、Adaptive LayerNorm 等条件注入方式，简单的加法融合更能利用 LiDAR 和输出深度之间的像素对齐特性。

2. **合成数据 LiDAR 仿真（Sparse Anchor Interpolation）**:

    - 功能：为合成数据（如 Hypersim）生成逼真的 LiDAR 深度输入
    - 核心思路：简单下采样 GT 深度无法模拟真实 LiDAR 的噪声特性，模型会退化为学习深度超分辨率。为此，作者提出稀疏锚点插值法：先将 GT 深度下采样到 LiDAR 分辨率（192×256），再在其上用扰动网格（步长 7）采样稀疏锚点，其余点通过 RGB 相似度 KNN 插值得到。这样模拟了 LiDAR 的噪声和插值伪影。
    - 设计动机：如果仿真的 LiDAR 过于"干净"，模型只学会做超分辨率而不学纠正噪声，在真实 LiDAR 上效果会打折扣。稀疏锚点插值引入了与真实 iPhone LiDAR 类似的噪声模式。

3. **伪 GT 深度生成 + 边缘感知损失**:

    - 功能：为只有粗糙 GT 深度的真实数据（ScanNet++）生成高质量的训练监督信号
    - 核心思路：ScanNet++ 的 FARO 激光扫描深度在纹理丰富的区域边缘质量差（遮挡导致的孔洞和伪影），而 Zip-NeRF 重建能产生高质量边缘但在无纹理区域不准确。边缘感知损失将两者优势结合：$\mathcal{L}_{edge} = L_1(\mathbf{D}_{gt}, \hat{\mathbf{D}}) + \lambda \cdot \mathcal{L}_{grad}(\mathbf{D}_{pseudo}, \hat{\mathbf{D}})$，其中 $L_1$ 项用 FARO 深度监督整体深度值（在无纹理平面区域准确），梯度项用 Zip-NeRF 伪 GT 的梯度监督输出深度的梯度（在边缘处准确）。
    - 设计动机：任何单一数据源都有缺陷，通过损失函数设计巧妙地组合了两种深度的互补优势，避免了伪 GT 中不可靠区域对训练的负面影响。

### 损失函数 / 训练策略

- 深度归一化：使用 LiDAR 深度的最小/最大值线性缩放到 [0,1]，输出也同步归一化，保证尺度一致
- 从 Depth Anything v2 的度量模型初始化，先进行 10K 步预热（微调到输出归一化深度），再训练 200K 步
- AdamW 优化器，ViT 骨干 lr=5e-6，其他参数 lr=5e-5，batch size=2，8 GPU
- $\lambda=0.5$

## 实验关键数据

### 主实验

ARKitScenes 数据集（768×1024 分辨率）：

| 方法 | 类型 | L1↓ | RMSE↓ |
|------|------|-----|-------|
| Depth Anything v2 (对齐后) | Post | 0.0771 | 0.0647 |
| Metric3D v2 (对齐后) | Post | 0.0524 | 0.1721 |
| DepthPro (微调) | w/o LiDAR | 0.0435 | 0.0665 |
| Depth Prompting | Net | 0.0253 | 0.0422 |
| ARKit Depth | - | 0.0250 | 0.0423 |
| **Ours (zero-shot)** | **Net** | **0.0163** | **0.0371** |
| **Ours** | **Net** | **0.0132** | **0.0315** |

ScanNet++ 数据集（深度估计 + TSDF 重建）：

| 方法 | L1↓ | RMSE↓ | F-score↑ |
|------|-----|-------|----------|
| Depth Anything v2 (微调) | 0.0510 | 0.1010 | 0.6595 |
| **Ours** | **0.0250** | **0.0829** | **0.7619** |
| Ours (zero-shot, 仅合成数据) | 0.0327 | 0.0966 | 0.7307 |

### 消融实验

| 配置 | L1↓ | RMSE↓ | 说明 |
|------|-----|-------|------|
| Full model (prompt fusion) | 0.0135 | 0.0326 | 完整模型 |
| Adaptive LayerNorm | - | - | 次优，不适合像素对齐的条件 |
| CrossAttention | - | - | 计算开销大，效果不如直接融合 |
| ControlNet | - | - | 复制编码器参数过多 |
| w/o 基础模型初始化 | 显著下降 | - | 证明继承预训练权重至关重要 |
| w/o edge-aware loss | 边缘质量下降 | - | 边缘感知损失对真实数据训练至关重要 |
| 简单下采样仿真 LiDAR | 退化为超分辨率 | - | 证明 LiDAR 噪声仿真的必要性 |

### 关键发现

- 即使是零样本模型（仅在合成数据上训练）也超过了在目标数据集上训练/微调的其他方法，证明了"提示基础模型"范式的强泛化能力
- 简单的加法融合比 ControlNet、CrossAttention 等复杂设计更有效，因为 LiDAR 和深度输出是像素对齐的，不需要跨模态的注意力机制
- 深度基础模型作为"局部形状学习器"配合 LiDAR 的"全局尺度锚点"角色分工明确，效果优于任何单一路线
- 方法可替换基础模型（如换成 DepthPro）和 LiDAR 类型（如车载 LiDAR），具有良好的扩展性

## 亮点与洞察

- **将 prompting 范式引入深度估计**：类比 NLP 中的 prompt tuning，用 LiDAR 作为"prompt"激活基础模型在度量深度估计上的能力，这种思路可以推广到其他需要额外信号引导的基础模型任务中
- **零初始化融合模块**：确保新增模块在初始化时不改变原模型输出，训练更稳定，且完整保留了预训练模型的泛化能力
- **互补数据源的损失函数设计**：用梯度损失只监督边缘、用 L1 损失只监督平坦区域的思路，适用于任何需要组合多种不完美监督信号的场景

## 局限与展望

- 依赖 LiDAR 传感器输入，纯单目场景下无法使用（虽然 iPhone 等设备普遍配备 LiDAR，但仍有很多场景没有）
- 目前主要在室内场景验证，室外大范围场景的泛化性有待进一步评估
- LiDAR 仿真策略相对简化，更复杂的 LiDAR 噪声模型（如考虑材质反射率、多路径干扰等）可能进一步提升性能
- 训练需要 Zip-NeRF 为真实数据生成伪 GT，该过程本身耗时且可能引入系统误差
- 未来可探索其他形式的"度量提示"，如稀疏 SfM 点、IMU 数据等

## 相关工作与启发

- **vs Depth Anything v2**: 本文在 DAv2 基础上增加了 LiDAR 提示，将其从相对深度模型升级为精确度量深度模型，证明了基础模型 + 提示的范式比直接微调更有效
- **vs Depth Prompting**: Depth Prompting 是在基础模型输出后用稀疏深度做后处理融合，不算真正的"提示"；本文在解码过程中多尺度融合 LiDAR 信息，是更深层的条件集成
- **vs 传统深度补全方法**: 传统方法（如 NLSPN、BPNet）将稀疏到稠密当作独立任务学习，没有利用预训练深度基础模型的强先验；本文将基础模型作为强正则化，大幅提升了泛化能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 prompting 概念引入深度基础模型是新颖的，但技术实现（多尺度特征融合）较为常规
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、多分辨率、多基线比较、丰富的消融和下游应用验证
- 写作质量: ⭐⭐⭐⭐⭐ 叙事流畅，motivation 清晰，图表信息丰富
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高，iPhone 普及的 LiDAR + 基础模型即可获得 4K 高质量深度，对 3D 重建和机器人抓取等下游应用有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](../../CVPR2026/autonomous_driving/ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[CVPR 2025\] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting](pansplat_4k_panorama_synthesis_with_feed-forward_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
