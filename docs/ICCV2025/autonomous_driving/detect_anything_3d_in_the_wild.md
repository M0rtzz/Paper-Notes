---
title: >-
  [论文解读] Detect Anything 3D in the Wild
description: >-
  [ICCV 2025][自动驾驶][3D检测基础模型] DetAny3D 是一个可提示（promptable）的3D检测基础模型，通过融合SAM和depth-pretrained DINO两个2D基础模型的先验知识，并提出2D Aggregator和Zero-Embedding Mapping机制实现稳定的2D-to-3D知识迁移，仅用单目图像即可在任意场景和相机配置下实现零样本3D目标检测，在新类别上零样本AP3D超越基线最多21%。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D检测基础模型
  - 零样本泛化
  - 单目3D检测
  - 2D-to-3D知识迁移
  - 开放世界检测
---

# Detect Anything 3D in the Wild

**会议**: ICCV 2025  
**arXiv**: [2504.07958](https://arxiv.org/abs/2504.07958)  
**代码**: [https://github.com/OpenDriveLab/DetAny3D](https://github.com/OpenDriveLab/DetAny3D)  
**领域**: 自动驾驶 / 3D目标检测  
**关键词**: 3D检测基础模型, 零样本泛化, 单目3D检测, 2D-to-3D知识迁移, 开放世界检测

## 一句话总结

DetAny3D 是一个可提示（promptable）的3D检测基础模型，通过融合SAM和depth-pretrained DINO两个2D基础模型的先验知识，并提出2D Aggregator和Zero-Embedding Mapping机制实现稳定的2D-to-3D知识迁移，仅用单目图像即可在任意场景和相机配置下实现零样本3D目标检测，在新类别上零样本AP3D超越基线最多21%。

## 研究背景与动机

3D目标检测是自动驾驶、机器人和增强现实的核心技术。理想的通用3D检测器应能从单目图像输入中检测任意物体，不依赖特定传感器参数。然而现有方法存在以下痛点：

- **闭集假设限制**：现有检测器（如Cube R-CNN, Omni3D）虽支持多数据集训练，但仍局限于预定义的类别空间，无法检测未见过的物体
- **相机配置敏感**：跨数据集部署时，因相机参数差异导致严重域差距
- **3D标注数据稀缺**：3D标注数据量仅百万级，比2D图像标注（数十亿级）少3-4个数量级，从零训练3D基础模型几乎不可行

核心矛盾在于：**3D数据不足以支撑基础模型训练，但2D基础模型（SAM、DINOv2）拥有丰富的形状和几何先验**。本文的切入角度是利用预训练2D基础模型的知识来弥补3D数据的不足，通过精心设计的架构实现有效的2D-to-3D知识迁移。

## 方法详解

### 整体框架

DetAny3D 以单目RGB图像和提示（box/point/text/intrinsic）作为输入。图像被两个基础模型并行编码：SAM（提供像素级形状信息，作为可提示骨干）和 depth-pretrained DINO（通过UniDepth预训练，提供几何深度先验）。两者特征通过 2D Aggregator 融合后，经 Depth/Camera Module 提取几何嵌入，最终由 3D Interpreter 解码为3D包围盒预测。

### 关键设计

1. **2D Aggregator（2D特征聚合器）**:

    - 功能：融合来自SAM和DINO的异构特征，消除表示冲突
    - 核心思路：采用层次化交叉注意力机制，包含4个级联对齐单元。每个单元通过可学习门控权重 $\alpha_i$（初始化为0.5）动态融合两个模型的特征：$\mathbf{F}_{\text{fused}}^{i}=\alpha_{i}\cdot\mathbf{F}_{s}^{i}+(1-\alpha_{i})\cdot\mathbf{F}_{d}^{i}$，然后以融合特征为KV、查询特征为Q进行交叉注意力
    - 设计动机：SAM擅长细粒度空间信息，DINO擅长深度几何信息，两者具有互补性但特征空间不同，需要自适应对齐和融合

2. **3D Interpreter 与 Zero-Embedding Mapping (ZEM)**:

    - 功能：将2D特征逐步注入3D几何信息，同时确保稳定的2D-to-3D知识迁移
    - 核心思路：包含 Two-Way Transformer（继承SAM解码器结构）和 Geometric Transformer。ZEM 使用零初始化的 $1\times1$ 卷积层，将几何嵌入 $\mathbf{G}$ 逐步注入到特征中：$\mathbf{G}'=\text{GeoTrans}(\mathbf{Q}, \text{ZEM}(\mathbf{G})+\mathbf{F}_s, \text{ZEM}(\mathbf{G})+\mathbf{F}_s)$
    - 设计动机：直接注入3D几何特征会干扰预训练的2D特征，导致灾难性遗忘。ZEM通过零初始化确保训练初期不改变原始2D特征，逐步学习几何注入，稳定跨数据集训练

3. **多模态提示交互**:

    - 功能：支持box、point、text和intrinsic四种提示方式
    - 核心思路：Box/Point提示遵循SAM的位置编码方式；Text提示通过Grounding DINO获取2D框后转换；Intrinsic提示为可选的相机内参，未提供时模型自行预测
    - 设计动机：借鉴SAM的可提示设计理念，实现灵活的用户交互和开放世界检测

### 损失函数 / 训练策略

总损失为三部分之和：
- **深度损失** $\mathcal{L}_{\text{depth}}$：SILog损失监督深度预测
- **相机内参损失** $\mathcal{L}_{\text{cam}}$：基于密集相机射线表示的SILog损失
- **检测损失** $\mathcal{L}_{\text{det}}$：Smooth L1损失（3D框参数回归）+ Chamfer损失（旋转矩阵）+ MSE损失（3D IoU分数）

训练细节：SAM编码器冻结，使用ViT-L DINOv2和ViT-H SAM初始化。8×8 A100 GPU，batch size 64，训练80 epoch约2周。DA3D数据集包含16个数据集，0.4M帧，20种相机配置。

## 实验关键数据

### 主实验

**零样本新类别检测（GT prompt）:**

| 数据集 | 指标 | DetAny3D | OVMono3D | 提升 |
|--------|------|----------|----------|------|
| KITTI | AP3D | 28.96 | 8.44 | +20.52 (3.4×) |
| SUNRGBD | AP3D | 39.09 | 17.16 | +21.93 (2.3×) |
| ARKitScenes | AP3D | 57.72 | 14.12 | +43.60 (4.1×) |

**零样本新相机配置检测（Grounding DINO prompt, target-aware metric）:**

| 数据集 | 指标 | DetAny3D | OVMono3D | 提升 |
|--------|------|----------|----------|------|
| Cityscapes3D | AP3D | 15.71 | 10.98 | +4.73 |
| Waymo | AP3D | 15.95 | 10.27 | +5.68 |
| 3RScan | AP3D | 9.58 | 8.48 | +1.10 |

**域内Omni3D检测（GT prompt）：AP3D = 34.38 vs OVMono3D 25.32（+9.06）**

### 消融实验

| 配置 | AP3D | 说明 |
|------|------|------|
| SAM基线（无附加组件） | 5.81 | 仅SAM+3D头 |
| + Depth & Camera模块 | 10.10 | 深度和相机模块提供+4.29 |
| + 合并DINO | 20.20 | DINO几何先验贡献巨大(+10.10) |
| + 2D Aggregator | 23.21 | 比直接加法融合更好(+3.01) |
| + ZEM | 25.80 | 稳定迁移带来+2.59 |

### 关键发现
- 合并depth-pretrained DINO是最大的增益来源，证明了几何先验对单目3D检测的关键作用
- ZEM机制在跨数据集训练时稳定性提升显著，避免了不同数据分布间的冲突
- 2D提示质量是性能瓶颈——使用GT 2D框时AP3D远高于使用Cube R-CNN检测结果
- DetAny3D的3D检测结果可用于下游任务如Sora视频生成的3D框引导

## 亮点与洞察
- **核心创新**：首个真正意义上的可提示3D检测基础模型，零样本泛化能力远超现有方法
- **巧妙的知识迁移**：ZEM的零初始化策略简单但极其有效，避免了2D预训练权重的灾难性遗忘
- **工程价值**：聚合了16个数据集构建DA3D统一基准，为3D检测基础模型的系统评估提供了标准
- 展示了从2D基础模型到3D任务的有效知识迁移路径，对其他3D任务有启发

## 局限与展望
- 依赖于2D提示的质量，当前2D检测器（如Cube R-CNN）的性能成为瓶颈
- SAM编码器在训练中冻结，可能限制了对3D任务的适应性
- 直线声学射线模型假设可能在复杂几何场景中不够精确
- 未探索点云或深度传感器等多模态输入的可能性
- 3RScan等命名歧义严重的数据集上效果仍然有限

## 相关工作与启发
- **Omni3D / Cube R-CNN**：多数据集统一训练的先驱，但限于闭集检测
- **OVMono3D**：开放词汇3D检测的尝试，但未充分利用2D基础模型先验
- **SAM / DINOv2**：2D基础模型，本文成功将其先验迁移到3D任务
- **UniDepth**：提供了depth-pretrained DINO和相机-深度联合估计的框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个可提示3D检测基础模型，ZEM机制新颖有效
- 实验充分度: ⭐⭐⭐⭐⭐ 16个数据集，零样本/域内/消融全面覆盖，结果显著
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 为3D检测领域开辟了基础模型方向，实际应用价值高

<!-- RELATED:START -->

## 相关论文

- [LabelAny3D: Label Any Object 3D in the Wild](../../NeurIPS2025/autonomous_driving/labelany3d_label_any_object_3d_in_the_wild.md)
- [Towards In-the-Wild 3D Plane Reconstruction from a Single Image](../../CVPR2025/autonomous_driving/towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)
- [SAM4D: Segment Anything in Camera and LiDAR Streams](sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [Segment Anything, Even Occluded](../../CVPR2025/autonomous_driving/segment_anything_even_occluded.md)
- [Cubify Anything: Scaling Indoor 3D Object Detection](../../CVPR2025/autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)

<!-- RELATED:END -->
