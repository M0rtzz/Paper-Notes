---
title: >-
  [论文解读] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation
description: >-
  [NeurIPS 2025][图像生成][9DoF 姿态控制] SceneDesigner 提出了一种基于 CNOCS 地图表示和两阶段强化学习训练的方法，首次实现了多物体 9D 姿态（位置、大小、朝向）的精确控制，在图像生成的可控性和质量上显著超越现有方法。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "9DoF 姿态控制"
  - "CNOCS 表示"
  - "多物体生成"
  - "强化学习微调"
  - "ControlNet"
---

# SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation

**会议**: NeurIPS 2025  
**arXiv**: [2511.16666](https://arxiv.org/abs/2511.16666)  
**代码**: [https://henghuiding.com/SceneDesigner/](https://henghuiding.com/SceneDesigner/) (项目页面)  
**领域**: 可控图像生成 / 3D 感知生成  
**关键词**: 9DoF 姿态控制, CNOCS 表示, 多物体生成, 强化学习微调, ControlNet

## 一句话总结
SceneDesigner 提出了一种基于 CNOCS 地图表示和两阶段强化学习训练的方法，首次实现了多物体 9D 姿态（位置、大小、朝向）的精确控制，在图像生成的可控性和质量上显著超越现有方法。

## 研究背景与动机
3D 感知的可控图像生成是一个重要但尚未充分解决的问题。现有方法存在以下局限：

**2D 控制主导**：大多数方法（GLIGEN、InstanceDiffusion）只能处理 2D 边界框，无法表达 3D 属性

**朝向控制不足**：LOOSECONTROL 使用 3D 边界框但缺少朝向信息，同一边界框可能对应正面或背面物体

**多物体场景困难**：Continuous 3D Words 和 COMPASS 只能处理单物体且视觉风格不真实

**方法兼容性差**：ORIGEN 依赖一步生成模型，无法与主流多步扩散框架兼容

核心切入点：设计一种高效的 9D 姿态编码表示（CNOCS 地图），结合专门的数据集和两阶段训练来实现精确的多物体姿态控制。

## 方法详解

### 整体框架
SceneDesigner 在预训练的 Stable Diffusion 3.5 基础上引入 ControlNet 分支网络，接收编码 9D 姿态的 CNOCS 地图作为控制条件。采用两阶段训练：第一阶段在 ObjectPose9D 数据集上学习基本控制能力，第二阶段通过强化学习微调改善低频姿态的生成质量。推理时采用 Disentangled Object Sampling 处理多物体场景。

### 关键设计

1. **CNOCS 地图（Cuboid Normalized Object Coordinate System）**

    - 动机：传统 NOCS 需要精确 3D 形状（CAD 模型），对用户不友好
    - 简化：CNOCS 只使用长方体（cuboid）近似物体形状，保留了关键几何信息
    - 构建过程：(1) 对每个像素找到其在 3D 边界框表面的交点坐标 → (2) 从相机坐标系转换到物体坐标系 → (3) 用边界框尺寸归一化到 [-1,1] → (4) 用编码函数 f 映射为最终像素值
    - 变体：C-CNOCS（常量函数，如欧拉角）、I-CNOCS（恒等函数，直接用坐标）、S-CNOCS（球谐函数）。实验选择 I-CNOCS
    - 优势：用户只需在 3D 空间操作长方体 mesh 即可指定物体姿态，直观友好

2. **ObjectPose9D 数据集**

    - 基础数据：从 OmniNOCS（Objectron + Cityscapes 子集）中选取约 110K 图像，具有准确的姿态标注
    - 扩展数据：对 MS-COCO 大规模标注 9D 姿态（约 65K 样本）以丰富物体类别和场景多样性
    - 标注流程：筛选合适物体 → Orient Anything 估计朝向 → MoGe 重建 3D 点云 → 计算 3D 边界框 → 人工校验
    - 总计 125,486 个训练样本

3. **两阶段训练策略**

    - **第一阶段**：在 ObjectPose9D 上用 flow matching 损失训练 ControlNet 分支（45K 迭代），学习基本姿态控制
    - **第二阶段**：RL 微调（5K 迭代），解决低频姿态（如多数动物的背面视角）生成不佳的问题
    - 奖励函数设计：位置/大小奖励 $r_{ls}$（用 Grounding DINO 检测 IoU）+ 朝向奖励 $r_o$（用 Orient Anything 估计 KL 散度）
    - 使用 randomized truncated backpropagation 和 gradient checkpointing 降低显存开销

4. **Disentangled Object Sampling（DOS）**

    - 动机：多物体场景中模型难以将每个物体与其对应姿态正确关联，且存在生成不足和概念混淆
    - 方法：每个去噪步骤中组合多个噪声 latent，分别基于全局条件或单个物体条件采样，通过区域掩码在 latent 空间融合
    - 可与用户个性化权重（LoRA）结合实现参考主体的定制姿态控制

### 损失函数 / 训练策略
- 第一阶段：标准 flow matching 损失 $\|v_\theta(x_t, t, c_p, \{P_i\}) - (\epsilon - x)\|^2$
- 第二阶段：最小化 $-\beta r(x, c_p, \{P_i\}) + L_{prior}$，其中 $L_{prior}$ 为第一阶段损失用于稳定训练
- AdamW 优化器，学习率 5e-6，分辨率 512×512，batch size 48，6 张 A800

## 实验关键数据

### 主实验（姿态对齐精度）

| 基准 | 方法 | Acc_ls(%)↑ | mIoU(%)↑ | Abs.Err↓ | Acc@22.5°(%)↑ |
|------|------|-----------|----------|----------|---------------|
| Single-Front | C3DW | 2.02 | 19.61 | 50.01 | 60.32 |
| Single-Front | LOOSECONTROL | 23.89 | 27.12 | 87.26 | 23.08 |
| Single-Front | **SceneDesigner** | **50.20** | **57.21** | **13.23** | **89.47** |
| Single-Back | LOOSECONTROL | 24.36 | 30.49 | 132.26 | 7.05 |
| Single-Back | **SceneDesigner** | **52.56** | **60.66** | **17.47** | **83.33** |
| Multi | LOOSECONTROL | 14.85 | 22.58 | 147.42 | 4.80 |
| Multi | **SceneDesigner** | **47.16** | **52.16** | **23.14** | **80.79** |

### 消融实验

| 配置 | Acc_ls(%)↑ | mIoU(%)↑ | Abs.Err↓ | Acc@22.5°(%)↑ |
|------|-----------|----------|----------|---------------|
| 无 MS-COCO | 41.69 | 50.07 | 74.89 | 24.32 |
| 无 RL 微调 | 43.18 | 50.32 | 43.85 | 52.36 |
| C-CNOCS | 40.45 | 49.86 | 37.86 | 73.70 |
| Pose embedding | 32.51 | 40.73 | 49.65 | 47.15 |
| 仅文本描述 | 12.90 | 14.32 | 88.43 | 25.31 |
| **SceneDesigner** | **51.12** | **58.55** | **14.87** | **87.10** |

### 关键发现
- I-CNOCS 地图远优于直接注入姿态嵌入（Acc@22.5°: 87.1% vs 47.2%），证实了空间表示的有效性
- RL 微调将朝向精度从 52.36% 提升到 87.10%，特别改善了背面姿态生成
- MS-COCO 数据扩展将朝向精度从 24.32% 大幅提升到 87.10%，类别多样性至关重要
- DOS 在多物体场景中将所有指标均大幅提升（Acc_ls: 36.68% → 47.16%）
- 用户研究中 SceneDesigner 在图像质量（0.96）、朝向保真（0.91）方面远超竞争方法

## 亮点与洞察
- CNOCS 地图是一个优雅的设计：用长方体代替精确 3D 形状，降低了使用门槛同时保留了几何可解释性
- 两阶段训练策略直接对齐姿态控制目标，用 RL 解决数据不平衡问题比数据重采样更高效
- DOS 在推理时以额外计算换取多物体场景质量，思路直观有效
- 支持与 DreamBooth/LoRA 结合进行个性化姿态控制，实用性强

## 局限与展望
- 无法控制物体的精确形状（只能控制长方体包围盒）
- 多物体场景受限于基础模型能力，物体数量增多时仍有概念混淆
- DOS 引入额外计算开销，每个物体需要独立采样
- 当前仅支持图像生成，扩展到视频需要额外工作

## 相关工作与启发
- 与 LOOSECONTROL 相比，CNOCS 补充了关键的朝向信息
- RL 微调策略可推广到其他需要对齐控制条件的生成任务
- CNOCS 的变体设计（常量/恒等/球谐函数）提供了表示工程的设计空间参考

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BootComp: Controllable Human Image Generation with Personalized Multi-Garments](../../CVPR2025/image_generation/controllable_human_image_generation_with_personalized_multi-garments.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](../../ICCV2025/image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[ECCV 2024\] Linearly Controllable GAN: Unsupervised Feature Categorization and Decomposition for Image Generation and Manipulation](../../ECCV2024/image_generation/linearly_controllable_gan_unsupervised_feature_categorization_and_decomposition_.md)
- [\[NeurIPS 2025\] OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data](osmgen_highly_controllable_satellite_image_synthesis_using_openstreetmap_data.md)
- [\[AAAI 2026\] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](../../AAAI2026/image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)

</div>

<!-- RELATED:END -->
