---
title: >-
  [论文解读] Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes
description: >-
  [CVPR 2025][3D视觉][3D场景交互] 本文提出通过在3D重建场景中记录人手交互的动作-声音对，训练基于rectified flow的生成模型，实现从3D手部轨迹预测对应交互声音，生成结果在人类评估中约47%无法与真实声音区分。 领域现状：当前3D重建方法（如NeRF、Gaussian Splatting）主要聚…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D场景交互"
  - "声音生成"
  - "手部动作"
  - "Gaussian Splatting"
  - "Rectified Flow"
---

# Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes

**会议**: CVPR 2025  
**arXiv**: [2506.09989](https://arxiv.org/abs/2506.09989)  
**代码**: [https://www.yimingdou.com/hearing_hands/](https://www.yimingdou.com/hearing_hands/) (项目页面)  
**领域**: 3D视觉  
**关键词**: 3D场景交互, 声音生成, 手部动作, Gaussian Splatting, Rectified Flow

## 一句话总结

本文提出通过在3D重建场景中记录人手交互的动作-声音对，训练基于rectified flow的生成模型，实现从3D手部轨迹预测对应交互声音，生成结果在人类评估中约47%无法与真实声音区分。

## 研究背景与动机

**领域现状**：当前3D重建方法（如NeRF、Gaussian Splatting）主要聚焦于静态场景的视觉表示，部分工作开始探索场景的可交互性，如物体关节运动、变形等视觉动态，但这些方法只关注视觉层面的变化。

**现有痛点**：现有的交互式3D重建忽略了声音这一重要的交互模态。声音能够传达视觉无法直接表达的物理属性，比如表面是硬还是软、光滑还是粗糙、中空还是实心。另一方面，视频到音频的生成方法需要输入视频作为条件，无法让用户自由指定想要模拟的交互动作。

**核心矛盾**：如何在没有真实视频输入的情况下，仅通过用户指定的3D手部动作轨迹，就能预测在特定场景位置进行该动作时会产生的声音？这要求模型同时理解场景的材质属性和动作的时序特征。

**本文目标** (1) 构建一个将3D手部轨迹与场景视觉信息配对的声音数据集；(2) 训练一个能够从3D手部动作条件生成声音的模型；(3) 生成的声音需要准确反映材质属性和动作时序。

**切入角度**：作者观察到材质的视觉外观与其被物理操作时产生的声音之间存在关联。通过将手部动作参数化为3D手部姿态序列，并结合Gaussian Splatting场景重建提供的视觉信息，可以建立动作到声音的映射。

**核心 idea**：将3D手部轨迹和场景视觉特征作为条件，通过rectified flow生成模型预测交互声音的mel频谱图。

## 方法详解

### 整体框架

系统包含两个部分：(1) 视觉神经场 $F_\theta$ 使用Gaussian Splatting重建3D场景，将3D点映射为RGB和深度；(2) 动作条件声音生成器 $F_\phi$，接收场景视频 $\mathbf{v}$ 和手部动作 $\mathbf{a}$ 作为输入，生成对应声音 $\mathbf{s}$。训练时，先采集真人在场景中用手交互的视频，提取3D手部姿态并注册到场景坐标系，再利用场景重建合成不同视角的交互视频与声音配对。

### 关键设计

1. **3D手部-场景数据采集与增强管线**:

    - 功能：构建3D配准的手部动作-声音对数据集
    - 核心思路：先用Gaussian Splatting重建场景，然后记录人手在场景中交互的视频，用HaMeR检测3D手部关键点 $\mathbf{a} \in \mathbb{R}^{2N \times 21 \times 3}$，通过COLMAP将交互相机注册到场景坐标系。关键的是，将检测到的3D手部投影到场景重建的渲染视图上，生成"模拟交互视频"——包含全局视图（带手部叠加）和局部视图（以手部为中心裁剪）。这样可以去除人体遮挡，并通过合成不同视角（俯视、侧视）实现3D一致的数据增强。共采集24个场景、9.1小时交互数据。
    - 设计动机：直接使用原始视频会包含人体遮挡和视角限制，通过3D注册+重新渲染可以获得干净的视觉输入并进行多视角增强。

2. **基于Rectified Flow的声音生成模型**:

    - 功能：从视觉+动作条件生成声音频谱图
    - 核心思路：基于Frieren视频到声音模型，但做了两个关键修改。首先用CLIP替代CAVP编码视频特征，因为CLIP具有更好的空间一致性和材质理解能力。然后显式地将3D手部动作 $\mathbf{a}$ 作为额外条件注入模型——通过线性层编码手部姿态到与帧嵌入相同维度，归一化为单位向量，上采样到与频谱图相同的时间频率（31.25Hz），最后与视觉嵌入逐元素求和作为条件向量。
    - 设计动机：原始Frieren在合成交互视频上效果差，因为模拟视频缺乏真实视频的低层细节。CLIP提供材质信息，30Hz的手部姿态提供高分辨率的动作细节，两者互补。

3. **双流视觉编码（全局+局部视图）**:

    - 功能：同时捕获场景整体布局和交互区域的材质细节
    - 核心思路：将全局视频 $\mathbf{v}_g$（含手部叠加的场景渲染）和局部视频 $\mathbf{v}_l$（以手部中心裁剪的近景）分别通过CLIP编码，得到两个特征向量后拼接送入模型。视频帧以4Hz下采样送入CLIP。
    - 设计动机：全局视图提供手在场景中的位置和整体语境，局部视图提供手接触区域的材质纹理细节，两者结合才能准确判断声音类型。

### 损失函数 / 训练策略

使用rectified flow matching的标准训练目标。模型从零训练，40个epoch，batch size 128，Adam优化器，学习率从 $10^{-5}$ 热启动到 $4 \times 10^{-4}$，再线性衰减到 $3.4 \times 10^{-4}$。推理时进行26步采样，guidance scale 4.5，通过预训练vocoder将频谱图转换为波形。

## 实验关键数据

### 主实验

| 方法 | STFT ↓ | Envelope ↓ | CLAP-acc All ↑ | CLAP-acc Action ↑ | CLAP-acc Material ↑ | 真实率 (%) |
|------|--------|-----------|----------------|-------------------|---------------------|-----------|
| RegNet | 0.62 | 0.77 | 1.08 | 42.55 | 3.52 | - |
| Frieren | 0.74 | 0.81 | 23.94 | 41.73 | 42.55 | 43.79±2.64 |
| **Ours** | **0.50** | **0.66** | **28.09** | **50.50** | **45.62** | **47.18±2.66** |

### 消融实验

| 配置 | STFT ↓ | Envelope ↓ | CLAP-acc All ↑ | CLAP-acc Material ↑ | CLAP-acc Action ↑ |
|------|--------|-----------|----------------|---------------------|-------------------|
| Full model | 0.50 | 0.66 | 28.09 | 45.62 | 50.50 |
| w/o CLIP | 0.68 | 0.77 | 18.25 | **31.80** (-13.82) | 43.90 |
| w/o hand pose | 0.69 | 0.77 | 20.96 | 39.11 | **38.21** (-12.29) |
| w/o synthetic-view | 0.62 | 0.73 | 24.12 | 40.56 | 47.61 |

### 关键发现

- 去掉CLIP特征导致CLAP材质准确率下降最大（45.62→31.80），说明CLIP主要提供材质信息
- 去掉手部姿态导致CLAP动作准确率下降最大（50.50→38.21），说明手部姿态主要编码动作时序
- 真实-伪造实验中，本文方法47.18%的误分类率接近50%随机基线，说明生成声音几乎无法与真实声音区分
- 在粗糙表面和柔软材质上优势尤为明显

## 亮点与洞察

- **单步声音提取的3D场景交互**：将声音生成与3D重建结合，让用户可以通过指定手部轨迹"试听"场景中不同物体的声音，这是一个新颖的AR/VR应用方向。巧妙之处在于用已有的手部检测+场景重建技术构建了一个完全自动化的数据采集管线。
- **3D一致数据增强**：利用3D重建消除人体遮挡并合成多视角训练数据，这种利用3D几何先验做数据增强的思路可以迁移到其他需要视角多样性的任务中。
- **CLIP替代CAVP的洞察**：发现通用视觉-语言模型（CLIP）在材质理解上优于专门的音视频对齐模型（CAVP），这暗示了材质理解更依赖视觉语义而非音视频对齐。

## 局限与展望

- 假设场景中物体在被操作时不移动或变形，这在操作小物体时经常被违反
- 依赖3D手部检测模型的精度，检测误差会导致数据集中的手部动作不准确
- 仅在24个场景上训练，泛化到完全不同的场景和材质可能受限
- 未考虑声音的空间传播效应（如混响、距离衰减），可结合声学重建工作

## 相关工作与启发

- **vs ObjectFolder**: ObjectFolder构建物体级多模态表示但只能处理刚体小物体和简单碰撞声音，本文处理场景级重建且支持复杂手部动作
- **vs Diff-Foley/Frieren**: 它们从视频生成声音，本文从3D手部轨迹生成声音，不需要视频输入，且通过3D约束获得更清晰的材质视图
- **vs Tactile-augmented radiance fields**: 同一作者之前的工作处理触觉，这里处理声音，且声音不是表面的固有属性而是动作的函数

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将声音生成引入3D交互场景重建，新问题formulation
- 实验充分度: ⭐⭐⭐⭐ 自动化指标+大规模人类评估，消融设计合理
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，pipeline描述详尽
- 价值: ⭐⭐⭐⭐ 在AR/VR和机器人领域有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[ICCV 2025\] Bolt3D: Generating 3D Scenes in Seconds](../../ICCV2025/3d_vision/bolt3d_generating_3d_scenes_in_seconds.md)
- [\[CVPR 2025\] Functionality Understanding and Segmentation in 3D Scenes](functionality_understanding_and_segmentation_in_3d_scenes.md)
- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)

</div>

<!-- RELATED:END -->
