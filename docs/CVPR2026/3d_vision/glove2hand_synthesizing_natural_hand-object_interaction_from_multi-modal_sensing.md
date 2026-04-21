---
title: >-
  [论文解读] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves
description: >-
  [CVPR 2026][3D视觉][手物交互] 提出 Glove2Hand 框架，将佩戴传感手套的第一人称视频翻译为逼真的裸手视频，同时保留触觉和 IMU 信号，并构建了首个多模态手物交互数据集 HandSense，显著提升下游裸手接触估计和遮挡手部追踪性能。
tags:
  - CVPR 2026
  - 3D视觉
  - 手物交互
  - 传感手套
  - 视频翻译
  - 3D高斯手模型
  - 扩散模型
---

# Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves

**会议**: CVPR 2026  
**arXiv**: [2603.20850](https://arxiv.org/abs/2603.20850)  
**代码**: https://mlzxy.github.io/glove2hand  
**领域**: 3D视觉  
**关键词**: 手物交互、传感手套、视频翻译、3D高斯手模型、扩散模型

## 一句话总结
提出 Glove2Hand 框架，将佩戴传感手套的第一人称视频翻译为逼真的裸手视频，同时保留触觉和 IMU 信号，并构建了首个多模态手物交互数据集 HandSense，显著提升下游裸手接触估计和遮挡手部追踪性能。

## 研究背景与动机

**领域现状**：手物交互（HOI）理解是计算机视觉、机器人和 AR/VR 的基础问题。当前主流方法通过采集第一人称视频来开发数据驱动算法，但这些系统基本只依赖视觉模态。

**现有痛点**：纯视觉 HOI 数据存在两大根本性缺陷：(1) 缺乏力和接触等物理量信息，现有方法如 ContactPose 只能估计二值指尖接触且仅适用于预扫描刚体；(2) 有限视角导致严重的手部遮挡，多相机工作室方案在野外不可行。传感手套虽能提供 IMU 和触觉信号，但手套与裸手之间巨大的外观差异使得在手套数据上训练的视觉模型无法泛化到裸手任务。

**核心矛盾**：传感手套提供了丰富的物理信号但引入了domain gap，而裸手视频有好的视觉但缺少物理信息——这两者之间存在无法调和的矛盾。

**本文目标** 如何将传感手套视频翻译为逼真裸手视频，同时保留触觉/IMU，使物理信号能用于裸手学习任务？具体子问题包括：(1) 实现跨帧时空一致性（而非只处理静态图像）；(2) 处理与未知/非刚性物体的复杂交互。

**切入角度**：关键观察是尽管手套和裸手外观差异巨大，但两者共享相同的关节结构（hand pose）。因此可以将问题分解为两步：先将手套视频变成一致的空中裸手序列（用3D重建保证一致性），再将裸手嵌入场景并修复交互细节（用扩散模型保证灵活性）。

**核心 idea**：结合3D高斯手模型的时空一致性和扩散手部修复器的生成灵活性，实现传感手套到裸手的视频翻译，同时保留多模态传感信号。

## 方法详解

### 整体框架
输入是佩戴传感手套的第一人称视频及其对应的手部姿态、手套/物体mask。输出是外观与裸手一致的高质量视频。流程分为两个阶段：(1) 3D 高斯手模型从手部姿态渲染出时空一致的纯手序列；(2) 扩散手部修复器将渲染的手无缝融合到场景中，修复手物交互边界和腕部连接。训练阶段使用裸手视频学习，推理阶段利用光流背景修复器先擦除手套区域再叠加渲染手。

### 关键设计

1. **Surface-Grounded 3D Gaussian Hand**:

    - 功能：从给定手部姿态渲染出时空一致、可重光照的裸手图像
    - 核心思路：将 3D 高斯分布直接定义在规范手部网格的三角面片表面上，每个高斯由重心坐标权重 $\mathbf{w}$、2D 尺度 $\mathbf{s}$ 和旋转 $\phi$ 参数化。手部变形时只变换网格三角形再重新计算高斯，避免了对每个高斯单独做线性蒙皮。通过 2D 仿射变形梯度 $\mathbf{A}=\mathbf{M}_{\text{deform}}\mathbf{M}_{\text{canon}}^{-1}$ 映射高斯椭圆。相比 2DGS 在3D空间定义高斯再正则化成面，本方法直接利用网格面作为更强先验
    - 设计动机：规范网格提供强几何先验但缺乏学习灵活性，高斯溅射灵活但缺乏结构。面片锚定统一了两者优势，且网格表面法线自然支持光照估计

2. **Relightable Hand Gaussians**:

    - 功能：处理自中心场景中动态变化的光照条件
    - 核心思路：用小型 MLP 根据手部姿态 $\mathbf{P}$ 预测球谐系数 $\mathbf{l}$，将颜色计算为固有色(albedo)与光照的乘积 $\mathbf{c}\odot\text{SH}(\mathbf{l},\mathbf{n})$。分别为手掌和手背预测两套独立环境贴图。由于法线来自网格几何而非高斯本身，大幅缓解了albedo-illumination模糊问题
    - 设计动机：LumiGauss 假设单一静态环境贴图，不适用于egocentric场景的动态光照；面片定义的高斯提供了一致的法线来源

3. **Diffusion Hand Restorer**:

    - 功能：将渲染的纯手序列无缝融合到场景中，修复手物交互和腕部连接
    - 核心思路：基于 ControlNet + AnimateDiff，在裸手视频上训练。将渲染手叠加到原始帧（膨胀mask、遮挡腕部区域）作为条件输入，网络学会从这种"损坏"输入恢复原始视频。推理时先用 SAM-2 + Grounding DINO 检测手套/物体mask，用 Propainter 光流修复擦除手套区域，保留物体像素，再叠加渲染手送入扩散修复器
    - 设计动机：直接叠加渲染手会产生穿透/悬浮等不合理交互、不自然腕部衔接、以及手套残留伪影。用扩散模型在像素域处理物体和背景比显式建模物体几何更灵活

### 损失函数 / 训练策略
3D 高斯手模型使用图像重建损失通过可微渲染训练，每个受试者单独优化一个模型。冻结受试者特定高斯手后，训练统一的扩散手部修复器。训练数据来自 HOT3D 和 HandSense。

## 实验关键数据

### 主实验

| 方法 | FID ↓ | FVD ↓ | FVD-long ↓ |
|------|-------|-------|------------|
| HandRefiner | 35.5 | 24.2 | 29.7 |
| BrushNet | 37.9 | 34.5 | 40.4 |
| Pix2Pix | 38.6 | 24.7 | 31.4 |
| **Glove2Hand (Ours)** | **30.1** | **19.5** | **24.5** |

### 消融实验

| 配置 | FID ↓ | FVD ↓ | FVD-long ↓ |
|------|-------|-------|------------|
| 2DGS | 91.1 | 50.0 | 62.9 |
| +Surface Grounding | 60.3 | 35.1 | 46.6 |
| +Relightable | 56.7 | 30.7 | 40.2 |
| +Diffusion | 32.3 | 19.8 | 22.7 |
| +Glove Removal | 31.2 | 20.9 | 25.0 |
| +Object Mask (Full) | 30.1 | 19.5 | 24.5 |

### 下游任务：接触估计

| 训练数据 | Contact IoU (%) | Precision (%) | Recall (%) |
|----------|----------------|---------------|------------|
| Glove only | 71.5 | 82.8 | 83.9 |
| G2H only | 75.6 | 90.6 | 82.0 |
| Hand only | 85.3 | 90.0 | 94.2 |
| **Hand + G2H** | **88.2** | **92.6** | **94.9** |

### 下游任务：遮挡手部追踪

| 方法 | MKPE (Occ) ↓ | MKPE (All) ↓ |
|------|-------------|-------------|
| UmeTrack | 19.2 | 19.5 |
| UmeTrack + Glove | 27.2 | 26.5 |
| **UmeTrack + G2H** | **16.6** | **17.8** |

### 关键发现
- Surface Grounding 和 Diffusion Restorer 贡献最大，分别带来 FID 从 91.1→60.3 和 56.7→32.3 的大幅下降
- 直接在手套数据上训练手部追踪器反而恶化性能（19.5→26.5），证实domain gap的严重性
- 合成裸手视频与真实裸手数据结合训练接触估计器达到最佳效果，验证了框架作为数据生成引擎的价值
- 人类评估显示生成的手在静态图像中几乎无法与真实手区分

## 亮点与洞察
- 将硬件传感（触觉+IMU）与视觉生成对齐，开辟了"传感器作为免标注ground-truth"的数据生成范式，这个思路可迁移到其他需要昂贵标注的领域
- 面片锚定高斯的设计非常优雅：一个简单的几何先验同时解决了时空一致性、重光照和变形三个问题，且实现极简
- 从 unpaired 数据学习 glove-to-hand 映射，避免了配对数据的采集成本

## 局限与展望
- 每个受试者需要单独优化一个3D高斯手模型，限制了直接推广到新用户
- 依赖 SAM-2 和 Grounding DINO 的自动化流水线，可能在极端遮挡或复杂背景下失败
- 数据集规模有限（5个受试者），泛化性还需更大规模验证
- 非刚性物体交互虽然在像素域灵活处理，但缺乏物理合理性保证（如力一致性）

## 相关工作与启发
- **vs HandRefiner**: HandRefiner 是扩散独有方法聚焦单帧手部修复，本文结合3D重建保证时空一致性，FID 低 5.4 点
- **vs 手部Avatar方法 (HandSplat等)**: 传统Avatar需要密集多视角和受控光照，本文的surface-grounded设计适应稀疏egocentric相机和动态光照
- **vs MeDM等视频翻译方法**: 通用视频翻译无法处理手套-裸手之间巨大的 embodiment 差异，本文利用共享关节结构解耦问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 面片锚定高斯+扩散修复的组合架构新颖，但各组件单独看是已有技术的良好整合
- 实验充分度: ⭐⭐⭐⭐⭐ 视频质量评估+人类评估+两个下游任务+完整消融，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法动机链完整，图示直观
- 价值: ⭐⭐⭐⭐ 为HOI领域提供了新的数据生成范式，HandSense数据集有长期价值

<!-- RELATED:START -->

## 相关论文

- [SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/3d_vision/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)
- [HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](../../CVPR2025/3d_vision/hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)
- [Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting](../../ICCV2025/3d_vision/contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)

<!-- RELATED:END -->
