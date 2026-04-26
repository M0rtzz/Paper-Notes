---
title: >-
  [论文解读] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation
description: >-
  [ICCV 2025][3D视觉][自中心运动] 提出 UniEgoMotion，首个统一的自中心运动模型，通过条件运动扩散框架和头部中心运动表示，在单一模型中实现自中心视角下的3D人体运动重建、预测和生成三项任务，并发布大规模EE4D-Motion数据集。
tags:
  - ICCV 2025
  - 3D视觉
  - 自中心运动
  - 扩散模型
  - 运动重建
  - 运动预测
  - 运动生成
  - 头部中心表示
---

# UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation

**会议**: ICCV 2025  
**arXiv**: [2508.01126](https://arxiv.org/abs/2508.01126)  
**代码**: [UniEgoMotion](https://chaitanya100100.github.io/UniEgoMotion/)  
**领域**: 3d_vision  
**关键词**: 自中心运动, 扩散模型, 运动重建, 运动预测, 运动生成, 头部中心表示

## 一句话总结

提出 UniEgoMotion，首个统一的自中心运动模型，通过条件运动扩散框架和头部中心运动表示，在单一模型中实现自中心视角下的3D人体运动重建、预测和生成三项任务，并发布大规模EE4D-Motion数据集。

## 研究背景与动机

**自中心人体运动理解**（从头戴设备的第一人称视角）对AR/VR、辅助技术和医疗健康至关重要，但面临三个核心挑战：

**自中心视角的固有困难**：头戴前向摄像头只能看到用户身体的极小部分，模型必须从动态的第一人称视角、频繁遮挡和运动模糊中推断全身运动。现有研究几乎未探索自中心运动预测和生成。

**场景上下文的忽视**：大多数现有方法依赖显式3D场景表示（点云、体素、SDF等），这在真实自中心应用中不可用。前人工作通常丢弃自中心图像中的语义信息，仅利用设备轨迹，导致在头部运动较少的活动（如烹饪、演奏乐器）中表现欠佳。

**任务孤立的模型设计**：运动重建、预测和生成被割裂为独立任务分别建模，缺乏统一框架来共享运动先验和场景理解能力。

**核心洞察**：自中心图像本身包含丰富的场景语义信息，无需显式3D场景重建即可推断合理的3D运动。通过条件扩散模型的灵活条件设计，可以在统一框架中实现三种自中心运动任务。

## 方法详解

### 整体框架

UniEgoMotion是一个基于Transformer的条件运动扩散模型。输入包含自中心视频帧 $\boldsymbol{I}_{1:N}$、6-DoF设备轨迹 $\boldsymbol{T}_{1:N}$ 和噪声运动 $\boldsymbol{X}^t_{1:N}$。核心思想是通过不同的条件组合支撑三种任务：

- **重建**: $\boldsymbol{C} = \{\boldsymbol{T}_{1:N}, \boldsymbol{I}_{1:N}\}$（全条件）
- **生成**: $\boldsymbol{C} = \{I_1\}$（单张图像，其余用可学习mask替代）
- **预测**: $\boldsymbol{C} = \{\boldsymbol{T}_{1:n}, \boldsymbol{I}_{1:n}\}$（部分观测，通过diffusion inpainting续写未来运动）

训练时随机mask条件输入实现三种任务的统一训练；推理时通过填充可学习mask token支持灵活推理。

### 扩散建模

采用标准条件扩散框架，目标为直接预测干净运动（x-prediction）：

$$\mathcal{L} = \mathbb{E}_{t, \boldsymbol{X}^t \sim q_t(\cdot|\boldsymbol{X})} \left[ \|\boldsymbol{X} - \mathcal{M}(\boldsymbol{X}^t, t, \boldsymbol{C})\|_2^2 \right]$$

预测任务使用diffusion repainting策略：先重建已观测运动 $\boldsymbol{X}_{1:n}$，然后在每步去噪时覆盖已知帧以确保一致性。

### 架构设计

- **运动输入**：每帧SMPL-X参数通过线性层投影为潜向量 $f_X(X_i)$
- **轨迹条件**：线性投影 $f_T(T_i)$ 与运动嵌入相加
- **图像条件**：使用预训练DINOv2初始化的ViT编码器 $f_I(I_i)$，通过cross-attention注入场景上下文
- **Transformer解码器**：多层解码器层处理运动token，cross-attend图像特征

关键设计选择：DINOv2的细粒度特征显著优于CLIP和EgoVideo编码器，表明精确的场景上下文比自中心动作语义更重要。

### 头部中心运动表示

传统骨盆中心的SMPL-X参数 $(R^r_i, t^r_i, \theta_i, \beta_i)$ 存在两个问题：(1) 与自中心设备位置不匹配；(2) 局部关节角度需要学习复杂的正向运动学。

UniEgoMotion提出头部中心表示：
1. 通过正向运动学计算所有关节的全局SE(3)变换
2. 消除pitch/roll/height得到规范参考帧 $_cM_i$（头部轨迹在地面上的投影）
3. 运动表示为 $(_cM_i, \; _cM_i \odot M^h_i, \; _cM_i \odot \boldsymbol{M}^j_i)$
4. 轨迹表示为相邻帧的残差，实现平移不变性

此表示消除了运动学链中的关节依赖，显著减少了脚部穿透地面和浮空伪影。

### EE4D-Motion数据集

从大规模EgoExo4D数据集处理而来，通过多视角优化和序列级平滑的SMPL-X拟合流水线，生成110+小时的同步自中心视频-3D运动数据对，覆盖真实世界活动。训练集143K样本，评估集4400样本。

## 实验

### 主实验：自中心运动重建

| 方法 | MPJPE↓ | MPJPE-PA↓ | MPJPE-H↓ | Foot Slide↓ | Semantic Sim.↑ | FID↓ |
|------|--------|-----------|----------|-------------|----------------|------|
| AvatarPoser | 0.116 | 0.068 | 0.240 | 7.85 | 0.872 | 0.082 |
| EgoEgo | 0.130 | 0.075 | 0.272 | 3.90 | 0.858 | 0.068 |
| EgoAllo | 0.163 | 0.071 | 0.273 | 4.10 | 0.885 | 0.043 |
| **UniEgoMotion** | **0.100** | **0.053** | **0.180** | **3.62** | **0.918** | **0.027** |

UniEgoMotion在所有指标上全面领先，MPJPE降低13.8%（vs AvatarPoser），语义相似度提升3.3%。

### 消融实验：运动预测与生成

| 方法 | Forecasting MPJPE(2-4s)↓ | Foot Slide↓ | FID↓ | Generation MPJPE(0-2s)↓ | Foot Slide↓ | FID↓ |
|------|--------------------------|-------------|------|--------------------------|-------------|------|
| LSTM | 0.238 | 7.23 | 0.058 | 0.216 | 6.83 | 0.090 |
| Two-stage | 0.253 | 3.55 | 0.038 | 0.222 | 4.35 | 0.037 |
| **UniEgoMotion** | **0.206** | **2.60** | 0.047 | 0.226 | **2.89** | 0.043 |
| Pelvis-centric | 0.245 | 3.56 | 0.042 | 0.232 | 3.94 | 0.039 |
| Global Repre. | 0.293 | 3.16 | 0.046 | 0.228 | 3.65 | 0.035 |
| CLIP encoder | 0.214 | 3.75 | 0.043 | 0.238 | 3.57 | 0.037 |

**关键发现**：
- LSTM确定性预测均值运动导致严重脚部滑动(7.23)，UniEgoMotion的Foot Slide仅2.60
- 头部中心表示 vs 骨盆中心表示：Foot Slide从3.56降至2.60，消除浮空和穿透伪影
- DINOv2编码器的细粒度特征优于CLIP和EgoVideo，表明场景上下文理解比自中心动作特征更关键
- Transformer解码器架构优于编码器架构和1D-UNet，验证了cross-attention在灵活条件设定中的优势

## 亮点与洞察

1. **统一框架的优雅设计**：通过条件mask策略和diffusion repainting，在单一模型中无缝支持重建/预测/生成三种任务，共享运动先验
2. **首次定义自中心运动预测与生成任务**：将场景感知运动建模扩展到更实际的自中心设置，无需显式3D场景输入
3. **头部中心表示的创新**：与自中心设备自然对齐，消除运动学链依赖，大幅减少物理不合理的伪影
4. **图像编码器的重要性**：DINOv2的细粒度特征优于文本对齐的CLIP和自中心专用的EgoVideo，揭示了场景上下文的精细理解比高层语义更重要

## 局限性

- EE4D-Motion数据集基于伪ground-truth（通过多视角优化拟合），运动精度受限于SMPL-X拟合质量
- 预测和生成任务中长时间序列（>2s）的评估困难，因多种合理运动均可能偏离ground-truth
- 未探索文本条件的运动生成，限制了交互性
- 单帧图像生成任务的motion多样性和场景推理能力仍有提升空间

## 相关工作

- **场景感知运动生成**：基于3D场景（点云/体素/SDF）的运动生成方法，仅一项工作从宽场景RGB图像生成运动
- **自中心运动重建**：EgoEgo、EgoAllo、AvatarPoser等，多数丢弃图像语义仅用设备轨迹
- **运动预测**：传统方法（MLP/RNN/GCN/Transformer）和扩散模型方法

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首次统一三种自中心运动任务，头部中心表示设计巧妙
- **技术质量**: ⭐⭐⭐⭐ — 方法简洁有效，消融充分，伪GT数据有局限
- **实验充分度**: ⭐⭐⭐⭐ — 三任务全面评估，多指标覆盖，消融细致
- **表达清晰度**: ⭐⭐⭐⭐⭐ — 问题定义清晰，soccer类比生动易懂
- **综合评分**: 8.5/10

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](../../CVPR2025/3d_vision/a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [\[NeurIPS 2025\] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](../../NeurIPS2025/3d_vision/gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](../../CVPR2025/3d_vision/hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)

<!-- RELATED:END -->
