---
title: >-
  [论文解读] SyncHuman: Synchronizing 2D and 3D Generative Models for Single-View Human Reconstruction
description: >-
  [NeurIPS 2025][3D视觉][单视图人体重建] SyncHuman首次将2D多视图生成模型与3D原生生成模型统一在一个框架中，通过像素对齐的2D-3D同步注意力机制实现互补增强，在复杂人体姿态下实现了高保真纹理网格重建，几何精度和视觉质量均超越现有方法。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 单视图人体重建
  - 2D-3D生成模型同步
  - 多视图生成
  - 3D高斯溅射
  - 纹理网格
---

# SyncHuman: Synchronizing 2D and 3D Generative Models for Single-View Human Reconstruction

**会议**: NeurIPS 2025  
**arXiv**: [2510.07723](https://arxiv.org/abs/2510.07723)  
**代码**: [GitHub](https://xishuxishu.github.io/SyncHuman.github.io)  
**领域**: 3D视觉 / 人体重建  
**关键词**: 单视图人体重建, 2D-3D生成模型同步, 多视图生成, 3D高斯溅射, 纹理网格

## 一句话总结

SyncHuman首次将2D多视图生成模型与3D原生生成模型统一在一个框架中，通过像素对齐的2D-3D同步注意力机制实现互补增强，在复杂人体姿态下实现了高保真纹理网格重建，几何精度和视觉质量均超越现有方法。

## 研究背景与动机

从单张RGB图像重建3D穿衣人体是AR/VR、虚拟试穿、游戏和影视制作中的关键任务。现有方法主要依赖SMPL体型估计来提供结构信息，再利用条件多视图生成模型来补全未见视角。然而这种范式存在两个核心矛盾：

**SMPL估计不准确**：尤其在遮挡和复杂姿态下，单视图姿态估计方法精度不足，且SMPL网格仅表示裸体，无法准确建模宽松服装

**2D与3D生成各有软肋**：多视图生成模型擅长捕获精细2D细节但跨视图一致性差；3D原生生成模型结构一致性好但细节粗糙、保真度低

核心idea：**让2D和3D生成模型相互监督——3D模型为2D提供结构引导以改善多视图一致性，2D模型为3D注入精细纹理以提升保真度**。

## 方法详解

### 整体框架

给定单张全身人体图像，SyncHuman分两阶段工作：
- **阶段一**：2D-3D跨空间生成模型同时生成多视图彩色+法线图和对齐的稀疏3D体素网格
- **阶段二**：多视图引导解码器将结构化隐变量解码为高质量纹理网格

### 关键设计

1. **2D-3D跨空间生成模型**:

    - **多视图生成分支**：基于PSHuman架构，以输入图像为前视图，生成前/后/左/右四个正交视角的彩色图和法线图，采用行级多视图注意力增强跨视图一致性
    - **3D结构生成分支**：基于Trellis架构，用DiT-based flow transformer从3D噪声网格生成稀疏结构隐变量，再解码为体素占用网格
    - **2D-3D同步注意力**：这是核心创新。包含两个方向的交叉注意力：
        - *2D→3D*：每个3D体素特征正交投影到四个视角平面获取对应2D特征，用体素特征作query、2D特征作key/value进行交叉注意力，用零初始化MLP处理后残差连接
        - *3D→2D*：每个2D多视图特征投影到3D空间中查询对应的体素列特征，用2D特征作query、体素特征作key/value进行交叉注意力

2. **联合训练策略**:

    - 将SD 2.1重新适配为与Trellis相同的flow matching框架
    - 使用flow matching目标联合训练2D和3D分支，损失函数为两个分支的速度预测误差之和

3. **多视图引导解码器 (MVGD)**:

    - 在Trellis的结构化隐变量生成之后，引入多视图特征注入机制
    - 提取生成多视图图像的DINOv2特征，经可训练MLP处理
    - 对每个3D体素查询对应四视图特征，与结构化隐变量拼接后送入原始解码器
    - 支持网格分支和3DGS分支分别训练，最终将3DGS渲染颜色烘焙到网格上

### 损失函数 / 训练策略

- **生成阶段**：2D和3D分支各自的flow matching loss联合优化
- **3DGS解码**：L1 + SSIM + LPIPS + 不透明度正则
- **网格解码**：前景掩码、深度图、法线图的L1/Huber损失
- 训练数据：THuman2.1、CustomHumans、THuman3.0、2K2K等3D人体扫描数据集
- 渲染分辨率：768×768，8个正交相机等距分布

## 实验关键数据

### 主实验

| 数据集 | 指标 | SyncHuman | PSHuman (之前SOTA) | 提升 |
|--------|------|-----------|-------------------|------|
| X-Humans | Chamfer Dist↓ | **0.835** | 1.438 | -42% |
| X-Humans | P2S↓ | **0.759** | 1.138 | -33% |
| X-Humans | NC↑ | **0.887** | 0.839 | +5.7% |
| X-Humans | PSNR↑ | **21.84** | 20.84 | +1.0 |
| X-Humans | LPIPS↓ | **0.079** | 0.098 | -19% |
| CAPE-NFP | Chamfer Dist↓ | **0.913** | 1.373 | -33% |
| CAPE-FP | Chamfer Dist↓ | **0.641** | 0.776 | -17% |

### 消融实验

| 配置 | PSNR↑ | Chamfer↓ | NC↑ | 说明 |
|------|-------|---------|-----|------|
| Trellis原始 | 17.08 | 2.004 | 0.772 | 3D原生模型细节差 |
| Trellis微调 | 20.34 | 1.135 | 0.848 | 微调后改善明显 |
| PSHuman | 20.84 | 1.438 | 0.839 | 2D模型几何弱 |
| SyncHuman完整 | **21.84** | **0.835** | **0.887** | 2D+3D互补最优 |

| 解码器配置 | PSNR↑ | Chamfer↓ | 说明 |
|-----------|-------|---------|------|
| 原始Trellis解码器 | 21.08 | 0.895 | 缺乏细节 |
| 微调Trellis解码器 | 21.36 | 0.887 | 略有改善 |
| MVGD (多视图引导) | **21.84** | **0.835** | 显著提升 |

### 关键发现

- 2D-3D同步注意力同时改善几何精度和纹理质量，验证了2D与3D互补的核心假设
- MVGD的多视图特征注入虽然简单，但能有效将高分辨率细节"雕刻"到3D形状上
- 相比SMPL估计，3D原生生成模型在复杂姿态下更鲁棒，避免了自交叉等伪影

## 亮点与洞察

- **2D-3D互补范式**：首次证明将像素级2D生成与体素级3D生成统一在同一去噪过程中可行且有效，为未来3D生成模型提供了新方向
- **零初始化注意力**：同步注意力层使用零初始化MLP，允许从预训练模型平滑过渡，训练更稳定
- **免SMPL依赖**：不再依赖人体姿态先验，直接利用3D生成模型提供结构信息，大幅提升复杂姿态下的鲁棒性
- **Flow matching统一训练**：将SD 2.1和Trellis统一到相同的flow matching框架下联合训练

## 局限与展望

- 训练数据仅约5000个扫描，规模较小，在极端光照条件下纹理质量下降
- 多视图生成基于SD 2.1微调，生成质量仍受限于基础模型能力
- 未来方向：使用视频生成模型或大规模多视图人体数据集进行扩展

## 相关工作与启发

- **vs PSHuman**: PSHuman仅用2D多视图扩散生成+重网格化，几何一致性差；SyncHuman添加3D分支协同改善
- **vs Trellis**: Trellis作为3D原生模型细节不足，SyncHuman通过2D多视图特征注入恢复细节
- **vs Human3Diff**: Human3Diff尝试在去噪过程中引入3D约束但仍基于2D去噪模型，SyncHuman则是真正的2D-3D联合生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将2D多视图与3D原生生成模型同步训练，idea清晰且有效
- 实验充分度: ⭐⭐⭐⭐ 三个数据集全面评估，消融实验充分，但缺少对更多3D生成基线的比较
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表质量高，方法描述详尽
- 价值: ⭐⭐⭐⭐⭐ 为3D人体重建和通用3D生成提供了有价值的2D-3D联合建模范式

<!-- RELATED:START -->

## 相关论文

- [Orientation Matters: Making 3D Generative Models Orientation-Aligned](orientation_matters_making_3d_generative_models_orientation-aligned.md)
- [ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [GAS: Generative Avatar Synthesis from a Single Image](../../ICCV2025/3d_vision/gas_generative_avatar_synthesis_from_a_single_image.md)

<!-- RELATED:END -->
