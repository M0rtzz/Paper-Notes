---
title: >-
  [论文解读] Disco4D: Disentangled 4D Human Generation and Animation from a Single Image
description: >-
  [CVPR 2025][3D视觉][4D人体生成] Disco4D 提出将服装（用 Gaussian 模型表示）与人体（用 SMPL-X 模型表示）解耦的 4D 人体生成框架，从单张图像生成可动画、可编辑、分层的3D穿衣人体模型，并支持逼真的4D服装动力学。
tags:
  - CVPR 2025
  - 3D视觉
  - 4D人体生成
  - 服装解耦
  - 高斯溅射
  - 人体动画
  - 单图重建
---

# Disco4D: Disentangled 4D Human Generation and Animation from a Single Image

**会议**: CVPR 2025  
**arXiv**: [2409.17280](https://arxiv.org/abs/2409.17280)  
**代码**: [https://disco-4d.github.io/](https://disco-4d.github.io/)  
**领域**: 3D视觉  
**关键词**: 4D人体生成、服装解耦、高斯溅射、人体动画、单图重建

## 一句话总结

Disco4D 提出将服装（用 Gaussian 模型表示）与人体（用 SMPL-X 模型表示）解耦的 4D 人体生成框架，从单张图像生成可动画、可编辑、分层的3D穿衣人体模型，并支持逼真的4D服装动力学。

## 研究背景与动机

- 现有单图3D人体重建方法（如 PIFu、ECON）将人体和服装作为单层整体重建，无法支持虚拟试穿、服装编辑等应用
- 整体重建产生的单层不可动画网格使得重新动画和动态定制极为困难
- 此前没有从单图或少量图像进行4D分层人体生成和动画的工作
- 需要一种能分离人体和服装的表示，同时支持高保真生成、精细编辑和逼真动画

## 方法详解

### 整体框架

Disco4D 采用自底向上的构建方式：(1) 使用 SMPL-X 参数化模型表示人体；(2) 使用 3D Gaussian 表示服装；(3) 将服装 Gaussian 绑定到 SMPL-X 网格并迭代优化；(4) 通过身份编码实现服装类别的分离和提取。完整人体表示为 $S_{human} = S_{body} \cup S_{cloth}$。

### 关键设计

1. **SMPL-X Gaussian 人体表示**:
    - 功能：提供稳定的人体锚点结构
    - 核心思路：将扁平3D Gaussian 直接绑定到 SMPL-X 网格的每个三角面片上（类似 SuGaR），Gaussian 的位置 $\mu_{body}$ 由预定义重心坐标决定，旋转由面法线导出，不透明度固定为1.0
    - 设计动机：SMPL-X 擅长捕捉人体结构和运动学，固定人体表示可保持完整性，让学习过程聚焦于服装

2. **服装 Gaussian 的网格嵌入与可分离优化**:
    - 功能：将服装建模为独立于人体的层
    - 核心思路：每个服装 Gaussian 嵌入到 SMPL-X 规范网格的一个三角面片上，使用局部坐标系 $\mu = O + \sigma i + \beta j + \gamma k$ 定位；通过 SDF 损失和剪枝确保服装在人体外部；引入身份编码 $e \in \mathbb{R}^{15}$ 关联每个 Gaussian 与其服装类别
    - 设计动机：将 Gaussian 嵌入三角面片使其自然随 SMPL-X 变形进行动画；身份编码使不同服装可独立提取和编辑

3. **扩散模型增强的纹理补全与4D动画**:
    - 功能：补全遮挡区域纹理并学习服装动力学
    - 核心思路：使用 SDS 损失从扩散模型中蒸馏遮挡区域的纹理细节；4D动画通过变形网络 $S'' = \phi(S', t)$ 预测服装 Gaussian 的位置、旋转和缩放变化，人体由 SMPL-X 驱动，服装既跟随人体又展现自身动力学
    - 设计动机：单张图像不可避免有遮挡区域，扩散模型提供合理的先验；服装与人体的解耦动画使其更加逼真

### 损失函数 / 训练策略

总损失函数：$\mathcal{L} = \mathcal{L}_{ori} + \mathcal{L}_{id} + \mathcal{L}_{ani} + \mathcal{L}_{sdf} + \mathcal{L}_{SDS}$

- $\mathcal{L}_{ori}$: 标准3D Gaussian 渲染损失
- $\mathcal{L}_{id} = \mathcal{L}_{2d} + \mathcal{L}_{3d}$: 身份编码损失，交叉熵分类 + 3D近邻一致性正则
- $\mathcal{L}_{ani}$: 各向异性约束，防止变形时出现过细的Gaussian核
- $\mathcal{L}_{sdf}$: SDF损失，确保服装Gaussian在SMPL-X外部
- $\mathcal{L}_{SDS}$: Score Distillation Sampling 损失，用于遮挡区域纹理补全

## 实验关键数据

### 主实验

| 方法 | CLIP(All)↑ | CLIP(Pants)↑ | PSNR(NV)↑ | SSIM(NV)↑ | LPIPS(NV)↓ |
|------|-----------|-------------|----------|----------|-----------|
| Disco4D(CloSe) | **0.856** | **0.858** | 20.10 | 0.918 | 0.081 |
| LGM | 0.829 | 0.727 | **20.50** | **0.939** | **0.077** |
| DreamGaussian | 0.734 | 0.693 | 20.08 | 0.939 | 0.089 |
| SHERF | 0.777 | 0.785 | 18.96 | 0.912 | 0.083 |

### 消融实验

| 配置 | CLIP(All)↑ | Assets↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-----------|---------|-------|-------|--------|
| Disco4D (reposed + deform) | **0.900** | **0.865** | **25.46** | **0.96** | **0.035** |
| Disco4D (reposed) | 0.853 | 0.774 | 23.94 | 0.95 | 0.049 |
| DG4D (Disco4D init) | 0.870 | 0.849 | 21.02 | 0.93 | 0.065 |
| GaussianAvatar | 0.822 | 0.768 | 20.01 | 0.93 | 0.069 |

### 关键发现

- Disco4D 在服装解耦方面（CLIP Assets 分数）显著优于所有基线
- 学习服装变形后（+learned deformations），PSNR 从23.94提升至25.46，LPIPS 从0.049降至0.035
- 用户研究中 Disco4D 的图像一致性评分达3.142（5分制），远超 LGM 的2.338和 DreamGaussian 的2.017
- SMPL-X 作为身体锚点明显提升了面部和肢体的几何精度

## 亮点与洞察

- **首次实现单图4D分层人体生成**：同时支持人体-服装分离、动画和编辑
- **身份编码机制巧妙**：15维可学习向量 + 2D分割监督 + 3D近邻正则，实现精细的服装类别分离
- **非人体专用数据集训练**：利用通用扩散模型，无需人体特定数据集训练
- **丰富的编辑能力**：支持服装移除、换色、材质修改、跨人物服装迁移

## 局限与展望

- 初始 SMPL-X 估计的精度会影响后续所有步骤的质量
- 宽松服装（如长裙）和复杂配饰的建模仍具挑战性
- 扩散模型蒸馏的纹理质量在背面等遮挡严重的区域可能不理想
- 服装动力学学习依赖视频输入，单图模式下只能通过 SMPL-X 简单驱动

## 相关工作与启发

- 与 SHERF/ELICIT 等 NeRF 方法相比，Gaussian 表示在效率和编辑灵活性上显著优势
- SMPL-X + Gaussian 的分层表示思路可扩展到其他分层重建任务
- 身份编码的设计可启发更细粒度的场景分解方法
- 为虚拟试衣、数字时尚和内容创作提供了实用框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将人体-服装解耦引入单图4D生成，框架设计优雅
- 实验充分度: ⭐⭐⭐⭐ 多数据集定量评估+用户研究，但缺少更多真实场景demo
- 写作质量: ⭐⭐⭐⭐ 结构清晰、方法描述详细
- 价值: ⭐⭐⭐⭐⭐ 实际应用价值很高，为虚拟试衣等领域提供了新范式

<!-- RELATED:START -->

## 相关论文

- [WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)
- [Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)

<!-- RELATED:END -->
