---
title: >-
  [论文解读] RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models
description: >-
  [ECCV 2024][图像生成][3D头像生成] 提出RodinHD，解决triplane解码器的灾难性遗忘问题，并通过层级化肖像表示注入实现高保真3D头像生成。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "3D头像生成"
  - "高保真"
  - "扩散模型"
  - "triplane"
  - "灾难性遗忘"
---

# RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2407.06938](https://arxiv.org/abs/2407.06938)  
**代码**: 有 (项目页面)  
**领域**: Image Generation  
**关键词**: 3D头像生成, 高保真, 扩散模型, triplane, 灾难性遗忘

## 一句话总结

提出RodinHD，解决triplane解码器的灾难性遗忘问题，并通过层级化肖像表示注入实现高保真3D头像生成。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：从单张肖像图片生成高保真的3D头像（avatar）是计算机图形学和计算机视觉交叉领域的重要问题。高质量的3D头像在虚拟现实、社交媒体、游戏、远程会议等场景中有广泛应用。

现有方法（如Rodin、DreamFusion等）利用扩散模型生成3D表示（如triplane或NeRF），但在生成细节（特别是发型等精细结构）方面仍不尽如人意。作者识别了一个被忽视的关键问题——**灾难性遗忘**。

具体来说，triplane-based 3D生成方法通常使用一个共享的MLP解码器将triplane特征解码为颜色和密度。当在大量不同头像上顺序训练（fitting）时，MLP解码器会遗忘之前学到的知识，导致渲染质量下降。这类似于持续学习中的灾难性遗忘问题，但在3D生成领域尚未被充分认识。

此外，现有方法在利用输入肖像图像引导3D生成时，通常只提取全局特征（如CLIP embedding），忽略了丰富的2D纹理细节。

## 方法详解

### 整体框架

RodinHD在以下pipeline上进行改进：(1) 从肖像图像中提取多尺度、层级化的视觉表示；(2) 使用3D扩散模型在triplane空间中生成3D头像；(3) 通过改进的MLP解码器将triplane渲染为最终图像。核心改进集中在解码器的遗忘问题和肖像特征的注入方式。

### 关键设计

1. **数据调度策略与权重固化正则化（Data Scheduling + Weight Consolidation）**:
    - 功能：解决MLP解码器在顺序fitting时的灾难性遗忘问题
    - 核心思路：(a) 数据调度策略：不再按顺序处理每个头像，而是设计一种动态的训练数据调度方案，混合不同头像的训练数据，避免模型在单一头像上过度拟合而遗忘其他头像。(b) 权重固化正则化（类似EWC）：在更新解码器参数时，对重要参数施加正则化约束，限制参数变化幅度
    - 设计动机：标准的顺序fitting导致解码器参数在新头像上偏移，正则化和数据调度有效缓解了这一问题

2. **层级化肖像表示注入（Hierarchical Portrait Representation Injection）**:
    - 功能：更充分地利用输入肖像图像指导3D生成
    - 核心思路：从输入肖像图像中提取多尺度的特征金字塔（从浅层的低级纹理特征到深层的高级语义特征），然后通过多层cross-attention将这些特征注入到3D扩散模型的不同层级中。浅层特征提供纹理细节，深层特征提供结构引导
    - 设计动机：单一的全局特征无法传递丰富的细节信息；多层注入使3D生成能够获得更精细的2D线索

3. **优化的噪声调度（Optimized Noise Schedule for Triplanes）**:
    - 功能：改善扩散模型在triplane数据上的训练
    - 核心思路：针对triplane数据的信噪比特性，调整扩散过程的噪声调度方案。triplane的数据分布与自然图像不同，标准的噪声调度可能不是最优的。通过分析triplane的频率特性，设计更适合的噪声计划
    - 设计动机：标准噪声调度为自然图像设计，直接应用到triplane可能导致训练不稳定或质量下降

### 损失函数 / 训练策略

- 扩散训练损失：在triplane空间中训练去噪网络
- 渲染损失：将triplane渲染为2D图像，计算与GT图像的像素级和感知级损失
- 权重固化正则化损失：约束MLP解码器重要参数的更新
- 训练规模：在46K个头像数据上训练

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA(Rodin) | 提升 |
|--------|------|------|----------|------|
| 合成头像 | FID ↓ | 显著更好 | Rodin | 明显改善 |
| 真实肖像 | 视觉质量 | 更锐利 | Rodin | 细节丰富 |
| 发型细节 | 视觉质量 | 大幅提升 | 其他方法 | 最突出改进 |
| 泛化性 | in-the-wild | 良好 | 有限 | 更好泛化 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无数据调度 | 遗忘严重 | 新头像质量好但旧头像退化 |
| 无权重固化 | 部分遗忘 | 参数漂移导致质量下降 |
| 单层特征注入 | 细节不足 | 缺少多尺度信息 |
| 标准噪声调度 | 训练不稳定 | triplane特性不匹配 |
| 完整RodinHD | 最优 | 所有改进互补 |

### 关键发现

- 灾难性遗忘是triplane-based方法中的一个被忽视但重要的问题
- 数据调度+权重固化的组合有效缓解了遗忘，显著提升了渲染锐度
- 层级化特征注入是提升细节质量（特别是发型）的关键
- 在46K头像上训练后，模型能泛化到in-the-wild肖像输入

## 亮点与洞察

- 识别并解决了triplane生成中灾难性遗忘这一被忽视的问题，这一发现具有一般性
- 在3D生成领域引入持续学习的解决方案（EWC/数据调度），跨领域方法迁移
- 层级化特征注入的设计直观且有效
- 在发型等精细结构的生成上取得了显著进步

## 局限与展望

- 方法专注于头像/胸像生成，对全身3D人体的适用性需进一步验证
- 46K头像的训练数据需求较大，数据获取本身是一个挑战
- triplane表示的分辨率限制了最终细节的上限
- 可以结合3D Gaussian Splatting等新兴表示方法
- 动态头像（表情、动作）的生成是重要的扩展方向

## 相关工作与启发

- **Rodin**: RodinHD的前序工作，首次使用扩散模型生成3D头像triplane
- **DreamFusion / Zero-1-to-3**: 2D-to-3D生成的代表性方法
- **EWC**: 弹性权重固化，持续学习中的经典正则化方法
- 启发：3D生成方法中的训练稳定性问题值得更多关注，持续学习的技术在此有用武之地

## 评分

- 新颖性: ⭐⭐⭐⭐ 识别灾难性遗忘问题有重要贡献，解决方案实用
- 实验充分度: ⭐⭐⭐⭐ 大规模训练和消融实验充分
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，方法描述清晰
- 价值: ⭐⭐⭐⭐ 对高保真3D头像生成有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models](udifftext_a_unified_framework_for_high-quality_text_synthesis_in_arbitrary_image.md)

</div>

<!-- RELATED:END -->
