---
title: >-
  [论文解读] RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation
description: >-
  [ICLR 2026][图像生成][3D 资产参考] 提出 RefAny3D，一个 3D 资产参考的图像生成框架，通过联合建模 RGB 图像和点图（point map）的双分支生成策略，实现生成图像与 3D 参考资产在几何和纹理上的精确一致性。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "3D 资产参考"
  - "双分支生成"
  - "点图 (point map)"
  - "域解耦"
  - "主题驱动生成"
---

# RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation

**会议**: ICLR 2026  
**arXiv**: [2601.22094](https://arxiv.org/abs/2601.22094)  
**代码**: [https://judgementh.github.io/RefAny3D](https://judgementh.github.io/RefAny3D)  
**领域**: 3D 引导图像生成 / 扩散模型  
**关键词**: 3D 资产参考, 双分支生成, 点图 (point map), 域解耦, 主题驱动生成

## 一句话总结

提出 RefAny3D，一个 3D 资产参考的图像生成框架，通过联合建模 RGB 图像和点图（point map）的双分支生成策略，实现生成图像与 3D 参考资产在几何和纹理上的精确一致性。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：现有的参考图像生成方法（如 IP-Adapter、OminiControl）依赖 2D 参考图像，无法有效利用 3D 资产。在实际应用中，创作者通常需要直接使用网格等 3D 资产作为参考来可视化物体在不同场景中的表现。

3D 资产参考生成面临三大挑战：

**一致性不足**：需要与 3D 资产的几何结构和纹理精确对齐

**视角受限**：单参考图像方法无法捕获物体的完整外观

**视角冲突**：多图像条件方法缺乏 3D 结构先验，导致跨视角不一致

## 方法详解

### 整体框架

基于 Flux.1-dev 模型构建，输入 3D 资产的多视角 RGB 图像和点图作为条件，同时生成目标 RGB 图像及其对应的点图。

### 关键设计一：空间对齐的双分支生成

将生成过程形式化为联合分布建模：$p(x_I, x_P | y, c)$

- $x_I$：目标 RGB 图像
- $x_P$：对应点图
- $y$：参考 3D 模型
- $c$：文本提示

**共享位置编码**：对 RGB 和点图的同一视角令牌施加共享位置编码，利用 DiT 中位置编码的特性自然地为相同位置的令牌分配更高注意力分数。引入统一位置偏移项 $(i-w, j)$ 避免条件令牌间距离不一致造成的偏差。

### 关键设计二：域解耦生成

RGB 和点图存在固有的信息不对称：点图仅定义 3D 几何和姿态，RGB 包含整个场景的写实细节。

- **域特定 LoRA**：引入 Reference-LoRA（为所有条件令牌激活）和 Domain-LoRA（仅为点图令牌激活），分别学习参考生成和点图域知识
- **文本无关注意力**：在点图分支引入注意力掩码，抑制文本令牌对点图的影响，避免背景信息泄露到点图中

### 数据集构建

基于 Subjects200K 数据集构建 3D 资产-姿态对齐数据集：
1. 用 GroundingDINO 提取目标物体
2. 用 Hunyuan3D 将物体转为 3D 资产
3. 用 FoundationPose 估计 3D 资产在图像中的姿态

## 实验

### 主实验（GPT 评估 + 视觉模型评估）

| 方法 | 纹理↑ | 几何↑ | 美学↑ | 总分↑ | CLIP Avg↑ | DINO Avg↑ | GIM↑ |
|------|------|------|------|------|----------|----------|------|
| Textual Inversion | 2.89 | 4.42 | 6.26 | 4.53 | 0.827 | 0.548 | 3360 |
| DreamBooth | 5.37 | 6.68 | 6.89 | 6.32 | 0.867 | 0.695 | 3483 |
| OminiControl | 5.63 | 6.58 | 6.89 | 6.37 | 0.855 | 0.665 | 3474 |
| **RefAny3D** | **6.32** | **7.37** | **7.69** | **7.12** | **0.873** | **0.720** | **3901** |

### 消融实验

| 设置 | 效果 |
|------|------|
| 无共享位置编码 | 点图与 RGB 像素级对应失败，几何一致性下降 |
| 无文本无关注意力 | 点图受文本影响，背景区域出现颜色混合 |
| 无域特定 LoRA | 单一 LoRA 同时学习两个域，导致背景伪影 |
| 无点图分支 | 缺乏 3D 线索，训练不稳定，3D 一致性差 |

### 用户研究

RefAny3D 在忠实度（4.655）、ID 保持（4.737）、美学质量（4.632）和整体排名（1.579）上均优于所有基线。

## 亮点与洞察

- 首次探索以 3D 资产为参考条件的图像生成任务
- 点图作为结构锚点的设计有效建立了跨视角的像素级对应
- 域解耦策略优雅地解决了 RGB 与点图的信息不对称问题
- 可与多视图到 3D 生成模型集成，形成完整工作流

## 局限与展望

- 对非刚性物体（如绳索、靠垫）效果较差，因数据集限制
- 大量视角条件输入带来显著的计算和时间开销
- 依赖 Hunyuan3D 和 FoundationPose 的质量进行数据构建

## 相关工作

- **主题驱动生成**：Textual Inversion、DreamBooth、IP-Adapter、OminiControl 等
- **3D 引导生成**：ThemeStation、Phidias 等侧重于 3D 资产生成而非图像生成
- **多模态生成**：Marigold、GeoWizard 等联合生成 RGB 和几何信息

## 评分

- 新颖性：⭐⭐⭐⭐ — 3D 资产参考图像生成是全新任务定义
- 技术性：⭐⭐⭐⭐ — 双分支 + 域解耦设计合理
- 实验：⭐⭐⭐⭐ — GPT 评估 + 视觉模型 + 用户研究，评估全面
- 实用性：⭐⭐⭐⭐ — 对 3D 内容创作有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](../../CVPR2025/image_generation/3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)
- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)

</div>

<!-- RELATED:END -->
