---
title: >-
  [论文解读] Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization
description: >-
  [CVPR 2025][图像生成][人体重光照] 提出基于预训练扩散模型的人体重光照和背景协调统一框架，通过粗到精策略（球谐函数 ControlNet 提供粗光照 + 扩散模型学习精细残差）和无监督运动 ControlNet 实现静态和视频场景的光照一致重光照。 领域现状：单目人体重光照旨在改变人物图像的光照条件…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "人体重光照"
  - "背景协调"
  - "粗到精扩散"
  - "时间一致性"
  - "视频重光照"
---

# Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization

**会议**: CVPR 2025  
**arXiv**: [2504.03011](https://arxiv.org/abs/2504.03011)  
**代码**: [https://junyingw.github.io/paper/relighting](https://junyingw.github.io/paper/relighting) (项目页)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 人体重光照, 背景协调, 粗到精扩散, 时间一致性, 视频重光照

## 一句话总结
提出基于预训练扩散模型的人体重光照和背景协调统一框架，通过粗到精策略（球谐函数 ControlNet 提供粗光照 + 扩散模型学习精细残差）和无监督运动 ControlNet 实现静态和视频场景的光照一致重光照。

## 研究背景与动机

**领域现状**：单目人体重光照旨在改变人物图像的光照条件。现有方法要么用物理模型（球谐函数等）精确但缺乏阴影等细节，要么用学习模型生成阴影但泛化差。背景协调（使人物光照与背景一致）通常作为独立任务处理。

**现有痛点**：(1) 端到端扩散方法直接从噪声中学习完整光照变换，但任务太难导致质量不稳定；(2) 静态图方法无法处理视频，因为逐帧重光照会导致时间闪烁；(3) 没有同时处理人体重光照和背景协调的统一框架。

**核心矛盾**：精确的物理光照模型（如球谐函数）可以处理漫反射但无法生成自遮挡阴影和镜面反射等细节，而扩散模型可以生成这些细节但难以保证物理正确性和时间一致性。

**本文目标** 统一处理人体重光照和背景协调，同时支持静态图和视频的时间一致重光照。

**切入角度**：粗到精分解——球谐函数提供粗光照变换（物理正确的基础），扩散模型仅需学习精细残差（如阴影、环境反射、镜面高光），降低学习难度。

**核心 idea**：用球谐函数 ControlNet 提供粗光照条件、扩散模型学习精细残差实现人体重光照，加上无监督运动 ControlNet 从真实视频学习光照时间一致性。

## 方法详解

### 整体框架
基于预训练 Stable Diffusion 微调。**Light ControlNet** 编码粗光照 shading（由球谐函数渲染）和目标背景图像，提供粗光照条件。扩散模型仅学习粗光照无法覆盖的精细残差。**Motion ControlNet** 从真实视频中无监督学习光照时间循环一致性。推理时用时空特征混合和引导精修保持高频细节。

### 关键设计

1. **粗到精光照分解**

    - 功能：降低扩散模型的学习难度，使其专注于精细光照效果
    - 核心思路：先用球谐函数从法线图和目标光照参数渲染粗 shading map，作为 Light ControlNet 的条件输入。扩散模型看到粗光照后只需预测残差（自遮挡阴影、环境反射等）。消融显示粗到精（PSNR 28.42）大幅优于端到端（26.42）和无扩散方案（17.10）
    - 设计动机：球谐函数处理漫反射快速准确但缺细节，扩散模型生成细节强但全量学习不稳定。分解让各取所长

2. **无监督时间一致性学习**

    - 功能：从无标注真实视频中学习光照变化的时间平滑性
    - 核心思路：Motion ControlNet 从真实视频帧序列中学习光照循环一致性——同一帧在不同光照下重光照后再循环回应保持一致。无需重光照真值标注。推理时用固定时空混合比例（空间 0.85:0.15，时间 0.5:0.5）融合相邻帧特征
    - 设计动机：不存在动态人体的重光照真值视频数据集，无监督方法绕过了数据限制

3. **引导精修（Guided Refinement）**

    - 功能：保持输出的高频细节不被扩散过程模糊
    - 核心思路：在去噪后期用原始图像的高频信息引导输出细节
    - 设计动机：扩散过程倾向于平滑高频纹理

### 损失函数 / 训练策略
标准扩散去噪损失 + 时间循环一致性损失。在合成数据（OpenIllumination, LightStage 等）+ 真实视频上训练，约 100K 训练样本。

## 实验关键数据

### 主实验

| 场景 | 方法 | PSNR↑ | SSIM↑ |
|------|------|-------|-------|
| 肖像 | DPR | 21.29 | 0.88 |
| 肖像 | **Ours** | **23.04** | **0.90** |
| 全身 | GFR | 28.57 | 0.95 |
| 全身 | **Ours** | **30.81** | **0.97** |
| 视频(动态光照+运动人物) | **Ours** | **26.61 PSNR / 38.32 tPSNR** | **0.94 / 0.98** |

### 消融实验

| 配置 | PSNR↑ |
|------|-------|
| 无扩散（仅球谐函数） | 17.10 |
| 端到端扩散 | 26.42 |
| **粗到精扩散** | **28.42** |
| + 背景 + 精修 | **28.78** |

### 关键发现
- 粗到精分解是核心贡献：比端到端扩散提升 2 PSNR，信明分解学习的有效性
- 无监督时间一致性在三个视频场景中都表现最优，无需重光照真值数据
- AMT 用户研究中 32.2% 用户选择本方法的重光照结果，接近真值 34.8%

## 亮点与洞察
- **粗到精的物理-学习混合**是简约高效的范式——用物理模型处理可建模的部分，学习模型只需补充残差
- **从真实视频无监督学时间一致性**绕过了标注数据瓶颈

## 局限与展望
- 训练数据以合成+LightStage 为主，极端真实场景可能泛化不足
- 时空混合比例为固定超参，可能不适用所有场景
- 无公开的动态人体重光照真值数据用于全面评估

## 相关工作与启发
- **vs DPR**: DPR 用球谐函数直接修改，缺乏阴影细节。本文在球谐函数基础上加扩散残差
- **vs GFR**: GFR 用条件 GAN，本文用扩散模型获得更好质量和泛化性

## 评分
- 新颖性: ⭐⭐⭐⭐ 粗到精分解和无监督视频一致性的组合设计有创新
- 实验充分度: ⭐⭐⭐⭐ 静态+视频+AMT 用户研究+消融，较全面
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰
- 价值: ⭐⭐⭐⭐ 统一框架对影视、AR 应用有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)
- [\[CVPR 2025\] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)
- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](../../NeurIPS2025/image_generation/unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)
- [\[CVPR 2025\] Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
