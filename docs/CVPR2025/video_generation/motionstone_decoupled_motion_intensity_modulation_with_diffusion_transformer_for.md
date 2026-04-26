---
title: >-
  [论文解读] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation
description: >-
  [CVPR 2025][图像到视频生成] 提出 MotionStone，通过训练独立的运动强度估计器将视频运动解耦为物体运动和相机运动两个维度，并以解耦方式注入 Diffusion Transformer，实现精细的运动强度可控 I2V 生成。
tags:
  - CVPR 2025
  - 图像到视频生成
  - 运动强度估计
  - 物体/相机运动解耦
  - 对比学习
  - 扩散模型
---

# MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation

**会议**: CVPR 2025  
**arXiv**: [2412.05848](https://arxiv.org/abs/2412.05848)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 图像到视频生成, 运动强度估计, 物体/相机运动解耦, 对比学习, 扩散模型

## 一句话总结

提出 MotionStone，通过训练独立的运动强度估计器将视频运动解耦为物体运动和相机运动两个维度，并以解耦方式注入 Diffusion Transformer，实现精细的运动强度可控 I2V 生成。

## 研究背景与动机

I2V 生成中，运动强度控制是关键但未被充分解决的问题。现有方法存在以下不足：

1. **传统运动度量不可靠**：SSIM、光流等传统指标难以泛化到任意视频，与人类感知的运动强度不一致
2. **运动解耦缺失**：真实视频中的运动是物体运动与相机运动的叠加，现有方法将整个场景的运动强度建模为单一值，无法区分两种运动类型
3. **标注困难**：让人类标注者为视频标注绝对运动强度分数不切实际，因为人们难以对抽象的运动强度给出一致的评分

核心洞察：人虽然很难给单个视频打绝对运动分数，但**比较两个视频哪个运动更强**是相对容易的。基于此，作者设计了一种相对标注 + 对比学习的框架来训练运动估计器。

## 方法详解

### 整体框架

MotionStone 包含两大模块：(1) 独立的运动强度估计器，预测视频的物体运动分数和相机运动分数（1-10 范围）；(2) 基于 CogVideoX 的 I2V 扩散模型，以解耦运动嵌入作为条件进行视频生成。

### 关键设计

1. **运动强度估计器（Motion Estimator）**：采用 TAdaConv 作为视频运动表征骨干网络，提取运动特征 $M = \text{TAdaConv}(\mathbf{x}; \phi)$，然后经过全局平均池化后分别输入两个 MLP 头：$s^{object} = \text{MLP}_{object}(\text{GAP}(M); \theta)$ 和 $s^{camera} = \text{MLP}_{camera}(\text{GAP}(M); \theta)$，分别预测物体和相机运动分数。**设计动机**：使用轻量级时序自适应卷积作为骨干，双头结构自然实现运动解耦。

2. **相对标注与对比训练**：构建 5000 个视频对，标注者仅需判断哪个视频的物体/相机运动更强。训练使用 pairwise ranking loss：$L_o = \max(0, s_2^{object} - s_1^{object})$（假设视频 1 运动更强），$L_c = \max(0, s_2^{camera} - s_1^{camera})$。为避免预测分数过于集中，额外使用跟踪轨迹生成的伪标签进行回归训练：$\mathcal{L}_r = \|s^{object} - y^{object}\|_2^2 + \|s^{camera} - y^{camera}\|_2^2$。总损失 $\mathcal{L}_{total} = \mathcal{L}_o + \mathcal{L}_c + \lambda \mathcal{L}_r$。

3. **解耦运动条件注入（Decoupled Motion Embedding）**：物体和相机运动分数分别通过独立的 MLP 映射到高维向量，拼接后加到时间步嵌入 $t$ 上，通过 adaptive LayerNorm 调制 DiT 中的视觉和文本特征。**设计动机**：物体运动和相机运动在空间维度上含义不同，混合注入会模糊各自贡献，解耦保持语义清晰。

### 损失函数 / 训练策略

- **运动估计器训练**：Ranking Loss + Regression Loss with 伪标签（$\lambda$ 平衡两项）
- **扩散模型训练**：基于 CogVideoX 框架 SFT 微调，100K 高质量视频，8 张 A100 GPU，batch size 16
- 每个视频采样 49 帧，分辨率 $480 \times 720$，中心裁剪
- 运动估计器预训练后冻结，推理时用户可自定义物体/相机运动强度分数

## 实验关键数据

### 主实验（WebVID 验证集，VBench 指标）

| 方法 | Background Consistency ↑ | Aesthetic Quality ↑ | Imaging Quality ↑ |
|------|-------------------------|--------------------|--------------------|
| I2VGen-XL | 90.93% | 40.14% | 58.35% |
| SVD | 93.17% | 42.38% | 59.61% |
| AnimateAnything | 93.89% | 46.04% | 61.69% |
| CogVideoX-5B | 94.91% | 45.88% | 61.99% |
| **MotionStone** | **95.76%** | **46.78%** | **62.29%** |

### 消融实验

| 配置 | BG Consistency ↑ | Aesthetic ↑ | Imaging ↑ | 说明 |
|------|------------------|-------------|-----------|------|
| w/o 运动估计器 (固定 5) | 95.13% | 45.61% | 60.15% | 训练数据运动多样性导致混淆 |
| w/ 特征差异估计 (S) | 94.97% | 46.13% | 60.73% | 与人类感知不一致 |
| w/ SSIM 估计 | 92.99% | 45.72% | 54.75% | SSIM 无法解耦，最差 |
| w/o 解耦注入 | 94.03% | 46.27% | 58.73% | 混合注入模糊运动贡献 |
| **MotionStone (完整)** | **95.76%** | **46.78%** | **62.29%** | 最优 |

### 运动估计准确率

| 方法 | 运动估计准确率 |
|------|---------------|
| SSIM | 44.56% |
| **本文 Motion Estimator** | **72.80%** |

### 关键发现

- 运动估计器在判断视频对运动关系上比 SSIM 高出 **28%** 准确率
- 解耦注入比混合注入在 Imaging Quality 上提升 **3.56%**（58.73% → 62.29%）
- 物体运动和相机运动可独立调节，用户设定分数 1-10 实现从静止到剧烈运动的连续控制
- 固定相机运动为 5、变化物体运动强度时，生成视频物体运动速度/幅度单调递增
- 固定物体运动为 5、变化相机运动强度时，zoom/pan 幅度单调递增

## 亮点与洞察

- **"比较式标注"思路巧妙**：绕过了绝对运动标注的困难，5000 个视频对即可训练有效的运动估计器
- **运动解耦的必要性**：实验充分证明物体和相机运动在不同空间维度上操作，混合建模显著降低性能
- **通用插件潜力**：训练好的运动估计器可作为数据预处理工具或其他视频生成模型的增强模块
- TAdaConv 骨干 + 双 MLP 头的轻量设计，几乎不增加推理开销

## 局限与展望

- 运动估计器仅支持整体场景的物体/相机运动评分，无法实现**逐物体**的运动控制
- 标注数据规模（5000 对）相对有限，增加数量和多样性可能进一步提升
- 运动强度分数为 1-10 离散整数，缺乏更细粒度的连续控制
- 未探讨复杂场景（多物体不同运动方向/速度）下的效果
- 仅在 CogVideoX 上验证，未在其他基础模型上测试通用性

## 相关工作与启发

- LivePhoto / Cinemo：文本 + SSIM 粗粒度运动控制的先驱，但 SSIM 不可靠
- AnimateAnything：支持粗粒度运动强度但生成视频常接近静止
- CogVideoX：本文基础模型，DiT 架构的强大时空建模能力
- 对比学习 + 排序损失范式可迁移至其他难以绝对标注的视觉任务（如视频美学评分）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 运动解耦估计 + 相对标注思路新颖
- **实验充分度**: ⭐⭐⭐⭐ 消融全面，验证了每个组件的有效性
- **写作质量**: ⭐⭐⭐⭐ 整体流畅，动机清晰
- **价值**: ⭐⭐⭐⭐ 运动估计器可作为通用插件，对 I2V 领域有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)
- [\[CVPR 2025\] MotiF: Making Text Count in Image Animation with Motion Focal Loss](motif_making_text_count_in_image_animation_with_motion_focal_loss.md)

<!-- RELATED:END -->
