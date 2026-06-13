---
title: >-
  [论文解读] SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning
description: >-
  [CVPR 2025][自监督学习][掩码视频建模] 提出 SMILE，通过合成运动增强（在视频上叠加沿随机轨迹运动的分割物体）和 CLIP 特征重建目标来增强掩码视频建模，结合轨迹引导的掩码策略，在 K400 线性探测上大幅提升至 56.2%（前 SOTA 47.5%）。 领域现状：自监督视频表示学习中…
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "掩码视频建模"
  - "合成运动增强"
  - "CLIP特征目标"
  - "轨迹掩码"
  - "VideoMAE"
---

# SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning

**会议**: CVPR 2025  
**arXiv**: [2504.00527](https://arxiv.org/abs/2504.00527)  
**代码**: [https://github.com/fmthoker/SMILE](https://github.com/fmthoker/SMILE)  
**领域**: 自监督学习 / 视频理解  
**关键词**: 掩码视频建模, 合成运动增强, CLIP特征目标, 轨迹掩码, VideoMAE

## 一句话总结

提出 SMILE，通过合成运动增强（在视频上叠加沿随机轨迹运动的分割物体）和 CLIP 特征重建目标来增强掩码视频建模，结合轨迹引导的掩码策略，在 K400 线性探测上大幅提升至 56.2%（前 SOTA 47.5%）。

## 研究背景与动机

**领域现状**：自监督视频表示学习中，掩码视频建模（如 VideoMAE）通过掩码-重建学习时空特征。但视频中的时间冗余严重——相邻帧几乎相同，模型可能通过"看邻居"作弊而不真正理解运动。

**现有痛点**：（1）像素重建目标关注低层纹理而非高层语义；（2）自然视频中物体运动不够多样，模型难以学到丰富的运动模式；（3）随机 tube masking 没有针对运动区域。

**核心矛盾**：掩码视频学习需要丰富的运动信号，但自然视频中大部分区域是静态背景。

**切入角度**：人工合成运动——从 Stable Diffusion 生成物体，用 X-Paste 分割后沿随机平滑轨迹叠加到视频上，强制引入运动信号。搭配 CLIP 特征替代像素作为重建目标。

**核心 idea**：合成物体运动增强 + CLIP 特征重建 + 轨迹掩码 = 运动感知的自监督视频学习。

## 方法详解

### 关键设计

1. **合成运动增强**:

    - 功能：人工增加视频中的运动多样性
    - 核心思路：用 Stable Diffusion 生成物体图像，X-Paste 分割出来，沿随机二次贝塞尔曲线轨迹叠加到视频帧上，带缩放和旋转变换。强制模型学习跟踪这些运动物体
    - 设计动机：消融显示合成运动在 K400 上贡献 +3.1%（像素目标），说明它有效打破了时间冗余

2. **CLIP 特征重建目标**:

    - 功能：用高层语义特征替代低层像素作为重建目标
    - 核心思路：用预训练 CLIP 图像编码器提取每帧特征，作为掩码 token 的重建目标：$\mathcal{L} = \frac{1}{|\mathcal{T}^{mask}|}\sum_{i \in \mathcal{T}^{mask}} \|f_i' - Y_i\|_2^2$
    - 设计动机：CLIP 特征比像素高 +12.1% 线性探测准确率——证明语义级目标远优于像素级

3. **轨迹引导掩码**:

    - 功能：在合成物体运动轨迹上额外施加掩码
    - 核心思路：在标准 tube masking 基础上，沿合成物体的运动轨迹额外掩码 token，迫使模型预测轨迹上的语义特征
    - 设计动机：在运动最活跃的区域施加掩码，最大化运动推理需求

### 损失函数 / 训练策略

CLIP 特征 L2 重建损失。ViT-B backbone，K400 上 600 epochs。渐进训练：先用合成运动+CLIP 目标预训练，再用原始视频微调。

## 实验关键数据

### 主实验

| 方法 | K400 线性探测 | UCF-101 | SSv2 |
|------|-------------|---------|------|
| VideoMAE | 40.2% | 73.1% | 17.8% |
| SIGMA (前SOTA) | 47.5% | 80.7% | 21.7% |
| **SMILE** | **56.2%** | **83.8%** | **23.7%** |

### 消融实验

| 配置 | K400 线性探测 |
|------|-------------|
| 像素目标 | 44.1% |
| + 合成运动 | 47.2% (+3.1%) |
| CLIP 目标 | 56.2% (+12.1%) |
| CLIP + 合成运动 | 56.2% |
| + 轨迹掩码 | +~1% |

### 关键发现
- **CLIP vs 像素目标差距巨大**：+12.1%，语义级重建目标是最大贡献因素
- **合成运动在像素目标下更有效**：+3.1%（像素），但在 CLIP 目标下增益饱和
- **SSv2 (运动敏感) 也提升**：23.7% vs 21.7%，说明运动理解确实增强

## 亮点与洞察
- **打破时间冗余的简单方案**——在视频上"贴"运动物体比收集新数据简单得多
- **CLIP 特征的碾压性优势**——从像素到 CLIP 就提升 12 个点，暗示 VideoMAE 的瓶颈不在掩码策略而在重建目标

## 局限与展望
- 合成物体运动不够真实（机械地沿轨迹移动）
- 主要在动作识别上评估，时序推理任务验证不足
- CLIP 特征继承了图文对齐的偏置

## 评分
- 新颖性: ⭐⭐⭐⭐ 合成运动+CLIP 目标的组合有效且新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集，详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 大幅刷新自监督视频学习 SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] From Linearity to Non-Linearity: How Masked Autoencoders Capture Spatial Correlations](../../ICCV2025/self_supervised/from_linearity_to_non-linearity_how_masked_autoencoders_capture_spatial_correlat.md)
- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](../../ECCV2024/self_supervised/vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)
- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)

</div>

<!-- RELATED:END -->
