---
title: >-
  [论文解读] Realistic Human Motion Generation with Cross-Diffusion Models
description: >-
  [ECCV 2024][图像生成] 提出 CrossDiff 框架，通过统一编码和交叉解码机制融合 3D 与 2D 运动信息，利用交叉扩散实现更精细的全身运动细节捕获，并支持从野外 2D 数据学习 3D 运动生成。
tags:
  - ECCV 2024
  - 图像生成
---

# Realistic Human Motion Generation with Cross-Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2312.10993](https://arxiv.org/abs/2312.10993)  
**领域**: 图像生成

## 一句话总结

提出 CrossDiff 框架，通过统一编码和交叉解码机制融合 3D 与 2D 运动信息，利用交叉扩散实现更精细的全身运动细节捕获，并支持从野外 2D 数据学习 3D 运动生成。

## 研究背景与动机

- 文本驱动的人体运动生成在游戏、VR、机器人等领域需求日增
- 现有方法（MDM、MLD、T2M-GPT）仅依赖 3D 运动信息训练，忽略细微运动细节
- **关键洞察**：仅使用 3D 表示时，模型容易关注主要运动而忽略局部细节（如手指、面部表情）；而 2D 投影可在不同视角下放大这些细微运动
- 收集高质量 3D 运动数据成本高昂，而 2D 运动数据容易从视频中提取
- **核心问题**：如何利用 2D 运动的互补信息增强 3D 运动生成的全身细节？

## 方法详解

### 整体框架

CrossDiff 由三个核心模块组成：

1. **混合表示（Mixed Representations）**：将 3D 运动数据通过正交投影到四个方向（前、左、右、后）获得对应的 2D 运动数据
2. **统一编码（Unified Encoding）**：两个独立编码器（$\mathcal{E}_{3D}$、$\mathcal{E}_{2D}$）分别处理 3D/2D 运动噪声，加上一个共享权重编码器 $\mathcal{E}_{share}$ 映射到统一特征空间
3. **交叉解码（Cross-Decoding）**：独立的 3D/2D 解码器可从任意维度的统一特征中输出对应维度的运动

### 关键设计

**1. 交叉扩散机制**

框架产生四种输出路径：3D→3D、2D→3D、3D→2D、2D→2D，实现跨维度的噪声逆转：

$$\hat{x}_{iD \to jD,0} = G_{iD \to jD}(x_{iD,t}, t, c) = \mathcal{D}_{jD}(\mathcal{E}_{share}(\mathcal{E}_{iD}(x_{iD,t}, t, c)))$$

**2. 两阶段训练策略**

- **阶段 I**：同时学习四个方向的逆扩散过程，建立 2D/3D 运动之间的映射关系
  $$\mathcal{L}_{stage I} = \mathcal{L}_{3D \to 3D} + w_{23}\mathcal{L}_{2D \to 3D} + w_{32}\mathcal{L}_{3D \to 2D} + w_{22}\mathcal{L}_{2D \to 2D}$$
- **阶段 II**：仅使用 3D 生成损失微调，聚焦 3D 去噪，同时保留从 2D 学到的丰富运动特征
  $$\mathcal{L}_{stage II} = \mathcal{L}_{3D \to 3D}$$

**3. 混合采样（Mixture Sampling）**

推理时可先在 2D 域去噪到时间步 $\alpha$，再通过 $G_{2D \to 3D}$ 投射到 3D 域继续去噪。该策略利用 2D 域的更丰富运动细节来引导 3D 生成。

**4. 从 2D 野外数据学习 3D 运动**

利用预训练的 $G_{2D \to 3D}$ 从视频中估计的 2D 姿态生成伪 3D 标签，实现在无 3D GT 数据的情况下微调生成域外运动。

### 损失函数

各路径采用简单的重建目标：

$$\mathcal{L}_{iD \to jD} = \mathbb{E}_{t \sim [1,T]} \|x_{jD,0} - G_{iD \to jD}(x_{iD,t}, t, c)\|_2^2$$

## 实验关键数据

### 主实验

HumanML3D 和 KIT-ML 数据集上与 SOTA 方法的比较：

| 方法 | R-Prec(top3)↑ | FID↓ | MM Dist↓ | DIV→ | FID-U↓ | FID-L↓ |
|------|-------------|------|----------|------|--------|--------|
| MDM | 0.611 | 0.544 | 5.566 | 9.559 | 0.825 | 0.840 |
| T2M-GPT | 0.775 | 0.141 | 3.121 | 9.722 | 0.145 | 0.607 |
| MLD | 0.772 | 0.473 | 3.196 | 9.724 | 0.541 | 0.553 |
| ReMoDiffuse | 0.795 | **0.103** | **2.974** | 9.018 | 0.125 | 0.565 |
| **CrossDiff** | 0.730 | 0.162 | 3.358 | 9.577 | **0.118** | **0.281** |

KIT-ML 数据集：

| 方法 | R-Prec(top3)↑ | FID↓ | MM Dist↓ | FID-U↓ | FID-L↓ |
|------|-------------|------|----------|--------|--------|
| MDM | 0.396 | 0.497 | 9.191 | 0.925 | 0.973 |
| T2M-GPT | 0.745 | 0.514 | 3.007 | 0.602 | 0.715 |
| ReMoDiffuse | 0.765 | **0.155** | **2.814** | 0.205 | 0.644 |
| **CrossDiff** | 0.704 | 0.474 | 3.308 | **0.434** | **0.625** |

### 消融实验

CrossDiff 各组件在 HumanML3D 上的影响：

| 设置 | R-Prec↑ | FID↓ | MM Dist↓ | DIV→ |
|------|---------|------|----------|------|
| MDM baseline | 0.611 | 0.544 | 5.566 | 9.559 |
| 50% 3D | 0.666 | 0.586 | 3.894 | 9.513 |
| 100% 3D | 0.685 | 0.224 | 3.690 | 9.445 |
| 50% 3D + 100% 2D | 0.672 | 0.422 | 3.708 | 9.345 |
| 100% 3D + 100% 2D | **0.730** | **0.162** | **3.358** | 9.577 |
| w/o 共享编码器 | 0.714 | 0.187 | 3.496 | 9.488 |
| w/ 共享编码器 | **0.730** | **0.162** | **3.358** | 9.577 |
| 1个视角(前) | 0.722 | 0.186 | 3.467 | 9.798 |

### 关键发现

1. **上下半身 FID 指标全面领先**：CrossDiff 的 FID-U=0.118（最佳）和 FID-L=0.281（最佳），表明全身运动生成质量更均衡
2. ReMoDiffuse 和 T2M-GPT 的上半身 FID 较低但下半身 FID 偏高，说明运动生成不均衡
3. 加入 2D 数据显著降低 FID（从 0.224 → 0.162），证实跨维度互补信息的有效性
4. 共享编码器是必要的——去掉后 FID 从 0.162 上升到 0.187
5. 用户研究中 CrossDiff 在运动活力和多样性方面获得最高偏好
6. 从 UCF101 野外 2D 数据可以成功生成域外 3D 运动（如引体向上、骑自行车）

## 亮点与洞察

- **核心创新在于跨维度互补**：2D 投影可从不同视角放大细微运动，补充 3D 表示容易忽略的局部细节
- **提出上下半身分离的 FID 评估指标**（FID-U/FID-L），更细粒度地分析全身运动生成质量
- **混合采样策略**可灵活切换 2D/3D 域，理论上支持从纯 2D 噪声生成 3D 运动
- **实用价值**：2D 运动数据大量可从视频提取，该方法大幅降低了 3D 运动数据收集成本

## 局限性

- 在传统指标（R-Precision、FID）上不如 ReMoDiffuse 等方法，优势主要体现在细粒度运动质量
- KIT-ML 数据集以"行走"为主，不适合展示方法的细节捕获优势
- 2D 到 3D 映射存在深度模糊性，纯 2D 数据训练的根节点估计不够精确
- 两阶段训练流程增加了训练复杂度

## 评分

- 创新性：⭐⭐⭐⭐ — 跨维度扩散机制新颖
- 实用性：⭐⭐⭐⭐ — 支持野外2D数据训练
- 表现力：⭐⭐⭐ — 传统指标不占优，细粒度指标领先
- 综合评分：7.5/10

<!-- RELATED:START -->

## 相关论文

- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)

<!-- RELATED:END -->
