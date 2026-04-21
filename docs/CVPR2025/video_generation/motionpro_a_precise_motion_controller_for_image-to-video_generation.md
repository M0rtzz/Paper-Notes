---
title: >-
  [论文解读] MotionPro: A Precise Motion Controller for Image-to-Video Generation
description: >-
  [CVPR 2025][图像到视频生成] 提出 MotionPro，利用区域级轨迹（region-wise trajectory）和运动掩码（motion mask）双重信号，实现细粒度、可区分物体/相机运动的精确可控图像到视频生成。
tags:
  - CVPR 2025
  - 图像到视频生成
  - 运动控制
  - 区域轨迹
  - 运动掩码
  - 扩散模型
---

# MotionPro: A Precise Motion Controller for Image-to-Video Generation

**会议**: CVPR 2025  
**arXiv**: [2505.20287](https://arxiv.org/abs/2505.20287)  
**代码**: https://github.com/zhw-zhang/MotionPro-page (有)  
**领域**: 视频理解  
**关键词**: 图像到视频生成, 运动控制, 区域轨迹, 运动掩码, 扩散模型

## 一句话总结

提出 MotionPro，利用区域级轨迹（region-wise trajectory）和运动掩码（motion mask）双重信号，实现细粒度、可区分物体/相机运动的精确可控图像到视频生成。

## 研究背景与动机

现有可控 I2V 方法（如 DragNUWA、DragAnything）主要依赖大核高斯滤波扩展稀疏轨迹作为条件信号，存在两大问题：
1. **细粒度运动不准确**：高斯滤波（核大小高达 $99 \times 99$）会将轨迹信号扩散到周围区域，导致精细运动细节丢失，生成的运动（如转头）不自然
2. **运动类别混淆**：仅依赖轨迹条件无法区分"物体运动"与"相机运动"。例如，行星上的向下轨迹既可以是相机下拉也可以是行星上升，单一轨迹信号易产生歧义

此外，MOFA-Video 虽引入了运动区域 mask，但仅用于后处理的光流掩码，而非作为生成条件注入网络，导致合成视频出现局部扭曲。

## 方法详解

### 整体框架

MotionPro 基于预训练 SVD（Stable Video Diffusion）构建。训练时从输入视频中提取**区域级轨迹**和**运动掩码**作为双重控制信号，通过 Motion Encoder 编码为多尺度特征，再通过自适应特征调制（Adaptive Feature Modulation）注入 3D-UNet 的各尺度特征层。同时使用 LoRA 微调所有注意力模块以增强运动-轨迹对齐。

### 关键设计

1. **区域级轨迹（Region-wise Trajectory）**：使用 DOT 光流跟踪模型估计视频的光流图 $f^i$ 和可见性掩码 $M^i$，计算全局可见性掩码 $M_g = \prod_{i=1}^{L} M^i$，将掩码后的光流图划分为 $k \times k$（默认 $k=8$）的局部区域，并通过随机区域选择掩码 $M_{sel}$（mask ratio 在 $[r_{min}, 1.0]$ 随机采样）稀疏选取区域内的轨迹。**核心优势**：相比高斯滤波，直接保留局部区域内的原始轨迹信息，保持精确的运动细节。

2. **运动掩码（Motion Mask）**：计算所有帧光流幅值的时间平均 $f_{avg} = \frac{1}{L} \sum_{i=1}^{L} \|f^i\|_2$，将 $f_{avg} > 1$ 的位置标记为运动区域得到 $M_{mot}$，重复 $L$ 次形成运动掩码序列 $\mathbf{M}_{mot} \in \{0,1\}^{L \times H \times W \times 1}$。**作用**：全局性地标识运动区域，明确目标运动类别（物体/相机运动），消除轨迹信号的歧义。

3. **自适应特征调制（Adaptive Feature Modulation）**：将轨迹和掩码拼接后通过轻量级 Motion Encoder 编码为多尺度特征 $l_s$，在每个尺度通过时空卷积层预测 scale $\gamma_s$ 和 bias $\beta_s$，对视频潜在特征进行调制：$h_s' = GN(h_s) \cdot \gamma_s + \beta_s + h_s$。初始化时 $\gamma_s = 0, \beta_s = 0$（零初始化），保证训练稳定性。

### 损失函数 / 训练策略

- 使用 EDM 训练协议的去噪分数匹配（DSM）损失：$\mathcal{L} = \mathbb{E}[\lambda_\sigma \|\hat{\mathbf{z}}_0 - \mathbf{z}_0\|_2^2]$
- LoRA rank 设为 32，仅训练 Motion Encoder 和 LoRA 层
- 学习率 $1 \times 10^{-5}$，AdamW 优化器
- 6 张 A800 GPU，batch size 48
- 视频分辨率 $320 \times 512$，16 帧，8 fps

## 实验关键数据

### 主实验 — 细粒度运动控制

| 数据集 | 指标 | MotionPro | MOFA-Video | DragNUWA | 提升 |
|--------|------|-----------|------------|----------|------|
| WebVid-10M | FVD ↓ | **59.88** | 87.70 | 96.65 | -27.82 |
| WebVid-10M | FID ↓ | **10.40** | 12.18 | 13.19 | -1.78 |
| MC-Bench | MD-Img ↓ | **10.56** | 13.94 | - | -3.38 |
| MC-Bench | MD-Vid ↓ | **8.34** | 10.50 | - | -2.16 |

### 主实验 — 物体级运动控制（MC-Bench）

| 方法 | MD-Img ↓ | MD-Vid ↓ | Frame Consis. ↑ |
|------|----------|----------|-----------------|
| MOFA-Video | 15.56 | 12.04 | 0.9951 |
| DragAnything | 12.30 | 11.37 | 0.9917 |
| **MotionPro** | **10.48** | **8.59** | 0.9943 |

### 消融实验

| 配置 | MD-Vid (fine) ↓ | MD-Vid (obj) ↓ | 说明 |
|------|----------------|----------------|------|
| $k=1$ | 较高 | 较高 | 区域太小，轨迹信号不足 |
| $k=4$ | 中等 | 中等 | 性能逐步提升 |
| $k=8$ | **最优** | **最优** | 精度与覆盖的最佳平衡 |
| $k=16$ | 略差 | 略差 | 区域过大影响细粒度控制 |
| MotionPro$_C$（拼接注入） | 较高 | 较高 | 特征拼接需严格时空对齐 |
| MotionPro$_+$（加法注入） | 中等 | 中等 | 加法效果次优 |
| **MotionPro**（调制注入） | **最优** | **最优** | 间接调制无需严格对齐 |

### 关键发现

- MotionPro 生成视频的平均光流幅值为 8.95，远高于 MOFA-Video 的 4.95，说明运动变化更丰富
- 运动掩码有效解决了如 DragNUWA 的运动类别混淆（将物体运动误判为相机运动）
- MC-Bench 基准包含 1.1K 用户标注的图像-轨迹对，更符合实际使用场景

## 亮点与洞察

- **区域级轨迹替代高斯滤波**：简单但有效的设计变更，直接使用局部区域内原始光流而非高斯扩展
- **运动掩码的条件化使用**：将 mask 作为生成条件注入而非仅用于后处理，从根本上避免运动类别歧义
- **MC-Bench 基准**：填补了可控 I2V 缺少用户标注评测基准的空白
- 特征调制（FiLM 风格）比拼接/相加注入更适合运动控制信号的融合

## 局限与展望

- 基于 SVD 架构，分辨率限制在 $320 \times 512$，未迁移到更强的 DiT 基础模型
- 训练需要 DOT 预计算光流，增加了数据预处理成本
- 仅支持单张参考图 + 轨迹/掩码的输入，不支持文本描述的运动控制
- 未探讨多物体独立运动控制的场景

## 相关工作与启发

- DragNUWA/DragAnything：高斯轨迹方法的代表，是本文主要对比对象
- MOFA-Video：两阶段框架（稀疏→稠密轨迹），mask 仅用于后处理是其局限
- Motion-I2V：类似的稀疏到稠密思路
- 本文的 region-wise 设计思路可推广到视频编辑、3D 场景运动控制等任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 区域级轨迹和运动掩码的双信号设计简洁有效
- **实验充分度**: ⭐⭐⭐⭐ 提出了新 benchmark MC-Bench，消融实验细致
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，问题动机明确
- **价值**: ⭐⭐⭐⭐ 对可控视频生成领域有实用参考价值

<!-- RELATED:START -->

## 相关论文

- [MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [MotiF: Making Text Count in Image Animation with Motion Focal Loss](motif_making_text_count_in_image_animation_with_motion_focal_loss.md)
- [Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)

<!-- RELATED:END -->
