---
title: >-
  [论文解读] GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation
description: >-
  [ECCV 2024][3D视觉][3D高斯] 提出GRM，一种基于纯Transformer架构的前馈式3D重建模型，将稀疏视图(4张图)的像素通过pixel-aligned Gaussians转化为稠密的3D高斯表示，约0.1秒完成重建，结合多视图扩散模型可实现文本/图像到3D生成。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D高斯
  - 稀疏视图重建
  - 3D生成
  - Transformer
  - 前馈式重建
---

# GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation

**会议**: ECCV 2024  
**arXiv**: [2403.14621](https://arxiv.org/abs/2403.14621)  
**代码**: https://justimyhxu.github.io/projects/grm/ (有)  
**领域**: 3D视觉  
**关键词**: 3D高斯, 稀疏视图重建, 3D生成, Transformer, 前馈式重建

## 一句话总结

提出GRM，一种基于纯Transformer架构的前馈式3D重建模型，将稀疏视图(4张图)的像素通过pixel-aligned Gaussians转化为稠密的3D高斯表示，约0.1秒完成重建，结合多视图扩散模型可实现文本/图像到3D生成。

## 研究背景与动机

**领域现状**: 3D资产创建在机器人、游戏、建筑等领域需求旺盛。基于优化的3D生成方法（如SDS）质量高但耗时数小时；前馈式方法速度快但多依赖triplane-NeRF表示，受限于体渲染效率和分辨率。

**现有痛点**: 
   - 优化方法（SDS-based）: 单个3D资产需要数小时优化
   - 前馈方法（如LRM）: 依赖triplane表示+体渲染，效率低、分辨率受限
   - 并发工作（LGM, Splatter Image）: 使用卷积U-Net架构，生成的高斯数量有限

**核心矛盾**: 如何在保持快速推理的同时，生成足够数量的高质量3D高斯以实现高保真重建？

**本文目标**: 设计一种可扩展的前馈框架，能从稀疏视图高效生成大量像素对齐的3D高斯，实现高质量快速3D重建和生成。

**切入角度**: 用3D高斯替代triplane-NeRF，用纯Transformer架构替代CNN-based模型，利用窗口注意力进行高效上采样。

**核心 idea**: Transformer编码器聚合多视图信息 + Transformer上采样器生成高分辨率特征 + pixel-aligned Gaussians沿视线射线约束高斯位置。

## 方法详解

### 整体框架

GRM的pipeline：
1. **输入**: 4张稀疏视图图像 + 相机位姿（可来自多视图扩散模型）
2. **Transformer编码器**: 提取并融合多视图特征
3. **Transformer上采样器**: 渐进式上采样到原始分辨率
4. **高斯属性预测**: 线性头预测深度、旋转、缩放、透明度、SH系数
5. **反投影**: 像素对齐的3D高斯沿视线射线放置到3D空间
6. **渲染**: 通过高斯溅射实时渲染任意视角

### 关键设计

1. **Pixel-aligned Gaussians（像素对齐高斯）**:

    - **功能**: 将3D高斯的位置约束在输入视线射线上，而非自由预测3D坐标
    - **核心思路**: 每个高斯的3D位置由相机中心和射线方向确定:
    $\boldsymbol{\mu} = \mathbf{c}_o + \tau \mathbf{r}$
   其中 $\mathbf{c}_o$ 是相机中心，$\mathbf{r}$ 是射线方向，$\tau$ 是预测的深度值。每个视图预测一个 $H \times W \times 12$ 的高斯属性图（深度+旋转+缩放+透明度+SH DC项），总共生成 $V \times H \times W$ 个3D高斯
    - **设计动机**: 直接预测无结构的3D坐标使优化困难（多种配置可产生相同视觉结果），射线约束建立了像素到3D空间的直接联系，降低了学习难度

2. **Transformer编码器**:

    - **功能**: 从多视图图像提取并融合全局特征，实现跨视图信息交换
    - **核心思路**: 
        - 用Plücker embedding注入相机信息到每个像素
        - 卷积tokenizer（kernel/stride=16）提取 $H/16 \times W/16$ 局部特征
        - 所有视图特征拼接为长度 $(V \times H/16 \times W/16)$ 的序列
        - 经过24层自注意力层实现跨视图信息交换:
    $\mathbf{F} = E_{\theta, \phi}(\mathcal{I}, \mathcal{C})$
    - **设计动机**: 全局自注意力等效于跨视图特征匹配，确保不同视图对同一3D点的预测一致

3. **Transformer上采样器**:

    - **功能**: 渐进式将低分辨率特征图上采样到原始输入分辨率，恢复高频细节
    - **核心思路**: 每个上采样块包含:
        - Linear层4倍扩展通道维度
        - PixelShuffle 2倍空间上采样
        - 窗口自注意力 + 移位窗口自注意力（类Swin Transformer）:
    $\mathbf{F} = \text{PixelShuffle}(\text{Linear}(\mathbf{F}), 2)$
    $\mathbf{F} = \text{SelfAttn}(\mathbf{F}, W)$
    $\mathbf{F} = \text{Shift}(\text{SelfAttn}(\text{Shift}(\mathbf{F}, W/2), W), -W/2)$
    - **设计动机**: 编码器的patch化操作丢失了高频细节；CNN上采样器无法捕获多视图对应关系；窗口注意力平衡了计算效率和非局部信息传递

### 损失函数 / 训练策略

- **图像重建损失**: $\mathcal{L}_\text{img} = L_2(\mathbf{I}, \hat{\mathbf{I}}) + 0.5 L_p(\mathbf{I}, \hat{\mathbf{I}})$（L2 + 感知损失）
- **透明度掩码损失**: $\mathcal{L}_\text{mask} = L_2(\mathbf{M}, \hat{\mathbf{M}})$，消除浮空点
- **缩放激活函数**: 将高斯缩放限制在 $[s_{min}, s_{max}]$ 之间:
$$\mathbf{s} = s_{min} \cdot \sigma(\mathbf{s}_o) + s_{max} \cdot (1 - \sigma(\mathbf{s}_o))$$
- 训练数据: Objaverse 100K高质量物体，32张随机视角渲染
- 训练配置: 32 × A100 GPU，4天，512×512分辨率，AdamW + cosine annealing
- 使用Deferred Back-propagation优化GPU内存

## 实验关键数据

### 主实验 — 稀疏视图重建 (GSO数据集, 100物体, 64视角)

| 方法 | 输入视图数 | PSNR↑ | SSIM↑ | LPIPS↓ | 推理时间 |
|------|-----------|-------|-------|--------|---------|
| GS (优化) | 4 | 21.22 | 0.854 | 0.140 | 9 min |
| IBRNet | 16 | 21.50 | 0.877 | 0.155 | 21 sec |
| SparseNeuS | 16 | 22.60 | 0.873 | 0.132 | 6 sec |
| LGM | 4 | 23.79 | 0.882 | 0.097 | 0.07 sec |
| MV-LRM | 4 | 25.38 | 0.897 | 0.068 | 0.25 sec |
| **GRM (Ours)** | **4** | **30.05** | **0.906** | **0.052** | **0.11 sec** |

### 主实验 — 单图像到3D (GSO数据集, 250物体)

| 方法 | PSNR↑ | LPIPS↓ | CLIP↑ | FID↓ | 时间 |
|------|-------|--------|-------|------|-----|
| DreamGaussian | 19.19 | 0.171 | 0.862 | 57.6 | 2 min |
| One-2-3-45++ | 17.79 | 0.219 | 0.886 | 42.1 | 1 min |
| LGM | 16.90 | 0.235 | 0.855 | 42.1 | 5 sec |
| **GRM (Ours)** | **20.10** | **0.136** | **0.932** | **27.4** | **5 sec** |

### 消融实验 (256分辨率训练)

| 配置 | PSNR | SSIM | LPIPS | 说明 |
|------|------|------|-------|------|
| 无Sigmoid缩放激活 | 24.43 | 0.638 | 0.133 | 指数激活导致不稳定 |
| +Sigmoid缩放激活 | 27.51 | 0.900 | 0.044 | +3dB，大幅改善 |
| +1个上采样块 | 29.11 | 0.922 | 0.037 | 上采样提升细节 |
| +3个上采样块 | 29.38 | 0.917 | 0.036 | 更多块继续提升 |
| +Alpha正则化 | 29.48 | 0.920 | 0.031 | 消除浮空点 |
| Conv上采样器 (替代) | 27.23 | 0.894 | 0.063 | 远差于Transformer上采样 |
| XYZ预测 (替代) | 28.61 | 0.910 | 0.037 | 不如深度预测 |

### 关键发现

- GRM在稀疏视图重建上PSNR高达30.05，比第二名MV-LRM高出近**5 dB**，且仅需4张输入（对手用16张仍逊色）
- 推理时间仅0.11秒，生成的高斯数量是LGM的**16倍**，重建保真度大幅领先
- Transformer上采样器比CNN上采样器高出2.25 dB PSNR，证明跨视图注意力对细节重建至关重要
- Pixel-aligned（深度预测）比自由XYZ预测高0.87 dB，射线约束有效降低了学习难度
- 文本到3D生成在用户偏好研究中(29.5%)超越优化方法MVDream(25.9%)，但速度快450倍

## 亮点与洞察

- **3D表示的选择至关重要**: 用3D高斯替代triplane-NeRF同时解决了效率和质量问题，实时渲染+更高保真度
- **纯Transformer架构的优势**: 自注意力天然适合跨视图特征匹配，CNN无法捕获多视图对应关系
- **像素对齐约束的简洁智慧**: 利用相机射线约束将自由3D问题降维为1D深度估计，大幅降低学习难度
- **模型规模化的效果**: 24层编码器+4块上采样器的大规模模型在大数据(100K物体)上训练，展现出强大的泛化能力
- **模块化设计**: 重建器可与任意多视图扩散模型即插即用，实现text/image-to-3D

## 局限与展望

- 输入视图不一致时重建质量下降（来自扩散模型的生成图可能不完全一致）
- 重建器是确定性的，未来可嵌入概率框架处理多模态不确定性
- 仅适用于以物体为中心的场景，无法处理大规模场景
- 训练成本较高（32×A100 GPU，4天），难以复现
- 仅使用SH的DC项（0阶），视角相关的外观效果有限

## 相关工作与启发

- **与LRM系列的对比**: LRM使用triplane-NeRF表示，需要昂贵的体渲染；GRM使用3D高斯，实时渲染且保真度更高
- **与LGM的对比**: LGM同样用3D高斯但采用CNN U-Net架构，生成高斯数量少（GRM是其16倍），质量差距大
- **Swin Transformer思想的迁移**: 移位窗口注意力在3D重建中首次用于高效上采样，平衡效率与非局部信息
- **启发**: 对于结构化输出预测问题，将输出约束在几何合理的空间（如射线方向）可大幅简化学习

## 评分

- 新颖性: ⭐⭐⭐⭐ pixel-aligned Gaussians+纯Transformer上采样器的组合设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 稀疏视图重建+单图to3D+文本to3D+用户研究+详尽消融，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，对比公平，展示充分
- 价值: ⭐⭐⭐⭐⭐ 在速度和质量上同时大幅超越已有方法，是feed-forward 3D重建的里程碑式工作

<!-- RELATED:START -->

## 相关论文

- [GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)
- [AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation](animatabledreamer_text-guided_non-rigid_3d_model_generation_and_reconstruction_w.md)
- [LaRa: Efficient Large-Baseline Radiance Fields](lara_efficient_large-baseline_radiance_fields.md)

<!-- RELATED:END -->
