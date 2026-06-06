---
title: >-
  [论文解读] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image
description: >-
  [ECCV 2024][3D视觉][3D生成] 提出一种高度压缩的 triplane 潜空间自编码器，配合两阶段扩散模型（先生成 shape embedding 再生成 triplane latent），仅需 7 秒即可从单张图像生成高质量 3D 资产，且训练数据和时间远少于同类方法。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D生成"
  - "扩散模型"
  - "Triplane"
  - "潜空间压缩"
  - "单图生成3D"
---

# Compress3D: a Compressed Latent Space for 3D Generation from a Single Image

**会议**: ECCV 2024  
**arXiv**: [2403.13524](https://arxiv.org/abs/2403.13524)  
**代码**: 有 ([https://compress3d.github.io/](https://compress3d.github.io/))  
**领域**: 3D视觉  
**关键词**: 3D生成, 扩散模型, Triplane, 潜空间压缩, 单图生成3D

## 一句话总结

提出一种高度压缩的 triplane 潜空间自编码器，配合两阶段扩散模型（先生成 shape embedding 再生成 triplane latent），仅需 7 秒即可从单张图像生成高质量 3D 资产，且训练数据和时间远少于同类方法。

## 研究背景与动机

从单张图像高效生成高质量 3D 资产是一个重要但困难的任务。现有方法存在以下问题：

**优化类方法**（如 DreamFusion、Magic3D）依赖 SDS 优化，单个资产需数分钟到数小时

**学习类方法**（如 Shap-E、3DGen）在潜空间上训练扩散模型，但没有实现高度压缩，导致训练和生成速度受限

**重建类方法**（如 LRM、Zero-1-to-3）需要多视角图像或额外重建步骤
4. 现有方法仅用图像 embedding 作为条件，缺乏 3D 几何信息，生成网格质量不高

核心动机：**设计一个高度压缩的 triplane 潜空间**，并同时利用 image embedding 和 shape embedding 两种条件来提升生成质量。

## 方法详解

### 整体框架

Compress3D 包含三个核心组件，分三个阶段训练：

1. **Triplane AutoEncoder**：将彩色点云编码到低维 triplane 潜空间，再解码回高质量 3D 模型
2. **Diffusion Prior Model**：根据图像 embedding 生成 shape embedding
3. **Triplane Diffusion Model**：以 image embedding 和 shape embedding 为条件生成 triplane latent

推理流程：输入单张图像 → CLIP 提取 image embedding → Prior 模型生成 shape embedding → Triplane 扩散模型生成 latent → 解码为带纹理 3D 模型。

### 关键设计

#### 1. Triplane Encoder 的可学习投影

之前方法直接用均值池化将 3D 点特征投影到 triplane，没有可学习参数导致信息丢失。Compress3D 的改进：

- 用 PointNet 提取 3D 点特征 $F$
- 通过加权投影构建 3D 特征体 $V \in \mathbb{R}^{r \times r \times r \times c}$，权重与点到网格距离成反比
- 对特征体做归一化消除点云密度不均的影响
- 用三个方向的 **3D 卷积**获取高分辨率 triplane 特征，例如 $T_{xy} = \text{3DConv}(V^n, k=(1,1,r), s=(1,1,r))$
- 经 ResBlock + 下采样得到低分辨率 triplane latent（32×32，通道数 32）

#### 2. 3D-aware Cross-Attention

为增强低分辨率 triplane latent 的表示能力：

- 将特征体下采样到低分辨率 $V_d^n$（分辨率 32）
- 用 triplane latent 作为 query，3D 特征体作为 key/value
- 每个 triplane 位置 $(i,j)$ 只查询对应的**局部立方体区域**，计算高效
- 产生残差特征并加到原始 latent 上：$T^e = A + T^l$
- 在低分辨率特征体上查询将每步训练时间从 2.295s 降到 0.824s，性能反而微升

#### 3. Triplane Decoder 与 FlexiCubes

- 通过 ResBlock + 上采样解码为 128×128 高分辨率 triplane
- 采用 **FlexiCubes** 表示，对每个立方体预测权重、SDF 和顶点形变
- 用 dual marching cubes 提取网格，MLP 预测表面颜色
- 通过可微渲染器端到端训练，无需预计算 SDF

#### 4. Diffusion Prior Model

- 用 OpenShape 提取 shape embedding $e_s \in \mathbb{R}^{1280}$ 和 image embedding $e_i \in \mathbb{R}^{1280}$
- MLP + 跳跃连接作为扩散骨干（25.8M 参数）
- 直接预测去噪后的 $e_s$，用 L1 损失：$L_{prior} = \mathbb{E}[\|f_\theta^p(e_s^{(t)}, t, e_i) - e_s\|]$

#### 5. Triplane Diffusion Model

- UNet 骨干（864M 参数），3D-aware convolution
- Shape/image embedding 通过 cross-attention 注入
- Classifier-free guidance：各 5% dropout，推理公式为双重引导

### 损失函数 / 训练策略

**AutoEncoder 渲染损失**：

$$L_R = \lambda_1 L_{rgb} + \lambda_2 L_{mask} + \lambda_3 L_{depth} - \lambda_{kl} D_{KL}(N(\mu,\sigma) \| N(0,1))$$

- $\lambda_1{=}10, \lambda_2{=}10, \lambda_3{=}0.1, \lambda_{kl}{=}1e{-}6$
- 40 个随机视角渲染 512×512 图像监督

**数据集**：手标 2500 个好/坏模型训练 MLP 分类器过滤 Objaverse，获得 **100K** 高质量模型。

**训练配置**：AutoEncoder 8×A100 训 6 天 | Prior 2×A100 训 18 小时 | Diffusion 8×A100 训 4 天。

## 实验关键数据

### 主实验

| 指标 | Shap-E | OpenLRM | **Compress3D** |
|---|---|---|---|
| FID (↓) | 146.14 | 94.47 | **53.21** |
| CLIP Similarity (↑) | 0.731 | 0.756 | **0.776** |
| 潜空间维度 (↓) | 1.05M | 0.98M | **0.10M** |
| 单形状秒数 | 11 | 5 | 7 |
| 训练数据量 | ≥1M | 0.951M | **0.095M** |
| 训练 GPU 时 | - | 9200 | **1900** |

| 3D-aware Cross-Attention 消融 | $L_{rgb}{\times}10^3$↓ | $L_{mask}{\times}10^3$↓ | $L_{depth}{\times}10^2$↓ | 秒/步 |
|---|---|---|---|---|
| w/o attention | 3.798 | 6.953 | 2.637 | 0.789 |
| **w/ attention** | **2.485** | **5.059** | **2.095** | 0.824 |

### 消融实验

**Prior Model**：FID 从 66.46→53.21，CLIP Sim 从 0.745→0.776，在非常规视角下尤为显著。

**Guidance Scale**：最优 $s_p{=}5.0, s_s{=}1.0$，FID=53.21；过大/过小引导值均降低性能。

**特征体分辨率**：128→64→32 分辨率，重建质量微升而训练时间从 2.295s→0.961s→0.824s/步。

### 关键发现

1. 潜空间压缩到 **0.10M**（Shap-E 的 1/10），生成质量反而更好
2. 仅 9.5 万训练数据，FID 低了近 100 分
3. 训练 GPU 时仅 1900 小时（OpenLRM 的 1/5）
4. 3D-aware cross-attention 仅增 0.035s/step 但大幅降低重建损失

## 亮点与洞察

1. **压缩率极高**：潜空间从百万级压至 10 万级，证明合理架构设计比暴力增大模型更重要
2. **Shape embedding 桥梁**：图像→shape→triplane 两阶段优于直接图像→triplane，因 shape embedding 含更丰富 3D 信息
3. **局部 3D cross-attention**：triplane 查询对应局部立方体而非全局注意力，兼顾表达力与效率
4. **数据质量 > 数量**：仅 2500 标注训练过滤器，从 Objaverse 筛选 100K 高质量数据

## 局限与展望

1. 依赖 CLIP/OpenShape 预训练模型，对不在训练分布内的物体可能效果有限
2. FlexiCubes 网格分辨率（90）限制最精细几何细节
3. Shape embedding 预测有上限，极端视角下仍可能失败
4. 仅用 100K 数据训练，扩大高质量数据集可进一步提升

## 相关工作与启发

- **Shap-E**：transformer encoder 编码到隐函数参数空间，潜空间过大（1.05M）
- **3DGen**：UNet 精炼 triplane 但计算量大，本文改为投影中加入可学习参数
- **OpenShape**：shape-text-image 对齐 embedding 被巧妙用作生成条件
- 启发：**两阶段条件生成**是提升质量的通用策略

## 评分

| 维度 | 分数 (1-10) |
|---|---|
| 创新性 | 7 |
| 技术深度 | 8 |
| 实验充分性 | 8 |
| 写作质量 | 7 |
| 实用价值 | 8 |
| **总分** | **7.6** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[ECCV 2024\] CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model](crm_single_image_to_3d_textured_mesh_with_convolutional_reconstruction_model.md)
- [\[ECCV 2024\] Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting](repaint123_fast_and_high-quality_one_image_to_3d_generation_with_progressive_con.md)

</div>

<!-- RELATED:END -->
