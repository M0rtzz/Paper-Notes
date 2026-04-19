---
title: >-
  [论文解读] GPS as a Control Signal for Image Generation
description: >-
  [CVPR 2025][image generation] 将照片 EXIF 中的 GPS 标签作为扩散模型的控制信号，实现基于地理位置的可控图像生成，并通过 GPS-to-3D 的 SDS 管线从 2D 模型提取 3D 地标重建。
tags:
  - CVPR 2025
  - GPS conditioning
  - diffusion model
  - 3D reconstruction
  - score distillation sampling
  - 地理位置控制
---

# GPS as a Control Signal for Image Generation

**会议**: CVPR 2025  
**arXiv**: [2501.12390](https://arxiv.org/abs/2501.12390)  
**代码**: [项目页面](https://cfeng16.github.io/gps-gen/)  
**领域**: image_generation / 3d_vision  
**关键词**: GPS conditioning, diffusion model, geotagged photos, NeRF, score distillation sampling, compositional generation

## 一句话总结

将照片 EXIF 元数据中的 GPS 坐标作为扩散模型的新型控制信号，训练 GPS+文本联合条件的图像生成模型，能捕捉城市内不同街区/地标的细粒度外观差异，并通过角度条件 SDS 从 2D 模型提取 3D 地标重建。

## 研究背景与动机

**领域现状**: 扩散模型已广泛使用文本、深度图、语义掩码、相机位姿等条件进行图像/视频/3D 生成。地理标签（GPS）是照片元数据中海量可用但被忽视的信号。

**现有痛点**:
- 文本条件无法精确控制场景的地理位置特征（如特定街区的建筑风格）
- 传统 SfM→NeRF 管线在无序旅游照片上容易因位姿估计失败而崩溃
- 已有的 GPS→图像工作仅限于卫星图像且需要校准训练数据

**核心矛盾**: GPS 隐含了丰富的视觉先验（地标位置、建筑风格、视角信息），但目前没有生成模型能利用这一信号。

**本文目标**: 证明 GPS 标签是图像生成的有用控制信号，并展示其在组合生成和 3D 重建中的应用。

**切入角度**: 在预训练 Stable Diffusion 上微调，将 GPS 坐标编码后拼接到 CLIP 文本嵌入中作为额外条件 token。

**核心 idea**: GPS 坐标提供了互补文本的位置先验，不仅能让生成模型捕捉城市内的细粒度地理变化，还能为 3D 重建提供隐式视角监督。

## 方法详解

### 整体框架

1. **数据收集**: 从 Flickr 收集带 GPS 标签的旅游照片（曼哈顿 50 万张，巴黎 31 万张）
2. **GPS-to-Image 扩散模型**: 在 SD v1.4 上微调，联合 GPS 和文本条件
3. **Angle-to-Image 扩散模型**: 针对特定地标，用方位角替代 GPS 作为条件
4. **GPS-guided 3D 重建**: 通过 SDS 从 angle-to-image 模型中提取 NeRF

### 关键设计

#### 1. GPS 条件编码

- 将 GPS 坐标 $(x, y)$（经纬度）归一化到 $[-1, 1]$
- 使用频率为 10 的位置编码 + 两层 MLP 编码为 $\mathbf{g} = [f(x), f(y)] \in \mathbb{R}^{2 \times D}$
- 将 GPS 嵌入 **拼接到 CLIP 文本 token 序列末尾**，作为"GPS" CLIP 文本条件
- 训练时随机丢弃条件：5% 仅文本 + 5% 仅 GPS + 5% 无条件

#### 2. 双条件 Classifier-Free Guidance

推理时采用 InstructPix2Pix 风格的双条件 CFG：

$$\tilde{\boldsymbol{\epsilon}}_\phi = \boldsymbol{\epsilon}_\phi(\varnothing, \varnothing) + \omega_{\mathbf{p}}(\boldsymbol{\epsilon}_\phi(\mathbf{p}, \varnothing) - \boldsymbol{\epsilon}_\phi(\varnothing, \varnothing)) + \omega_{\mathbf{g}}(\boldsymbol{\epsilon}_\phi(\mathbf{p}, \mathbf{g}) - \boldsymbol{\epsilon}_\phi(\mathbf{p}, \varnothing))$$

三次前向分别用于无条件、仅文本、文本+GPS，通过权重 $\omega_\mathbf{p}$ 和 $\omega_\mathbf{g}$ 分别控制语义和地理引导强度。

#### 3. GPS-guided 3D 地标重建

- 将 GPS 参数化为相对于地标中心的**方位角** $\alpha = \arctan\frac{x-x_o}{y-y_o}$
- 训练 angle-to-image 扩散模型（每个地标单独训练）
- 加入 DreamBooth 风格的 **prior preservation loss** 防止微调过拟合
- 用 SDS 损失驱动 NeRF 优化：每次渲染随机视角 → 计算方位角 → GPS 条件生成 → SDS 梯度回传
- GPS 条件替代传统的 view-dependent prompting，提供更准确的视角先验，避免 Janus 问题

### 损失函数

- GPS-to-image 训练：$\mathcal{L}_{recon} = \mathbb{E}[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\phi(\mathbf{z}_t; \mathbf{p}, \mathbf{g}, t)\|_2^2]$
- 地标 3D 重建：$\mathcal{L} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{preservation}$，$\lambda = 1.0$

## 实验关键数据

### 主实验表（GPS-to-Image 生成）

| 方法 | CLIP Score ↑ | GPS Score ↑ | Avg ↑ |
|------|-------------|-------------|-------|
| GPS NN | 18.77 | **13.66** | 16.22 |
| SD (Text+Address) | **26.65** | 4.25 | 15.45 |
| SD (Text) | 29.13 | 1.21 | 15.17 |
| **Ours** | 27.88 | 8.15 | **18.02** |
| Ours (w/o text) | – | 13.71 | – |

综合 CLIP Score 和 GPS Score，本方法最优。

### 3D 地标重建对比

| 方法 | CLIP Score ↑ | PQ ↑ | Tourist Score ↑ |
|------|-------------|------|----------------|
| NeRF (SfM-based) | 20.57 | 1.32 | 1.36 |
| DreamFusion | 29.49 | 2.21 | 2.09 |
| **Ours** | **31.87** | **3.31** | **3.45** |

### 消融表

- **Angle-to-Image 方位角精度**: 本方法 22.36% vs. SD 3.06% vs. 随机 2.78%
- **GPS vs. Text Address**: GPS Score 8.15 vs. 4.25，连续 GPS 远优于文本地址
- **Prior preservation loss**: 移除后 3D 重建质量显著下降

### 关键发现

1. GPS 条件与文本条件高度互补：文本控制语义，GPS 控制地理外观（注意力图可视化验证）
2. 连续 GPS 坐标远优于离散化的地址名称作为条件
3. GPS 条件有效缓解 DreamFusion 的 Janus 多面问题
4. SfM-based 方法在 6 个地标中 3 个完全失败，而 GPS-guided SDS 全部成功
5. 平均图像（compositional generation）能捕捉特定街区的建筑风格

## 亮点与洞察

1. **全新的控制信号**: 首次系统性地将 GPS 作为图像生成的条件信号，开辟新方向
2. **一石二鸟**: 同一个 GPS 条件既可用于可控生成，又可为 3D 重建提供视角先验
3. **数据易获取**: 利用照片 EXIF 中已有的 GPS 标签，无需额外标注，信号丰富且免费
4. **组合生成能力**: "superman"在 MoMA 生成雕塑，在时代广场生成 Cosplay 人物，展示了 GPS+文本的组合语义
5. **Average Image**: 通过对一个区域多个 GPS 的噪声估计取平均，生成该区域的"代表性图像"，捕捉建筑风格

## 局限与展望

1. 依赖大量 GPS 标注的照片集合，对 GPS 数据稀疏的区域效果有限
2. SDS 生成的 3D 模型存在颜色饱和问题（SDS 的固有局限）
3. GPS 标签中蕴含的语义信息难以与文本完全解耦
4. 仅在曼哈顿和巴黎两个城市验证，泛化到其他地理区域未验证
5. 基于 SD v1.4，分辨率和质量受基础模型限制，升级到更新模型可能效果更好

## 相关工作与启发

1. **DreamFusion** (Poole et al., 2022): SDS 框架的来源，本文扩展为 GPS-guided SDS
2. **InstructPix2Pix** (Brooks et al., 2023): 双条件 CFG 推理策略的来源
3. **Snavely et al. (2006)**: Photo Tourism，从地理标签照片集重建 3D，经典先驱工作
4. **DreamBooth** (Ruiz et al., 2023): Prior preservation loss 防止微调过拟合

**启发**: 照片元数据（不仅是 GPS，还有时间戳、相机参数等）可能包含更多未被利用的控制信号。GPS→Image 的思路可扩展到街景生成、自动驾驶场景合成等应用。

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐⭐ — 全新控制信号，问题定义本身就是贡献
- **实验充分度**: ⭐⭐⭐⭐ — 生成+3D 两个任务，定性定量+用户研究，但城市数量偏少
- **论文写作**: ⭐⭐⭐⭐ — 图示清晰，展示效果令人印象深刻
- **实用价值**: ⭐⭐⭐ — 应用场景明确但相对小众（旅游照片/地标重建）

<!-- RELATED:START -->

## 相关论文

- [Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)
- [LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](../../ICCV2025/image_generation/larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment](../../ECCV2024/image_generation/dreamdiffusion_high-quality_eeg-to-image_generation_with_temporal_masked_signal_.md)
- [Collaborative Control for Geometry-Conditioned PBR Image Generation](../../ECCV2024/image_generation/collaborative_control_for_geometry-conditioned_pbr_image_generation.md)

<!-- RELATED:END -->
