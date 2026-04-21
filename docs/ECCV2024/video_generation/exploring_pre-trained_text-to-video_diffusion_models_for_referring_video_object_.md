---
title: >-
  [论文解读] Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation
description: >-
  [ECCV 2024][文本到视频扩散模型] 本文首次探索预训练文本到视频（T2V）扩散模型的视觉特征用于视频理解任务，提出 VD-IT 框架，通过文本引导的图像投影和视频特定噪声预测两项关键设计，从固定的 T2V 扩散模型中提取具有优越时序语义一致性的视觉特征，在 R-VOS 四大基准上超越了使用判别式预训练视频骨干网络（如 Video Swin Transformer）的 SOTA 方法。
tags:
  - ECCV 2024
  - 文本到视频扩散模型
  - 指代视频目标分割
  - 时序一致性
  - 视觉特征提取
  - 生成式预训练
---

# Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation

**会议**: ECCV 2024  
**arXiv**: [2403.12042](https://arxiv.org/abs/2403.12042)  
**代码**: https://github.com/buxiangzhiren/VD-IT (有)  
**领域**: Segmentation / Video Understanding  
**关键词**: 文本到视频扩散模型, 指代视频目标分割, 时序一致性, 视觉特征提取, 生成式预训练

## 一句话总结
本文首次探索预训练文本到视频（T2V）扩散模型的视觉特征用于视频理解任务，提出 VD-IT 框架，通过文本引导的图像投影和视频特定噪声预测两项关键设计，从固定的 T2V 扩散模型中提取具有优越时序语义一致性的视觉特征，在 R-VOS 四大基准上超越了使用判别式预训练视频骨干网络（如 Video Swin Transformer）的 SOTA 方法。

## 研究背景与动机
预训练文本到图像扩散模型的内部表征已被证明对图像理解任务有价值，但文本到视频（T2V）扩散模型在视频理解任务中的潜力尚未被充分探索。视频理解比图像理解更具挑战性，因为需要同时处理空间和时序信息。现有 R-VOS 方法普遍使用判别式预训练的视频骨干（如 Video Swin Transformer），这些骨干虽然经过分类任务预训练，但在时序语义一致性方面存在固有缺陷——容易受光照变化和相机运动影响导致帧间特征漂移。核心矛盾在于：判别式预训练关注帧级别的分类特征，而视频分割需要跨帧的语义一致性。本文的切入角度是观察到 T2V 扩散模型在训练中使用全局文本提示引导生成语义一致的视频帧序列，因此其内部表征天然蕴含了丰富的时序一致性先验。核心 idea 可概括为"what I cannot create, I do not understand"——能生成连贯视频的模型必然理解时序一致性。

## 方法详解

### 整体框架
VD-IT 建立在开源的 ModelScopeT2V 扩散模型之上，包含两大核心组件：(1) 视觉特征提取——利用固定的 T2V 扩散模型 U-Net 的中间层特征，输入为噪声添加后的视频和文本-图像融合的提示嵌入；(2) 掩码预测头——基于 query 的分割模型设计，从指代文本中提取实例 query，与视觉特征融合生成最终的分割掩码。T2V 模型的参数在训练中保持冻结。

### 关键设计
1. **文本引导的图像投影（Text-Guided Image Projection）**:
    - 功能：生成同时包含时序语义一致性和帧级细节的条件提示嵌入
    - 核心思路：对每帧用 CLIP 视觉模型提取视觉 tokens $p_{v,t}$，用 T2V 模型自带的文本编码器提取指代文本 tokens $p_e$，通过交叉注意力机制将两者融合：$p_{ve,t} = \mathrm{MLP}(p_e + \mathrm{SoftMax}(\frac{p_e W^Q (p_{v,t} W^K)^T}{\sqrt{d_k}}) p_{v,t} W^V)$，其中文本 tokens 为 query，图像 tokens 为 key/value
    - 设计动机：仅用文本（VD-T）导致低层特征缺乏细粒度细节，掩码边界不精确；仅用图像（VD-I）导致高层特征缺乏语义区分，容易混淆不同实例。两者结合（VD-IT）兼顾了时序语义一致性和空间细节丰富性

2. **视频特定噪声预测（Video-specific Noise Prediction）**:
    - 功能：用可学习的视频相关噪声替代标准高斯噪声，保留提取特征的保真度
    - 核心思路：将视频潜在向量 $\mathcal{F}_o$ 通过卷积层得到 $\mathcal{F}_n$，然后对其进行归一化生成预测噪声：$n_t = (f_{n,t} W^N - \mu(f_{n,t} W^N)) / (\sigma(f_{n,t} W^N) + \epsilon)$，其中 $W^N \in \mathbb{R}^{4 \times 4}$ 为可训练权重。最终扩散输入为 $\alpha_0 \mathcal{F}_o + (1-\alpha_0) \mathcal{N}$，使用 step=0 以最小化噪声强度
    - 设计动机：标准高斯噪声与输入视频无关，会模糊关键细节；而视频相关噪声能更好地保留原始信号中的结构信息

3. **掩码预测头（Mask Prediction Head）**:
    - 功能：将视觉特征和文本信息融合生成分割掩码
    - 核心思路：使用 $Q$ 个可学习的实例 query 向量与 RoBERTa 提取的文本特征做交叉注意力得到实例 query $q_e$；然后通过 Deformable Transformer 编码器处理多尺度视觉特征，解码器中以实例 query 为 query、视觉特征为 key/value 得到跨模态特征；最后分别输入分类头、边界框头和动态卷积掩码头生成预测
    - 设计动机：沿用成熟的 query-based 分割设计，使本文的贡献聚焦在特征提取端，便于公平对比

### 损失函数 / 训练策略
使用匈牙利算法从 $Q$ 个实例 query 中匹配目标对象。训练损失包括：Dice loss 和 Focal loss 用于掩码 $\mathcal{M}$，Focal loss 用于置信度分数 $\mathcal{S}$，L1 和 GIoU loss 用于边界框 $\mathcal{B}$。训练使用 2 块 A100 GPU，每个 clip 5 帧，共 9 个 epoch。T2V 扩散模型参数冻结，仅训练投影层、噪声预测模块和掩码预测头。

## 实验关键数据

### 主实验
**Ref-YouTube-VOS & Ref-DAVIS17**：

| 方法 | 骨干 | YouTube $\mathcal{J}\&\mathcal{F}$ | YouTube $\mathcal{J}$ | YouTube $\mathcal{F}$ | DAVIS $\mathcal{J}\&\mathcal{F}$ |
|------|------|------|------|------|------|
| SgMg | V-Swin-T | 58.9 | 57.7 | 60.0 | 56.7 |
| SgMg | V-Swin-B | 61.6 | 59.7 | 63.5 | - |
| **VD-IT** | **Video Diffusion** | **64.8** | **63.1** | **66.6** | **63.0** |

预训练 RefCOCO/+/g 后：

| 方法 | 骨干 | YouTube $\mathcal{J}\&\mathcal{F}$ | DAVIS $\mathcal{J}\&\mathcal{F}$ |
|------|------|------|------|
| SgMg | V-Swin-B | 65.7 | 63.3 |
| **VD-IT** | **Video Diffusion** | **66.5** | **69.4** |

**A2D-Sentences & JHMDB-Sentences**：

| 方法 | A2D mAP | A2D Overall | JHMDB mAP | JHMDB Overall |
|------|---------|-------------|-----------|---------------|
| SgMg (V-Swin-B) | 58.5 | 79.9 | 45.0 | 73.7 |
| **VD-IT** | **61.4** | **81.5** | **46.5** | **74.4** |

### 消融实验
**组件消融**（Ref-YouTube-VOS, $\mathcal{J}\&\mathcal{F}$）：

| Image-Cond | Text-Cond | Noise Pred | $\mathcal{J}\&\mathcal{F}$ | 说明 |
|:---:|:---:|:---:|------|------|
| ✓ | | | 59.7 | VD-I：细节好但实例混淆 |
| | ✓ | | 61.9 | VD-T：时序一致但细节差 |
| ✓ | ✓ | | 63.8 | VD-IT (无噪声预测)：兼顾两者 |
| ✓ | ✓ | ✓ | **64.8** | 完整 VD-IT |

**时序一致性分析**（帧间 IoU 差异↓）：

| 方法 | 间隔1帧 | 间隔5帧 |
|------|---------|---------|
| SgMg (V-Swin) | 7.24 | 11.15 |
| VD-I | 6.52 | 9.43 |
| VD-IT | **5.19** | **7.89** |

### 关键发现
- VD-IT 在 Ref-YouTube-VOS 上超越之前 SOTA（SgMg with V-Swin-T）3.2 个 $\mathcal{J}\&\mathcal{F}$ 点，在 Ref-DAVIS17 with RefCOCO 预训练后超越 6.1 个点
- 仅用图像条件（VD-I）的性能不如使用 V-Swin 的 SgMg，说明 SOTA 性能并非来自模型容量或数据曝光量的增加
- 文本引导是关键——VD-T 比 VD-I 高 2.2 个点，验证了指代文本在特征提取中的重要性
- VD-IT 的帧间 IoU 差异比 SgMg 低约 2 个点（间隔1帧：5.19 vs 7.24），定量证明了 T2V 扩散特征的时序一致性优势
- 在 RefCOCO 图像分割任务上，VD-IT 与 SgMg 性能相近，说明优势主要来自时序一致性而非单帧质量

## 亮点与洞察
- **范式创新**：首次系统验证了"生成式 T2V 模型的特征可以用于视频理解"这一假说，为视频分析提供了全新的特征提取范式
- **深刻的特征分析**：通过 K-Means 聚类可视化、余弦相似度曲线和光照鲁棒性实验，多维度证明了扩散特征的时序优势来自于全局文本条件引导和内在的去噪鲁棒性
- **VD-I vs VD-T vs VD-IT 的层次分析**：清晰展示了低层特征需要视觉细节、高层特征需要语义一致的互补关系，这对后续工作有指导价值
- T2V 扩散模型固定参数，仅训练轻量级模块，训练效率较高

## 局限与展望
- 推理速度较慢（21 FPS），远低于 SgMg（65 FPS），因为需要扩散模型前向传播提取特征
- 仅使用了 ModelScopeT2V 这一个 T2V 模型，未探索更新的模型（如 Stable Video Diffusion）是否能带来进一步提升
- 掩码预测头的设计较为标准，没有针对扩散特征进行专门优化
- 固定的扩散时间步（step=0）可能不是所有场景的最优选择，多步特征融合可能进一步提升
- 扩散 U-Net 的多尺度特征只用了 3 个层级，还有压缩空间未被充分利用

## 相关工作与启发
- 与 VPD、OVDiff 等利用预训练 T2I 扩散模型做图像理解的工作一脉相承，但本文扩展到了视频维度，填补了 T2V 扩散模型用于视频理解的空白
- SgMg 是最强 baseline，两者的差异主要在于视觉骨干（V-Swin vs T2V Diffusion），掩码预测头设计相近，因此对比公平
- 启发 1：T2V 扩散模型的时序一致性先验可能对其他视频理解任务（如视频目标跟踪、动作识别）同样有价值
- 启发 2：生成模型和判别模型的统一趋势——生成模型的内部表征不仅能"创造"还能"理解"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次探索 T2V 扩散模型用于视频理解，假设驱动的研究思路令人耳目一新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、系统消融、时序一致性分析、光照鲁棒性实验、特征可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机论证有说服力，VD-I/VD-T/VD-IT 的渐进分析引人入胜，但部分段落略冗长
- 价值: ⭐⭐⭐⭐⭐ 开辟了"生成式 T2V 模型用于视频理解"的新方向，对领域有重要启示意义

<!-- RELATED:START -->

## 相关论文

- [FreeInit: Bridging Initialization Gap in Video Diffusion Models](freeinit_bridging_initialization_gap_in_video_diffusion_models.md)
- [VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [Kalman-Inspired Feature Propagation for Video Face Super-Resolution](kalman-inspired_feature_propagation_for_video_face_super-resolution.md)
- [Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)
- [MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)

<!-- RELATED:END -->
