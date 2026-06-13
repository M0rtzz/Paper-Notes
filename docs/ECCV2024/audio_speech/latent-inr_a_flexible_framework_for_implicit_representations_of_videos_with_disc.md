---
title: >-
  [论文解读] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics
description: >-
  [ECCV2024][音频/语音][Implicit Neural Representations] 提出 Latent-INR 框架，通过为视频每帧学习一个隐式 latent code 并结合 hypernetwork 进行低秩权重调制，将视频 INR 的空间与时间建模解耦，在保持压缩性能的同时赋予表征语义判别能力，支持检索、视频插帧和任意分辨率推理等多种下游任务。
tags:
  - "ECCV2024"
  - "音频/语音"
  - "Implicit Neural Representations"
  - "Video Compression"
  - "Video Retrieval"
  - "Hypernetwork"
  - "CLIP Alignment"
---

# Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics

**会议**: ECCV2024  
**arXiv**: [2408.02672](https://arxiv.org/abs/2408.02672)  
**代码**: 待确认  
**领域**: 音频语音  
**关键词**: Implicit Neural Representations, Video Compression, Video Retrieval, Hypernetwork, CLIP Alignment

## 一句话总结

提出 Latent-INR 框架，通过为视频每帧学习一个隐式 latent code 并结合 hypernetwork 进行低秩权重调制，将视频 INR 的空间与时间建模解耦，在保持压缩性能的同时赋予表征语义判别能力，支持检索、视频插帧和任意分辨率推理等多种下游任务。

## 背景与动机

- **Implicit Neural Representations (INR)** 通过为每个视频训练一个小型网络来避免泛化问题，但现有视频 INR 方法主要聚焦于压缩任务，学到的表征缺乏语义信息，无法直接用于检索等下游任务
- 传统编解码器（HEVC、H.264、AV1）虽然成熟，但基于 ML 的编解码器在实际部署中缺乏除压缩以外的额外优势，导致难以被广泛采纳
- 已有视频 INR 方法存在两个关键挑战：(1) 对长视频的架构扩展性差；(2) 每个视频都需要训练一个网络，编码时间过长
- 作者的核心思路不是直接缩短编码时间，而是证明花费的计算量是值得的——让 INR 不仅能压缩，还能支持语义下游任务

## 核心问题

如何设计一种视频 INR 框架，使其在保持良好压缩性能的同时，学到的表征具有语义判别性，能够支持检索、问答、视频插帧和任意分辨率推理等多种任务？

## 方法详解

### 整体架构：Auto-Decoder + Hypernetwork

框架由两部分组成：

1. **帧级可学习 latent 字典**：为视频的每一帧学习一个 latent code $z_t \in \mathbb{R}^D$
2. **共享 hypernetwork 集合**：给定 latent $z_t$，hypernetwork 预测帧特定的权重调制参数，用于修改共享 base network 的权重

基本公式为：

$$f_\theta((x,y)|\theta_t) = Y_t, \quad \theta_t = h(z_t)$$

其中 base network $f_\theta$ 接收空间坐标网格 $(x,y)$ 作为输入，输出对应帧。这种设计将视频的空间和时间建模分离——hypernetwork 学习视频的整体结构和风格，latent 负责条件化输出特定帧。

### 低秩权重调制

直接用 hypernetwork 预测 base network 全部权重代价太高，因此采用低秩矩阵调制：

$$\theta_t^l = \sigma(P^l \times Q^l) \cdot \theta^l, \quad h_l(z_t) = [P^l, Q^l]$$

其中 $P^l \in \mathbb{R}^{N \times r}$，$Q^l \in \mathbb{R}^{M \times r}$，$r \ll (N, M)$。秩 $r$ 和被调制的层数是控制压缩-性能权衡的超参数。这种方式类似子网络选择机制。

### 网络细节

- **Base network**：6 层 MLP，层宽 512，使用 ReLU 激活，部分变体加入卷积上采样模块（PixelShuffle）处理 patch centroid 输入
- **Hypernetwork**：每个被调制层对应一个 hypernetwork，含一个 128 维隐藏层 + tanh 非线性
- **位置编码**：使用 hash-grid 实现高质量重建（也可替换为 Fourier features 获得更快训练速度）
- **Latent 初始化**：标准正态分布 + 小方差，有助于加速收敛

### 模型压缩

端到端用 MSE loss 训练后，对所有网络参数做标准量化（指定 bit width $b$），再用 Huffman 编码进一步压缩。

### 视频插帧

对帧 latent 做线性插值生成中间帧：

$$z_{\text{inter}} = \beta_i \cdot z_t + (1 - \beta_i) \cdot z_{t-1}$$

将插值后的 latent 送入 hypernetwork 即可获得中间帧的权重调制，支持 $\alpha \in \{2, 4, 8\}$ 倍插帧。

### 语义对齐

- **CLIP 对齐（检索）**：在训练损失中加入 latent 与 CLIP 图像嵌入的余弦相似度损失 $L = L_{\text{MSE}} + \lambda \cdot L_{\text{clip}}$，$\lambda = 0.01$
- **VideoLlama 对齐（问答/Chat）**：将 latent 字典作为 token 投影到 VideoLlama 空间，替代原始视频输入 token，支持开放式文本对话

## 实验关键数据

### 视频压缩（UVG 数据集）

- 在 PSNR/BPP 率失真曲线上与 NVP（当时 SOTA）性能相当，同时解码 FPS 更优
- 支持 **任意分辨率推理**：同一模型无需修改即可在不同分辨率下解码（传统编码器需对每个分辨率单独编码）

### 视频插帧

| 数据集 | $\alpha$ | NeRV | NIRVANA | NVP | **Ours** |
|--------|----------|------|---------|-----|----------|
| Bunny  | 2        | 15.92| 19.14   |20.10| **33.17**|
| Bunny  | 4        | 15.43| 18.90   |19.11| **28.08**|
| Bunny  | 8        | 13.68| 18.67   |18.08| **25.88**|
| TaiChi | 2        | 16.91| 18.19   |19.33| **35.13**|
| TaiChi | 8        | 15.72| 16.21   |17.70| **27.72**|

相比其他 INR 方法有巨大优势（$\alpha=2$ 时超出 NVP 约 13-16 dB）。

### 视频检索

- 在 MSR-VTT 上 T2V R@1=30.2，与直接使用 CLIP 特征（30.1）基本持平
- 在 COIN 类级别检索上 R@1=34.4 甚至超过 CLIP 的 31.6
- 在 ActivityNet 上长视频检索也与 CLIP 特征表现一致

### CLIP $\lambda$ 消融

$\lambda=0.01$ 为最佳平衡点：PSNR 仅从 30.03 降至 29.46（-0.57 dB），但 T2V R@1 从 0.1 飙升至 30.2。

## 亮点

- **一体多用**：首次在同一视频 INR 框架中同时实现压缩、超分辨率、任意分辨率推理、视频插帧、文本检索和视频问答
- **latent 空间语义丰富**：UMAP 可视化表明即使仅用压缩目标训练，帧 latent 也能捕捉场景语义（重复模式聚集、动态场景沿轨迹展开）
- **插帧性能碾压**：latent 空间的线性插值特性使其在视频插帧上远超其他 INR 方法
- **灵活对齐**：latent 可与任意大模型（CLIP、VideoLlama）对齐，具有良好的可扩展性

## 局限与展望

- 编码时间仍然较长（每个视频需单独训练整个系统），未从根本上解决 INR 的训练效率问题
- 压缩性能虽可比，但并未显著超越 NVP 等方法，额外的语义能力需要额外计算代价
- Chat 功能的质量主要受限于对齐的 LLM 本身（VideoLlama 局限性），实验结果多为定性展示
- 仅在 UVG（7 个视频）上评估压缩，数据集规模有限
- 领域分类为 audio_speech 可能不准确，本文核心是视频表征与压缩

## 与相关工作的对比

| 方法 | 压缩 | 插帧 | 检索 | 任意分辨率 | Chat |
|------|------|------|------|-----------|------|
| NeRV | ✓ | ✗ | ✗ | ✗ | ✗ |
| NIRVANA | ✓ | ✗ | ✗ | ✗ | ✗ |
| NVP | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Latent-INR** | ✓ | ✓ | ✓ | ✓ | ✓ |

- 与 NeRV/NVP/NIRVANA 的核心区别在于引入了 auto-decoder 框架和 latent 字典，将时空解耦
- 低秩调制思路与 LoRA 类似，但应用于 INR 场景，且 latent 不是冗余的中间表征而是核心信息载体
- 与传统编解码器相比，最大优势是同一模型支持任意分辨率和语义任务

## 启发与关联

- latent 作为权重代理（proxy for weights）的思路值得关注——通过在 latent 空间上操作来间接操控网络输出，是一种优雅的设计范式
- CLIP 对齐损失仅用 $\lambda=0.01$ 即可获得良好检索性能且几乎不损害重建质量，说明语义信息和重建信息在表征空间中可以共存
- 任意分辨率推理是对传统编解码器的显著优势，在实际流媒体场景（自适应码率）中有潜在价值
- 可关注后续是否有工作将此框架扩展到更大规模视频数据集或引入更高效的训练策略

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在视频 INR 中融合压缩与语义任务，框架设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖压缩/插帧/检索/Chat 四大任务，有消融和可视化，但压缩评估数据集偏小
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 为 ML-based codec 的实际采纳提供了有说服力的论据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning](../../ICML2026/audio_speech/position_text_embeddings_should_capture_implicit_semantics_not_just_surface_mean.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[ACL 2025\] In-the-wild Audio Spatialization with Flexible Text-guided Localization](../../ACL2025/audio_speech/tas_audio_spatialization.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)

</div>

<!-- RELATED:END -->
