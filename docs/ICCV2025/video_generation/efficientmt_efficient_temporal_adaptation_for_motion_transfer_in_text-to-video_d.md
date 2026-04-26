---
title: >-
  [论文解读] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models
description: >-
   提出 EfficientMT，一个高效的端到端视频运动迁移框架，通过复用预训练 T2V 模型骨干提取时序运动特征，结合 scaler 模块和时序集成机制，仅用少量合成配对数据即可实现零样本运动迁移，推理时间较优化方法提速 10 倍以上。
tags:

---

# EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models

## 论文信息

- **会议**: ICCV 2025
- **arXiv**: 2503.19369
- **代码**: [https://github.com/PrototypeNx/EfficientMT](https://github.com/PrototypeNx/EfficientMT)
- **领域**: 视频生成 / 运动迁移
- **关键词**: motion transfer, text-to-video, diffusion model, temporal attention, end-to-end

## 一句话总结

提出 EfficientMT，一个高效的端到端视频运动迁移框架，通过复用预训练 T2V 模型骨干提取时序运动特征，结合 scaler 模块和时序集成机制，仅用少量合成配对数据即可实现零样本运动迁移，推理时间较优化方法提速 10 倍以上。

## 研究背景与动机

视频运动迁移旨在将参考视频的运动模式迁移到不同主体和场景上。现有方法分为两类：

**密集视觉条件方法**（VideoComposer, Control-A-Video 等）：使用深度图、光流等密集条件端到端训练，推理高效但结构约束过强，难以跨场景迁移

**优化方法**（MotionDirector, MotionClone, MOFT 等）：从 T2V 模型中提取隐式运动表征，效果好但需要针对每个参考视频做样本级优化（数分钟到数十分钟）

核心矛盾在于：端到端方法高效但迁移能力弱，优化方法迁移好但计算开销大。EfficientMT 的目标是结合两者优势——用隐式运动表征实现鲁棒迁移，同时保持端到端推理效率。

## 方法详解

### 整体框架

EfficientMT 基于 AnimateDiff / VideoCrafter2 两个 T2V 基线模型，包含三个核心组件：

1. **运动特征提取**：复用 T2V 骨干作为参考视频的特征提取器
2. **Scaler 模块**：对参考特征进行细粒度缩放，过滤运动无关信息
3. **时序集成机制**：将参考运动特征注入生成过程的时序注意力层

### 关键设计 1：运动表征提取

直接复用预训练 T2V 模型骨干 $\hat{\epsilon}$ 提取参考视频的时序注意力层输入作为运动特征：

$$\Gamma = \hat{\epsilon}(x, t, \tau_\theta(y^r)) = \{f\}_{upblocks}^{temporal}$$

其中 $t$ 设为去噪后期时间步，参考提示 $y^r$ 设为空文本以更好捕获运动动态。这种设计的优势在于提取的特征与生成过程天然对齐，无需额外训练特征提取器。

### 关键设计 2：Scaler 模块

参考特征中包含大量运动无关信息（如纹理、形状），直接注入会导致过拟合。Scaler 预测一个细粒度的缩放图：

$$\alpha = \mathcal{S}(f^r \otimes f^g) \in \mathbb{R}^{h \times w \times n \times 1}$$

其中 $f^r, f^g$ 分别是参考特征和生成特征，$\otimes$ 为通道拼接。缩放后的特征 $\hat{f}^r = \alpha \cdot f^r$ 自适应保留运动信息并抑制无关信息。

可视化分析表明：在低分辨率块中，scaler 倾向于全选或全弃（scale 接近 0 或 1）；在高分辨率块中，scaler 选择性关注运动相关区域（如移动的头部、行走的腿部）。

### 关键设计 3：时序集成机制

将过滤后的参考特征沿时间轴与生成特征拼接：

$$f^{int} = f^g \otimes \hat{f}^r \in \mathbb{R}^{h \times w \times 2n \times c}$$

Query 从原始 $f^g$ 投影，Key 和 Value 从集成特征 $f^{int}$ 投影：

$$Q = W_Q \cdot f^g, \quad K = W_K \cdot f^{int}, \quad V = W_V \cdot f^{int}$$

这等价于将参考帧视为额外的视频帧，在时序注意力中提供运动引导。训练时仅微调所有上采样块的时序注意力层。

### 损失函数

使用标准扩散损失：

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y), \Gamma)\|_2^2\right]$$

### 训练数据构建

收集 217 个真实参考视频，使用 MotionClone 和 MotionInversion 合成配对运动迁移数据。经运动对齐度（motion fidelity score）和时间一致性（CLIP 帧间特征）双重筛选 + 人工选择，最终仅保留约 150 个高质量样本。

## 实验关键数据

### 主实验：定量对比

| 方法 | Temporal Consistency ↑ | Text Alignment ↑ | Motion Fidelity ↑ | Time Cost ↓ |
|------|----------------------|-----------------|-------------------|------------|
| ControlVideo | 0.9213 | 0.2483 | 0.5533 | 80s |
| VideoComposer | 0.9192 | 0.2635 | 0.6356 | 18s |
| MotionDirector | 0.9327 | 0.2525 | 0.8361 | 473s |
| MotionClone | 0.9108 | 0.2637 | 0.8569 | 190s |
| MOFT | 0.9283 | 0.2581 | 0.7698 | 127s |
| **Ours (AnimateDiff)** | **0.9291** | **0.2712** | **0.8470** | **16s** |
| DMT | 0.9275 | 0.2479 | 0.6974 | 203s |
| MotionInversion | 0.9329 | 0.2558 | 0.7373 | 418s |
| **Ours (VideoCrafter2)** | **0.9456** | **0.2677** | **0.7116** | **21s** |

EfficientMT 在文本对齐度上取得最佳，运动保真度与优化方法相当，时间开销仅 16-21 秒（优化方法需 127-473 秒）。

### 消融实验

| 配置 | TC ↑ | TA ↑ | MF ↑ |
|------|------|------|------|
| w/o Scaler | 0.9244 | 0.2638 | 0.8135 |
| w/o Data Filter | 0.9237 | 0.2649 | 0.8278 |
| Inject Upblock.1 | 0.9013 | 0.2789 | 0.6374 |
| Inject Upblock.1,2 | 0.9198 | 0.2755 | 0.7236 |
| **Full Model** | **0.9291** | **0.2712** | **0.8470** |

**关键发现**：
- 移除 Scaler 导致运动保真度下降（0.847→0.814），生成出现伪影
- 仅注入单个上采样块（upblock1）运动保真度急剧下降至 0.637
- 数据质量比数据规模更重要：不做数据清洗反而降低所有指标

### 用户研究

与 VideoComposer(VC)、MotionDirector(MD)、MotionClone(MC)、MotionInversion(MI) 对比，EfficientMT 在时间一致性、文本对齐和运动保真度三方面均获得多数用户偏好（55-80%）。

## 亮点与洞察

1. **高效复用策略**：复用 T2V 骨干作为特征提取器，无需额外编码器，训练参数量小
2. **少量数据即可训练**：仅需约 150 个高质量合成配对样本即可完成训练
3. **10 倍以上提速**：16 秒 vs 优化方法的 127-473 秒，使运动迁移具备实际应用价值
4. **Scaler 的可视化分析**：揭示了不同分辨率层对运动信息的差异化处理模式

## 局限性

- 在剧烈运动变化场景下（如快速旋转、突然位移），生成结果容易出现撕裂伪影
- 运动迁移质量依赖参考视频的运动明确程度
- 基于 AnimateDiff/VideoCrafter2，未验证在更新架构（如 DiT-based T2V）上的效果

## 相关工作与启发

- 与 MotionClone/MotionInversion 的关系：用它们生成训练数据（teacher），训练端到端模型（student），本质上是知识蒸馏
- Scaler 模块的设计灵感来自 SmartControl，通过细粒度缩放替代粗粒度的全局注入强度调整
- 时序注意力的 KV 拼接机制与 IP-Adapter 的空间注意力注入思路类似，但作用在时间维度

## 评分

⭐⭐⭐⭐ — 实用性极强的工作，将运动迁移从分钟级优化压缩到秒级推理，少量训练数据的设计也很优雅。局限在于仅验证了两个较旧的 T2V 基线，缺乏对最新视频生成架构的适配。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)

<!-- RELATED:END -->
