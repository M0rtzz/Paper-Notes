---
title: >-
  [论文解读] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer
description: >-
  [ICCV 2025][视频生成][Transformer] 针对 DiT 模型中 3D 全注意力机制导致的运动-外观难以解耦问题，提出共享时序核（Shared Temporal Kernel）和稠密点跟踪损失（Dense Point Tracking Loss）…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "Transformer"
  - "运动迁移"
  - "时序核"
  - "轨迹跟踪"
  - "benchmark"
---

# Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer

**会议**: ICCV 2025  
**arXiv**: [2503.17350](https://arxiv.org/abs/2503.17350)  
**代码**: [项目页面](https://shi-qingyu.github.io/DeT.github.io)  
**领域**: Video Generation / Motion Transfer  
**关键词**: 扩散 Transformer, 运动迁移, 时序核, 轨迹跟踪, benchmark

## 一句话总结

针对 DiT 模型中 3D 全注意力机制导致的运动-外观难以解耦问题，提出共享时序核（Shared Temporal Kernel）和稠密点跟踪损失（Dense Point Tracking Loss），同时建立了更全面的运动迁移基准 MTBench 和混合运动保真度指标。

## 研究背景与动机

运动迁移任务旨在将源视频的运动模式迁移到新生成的视频中，同时允许通过文本控制前景和背景外观。核心挑战在于将运动与外观解耦。

现有方法的困境：
- **基于 3D U-Net 的方法**（如 MotionDirector、SMA）利用独立的空间/时序自注意力实现解耦——冻结空间注意力只训练时序注意力。但这些方法与最新的 DiT 模型不兼容
- **DiT 模型**（如 CogVideoX、HunyuanVideo）使用 3D 全注意力机制，**不显式分离时空信息**，空间和时序维度的交互使运动-外观解耦变得极其困难
- 现有基准（如 DMT、MotionDirector benchmark）规模小且运动类型单一，难以全面评估

作者通过可视化 DiT 特征发现了关键问题：**前景和背景特征在 3D 全注意力中难以区分**，背景特征在去噪过程中存在时序不一致性，导致某些帧中前景和背景混淆，进而使背景外观与前景运动纠缠。

## 方法详解

### 整体框架

DeT 在预训练 DiT 模型上进行微调，核心包含两个组件：共享时序核用于解耦运动/外观并学习运动模式，稠密点跟踪损失用于增强前景运动一致性。推理时移除后 65% DiT 块中的时序核以提升编辑保真度。

### 关键设计

1. **共享时序核（Shared Temporal Kernel）**：

   核心观察：通过分析 3D 注意力图发现，**显著注意力分数仅出现在相邻帧的对角线上**，说明建模 DiT 特征的时序变化只需要局部时序感受野而非空间感受野。

   设计思路：对 DiT 特征 $\mathcal{I}' \in \mathbb{R}^{hw \times t \times c}$ 沿时序维度做卷积平滑。从流形学习角度，时序核等价于时序轴上的拉普拉斯平滑算子：

    $\hat{\mathcal{I}}_{xy,i} = \mathcal{I}'_{xy,i} + \sum_{j=-\frac{k-1}{2}}^{\frac{k-1}{2}} \mathcal{I}'_{xy,i+j} \times \mathcal{K}^{t}_j$

   实现上采用 down-and-up 架构以减少参数和内存：$\mathcal{K}^{t}_{down} \in \mathbb{R}^{k \times c \times m}$，$\mathcal{K}^{t}_{up} \in \mathbb{R}^{k \times m \times c}$，中间加 GELU 激活：

    $\tilde{\mathcal{I}} = \mathcal{K}^{t}_{up} * \sigma(\mathcal{K}^{t}_{down} * \mathcal{I}) + \text{Attention}(\mathcal{I}, \mathcal{E}_{text})$

   **双重作用**：（1）沿时序维度平滑使前景/背景特征在时间上一致，便于区分；（2）时序 1D 卷积有效捕获帧间变化，即运动信息，且不引入空间维度信息避免外观记忆。

2. **稠密点跟踪损失（Dense Point Tracking Loss）**：

   基于发现前景 DiT 特征在时间上应保持一致，引入显式监督增强前景运动一致性：
    - 使用 CoTracker 跟踪源视频前景，生成轨迹集 $\mathcal{T} \in \mathbb{R}^{N \times T \times 2}$ 和可见性矩阵 $\mathcal{V} \in \{0,1\}^{N \times T \times 1}$
    - 在预测的潜特征 $\hat{\mathcal{E}}(\mathcal{S})$ 上沿轨迹对齐特征，使用考虑遮挡的 L2 距离：
    $\mathcal{L}_{TL} = \|\min(\mathcal{V}(t+1), \mathcal{V}(t)) \times [\hat{\mathcal{E}}(\mathcal{S})[\mathcal{T}(t+1)] - \hat{\mathcal{E}}(\mathcal{S})[\mathcal{T}(t)]]\|_2^2$
    - 最终损失：$\mathcal{L} = \lambda_{DL} \mathcal{L}_{DL} + \lambda_{TL} \mathcal{L}_{TL}$（$\lambda_{DL}=1.0$，$\lambda_{TL}=0.1$）

3. **MTBench 基准构建**：

    - 来源于 DAVIS 和 YouTube-VOS，包含 100 个高质量视频 + 500 个评估提示
    - 使用 Qwen2.5-VL-7B 生成描述，Qwen2.5-14B 生成评估提示
    - SAM + CoTracker 标注前景轨迹，基于距离加权采样初始点（确保覆盖肢体等窄区域）
    - K-means 聚类将运动分为 Easy/Medium/Hard 三个难度级别

### 混合运动保真度指标

结合 Fréchet 距离（全局形状相似性）和速度方向余弦相似度（局部运动方向一致性）：

$$\mathcal{M}(\mathcal{T}_i, \mathcal{T}_j) = \frac{1}{N} \sum_{n=1}^{N} [\alpha \cdot e^{-d_F(\mathcal{T}_i^n, \mathcal{T}_j^n)} + (1-\alpha) \cdot \bar{c}_n]$$

其中 $\alpha = 0.5$ 平衡两个分量。

### 训练策略

在单个源视频上训练 500 步，AdamW 优化器，学习率 1e-5，权重衰减 1e-2。中间维度 128，核大小 3。推理时移除最后 27 块（65%）的时序核以提高编辑保真度。DDIM 调度器 50 步去噪，CFG scale 6.0。单卡 A100 约 1 小时完成训练。

## 实验关键数据

### 主实验：MTBench 全量结果

| 方法 | Edit Fidelity | Temporal Consistency | Motion Fidelity |
|------|--------------|---------------------|----------------|
| MotionDirector (U-Net) | 31.9 | 91.7 | 67.7 |
| SMA (U-Net) | 31.6 | 82.9 | 55.1 |
| MotionClone (U-Net) | 30.8 | 80.9 | 78.9 |
| MOFT (U-Net) | 33.0 | 91.1 | 52.5 |
| DreamBooth (CogVideoX) | 28.4 | 85.6 | 80.4 |
| MotionInversion (CogVideoX) | 26.6 | 85.4 | 85.0 |
| **DeT (CogVideoX)** | **31.2** | **89.6** | **85.8** |
| **DeT (HunyuanVideo)** | **31.9** | **91.9** | **85.9** |

### 消融实验

| 运动学习方式 | Edit Fidelity | Motion Fidelity | 说明 |
|-------------|--------------|----------------|------|
| LoRA | 28.4 | 80.4 | 全参数适配，外观过拟合 |
| Conv3D | 27.1 | 84.9 | 3D 卷积引入空间信息 |
| Local Attention | 31.3 | 73.1 | 局部注意力运动保真度差 |
| **Temporal Conv1D** | **31.6** | **85.6** | 最佳平衡 |

| 超参数 | 最优值 | 说明 |
|--------|-------|------|
| 层丢弃比例 | 65% | 推理时移除后 65% 块的时序核 |
| $\lambda_{TL}$ | 1e-1 | 跟踪损失权重 |
| 核大小 $k$ | 3 | 时序卷积核大小 |
| 中间维度 $m$ | 128 | down-up 架构中间维度 |

### 关键发现

- DeT 在运动保真度与编辑保真度之间取得最优平衡，MotionInversion 运动保真度高但严重过拟合外观
- 跨类别运动迁移（人 → 熊猫、火车 → 船）表现优秀，说明时序核有效提取了类别无关的运动模式
- MTBench 的难度分级验证了运动复杂度增加时运动保真度下降、编辑保真度提高的趋势

## 亮点与洞察

- 从 3D 注意力图的对角结构出发设计时序 1D 卷积，既有理论支撑又Implementation简洁高效
- 时序核的"平滑 + 学习"双重功能优雅地解决了解耦与运动捕获的矛盾
- 推理时选择性移除时序核（保留前 35%）是一个巧妙的工程设计，平衡了运动迁移和文本编辑
- MTBench 的构建方法（距离加权采样 + 聚类难度分级）对后续基准建设有参考价值

## 局限与展望

- 需要对每个源视频单独微调 500 步（约 1 小时），无法做到零样本运动迁移
- 仅在 49 帧视频上验证，更长视频的运动迁移效果未知
- 前景轨迹依赖 CoTracker 的跟踪质量，复杂遮挡场景可能受限
- MTBench 虽然比前作更大，但 100 个视频对于全面评估仍有局限

## 相关工作与启发

- **MotionDirector** 和 **SMA** 代表了基于 U-Net 时序注意力解耦的传统路线，DeT 首次将运动迁移推广到 DiT 架构
- 时序核的设计灵感可推广到其他需要解耦时空信息的视频编辑任务
- 稠密点跟踪损失的思路（光流在像素和特征空间的对应性）可扩展到视频风格迁移等领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 时序核用于同时解耦和学习运动的思路新颖，基准构建方法扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 DiT 基模型验证 + 详尽消融 + 新基准新指标
- 写作质量: ⭐⭐⭐⭐ 可视化分析驱动的方法设计逻辑清晰
- 价值: ⭐⭐⭐⭐ 为 DiT 时代的运动迁移提供了基础方法和评估体系

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](../../CVPR2025/video_generation/video_motion_transfer_with_diffusion_transformers.md)
- [\[ICCV 2025\] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers](magicmirror_id-preserved_video_generation_in_video_diffusion_transformers.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)

</div>

<!-- RELATED:END -->
