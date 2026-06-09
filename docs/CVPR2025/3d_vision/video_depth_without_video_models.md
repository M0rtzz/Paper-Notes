---
title: >-
  [论文解读] Video Depth Without Video Models
description: >-
  [CVPR 2025][3D视觉][视频深度估计] 本文提出 RollingDepth，不使用视频扩散模型而是将单帧潜在扩散模型（Marigold）扩展为多帧 snippet 处理器，配合多尺度膨胀采样和鲁棒全局对齐算法，将短片段深度拼接为时序一致的长视频深度，在多个基准上超越了专门的视频深度模型和单帧模型。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "视频深度估计"
  - "单目深度"
  - "潜在扩散模型"
  - "时序一致性"
  - "滚动推理"
---

# Video Depth Without Video Models

**会议**: CVPR 2025  
**arXiv**: [2411.19189](https://arxiv.org/abs/2411.19189)  
**代码**: [https://rollingdepth.github.io](https://rollingdepth.github.io)  
**领域**: 3D视觉 / 深度估计  
**关键词**: 视频深度估计, 单目深度, 潜在扩散模型, 时序一致性, 滚动推理

## 一句话总结

本文提出 RollingDepth，不使用视频扩散模型而是将单帧潜在扩散模型（Marigold）扩展为多帧 snippet 处理器，配合多尺度膨胀采样和鲁棒全局对齐算法，将短片段深度拼接为时序一致的长视频深度，在多个基准上超越了专门的视频深度模型和单帧模型。

## 研究背景与动机

**领域现状**：单帧单目深度估计近年取得巨大进展，得益于大规模预训练基础模型（DINOv2、StableDiffusion）和合成训练数据。Marigold、Depth Anything 等方法展现了优秀的零样本泛化能力。然而，逐帧应用这些方法到视频上会导致深度闪烁和漂移。

**现有痛点**：(1) 逐帧方法没有时序一致性概念，相邻帧深度可能突然跳变，特别是深度范围因相机运动突变时（如前景物体进入或相机转向窗外）；(2) 基于视频扩散模型的方法（ChronoDepth、DepthCrafter）虽然局部时序一致性好，但训练推理代价大、只能处理固定短序列长度，需要分段拼接方案，容易产生低频闪烁和漂移；(3) 视频深度模型在远处场景准确度较差。

**核心矛盾**：时序一致性要求帧间信息交换，但视频扩散模型的代价太高且受限于固定长度；单帧模型精确但缺乏时序连贯性。

**本文目标**：在不使用视频扩散模型的情况下，将单帧 LDM 扩展为能处理任意长度视频的时序一致深度估计器。

**切入角度**：既然视频扩散模型的核心问题是固定长度限制和高训练成本，那么能否用单帧模型的小幅扩展（处理 2-3 帧的短片段）+ 智能的全局对齐算法来达到甚至超越视频模型的效果？通过不同帧率采样片段来覆盖长短程时序关系。

**核心 idea**：将 Marigold 扩展为处理极短片段（通常 3 帧）的多帧 LDM，以不同膨胀率从视频中采样重叠片段，然后用鲁棒优化对齐所有片段的尺度和偏移，组装成一致的长视频深度。

## 方法详解

### 整体框架

输入一段任意长度的 RGB 视频，RollingDepth 执行三步处理：(1) **片段推理**：用膨胀滚动窗口以不同帧间距从视频中采样大量重叠的 3 帧 snippet，用多帧 LDM 对每个 snippet 做 1 步去噪得到初始深度片段；(2) **全局对齐**：对所有片段联合优化各自的尺度和偏移参数，使重叠帧上的深度值在全局一致；(3) **可选精修**：将对齐后的深度视频加入适度噪声后再次用 LDM 去噪，以恢复精细细节。

### 关键设计

1. **多帧 LDM (Snippet Depth Estimator)**:

    - 功能：从 Marigold 单帧模型扩展为能处理短片段的深度估计器
    - 核心思路：修改 self-attention 层，将 snippet 中所有帧的 token 展平为一个序列，使注意力机制跨帧运算，捕获空间和时序交互。这种设计不同于视频扩散模型的 factorized spatial-temporal attention，能处理任意时间间距的帧，适用于不同帧率的 snippet。同时将 Marigold 的仿射不变深度改为预测逆深度（inverse depth），对远场更鲁棒
    - 设计动机：跨帧 self-attention 是实现帧间信息交换的最简洁方案。预测逆深度比仿射不变深度更适合视频场景，因为视频中深度范围经常突变

2. **膨胀滚动核采样 (Dilated Rolling Kernel)**:

    - 功能：以多种时间分辨率从视频中构建大量重叠的短片段
    - 核心思路：对于 3 帧 snippet，使用不同的膨胀率 $g \in \{1, 10, 25\}$ 和步幅 $h$ 构建片段。膨胀率 1 的片段捕获相邻帧间的短程关系，膨胀率 25 的片段跨越约 1 秒的长程关系。不同膨胀率的片段在同一帧上重叠，为后续对齐提供约束
    - 设计动机：只用相邻帧无法覆盖长程时序依赖；只用大间距帧会丢失局部平滑性。多尺度采样兼顾两者，且保持恒定内存开销

3. **鲁棒全局深度对齐 (Depth Co-alignment)**:

    - 功能：将所有独立推理的 snippet 深度统一到一个全局一致的尺度和偏移
    - 核心思路：每个 snippet $k$ 有独立的 scale $s_k$ 和 shift $t_k$。通过最小化所有帧上所有重叠深度预测的 L1 损失来联合优化 $N_T$ 对参数。用 Adam 梯度下降求解（2000 步），高膨胀率片段给予更大权重以稳定优化。对齐后在每帧取所有重叠深度的逐像素均值得到最终深度
    - 设计动机：L1 损失比 L2 更鲁棒，不受离群值影响。高膨胀率片段提供长程约束，对稳定全局尺度至关重要

### 损失函数 / 训练策略

- 使用 TartanAir（合成视频 18 场景 369 序列）和 Hypersim（合成单帧 365 场景）微调 Marigold
- 训练图像 480×640，AdamW 优化器，学习率 $3 \times 10^{-5}$，4 张 A100 训练约 2 天
- 关键技巧：snippet 内联合归一化逆深度（而非逐帧归一化），使模型能处理深度范围突变
- 深度范围增强：随机压缩和偏移归一化深度范围

## 实验关键数据

### 主实验

**零样本视频深度估计对比 (AbsRel % ↓)**:

| 方法 | 类型 | PointOdyssey(250) | ScanNet(90) | Bonn(110) | DyDToF(200) | DyDToF(100) |
|------|------|---------|---------|------|---------|---------|
| Marigold | 单帧 | 14.9 | 14.9 | 10.5 | 25.3 | 16.4 |
| DepthAnythingv2 | 单帧 | 14.4 | 13.3 | 10.5 | 24.8 | 16.0 |
| ChronoDepth | 视频 | 51.7 | 16.8 | 10.9 | 26.9 | 19.9 |
| DepthCrafter | 视频 | 36.3 | 12.7 | **6.6** | 22.1 | 16.2 |
| **RollingDepth(fast)** | 扩展 | **9.6** | 10.1 | 7.9 | 17.7 | 12.7 |
| **RollingDepth** | 扩展 | **9.6** | **9.3** | 7.9 | **17.3** | **12.3** |

### 消融实验

| 膨胀率 | PointOdyssey ↓ | ScanNet ↓ | 说明 |
|--------|---------------|-----------|------|
| {1} | 16.7 | 12.8 | 仅相邻帧，缺长程信息 |
| {1, 25} | 10.2 | 10.6 | 加入长程约束，大幅提升 |
| {1, 10, 25} | **10.2** | **9.9** | 再加中程，进一步改善 |

| 对齐 | 精修 | PointOdyssey ↓ | ScanNet ↓ |
|------|------|---------------|-----------|
| × | × | 13.0 | 12.4 |
| ✓ | × | 10.2 | 9.9 |
| ✓ | ✓ | **9.6** | **9.3** |

### 关键发现

- 全局对齐是核心贡献：约 3 个百分点的误差降低来自对齐，精修仅带来边际改善
- 加入膨胀率 25 的片段带来巨大提升（PointOdyssey 16.7→10.2），长程约束至关重要
- 视频深度模型在 PointOdyssey 上表现极差（ChronoDepth 51.7），甚至不如单帧方法。说明视频先验在深度范围突变场景下反而是阻碍
- RollingDepth fast 版在 250 帧视频上仅需 81 秒，快于 ChronoDepth (121s) 和 DepthCrafter (284s)
- DepthCrafter 在室内前景人物场景（Bonn）表现最优，说明其视频先验对这类场景有利

## 亮点与洞察

- **反直觉的成功**：不用视频模型反而比视频模型做得更好，证明了"轻量扩展+智能后处理"路线的可行性
- **snippet 内联合归一化是关键技巧**：使模型能理解深度范围的帧间变化，这对长视频处理至关重要
- **全局对齐的优雅**：将复杂的时序一致性问题转化为简单的尺度-偏移优化，利用多尺度片段的重叠关系提供充分约束
- **1步推理**：snippet LDM 只做 1 步去噪即可获得初始深度估计，速度极快

## 局限与展望

- 对齐步骤依赖重叠片段间的深度一致性假设，当大幅物体运动时可能失效
- 在室内近景人物场景中略逊于 DepthCrafter，前景主导场景的建模仍有空间
- 推理时间随膨胀率数量和精修步骤增加而增长
- 可以考虑自适应选择膨胀率，根据视频运动特性动态调整
- 未探索与度量深度估计的结合，目前仍输出仿射不变深度

## 相关工作与启发

- **vs Marigold**: RollingDepth 的基础模型，本文展示了如何最小化改动将其扩展到视频
- **vs DepthCrafter**: 基于 SVD 视频扩散模型，局部一致性好但受限于固定长度，训练成本高。RollingDepth 更灵活且可处理 1000+ 帧长视频
- **vs ChronoDepth**: 同为扩散方法但效果较差，产生层状深度图，在深度范围变化大的场景表现极差

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "不用视频模型做视频深度"的思路反直觉但极其有效
- 实验充分度: ⭐⭐⭐⭐⭐ 6个对比方法、4个数据集、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 方法动机清晰，组件贡献层层递进
- 价值: ⭐⭐⭐⭐⭐ 重新定义了视频深度估计的方法论，证明了简单扩展路线的可行性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[ECCV 2024\] FutureDepth: Learning to Predict the Future Improves Video Depth Estimation](../../ECCV2024/3d_vision/futuredepth_learning_to_predict_the_future_improves_video_depth_estimation.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[CVPR 2025\] Generative Omnimatte: Learning to Decompose Video into Layers](generative_omnimatte_learning_to_decompose_video_into_layers.md)
- [\[CVPR 2025\] HD-EPIC: A Highly-Detailed Egocentric Video Dataset](hd-epic_a_highly-detailed_egocentric_video_dataset.md)

</div>

<!-- RELATED:END -->
