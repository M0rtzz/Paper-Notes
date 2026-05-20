---
title: >-
  [论文解读] Arbitrary Generative Video Interpolation
description: >-
  [ICLR 2026][视频生成][Video Frame Interpolation] ArbInterp 提出了一种支持任意时间戳、任意长度的生成式视频帧插值框架，通过时间戳感知旋转位置编码（TaRoPE）实现精准时间控制，并通过外观-运动解耦的条件注入策略实现长序列的无缝拼接。
tags:
  - "ICLR 2026"
  - "视频生成"
  - "Video Frame Interpolation"
  - "Generative VFI"
  - "RoPE"
  - "Temporal Conditioning"
  - "Any-length Generation"
---

# Arbitrary Generative Video Interpolation

**会议**: ICLR 2026  
**arXiv**: [2510.00578](https://arxiv.org/abs/2510.00578)  
**代码**: [项目主页](https://mcg-nju.github.io/ArbInterp-Web/)  
**领域**: 视频理解 / 视频生成  
**关键词**: Video Frame Interpolation, Generative VFI, RoPE, Temporal Conditioning, Any-length Generation

## 一句话总结

ArbInterp 提出了一种支持任意时间戳、任意长度的生成式视频帧插值框架，通过时间戳感知旋转位置编码（TaRoPE）实现精准时间控制，并通过外观-运动解耦的条件注入策略实现长序列的无缝拼接。

## 研究背景与动机

视频帧插值（Video Frame Interpolation, VFI）是视频生成领域的基础任务，给定起始帧和结束帧，生成中间过渡帧。近年来，基于扩散模型的生成式 VFI 方法（如 DynamiCrafter、TRF、GI 等）展示了生成高质量中间帧的能力。

然而现有生成式 VFI 方法存在两个关键限制：

**固定插值数量**: 现有方法只能一次性生成固定数量的中间帧（如一次生成 7 帧或 15 帧），无法灵活调整生成帧率或总序列时长。例如，用户可能需要在两帧之间插 2 帧（2x），也可能需要插 31 帧（32x），现有方法难以统一处理。

**长序列不连贯**: 当需要大量插值帧时（如 32x 插值），直接生成长序列面临显存和质量问题。分段生成是自然的解决方案，但不同片段之间的时空连贯性难以保证，容易出现运动不自然、外观不一致等问题。

ArbInterp 的目标是构建一个统一的生成式 VFI 框架，同时解决"任意时间戳"和"任意长度"两个挑战。

## 方法详解

### 整体框架

ArbInterp 基于视频扩散模型构建。整体流程分为两个层面：
- **单段插值**：给定起止帧和目标时间戳序列，使用 TaRoPE 精确控制每帧对应的时间位置
- **长序列插值**：将长序列分解为多个片段，通过外观-运动解耦的条件策略确保片段间的时空过渡一致性

### 关键设计

1. **Timestamp-aware Rotary Position Embedding（TaRoPE）**: 

    - **功能**: 让模型能感知并生成任意连续时间戳位置的帧
    - **核心思路**: 传统位置编码将帧位置映射为固定整数位置（0, 1, 2, ...），TaRoPE 将目标归一化时间戳（如 0.25, 0.5, 0.75）调制到 RoPE 的旋转角度中。具体来说，对时间维度的 RoPE，用目标时间戳替换离散位置索引，使得旋转角度与连续时间对齐。
    - **设计动机**: 固定位置范式（如 DynamiCrafter 的整数位置编码）只能生成等间距帧，无法处理非均匀时间戳（如只想要 t=0.3 的帧）。TaRoPE 使位置编码从离散跳跃变为连续可控，一个模型即可覆盖 2x 到 32x 的任意插值倍率。

2. **分段式帧合成（Segment-wise Frame Synthesis）**:

    - **功能**: 将长序列（如 32x 插值 = 生成 31 帧）分解为多个短段依次生成
    - **核心思路**: 每段生成若干帧，前一段的最后一帧作为后一段的起始边界条件。核心挑战在于如何使不同段之间平滑过渡。
    - **设计动机**: 直接生成超长序列受显存限制且质量下降。分段生成在计算上可行，但需要额外的机制来确保段间一致性。

3. **外观-运动解耦条件策略（Appearance-Motion Decoupled Conditioning）**:

    - **功能**: 分别处理段间的外观一致性和运动连贯性
    - **核心思路**: 
        - **外观条件**: 利用前一段的端点帧（起始帧和结束帧）作为视觉条件，强制新段的外观风格与已生成内容一致
        - **运动条件**: 利用时间语义信息（即归一化时间戳）维持全局运动的连贯性，确保速度和方向在段间自然过渡
    - **设计动机**: 单纯使用前一段最后帧作条件可能导致外观漂移或运动跳变。将外观和运动信号解耦后分别注入，能更精确地控制两个维度的连续性。

### 损失函数 / 训练策略

论文未在摘要和项目页面详述损失函数细节，但作为基于扩散模型的方法，核心训练策略包括：
- 标准的去噪扩散损失
- TaRoPE 在训练时随机采样不同时间戳，使模型学会处理任意时间位置
- 训练时使用不同长度的片段和不同的插值倍率，增强泛化性

## 实验关键数据

### 评估基准

作者构建了两个综合性基准：

1. **MultiInterp Benchmark**: 多尺度帧插值评估（2x, 4x, 8x, 16x, 32x），测试模型在不同插值倍率下的泛化能力
2. **StreamInterp Benchmark**: 流式/长序列插值评估，测试分段生成时的时空连贯性

### 主实验

| 方法 | 2x 质量 | 8x 质量 | 16x 质量 | 32x 质量 | 评估维度 |
|------|---------|---------|----------|----------|----------|
| DynamiCrafter | 基线 | 基线 | 基线 | 基线 | 固定帧数限制 |
| TRF | 对比 | 对比 | 对比 | 对比 | 固定位置编码 |
| GI | 对比 | 对比 | 对比 | 对比 | 生成式插值 |
| **ArbInterp** | **最优** | **最优** | **最优** | **最优** | 统一模型全覆盖 |

根据论文摘要，ArbInterp 在所有插值场景下均超越先前方法，展现更高保真度和更无缝的时空连续性。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o TaRoPE（固定位置编码） | 质量下降 | 无法适应不同倍率 |
| w/o 外观-运动解耦条件 | 段间不连贯 | 运动跳变、外观漂移 |
| 不同段长度 | 影响效率和质量权衡 | 较短段更灵活但可能累积误差 |

### 关键发现

1. **TaRoPE 的连续可控性**: 单一模型可以处理 2x 到 32x 的任意插值，无需针对每个倍率单独训练
2. **解耦条件的必要性**: 如果只用前一段末帧做条件（不区分外观和运动），长序列会出现渐进的质量退化
3. **从定量到定性的全面优势**: 在多个对比方法中，ArbInterp 不仅指标更好，视觉效果也更自然流畅

## 亮点与洞察

- **TaRoPE 方案优雅**: 将连续时间戳编码到 RoPE 中是一个简洁但有效的设计，几乎零额外参数即可实现任意时间戳控制。这个思路可以推广到其他需要连续化离散位置的生成任务。
- **解耦设计思想**: 外观一致性和运动连贯性是两个正交的需求，将它们解耦处理比混合处理更加可控。这种思路在视频编辑、视频续写等任务中也有借鉴意义。
- **实用性强**: 任意倍率 + 任意长度 = 一个模型适配所有帧插值需求，大幅降低部署复杂度。
- **基准构建**: MultiInterp 和 StreamInterp 两个 benchmark 的构建也是一个贡献，有助于后续工作的公平比较。

## 局限与展望

1. **依赖扩散模型的生成速度**: 生成式方法逐帧去噪的推理速度远慢于传统光流方法（如 RIFE、IFRNet），高倍率插值时的推理时间可能成为瓶颈
2. **分段累积误差**: 虽然有解耦条件策略，但超长序列（如 64x, 128x）下是否会出现渐进退化尚不确定
3. **场景多样性**: 从项目页面看，演示主要集中在驾驶和运动场景，复杂遮挡、场景切换等极端情况的表现未知
4. **与非生成式方法的对比**: 论文主要对比生成式 VFI 方法，与传统高效的光流 VFI 方法（RIFE 等）的全面定量对比会更有说服力
5. **训练数据需求**: 生成式模型通常需要大规模视频数据预训练，训练成本和数据来源值得关注

## 相关工作与启发

- **与 DynamiCrafter 的关系**: DynamiCrafter 是生成式 VFI 的代表方法之一，但受限于固定帧位置编码。ArbInterp 的 TaRoPE 直接解决了这个根本限制。
- **与 TRF、GI 的关系**: TRF（Time-Reversal Fusion）和 GI（Generative Interpolation）也尝试了生成式插值，但同样受固定长度约束。
- **RoPE 的时间维度扩展**: 原始 RoPE 在 LLM 中用于序列位置编码，ArbInterp 将其扩展到视频的时间维度并支持连续值，这种跨领域的技术迁移值得注意。
- **对视频生成的启发**: TaRoPE 和解耦条件策略不仅适用于帧插值，也可能对视频预测、视频续写等任务有帮助。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TeachMaster: Generative Teaching via Code](../../ACL2026/video_generation/teachmaster_generative_teaching_via_code.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](../../CVPR2026/video_generation/generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](../../ICCV2025/video_generation/motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[CVPR 2026\] LightMover: Generative Light Movement with Color and Intensity Controls](../../CVPR2026/video_generation/lightmover_generative_light_movement_with_color_and_intensity_controls.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](../../ICCV2025/video_generation/recammaster_camera-controlled_generative_rendering_from_a_single_video.md)

</div>

<!-- RELATED:END -->
