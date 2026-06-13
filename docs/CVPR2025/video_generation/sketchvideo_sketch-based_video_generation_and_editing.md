---
title: >-
  [论文解读] SketchVideo: Sketch-Based Video Generation and Editing
description: >-
  [CVPR 2025][视频生成][草图控制] 基于 DiT 视频生成架构，提出内存高效的草图条件网络和帧间注意力机制，实现通过 1-2 张关键帧草图对视频进行精细的空间布局和几何细节控制，同时支持基于草图的视频局部编辑。 领域现状：扩散模型驱动的文本到视频生成取得了显著进展，CogVideoX 等基于 DiT 的视频生成模…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "草图控制"
  - "视频编辑"
  - "DiT架构"
  - "帧间注意力"
---

# SketchVideo: Sketch-Based Video Generation and Editing

**会议**: CVPR 2025  
**arXiv**: [2503.23284](https://arxiv.org/abs/2503.23284)  
**代码**: [http://geometrylearning.com/SketchVideo/](http://geometrylearning.com/SketchVideo/)  
**领域**: 视频生成  
**关键词**: 草图控制, 视频生成, 视频编辑, DiT架构, 帧间注意力

## 一句话总结

基于 DiT 视频生成架构，提出内存高效的草图条件网络和帧间注意力机制，实现通过 1-2 张关键帧草图对视频进行精细的空间布局和几何细节控制，同时支持基于草图的视频局部编辑。

## 研究背景与动机

**领域现状**：扩散模型驱动的文本到视频生成取得了显著进展，CogVideoX 等基于 DiT 的视频生成模型能生成时间上连贯的高质量视频。然而，文本提示只能描述高层语义，无法精确控制场景布局和几何细节。

**现有痛点**：现有视频生成方法要么依赖图像作为额外条件（但如何生成输入图像本身就是问题），要么像 SparseCtrl 那样用白色占位符填充缺失帧的条件（效果不佳），要么要求提供所有帧的条件输入（对草图交互而言太繁琐）。此外，在 DiT 架构上直接复制半数预训练模块作为条件网络（如 PIXART-δ 的做法）会导致显存溢出。

**核心矛盾**：用户只需在 1-2 个关键帧上画草图即可传达空间结构和运动信息，但如何从这种时间上极度稀疏的草图条件传播到所有视频帧是一个关键挑战。同时必须在控制精度和内存效率之间取得平衡。

**本文目标**：(1) 设计内存高效的草图条件网络适配 DiT 视频架构；(2) 从稀疏关键帧草图传播控制信号到所有帧；(3) 支持基于草图的精细视频局部编辑。

**切入角度**：作者观察到 DiT 网络不同深度的 block 处理不同层级的特征，与其像 PIXART-δ 那样借用连续的前半部分 block，不如均匀跳跃选取少量 block 来覆盖多个特征层级。同时利用视频帧间内在的相似性来传播稀疏草图条件。

**核心 idea**：用 5 个均匀分布的草图控制 block（跳跃连接到 30 个 DiT block 中的第 0/6/12/18/24 个）替代复制半数模型的做法，并通过一种新颖的帧间注意力机制（Q/K 来自噪声隐状态、V 来自草图特征）将关键帧草图条件传播到所有帧。

## 方法详解

### 整体框架

输入为文本提示和 1-2 张关键帧草图（可指定任意时间点），输出为与草图几何一致的视频。整体流程分为两个分支：(1) 基础视频生成网络 CogVideoX-2b（30 个 DiT block），(2) 草图条件网络（5 个草图控制 block）。草图先经 VAE 编码为隐空间表示，再通过 patchify 和时间感知位置编码得到草图 latent。5 个草图控制 block 分别预测残差特征并注入对应的 DiT block。编辑模式额外引入视频插入模块和 latent 融合策略。

### 关键设计

1. **跳跃残差结构 (Skip Residual Structure)**:

    - 功能：以极少参数实现多层级的草图控制信号注入
    - 核心思路：不同于 PIXART-δ 使用前半部分连续 block 作为编码器，本文认为 DiT 不同深度的 block 处理不同层级特征，因此只用 5 个草图控制 block，均匀分布在第 0/6/12/18/24 个 DiT block 的位置预测残差特征。这样只需 5/30 = 1/6 的额外参数，显著减少显存占用
    - 设计动机：传统 ControlNet 做法（复制半数预训练模型）无法应用于视频生成的 DiT 架构，因为参数量过大导致 OOM。同时跳跃结构能覆盖浅层到深层的多个特征层级

2. **帧间注意力机制 (Inter-frame Attention)**:

    - 功能：将 1-2 张关键帧的草图条件传播到所有视频帧
    - 核心思路：与传统 cross-attention（K 和 V 都来自条件信号）不同，本文的 Q 来自所有帧的噪声隐特征 $h_i^{1:N}$，K 来自关键帧对应的噪声隐特征 $h_i^{t_1,t_2}$，V 来自经 DiT block 处理后的关键帧草图特征 $c_i^{t_1,t_2}$。这样注意力权重基于帧间相似性计算，而传播的是草图特征信息。草图控制 block 中，先用 FeedForward 更新草图特征，再用可训练的 DiT block 副本处理草图（不处理白色占位符），最后通过帧间注意力传播到所有帧
    - 设计动机：直接用白色占位符处理缺失帧（如 SparseCtrl）会导致网络同时处理性质完全不同的输入，学习效果差。利用帧间隐特征相似性来决定传播权重是更自然的方式

3. **视频插入模块 (Video Insertion Module)**:

    - 功能：在编辑场景中保持新编辑内容与原始视频的时空一致性
    - 核心思路：对于编辑任务，额外引入一个可训练的 DiT block 副本处理被 mask 遮挡编辑区域后的原始视频 latent。然后将草图分支输出 $\tilde{c}_i^{1:N}$ 与视频分支输出 $\tilde{v}_i^{1:N}$ 按 mask 拼接：$\text{Concat}(\tilde{c}_i^{1:N} * M^{1:N}, \tilde{v}_i^{1:N} * \bar{M}^{1:N})$，送入 FeedForward 生成最终残差特征
    - 设计动机：纯草图生成网络缺乏对原始视频信息的感知，无法保证编辑区域与未编辑区域的一致性

### 损失函数 / 训练策略

- **两阶段训练**：第一阶段混合图像数据和视频数据训练，加速收敛并弥补视频数据不足；第二阶段仅用视频数据提升时间一致性
- **编辑网络微调**：从生成网络预训练权重初始化，新增视频插入模块。预训练模型已具备草图保真度，只需学习视频信息。使用随机 mask 的自监督修复方式训练
- **推理时 Latent 融合**：对编辑任务，在第 25 和 49 步（共 50 步）用 DDIM 反演的原始视频 latent 替换未编辑区域，精确保留原始视频细节
- 训练损失沿用 CogVideoX 的扩散目标（v-prediction + zero SNR）

## 实验关键数据

### 主实验

| 方法 | LPIPS ↓ | CLIP ↑ | Fidelity ↓ | Consistency ↓ | Realism ↓ |
|------|---------|--------|-----------|--------------|----------|
| AMT | 29.17 | 96.12 | 3.13 | 3.51 | 3.57 |
| SparseCtrl | 44.85 | 96.48 | 2.79 | 2.94 | 2.83 |
| Ctrl-CogVideo | 32.23 | 98.04 | 2.86 | 2.47 | 2.50 |
| **SketchVideo** | **27.56** | **98.31** | **1.21** | **1.08** | **1.11** |

编辑对比（LPIPS/CLIP ×100，PSNR 评估未编辑区域保持）：

| 方法 | LPIPS ↓ | CLIP ↑ | PSNR ↑ | Fidelity ↓ | Preservation ↓ | Realism ↓ |
|------|---------|--------|--------|-----------|----------------|----------|
| InsV2V | 13.61 | 95.39 | 16.84 | 2.58 | 2.26 | 2.61 |
| AnyV2V | 11.92 | 93.47 | 13.68 | 2.35 | 2.69 | 2.34 |
| **SketchVideo** | **9.74** | **98.34** | **36.48** | **1.07** | **1.05** | **1.04** |

### 消融实验

| 变体 | LPIPS ↓ | CLIP ↑ |
|------|---------|--------|
| 去掉帧间注意力 | 36.33 | 98.10 |
| 用传统 Sketch K,V 的 cross-attn | 32.59 | 98.19 |
| 去掉跳跃结构（前5个连续block）| 31.91 | 97.60 |
| 去掉图像数据训练 | 34.58 | 98.24 |
| **完整模型** | **30.79** | **98.48** |

### 关键发现

- 帧间注意力对草图保真度影响最大（去掉后 LPIPS 从 30.79 上升到 36.33）
- 跳跃结构对视频时间一致性至关重要（去掉后 CLIP 降至 97.60）
- 混合图像训练数据显著改善几何匹配精度
- 编辑任务中，生成预训练是保持草图保真度的关键，latent 融合是保持未编辑区域的关键（去掉后 PSNR 从 36.48 降至 31.69）
- 用户研究中本方法在所有评价维度上均排名第一

## 亮点与洞察

- **极简高效的设计思路**：仅 5 个控制 block（原模型的 1/6）即可实现有效控制，打破了"需要复制半数模型"的惯性思维
- **巧妙的帧间注意力设计**：Q/K 来自原始噪声特征保证帧间关系建模，V 来自草图特征实现条件注入，二者优雅解耦
- **统一框架**：生成和编辑共用同一套草图控制 block，编辑只需额外添加视频插入模块
- 草图可在任意时间点指定，不限于首尾帧，支持运动内插和外推

## 局限与展望

- 生成能力受限于预训练的文本到视频模型（CogVideoX-2b）的质量上限
- 复杂场景（如人手、多物体交互）效果仍不理想
- 目前只关注几何控制，未来可探索颜色笔触等外观定制
- 仅支持约 6 秒的短视频片段，长视频生成有待探索
- 引入 3D 先验（如 SMPL-X）可能改善人体场景效果

## 相关工作与启发

- **ControlNet 系列**：本文是 ControlNet 思想在 DiT 视频架构上的创新适配，跳跃结构和帧间传播是核心创新
- **SparseCtrl**：最直接的对比基线，白色占位符方案在 DiT 上效果不佳
- **DiT 架构的条件控制**：PIXART-δ 的前半部分 block 方案不适用于视频，而均匀跳跃是更好选择
- 启发：对 DiT 类架构进行条件控制时，不必拘泥于"前半编码器"的范式，可根据特征层级灵活设计注入点

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 技术深度 | 8 |
| 实验充分度 | 8 |
| 写作质量 | 8 |
| 实用价值 | 8 |
| 总评 | 8.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)
- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[CVPR 2025\] VEU-Bench: Towards Comprehensive Understanding of Video Editing](veu-bench_towards_comprehensive_understanding_of_video_editing.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)

</div>

<!-- RELATED:END -->
