---
title: >-
  [论文解读] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities
description: >-
  [ICCV 2025][3D视觉][3D场景理解] 本文提出 LLaVA-3D，通过将 3D 位置嵌入注入 2D CLIP patch 特征构建"3D Patch"，以最小改动将 2D LMM（LLaVA-Video）扩展为统一的 2D/3D 理解模型，训练收敛速度比现有 3D LMM 快 3.5 倍，在多个 3D 基准上达到 SOTA 且保持 2D 能力不下降。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D场景理解
  - 大型多模态模型
  - 3D Patch
  - 位置编码
  - 视觉定位
---

# LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities

**会议**: ICCV 2025  
**arXiv**: [2409.18125](https://arxiv.org/abs/2409.18125)  
**代码**: https://zcmax.github.io/projects/LLaVA-3D  
**领域**: 3D视觉 / 多模态VLM  
**关键词**: 3D场景理解, 大型多模态模型, 3D Patch, 位置编码, 视觉定位

## 一句话总结

本文提出 LLaVA-3D，通过将 3D 位置嵌入注入 2D CLIP patch 特征构建"3D Patch"，以最小改动将 2D LMM（LLaVA-Video）扩展为统一的 2D/3D 理解模型，训练收敛速度比现有 3D LMM 快 3.5 倍，在多个 3D 基准上达到 SOTA 且保持 2D 能力不下降。

## 研究背景与动机

**领域现状**：2D 大型多模态模型（LMM）如 LLaVA 系列在图像和视频理解上取得了显著进展，但它们只停留在虚拟层面的视觉交互，缺乏与 3D 物理世界的交互能力。现有 3D LMM 的主流方案包括：(a) 使用 3D 点云编码器（如 LL3DA、LEO），(b) 利用离线 3D 实例分割提取目标特征（如 Chat-Scene），(c) 从多视角图像中通过手动设计的 2D 分割聚合 CLIP 特征构建 3D 表示（如 3D-LLM）。

**现有痛点**：(a) 缺乏大规模 3D 视觉-语言数据集，规模远小于 2D 数据集；(b) 没有像 CLIP 那样强大的预训练 3D 编码器；(c) 现有 3D LMM pipeline 复杂且计算密集，如需要先做 3D 实例分割再聚合特征，单个场景特征构建需 900 秒；(d) 3D 视觉定位依赖离线 3D 分割器，不适合实际应用。

**核心矛盾**：如何在不放弃 2D LMM 强大理解能力的前提下，高效地赋予其 3D 场景理解能力？直接训练 3D LMM 面临数据不足和编码器薄弱的双重困境。

**本文目标** (a) 如何高效地将 2D LMM 的 2D 视觉理解先验迁移到 3D 场景理解？(b) 如何构建统一的 2D+3D 架构，同时支持两种视觉理解任务？(c) 如何直接输出精确的 3D 空间感知结果（如 3D bbox），而不依赖耗时的离线 3D 分割器？

**切入角度**：受 ODIN 启发——ODIN 通过为 2D 和 3D 使用不同的位置编码实现统一分割。类似地，LLaVA-3D 通过在 2D CLIP patch 上添加 3D 位置嵌入来构建"3D Patch"，用最小改动桥接 2D 和 3D 世界。

**核心 idea**：直接在 LLaVA 的 2D CLIP patch 特征上叠加可学习的 3D 位置嵌入，构建同时编码语义和空间信息的 3D Patch，从而以最小代价将 2D LMM 升级为 3D LMM。

## 方法详解

### 整体框架

基于 LLaVA-Video 构建。输入为多视角 RGB-D 图像，CLIP 编码器提取 2D patch 特征后，通过投影层对齐到 LLM 空间，再叠加从 3D 坐标编码的 3D 位置嵌入形成"3D Patch"。3D Patch 经过可选的池化压缩后送入 LLM 进行多模态推理。对于需要 3D 坐标输入/输出的任务，分别使用 3D Coordinate Token 和 Grounding Decoder 处理。

### 关键设计

1. **3D Patch 构建**:

    - 功能：将 2D 视觉 patch 特征增强为包含 3D 空间信息的 3D patch 表示
    - 核心思路：给定多视角 2D patch 特征 $X'_p \in \mathbb{R}^{V \times d \times w \times h}$（V 为视角数），利用已知的相机内外参和深度图获取每个 patch 的 3D 位置 $P \in \mathbb{R}^{V \times 3 \times w \times h}$，通过一个两层 MLP 编码为 3D 位置嵌入 $P'$，然后直接相加：$X'_{3D} = X'_p + P'$
    - 设计动机：通过加法融合而非复杂的特征聚合，最大限度保留 2D CLIP patch 的丰富语义信息，同时注入 3D 空间位置信息。这种设计使得同一模型可以通过是否添加 3D 位置嵌入来切换 2D/3D 模式

2. **3D Patch 池化**:

    - 功能：压缩 3D Patch 的数量以适应 LLM 的上下文长度限制
    - 核心思路：提出两种无参数池化策略——(a) 体素化池化（Voxelization Pooling）：将 3D 空间离散化为体素网格，对每个占用体素内的 3D patch 做平均池化，token 数量取决于体素大小而非图像数量；(b) FPS 池化：通过最远点采样选取固定数量的代表性 3D patch
    - 设计动机：与 2D 中常用的空间/时间维度池化不同，基于 3D 位置的池化确保压缩后的特征尽可能完整覆盖整个场景结构

3. **3D 坐标 Token 编码 & Grounding Decoder**:

    - 功能：处理 3D 坐标的输入（如"描述位置 [x,y,z] 处的物体"）和输出（3D bbox 预测）
    - 核心思路：输入端通过 3D Position Encoding Layer 将 3D 坐标编码为 3D Coordinate Token，与 3D patch token 和文本 token 一起送入 LLM。输出端设计了 Grounding Decoder：用 FPS 从 3D patch 中采样 instance query，通过 cross-attention 聚合 3D patch 的几何信息和 LLM 的语义信息，使用多尺度 3D k-NN attention 建模局部几何结构，最终预测 3D bbox
    - 设计动机：直接让 LLM 输出 3D 坐标的效果很差（ScanRefer 上仅 7.8 Acc@0.25），因此需要专门的解码器从 3D patch 中提取几何信息。该方案避免了依赖离线 3D 分割器的两阶段方法

### 损失函数 / 训练策略

两阶段训练：
- **Stage 1 多任务指令微调**：在 2D（LLaVA-Video 数据）和 3D（LLaVA-3D-Instruct-86K）数据上联合微调，训练 3D 位置编码层和其他模块。3D 数据覆盖 3D QA、3D dense captioning、3D visual grounding 等任务
- **Stage 2 Decoder 单独微调**：由于 Grounding Decoder 在 Stage 1 的单 epoch 训练中未充分收敛，冻结其他组件，仅用 3D visual grounding 数据进一步训练 decoder 和 location token

## 实验关键数据

### 主实验

| 基准 | 指标 | LLaVA-3D | 之前 SOTA | 提升 |
|------|------|----------|----------|------|
| ScanQA (val) | CIDEr | **103.1** | 101.4 (LEO) | +1.7 |
| ScanQA (val) | EM@1 | **30.6** | 27.2 (Scene-LLM) | +3.4 |
| SQA3D (test) | EM@1 | **60.1** | 54.7 (Chat-3D v2) | +5.4 |
| MMScan QA | EM@1 | **50.1 (54.9)** | 36.6 (44.5) (LEO) | +13.5 |
| OpenEQA | Accuracy | 53.2 | 55.3 (GPT-4V, 50帧) | 接近 |
| Scan2Cap | C@0.5 | **84.1** | 77.2 (ChatScene) | +6.9 |
| Multi3DRefer | Acc@0.25 | **49.8** (单阶段) | 57.1 (ChatScene, 两阶段) | - |
| MVBench (2D) | - | 58.1 | 58.6 (LLaVA-Video) | -0.5 |

### 消融实验

| 配置 | ScanQA EM@1 | SQA3D EM@1 | ScanRefer Acc@0.25 | 推理时间 |
|------|-------------|------------|-------------------|---------|
| SAM+CLIP w/ PE + Q-Former + Vicuna-7B | 21.9 | 49.3 | - | 900s |
| CLIP w/ PE + Pooling+MLP + Vicuna-7B | 23.4 | 51.2 | 43.8 | 0.2s |
| CLIP w/ PE + Pooling+MLP + LLaVA-1.5-7B | 27.0 | 55.6 | 47.9 | 0.2s |
| CLIP w/ PE + MLP + LLaVA-Video-7B | **30.6** | **60.1** | **50.1** | 0.2s |
| 2D Patch (无3D位置嵌入) | - | 59.8 | - | - |
| 3D Patch | - | 60.1 (+0.3) | - | - |
| 3D Patch (MMScan QA) | **55.4** vs 42.1 | - | - | - |
| 3D Patch (Scan2Cap) | **84.1** vs 29.7 | - | - | - |

### 关键发现

- **3D Patch 的效果取决于任务类型**：在 ScanQA/SQA3D 这类主要依赖语言描述的简单 3D QA 上，3D 位置嵌入仅带来微小提升（+0.3~0.4）。但在需要真正 3D 空间理解的任务上（如 MMScan QA +13.3, Scan2Cap +54.4），3D Patch 至关重要
- **从 2D LMM 而非 LLM 构建 3D LMM 优势明显**：从 LLaVA-1.5 出发比从 Vicuna-7B 出发，ScanRefer 提升 4.1%，验证了 2D 预训练先验的价值
- **视频 LMM 是更好的基座模型**：LLaVA-Video 作为基座比 LLaVA-1.5 和 InternVL2.5 效果更好，因为多视角 3D 场景表示与视频具有内在一致性
- **推理速度大幅提升**：相比需要 SAM+CLIP 的方案（900s），仅用 CLIP 的方案推理仅需 0.2s，且性能更好
- **2D 能力几乎无损**：LLaVA-3D 在 MVBench 和 VideoMME 上仅比 LLaVA-Video 低 0.5，验证了联合训练策略的有效性

## 亮点与洞察

- **极简但高效的 3D 扩展方案**：仅通过"2D patch + 3D 位置嵌入 = 3D Patch"这一简单加法操作，就将 2D LMM 升级为 3D LMM，避免了复杂的 3D 编码器和特征聚合 pipeline。这种"最小改动、最大复用"的设计哲学值得借鉴
- **Grounding Decoder 的必要性**：实验证明 LLM 直接输出 3D 坐标的效果极差（7.8 vs 50.1），说明 3D 空间感知需要专门的解码器从视觉特征中提取几何信息，不能简单依赖语言模型的数字生成能力
- **通用扩展能力**：LLaVA-3D 可以作为通用方法适配不同 2D LMM（LLaVA-1.5、InternVL2.5、LLaVA-Video），且能从更强的 2D 基座中获益

## 局限与展望

- 需要已知的相机参数和深度信息来构建 3D 位置嵌入，限制了对无深度信息场景的适用性
- 当前仅在 ScanNet 等室内场景数据集上验证，对大规模室外场景的泛化能力未知
- Grounding Decoder 需要额外的 Stage 2 训练才能收敛，增加了训练复杂度
- 3D Patch 池化策略（如体素化）引入了信息损失，对需要精细空间理解的任务可能有影响

## 相关工作与启发

- **vs LEO/Chat-Scene**: 这些方法依赖离线 3D 实例分割来提取目标特征，pipeline 复杂且耗时。LLaVA-3D 直接从多视角图像构建 3D 表示，避免了 3D 分割的瓶颈
- **vs 3D-LLM/Scene-LLM**: 需要复杂的 2D 分割+特征聚合 pipeline，场景构建耗时 900s。LLaVA-3D 仅需 0.2s
- **vs 纯 2D LMM（GPT-4V等）**: GPT-4V 等在 3D QA 上也有不错表现（ScanQA 上甚至超过一些 3D 专用方法），但在需要精确 3D 空间理解的任务上仍与 LLaVA-3D 有明显差距

## 评分

- 新颖性: ⭐⭐⭐⭐ 思路简单但效果显著，3D Patch 的概念很优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 6+ 个 3D 基准，消融详细，还做了 2D 保持性验证
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图表精心设计，动机论证充分
- 价值: ⭐⭐⭐⭐⭐ 为 2D→3D LMM 的扩展提供了一条简洁高效的路径，影响力大

<!-- RELATED:START -->

## 相关论文

- [Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)
- [3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer](../../CVPR2025/3d_vision/3d-llava_towards_generalist_3d_lmms_with_omni_superpoint_transformer.md)
- [Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness](ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)
- [GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

<!-- RELATED:END -->
