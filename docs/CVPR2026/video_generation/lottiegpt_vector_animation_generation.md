---
title: >-
  [论文解读] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation
description: >-
  [CVPR 2026][视频生成][矢量动画] 提出首个矢量动画自回归生成框架 LottieGPT，设计了 Lottie 分词器将层级几何体、变换和关键帧运动编码为紧凑 token 序列，构建 660K 动画数据集，基于 Qwen-VL 微调实现从文本/图像直接生成可编辑矢量动画。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "矢量动画"
  - "Lottie"
  - "自回归生成"
  - "分词器"
  - "多模态"
---

# LottieGPT: Tokenizing Vector Animation for Autoregressive Generation

**会议**: CVPR 2026  
**arXiv**: [2604.11792](https://arxiv.org/abs/2604.11792)  
**代码**: [https://lottiegpt.github.io/](https://lottiegpt.github.io/)  
**领域**: 视频生成  
**关键词**: 矢量动画, Lottie, 自回归生成, 分词器, 多模态

## 一句话总结

提出首个矢量动画自回归生成框架 LottieGPT，设计了 Lottie 分词器将层级几何体、变换和关键帧运动编码为紧凑 token 序列，构建 660K 动画数据集，基于 Qwen-VL 微调实现从文本/图像直接生成可编辑矢量动画。

## 研究背景与动机

**领域现状**：视频生成领域（Sora、Kling 等）已能生成高质量光栅视频，但所有现有生成模型均在像素空间操作，无法生成矢量动画——一种分辨率无关、可编辑、紧凑的多媒体主流形式。

**现有痛点**：矢量动画（如 UI 动效、品牌动画、After Effects 动态图形）具有像素视频无法提供的关键属性：无限分辨率、语义可操作性、参数化运动和小文件体积。现有 SVG 生成方法仅限于静态输出，缺乏时间建模能力。

**核心矛盾**：矢量动画既包含层级结构又包含时间依赖的变换逻辑，如何将其编码为适合自回归建模的 token 序列是核心挑战。此外，大规模矢量动画数据集的缺失也是主要瓶颈。

**本文目标**：(1) 设计能统一编码层级几何和时间运动的分词器；(2) 构建大规模矢量动画数据集；(3) 训练首个矢量动画生成的多模态模型。

**切入角度**：采用 Lottie 格式（广泛部署的 JSON 动画标准），利用其关键帧+缓动函数的参数化表示实现紧凑编码。

**核心 idea**：用关键帧和插值函数替代逐帧数据来 token 化矢量动画，大幅减少序列长度同时保留结构保真度。

## 方法详解

### 整体框架

LottieGPT 基于 Qwen2.5-VL 架构，包含三个组件：(1) Lottie 分词器将 JSON 动画编码为紧凑 token 序列；(2) 视觉-语言骨干处理多模态输入；(3) 两阶段训练策略（先静态后动态）。输入为文本/图像/关键帧视频，输出为 Lottie token 序列，可解码为完整可编辑的矢量动画。

### 关键设计

1. **Lottie 分词器 - 层级结构编码**:

    - 功能：将 Lottie JSON 的层级结构（动画元数据→资产→层→形状→属性）编码为离散 token 序列
    - 核心思路：使用特殊 token（如 `<|LAYER|>`, `<|ty|>`）标记层级边界和关系，直接对应 Lottie schema。支持完整的形状原语编码（椭圆、填充、渐变、描边等），不像 OmniSVG 需要分解为原子命令
    - 设计动机：保留语义信息和层级组织，使模型能学习结构模式而非任意文本序列

2. **关键帧运动压缩**:

    - 功能：实现时间维度的紧凑编码，区别于逐帧编码
    - 核心思路：只存储关键帧时间点 `<|t|>`、属性值和贝塞尔缓动函数 `<|ease|>`，而非密集的逐帧数据。对于 100 帧动画仅需 6 个关键帧的 token，300 帧时压缩率达 98%。缓动函数作为一等原语编码，使相同关键帧可产生截然不同的运动感觉
    - 设计动机：矢量动画的本质是关键帧+插值，这种编码在大幅减少 token 数的同时保留了动画的完整信息

3. **静态到动态的渐进训练**:

    - 功能：通过课程学习策略稳定训练
    - 核心思路：Stage 1 训练静态矢量图形（50%文本到Lottie + 50%图像到Lottie），Stage 2 引入时间动态（34%纯文本 + 33%文本+首帧 + 33%文本+视频关键帧）
    - 设计动机：直接混合训练静态+动态数据会导致收敛不稳定，因为动画样本的 token 数远多于静态图形

### 损失函数 / 训练策略

标准因果语言模型交叉熵损失：$\mathcal{L} = -\sum_{i=1}^{N} \log P(t_i | t_{<i}, \mathbf{c})$，其中 $\mathbf{c}$ 为多模态条件。分词器支持无损往返：解码后的动画与原始渲染完全一致。

## 实验关键数据

### 主实验

| 方法 | 输入 | CLIP↑ | SSIM↑ | LPIPS↓ | DINOv2↑ | JSON↑ | 有效率 |
|------|------|-------|-------|--------|---------|-------|-------|
| OmniSVG-7B | 文本 | 0.832 | 0.563 | 0.512 | 0.727 | N/A | N/A |
| LottieGPT-7B | 文本 | **0.933** | **0.810** | **0.176** | **0.857** | 0.824 | 98.3% |
| StarVector-8B | 图像 | 0.766 | 0.385 | 0.465 | 0.529 | N/A | N/A |
| OmniSVG-7B | 图像 | 0.900 | 0.705 | 0.251 | 0.848 | N/A | N/A |
| LottieGPT-7B | 图像 | **0.945** | **0.835** | **0.154** | **0.876** | 0.843 | 98.8% |

### 消融实验

| 配置 | CLIP↑ | SSIM↑ | 有效率 |
|------|-------|-------|-------|
| 完整模型 (Stage1+2) | 0.933 | 0.810 | 98.3% |
| 仅 Stage 1 (无动画) | 0.928 | 0.805 | 97.5% |
| 无层级编码 | 0.891 | 0.752 | 92.1% |
| 逐帧编码替代关键帧 | 0.875 | 0.701 | 85.6% |

### 关键发现

- 关键帧编码对有效率的提升至关重要：逐帧编码导致序列过长，有效率从 98.3% 降至 85.6%
- 时间建模增强了静态矢量理解：LottieGPT 在 SVG 生成上也达到新 SOTA
- JSON 结构分数证明生成的 Lottie 文件具有高结构保真度

## 亮点与洞察

- 关键帧+缓动函数编码是一个优雅的设计：它既保留了动画的完整语义（运动曲线是一等原语），又实现了极高的压缩率，这个思路可推广到其他参数化表示的生成任务
- 数据集贡献巨大：660K 矢量动画 + 15M 静态矢量图形，是该领域首个大规模资源
- 将 2D 动画生成类比为 3D 动画的生产范式（先生成结构再添加动画），是一个有启发性的视角

## 局限与展望

- 仅支持 Lottie 格式，未涵盖 SVG SMIL 动画或 CSS 动画
- 复杂动画的 token 序列仍然较长，受限于 VLM 的上下文窗口
- 未评估生成动画的时间一致性和运动自然度的人类评价
- 可扩展到交互式动画编辑和条件生成

## 相关工作与启发

- **vs OmniSVG/StarVector**: 这些方法仅能生成静态 SVG，LottieGPT 首次支持时间建模和动画生成
- **vs 像素视频生成**: 像素方法生成固定分辨率、不可编辑的输出，LottieGPT 输出可无限缩放且完全可编辑

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个矢量动画自回归生成框架，开辟新方向
- 实验充分度: ⭐⭐⭐⭐ 提出 LottieBench，多维度评估
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，贡献明确
- 价值: ⭐⭐⭐⭐⭐ 数据集+基准+方法的完整贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens](omnilottie_generating_vector_animations_via_parameterized_lottie_tokens.md)
- [\[ICML 2026\] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](../../ICML2026/video_generation/vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](../../CVPR2025/video_generation/teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)

</div>

<!-- RELATED:END -->
