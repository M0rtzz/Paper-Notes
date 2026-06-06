---
title: >-
  [论文解读] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation
description: >-
  [ICCV 2025][模型压缩][sketch generation] > 提出 VQ-SGen，将每个笔画视为独立实体并解耦其形状与位置信息，通过向量量化（VQ）构建紧凑离散的笔画码本，再用级联自回归 Transformer 逐步生成笔画的语义标签、形状和位置…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "sketch generation"
  - "量化"
  - "stroke representation"
  - "autoregressive"
  - "creative sketch"
---

# VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation

**会议**: ICCV 2025  
**arXiv**: [2411.16446](https://arxiv.org/abs/2411.16446)  
**代码**: [项目页面](https://enigma-li.github.io/projects/VQ-SGen/VQ-SGen.html)  
**领域**: 模型压缩  
**关键词**: sketch generation, vector quantization, stroke representation, autoregressive, creative sketch  

## 一句话总结

> 提出 VQ-SGen，将每个笔画视为独立实体并解耦其形状与位置信息，通过向量量化（VQ）构建紧凑离散的笔画码本，再用级联自回归 Transformer 逐步生成笔画的语义标签、形状和位置，在 CreativeSketch 数据集上显著超越现有方法。

## 研究背景与动机

创意素描生成（Creative Sketch Generation）旨在生成多样、复杂且富有美感的手绘草图，区别于传统的简单草图生成。现有方法存在以下问题：

**像素级方法**（DoodlerGAN、DoodleFormer）：将草图作为整体图像或分部件像素生成，忽略了笔画之间的**内在结构关系**（形状、相对位置），导致局部模糊、笔画散乱或断裂。

**笔画点序列方法**（SketchKnitter）：通过扩散模型重排笔画点，但没有"笔画实体"的概念，在复杂创意草图上效果很差。

**缺少紧凑表示**：连续的笔画嵌入空间过大，生成器难以高效采样。

核心想法：**将每个笔画视为独立实体**（entity），解耦形状和位置，通过 VQ 构建离散紧凑的码本空间。离散表示不仅减少了冗余，还自然形成了语义感知的聚类——相似形状的笔画（如鸟嘴、眼睛、翅膀）聚在一起，为生成器提供了理想的采样基础。

## 方法详解

### 整体框架（两阶段）

**Stage 1：解耦表示学习**
- 每个笔画 $\bm{s}_i = (\bm{I}_i, b_i, l_i)$：形状图像、位置（边界框中心+尺寸）、语义标签
- 解耦：平移笔画使边界框居中 → 形状图像 $\bm{I}_i$；位置 $b_i = (w_i/2, h_i/2, x_i, y_i)$
- 分别学习形状码本 $\bm{D}_s$ 和位置码本 $\bm{D}_l$

**Stage 2：自回归生成**
- 级联 Transformer 解码器逐步生成：标签 → 形状码 + 位置码

### 关键设计1：向量量化笔画表示

**笔画潜在嵌入**：CNN 自编码器将笔画图像 $\bm{I}_i$ 编码为潜在嵌入 $\bm{e}_i^s$，训练目标为重建损失 + CoordConv + 距离场监督。

**形状码本学习**：1D CNN 编码器 $\mathcal{E}^f$ 将笔画嵌入序列压缩后，通过最近邻量化映射到码本 $\bm{D}_s$（$V$ 个码字）：

$$v_i^s = \arg\min_{j \in [0, V)} \|\bm{z}_i^s - \bm{c}_j\|$$

**VQ 训练损失**：

$$\mathcal{L}_{VQ} = \frac{1}{N}\sum_{i=1}^{N} \alpha(\|\bm{z}_i^s - \text{sg}[\bm{c}_{v_i^s}]\|_2^2 + \|\text{sg}[\bm{z}_i^s] - \bm{c}_{v_i^s}\|_2^2) + \|\bm{z}_i^s - \mathcal{D}^f(\bm{c}_{v_i^s})\|_2^2$$

包含承诺损失、码本损失和重建损失。位置码本 $\bm{D}_l$ 使用相同流程学习。

### 关键设计2：级联自回归生成器

将草图生成分解为：

$$p(\bm{S}) = \prod_{i=1}^{N} p(v_i^s, v_i^l | l_i) \cdot p(l_i)$$

**标签 Transformer $\mathcal{T}^l$**：
- 输入：形状码嵌入 + 位置码嵌入 + 标签嵌入（三者相加融合）
- 输出：下一个笔画的语义标签 $l_{i+1}$

**码字 Transformer $\mathcal{T}^c$**：
- 接受两个输入：① 预测的标签 $l_{i+1}$ 提供语义引导；② $\mathcal{T}^l$ 的融合特征提供历史上下文
- 输出：两个分支分别预测形状码索引 $v_{i+1}^s$ 和位置码索引 $v_{i+1}^l$

**生成损失**：负对数似然

$$\mathcal{L}_{gen} = -\log p(\bm{S})$$

### 码空间分析

UMAP 可视化显示形状码本自动形成了语义感知的聚类：鸟嘴、眼睛、尾巴、翅膀、腿分别聚成不同cluster，头和身体形成内部大cluster。这说明 VQ 过程在无监督情况下捕获了有意义的语义结构。

## 实验关键数据

### 主实验：CreativeSketch 数据集对比

| 方法 | Creative Birds | | | Creative Creatures | | | |
|------|------|------|------|------|------|------|------|
| | FID↓ | GD↑ | CS↑ | FID↓ | GD↑ | CS↑ | SDS↑ |
| 训练数据 | - | 19.40 | 0.45 | - | 18.06 | 0.60 | 1.91 |
| SketchKnitter | 74.42 | 14.23 | 0.14 | 64.34 | 12.34 | 0.42 | 1.32 |
| DoodlerGAN | 39.95 | 16.33 | 0.69 | 43.94 | 14.57 | 0.55 | 1.45 |
| DoodleFormer | 17.48 | 17.83 | 0.57 | 20.43 | 16.23 | 0.53 | 1.68 |
| **VQ-SGen** | **15.78** | **18.92** | 0.53 | **17.61** | **17.42** | **0.57** | **1.86** |

- Creative Birds：FID 降低 1.7（15.78 vs 17.48），GD 提升 1.09
- Creative Creatures：FID 降低 2.82，GD 提升 1.19，全部四个指标最优

### 消融实验：核心组件贡献

| 配置 | Creative Birds FID↓ | Creative Birds GD↑ | Creative Creatures FID↓ | Creative Creatures GD↑ |
|------|---------------------|---------------------|-------------------------|------------------------|
| w/o VQ（连续空间） | 48.53 | 13.34 | 54.56 | 14.02 |
| w/o Decouple（不解耦） | 17.14 | 18.12 | 19.42 | 16.42 |
| w/o $\mathcal{T}^l$（无标签） | 16.51 | 18.35 | 19.21 | 16.14 |
| 2048×512 码本 | 26.23 | 15.34 | 43.21 | 14.51 |
| 4096×512 码本 | 16.92 | 18.27 | 18.44 | 16.14 |
| **8192×512 (VQ-SGen)** | **15.78** | **18.92** | **17.61** | **17.42** |

### 关键发现

- **VQ 是核心**：去掉 VQ 后 FID 从 15.78 暴涨至 48.53，证明离散紧凑空间对生成至关重要
- **解耦很关键**：不解耦形状和位置导致 FID 从 15.78 升至 17.14（CB）和 17.61 升至 19.42（CC）
- **标签 Transformer 锦上添花**：去掉标签后影响较小（第二好性能），说明方法可扩展到无标签数据集
- 码本大小从 2048 增至 8192，FID 持续改善，但更大码本可能带来利用率问题
- 用户研究中 50 人评测，VQ-SGen 在创意性、结构完整性和整体偏好上均优于 DoodleFormer

## 亮点与洞察

1. **笔画级建模范式**：与像素级和点序列级方法对比，将笔画作为独立实体是更自然的抽象层次，能同时捕获局部精细结构和全局语义关系。
2. **解耦设计优雅**：形状和位置的解耦让 VQ 码本专注于形状变化，而位置信息单独编码，避免了两者干扰。
3. **零监督语义聚类**：VQ 码本自动形成语义感知的聚类是一个令人惊喜的发现，为可控生成提供了天然支持。
4. **灵活的条件生成**：框架天然支持类别标签条件、文本条件和笔画补全三种模式，展示了良好的扩展性。

## 局限性

- 仅在 CreativeSketch 数据集（birds + creatures 两类）上验证，数据多样性有限。
- 码本大小的选择需要权衡——太小表示力不足，太大可能导致码字利用率低。
- 对形状变化极大的笔画（如复杂的动物耳朵），VQ 重建仍有失真。
- 生成结果的创意性主要依赖训练数据分布，无法产生超出训练数据范畴的全新概念。
- 自回归生成的顺序依赖原始绘制顺序，未探索更优的笔画排序策略。

## 相关工作与启发

- ContextSeg 首次提出单笔画实体概念用于语义分割，SketchXAI 用于可解释性分析，VQ-SGen 将其扩展到生成任务。
- StrokeNUWA 也使用 VQ-VAE 处理矢量图形但基于 LLM 生成；VQ-SGen 的方法更轻量且专注于创意草图。
- VQ 表示 + 自回归 Transformer 的 two-stage 范式可以迁移到其他结构化序列生成任务（如矢量图标设计、简笔画动画）。
- 文本到草图生成的 CLIP 条件化方式值得在交互式创作工具中探索。

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation](../../ACL2025/model_compression/brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re.md)
- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](../../CVPR2025/model_compression/sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)

</div>

<!-- RELATED:END -->
