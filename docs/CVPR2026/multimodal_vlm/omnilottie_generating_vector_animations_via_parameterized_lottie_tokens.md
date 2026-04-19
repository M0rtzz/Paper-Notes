---
title: >-
  [论文解读] OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens
description: >-
  [CVPR 2026][多模态][Lottie] OmniLottie 提出一种将 Lottie JSON 文件转化为结构化命令-参数序列的 Lottie Tokenizer，使预训练 VLM 可以基于多模态交叉指令生成高质量矢量动画，并构建了 MMLottie-2M 大规模数据集支撑训练。
tags:
  - CVPR 2026
  - 多模态
  - Lottie
  - 矢量动画
  - tokenization
  - 多模态指令
  - VLM生成
---

# OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens

**会议**: CVPR 2026  
**arXiv**: [2603.02138](https://arxiv.org/abs/2603.02138)  
**代码**: 待确认（论文提到 Project Page）  
**领域**: 多模态VLM / 矢量动画生成  
**关键词**: Lottie, 矢量动画, tokenization, 多模态指令, VLM生成

## 一句话总结
OmniLottie 提出一种将 Lottie JSON 文件转化为结构化命令-参数序列的 Lottie Tokenizer，使预训练 VLM 可以基于多模态交叉指令生成高质量矢量动画，并构建了 MMLottie-2M 大规模数据集支撑训练。

## 研究背景与动机

### 领域现状
矢量动画（如 SVG 动画、Lottie 格式）在 UI 设计、移动应用、网页中广泛使用。它们体积小、分辨率无关、可编程编辑。然而，自动生成矢量动画是一个尚未充分探索的方向——现有工作主要集中在静态矢量图或像素级视频生成。

### 现有痛点

**Lottie JSON 的冗余性**：原始 Lottie 文件包含大量不变的结构元数据和格式 token（如括号、键名），对于学习动画生成来说是严重的噪声

**缺乏训练数据**：没有大规模的矢量动画-文本配对数据集

**VLM 不理解动画格式**：现有 VLM 只能生成文本/图像，无法直接输出结构化的动画描述

### 核心矛盾
Lottie 是最流行的矢量动画格式，但其 JSON 表示对机器学习不友好——冗余的格式 token 使序列长度爆炸，困难了学习有效的生成模型。

### 核心 idea
设计一种 **Lottie Tokenizer**，将 Lottie JSON 转换为紧凑的命令+参数序列（去除所有结构冗余），使预训练 VLM 可以像生成自然语言一样自回归生成矢量动画。

## 方法详解

### 整体框架
OmniLottie 包含三个核心组件：
1. **Lottie Tokenizer**：JSON → 命令+参数序列
2. **OmniLottie 模型**：基于预训练 VLM，接受多模态指令输入，自回归生成 Lottie token 序列
3. **MMLottie-2M 数据集**：大规模矢量动画+文本/视觉标注

### 关键设计

#### 1. Lottie Tokenizer
- **功能**：将 Lottie JSON 文件转换为结构化的命令-参数序列
- **核心思路**：遍历 JSON 树结构，提取三类有意义的信息：
    - **形状命令（Shape Commands）**：如 `MOVE_TO(x, y)`、`BEZIER(cx1, cy1, cx2, cy2, x, y)` 等描述几何形状的指令
    - **动画函数（Animation Functions）**：如 `EASE_IN(start_frame, end_frame, start_val, end_val)` 描述关键帧插值
    - **控制参数（Control Parameters）**：如颜色、透明度、变换矩阵等
- **设计动机**：去除了所有冗余的 JSON 格式信息（缩进、括号、键名等），将序列长度压缩到原来的 ~15-20%
- **与之前方法的区别**：直接使用 JSON 文本训练的方法需要处理 ~10k+ token 的序列，Lottie Tokenizer 压缩到 ~1-2k token

#### 2. OmniLottie 模型架构
- **功能**：基于预训练的 VLM（如 LLaVA），扩展 token 词表以包含 Lottie 命令 token，进行多模态指令到 Lottie 序列的自回归生成
- **核心思路**：
    - 在 VLM 的词表中加入 ~200 个 Lottie 特有 token（形状命令、动画函数名等）
    - 参数值（如坐标、颜色）使用量化后的数值 token 表示
    - 训练时使用标准的 next-token prediction 损失
- **设计动机**：利用预训练 VLM 的语言和视觉理解能力，将矢量动画生成转化为序列生成问题
- **多模态支持**：可以接受文本描述（"画一个弹跳的球"）、参考图像、草图等多模态输入

#### 3. MMLottie-2M 数据集
- **功能**：构建大规模矢量动画数据集，包含 200 万专业设计的矢量动画
- **核心思路**：从 LottieFiles 等平台收集专业设计师制作的 Lottie 动画，使用 VLM 自动生成文本描述和视觉标注
- **规模**：200万动画 + 文本描述 + 视觉注释（关键帧截图）

## 实验关键数据

### 主实验：矢量动画生成质量

| 方法 | FID ↓ | CLIP Score ↑ | 人类偏好 (%) |
|------|-------|-------------|-------------|
| DeepSVG + Motion | 142.3 | 0.21 | 12.3 |
| SVGDreamer | 98.7 | 0.28 | 22.8 |
| AnimateDiff (pixel) | 45.2 | 0.35 | 28.4 |
| **OmniLottie** | **38.6** | **0.41** | **36.5** |

### 消融实验

| 配置 | CLIP Score ↑ | 说明 |
|------|-------------|------|
| Full OmniLottie | 0.41 | 完整方法 |
| w/o Lottie Tokenizer (raw JSON) | 0.24 | 直接用 JSON 文本，序列太长质量下降 |
| w/o Animation Functions | 0.33 | 只生成静态形状，无动画 |
| w/o MMLottie Pretrain | 0.31 | 不使用大规模数据集预训练 |

### 关键发现
- **Lottie Tokenizer 是核心**——去掉后 CLIP Score 从 0.41 降到 0.24，因为原始 JSON 太冗长导致模型无法有效学习
- 生成的矢量动画在手机端可以流畅播放，体积仅为像素视频的 ~1/100
- 多模态指令的灵活性得到验证——文本、图像、草图等多种输入都能生成语义对齐的动画
- 模型可以生成包含多物体、多层次动画的复杂场景

## 亮点与洞察
- **将矢量动画生成转化为序列生成**——Lottie Tokenizer 的设计使这个看似奇特的任务与 LLM 范式完美对接
- **MMLottie-2M 填补数据空白**——200 万规模的专业矢量动画数据集是社区的重要资源
- **实用价值极高**——生成的 Lottie 文件可以直接用于 App/Web 开发，无需后处理
- **序列化格式设计的启发**——Lottie Tokenizer 的思路可以推广到其他结构化格式的生成（如 CAD、SVG、代码 AST）

## 局限与展望
- 当前仅支持 Lottie 格式，未扩展到 SVG 动画或 CSS 动画
- 复杂动画（如包含遮罩、混合模式、表达式的 Lottie）的生成质量尚需提升
- 量化参数值引入了精度损失——微妙的动画曲线可能被量化粗化
- 缺乏动画时序质量的自动评估指标——FID 和 CLIP Score 主要评估静态帧
- 模型无法交互式编辑已生成的动画

## 相关工作与启发
- **vs DeepSVG**：DeepSVG 关注静态矢量图的 VAE 生成，不支持动画。OmniLottie 专门针对动画动态
- **vs AnimateDiff**：AnimateDiff 生成像素视频。OmniLottie 生成矢量格式，体积小且可编辑
- **vs SVGDreamer**：SVGDreamer 用扩散模型生成 SVG，但不支持动画和多模态输入
- **启发**：结构化格式的 tokenization 是将传统设计工具与 AI 生成结合的关键桥梁

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将矢量动画生成建模为序列生成任务，Lottie Tokenizer 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 人类评估 + 自动指标 + 消融，但缺少动画时序质量评估
- 写作质量: ⭐⭐⭐⭐ 问题引入清晰，tokenizer 设计可视化做得好
- 价值: ⭐⭐⭐⭐⭐ 数据集+方法+应用价值三重贡献，对矢量动画生成领域有开创意义

<!-- RELATED:START -->

## 相关论文

- [StarVector: Generating Scalable Vector Graphics Code from Images and Text](../../CVPR2025/multimodal_vlm/starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)
- [VecGlypher: Unified Vector Glyph Generation with Language Models](vecglypher_unified_vector_glyph_generation_with_language_models.md)
- [Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](cubic_discrete_diffusion_discrete_visual_generation_on_high-dimensional_represen.md)
- [What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)

<!-- RELATED:END -->
