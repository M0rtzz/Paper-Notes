---
title: >-
  [论文解读] OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens
description: >-
  [CVPR 2026][视频生成][Lottie] OmniLottie 提出一种将 Lottie JSON 文件转化为结构化命令-参数序列的 Lottie Tokenizer，使预训练 VLM 可以基于多模态交叉指令生成高质量矢量动画，并构建了 MMLottie-2M 大规模数据集支撑训练。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "Lottie"
  - "矢量动画"
  - "tokenization"
  - "多模态指令"
  - "VLM生成"
---

# OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens

**会议**: CVPR 2026  
**arXiv**: [2603.02138](https://arxiv.org/abs/2603.02138)  
**代码**: 待确认（论文提到 Project Page）  
**领域**:视频生成
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
OmniLottie 想把"生成一段矢量动画"这件看似奇特的事，变成预训练 VLM 已经擅长的"自回归吐序列"。它的关键观察是：Lottie 本质是一份 JSON，但这份 JSON 里绝大多数字符是括号、键名、缩进这类格式噪声，真正承载动画语义的只有形状轮廓、关键帧插值和颜色变换。于是整条 pipeline 先用一个 Tokenizer 把 Lottie JSON 榨成紧凑的"命令 + 参数"序列，再让一个词表被扩展过的 VLM 接收文本/图像/草图等多模态指令，自回归地把这串 token 一个个生成出来，最后反解码回合法的 Lottie 文件直接送进渲染器。为了让这套模型有东西可学，作者还配套造了一个 200 万规模的矢量动画数据集。

### 关键设计

**1. Lottie Tokenizer：把冗长 JSON 压成命令-参数序列，让序列短到模型学得动。**

直接拿原始 Lottie JSON 喂模型的最大问题是序列爆炸——一个普通动画的 JSON 文本动辄 10k+ token，里面的括号、键名、缩进对动画语义毫无贡献，却把有效信号稀释得无法学习。Tokenizer 的做法是遍历 JSON 树，只抽出三类真正有意义的信息：描述几何轮廓的**形状命令**（如 `MOVE_TO(x, y)`、`BEZIER(cx1, cy1, cx2, cy2, x, y)`）、描述关键帧插值的**动画函数**（如 `EASE_IN(start_frame, end_frame, start_val, end_val)`），以及颜色、透明度、变换矩阵这类**控制参数**。所有格式冗余被丢弃后，序列长度压到原来的 ~15-20%，也就是从 10k+ token 缩到 ~1-2k token。序列短一个数量级，自回归模型才能在有限上下文里看清一整个动画的结构，这也是后面消融里它被证明为最关键组件的原因。

**2. OmniLottie 模型：在预训练 VLM 词表里塞进 Lottie token，把动画生成接进语言建模范式。**

有了紧凑序列，下一步是让一个已经懂语言和图像的 VLM（如 LLaVA）去生成它，而不是从头训一个专用模型。具体做法是在 VLM 原有词表上扩出 ~200 个 Lottie 专属 token，对应各种形状命令名和动画函数名；坐标、颜色这类连续参数值则先量化再用数值 token 表示，这样整个 Lottie 序列就被纳入同一个离散词表，训练时只需标准的 next-token prediction 损失。复用预训练 VLM 的好处是直接继承了它的多模态理解力——输入可以是文本描述（"画一个弹跳的球"）、参考图像，也可以是一张草图，模型都能把它对齐到要生成的动画语义上。

**3. MMLottie-2M：补上 200 万级矢量动画-文本配对，把"没有训练数据"这块短板填掉。**

这个任务此前几乎无人做的现实原因之一是没有大规模配对数据。作者从 LottieFiles 等平台收集了 200 万个专业设计师制作的 Lottie 动画，再用 VLM 自动为每个动画补上文本描述和视觉标注（关键帧截图），凑成"动画—文本—视觉"三元对。规模和专业性是这个数据集的价值所在：消融显示去掉它做预训练后 CLIP Score 从 0.41 掉到 0.31，说明模型很大程度上是靠这批数据学会了动画该长什么样。

### 一个完整示例

以指令"画一个弹跳的球"为例走一遍：模型先把这句文本编码进上下文，然后自回归地吐出 Lottie token——先用一串 `MOVE_TO` / `BEZIER` 命令勾出小球的圆形轮廓，再用 `EASE_IN` / `EASE_OUT` 动画函数给纵坐标排上几个关键帧（落下时缓出、触底时回弹），最后附上颜色和变换参数。整段输出大约 1.5k token，而同一个动画若用原始 JSON 表示需要 10k+ token。反解码后得到一份合法 Lottie 文件，直接丢进渲染器就能在手机端流畅播放，体积只有等效像素视频的约 1/100。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](lottiegpt_vector_animation_generation.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[ICML 2026\] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](../../ICML2026/video_generation/vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)

</div>

<!-- RELATED:END -->
