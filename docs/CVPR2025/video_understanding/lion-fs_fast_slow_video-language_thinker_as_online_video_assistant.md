---
title: >-
  [论文解读] LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant
description: >-
  [CVPR 2025][视频理解][在线视频助手] 提出 LION-FS 在线视频助手框架，借鉴"快思考-慢思考"认知理论，用 Fast Path（基于路由的 Token 聚合与丢弃）实现高效实时响应判断，用 Slow Path（多粒度关键帧增强）在响应生成时注入细粒度空间和交互特征，在 Ego4D/Ego-Exo4D 基准上全面超越现有方法。
tags:
  - CVPR 2025
  - 视频理解
  - 在线视频助手
  - 快慢思维
  - Token路由
  - 关键帧增强
  - 第一人称视频
---

# LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant

**会议**: CVPR 2025  
**arXiv**: [2503.03663](https://arxiv.org/abs/2503.03663)  
**代码**: https://github.com/JiuTian-VL/LION-FS  
**领域**: 视频理解 / 多模态大模型  
**关键词**: 在线视频助手, 快慢思维, Token路由, 关键帧增强, 第一人称视频

## 一句话总结

提出 LION-FS 在线视频助手框架，借鉴"快思考-慢思考"认知理论，用 Fast Path（基于路由的 Token 聚合与丢弃）实现高效实时响应判断，用 Slow Path（多粒度关键帧增强）在响应生成时注入细粒度空间和交互特征，在 Ego4D/Ego-Exo4D 基准上全面超越现有方法。

## 研究背景与动机

**领域现状**：在线视频助手需要持续接收第一人称视频流，实时判断何时需要回复用户，并给出专业准确的响应。VideoLLM-online 提出的 LIVE 框架是该领域的先驱工作，建立了视频流对话的基础范式。

**现有痛点**：LIVE 存在三个严重问题：(1) 响应判断精度低——仅使用低帧率图像特征，LLM 难以捕捉帧间时序关系；(2) 响应内容不精确——每帧固定保留少量 token，未利用第一人称视角的特殊性，无法捕捉自适应的细粒度信息；(3) 训练/推理效率差——为提升效果而扩展所有帧的 token，但响应判断阶段本不需要这么多 token，token 扩展应集中在关键帧的响应生成阶段。

**核心矛盾**：在线视频助手要同时满足实时性（高帧率、低延迟）和准确性（细粒度理解、精准回复），二者存在天然冲突——更多 token 带来更好理解但更慢推理。

**本文目标** (1) 如何高效处理高帧率视频流并准确判断何时需要回复；(2) 如何在不影响效率的前提下提升响应的精度和细粒度。

**切入角度**：借鉴 Kahneman 的"快思考/慢思考"理论——简单的响应判断（是否需要回复）对应快速直觉的 System 1，复杂的响应生成对应深思熟虑的 System 2。将两种任务解耦，分别用不同策略优化。

**核心 idea**：将在线视频对话解耦为快速路径（路由驱动的高效响应判断）和慢速路径（多粒度关键帧增强的精细响应生成），各自优化效能与效率。

## 方法详解

### 整体框架

LION-FS 的整体流程分为两条路径。Fast Path：用双编码器（SigLIP 2FPS + EgoVLPv2 8FPS）提取通用空间特征和第一人称时序特征，通过 Token Aggregation Router 自适应融合两类特征（不增加 token 数），再通过 Token Dropping Router 丢弃冗余 token 实现稀疏解码，高效地逐帧判断是否需要回复。Slow Path：当判断需要回复时，将当前帧定义为关键帧，对其进行多粒度增强——全局网格增强（Grid Tokens）和局部目标增强（Box Tokens），注入多模态 Thinking Template 引导更精准的响应生成。

### 关键设计

1. **Token Aggregation Router（Token 聚合路由器）**:

    - 功能：自适应融合通用图像编码器和第一人称视频编码器的特征，不增加 token 数量
    - 核心思路：SigLIP（2FPS）提取每帧 10 个 token（1 CLS + 9 个 3×3 池化 token），EgoVLPv2（8FPS）每 4 帧一组提取 10 个 token。两种特征在时间上对齐后，用一个 MLP 路由器根据 SigLIP 的 CLS token（Visual Guidance）生成权重比例，对两类 token 进行加权融合：$[\text{Frm}]_i = G_f(\text{[VG]})_0 \times [\text{Frm}_s]_i + G_f(\text{[VG]})_1 \times [\text{Frm}_t]_i$。
    - 设计动机：简单拼接两类 token 会翻倍序列长度影响 LLM 解码效率；直接相加忽略了不同场景下两类特征重要性不同。路由器可以根据场景内容动态决定视角切换时更信赖哪种编码器。

2. **Token Dropping Router（Token 丢弃路由器）**:

    - 功能：在 LLM 解码的每层 Transformer 中自适应丢弃冗余视觉 token，加速推理
    - 核心思路：在每层为每个 token 计算路由权重 $r_{(i,n)}^l = w_\theta^T [\text{Frm}]_{(i,n)}^l$，只保留权重高于 $\beta$ 分位数阈值的 token 参与注意力和 FFN 计算。低于阈值的 token 直接跳过当前层，保持上一层的表示不变。$\beta$ 控制丢弃比例。
    - 设计动机：第一人称场景中通常只有手和交互区域是关键信息，大量 token 表示的是低信息量的背景或近乎静止的连续帧重复信息。丢弃这些冗余 token 可以显著减少 FLOPs。

3. **Multi-granularity Keyframe Augmentation（多粒度关键帧增强）**:

    - 功能：在响应生成时为关键帧注入细粒度全局和局部特征（**training-free**）
    - 核心思路：全局增强——将关键帧分成 4 个网格，每个网格做 3×3 池化得到 Grid Tokens，等于把 1 帧变成 4 帧的信息量。局部增强——用 Faster R-CNN 检测手部位置，根据距离匹配交互物体的 bounding box，从 576 个 patch token 中选取 box 区域内的 token 进行全局池化得到 Box Tokens。两种 token 被组装进 Multimodal Thinking Template："Stream: [Frame Tokens] [Grid Tokens] User: Please focus on [Box Tokens]. Assistant: "，用作引导精细响应生成的多模态提示。
    - 设计动机：Fast Path 每帧仅 10 个 token，信息量不足以支撑精细的响应生成。在所有帧都增加细粒度特征不现实（影响实时性），但关键帧是动作/事件转折点，集中在关键帧增强是性价比最高的策略。

### 损失函数 / 训练策略

训练目标包含两部分：Streaming Loss（响应判断）监督模型在每帧预测 EOS token 的概率，LM Loss（语言建模）监督自回归生成响应文本。总损失 $\text{Loss} = \frac{1}{N}\sum_j(-ws_j\log P_j^{[\text{EOS}]} - l_{j+1}\log P_j^{[\text{Txt}]_{j+1}})$。Slow Path 是 training-free 的，只有 Fast Path 需要训练。

## 实验关键数据

### 主实验

| 数据集 | 方法 | LL-PPL↓ | TimeDiff↓ | Fluency↑ | LM-Correctness↑ |
|--------|------|---------|-----------|----------|------------------|
| Ego-Exo4D | VideoLLM-online | 2.24 | 0.78 | 33.7% | 44.8% |
| Ego-Exo4D | VideoLLM-MoD | 2.12 | 0.82 | 33.8% | 45.3% |
| Ego-Exo4D | **LION-FS** | **2.04** | **0.74** | **36.5%** | **48.2%** |
| Ego4D | VideoLLM-online | 2.40 | 2.04 | 45.3% | 49.0% |
| Ego4D | **LION-FS** | **2.09** | 2.15 | **46.1%** | **52.4%** |

### 消融实验

| 配置 | LL-PPL↓ | TimeDiff↓ | Fluency↑ | LM-Correctness↑ |
|------|---------|-----------|----------|------------------|
| 仅 SigLIP (10 tokens) | 2.24 | 0.78 | 33.7% | 44.8% |
| 仅 EgoVLP (10 tokens) | 2.29 | 1.05 | 36.8% | 47.8% |
| 简单拼接 (20 tokens) | 2.25 | 1.65 | 27.7% | 45.8% |
| Adaptive Routing (10 tokens) | **2.25** | **0.67** | **38.1%** | **48.0%** |
| + Token Dropping β=0.5 | 2.16 | 0.74 | 36.5% | 47.0% |

### 关键发现

- **Adaptive Routing 是最优聚合策略**：相比拼接（20 tokens 但 Fluency 暴跌至 27.7%）和简单加法，路由聚合在不增加 token 数的前提下同时提升了准确性和时序感知
- **EgoVLPv2 的第一人称特征对 TimeDiff 帮助最大**（0.67 vs 0.78），说明第一人称预训练捕捉了关键的动作时序信号
- **Token Dropping 在 β=0.5 时取得最佳平衡**：FLOPs 降低 16%（61.44T→51.40T），训练加速 1.12×，性能仅有微小下降
- Slow Path 是 training-free 的，直接在推理时增强关键帧，部署灵活

## 亮点与洞察

- **"快慢解耦"范式非常优雅**：将响应判断（简单任务）和响应生成（复杂任务）解耦，分别用高效路由和精细增强来优化，避免了"要么全加 token 效率低、要么全精简效果差"的困境。这种任务难度自适应的思路可迁移到其他 VLM streaming 场景。
- **Training-free 的 Slow Path 巧妙地利用了任务特性**：只有在需要回复时才做昂贵的细粒度增强，通过 Thinking Template 将增强 token "注入"到 LLM 的生成前缀中。无需额外训练即可显著提升响应质量。
- **双编码器路由融合**不只是特征拼接，而是让 SigLIP 的 CLS token 作为"调度员"根据场景内容自动选择信赖哪个编码器的信息，这种"视觉引导"的路由策略很直观且有效。

## 局限与展望

- Slow Path 依赖 Faster R-CNN 做手部和物体检测，增加了推理时的额外延迟和系统复杂度
- Box Tokens 提取基于物体检测模型，如果检测失败会级联影响响应质量
- Token Dropping Router 的丢弃比例 $\beta$ 是全局固定的，不同场景的复杂度差异可能需要自适应的 $\beta$
- 仅在 Ego4D/Ego-Exo4D 两个第一人称数据集上验证，第三人称场景的泛化性未知
- Thinking Template 的格式是手工设计的，更灵活的 prompt 设计或可进一步提升生成质量

## 相关工作与启发

- **vs VideoLLM-online (LIVE)**: LIVE 处理低帧率视频、不区分响应判断和生成、使用固定 token 数。LION-FS 在这三个维度上全面改进——4× 帧率提升、快慢路径解耦、关键帧差异化增强
- **vs VideoLLM-MoD**: MoD 引入 Mixture-of-Depths 策略减少计算，但缺乏第一人称特征和多粒度增强。LION-FS 的 Token Dropping Router 理念类似但结合了 egocentric 编码器
- **vs SlowFast-LLaVA**: SlowFast-LLaVA 在不同帧率上做不同粒度池化来丰富特征，但是离线方法。LION-FS 将 Fast-Slow 概念拓展到在线视频流场景，强调实时性

## 评分

- 新颖性: ⭐⭐⭐⭐ 快慢解耦+双编码器路由+training-free增强组合新颖，但各组件独立看并非全新
- 实验充分度: ⭐⭐⭐⭐ 消融非常详细，但仅两个数据集略显不足
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、图示精美、motivation 阐述到位
- 价值: ⭐⭐⭐⭐ 为在线视频助手提供了实用的快慢推理框架，对可穿戴AI有启发意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] EgoLife: Towards Egocentric Life Assistant](egolife_towards_egocentric_life_assistant.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)

<!-- RELATED:END -->
