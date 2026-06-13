---
title: >-
  [论文解读] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval
description: >-
  [CVPR 2025][视频理解][视频-文本检索] 提出 AVIGATE 框架，通过门控注意力机制选择性地融合音频与视觉信息（过滤无用音频噪声），并设计自适应间距对比损失处理视频-文本之间模糊的正负关系，在多个视频-文本检索基准上取得 SOTA。 领域现状：视频-文本检索主要依赖视觉和文本特征进行跨模态对齐…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频-文本检索"
  - "音频引导"
  - "门控注意力"
  - "对比学习"
  - "多粒度对齐"
---

# Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval

**会议**: CVPR 2025  
**arXiv**: [2504.02397](https://arxiv.org/abs/2504.02397)  
**代码**: [https://github.com/](https://github.com/) (项目主页: [http://cvlab.postech.ac.kr/research/AVIGATE](http://cvlab.postech.ac.kr/research/AVIGATE))  
**领域**: 视频理解 / 多模态  
**关键词**: 视频-文本检索, 音频引导, 门控注意力, 对比学习, 多粒度对齐

## 一句话总结

提出 AVIGATE 框架，通过门控注意力机制选择性地融合音频与视觉信息（过滤无用音频噪声），并设计自适应间距对比损失处理视频-文本之间模糊的正负关系，在多个视频-文本检索基准上取得 SOTA。

## 研究背景与动机

**领域现状**：视频-文本检索主要依赖视觉和文本特征进行跨模态对齐，大量方法基于 CLIP 预训练特征构建多粒度匹配方案（如 CLIP4Clip、X-Pool、UATVR 等），在检索精度上持续提升。

**现有痛点**：视频中的音频信号（如说话人身份、背景声音、情绪线索等）被大多数方法忽略。少数利用音频的方法（如 ECLIPSE、TEFAL）盲目地假设音频总是有帮助的，但现实中许多视频的音频是无关的背景音乐或噪声，盲目融合反而会损害视频表征质量。此外，TEFAL 需要文本和视频/音频联合处理，导致检索时每来一个新 query 都要重新跑整个数据库，效率极低。

**核心矛盾**：音频模态对视频理解确实有价值（如对话内容、环境声音），但并非所有视频的音频都是有用的——存在"信息性音频 vs 噪声音频"的区分问题，现有方法缺乏对音频质量的动态判断。

**本文目标** (1) 如何动态判断音频是否有用并选择性融合；(2) 如何在对比学习中更好地处理语义相近的负样本对。

**切入角度**：作者观察到，视觉-音频的相关性在不同视频中差异很大，因此可以设计一个门控机制来"自动调节音频的贡献权重"——当音频有信息量时放大其影响，当音频是噪声时关闭其通道。

**核心 idea**：用门控注意力机制动态过滤音频噪声，并用自适应间距对比损失感知负样本之间的语义亲近度，实现高效且精准的视频-文本检索。

## 方法详解

### 整体框架

AVIGATE 使用三个独立编码器分别处理三种模态：CLIP 图像编码器提取帧级嵌入，CLIP 文本编码器提取文本嵌入，AST（Audio Spectrogram Transformer）提取音频嵌入。音频嵌入经过 audio resampler 压缩到固定数量的 token 后，与帧嵌入一起输入门控融合 Transformer，输出最终的视频表征，再通过多粒度对齐方案与文本嵌入计算相似度。

### 关键设计

1. **Audio Resampler（音频重采样器）**:

    - 功能：将 AST 输出的密集音频嵌入压缩为固定长度 $M$ 个嵌入
    - 核心思路：使用基于 query 的 Transformer，以 $M$ 个可学习 query 嵌入通过交叉注意力机制从原始音频嵌入中提取信息。这样既保留了关键音频特征，又大幅降低了后续融合的计算量。AST 参数在训练中冻结。
    - 设计动机：音频采样率远高于视频帧率，直接融合所有音频 token 计算开销太大，需要一个信息压缩层。

2. **Gated Fusion Transformer（门控融合 Transformer）**:

    - 功能：将音频嵌入与帧嵌入进行选择性融合，动态决定音频的贡献程度
    - 核心思路：由 $L$ 层 Gated Fusion Block 组成。每层包含"融合过程"和"精炼过程"。融合过程中，帧嵌入作为 query、音频嵌入作为 key/value 进行多头交叉注意力，输出乘以门控分数 $g_{\text{mha}}$ 后加残差；再经 FFN 并乘以 $g_{\text{ffn}}$。精炼过程通过自注意力进一步增强帧间关系。门控分数由 Gating Function 生成：将音频和帧嵌入分别平均池化后拼接，通过两个独立 MLP + tanh 激活得到两个标量门控值。当门控分数高时强调音频贡献，低时保护视觉内容不受噪声干扰。
    - 设计动机：和 ECLIPSE/TEFAL 的静态融合不同，门控机制让模型学会"什么时候该听音频、什么时候该忽略音频"，解决了噪声音频干扰的核心问题。

3. **Adaptive Margin-based Contrastive Loss（自适应间距对比损失）**:

    - 功能：在对比学习中为每对负样本动态设置不同的 margin
    - 核心思路：对每对负样本 $(V_i, T_j)$，计算视觉模态内相似度 $c_{ij}^v$ 和文本模态内相似度 $c_{ij}^t$，然后令自适应间距 $m_{ij} = \min(\lambda(1 - (c_{ij}^v + c_{ij}^t)/2), \delta)$。语义越不相似的负样本 margin 越大，迫使模型拉开它们的距离；语义相近的负样本 margin 较小，避免过度挤压造成泛化性下降。
    - 设计动机：传统对比损失对所有负样本一视同仁，但实际中许多"负样本"之间存在语义关联（如两段描述类似场景的视频），固定 margin 会迫使这些语义相关的样本也被推得很远，影响泛化。

### 损失函数 / 训练策略

最终损失为双向对比损失（video-to-text + text-to-video），每个方向都在负样本相似度上加了自适应间距 $m_{ij}$。相似度分数通过多粒度对齐方案计算：全局对齐（帧嵌入平均池化后与文本余弦相似度）+ 局部对齐（每帧与文本的相似度用 log-sum-exp 聚合）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | AVIGATE | 之前SOTA (UATVR) | 提升 |
|--------|------|---------|-----------|------|
| MSR-VTT (ViT-B/16) | T2V R@1 | **52.1** | 50.8 | +1.3 |
| MSR-VTT (ViT-B/16) | V2T R@1 | **51.2** | 48.1 | +3.1 |
| MSR-VTT (ViT-B/16) | RSum | **429.0** | 422.4 | +6.6 |
| MSR-VTT (ViT-B/32) | T2V R@1 | **50.2** | 47.5 | +2.7 |
| MSR-VTT (ViT-B/32) | V2T R@1 | **49.7** | 46.9 | +2.8 |

### 消融实验

| 配置 | RSum (MSR-VTT) | 说明 |
|------|----------------|------|
| Full model (AVIGATE) | **429.0** | 完整模型 |
| w/o audio | ~422 | 去掉音频融合，退化为纯视觉-文本模型 |
| w/o gating (blind fusion) | ~420 | 去掉门控，盲目融合音频反而降低性能 |
| w/o adaptive margin | ~425 | 使用固定 margin 对比损失，性能下降 |

### 关键发现

- 门控机制是关键：盲目融合音频反而不如纯视觉模型，门控机制让音频融合真正带来正向增益
- 自适应 margin 相比固定 margin 和无 margin 均有提升，在语义相近样本多的数据集上优势更明显
- AVIGATE 保持了高检索效率——视频和文本独立编码，不需要像 TEFAL 那样每个 query 都重新跑数据库

## 亮点与洞察

- **门控融合设计非常巧妙**：用 tanh 激活的标量门控值来调控音频影响，简洁有效地解决了"音频何时有用"的判断问题。这个思路可以迁移到任何多模态融合场景中，只要存在"某个模态不总是有用"的情况。
- **自适应间距的核心insight**：模态内相似度可以作为跨模态语义关联的代理信号——视觉上相似的视频，其对应文本也倾向于语义相近。这个 prior 让 margin 的设计有了合理依据。
- **多粒度对齐 + 独立编码的组合**确保了检索效率和精度的良好平衡，在实际部署中很重要。

## 局限与展望

- 仅在检索任务上验证，未测试在视频字幕生成、视频问答等下游任务中音频门控融合的效果
- 门控分数是一个全局标量，不能对音频的不同时段/频段进行细粒度的选择
- 音频编码器 AST 冻结不训练，可能导致音频表征与 CLIP 空间不完全对齐
- 未探索更强的音频-文本直接交互（本文音频仅与视觉融合，不直接与文本匹配）

## 相关工作与启发

- **vs ECLIPSE**: ECLIPSE 用交叉注意力融合音频-视觉，但没有门控机制来过滤噪声音频，本文的门控设计是核心差异
- **vs TEFAL**: TEFAL 需要视频和文本联合处理生成表征，检索时效率低；AVIGATE 独立编码视频和文本，效率更高
- **vs UATVR**: UATVR 通过分布匹配处理不确定性，但不用音频；AVIGATE 在 UATVR 的基础上引入音频模态取得了进一步提升

## 评分

- 新颖性: ⭐⭐⭐⭐ 门控融合和自适应 margin 各有创新但整体偏增量
- 实验充分度: ⭐⭐⭐⭐ 多个基准+消融较完整，但缺乏门控分数的可视化分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图示直观
- 价值: ⭐⭐⭐⭐ 门控融合思路有实用价值，可迁移到其他多模态场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](../../ECCV2024/video_understanding/text-guided_video_masked_autoencoder.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)

</div>

<!-- RELATED:END -->
