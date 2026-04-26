---
title: >-
  [论文解读] Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval
description: >-
  [CVPR 2025][文本视频检索] 将文本检索中ColBERT的后交互引入文本-视频检索，提出Video-ColBERT，通过在帧级和视频级两个层面执行MeanMaxSim交互，配合双Sigmoid损失训练独立且兼容的多粒度表征，在多个T2VR benchmark上超越现有双编码器方法。
tags:
  - CVPR 2025
  - 文本视频检索
  - 后交互
  - 多向量表示
  - ColBERT
  - Sigmoid损失
---

# Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval

**会议**: CVPR 2025  
**arXiv**: [2503.19009](https://arxiv.org/abs/2503.19009)  
**代码**: 无  
**领域**: 视频理解 / 多模态检索  
**关键词**: 文本视频检索, 后交互, 多向量表示, ColBERT, Sigmoid损失

## 一句话总结

将文本检索中ColBERT的后交互引入文本-视频检索，提出Video-ColBERT，通过在帧级和视频级两个层面执行MeanMaxSim交互，配合双Sigmoid损失训练独立且兼容的多粒度表征，在多个T2VR benchmark上超越现有双编码器方法。

## 研究背景与动机

**领域现状**：T2VR的高效方案多采用双编码器架构。主流方法基于CLIP适配——CLIP4Clip、X-CLIP、DRL等。

**现有痛点**：(1) 单向量表征难以编码复杂视频内容和文本概念；(2) 已有token级交互方法机制过复杂或最终表征受限——只对时间建模后的特征做交互，忽略独立帧的空间信息；(3) InfoNCE损失对噪声数据敏感。

**核心矛盾**：细粒度交互效果好但慢，单向量快但表达不足。ColBERT式后交互是中间地带，但现有T2VR方法未充分利用。

**本文目标**：(1) 空间和时空两层token级交互；(2) 合适的训练目标使两路表征都够强；(3) 保持双编码器效率。

**切入角度**：ColBERT的MaxSim让每个query token独立扫描文档。扩展到视频：对独立帧特征（空间）和时间建模后的特征（时空）分别做MaxSim，两路得分相加。

**核心 idea**：双层MeanMaxSim交互（MMS_F + MMS_V = MMS_FV），用双sigmoid损失独立训练两路表征。

## 方法详解

### 整体框架

双编码器：文本编码为token表征序列；视频侧图像编码器提取帧[CLS] token，temporal transformer产生上下文化视频特征。通过帧级和视频级双层MeanMaxSim计算相似度。

### 关键设计

1. **双层MeanMaxSim交互**:

    - 功能：空间和时空两粒度的query-video匹配
    - 核心思路：$MMS_F = \frac{1}{M}\sum_j \max_i (\mathbf{q}_j \cdot \mathbf{f}_i)$对帧特征，$MMS_V$对视频特征做同样操作。相比ColBERT的求和改用均值适应变长查询。最终$MMS_{FV} = MMS_F + MMS_V$。只从query→video单向交互
    - 设计动机：MMS_F覆盖空间信息后，temporal transformer可专注编码时间动态，形成功能分工

2. **查询和视觉扩展Token**:

    - 功能：增强query和video表征
    - 核心思路：query端添加padding token参与MMS，能学到隐含查询扩展。video端向temporal transformer添加可学习视觉扩展token
    - 设计动机：T2VR查询简短且抽象，query expansion推断未明确表达的概念

3. **双Sigmoid损失**:

    - 功能：独立训练帧级和视频级表征
    - 核心思路：分别对$MMS_F$和$MMS_V$的相似度矩阵计算sigmoid损失，$\mathcal{L}_D = \lambda_F \mathcal{L}_F + \lambda_V \mathcal{L}_V$。每个sigmoid损失将对比学习转为独立二分类
    - 设计动机：组合得分直接算损失会有梯度不均衡。分开计算让两路各自学到判别性，推理时相加组合

### 损失函数 / 训练策略

双sigmoid损失：每个损失$\mathcal{L} = -\frac{1}{|B|}\sum_i\sum_j \log\frac{1}{1+e^{z_{ij}(-t \cdot MMS + b)}}$，$z_{ij}$标签、$t$可学习scaling、$b$可学习bias。单向MaxSim体现query-video非对称性。

## 实验关键数据

### 主实验

| 方法 | MSR-VTT R@1 | MSVD R@1 | VATEX R@1 |
|------|------------|---------|----------|
| X-CLIP (B/16) | 49.3 | 50.4 | — |
| DRL (B/16) | 50.2 | 50.0 | 65.7 |
| **V-ColBERT (CLIP-B/16)** | **51.0** | 50.2 | 66.8 |
| **V-ColBERT (SigLIP-B/16)** | **51.5** | **55.2** | **68.0** |

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅MMS_F | 纯帧级空间匹配，~48 R@1 |
| 仅MMS_V | 仅时空特征，~49 R@1 |
| MMS_FV双层 | **~51 R@1**，两层互补 |
| InfoNCE损失 | 不如sigmoid |
| SMS(求和) | 变长query不公平 |
| 无query expansion | 失去隐含概念扩展 |

### 关键发现

- 双层交互vs单层提升2-3个R@1点，帧级空间和时空信息互补
- 双sigmoid优于InfoNCE和单sigmoid
- 单向MaxSim优于双向——视频中无关帧不应拉低得分
- SigLIP作为backbone比CLIP更好

## 亮点与洞察

- **ColBERT到视频的迁移**证明多向量后交互在视频模态有效
- **分层交互的功能分工**：MMS_F覆盖空间→temporal transformer专注时间
- **双sigmoid损失**简单巧妙——独立训练保证各自判别性，相加保证兼容性

## 局限与展望

- 存储开销比单向量方法大，每个视频存双倍特征
- 未探索只存TopK帧特征等高效近似
- 未与InternVideo2等大规模模型对比
- 查询扩展token的可解释性未探讨

## 相关工作与启发

- **vs DRL**: DRL加权tokenwise交互但只对时间建模后特征。Video-ColBERT双层交互更全面
- **vs X-CLIP**: X-CLIP多粒度交互更复杂，Video-ColBERT更简洁但更好
- **vs CLIP4Clip**: 后交互比单向量R@1提升5-8点

## 评分

- 新颖性: ⭐⭐⭐⭐ ColBERT到视频的迁移+双层交互+双sigmoid组合巧妙
- 实验充分度: ⭐⭐⭐⭐ 5个数据集、多backbone、丰富消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法简洁
- 价值: ⭐⭐⭐⭐ 提供了强大且简洁的后交互视频检索方案

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation](the_devil_is_in_the_prompts_retrieval-augmented_prompt_optimization_for_text-to-.md)
- [\[ICCV 2025\] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization](../../ICCV2025/video_generation/quantifying_and_narrowing_the_unknown_interactive_text-to-video_retrieval_via_un.md)
- [\[ACL 2025\] Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval](../../ACL2025/video_generation/q2e_query-to-event_decomposition_for_zero-shot_multilingual_text-to-video_retrie.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[CVPR 2025\] TransPixeler: Advancing Text-to-Video Generation with Transparency](transpixeler_advancing_text-to-video_generation_with_transparency.md)

<!-- RELATED:END -->
