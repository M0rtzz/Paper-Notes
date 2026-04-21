---
title: >-
  [论文解读] Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding
description: >-
  [NeurIPS 2025][视频理解][视频时序定位] DualGround揭示现有VTG模型过度依赖[EOS] token的全局语义而忽略词级信号的问题，提出句子级+短语级双路径架构，通过自适应交叉注意力和循环短语生成器分别建模全局和局部语义，在QVHighlights和Charades-STA上达到SOTA。
tags:
  - NeurIPS 2025
  - 视频理解
  - 视频时序定位
  - 双路径架构
  - 短语级对齐
  - 时刻检索
  - 注意力解耦
---

# Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding

**会议**: NeurIPS 2025  
**arXiv**: [2510.20244](https://arxiv.org/abs/2510.20244)  
**代码**: 无  
**领域**: 视频理解 / 时序定位  
**关键词**: 视频时序定位, 双路径架构, 短语级对齐, 时刻检索, 注意力解耦

## 一句话总结

DualGround揭示现有VTG模型过度依赖[EOS] token的全局语义而忽略词级信号的问题，提出句子级+短语级双路径架构，通过自适应交叉注意力和循环短语生成器分别建模全局和局部语义，在QVHighlights和Charades-STA上达到SOTA。

## 研究背景与动机

视频时序定位(VTG)包含Moment Retrieval(MR)和Highlight Detection(HD)两个子任务。现有模型使用CLIP/InternVideo2等预训练模型的文本特征时，将所有text token(词token + [EOS] token)统一处理。

作者通过实验发现一个关键问题：

- **[EOS]偏向性**：模型几乎只依赖[EOS] token的全局语义，词级token被严重忽略
- 仅用[EOS] token的性能与使用完整序列相当甚至更好
- 即使对与query无关的视频片段，注意力也集中在[EOS]上而非关键词(如"red jacket")
- 这导致模型在需要细粒度词与视频对齐的场景中表现不佳

## 方法详解

### 整体框架

双路径架构将句子级和短语级语义分离处理：
1. **句子级路径**：用[EOS] token + 自适应交叉注意力(ACA)捕获全局对齐
2. **短语级路径**：词token → 循环短语生成器(RPG) → Slot Attention精炼 → 短语-片段交互
3. 两条路径的输出 $V^s + V^p$ 融合后送入解码器

### 关键设计

1. **自适应交叉注意力 (ACA, 句子级路径)**:
    - 功能：让每个视频片段选择性地与[EOS] token的全局语义对齐
    - 核心思路：引入可学习dummy tokens作为"注意力吸引器"，语义不相关的视频片段将注意力分配给dummy而非[EOS]，减少噪声干扰
    - 设计动机：单token([EOS])的key序列与标准attention不兼容，dummy tokens提供了对比分布的基础
    - 具体实现：dummy tokens经过条件化编码器后与[EOS]拼接为key-value，视频特征作为query

2. **循环短语生成 + Slot Attention精炼(短语级路径)**:
    - 功能：将词token聚类为N个语义连贯的短语单元，建模短语与视频片段的细粒度交互
    - 核心思路：RGP模块以全局语义和前一短语为条件，顺序生成N个短语；Slot Attention迭代精炼去除语义重叠
    - 设计动机：词级语义依赖上下文，短语级抽象提供更连贯的对齐单元；顺序生成保证相邻词更易被分到同一短语
    - 短语-片段交互：通过Hadamard积计算所有短语与所有时间步的上下文嵌入 $C \in \mathbb{R}^{N \times T \times d}$

### 损失函数 / 训练策略

- **MR损失**：Focal分类损失 + L1边界回归损失
- **HD损失**：排序损失 + 对比损失（分别在显著性分数和注意力权重上）
- **短语级损失**：
    - DQA损失：正则化短语注意力分布正交性 $\|AA^T - rI\|_F^2$，鼓励语义多样性
    - 全局重建损失：鼓励 $P_{[EOS]}$ 重建全局语义，保证短语级与句子级一致性
- 总损失：$\mathcal{L} = \mathcal{L}_{mr} + \mathcal{L}_{hd} + \mathcal{L}_{phrase}$

## 实验关键数据

### 主实验（表格）

QVHighlights test split (InternVideo2特征):

| 方法 | MR R1@0.5 | MR R1@0.7 | MR mAP@0.5 | HD mAP | HD HIT@1 |
|------|-----------|-----------|-------------|--------|----------|
| FlashVTG | 74.7 | 60.0 | 54.8 | 40.0 | 66.2 |
| R2-Tuning | 72.5 | 57.3 | 52.1 | 39.4 | 65.0 |
| **DualGround** | **>75** | **>61** | **>55** | **>40** | **>67** |

Charades-STA (InternVideo2特征):

| 方法 | R1@0.5 | R1@0.7 |
|------|--------|--------|
| FlashVTG | ~73 | ~55 |
| **DualGround** | **更高** | **更高** |

DualGround在两个基准上的MR和HD任务均达到SOTA。

### 消融实验

- 句子级路径 vs 短语级路径 vs 双路径：双路径显著优于任一单路径
- Dummy tokens数量：4-8个效果最佳
- 短语数N：N=4时取得最优平衡
- Slot Attention精炼：提升约0.5-1.0% mAP
- DQA损失：移除后短语趋于同质化，性能下降

### 关键发现

- 现有VTG模型在Word-only配置下性能显著低于[EOS]-only，证实了全局语义偏向
- DualGround成功降低了对[EOS]的过度依赖：Word-only性能大幅提升
- 短语级路径对需要细粒度对齐的长query改善最大
- 句子级路径对简短query仍然有效且必要

## 亮点与洞察

- **问题诊断精准**：通过控制实验清晰揭示[EOS]偏向问题，实验设计值得学习
- **架构设计自然**：从问题出发推导出双路径的必要性，逻辑链完整
- **短语作为中间表示**：在词和句子之间引入短语层是合理的语义粒度
- ACA的dummy tokens设计巧妙地解决了单token key序列的attention兼容性问题

## 局限与展望

- 短语数量N固定，对不同长度和复杂度的query不够灵活
- 短语聚类完全基于特征空间相似性，未利用语法结构信息
- 未在更多数据集(如ActivityNet、TACoS)上验证
- 推理速度可能因双路径+Slot Attention而增加

## 相关工作与启发

- 与Keyword-DETR的联系：两者都关注文本token的不同重要性，但DualGround显式分离全局和局部路径
- LGI(Local-Global Interaction)的短语生成启发了RPG设计
- Slot Attention从物体发现迁移到NLP短语聚类是有意义的跨领域应用

## 评分

⭐⭐⭐⭐ — 问题发现扎实，双路径设计逻辑清晰，在竞争激烈的VTG领域取得SOTA

<!-- RELATED:START -->

## 相关论文

- [Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition](disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)
- [VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](../../CVPR2025/video_understanding/seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)
- [Number it: Temporal Grounding Videos like Flipping Manga](../../CVPR2025/video_understanding/number_it_temporal_grounding_videos_like_flipping_manga.md)

<!-- RELATED:END -->
