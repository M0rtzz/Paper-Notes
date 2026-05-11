---
title: >-
  [论文解读] DualGround: Structured Phrase and Sentence-Level Temporal Grounding
description: >-
  [NeurIPS 2025][视频理解][视频时间定位] 本文发现现有视频时间定位模型过度依赖 [EOS] token 的全局句子语义而忽略词级信号，提出 DualGround 双分支架构，通过句子级路径（自适应交叉注意力）和短语级路径（循环短语生成+Slot Attention）显式分离全局和局部语义，在 QVHighlights 和 Charades-STA 上实现 SOTA。
tags:
  - NeurIPS 2025
  - 视频理解
  - 视频时间定位
  - 短语级语义
  - 双分支架构
  - 注意力解耦
  - 高亮检测
---

# DualGround: Structured Phrase and Sentence-Level Temporal Grounding

**会议**: NeurIPS 2025  
**arXiv**: [2510.20244](https://arxiv.org/abs/2510.20244)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频时间定位, 短语级语义, 双分支架构, 注意力解耦, 高亮检测

## 一句话总结
本文发现现有视频时间定位模型过度依赖 [EOS] token 的全局句子语义而忽略词级信号，提出 DualGround 双分支架构，通过句子级路径（自适应交叉注意力）和短语级路径（循环短语生成+Slot Attention）显式分离全局和局部语义，在 QVHighlights 和 Charades-STA 上实现 SOTA。

## 研究背景与动机

1. **领域现状**: 视频时间定位（VTG）借助 CLIP/InternVideo2 等预训练 VLM 取得显著进展，但模型在跨模态注意力中对所有文本 token 一视同仁。

2. **现有痛点**: 实验揭示 VTG 模型高度偏向 [EOS] token 的全局语义——仅使用 [EOS] 的性能与使用全部 token 相当甚至更好，说明词级线索被严重低效利用。

3. **核心矛盾**: [EOS] token 是独立于视觉上下文的句子摘要，可能无法反映视觉显著的文本线索（如"red jacket"），导致细粒度定位不足。

4. **本文目标**: 设计显式利用词级/短语级语义的架构，平衡全局和局部对齐。

5. **切入角度**: 将文本表征分为句子级（[EOS]）和短语级（词聚类），分别建模与视频的对齐。

6. **核心 idea**: 双路径架构——句子路径用 dummy tokens 增强 [EOS] 注意力，短语路径将词聚类为语义短语并计算短语-片段上下文。

## 方法详解

### 整体框架
输入视频特征 $V \in \mathbb{R}^{T \times d}$ 和文本特征（含 [EOS] 和词 tokens），分别通过句子级路径和短语级路径处理，融合后进行时间金字塔预测。

### 关键设计

1. **句子级路径（ACA + Dummy Tokens）**:
    - 功能: 强化 [EOS] 与视频片段的全局对齐
    - 核心思路: 添加 $L_d$ 个可学习 dummy tokens 作为注意力吸收器。语义无关的视频片段关注 dummy tokens，相关片段关注 [EOS]，产生更尖锐的对齐信号 $\alpha_i$
    - 设计动机: 单 token 的 [EOS] 与标准注意力机制不兼容，dummy tokens 提供必要的对比背景

2. **短语级路径（RPG + Slot Attention）**:
    - 功能: 从词 tokens 构建语义连贯的短语表示
    - 核心思路: 循环短语生成器（RPG）以 [EOS] 和上一短语为引导向量，对词 tokens 做软注意力聚合。Slot Attention 迭代精炼短语，解耦重叠语义。最后计算短语-片段上下文嵌入 $C = f_{ctx}(f_p(P) \odot f_v(V)) \in \mathbb{R}^{N \times T \times d}$
    - 设计动机: 词级语义依赖上下文，短语级抽象比单词更适合与视觉内容对齐

3. **短语引导聚合**:
    - 功能: 将 N 个短语-片段上下文聚合为统一表示
    - 核心思路: 用重建的全局 token $P_{[EOS]}$ 对短语计算重要性权重，按权聚合：$v_{p,t} = \sum_{n=1}^N \text{softmax}(\langle W_q P_{[EOS]}, W_k p^{(n)} \rangle / \sqrt{d}) \cdot C_{n,t}$
    - 设计动机: 不同短语对不同时间片段的重要性不同，需要自适应聚合

### 损失函数 / 训练策略
- MR 损失: Focal loss（分类）+ L1（回归）
- HD 损失: 排序损失 + 对比损失（分别对显著性分数和注意力权重）
- 短语损失: DQA 损失（正交化短语注意力）+ EOS 重建损失（InfoNCE 对齐 $P_{[EOS]}$ 和 $e_{[EOS]}$）
- 总损失: $\mathcal{L} = \lambda_{mr}\mathcal{L}_{mr} + \lambda_{hd}\mathcal{L}_{hd} + \lambda_{phrase}(\mathcal{L}_{DQA} + \mathcal{L}_{EOS})$

## 实验关键数据

### 主实验

| 数据集 | 指标 | DualGround | FlashVTG | 提升 |
|--------|------|-----------|----------|------|
| QVHighlights Test (IV2) | R1@0.7 | **56.94%** | 53.96% | +2.98% |
| QVHighlights Test (IV2) | mAP | **52.73%** | 52.00% | +0.73% |
| QVHighlights Val (IV2) | R1@0.7 | **58.97%** | 56.06% | +2.91% |
| Charades-STA (IV2) | R1@0.7 | SOTA | - | - |

### 消融实验

| 配置 | R1@0.7 | mAP | 说明 |
|------|--------|-----|------|
| FlashVTG baseline | 56.13 | 52.24 | 全 token 序列 |
| + RPG + Slot + DQA + EOS | **58.97** | **53.26** | 完整 DualGround |
| - DQA loss | 56.55 | 52.53 | 短语不够分离 |
| - EOS reconstruction | 58.02 | 53.11 | 全局一致性略降 |
| - Slot Attention | 57.83 | 53.02 | 精炼不足 |

### 关键发现
- 长查询（>20 词）性能优势更显著，短语路径帮助处理复杂查询
- 注意力可视化清晰展示基线方法忽略"red jacket"等视觉显著词
- HD 对短语路径敏感——未正则化的短语可能注入噪声
- DQA损失贡献+2.42 R1@0.7，EOS重建损失贡献+0.95 R1@0.7，Slot Attention贡献+1.14 R1@0.7
- 模型假设固定短语数N，不同查询最优N可能不同；推理时双路径增加计算开销
- 与FlashVTG baseline（InternVideo2特征）对比：R1@0.7从56.13%提升至58.97%（+2.84），mAP从52.24%提升至53.26%（+1.02）
- 控制实验揭示了VTG模型的[EOS]偏差：仅使用[EOS] token的性能与全部token相当甚至更好，说明词级线索被严重低效利用，这一发现是DualGround双路径设计的直接动机

## 亮点与洞察
- 通过控制实验（仅词/仅EOS/全部）清晰揭示了 VTG 模型的 [EOS] 偏差
- Slot Attention 用于精炼短语聚类是巧妙的设计迁移
- 双路径的句子+短语设计可推广到其他视觉-语言对齐任务
- DQA 损失和 EOS 重建损失提供了有效的正则化
- 使用 Focal loss（分类）+ L1（回归）用于 MR，排序损失+对比损失用于 HD。总损失 $\mathcal{L} = \lambda_{mr}\mathcal{L}_{mr} + \lambda_{hd}\mathcal{L}_{hd} + \lambda_{phrase}(\mathcal{L}_{DQA} + \mathcal{L}_{EOS})$

## 局限与展望
- 短语数量 N 需要预设，不同查询的最优 N 可能不同
- 推理时双路径增加了计算开销
- 未与 MLLM-based 方法（如 TimeZero）对比
- 仅验证 MR+HD 任务，未扩展到视频 QA 等更广泛任务

## 相关工作与启发
- **vs FlashVTG**: FlashVTG 用多尺度时间金字塔但统一处理文本，DualGround 分离句子/短语语义
- **vs CG-DETR**: CG-DETR 提出 ACA 但仅用于句子级，DualGround 扩展到完整双路径
- **vs Keyword-DETR**: 强调视觉显著关键词但未做短语聚类，DualGround 通过 Slot Attention 实现了语义连贯的短语抽象

## 评分

### 实现细节
基于FlashVTG框架，使用InternVideo2/CLIP特征。$L_d$个可学习dummy tokens作为注意力吸收器。
- 新颖性: ⭐⭐⭐⭐ 句子/短语双路径解耦是有效的设计
- 实验充分度: ⭐⭐⭐⭐ 多基准对比、详细消融、可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机分析扎实，控制实验有说服力
- 价值: ⭐⭐⭐⭐ 对 VTG 和视觉-语言对齐领域有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding](dualground_phrase_temporal.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)

</div>

<!-- RELATED:END -->
