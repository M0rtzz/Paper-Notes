---
title: >-
  [论文解读] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering
description: >-
  [CVPR 2025][视频理解][音视觉问答] 提出 QA-TIGER 框架，通过混合高斯专家（MoE）对视频时序进行连续自适应加权建模，并在编码早期即注入问题信息实现渐进式语义精炼，在多个 AVQA 基准上达到 SOTA。
tags:
  - CVPR 2025
  - 视频理解
  - 音视觉问答
  - 高斯专家混合
  - 时序建模
  - 问题感知注意力
  - 时序定位
---

# QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering

**会议**: CVPR 2025  
**arXiv**: [2503.04459](https://arxiv.org/abs/2503.04459)  
**代码**: [项目页面](https://aim-skku.github.io/QA-TIGER/)  
**领域**: 视频理解/多模态问答  
**关键词**: 音视觉问答, 高斯专家混合, 时序建模, 问题感知注意力, 时序定位

## 一句话总结

提出 QA-TIGER 框架，通过混合高斯专家（MoE）对视频时序进行连续自适应加权建模，并在编码早期即注入问题信息实现渐进式语义精炼，在多个 AVQA 基准上达到 SOTA。

## 研究背景与动机

音视觉问答（AVQA）需要基于问题进行多模态推理和精确的时序定位。然而现有方法存在两个关键局限：

- **问题信息利用不足**：大多数方法仅在最终推理阶段以简单乘法整合问题信息，中间编码过程缺乏问题引导，无法渐进式聚焦相关特征
- **时序采样策略受限**：均匀帧采样忽略问题特异性；Top-K 帧选择虽考虑问题相关性，但离散采样丢失了帧间连续性，且仅基于视觉线索选择，忽略了音频信息

核心挑战：如何灵活捕获问题相关的连续和非连续时序片段，同时在编码流水线中全程保持问题上下文。

## 方法详解

### 整体框架

QA-TIGER 由三个模块组成：(1) Question-Aware Fusion——在编码早期注入问题上下文到视觉和音频特征；(2) Temporal Integration of Gaussian Experts——用多个高斯分布对时间轴加权，通过 MoE 路由自适应选择专家；(3) Question-Guided Reasoning——融合时序特征进行答案预测。

### 关键设计一：问题感知融合（Question-Aware Fusion）

- **功能**：在编码流水线早期将问题上下文注入视觉和音频特征
- **核心思路**：对视觉/音频特征先做自注意力增强内部关系，再做两轮交叉注意力——先与另一模态交互，再与词级问题特征 $q_w$ 交互。$\mathbf{v}_q = \mathbf{v} + SA(\mathbf{v}) + CA(\mathbf{v}, \mathbf{a}) + CA(\mathbf{v}, q_w)$，音频特征类似处理。随后用问题感知的帧级特征 $\mathbf{v}_q, \mathbf{a}_q$ 作为 query 对 patch 级特征做交叉注意力，精炼空间细节
- **设计动机**：早期注入问题上下文让模型在整个处理过程中持续对齐问题相关线索，避免仅在最终阶段才引入问题信息的"过晚"问题

### 关键设计二：高斯专家时序整合（Temporal Integration of Gaussian Experts）

- **功能**：对时间轴进行自适应的连续加权，聚焦问题相关的时序片段
- **核心思路**：为视觉和音频模态各生成 $E$ 个高斯分布 $g_m^i = \mathcal{N}(\mu_m^i, (\sigma_m^i)^2)$，中心位置沿时间轴分布并加偏移量。通过 MoE 路由器为每个专家分配权重 $r_m^i$，最终时序特征为所有专家加权求和：$\mathcal{G}_m(x) = \sum_{i=1}^{E} g_m^i \cdot r_m^i \cdot \mathcal{E}_m^i(x)$，其中 $\mathcal{E}_m^i$ 为每个专家的 MLP
- **设计动机**：均匀采样和 Top-K 选择都是离散的，丢失时序连续性。多高斯分布提供软掩码，可同时覆盖连续片段和分散的关键帧，且视觉和音频有独立的时序定位

### 关键设计三：问题引导推理（Question-Guided Reasoning）

- **功能**：融合时序整合后的视觉和音频特征进行答案预测
- **核心思路**：先用句子级问题特征 $q_s$ 对两种视觉时序特征 $\tilde{v}_{p_v}, \tilde{v}_{p_a}$ 做 CA 融合得到 $F_v$，再将 $F_v$ 与音频时序特征 $\tilde{a}$ 融合得到最终表示 $F_{va}$，用线性层+softmax 预测答案
- **设计动机**：层次化融合（先视觉内融合，再音视觉融合）配合残差连接，防止过度依赖单一模态

### 损失函数

标准交叉熵损失：$\mathcal{L}_{qa} = -\sum_{c=1}^{C} y_c \log \mathcal{P}_c$。

## 实验关键数据

### 主实验：MUSIC-AVQA 数据集

| 方法 | Audio | Visual | A-V | Overall Acc. |
|------|-------|--------|-----|-------------|
| AVST | 65.4 | 63.7 | 60.5 | 62.3 |
| APL | 72.3 | 71.9 | 70.1 | 70.9 |
| TSPM | 71.8 | 74.2 | 71.4 | 72.2 |
| **QA-TIGER** | **73.5** | **76.1** | **73.8** | **74.3** |

### MUSIC-AVQA-v2.0 (去偏)

| 方法 | Balanced Acc. |
|------|-------------|
| AVST | 48.3 |
| COCA | 52.1 |
| **QA-TIGER** | **56.2** |

### 消融实验

| 组件 | Overall Acc. |
|------|-------------|
| 基线 (均匀采样 + 晚期问题融合) | ~70 |
| + Question-Aware Fusion | +2.1 |
| + Gaussian Experts | +1.8 |
| + 两者结合 | **74.3** |

### 关键发现

- 在 MUSIC-AVQA、MUSIC-AVQA-R、MUSIC-AVQA-v2.0 三个数据集上均达到 SOTA
- 早期问题注入和高斯专家各自贡献 ~2% 提升，组合后协同效果更强
- 视觉和音频独立的时序定位比强制对齐效果更好
- 在去偏数据集（v2.0 balanced）上的优势更明显，说明方法不依赖数据集偏差

## 亮点与洞察

1. **高斯分布的连续时序建模**：相比离散 Top-K 选帧，高斯软掩码天然适合建模时序连续性
2. **问题早注入的价值**：让整个编码流水线都在问题引导下工作，信息利用更充分
3. **音视觉独立的时序定位**：视觉和音频的关键时刻可能不完全对齐（如声音先于动作），独立建模更灵活

## 局限与展望

- 高斯专家数量 $E$ 为超参数，不同视频复杂度可能需要不同的专家数
- 依赖预训练特征提取器（CLIP, VGGish），受底层模型能力限制
- 未考虑长视频（>1 分钟）场景，高斯覆盖范围可能不足
- 可探索与 LLM-based 推理方法的结合

## 相关工作与启发

- **PSTP, TSPM**：Top-K 帧选择方法，仅基于视觉线索
- **COCA**：因果图建模多模态协作
- **APL**：自适应正例学习对齐问题-对象语义
- 高斯专家的思路可推广到其他需要时序定位的多模态推理任务

## 评分

⭐⭐⭐⭐ — 高斯专家的时序建模设计新颖且有效，问题早注入策略简单但贡献显著。在三个数据集上一致的 SOTA 验证了方法的鲁棒性。

<!-- RELATED:START -->

## 相关论文

- [EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [Bimba: Selective-Scan Compression for Long-Range Video Question Answering](bimba_selective-scan_compression_for_long-range_video_question_answering.md)
- [Cross-modal Causal Relation Alignment for Video Question Grounding](cross-modal_causal_relation_alignment_for_video_question_grounding.md)
- [Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](../../NeurIPS2025/video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)
- [EgoPointVQA: Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/egopointvqa_gesture_based_egocentric_video_qa.md)

<!-- RELATED:END -->
