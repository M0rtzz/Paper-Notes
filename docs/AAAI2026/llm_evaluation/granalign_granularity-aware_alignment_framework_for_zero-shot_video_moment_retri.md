---
title: >-
  [论文解读] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval
description: >-
  [AAAI 2026][零样本视频时刻检索] 提出一个无需训练的粒度感知对齐框架GranAlign，通过将查询重写为简化版和细化版并分别匹配无关/感知查询的视频描述，解决了零样本视频时刻检索中语义粒度不匹配的核心难题，在QVHighlights上mAP@avg提升3.23%。
tags:
  - AAAI 2026
  - 零样本视频时刻检索
  - 语义粒度对齐
  - 查询重写
  - LLM
  - 视觉语言模型
---

# GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval

**会议**: AAAI 2026  
**arXiv**: [2601.00584](https://arxiv.org/abs/2601.00584)  
**代码**: 无  
**领域**: 视频理解 / 视频时刻检索  
**关键词**: 零样本视频时刻检索, 语义粒度对齐, 查询重写, LLM, 视觉语言模型

## 一句话总结

提出一个无需训练的粒度感知对齐框架GranAlign，通过将查询重写为简化版和细化版并分别匹配无关/感知查询的视频描述，解决了零样本视频时刻检索中语义粒度不匹配的核心难题，在QVHighlights上mAP@avg提升3.23%。

## 研究背景与动机

视频时刻检索（VMR）的目标是根据自然语言查询在未剪辑视频中定位相应的时间片段。传统监督方法依赖昂贵的标注数据，因此零样本VMR（ZVMR）借助VLM和LLM预训练知识成为了极具价值的范式。

然而ZVMR面临一个根本性挑战——"粒度不匹配"（Granularity Mismatch）。具体来说，用户可能用不同抽象程度描述同一事件："一只可爱的狗" vs "一只金毛幼犬在走来走去"。粗粒度查询有高召回但低精度（覆盖面广但无法精确定位），细粒度查询有高精度但低召回（语义精确但细节稍有偏差就失配）。这形成了不可避免的trade-off。

作者通过定量分析发现，按查询类型（简单/详细/错误/其他）分类后，现有方法在不同类型上表现差异很大——证明了粒度问题的实际影响。之前的方法要么只做查询重写但停留在单一粒度层面（"one-size-fits-all"），要么只用query-agnostic的视频描述缺乏与查询的语义对齐。这种单通道推理是阻碍鲁棒检索的核心瓶颈。

本文的核心idea：抛弃单通道设计，在查询侧和视频侧同时建立粗-细双通道，并按粒度层级进行匹配对齐——简化查询配通用描述（高召回），详细查询配感知描述（高精度），二者取长补短。

## 方法详解

### 整体框架

GranAlign是一个完全无需训练的三阶段框架：（1）粒度感知对齐——将查询重写为简化版和详细版，同时生成query-agnostic和query-aware两种视频帧描述；（2）时刻提案生成——基于帧级语义相似度分数生成候选时间段；（3）后处理——通过NMS筛选最终预测结果。

### 关键设计

1. **基于粒度的查询重写（Granularity-based Query Rewriting）**:

    - 功能：将原始查询通过LLaMA-3重写为两个语义互补的版本
    - 核心思路：简化查询 $Q_s$ 通过指令替换罕见词为常见词、保留核心实体和动作、去除偶然细节——提供高泛化性以检索广泛相关的候选段；详细查询 $Q_d$ 保留细粒度表达、时序上下文和具体词汇选择——实现更精确的对齐和定位
    - 设计动机：单一粒度的重写无法同时具备高召回和高精度。使用多个人工设计的指令对（prompt pairs）来生成两种版本，实验表明框架对具体指令对的选择并不敏感

2. **查询感知描述生成（Query-Aware Captioning）**:

    - 功能：为视频帧生成两种描述——通用描述和查询感知描述
    - 核心思路：首先对所有帧生成query-agnostic描述 $C_{agn} \in \mathbb{R}^{L_v \times l}$ 作为基线。然后选取与查询相似度最高的Top-K%帧（设 $L_k$ 为候选帧数），仅对这些帧用Qwen2.5-VL生成query-aware描述 $C_{awr} \in \mathbb{R}^{L_k \times l}$。生成时以从查询中提取的实体和动作作为语义引导
    - 设计动机：对所有帧生成query-aware描述计算成本过高。这种混合策略在关键区域利用语义精确的描述，同时保持全局计算效率。但query-aware描述可能出现幻觉或过度模仿查询的语言结构，因此需要在评分阶段设计容错机制

3. **粒度时刻评分（Granular Moment Scoring）**:

    - 功能：融合两组查询-描述对的相似度分数
    - 核心思路：对每一帧 $f$ 计算综合相似度分数：$S_f = \frac{1}{2m}\sum_{i=1}^{m}[g(q_s^{(i)}, C_{agn,f}) + g(q_d^{(i)}, C_{awr,f})]$，其中 $m$ 是重写对数量，$g(\cdot, \cdot)$ 是归一化余弦相似度
    - 设计动机：简化-通用对 $(Q_s, C_{agn})$ 提供广覆盖高召回，详细-感知对 $(Q_d, C_{awr})$ 提供精确对齐但容易受幻觉影响。融合两者互补的分数消除了单一对可能带来的偏差和误报

4. **时刻提案生成与后处理（Moment Proposal Generation）**:

    - 功能：从帧级分数生成并筛选候选时间段
    - 核心思路：相邻高分帧若间隔不超过阈值 $\tau$ 则合并为同一提案，丢弃平均相似度在底部n%的提案。每个候选段的评分为 $\text{Score}(p) = (1-\lambda)\mu_p + \lambda\rho_p$，其中 $\mu_p$ 是平均语义相似度，$\rho_p$ 是归一化长度正则项，$\lambda = 0.3$。最后用NMS去除冗余提案
    - 设计动机：长度正则化可防止过长的低质量提案和过短的碎片提案

### 损失函数 / 训练策略

GranAlign是完全零样本的无训练框架——不需要任何训练数据或微调。查询重写用LLaMA3-8B，描述生成用Qwen2.5-VL-7B，初始帧过滤用CLIP ViT-B/32。推理时先离线生成query-agnostic描述，在线阶段只为关键帧生成query-aware描述，实现了比Moment-GPT更短的推理时间（6.2s vs 16.1s）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | GranAlign | Moment-GPT (前SOTA) | 提升 |
|--------|------|-----------|---------------------|------|
| QVHighlights val | R1@0.5 | 61.94 | 58.9 | +3.04 |
| QVHighlights val | mAP@avg | 39.12 | 35.9 | +3.22 |
| QVHighlights test | R1@0.5 | 59.92 | 58.3 | +1.62 |
| QVHighlights test | mAP@avg | 38.23 | 35.0 | +3.23 |
| Charades-STA | R1@0.5 | 39.6 | 38.4 | +1.2 |
| Charades-STA | mIoU | 38.0 | 36.5 | +1.5 |
| ActivityNet | R1@0.5 | 34.0 | 31.1 | +2.9 |
| ActivityNet | mIoU | 33.1 | 30.8 | +2.3 |

### 消融实验（QVHighlights val）

| 查询→ $C_{agn}$ | 查询→ $C_{awr}$ | R1@0.5 | mAP@avg |
|-----------------|-----------------|--------|---------|
| $Q_r$ (原始查询) | - | 57.94 | 31.80 |
| - | $Q_r$ | 58.19 | 32.13 |
| $Q_s$ (简化) | - | 58.97 | 37.13 |
| - | $Q_d$ (详细) | 59.48 | 37.65 |
| $Q_s$ | $Q_d$ (完整GranAlign) | **61.94** | **39.12** |

### 关键发现
- 粒度匹配很关键：将简化查询配query-aware描述（粒度不匹配）时性能下降，而匹配粒度时性能明显提升
- 双通道融合始终优于任何单通道：简化对高召回、详细对高精度，融合后两方面都好
- 在VHD（视频高光检测）任务上GranAlign的mAP达39.35%，甚至超越了完全监督的QD-DETR（39.04%），展示了零样本方法的巨大潜力
- 推理效率优异：6.2s推理时间远优于Moment-GPT的16.1s，得益于两阶段描述生成策略
- 框架对超参数选择并不敏感（重写数量 $m=3$，$\lambda=0.3$ 附近均表现稳定）

## 亮点与洞察
- "粒度不匹配"这一问题的formulation非常精准，通过查询类型分类分析（Error/Simple/Detail/Else）提供了定量证据
- 双通道粒度对齐的设计既直觉又有效——简单和详细各管一路，最后取平均，避免了复杂的融合机制
- 完全零样本且无训练成本，但在QVHighlights上已接近甚至在某些指标上超越完全监督方法，非常实用
- 两阶段描述生成（离线通用+在线关键帧感知）是一个很好工程权衡

## 局限与展望
- query-aware描述可能出现幻觉——生成了视频中不存在的视觉内容，或过度模仿查询的语言结构
- LLM查询重写可能改变原始意图，需要语义验证步骤
- 依赖多个大模型（LLaMA3 + Qwen2.5-VL + CLIP + SentenceTransformer），部署成本较高
- 对长视频中事件密集的场景，简化查询可能覆盖太多无关内容

## 相关工作与启发
- Moment-GPT是直接前驱工作，用LLaMA-3重写查询+Video-ChatGPT评分，但仍是单通道单粒度设计
- 本文的粒度感知思路可以推广到其他检索场景——如文本检索中同一问题的不同表述、图像检索中的抽象/具体查询对
- 将"生成多粒度表示再对齐"的范式与RAG或多步推理结合，可能产生更强的多模态理解系统

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zero-shot_learning.md)
- [Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)
- [RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)
- [NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)

<!-- RELATED:END -->
