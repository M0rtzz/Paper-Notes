---
title: >-
  [论文解读] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction
description: >-
  [CVPR 2025][LLM/NLP][自回归视觉生成] 本文深入分析LLM框架下视觉embedding空间的特性，发现视觉token间的相关性有助于实现更稳定的生成，据此提出IAR方法，通过码本重排（Codebook Rearrangement）和簇导向token预测（Cluster-Oriented Token Prediction）提升自回归视觉生成的效率和质量。
tags:
  - CVPR 2025
  - LLM/NLP
  - 自回归视觉生成
  - 码本重排
  - 簇导向预测
  - LLM
  - token空间
---

# Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 视觉自回归生成  
**关键词**: 自回归视觉生成, 码本重排, 簇导向预测, LLM, token空间

## 一句话总结

本文深入分析LLM框架下视觉embedding空间的特性，发现视觉token间的相关性有助于实现更稳定的生成，据此提出IAR方法，通过码本重排（Codebook Rearrangement）和簇导向token预测（Cluster-Oriented Token Prediction）提升自回归视觉生成的效率和质量。

## 研究背景与动机

**领域现状**：利用LLM进行视觉生成已成为研究热点，代表方法如LlamaGen、VAR等将图像token化后用自回归模型逐步生成。这类方法试图将LLM在语言生成中的成功经验迁移到视觉领域。

**现有痛点**：(1) 语言与视觉的根本差异被忽视——语言token具有离散语义，相邻token ID可能语义完全不同；但视觉token具有连续性，在embedding空间中语义相似的视觉token应该有相近的表示。现有方法直接照搬LLM架构而未考虑这一差异。(2) 码本（codebook）中token的排列是随机的——VQVAE/VQGAN等量化器训练后的码本中，ID相邻的token在特征空间中可能距离很远，导致交叉熵损失对相似token的预测误差惩罚不合理。(3) 分类头效率低——对大码本（如8192个token）使用softmax分类在训练和推理时都是计算瓶颈。

**核心矛盾**：LLM的交叉熵损失将所有非目标token等价对待，但在视觉生成中，预测一个语义相近的token（如同一纹理区域的不同量化值）和预测一个完全不相关的token的"错误程度"显然不同。

**本文目标** 如何在LLM框架下更好地利用视觉token间的相关性来提升生成质量？

**切入角度**：首先研究视觉embedding空间的结构特性，发现视觉token自然形成语义簇（cluster），由此设计簇导向的训练和预测策略。

**核心 idea**：对码本按语义相似性重排，先预测token所属簇再在簇内预测具体token，实现由粗到细的视觉token生成。

## 方法详解

### 整体框架

IAR包含三个核心组件：(1) 码本重排——对VQGAN码本中的token按embedding相似性重排序，使ID相邻的token在特征空间中也相近。(2) 簇导向token预测——将码本划分为K个语义簇，生成时先预测token所属的簇（粗分类），再在簇内预测具体token ID（细分类）。(3) 与标准LLM自回归流程结合，保持next-token prediction范式不变。

### 关键设计

1. **码本重排（Codebook Rearrangement）**：
    - 功能：使码本中ID相邻的token在embedding空间中也相近
    - 核心思路：对预训练VQGAN码本中的N个embedding向量进行聚类和排序。先用K-means将码本分为K个簇，然后对簇进行排序（如按簇中心的特征值），最后在每个簇内部按与簇中心的距离排序。重排后的码本保证了ID的连续性反映特征的相似性
    - 设计动机：标准交叉熵中softmax label smoothing等技术对相邻ID有更高容忍度，码本重排使这种"邻近容忍"具有语义意义——预测错一个位置的惩罚应比预测错一百个位置要小

2. **簇导向Token预测（Cluster-Oriented Prediction）**：
    - 功能：将大规模分类问题分解为层次化的两步预测
    - 核心思路：将码本的K个簇作为粗分类目标，训练时同时预测token所属簇和簇内精确ID。推理时先通过簇预测头确定top-k个候选簇，然后仅在这些簇对应的token子集上做精确预测。通过两个分类头实现：簇预测头（K路分类）和token预测头（N路分类，但推理时只计算候选簇内token）
    - 设计动机：直接对8192个token做softmax效率低且梯度稀疏。分层预测降低了每一步的分类空间大小，提升训练效率和预测稳定性

3. **簇感知损失函数**：
    - 功能：利用token间的语义关系提供更细粒度的监督信号
    - 核心思路：在标准交叉熵基础上引入簇级交叉熵损失和软标签机制。当预测的token与ground truth属于同一簇时给予较小惩罚，不同簇时给予较大惩罚。总损失 = token级CE + λ × 簇级CE
    - 设计动机：标准CE对所有错误等价惩罚不符合视觉语义的连续性

## 实验关键数据

### 关键发现

- 在ImageNet 256×256生成任务上，IAR显著提升FID和IS指标
- 码本重排在不改变码本内容的情况下提升FID约2-3点
- 簇导向预测在相同计算量下将收敛速度提升约1.5倍
- 方法对不同LLM骨干（GPT-2、LLaMA等）和码本大小均有效
- 生成图像在局部纹理一致性上明显优于baseline

## 亮点与洞察

- **洞察深刻**：揭示了语言和视觉token空间的根本差异——语言token的离散性vs视觉token的连续性
- **码本重排思路简洁有效**：不修改码本内容，仅重排顺序就有显著提升
- **层次预测实用**：将大码本分类分解为两步，既提升精度又降低计算量

## 局限与展望

- 簇的数量K是需要调优的超参数
- 码本重排策略对不同VQGAN模型需要重新计算
- 当前仅验证了图像生成，视频等序列生成场景待验证
- 层次化预测引入的额外参数（两个分类头）在超大码本时成本显著

<!-- RELATED:START -->

## 相关论文

- [ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)
- [Test-Time Visual In-Context Tuning](test-time_visual_in-context_tuning.md)
- [AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](../../ICLR2026/llm_nlp/assetformer_modular_3d_assets_generation_with_autoregressive_transformer.md)
- [Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)

<!-- RELATED:END -->
