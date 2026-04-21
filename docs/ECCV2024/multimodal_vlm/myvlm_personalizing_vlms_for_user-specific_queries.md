---
title: >-
  [论文解读] MyVLM: Personalizing VLMs for User-Specific Queries
description: >-
  [ECCV 2024][多模态][VLM个性化] 提出MyVLM，通过外部概念识别头（concept head）和可学习的概念嵌入向量（concept embedding），在不修改VLM原始权重的情况下实现个性化视觉语言交互——仅需3-5张图片即可让VLM识别并描述用户特定概念（如"你的狗"、"你的朋友"），在BLIP-2和LLaVA上均取得了显著的个性化效果。
tags:
  - ECCV 2024
  - 多模态
  - VLM个性化
  - 概念学习
  - 图像描述
  - 视觉问答
  - 少样本学习
---

# MyVLM: Personalizing VLMs for User-Specific Queries

**会议**: ECCV 2024  
**arXiv**: [2403.14599](https://arxiv.org/abs/2403.14599)  
**代码**: https://snap-research.github.io/MyVLM/ (有项目页面)  
**领域**: 多模态VLM  
**关键词**: VLM个性化, 概念学习, 图像描述, 视觉问答, 少样本学习

## 一句话总结
提出MyVLM，通过外部概念识别头（concept head）和可学习的概念嵌入向量（concept embedding），在不修改VLM原始权重的情况下实现个性化视觉语言交互——仅需3-5张图片即可让VLM识别并描述用户特定概念（如"你的狗"、"你的朋友"），在BLIP-2和LLaVA上均取得了显著的个性化效果。

## 研究背景与动机

大规模视觉语言模型（VLM）如BLIP-2、LLaVA已经展现出出色的图像理解和文本生成能力，但它们只具备通用知识，缺乏对用户个人概念的理解。例如，VLM能识别"一只狗"，但无法知道这是"你的狗"。

**核心痛点**：
1. 直接微调VLM计算开销大，且容易导致灾难性遗忘
2. 模型编辑（model editing）技术仅关注修改特定查询的回答，无法处理视觉多样性
3. VLM的视觉编码器特征空间不够表达性，难以区分语义相似的对象

**核心矛盾**：需要让模型学会识别新概念，同时不破坏其原有的通用视觉-语言能力。

**切入角度**：将个性化分解为两个阶段——"识别"（Recognizing）和"表达"（Communicating），通过外部模块化设计实现概念的识别和嵌入，完全不修改VLM原始权重。

## 方法详解

### 整体框架
MyVLM采用两阶段pipeline：(1) 外部概念头识别图像中是否存在目标概念；(2) 可学习的概念嵌入向量被追加到视觉特征后，引导语言模型生成包含目标概念标识符的个性化文本响应。

### 关键设计

1. **概念识别头（Concept Head）**：为每个用户概念训练一个独立的分类头，用于判断图像中是否存在该概念。对于物体使用基于CLIP嵌入的线性分类器，对于人物使用预训练人脸识别网络（RetinaFace + ArcFace）。独立的头设计使得系统可以灵活扩展到更多概念。

2. **概念嵌入向量（Concept Embedding）**：在VLM的中间特征空间中学习一个d=768维的嵌入向量 $e_*$，追加到视觉编码器输出后，通过Q-Former的交叉注意力层传递给语言模型。优化目标为标准交叉熵损失：
    $e_* = \arg\min_e \sum_{i=1}^{N} \mathcal{L}_{CE}(t_i, o(I_i, e))$
   其中 $t_i$ 为目标描述，$o(I_i, e)$ 为给定概念嵌入后生成的描述。

3. **泛化增强机制**：

    - **Key/Value归一化**：在Q-Former交叉注意力层中，概念嵌入对应的key和value向量范数远大于冻结图像特征，因此将其归一化到原始key/value的平均范数：$\hat{k}_* = \frac{k_*}{\|k_*\|} \cdot n_k$
    - **注意力正则化**：对概念token的注意力概率施加L2正则化，防止query token过度关注概念token而忽略原始图像token：$\mathcal{L}_{reg} = \|\text{softmax}(Q \cdot \hat{k}_*)\|_2^2$

### 在LLaVA上的适配
- 概念嵌入追加到线性投影层输出之后（而非视觉编码器之后）
- 由于LLaVA无交叉注意力，改为将概念嵌入的向量范数缩放到与ViT的[CLS] token一致
- 注意力正则化改为鼓励其他token（包括语言token和图像token）对概念嵌入给予较低注意力

### 损失函数 / 训练策略
- 仅优化概念嵌入向量，VLM所有权重保持冻结
- 训练数据：每个概念3-5张图片及对应描述
- 总损失：$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{reg}$
- 可扩展到VQA：定义10组问答对，每步随机采样一组进行优化

## 实验关键数据

### 主实验 — 个性化描述召回率

| VLM | 方法 | 物体 | 人物 | 总体 |
|-----|------|------|------|------|
| BLIP-2 | Simple Replacement | 29.30 | 84.33 | 59.33 |
| BLIP-2 | LLM-Guided | 51.55 | 56.91 | 54.37 |
| BLIP-2 | **MyVLM** | **95.10** | **79.76** | **87.11** |
| LLaVA | Simple Replacement | 25.86 | 18.13 | 21.68 |
| LLaVA | LLM-Guided | 65.38 | 29.11 | 46.23 |
| LLaVA | **MyVLM** | **94.76** | **97.08** | **95.97** |

### 消融实验 — 训练样本数量

| VLM | 训练样本数 | Recall↑ | Image Sim↑ | Text Sim↑ |
|-----|-----------|---------|-----------|----------|
| BLIP-2 | 1 | 75.42 | 24.20 | 57.37 |
| BLIP-2 | 2 | 84.27 | 24.91 | 61.01 |
| BLIP-2 | 4 | 87.11 | 25.42 | 62.61 |
| LLaVA | 1 | 88.93 | 23.44 | 50.39 |
| LLaVA | 4 | 95.97 | 25.24 | 56.98 |

### 关键发现
- MyVLM在BLIP-2上比LLM-Guided基线高44%召回率，在LLaVA上高30%
- 即使仅用1张训练图片，MyVLM仍大幅超越所有基线
- 与GPT-4V对比，GPT-4V容易将"文本描述相似但视觉不同"的物体误识别为目标概念，说明GPT-4V更依赖文本描述而非视觉特征
- 在LLaVA上，自注意力可视化显示概念嵌入确实关注了图像中概念所在区域

## 亮点与洞察
1. **模块化设计精妙**：概念头和概念嵌入作为可插拔模块，不修改VLM任何权重，保留原始能力
2. **任务泛化性好**：仅用描述训练的嵌入可直接迁移到VQA和目标定位（REC），证明嵌入确实捕获了概念的语义表示
3. **揭示了VLM内部机制**：发现VLM视觉编码器的特征空间不足以区分语义相似对象，以及GPT-4V过度依赖文本描述的有趣现象
4. **注意力正则化思路**：通过控制注意力分布防止概念嵌入主导所有query token，是一种优雅的正则化手段

## 局限与展望
1. 依赖VLM本身的偏见（如将一男一女默认为夫妻）
2. 概念头的识别质量直接影响最终效果
3. 训练上下文可能泄漏到推理中（如训练图中的"纽约"出现在新描述中）
4. 多人场景中识别目标个体仍有困难
5. VQA对训练时见过的问题类型表现更好，泛化到全新问题仍有提升空间

## 相关工作与启发
- 与DreamBooth等图像生成个性化方法思路相似，但目标是VLM个性化
- 与模型编辑（MEMIT等）不同，个性化需要适应新图像的视觉多样性，而非固定查询
- 启发：该框架可结合开放集目标检测进一步提升概念头的鲁棒性；注意力正则化思想可推广到其他参数高效微调场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性提出VLM个性化任务，分解为识别+表达两阶段
- 实验充分度: ⭐⭐⭐⭐ 对比了多种基线和GPT-4V，但缺少更大规模数据集的验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法动机和设计逻辑链条完整
- 价值: ⭐⭐⭐⭐⭐ 开创VLM个性化新方向，具有广泛应用前景

<!-- RELATED:START -->

## 相关论文

- [Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](metaprompting_for_automating_zeroshot_visual_recognitio.md)
- [FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_highresolution_visionlanguage_mo.md)
- [DeCUR: Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](decoupling_common_and_unique_representations_for_multimodal_.md)
- [UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)
- [Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models](groma_localized_visual_tokenization_for_grounding_multimodal.md)

<!-- RELATED:END -->
