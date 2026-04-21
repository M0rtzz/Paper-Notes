---
title: >-
  [论文解读] Joint Vision-Language Social Bias Removal for CLIP
description: >-
  [CVPR 2025][CLIP去偏] 本文揭示了CLIP模型中图像和文本偏见分布不一致导致的"过度去偏"问题，提出一种双模态偏见对齐+反事实去偏的联合框架，在有效减少性别/年龄/种族偏见的同时保持视觉-语言对齐能力，并设计了ABLE指标综合评估去偏效果与下游性能。
tags:
  - CVPR 2025
  - CLIP去偏
  - 社会偏见消除
  - 视觉语言对齐
  - 反事实去偏
  - 公平性
---

# Joint Vision-Language Social Bias Removal for CLIP

**会议**: CVPR 2025  
**arXiv**: [2411.12785](https://arxiv.org/abs/2411.12785)  
**代码**: https://github.com/ (有)  
**领域**: 多模态VLM  
**关键词**: 社会偏见消除, CLIP去偏, 视觉-语言对齐, 公平性, 反事实去偏

## 一句话总结

本文揭示了现有CLIP去偏方法因图文偏差分布不一致导致的"过度去偏"问题，提出先对齐图文偏差再联合移除的双模态去偏框架，在多个骨干网络上显著提升ABLE综合指标，实现了偏差消除与V-L对齐能力的良好平衡。

## 研究背景与动机

CLIP等视觉-语言预训练模型在下游任务中表现优异，但因训练数据中的社会刻板印象，模型嵌入中包含性别、年龄、种族等社会偏见 → 现有去偏方法（如Biased-prompts投影法、CLIP-clip互信息裁剪法）虽能降低偏差水平，但严重损害了V-L对齐能力，导致下游任务性能大幅下降，这一现象被作者称为"过度去偏"（over-debiasing） → 关键矛盾在于：图像和文本中的社会偏差分布是**不对齐**的（如性别-职业偏差在图像中显著，而性别-科学偏差在文本中显著），仅去偏单模态或对两模态做相同处理都会破坏V-L对齐 → 本文切入角度：先将两模态的偏差分布对齐，再联合移除 → 核心idea：bias alignment then removal。

## 方法详解

### 整体框架

冻结CLIP原始编码器，在其后接一个可训练的偏差对齐模块（Bias Alignment Module, BA），将图文嵌入分解为偏差分量和中性分量。通过KL散度对齐两模态偏差分布，并通过反事实去偏损失拉近去偏后嵌入的V-L对齐。推理时BA模块作为plug-and-play使用：$\bar\phi(t) = f(t) - \mathrm{BA}(f(t); \theta_{ba})$。

### 关键设计

1. **偏差对齐模块（Dual-Bias Alignment）**:
    - 功能：将图像和文本嵌入中的偏差信息映射到对齐的分布空间
    - 核心思路：维护moving queue（类似MoCo/ALBEF），为每个偏差嵌入构造与队列的相似度伪分布 $p(t_i)$ 和 $p(v_i)$，然后最小化KL散度 $\mathcal{L}_{ba} = \frac{1}{N}\sum_{i=1}^N D_{KL}(p(t_i) \| p(v_i))$ 实现对齐
    - 设计动机：直接用MSE或cosine对齐偏差嵌入会丢失背景信息和特征多样性；通过队列建立全局视角的伪分布再对齐，既能捕获偏差方向又不损失信息

2. **反事实去偏（Counterfactual Debiasing）**:
    - 功能：消除文本中不同属性之间的嵌入差距，同时保持V-L对齐
    - 核心思路：对每个文本 $t_i$ 构造反事实文本 $t_i'$（替换属性关键词，如male→female），用交叉熵损失使去偏后的文本嵌入与原始嵌入保持相同的text-to-image相似度分布：$\mathcal{L}_{cd}^t = -\frac{1}{N}\sum_{i}\sum_{v} s_t(t_i,v) \log \bar{s}_t(a(t_i,t_i'),v)$
    - 设计动机：直接拉近反事实对的嵌入会丢失V-L信息；以原始相似度为soft target做蒸馏，既去偏又保对齐

3. **ABLE综合评估指标**:
    - 功能：同时量化去偏效果和V-L对齐能力
    - 核心思路：ABLE = $\frac{2}{\frac{1}{acc} + \frac{1}{\exp(-\text{MaxSkew}@k)}}$，即ImageNet准确率与MaxSkew负指数的调和平均
    - 设计动机：现有评估要么只看公平性要么只看性能，ABLE仿照F1-score思路做综合评价

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \alpha \mathcal{L}_{cd} + (1-\alpha) \mathcal{L}_{ba}$，$\alpha \in [0,1]$ 为超参数。训练时CLIP编码器完全冻结，仅训练BA模块参数 $\theta_{ba}$。训练数据使用FairFace/UTKFace（带性别、年龄、种族标签的人脸图像）。

## 实验关键数据

### 主实验（ViT-B/16, FairFace训练, Gender去偏）

| 数据集/指标 | 指标 | 本文 | CLIP-clip | 原始CLIP | 说明 |
|---|---|---|---|---|---|
| FairFace | MaxSkew↓ | **0.080** | 0.103 | 0.218 | 偏差降低最多 |
| UTKFace (OOD) | MaxSkew↓ | **0.040** | 0.083 | 0.114 | 域外泛化也最优 |
| ImageNet-1K | Top-1 Acc(%)↑ | 68.05 | 68.00 | 68.31 | V-L性能几乎无损 |
| Flickr | TR R@5(%)↑ | **96.6** | 95.4 | 96.4 | 甚至优于原始CLIP |
| 综合 | ABLE(%)↑ | **78.35** | 77.55 | 73.87 | 综合指标最优 |

### 消融实验（ViT-B/16, FairFace, Gender）

| 配置 | MaxSkew↓ | ABLE(%)↑ | 说明 |
|---|---|---|---|
| 完整方法 | 0.080 | 78.35 | 偏差与性能的最佳平衡 |
| w/o $\mathcal{L}_{cd}$ | 0.167 | 75.58 | 去偏效果明显下降 |
| w/o $\mathcal{L}_{ba}$ | 0.095 | 77.71 | ABLE略低于完整 |
| 原始CLIP | 0.218 | 73.87 | 基线 |

### 关键发现

- 图文偏差在方向和强度上都显著不同（如Gender-Career在图像中显著，Gender-Science在文本中显著），CLIP-clip假设两模态偏差维度相同是不成立的
- 偏差对齐损失 $\mathcal{L}_{ba}$ 对age类偏差消除贡献更大（ABLE从56.14提升到60.61），而反事实损失 $\mathcal{L}_{cd}$ 对gender去偏效果更显著
- 方法在ViT-B/16到ViT-H/14四种骨干上均一致有效，且支持多类偏差联合去除

## 亮点与洞察

- 对"过度去偏"问题的诊断非常精准：通过SEAT/IEAT定量证实了图文偏差的不对齐性
- 基于moving queue的分布对齐思路优雅避免了直接元素级对齐的信息损失
- ABLE指标设计简洁实用，填补了去偏方法综合评价的空缺
- 推理阶段BA模块作为plug-and-play模块，不改变CLIP本身结构

## 局限与展望

- 反事实图像不可用（生成模型成本高且不可靠），图像端去偏仅通过 $\mathcal{L}_{cd}^v$ 间接实现
- 偏差类型（gender/age/race）需要预定义属性关键词和标注数据
- 目前偏差检测和消除基于人脸数据集，对非人脸场景的偏差（如地域、文化偏见）未涉及
- moving queue大小 $M$ 对结果的影响未充分探讨

## 相关工作与启发

- **Hard-Debiasing / SentenceDebias**: 早期词/句嵌入去偏方法，启发了投影子空间思路
- **CLIP-clip**: 通过互信息定位偏差维度的去偏方法，但假设图文偏差维度相同是其缺陷
- **MoCo/ALBEF**: 提供了moving queue机制的灵感
- 本文对多模态公平性问题提供了"先对齐再移除"的新思路，可推广到其他V-L模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 对过度去偏问题的剖析有洞察力，分布对齐+反事实的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4个骨干网络×3种偏差类型×3个数据集，包含域内域外评估和消融
- 写作质量: ⭐⭐⭐⭐ 论证逻辑清晰，从问题诊断到解决方案的推进自然流畅
- 价值: ⭐⭐⭐⭐ 填补了多模态公平性中偏差对齐问题的空白，ABLE指标有通用价值
# Joint Vision-Language Social Bias Removal for CLIP

**会议**: CVPR 2025  
**arXiv**: [2411.12785](https://arxiv.org/abs/2411.12785)  
**代码**: https://github.com/ (有，论文中提及)  
**领域**: 多模态VLM  
**关键词**: CLIP去偏、社会偏见消除、视觉语言对齐、反事实去偏、公平性

## 一句话总结
本文揭示了CLIP模型中图像和文本偏见分布不一致导致的"过度去偏"问题，提出一种双模态偏见对齐+反事实去偏的联合框架，在有效减少性别/年龄/种族偏见的同时保持视觉-语言对齐能力，并设计了ABLE指标综合评估去偏效果与下游性能。

## 研究背景与动机
CLIP等视觉-语言预训练模型在分类、检索等下游任务上表现优异，但从web数据中继承了严重的社会偏见（如将"职业"与特定性别关联）。现有去偏方法主要从单一模态的embedding中移除偏见信息，但这带来一个核心矛盾：**去偏后V-L对齐能力大幅下降**，即所谓的"过度去偏"（over-debiasing）问题。

作者进一步探究发现：(1) 社会偏见同时存在于图像和文本两个模态中；(2) 两个模态中的偏见分布差异很大（如gender-career偏见在图像中显著、gender-science偏见在文本中显著）。因此，像CLIP-clip那样假设两个模态偏见相同并用相同维度去偏是不合理的。

**核心idea**: 先对齐两个模态的偏见分布，再联合移除偏见，同时通过反事实目标保持V-L对齐能力。

## 方法详解

### 整体框架
冻结原始CLIP编码器，在其后接一个可学习的偏见对齐模块 $\mathrm{BA}(\cdot;\theta_{ba})$。训练数据为带属性标签的人脸图像-文本对（如FairFace）。训练时通过偏见对齐损失 $\mathcal{L}_{ba}$ 和反事实去偏损失 $\mathcal{L}_{cd}$ 联合优化；推理时通过 $\bar{\phi}(t) = f(t) - \mathrm{BA}(f(t))$ 得到去偏embedding。

### 关键设计
1. **偏见信息解耦**:
    - 功能：将CLIP embedding分解为偏见分量和中性分量
    - 核心思路：$f(t) = \phi(t) + \bar{\phi}(t)$，其中 $\phi(t)$ 为偏见信息，$\bar{\phi}(t)$ 为中性信息。BA模块输出 $\phi(t)$，减去即得去偏embedding
    - 设计动机：社会偏见作为可加性分量嵌入embedding中，可通过学习并减去的方式消除

2. **双模态偏见对齐（Dual-Bias Alignment）**:
    - 功能：在去偏前先将图像和文本的偏见分布对齐
    - 核心思路：维护图像和文本embedding队列 $\mathcal{Q}_v, \mathcal{Q}_t$（类似MoCo），计算偏见embedding与队列的相似度伪分布 $p(t_i), p(v_i)$，通过KL散度损失 $\mathcal{L}_{ba} = \frac{1}{N}\sum D_{KL}(p(t_i) \| p(v_i))$ 对齐两个分布
    - 设计动机：直接element-wise匹配会丢失背景信息和特征多样性，通过分布层面的对齐更灵活且保留信息

3. **反事实去偏（Counterfactual Debiasing）**:
    - 功能：拉近同一中性概念不同属性的去偏embedding，同时保持V-L对齐
    - 核心思路：对文本构造反事实对（如"male dancer"↔"female dancer"），用交叉熵损失拉近去偏后的相似度分布与原始分布：$\mathcal{L}_{cd}^t = -\frac{1}{N}\sum\sum s_t(t_i,v,\mathcal{V}_q)\log\bar{s}_t(a(t_i,t'_i),v,\mathcal{V}_q)$，其中 $a(t_i,t'_i)$ 以50%概率随机选择文本或其反事实版本
    - 设计动机：仅对齐偏见不够，还需确保去偏后保持原始的V-L对齐能力，避免下游任务性能退化

### 损失函数 / 训练策略
总损失为：$\mathcal{L} = \alpha \mathcal{L}_{cd} + (1-\alpha)\mathcal{L}_{ba}$，其中 $\alpha \in [0,1]$ 平衡两个目标。CLIP编码器始终冻结，仅训练BA模块参数 $\theta_{ba}$。推理时BA模块作为即插即用组件。

## 实验关键数据

### 主实验（ViT-B/16，FairFace训练）

| 设置 | 方法 | MaxSkew↓(域内) | NDKL↓(域内) | IN1K Top1↑ | Flickr TR↑ | ABLE↑ |
|------|------|-------------|------------|-----------|-----------|-------|
| Gender | Original CLIP | 0.218 | 0.088 | 68.31 | 96.4 | 73.87 |
| Gender | CLIP-clip | 0.103 | 0.026 | 68.00 | 95.4 | 77.55 |
| Gender | Biased-prompts | 0.161 | 0.048 | 65.07 | 94.3 | 73.78 |
| Gender | **Ours** | **0.080** | **0.025** | 68.05 | **96.6** | **78.35** |
| Age | Original CLIP | 0.657 | 0.433 | 68.31 | 96.4 | 58.94 |
| Age | **Ours** | **0.608** | **0.294** | **68.34** | 96.0 | **60.61** |

### 消融实验（ViT-B/16, Gender, FairFace）

| 配置 | MaxSkew↓ | NDKL↓ | IN1K Top1↑ | ABLE↑ | 说明 |
|------|---------|-------|-----------|-------|------|
| Ours (complete) | 0.080 | 0.025 | 68.05 | 78.35 | 完整方法 |
| w/o $\mathcal{L}_{cd}$ | 0.167 | 0.056 | 68.28 | 75.58 | 去掉反事实损失，偏见增加 |
| w/o $\mathcal{L}_{ba}$ | 0.095 | 0.033 | 67.84 | 77.71 | 去掉对齐损失，性能略降 |

### 关键发现
- 两个损失均不可或缺：$\mathcal{L}_{cd}$ 对减少偏见贡献更大，$\mathcal{L}_{ba}$ 对保持V-L对齐更关键
- 方法在4种ViT backbone（B/16, B/32, L/14, H/14）上均一致有效
- 域外泛化性强：在FairFace上训练，UTKFace和FACET上也能有效去偏
- 可同时去除多种偏见（性别+年龄+种族），更适合实际部署

## 亮点与洞察
- **问题发现有价值**：证明了V-L模型中偏见在两个模态的分布不同，直接解释了现有方法的失败原因
- **ABLE指标设计巧妙**：用调和平均数综合评估去偏程度和下游性能，解决了以往只看一面的问题
- **方法简洁高效**：仅需训练一个轻量BA模块，CLIP完全冻结，可即插即用

## 局限与展望
- 依赖带属性标签的人脸数据集（FairFace/UTKFace）训练
- 图像侧无法构造反事实样本（生成模型质量不够），只能用单向图像去偏损失
- 仅在检索和分类任务上评估，未验证对文生图等生成任务的影响
- 偏见类型受限于训练数据的标注类别

## 相关工作与启发
- 与CLIP-clip（基于互信息维度裁剪）和Biased-prompts（基于投影矩阵）形成互补
- 分布对齐思路借鉴了MoCo的动量队列机制，应用到偏见对齐场景
- 反事实去偏思路可推广到其他V-L模型（如BLIP系列）

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题分析深入，双模态偏见对齐思路新颖，但基本框架（对齐+去偏）并不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 4种backbone、3种偏见类型、域内域外评估、消融实验齐全
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从问题分析到方法设计环环相扣
- 价值: ⭐⭐⭐⭐ 为V-L模型公平性研究提供了新的分析视角和实用方法

<!-- RELATED:START -->

## 相关论文

- [HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](../../ICLR2026/information_retrieval/efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](../../ACL2025/information_retrieval/llm_reranking_harmful_content.md)
- [ViLU: Learning Vision-Language Uncertainties for Failure Prediction](../../ICCV2025/information_retrieval/vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](../../ACL2025/information_retrieval/evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)

<!-- RELATED:END -->
