---
title: >-
  [论文解读] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks
description: >-
  [CVPR 2026][多模态][VLM去偏] 提出一种在VLM跨模态空间中具有闭式解的去偏方法，在无需训练、无需标注数据的条件下，通过正交分解实现Pareto最优的公平性与效用权衡，同时为效用损失提供理论上界。
tags:
  - CVPR 2026
  - 多模态
  - VLM去偏
  - 公平性
  - 闭式解
  - Pareto最优
  - 跨模态对齐
---

# A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks

**会议**: CVPR 2026  
**arXiv**: [2603.12998](https://arxiv.org/abs/2603.12998)  
**代码**: [有](https://github.com/Supltz/Debias_VLM)  
**领域**: 多模态VLM  
**关键词**: VLM去偏、公平性、闭式解、Pareto最优、跨模态对齐

## 一句话总结

提出一种在VLM跨模态空间中具有闭式解的去偏方法，在无需训练、无需标注数据的条件下，通过正交分解实现Pareto最优的公平性与效用权衡，同时为效用损失提供理论上界。

## 研究背景与动机

**领域现状**：CLIP等VLM通过海量网络图文对训练，在零样本分类、图文检索、文生图等任务上取得优异表现，但不可避免地从训练数据中继承社会偏见（如"nurse"与"female"异常相似、"doctor"与"male"异常相似）。

**痛点**：现有去偏方法存在多重局限——需要额外训练辅助网络（DeAR、FairerCLIP、PromptArray）、需要敏感属性标注数据（SFID、CLIP-clip）、仅关注单一模态（SANER、BiasedPrompt只去偏文本）、仅适用于单一下游任务（PRISM只做零样本分类）、且无法保证去偏后效用不显著下降。

**核心矛盾**：公平性（fairness）与效用（utility）之间存在固有trade-off——去除敏感属性信息可能同时损害语义内容。先前方法通过正交投影到整个子空间$\mathcal{S}$，这个子空间不仅包含属性信息，还包含需要保留的语义内容（如"doctor"），导致过度去偏。

**要解决什么**：设计一个同时满足"免训练 + 免标注 + 双模态去偏 + 多任务适用 + 效用可证明有界"的统一去偏方法。

**切入角度**：从跨模态嵌入空间的几何结构出发，将去偏问题形式化为单位超球面上的优化问题，利用正交分解将高维搜索空间降维到二维。

**核心idea**：只投影到**属性子空间**$\mathcal{A}$（由组间差异方向张成），而非整个组原型子空间$\mathcal{S}$，从而在去除偏见的同时保留语义内容；进一步通过Chebyshev scalarisation求解minimax问题，得到对任意权重鲁棒的闭式最优解。

## 方法详解

### 整体框架

整个方法分为两步：(1) LLM引导的组原型构建——为每个敏感属性组构建代表性文本嵌入；(2) 闭式求解去偏嵌入——在单位超球面$\mathbb{S}^{d-1}$上找到Pareto最优的去偏向量$\vec{u}^{\star}$。方法无需任何训练步骤，无需标注数据，可直接插入任何VLM。

### 关键设计

#### 1. LLM引导的组原型构建（Group Prototype Construction）

- **功能**：为每个敏感属性组$g$构建一个鲁棒的文本原型$\vec{p}_g$。
- **核心思路**：给定输入prompt（如"a photo of a doctor"），用LLM（GPT-5）生成组特定提示（如"a photo of a male doctor"）及其多个同义变体（"a photo of a man doctor"、"a photo of a masculine doctor"等），然后取所有变体嵌入的球面均值作为组原型。
- **设计动机**：先前方法直接用单个组提示作为原型，忽略了属性表达的语言多样性。SANER虽用语料库扩充，但词集不依赖输入上下文。本方法利用LLM推理能力生成上下文对齐的变体，使原型更具代表性。

#### 2. 属性子空间构建与正交分解

- **功能**：定义属性子空间$\mathcal{A} = \text{span}\{\vec{a}_2, \ldots, \vec{a}_n\}$，其中$\vec{a}_i = \vec{p}_{g_i} - \vec{p}_{g_1}$为组间差异方向。
- **核心思路**：将原始嵌入$\vec{e}$正交分解为$\vec{e}_{\mathcal{A}_\parallel}$（属性泄露分量）和$\vec{e}_{\mathcal{A}_\perp}$（中性语义分量），去偏即为减小前者、保留后者。
- **设计动机**：区别于之前投影到整个$\mathcal{S}$子空间（会丢失语义），只投影到属性子空间$\mathcal{A}$可精准去除偏见而不损害内容。

#### 3. 闭式最优解求解

- **功能**：在$\mathbb{S}^{d-1}$上求解同时最小化属性泄露（公平性）和self-utility loss（效用）的Pareto最优嵌入。
- **核心思路**：
    - 通过Lemma 1将搜索空间从高维超球面降至$\text{span}\{\vec{e}_{\mathcal{A}_\parallel}, \vec{e}_{\mathcal{A}_\perp}\}$上的二维单位圆。
    - 用标量$\alpha$参数化解：$\vec{u} = \alpha \frac{\vec{e}_{\mathcal{A}_\parallel}}{\|\vec{e}_{\mathcal{A}_\parallel}\|} + \sqrt{1-\alpha^2} \frac{\vec{e}_{\mathcal{A}_\perp}}{\|\vec{e}_{\mathcal{A}_\perp}\|}$
    - 为避免任务特定调参，通过Chebyshev scalarisation求解minimax问题，得到对任意权重$(w_1, w_2)$鲁棒的闭式解$\alpha^{\star}$（Theorem 1）。
- **设计动机**：先前方法要么完全投影（$\alpha=0$，公平性最优但效用最差），要么不去偏（$\alpha=\|\vec{e}_{\mathcal{A}_\parallel}\|$，效用最优但公平性最差）。本方法在Pareto前沿上找到minimax最优点，无需手动调参。

### 损失函数 / 训练策略

本方法**无需训练**。优化目标为：

$$\min_{0 \leq \alpha \leq \|\vec{e}_{\mathcal{A}_\parallel}\|} \sup_{w_1+w_2=1} \{w_1 L(\alpha) + w_2 V(\alpha)\}$$

其中$L(\alpha) = \alpha$为属性泄露，$V(\alpha) = 1 - \alpha\|\vec{e}_{\mathcal{A}_\parallel}\| - \sqrt{1-\alpha^2}\|\vec{e}_{\mathcal{A}_\perp}\|$为self-utility loss。该问题有闭式解，无需迭代优化。效用理论保证：self-utility loss ≤ $1 - \|\vec{e}_{\mathcal{A}_\perp}\|$，cross-utility loss通过Proposition 1由两个模态的self-utility loss上界约束。

## 实验关键数据

### 主实验

**零样本图像分类**（CelebA + FACET，CLIP ViT-L/14）：

| 方法 | CelebA F1↑ | ΔEO_Avg (G×A)↓ | ΔEO_Max (G×A)↓ | FACET F1↑ | ΔEO_Avg (G)↓ | ΔEO_Max (G)↓ |
|------|-----------|----------------|----------------|----------|--------------|--------------|
| CLIP Baseline | 54.0 | 25.1 | 45.0 | 70.8 | 8.9 | 49.8 |
| RoboShot | 52.3 | **23.3** | **40.0** | 69.3 | 8.5 | **47.3** |
| FairerCLIP | 53.1 | 24.0 | 41.4 | 69.8 | 9.2 | 50.1 |
| **Ours** | **56.5** | 23.6 | 40.1 | **70.7** | **8.3** | 47.5 |

**文图检索**（COCO2017 + Flickr30K，CLIP ViT-L/14）：

| 方法 | COCO R@5↑ | COCO R@10↑ | MS@1000 (G×ST)↓ | Flickr R@5↑ | Flickr R@10↑ | MS@1000 (G)↓ |
|------|----------|-----------|-----------------|------------|-------------|--------------|
| CLIP Baseline | 83.8 | 90.1 | 13.4 | 91.0 | 95.4 | 20.3 |
| CLIP-clip | 76.1 | 85.2 | **9.9** | 87.7 | 91.5 | **11.7** |
| FairerCLIP | 76.8 | 85.4 | 10.2 | 87.9 | 92.5 | 12.2 |
| **Ours** | **81.1** | **89.0** | 10.1 | **90.4** | **94.9** | 11.8 |

### 消融实验

在Flickr30K和CelebA上，分别对比了组原型构建和模态去偏的消融：

| 消融条件 | MS@1000 (G)↓ | ΔEO_Max (G×A)↓ |
|---------|--------------|----------------|
| Baseline | 20.3 | 45.0 |
| 仅用锚点嵌入$\vec{p}_a$ | 13.4 | 41.1 |
| 仅用均值嵌入$\vec{p}_m$ | 14.1 | 41.8 |
| 仅去偏图像$\vec{u}_I$ | 13.4 | 41.7 |
| 仅去偏文本$\vec{u}_T$ | 13.3 | 41.1 |
| 完整方法 | **11.8** | **40.1** |

不同LLM（DeepSeek v3.2、Gemini 2.5 Pro）对结果影响极小，说明原型生成的鲁棒性。

### 关键发现

1. **效用保持突出**：在零样本分类中F1达56.5%（baseline 54.0%，反而提升），检索中R@5/R@10几乎无损（81.1 vs 83.8、90.4 vs 91.0），远优于其他方法的大幅下降。
2. **跨任务一致性**：在分类、检索、文生图三个任务上均达到最佳或次佳公平性。
3. **双模态去偏必要性**：仅去偏单一模态的公平性指标均弱于双模态联合去偏。
4. **有标注数据无系统优势**：需要标注数据的方法（如FairerCLIP、PromptArray）在跨域（face-centric→full-body）场景下泛化力差。
5. **文生图**：SP↓ 39.7 vs baseline 47.9，同时CLIP score 24.2、AccG 74.6%远优于Orth-Proj（19.7/53.4%）和Orth-Cali（20.7/56.6%）。

## 亮点与洞察

- **数学优雅**：从几何视角将VLM去偏形式化为单位超球面上的优化，通过正交分解+降维得到闭式解，避免了迭代优化和超参搜索。
- **效用可证明有界**：首次为VLM去偏方法提供self-utility loss和cross-utility loss的理论上界，而非依赖经验观察。
- **真正的plug-and-play**：无训练、无标注、双模态、多任务，可直接应用于任何VLM（验证了CLIP、BLIP等多种架构）。
- **Chebyshev scalarisation的巧用**：通过minimax避免了task-specific权重调参，使解对任意公平-效用偏好都鲁棒。
- **属性子空间 vs 组原型子空间**的区分是关键insight——先前方法投影到$\mathcal{S}$会丢失"doctor"等语义，只投影到$\mathcal{A}$精准去除性别差异。

## 局限与展望

1. **效用保证在嵌入空间而非任务指标**：理论上界约束的是cosine similarity意义下的效用，不直接保证F1、R@K等具体指标；实际上文实验表明两者高度正相关，但gap仍存在。
2. **仅限编码器侧**：方法在VLM编码器嵌入空间操作，尚未扩展到解码器（如直接对生成模型的decoder去偏）。
3. **属性组的覆盖有限**：受限于现有公平性数据集，评估涉及的敏感属性（性别、年龄、肤色）有限，未覆盖更复杂的intersectional组合。
4. **可探索方向**：将闭式解推广到decoder空间；结合更丰富的属性分类体系；探索动态权重$\alpha$（根据下游任务自适应调整）。

## 相关工作与启发

- **投影去偏系列**（Orth-Proj/Orth-Cali、PRISM-mini）：本文的"完美公平"极端情况$\alpha=0$恰好对应这些方法——说明它们牺牲了所有属性方向上的信息。
- **对抗训练系列**（DeAR、PromptArray）：需要标注数据+训练，且泛化能力受限于训练域。
- **对NLP去偏的启发**：Bolukbasi等人的词嵌入去偏思路被推广到多模态空间，但本文加入了效用理论保证。
- **对公平性ML的启发**：Chebyshev scalarisation在multi-objective optimization中常用，本文首次将其用于VLM去偏的fairness-utility平衡。

## 评分

⭐⭐⭐⭐ 理论完备且实用性强的VLM去偏方法。闭式解优雅，效用保证有理论支撑，实验全面覆盖三个下游任务+intersectional fairness。唯一不足是效用保证停留在嵌入空间层面。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)

<!-- RELATED:END -->
