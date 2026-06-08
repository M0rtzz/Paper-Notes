---
title: >-
  [论文解读] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP
description: >-
  [CVPR 2026][可解释性][稀疏自编码器] 提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS）…
tags:
  - "CVPR 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "CLIP"
  - "信息范围"
  - "上下文依赖性"
  - "outlier token"
---

# Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP

**会议**: CVPR 2026  
**arXiv**: [2604.05724](https://arxiv.org/abs/2604.05724)  
**代码**: 无  
**领域**: 模型压缩/可解释性  
**关键词**: 稀疏自编码器, CLIP, 信息范围, 上下文依赖性, outlier token

## 一句话总结
提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS），揭示两类特征在分类、分割、深度估计中的差异化功能角色。

## 研究背景与动机
**领域现状**：稀疏自编码器（SAE）已成为解释CLIP等视觉模型内部表示的核心工具，能将稠密多义（polysemantic）表示分解为稀疏单义（monosemantic）特征。

**现有痛点**：当前SAE可解释性研究几乎只关注特征的**语义身份**（"这个特征代表什么概念"），但一个标记为"dog"的特征可能是对整个物体的全局编码，也可能仅对局部纹理（如毛发）响应——后语义分析无法区分这两者。

**关键观察**：Vision Transformer中的**outlier token**（异常高范数的patch token）在微小上下文偏移（Shifted Context Cropping，SCC）下表现出极强的**空间不稳定性**——位置随上下文变化剧烈。这暗示全局信号对上下文高度敏感，而局部信号则稳定锚定于视觉内容。

**核心idea**：利用上下文偏移下的空间稳定性差异，量化每个SAE特征的"信息范围"——是聚合局部证据还是全局证据。

## 方法详解

### 整体框架
这篇论文想回答一个被以往SAE分析忽略的问题：一个被标注为"dog"的特征，到底是在编码整个物体的全局信息，还是只对局部纹理响应？语义标签答不了这个问题，于是作者换了个角度——看特征对**上下文扰动**的敏感程度。整条流程是这样转的：给定一张图，先用Shifted Context Cropping生成两个内容重叠、但上下文略有偏移的裁剪；两个裁剪分别过ViT编码、再经SAE解码，得到每个特征在两张图上的空间激活图；最后比较同一特征在两张图重叠区域内激活图的差异，差异越大说明这个特征越依赖上下文。把这个差异量化成一个分数，就是Contextual Dependency Score（CDS），它沿"局部↔全局"这条信息范围的轴，把字典里的特征一刀分开。

拿到分区之后，作者用一组下游分析来验证它的意义：冻结CLIP骨干，分别把高CDS组、低CDS组的特征置零，再用线性探针在ImageNet分类、ADE20K语义分割、NYUd深度估计三个任务上看性能怎么变——如果两组特征真的承担不同角色，移除它们对不同任务的影响就应该截然不同。

### 关键设计

**1. Shifted Context Cropping (SCC)：用同一图像的两个重叠裁剪隔离出"纯上下文"这一个变量。**

要判断一个特征是不是依赖上下文，最干净的办法是只改上下文、不改内容，看它怎么反应。SCC正是为此设计：先把图像缩放到 $(p+s)n \times (p+s)n$，再从中裁出两个 $pn \times pn$ 的窗口，彼此偏移 $sn$ 像素。这样两个裁剪会共享一块 $(p-s) \times (p-s)$ 的完全相同的patch区域——同一块视觉内容，在两张图里像素一模一样。既然内容不变，那么这块重叠区域里特征激活的任何差异，就只能来自位置编码的偏移和注意力所见上下文的变化。SCC因此把"内容差异"这个混杂因素彻底排除，留下纯粹的上下文扰动作为探针。

**2. Contextual Dependency Score (CDS)：用激活图之间的搬土距离量化特征对上下文的依赖。**

有了SCC这把探针，下一步是把"激活图变了多少"变成一个可比的数。对每个SAE特征 $f_j$，先挑出让它激活最强的 $k_{CDS}$ 张图像；对每张图做一次SCC，取出重叠区域内该特征的两张激活图 $M_{j,1}^{(m)}$ 和 $M_{j,2}^{(m)}$，归一化成概率分布后，用Earth Mover's Distance（EMD，搬土距离）衡量两者的空间错位——EMD越大，说明同一块内容上特征的响应位置被上下文推得越远。最后用网格对角线长度 $D_{grid}$ 归一化、对 $k_{CDS}$ 张图取平均：

$$CDS_j = \frac{1}{k_{CDS} \cdot D_{grid}} \sum_{m=1}^{k_{CDS}} \text{EMD}(\mathcal{N}(M_{j,1}^{(m)}), \mathcal{N}(M_{j,2}^{(m)}))$$

CDS的物理含义很直白：低CDS意味着激活在两次裁剪间几乎纹丝不动，特征稳定锚定在视觉内容上，是**局部范围**特征；高CDS意味着激活随上下文剧烈漂移，特征聚合的是全局证据，是**全局范围**特征。用EMD而非逐像素差，是因为它对"激活整体平移"这种空间错位敏感，恰好捕捉作者关心的位置不稳定性。

**3. 特征分区与outlier验证：CDS直方图天然分成两簇，且与outlier token现象对上。**

把全字典的CDS画成直方图，作者发现它并非平滑单峰，而是呈多峰分布——这本身就暗示特征在信息范围上存在自然的群聚。于是用一个阈值 $\gamma$ 把特征切成低CDS组和高CDS组。更关键的是，这条分界线能和ViT里早被观察到的outlier token现象对上：检查两组特征在outlier与non-outlier token上的激活模式，发现高CDS特征几乎只在异常高范数的outlier token上点亮，低CDS特征则主要在正常token上响应。这就给"特征对上下文的依赖"找到了一个独立的物理对应物，也反过来为outlier token提供了特征级的机理解释——它们正是全局信息的载体。

## 实验关键数据

### 主实验（特征组移除 → 线性探针性能）

| 模型 | 嵌入类型 | ImageNet Top-1↑ | ADE20K mIoU↑ | NYUd RMSE↓ |
|------|---------|----------------|-------------|-----------|
| CLIP-B/16 | 原始 | 74.82 | 25.87 | 0.8841 |
| | 移除高CDS | **75.54** | **26.02** | **0.8616** |
| | 移除低CDS | 64.86 | 11.65 | 0.9481 |
| CLIP-L/14 | 原始 | 80.82 | 26.66 | 0.8029 |
| | 移除高CDS | **81.28** | 26.44 | **0.7994** |
| | 移除低CDS | 78.30 | 13.89 | 0.8878 |

### 消融实验

| 分析 | 关键发现 | 说明 |
|------|---------|------|
| Outlier vs Non-outlier EMD | outlier EMD >> non-outlier EMD | outlier token空间极不稳定 |
| 高CDS在outlier上激活 | 83.45 vs 1.66 (B/16) | 高CDS特征选择性响应outlier |
| 低CDS在normal token上激活 | 31.51 vs 10.39 (B/16) | 低CDS特征编码局部信息 |
| 移除高CDS提升分类 | +0.72 ~ +1.42% | 全局噪声移除反而有益 |

### 关键发现
- **移除低CDS特征**对分割和深度估计的破坏性极大（mIoU从26→12），证明空间细粒度信息主要由低CDS特征携带
- **移除高CDS特征**反而略微提升分类性能，说明全局特征可能包含冗余信息
- CDS分区与outlier token现象高度吻合，提供了outlier token的**特征级机理解释**

## 亮点与洞察
- 提出信息范围作为语义之外的正交可解释性维度，弥补了现有SAE分析的盲区
- CDS指标巧妙地利用SCC隔离上下文因素，设计精妙且物理意义明确
- 建立了SAE特征→outlier token→全局/局部信息的清晰链条

## 局限与展望
- CDS计算需要对每个特征选取top-k图像并做双向推理，计算成本随字典规模增长
- 仅分析了CLIP模型，DINOv2等自监督ViT的信息范围特性待探索
- 二分法（局部vs全局）可能过于粗糙，连续谱上的更细粒度分析有价值
- 未探索CDS引导的特征选择在实际下游任务中的应用

## 相关工作与启发
- 与ViT outlier token研究（Darcet et al.）形成互补：从现象描述深入到特征级机理
- CDS思想可推广到NLP中分析LLM的SAE特征（局部token vs 全局上下文）
- 为SAE特征的"质量控制"（哪些特征值得信任）提供了量化工具

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息范围维度和CDS指标均为全新贡献，视角独特
- 实验充分度: ⭐⭐⭐⭐ 三个CLIP模型、三个下游任务，但缺少非CLIP模型验证
- 写作质量: ⭐⭐⭐⭐⭐ 从现象→假说→指标→验证，逻辑链极为清晰
- 价值: ⭐⭐⭐⭐ 为模型可解释性研究开辟新方向，但实际应用场景还需探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Sparse Autoencoders are Topic Models](../../ICML2026/interpretability/sparse_autoencoders_are_topic_models.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](../../ICML2026/interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](../../ACL2026/interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)
- [\[CVPR 2026\] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)

</div>

<!-- RELATED:END -->
