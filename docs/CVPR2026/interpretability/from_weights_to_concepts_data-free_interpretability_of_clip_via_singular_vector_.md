---
title: >-
  [论文解读] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition
description: >-
  [CVPR 2026][CLIP可解释性] 本文提出 SITH（Semantic Inspection of Transformer Heads），一个完全无需数据和训练的 CLIP 可解释性框架：直接对注意力头的 Value-Output 权重矩阵做 SVD 分解，然后用自研的 COMP 算法将每个奇异向量解释为语义一致的概念稀疏组合，实现了比现有方法更细粒度的 intra-head 级别可解释性，并支持精准的权重编辑来改善下游性能。
tags:
  - CVPR 2026
  - CLIP可解释性
  - 奇异值分解
  - 注意力头分析
  - 权重空间编辑
  - 数据无关
---

# From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition

**会议**: CVPR 2026  
**arXiv**: [2603.24653](https://arxiv.org/abs/2603.24653)  
**代码**: https://frangente.github.io/SITH  
**领域**: 多模态VLM / 模型可解释性  
**关键词**: CLIP可解释性, 奇异值分解, 注意力头分析, 权重空间编辑, 数据无关

## 一句话总结
本文提出 SITH（Semantic Inspection of Transformer Heads），一个完全无需数据和训练的 CLIP 可解释性框架：直接对注意力头的 Value-Output 权重矩阵做 SVD 分解，然后用自研的 COMP 算法将每个奇异向量解释为语义一致的概念稀疏组合，实现了比现有方法更细粒度的 intra-head 级别可解释性，并支持精准的权重编辑来改善下游性能。

## 研究背景与动机

1. **领域现状**：CLIP 等视觉-语言模型（VLM）已被广泛用于各种下游任务。机制可解释性（mechanistic interpretability）试图理解这些模型内部如何表示和处理概念。现有方法主要分为两类：（1）基于激活的方法（如 Sparse Autoencoders）依赖数据集计算激活来分析；（2）TextSpan 将注意力头输出激活与文本概念对齐，但只能给出粗粒度的 head 级别解释。

2. **现有痛点**：（1）基于激活的方法依赖大规模数据集，解释结果会被数据偏差影响；（2）SAE 存在严重的不稳定性，不同数据训练出完全不同的字典；（3）TextSpan 只能解释到"这个 head 关注颜色"的粗粒度，无法区分 head 内部哪些子结构编码了红色、哪些编码了绿色；（4）尚无方法可以在不看数据的情况下直接从权重理解 CLIP 的内部机制。

3. **核心矛盾**：现有可解释性方法要么需要数据（受数据偏差影响），要么只能给出粗粒度解释（head 级别）——缺乏一个既 data-free 又 fine-grained 的分析框架。

4. **本文目标**（1）能否不看任何数据，直接从权重理解 CLIP 注意力头的功能？（2）这种理解能否达到 head 内部的单个特征级别？（3）理解后能否做精准的模型编辑？

5. **切入角度**：基于 Elhage et al. 的洞察——注意力头的计算可以表示为输入 patch 经过 Value-Output（VO）矩阵变换后的加权组合。分析 VO 矩阵就能理解 head "能提取和写入什么特征"，这完全不依赖于输入数据。

6. **核心 idea**：对 CLIP 注意力头的 VO 矩阵做 SVD，再用语义一致的稀疏编码（COMP）将每个奇异向量映射到人类可理解的概念组合，实现无需数据的细粒度权重空间可解释性。

## 方法详解

### 整体框架
SITH 的分析流程为三步：（1）隔离每个注意力头的 VO 矩阵 $\mathbf{W}_{VO}^{l,h} = \mathbf{W}_V^h \mathbf{W}_O^h$；（2）对该矩阵做 SVD 分解 $\mathbf{W}_{VO} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$，得到右奇异向量（输出方向）和对应的奇异值（重要性）；（3）将每个奇异向量投影到 CLIP 的多模态空间后，用 COMP 算法将其表示为概念池中 $K$ 个语义一致概念的稀疏非负组合。最终得到每个 head 内部每个主要计算方向的人类可理解解释。

### 关键设计

1. **VO 矩阵的 SVD 分解**

    - 功能：发现注意力头内部最重要的信息流动方向
    - 核心思路：VO 矩阵 $\mathbf{W}_{VO}$ 是线性变换，SVD 将其分解为读取方向（$\mathbf{u}_i$，head 从哪里读信息）、写入方向（$\mathbf{v}_i$，head 往哪里写信息）和放大系数（$\sigma_i$，该方向的重要性）。按奇异值从大到小排列，可以识别 head 最重要的计算方向。整个过程完全基于权重，不依赖任何输入数据
    - 设计动机：注意力矩阵 $\mathbf{A}^h$ 决定"信息从哪个 patch 流向哪个 patch"（路由），而 VO 矩阵决定"流的是什么信息"（内容）。分析 VO 即可获得 input-independent 的理解

2. **COMP（Coherent Orthogonal Matching Pursuit）算法**

    - 功能：将每个奇异向量解释为语义一致的概念稀疏组合
    - 核心思路：给定奇异向量 $\hat{\mathbf{v}}$ 和概念嵌入矩阵 $\hat{\mathbf{\Gamma}}$，寻找稀疏非负系数 $\mathbf{c}$ 使得 $\hat{\mathbf{v}} \approx \hat{\mathbf{\Gamma}}^T \mathbf{c}$。标准 NNOMP 贪心地选最大相关性概念，但可能选出语义不相关的概念集。COMP 修改评分函数为 $\text{score}(\hat{\gamma}_i) = \langle \mathbf{r}_{k-1}, \hat{\gamma}_i \rangle + \frac{\lambda}{|S_{k-1}|}\sum_{j \in S_{k-1}} \langle \hat{\gamma}_i, \hat{\gamma}_j \rangle$，第二项鼓励新选概念与已选概念语义相似，超参数 $\lambda$ 控制"重建精度 vs 语义一致性"的权衡
    - 设计动机：单纯的 top-k 相似度选择只能捕捉奇异向量的局部语义，NNOMP 重建好但概念不连贯。COMP 在两者之间找到最佳平衡

3. **基于 SITH 的权重编辑**

    - 功能：通过调整奇异值实现精准的概念级模型干预
    - 核心思路：利用 SITH 的概念解释，用 LLM 评估每个奇异向量的概念集与下游任务的相关性，然后放大相关方向的奇异值、抑制不相关方向的奇异值。完全不需要训练数据或梯度更新
    - 设计动机：相比 TextSpan 需要移除整个 head，SITH 可以精确到 head 内部的单个奇异向量，实现更精准的"手术刀式"编辑

### 损失函数 / 训练策略
- SITH 本身无需训练。COMP 算法是确定性迭代过程，超参数为概念数 $K$（默认 5）和一致性系数 $\lambda$（默认 0.3）
- 概念池使用 ConceptNet 5.5
- 分析聚焦于 OpenCLIP ViT-L/14 的最后 4 层（L=24, H=16, r=64）

## 实验关键数据

### 主实验——可解释性 vs 保真度

COMP 在 $\lambda=0.3, K=5$ 时取得最佳平衡：
- 可解释性（LLM评分 5分制）：COMP ≈ 3.8, NNOMP ≈ 3.0, top-k ≈ 4.2
- 重建保真度（余弦相似度）：COMP ≈ 0.6, NNOMP ≈ 0.65, top-k ≈ 0.35
- 用 SITH 重建的奇异向量替换原始向量后，零样本分类精度几乎无下降

### 权重编辑应用

| 任务 | 原始 OpenCLIP | TextSpan编辑 | SITH编辑 |
|------|-------------|------------|---------|
| Waterbirds (Overall Acc) | 73.5 | 81.8 | **82.7** |
| Waterbirds (Worst-group Acc) | 47.9 | 68.0 | **70.6** |
| Flowers 102 (零样本) | 76.5 | - | **77.5** |
| FGVC-Aircraft (零样本) | 36.6 | - | **36.9** |
| DTD (零样本) | 50.1 | - | **50.9** |

### 消融实验——NSFW 内容抑制

| 方法 | 安全查询→检索 | 不安全查询→检索 |
|------|-------------|--------------|
| Safe-CLIP (训练方法) | T→V: 69.2 | T*→V: 46.3 |
| OpenCLIP (原始) | T→V: 75.1 | T*→V: 29.3 |
| **SITH (无训练)** | T→V: **74.5** | T*→V: 29.5 |

SITH 在不牺牲安全查询性能的前提下，通过权重编辑抑制 NSFW 概念。

### 关键发现
- 单个奇异向量确实对应人类可理解的语义概念（如"粉红色"、"冬季穿着"、"海洋海滩"、"两个物体"），验证了权重空间分析的有效性
- 微调（full FT 和 LoRA）主要是重新加权已有的语义基底，而非学习全新特征——奇异向量空间高度稳定
- 微调引入的权重变化 $\Delta \mathbf{W}$ 的奇异向量与微调任务高度对齐（如 Flowers 102 微调后出现"alpine flowers"等概念）
- SITH 的"手术刀式"编辑优于 TextSpan 的整体 head 消融，因为后者可能误伤同一 head 中的有用特征

## 亮点与洞察
- **完全 data-free 的可解释性**：不看任何图像就能理解 CLIP 内部在做什么，这从根本上避免了数据偏差问题。在大模型透明度日益重要的背景下，这一方向极具价值
- **COMP 算法的巧妙设计**：在稀疏编码的贪心选择中加入语义一致性正则，idea 简洁但效果显著，从"apple + red"这样不连贯的解释变为"pink red + scarlet reds + red background"这样连贯的解释
- **从可解释性到可干预性的闭环**：不只是看懂模型，还能基于理解做精准编辑（抑制虚假相关、移除NSFW、提升分类性能），形成了完整的"理解→干预"pipeline
- **微调机制的新理解**：微调不是在学新东西，而是在已有语义基底上重新分配权重——这一发现对理解 LoRA 等方法的工作原理有启发

## 局限与展望
- 仅分析 VO 矩阵的右奇异向量（写入方向），未深入分析 QK 矩阵和注意力路由模式
- FFN 层未纳入分析范围，但 FFN 也存储了大量知识
- 概念池的覆盖度影响解释质量，ConceptNet 可能对某些领域覆盖不足
- 权重编辑的效果虽然一致但幅度有限（通常 1-2 个百分点的提升），实际应用中可能需要与其他方法结合
- 目前仅在 CLIP ViT 上验证，是否适用于 decoder-only VLM 或其他架构需要进一步验证

## 相关工作与启发
- **vs TextSpan**：TextSpan 需要 ImageNet 级数据来计算激活，解释粒度为 head 级别；SITH 完全 data-free，解释到 intra-head 的奇异向量级别，更精细也更不受数据偏差影响
- **vs Sparse Autoencoders (SAE)**：SAE 需要训练，存在不稳定性（不同数据训练出不同字典），且是 sample-level 解释；SITH 是确定性分析，给出模型层面的全局解释
- **vs 单模态权重分析**：之前的 SVD 权重分析限于语言模型，用简单的 nearest-neighbor 搜索做解释；SITH 的 COMP 算法提供更完整的语义覆盖

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次实现 CLIP 的完全 data-free intra-head 可解释性
- 实验充分度: ⭐⭐⭐⭐⭐ 可解释性验证+权重编辑应用+微调分析，实验全面
- 写作质量: ⭐⭐⭐⭐⭐ 图表精美，概念清晰，从方法到应用层层递进
- 价值: ⭐⭐⭐⭐ 对理解 VLM 内部机制有重要贡献，权重编辑应用有实用性但效果幅度有限

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)
- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)

<!-- RELATED:END -->
