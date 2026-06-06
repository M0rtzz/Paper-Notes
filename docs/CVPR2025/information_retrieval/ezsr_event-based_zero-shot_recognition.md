---
title: >-
  [论文解读] EZSR: Event-based Zero-Shot Recognition
description: >-
  [CVPR 2025][信息检索/RAG][事件相机] 提出 EZSR 框架用于事件相机数据的零样本物体识别，通过标量级调制（scalar-wise modulation）策略解决事件嵌入与 CLIP 文本嵌入之间的语义错位问题，并通过从静态 RGB 图像大规模合成事件数据来突破训练数据稀缺限制…
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "事件相机"
  - "零样本识别"
  - "CLIP蒸馏"
  - "标量调制"
  - "数据合成"
---

# EZSR: Event-based Zero-Shot Recognition

**会议**: CVPR 2025  
**arXiv**: [2407.21616](https://arxiv.org/abs/2407.21616)  
**代码**: [https://yan98.github.io/EZSR/](https://yan98.github.io/EZSR/)  
**领域**: 信息检索  
**关键词**: 事件相机, 零样本识别, CLIP蒸馏, 标量调制, 数据合成

## 一句话总结
提出 EZSR 框架用于事件相机数据的零样本物体识别，通过标量级调制（scalar-wise modulation）策略解决事件嵌入与 CLIP 文本嵌入之间的语义错位问题，并通过从静态 RGB 图像大规模合成事件数据来突破训练数据稀缺限制，在 N-ImageNet 上以 ViT-B/16 达到 47.84% 零样本准确率。

## 研究背景与动机

**领域现状**：事件相机以异步方式捕获像素级亮度变化，具有高时间分辨率、无运动模糊、低功耗等优势。将 CLIP 的零样本能力扩展到事件域是近期研究热点，主要有两种方案：(1) 重建方法——从事件数据重建灰度图喂给 CLIP 图像编码器；(2) 对比学习方法——用配对事件-RGB 数据蒸馏事件编码器。

**现有痛点**：重建方法因重建质量低和误差累积导致零样本性能差。对比学习方法在理论上存在根本缺陷——事件数据空间稀疏导致事件嵌入之间过度相似，对比学习的负样本推远目标无法有效区分它们。更关键的是，优化 InfoNCE 目标（使事件嵌入对齐 RGB 嵌入）并不意味着事件嵌入自动对齐 CLIP 的文本嵌入（Lemma 1）。此外，配对事件-RGB 数据集稀缺，现有方法常在同一数据集上训练和测试。

**核心矛盾**：对比学习的 InfoNCE 损失不仅优化配对数据的相似性，还推远非配对数据。但由于事件嵌入间相似度本身很高，推远操作给嵌入空间引入了与 CLIP 文本空间不对齐的自由度——即使事件-RGB 对齐成功，事件-文本对齐也可能失败。

**本文目标**：设计一种不依赖重建网络、能直接从事件数据进行零样本识别的事件编码器，并解决训练数据稀缺问题。

**切入角度**：通过标量级（逐维度）调制替代对比学习中的推远操作，直接将事件嵌入的每个维度对齐到 RGB 嵌入对应维度，消除自由度导致的语义错位。同时用简单的仿射变换从静态 RGB 图像合成事件数据，大规模扩展训练集。

**核心 idea**：让事件编码器输出的每个标量维度直接逼近 CLIP 图像编码器的对应维度，从而继承 RGB-文本的对齐关系——因为 CLIP 中 RGB 和文本已经对齐，如果事件和 RGB 在每个维度上都对齐，那么事件和文本也自然对齐。

## 方法详解

### 整体框架
输入配对的事件数据和 RGB 图像，分别通过可训练的事件编码器 $f^{evt}$ 和冻结的 CLIP 图像编码器 $f^{img}$ 提取嵌入。通过标量级调制损失使事件嵌入逐维度对齐 RGB 嵌入。推理时，用事件编码器替换 CLIP 图像编码器，配合 CLIP 文本编码器实现零样本分类。训练数据通过从静态 RGB 图像合成事件数据获得。

### 关键设计

1. **标量级调制（Scalar-wise Modulation）**:

    - 功能：逐维度对齐事件嵌入和 RGB 嵌入，消除对比学习带来的语义错位
    - 核心思路：不使用 InfoNCE 的负样本推远目标，而是直接优化 $\mathcal{L}_{mod} = \| \hat{\mathbf{x}}^{evt} - \mathbf{x}^{img} \|$ 的某种标量级度量。具体来说，对嵌入的每个维度 $d$ 计算事件嵌入和 RGB 嵌入之间的对齐损失，使网络自适应地挖掘每个维度上的语义对应关系。这不同于仅优化向量级余弦相似度——余弦相似度只约束方向不约束各维度的单独对齐
    - 设计动机：Lemma 1 证明了对比学习即使成功最小化 InfoNCE，也不能保证事件-文本对齐。标量级调制直接消除了嵌入空间的自由度，如果事件嵌入的每个标量维度都等于 RGB 嵌入，那么与文本的相似度也一致

2. **k-NN 嵌入翻译（Remark 1）**:

    - 功能：推理时进一步缓解事件-文本嵌入的错位
    - 核心思路：维护一个预计算的 RGB 图像嵌入池。推理时，对事件嵌入找 k 个最近邻的 RGB 嵌入，按相似度加权平均得到翻译后的事件嵌入 $\tilde{\mathbf{x}}^{evt}$，用翻译后的嵌入与文本匹配。这相当于借助 RGB 嵌入池做了一次"中继校准"
    - 设计动机：即使训练时对齐不完美，通过参考池中语义相近的 RGB 嵌入可以进一步修正事件嵌入

3. **静态图像事件合成**:

    - 功能：大规模生成配对事件-RGB 训练数据，突破数据稀缺限制
    - 核心思路：对静态 RGB 图像随机生成仿射变换（平移、旋转、缩放），通过插值生成图像序列，再差分生成事件数据。相比传统方法需要视频+预训练帧插值网络，此方法计算成本低且引入更大多样性
    - 设计动机：零样本识别场景下事件数据通常持续时间短（毫秒级），包含近似线性运动。简单仿射变换即可模拟这一特性。大规模合成数据使模型展现出良好的参数和数据扩展性

### 损失函数 / 训练策略
最终损失为标量级调制损失（主要），可选地加上 InfoNCE 损失。CLIP 图像编码器完全冻结，仅训练事件编码器。事件编码器初始化自 CLIP 的图像编码器权重。训练数据为合成的事件-RGB 对。

## 实验关键数据

### 主实验

| 数据集 | EZSR (ViT-B/16) | 之前最佳零样本 | 之前最佳有监督 |
|--------|-----------------|---------------|---------------|
| N-ImageNet | 47.84% | ~35% (ExACT) | ~60% (有监督) |
| N-Caltech101 | 高于SOTA | — | — |
| CIFAR10-DVS | 高于SOTA | — | — |

### 消融实验（N-ImageNet）

| 配置 | 准确率 | 说明 |
|------|--------|------|
| 仅 InfoNCE baseline | 9.57% | 严重的事件-文本错位 |
| + Remark 1 (k-NN翻译) | 43.48% | k-NN 校准大幅提升 |
| 仅标量级调制 | 47.80% | 核心贡献 |
| 标量级调制 + Remark 1 | 48.63% | 进一步微小提升 |
| 全部组合 | 48.86% | 最优 |

### 关键发现
- 基线 InfoNCE 仅 9.57%，证明了对比学习在事件域的根本问题——事件-文本语义错位
- 标量级调制单独就能达到 47.80%（38+ 百分点提升），是核心创新
- 模型展现出良好的扩展性——增加参数和合成数据都能持续提升性能
- 在 9 个标准事件数据集上评估，零样本性能甚至超过部分数据集特定的有监督方法

## 亮点与洞察
- **对事件域对比学习失败的理论分析**：通过 Lemma 1 严格证明了 InfoNCE 最小化不能保证零样本的事件-文本对齐，这为理解跨域蒸馏的局限性提供了理论基础
- **标量级调制的简洁优雅**：不需要复杂的网络结构改动，仅改变损失函数从向量级到标量级就获得 38+ 百分点提升，体现了"对症下药"的重要性
- **从静态图像合成事件数据**：方法极其简单（仿射变换+差分），但有效解决了事件域训练数据稀缺的核心问题，且可以利用任何 RGB 图像数据集

## 局限与展望
- 合成事件数据使用简单仿射变换，无法捕获真实世界中的复杂运动模式和噪声特性
- k-NN 嵌入翻译需要维护一个大的 RGB 嵌入参考池，增加了推理时的存储和计算成本
- 零样本性能（48.86%）相比有监督方法（~60%）仍有较大差距
- 未探索动作识别等需要更长时间序列的事件任务中的表现

## 相关工作与启发
- **vs ExACT**: ExACT 也做事件域零样本识别，但依赖对比学习，受限于同一数据集训练测试。EZSR 通过合成数据和标量调制根本性地提升了性能
- **vs 重建方法 (E2VID+CLIP)**: 重建方法引入额外网络且误差累积。EZSR 直接学习事件编码器，端到端更简洁
- **vs CLIP 直接应用事件帧**: 事件帧稀疏且与 RGB 域差异大，直接用 CLIP 效果很差。EZSR 通过训练专用编码器弥合域差异

## 评分
- 新颖性: ⭐⭐⭐⭐ 对对比学习失败的理论分析和标量级调制方案都是有洞察力的贡献
- 实验充分度: ⭐⭐⭐⭐ 9 个数据集评估，消融清晰，扩展性实验有说服力
- 写作质量: ⭐⭐⭐⭐ 理论和实验结合好，逐步构建方法的叙述方式清晰
- 价值: ⭐⭐⭐⭐ 为事件相机的零样本学习提供了强基线，合成数据管线有很高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)
- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ICML 2026\] BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](../../ICML2026/information_retrieval/blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)
- [\[CVPR 2025\] COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation](cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/information_retrieval/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)

</div>

<!-- RELATED:END -->
