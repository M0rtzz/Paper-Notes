---
title: >-
  [论文解读] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism
description: >-
  [CVPR 2026][自监督学习][generalized category discovery] 提出 OmniGCD，首个模态无关的广义类别发现方法，利用合成数据训练的 GCDformer 在测试时将任意模态的 GCD 潜空间变换为更适合聚类的表示，在 16 个跨四种模态的数据集上实现零样本 GCD。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "generalized category discovery"
  - "modality-agnostic"
  - "zero-shot"
  - "Transformer"
  - "synthetic training"
---

# OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism

**会议**: CVPR 2026  
**arXiv**: [2604.14762](https://arxiv.org/abs/2604.14762)  
**代码**: [github.com/Jordan-HS/OmniGCD](https://github.com/Jordan-HS/OmniGCD)  
**领域**: 自监督学习/表示学习  
**关键词**: generalized category discovery, modality-agnostic, zero-shot, transformer, synthetic training

## 一句话总结

提出 OmniGCD，首个模态无关的广义类别发现方法，利用合成数据训练的 GCDformer 在测试时将任意模态的 GCD 潜空间变换为更适合聚类的表示，在 16 个跨四种模态的数据集上实现零样本 GCD。

## 研究背景与动机

广义类别发现 (GCD) 模拟人类的类别学习能力，在部分标注数据下同时识别已知类和发现新类。神经科学研究表明人类的类别形成是独立于感觉输入的抽象过程。然而现有 GCD 方法都在单一模态内操作并需要数据集特定的微调，忽视了类别学习的根本抽象性。这促使设计模态无关的方案——训练一次即可跨视觉、文本、音频、遥感等模态的零样本 GCD。

## 方法详解

### 整体框架

OmniGCD 使用模态特定编码器将输入映射到特征空间，降维映射到低维 GCD 潜空间，拼接标签嵌入/掩码 token 后送入 GCDformer。GCDformer 在测试时（无梯度更新）变换潜空间使其更适合 k-means 聚类。

### 关键设计

1. **GCDformer Transformer**: 基于 GPT-2 架构的非因果自注意力 Transformer。输入为数据 token（降维后特征 $d$ 维）与标签 token（正弦位置编码或可学习掩码 $d_l$ 维）的拼接。不编码位置信息（因为 GCD 任务中输入是集合而非序列）。训练目标为对比损失——同类点拉近、异类点推远。

2. **合成数据训练**: 完全使用合成数据训练 GCDformer 以保持模态无关性。合成数据需满足两个关键属性：(1) 充分覆盖 GCD 潜空间；(2) 与真实数据分布对齐。选择低维潜空间（使采样空间可控）和适当的降维方法。

3. **降维方法选择**: 对比 PCA、UMAP、t-SNE。t-SNE 在合成-真实分布 KL 散度（1.41）和簇分离度/簇扩展度/簇重叠度等指标上整体最优。t-SNE 的非线性特性和重尾 t 分布更好地保留局部结构并缓解拥挤问题。

### 损失函数 / 训练策略

对比损失结合 margin 参数：同类对使用 L2 距离拉近，异类对使用 margin 约束推远。GCDformer 仅训练一次，对所有 16 个数据集使用同一模型。

## 实验关键数据

### 主实验

在 16 个数据集、4 种模态上的平均准确率提升（pp）：

| 模态 | 已知类提升 | 新类提升 |
|------|-----------|---------|
| 视觉 | +6.2pp | +6.2pp |
| 文本 | +17.9pp | +17.9pp |
| 音频 | +1.5pp | +1.5pp |
| 遥感 | +12.7pp | +12.7pp |

首次在音频模态上实现 GCD。

### 消融实验

- t-SNE 降维优于 PCA 和 UMAP（KL 散度最低、簇质量最优）
- GCDformer 在合成数据上快速过拟合，需要足够的采样多样性
- 编码器质量直接影响最终 GCD 性能

### 关键发现

- 将 GCD 抽象为表示空间变换问题的视角新颖
- 合成数据训练实现了真正的模态无关性
- 编码器质量是性能瓶颈——更好的"眼睛"直接带来更好的 GCD

## 亮点与洞察

- 受人脑前额叶皮层抽象类别形成启发的研究动机自然
- 将表示学习与类别发现解耦的设计使模态编码器和 GCD 能力可独立进步
- 零样本 GCD 的新设定填补了文献空白

## 局限与展望

- t-SNE 降维的非参数性质限制了对新数据的即时推理
- 低维潜空间可能丢失对某些细粒度区分关键的信息
- 合成数据的生成策略仍需人工设计

## 相关工作与启发

- Transformer 作为通用集合处理器的思路可推广到其他需要分组/聚类的任务
- 合成训练+零样本推理的范式对跨域泛化有启发
- 模态无关 GCD 的基准为后续工作提供了评测框架

## 评分

7/10 — 问题定义新颖，跨模态泛化令人印象深刻，但 t-SNE 依赖和低维限制需解决。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2025\] Hyperbolic Category Discovery](../../CVPR2025/self_supervised/hyperbolic_category_discovery.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](../../ICML2026/self_supervised/partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[CVPR 2026\] Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)

</div>

<!-- RELATED:END -->
