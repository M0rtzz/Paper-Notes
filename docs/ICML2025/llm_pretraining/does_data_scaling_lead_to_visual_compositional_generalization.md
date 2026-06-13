---
title: >-
  [论文解读] Does Data Scaling Lead to Visual Compositional Generalization?
description: >-
  [ICML 2025][预训练][组合泛化] 本文通过受控实验系统研究了数据规模与数据多样性对视觉模型组合泛化能力的影响，发现组合泛化的关键驱动力是数据多样性而非数据量，并证明当表示呈线性分解结构时仅需每个概念值2个组合样本即可完美泛化。 领域现状：主流范式是"扩大数据和模型规模即可提升性能"。但组合理解——通过组合已知简单…
tags:
  - "ICML 2025"
  - "预训练"
  - "组合泛化"
  - "数据多样性"
  - "线性分解表示"
  - "视觉推理"
  - "预训练模型"
---

# Does Data Scaling Lead to Visual Compositional Generalization?

**会议**: ICML 2025  
**arXiv**: [2507.07102](https://arxiv.org/abs/2507.07102)  
**代码**: [https://github.com/oshapio/visual-compositional-generalization](https://github.com/oshapio/visual-compositional-generalization)  
**领域**: LLM预训练  
**关键词**: 组合泛化, 数据多样性, 线性分解表示, 视觉推理, 预训练模型

## 一句话总结
本文通过受控实验系统研究了数据规模与数据多样性对视觉模型组合泛化能力的影响，发现组合泛化的关键驱动力是数据多样性而非数据量，并证明当表示呈线性分解结构时仅需每个概念值2个组合样本即可完美泛化。

## 研究背景与动机
**领域现状**：主流范式是"扩大数据和模型规模即可提升性能"。但组合理解——通过组合已知简单概念理解新场景——是人类智能基石，现有视觉模型表现不佳。  

**现有痛点**：即使 LAION-400M 这样的大规模数据集，概念组合覆盖上也存在严重稀疏性。组合爆炸导致大多数可能组合在训练集中缺失。  

**核心矛盾**：Scaling laws 预测扩大数据能持续提升性能，但概念组合空间指数增长，堆量可能无法解决组合稀疏。  

**本文目标**：精确回答"视觉模型能否组合泛化？在什么条件下？"  

**切入角度**：设计 (n,k) 框架参数化控制概念空间复杂性和训练数据多样性。  

**核心 idea**：组合泛化不是靠"堆量"解决的，而是靠增加概念组合多样性迫使模型发现线性分解表示结构。

## 方法详解

### 整体框架
两个设置：(1) 从零训练 ResNet-50 控制 (n,k) 参数；(2) 评估预训练大模型（DINO, CLIP）的组合泛化。度量三维度：泛化精度（ID vs OOD）、表示结构（线性度 R²、正交性、可解码性）、理论保证。

### 关键设计

1. **(n,k) 实验框架**: 

    - 功能：参数化控制概念空间复杂度和组合多样性
    - 核心思路：n = 每概念可取值数，k = 每概念值出现的训练组合数。n² 个可能组合中仅 nk 个用于训练
    - 设计动机：独立控制 n 和 k，精确分离"数据量"与"数据多样性"的贡献

2. **线性分解表示（Linearly Factored Embeddings）**: 

    - 功能：检测模型是否学到"概念可加"结构
    - 核心思路：复合概念表示 = 各概念表示的向量和 u_c = u_{c1} + u_{c2}
    - 度量方式：R² 系数衡量实际表示与线性重构的吻合程度

3. **三阶段特征学习动态**: 

    - 阶段一（0-10%覆盖）：虚假特征，decoded accuracy < 80%，零样本泛化随机水平
    - 阶段二（25-75%覆盖）：可判别但非线性分解，零样本精度60-80%
    - 阶段三（75-100%覆盖）：强线性（R² > 0.8）和正交，零样本精度 > 90%

4. **最小组合学习命题（Proposition 4.1）**: 

    - 证明理想线性分解表示下 k=2 即可完美泛化到所有未见组合
    - 条件：概念表示的联合张成空间维度为 2n-1

### 损失函数 / 训练策略
- 从零训练：ResNet-50 + 双线性分类头，交叉熵损失同时预测两概念
- 预训练评估：冻结特征 + MLP 探针，oracle 模型选择
- 数据集：DSprites, 3DShapes, PUG, Colored-MNIST, FSprites

## 实验关键数据

### 主实验

| 数据集 | 设置(n,k) | ID精度 | OOD精度 | 精度下降 |
|--------|-----------|--------|---------|---------|
| CMNIST | (3,2) | ~100% | ~22% | -78% |
| FSprites | (3,2) | ~100% | 不等 | 各概念差异大 |
| DSprites | (3,2) | ~100% | 不等 | 3%-40%不等 |
| Shapes3D | (3,2) | ~100% | 不等 | 部分仅-17% |

| 预训练模型 | 优势概念 | 劣势概念 |
|-----------|---------|---------|
| CLIP-ViT-L/14 | 颜色类概念最优 | 形状类较弱 |
| DINOv2-ViT-L/14 | 形状/尺度/朝向最优 | 颜色类较弱 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 增加ID数据量4倍 | OOD精度无改善 | 证明"堆量"无效 |
| 增加n（更多概念值） | OOD精度提升 | 多样性驱动泛化 |
| 增加k（更多组合） | OOD精度提升 | 组合多样性同样有效 |
| ViT vs ResNet | ViT无优势 | 非架构问题 |

### 关键发现
- 4倍ID数据量增加几乎不改善OOD泛化
- 线性分解结构只在高多样性训练下自然涌现
- 三阶段学习动态揭示了 simplicity bias 的新理解
- 预训练模型展现部分线性分解但远非完美
- CLIP 擅长颜色，DINOv2 擅长形状——组合能力有"概念选择性"

## 亮点与洞察
- 精心设计的受控实验做出因果性声明（非仅相关性分析）
- Proposition 4.1：k=2 就够了（理想条件下），理论结果优美
- 三阶段特征学习动态与 grokking/phase transition 有趣类比
- 对"Scaling Laws能解决一切"的流行观点提出有力反例

## 局限与展望
- 仅考虑两个概念组合，更多概念可能更具挑战
- 数据集主要为合成数据
- Proposition 4.1 假设概念数量大时可能不成立
- 缺乏自然图像组合泛化的深入分析

## 相关工作与启发
- Trager et al. (2023), Stein et al. (2024) 发现大模型中部分线性分解结构
- Geirhos et al. (2020) 的 simplicity bias 工作解释了模型倾向学虚假特征
- 启发：数据集策展（curation）比数据集规模更重要

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [\[ICML 2025\] Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)

</div>

<!-- RELATED:END -->
