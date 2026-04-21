---
title: >-
  [论文解读] Hierarchical Concept-based Interpretable Models
description: >-
  [ICLR 2026][概念嵌入模型] HiCEMs引入层级概念嵌入模型，通过Concept Splitting方法在预训练CEM的嵌入空间中自动发现细粒度子概念（无需额外标注），构建层级概念结构，使模型能在不同粒度层次进行测试时概念干预以提升任务性能。
tags:
  - ICLR 2026
  - 概念嵌入模型
  - 层级概念
  - 概念分裂
  - 子概念发现
  - 概念干预
---

# Hierarchical Concept-based Interpretable Models

**会议**: ICLR 2026  
**arXiv**: [2602.23947](https://arxiv.org/abs/2602.23947)  
**代码**: 无  
**领域**: 可解释AI / 概念模型  
**关键词**: 概念嵌入模型, 层级概念, 概念分裂, 子概念发现, 概念干预

## 一句话总结
HiCEMs引入层级概念嵌入模型，通过Concept Splitting方法在预训练CEM的嵌入空间中自动发现细粒度子概念（无需额外标注），构建层级概念结构，使模型能在不同粒度层次进行测试时概念干预以提升任务性能。

## 研究背景与动机
现代深度神经网络因其潜在表征的不透明性而难以解释，阻碍了模型理解、调试和去偏。概念嵌入模型（CEM）通过将输入映射到人类可理解的概念表征来解决这个问题。然而，CEM存在两个根本性局限：(1) 无法表示概念间的关系——将所有概念视为扁平的、独立的，忽略了概念天然的层级结构（如"羽毛颜色"→"胸部红色"/"翅膀蓝色"）；(2) 需要不同粒度级别的概念标注来训练层级模型，标注成本极高。核心矛盾是：层级概念结构对深入理解和精准干预至关重要，但获取多层级标注数据不切实际。本文的核心idea是：通过Concept Splitting在已有CEM的嵌入空间中自动发现子概念，无需任何额外标注即可构建层级概念结构。

## 方法详解

### 整体框架
HiCEMs的pipeline分为三步：(1) 训练标准CEM获得可靠的概念嵌入空间；(2) 对训练好的CEM应用Concept Splitting，在无额外监督下发现子概念；(3) 使用发现的子概念训练HiCEM，获得支持层级概念干预的可解释模型。输入为图像，概念瓶颈层输出层级化的概念预测（父概念+子概念），最终标签预测层基于整个层级结构进行分类。

### 关键设计

1. **概念嵌入模型 (CEM) 基础**: CEM将每个概念学习为高维嵌入空间中的向量，而非简单的二值标量。对于输入图像$x$，概念编码器$g$产出概念嵌入$c_i = g_i(x) \in \mathbb{R}^d$，然后通过与正/负概念原型向量的相似度来判断概念是否存在。相比标准CBM（Concept Bottleneck Model）的二值概念预测，CEM的连续嵌入空间包含了更丰富的语义信息——正是这个信息丰富的嵌入空间为后续发现子概念提供了可能。

2. **Concept Splitting**: 给定一个训练好的CEM中某个概念$c$的嵌入空间，Concept Splitting分析该概念在整个训练集上的激活模式分布。核心假设是：如果一个粗粒度概念实际上包含多个子概念，那么其嵌入向量在空间中会形成多个可分离的簇。算法过程为：(a) 收集概念$c$在训练集上的所有嵌入向量$\{c^{(j)}\}$；(b) 对这些向量进行聚类分析（如k-means或层次聚类），识别有意义的子簇；(c) 每个子簇对应一个子概念，其聚类中心作为子概念的原型向量。设计动机是利用CEM嵌入空间中已有的但未被显式利用的结构信息——CEM在学习概念预测时，自然地在嵌入空间中捕获了子概念级别的区分度。通过聚类操作，这些隐式的子概念结构被显式化。

3. **HiCEMs架构**: 在标准CEM基础上引入层级结构。对于每个父概念$c_i$及其通过Concept Splitting发现的子概念$\{c_{i,1}, c_{i,2}, \ldots, c_{i,K}\}$，HiCEM同时预测父概念和子概念的存在性。架构上的关键设计包括：**(a) 层级一致性**：子概念的预测与父概念保持逻辑一致——如果父概念"翅膀颜色"不存在，其子概念也不应被激活；**(b) 层级聚合**：标签预测层可以选择性地利用不同粒度的概念信息——对于某些任务，粗粒度就够了，对于另一些任务，细粒度子概念提供了关键判别信息；**(c) 多粒度干预接口**：测试时，用户可以在任意层级进行概念干预——修正父概念会级联影响子概念，修正子概念则只影响局部。

4. **PseudoKitchens数据集**: 为验证HiCEM，本文提出了一个新的基于概念的数据集。使用3D厨房渲染引擎生成图像，包含多层级的厨具和食品概念。该数据集的特点是概念间存在天然的层级关系（如"容器"→"杯子"/"碗"），适合验证层级概念模型的有效性。

### 损失函数 / 训练策略
HiCEM的训练损失包含三部分：(1) **概念预测损失**：对父概念和子概念分别计算二元交叉熵，$\mathcal{L}_{concept} = \mathcal{L}_{parent} + \lambda \mathcal{L}_{sub}$；(2) **任务预测损失**：基于层级化概念表征的分类交叉熵；(3) **层级一致性正则化**：鼓励子概念预测与父概念预测之间的逻辑一致性。训练流程为：先训练标准CEM至收敛→运行Concept Splitting发现子概念→在发现的层级上训练HiCEM。子概念数量$K$通过验证的方法确定——尝试不同的$K$值，选择验证集上概念可分离性最好的设置。

## 实验关键数据

### 主实验

| 数据集 | 指标 | HiCEM | 标准CEM | CBM | 说明 |
|--------|------|------|----------|------|------|
| MNIST-ADD | Task Acc | ~高 | 基线 | 较低 | 数字加法任务 |
| SHAPES | Task Acc | ~高 | 基线 | 较低 | 形状属性识别 |
| CUB-200 | Task Acc | 竞争力 | 基线 | 较低 | 鸟类细粒度分类 |
| AwA2 | Task Acc | 竞争力 | 基线 | 较低 | 动物属性预测 |
| PseudoKitchens | Task Acc | 最优 | 基线 | 较低 | 新提出的3D厨房数据集 |

注：HiCEM在所有数据集上保持了与CEM相当或更好的准确率，同时提供了更细粒度的解释。

### 概念干预实验

| 数据集 | 干预方式 | 无干预 | 粗粒度干预 | 细粒度干预(HiCEM) | 说明 |
|--------|---------|------|---------|---------|------|
| CUB-200 | 随干预数增加 | 基线 | 提升 | 更大提升 | 细粒度干预效果更好 |
| AwA2 | 随干预数增加 | 基线 | 提升 | 更大提升 | 层级干预的累积效应 |
| SHAPES | 随干预数增加 | 基线 | 提升 | 更大提升 | 尤其在中等干预数量时优势明显 |

### 用户研究（User Study）

| 评估维度 | 结果 | 说明 |
|---------|------|------|
| 子概念可理解性 | 用户能为多数子概念赋予有意义的名称 | 验证了Concept Splitting发现的子概念具有人类可理解的语义 |
| 解释有用性 | HiCEM的层级解释比CEM的扁平解释更受青睐 | 层级结构提供了更直观的错误追踪路径 |
| 干预效率 | 细粒度干预需要更少的修正次数 | 精准定位出错的子概念比修正粗粒度概念更高效 |

### 关键发现
- Concept Splitting发现的子概念具有很高的人类可理解性——在CUB数据集上，"翅膀颜色"被分裂为"翅膀条纹"和"翅膀纯色"等子概念，用户可以直观理解
- 细粒度概念干预比粗粒度干预更有效：在CUB上，干预5个细粒度子概念的效果优于干预5个粗粒度父概念
- HiCEM在不牺牲任务准确率的前提下提供了更丰富的解释，打破了"可解释性 vs 准确率"的常见trade-off
- 在PseudoKitchens上的实验表明，具有天然层级概念的域中HiCEM的优势最为明显
- CEM嵌入空间中确实存在有意义的子概念结构——这验证了CEM在训练过程中隐式学习了超出标注粒度的信息
- 不同概念的最优分裂数量不同：有些概念自然地包含多个子概念，有些则是"原子"概念不需要进一步分裂

## 亮点与洞察
- **零额外标注的子概念发现**：这是本文最大的贡献——仅利用CEM训练过程中自然形成的嵌入空间结构，不需要任何新的标注就能发现细粒度子概念
- **可解释性的层级化**：从"模型使用了哪些概念"到"模型具体使用了概念的哪个方面"，这是可解释AI的重要进步
- **概念干预的精细化**：测试时干预从"修正一个概念"进化为"在正确的层级修正正确的子概念"，大幅提高了干预效率
- **新数据集PseudoKitchens**：为概念层级研究提供了一个可控的实验环境（3D渲染可精确控制概念组合），填补了领域空白
- **理论洞察**：CEM嵌入空间天然包含比标注更丰富的信息这一发现，启发了对其他representation learning方法的类似探索

## 局限与展望
- Concept Splitting的质量高度依赖初始CEM嵌入空间的质量——如果CEM学得不好，聚类出的子概念可能没有意义
- 目前仅支持一层分裂（父→子），未扩展到多层分裂（由同组Workshop论文"Digging Deeper"探索了多层扩展）
- 聚类算法的选择和超参数（如$K$值）仍需人工调优或验证
- 在大规模数据集（如ImageNet）上的可扩展性未验证
- 层级一致性约束可能过于严格——现实中子概念不一定严格从属于父概念
- 未与基于注意力的可解释方法（如GradCAM）或特征归因方法（如SHAP）进行系统比较

## 相关工作与启发
- **Concept Bottleneck Models (CBM)**: 可解释AI的基础框架，HiCEM在其上引入了层级结构
- **Concept Embedding Models (CEM)**: HiCEM的直接前身，通过连续嵌入而非二值标量表示概念
- **Digging Deeper (ICLR 2026 Workshop)**: 同组后续工作，将Concept Splitting扩展到多层次（MLCS），配合Deep-HiCEMs架构
- **Concept Activation Vectors (TCAV)**: 另一种概念发现方法，但不构建层级结构
- **启发**：概念层级的自动发现思路可以推广到：(1) 公平性分析——发现敏感属性的子群体；(2) 模型调试——定位模型错误的精确概念层级；(3) 数据增强——基于概念层级的结构化采样

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Digging Deeper: Learning Multi-Level Concept Hierarchies](digging_deeper_learning_multi-level_concept_hierarchies.md)
- [Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe](../../NeurIPS2025/information_retrieval/hierarchical_retrieval_the_geometry_and_a_pretrain-finetune_recipe.md)
- [TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

<!-- RELATED:END -->
