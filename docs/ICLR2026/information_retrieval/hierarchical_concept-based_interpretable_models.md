---
title: >-
  [论文解读] Hierarchical Concept-based Interpretable Models
description: >-
  [ICLR 2026][信息检索/RAG][概念嵌入模型] HiCEMs引入层级概念嵌入模型，通过Concept Splitting方法在预训练CEM的嵌入空间中自动发现细粒度子概念（无需额外标注），构建层级概念结构，使模型能在不同粒度层次进行测试时概念干预以提升任务性能。
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "概念嵌入模型"
  - "层级概念"
  - "概念分裂"
  - "子概念发现"
  - "概念干预"
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
HiCEMs不直接从多层级标注学习，而是先训练一个标准CEM拿到可靠的概念嵌入空间，再用Concept Splitting在这个空间里无监督地把粗概念拆成子概念，最后在拆出来的层级上重新训练得到支持多粒度干预的HiCEM。输入仍是图像，区别在于概念瓶颈层不再输出扁平的概念向量，而是输出"父概念+子概念"的层级化预测，标签层据此分类。

### 关键设计

**1. 用 CEM 嵌入替代二值概念：让"拆分"成为可能**

子概念发现之所以能不依赖额外标注，前提是概念表征里本就藏着比标注更细的信息，而这正是CEM相对CBM的差别。标准CBM把每个概念压成一个0/1标量，丢掉了所有细粒度语义；CEM则把每个概念学成高维向量：对输入图像$x$，概念编码器$g$输出概念嵌入$c_i = g_i(x) \in \mathbb{R}^d$，再用它与正/负概念原型的相似度判断概念是否存在。连续嵌入保留了"翅膀是条纹还是纯色"这类标注里没有但视觉上可分的差异，正是这个信息丰富的空间给后续聚类拆分留出了余地。

**2. Concept Splitting：在嵌入空间里把一个概念聚成几个子概念**

核心假设很直接——如果一个粗概念实际混了多个子概念，它在训练集上的嵌入向量就会在空间里聚成几个可分的簇。于是对概念$c$，先收集它在全体训练样本上的嵌入$\{c^{(j)}\}$，对这些向量做聚类（k-means 或层次聚类），每个有意义的子簇即对应一个子概念，簇中心则当作该子概念的原型向量。这一步没有引入任何新标注，只是把CEM训练时已经隐式学到、却从未被显式利用的子概念结构显式化出来。子概念数$K$不是全局固定的：对每个概念分别尝试不同$K$，按验证集上的概念可分离性挑最优，因此"翅膀颜色"可能裂成两三个子概念，而本就原子化的概念会被判定为无需再分。

**3. HiCEM 架构：把发现的层级接回预测，并保持父子一致**

拿到父概念$c_i$及其子概念$\{c_{i,1}, \ldots, c_{i,K}\}$后，HiCEM同时预测父、子两层概念的存在性，并靠三件事让层级真正可用：层级一致性要求子概念预测服从父概念——父概念"翅膀颜色"若不存在，其子概念也不该被激活；层级聚合让标签层按任务自取粒度，粗概念够用的任务就用粗概念，需要判别细节的任务才调用子概念；多粒度干预接口则允许测试时在任意层修正概念，改父概念会级联影响子概念，改子概念只作用于局部，这正是后面细粒度干预更高效的来源。

**4. PseudoKitchens 数据集：造一个层级天然可控的测试场**

真实数据里很难精确控制概念的层级组合，于是本文用3D厨房渲染引擎合成了一个新数据集，里面的厨具与食品概念带有天然层级（如"容器"含"杯子""碗"）。渲染可逐项控制概念出现与否，便于干净地检验层级概念模型是否真的利用了子概念结构。

### 损失函数 / 训练策略
训练分两阶段：先把标准CEM训到收敛、跑Concept Splitting定下层级，再在该层级上训HiCEM。HiCEM的损失含三项：概念预测损失对父、子概念各算二元交叉熵并加权合成，$\mathcal{L}_{concept} = \mathcal{L}_{parent} + \lambda \mathcal{L}_{sub}$；任务预测损失是层级化概念表征上的分类交叉熵；层级一致性正则则鼓励子概念预测逻辑上从属于父概念预测，把上面架构里的一致性约束落到优化目标上。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Digging Deeper: Learning Multi-Level Concept Hierarchies](digging_deeper_learning_multi-level_concept_hierarchies.md)
- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

</div>

<!-- RELATED:END -->
