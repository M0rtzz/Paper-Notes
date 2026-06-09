---
title: >-
  [论文解读] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification
description: >-
  [CVPR 2026][可解释性][可解释分类] 提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配…
tags:
  - "CVPR 2026"
  - "可解释性"
  - "可解释分类"
  - "DINOv2"
  - "二次规划"
  - "视觉基础模型"
  - "特征稀疏化"
---

# DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification

**会议**: CVPR 2026  
**arXiv**: [2604.07166](https://arxiv.org/abs/2604.07166)  
**代码**: [https://github.com/RobertZimm/DINO-QPM](https://github.com/RobertZimm/DINO-QPM)  
**领域**: Model Interpretability / 模型可解释性  
**关键词**: 可解释分类, DINOv2, 二次规划, 视觉基础模型, 特征稀疏化

## 一句话总结

提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配，在 CUB-2011 和 Stanford Cars 上同时超越了 DINOv2 线性探测的准确率和所有可比方法的可解释性。

## 研究背景与动机

视觉基础模型（如 DINOv2）作为特征提取器表现卓越，但其复杂、高维的表示为可解释性带来了巨大挑战。现有方法面临以下问题：

**后验解释方法不可靠**：注意力图、Grad-CAM 等方法是外部近似，并非模型决策过程的真实反映；注意力图与下游任务无关，经常忽略分类所需的关键信息

**端到端可解释模型资源消耗大**：如原型网络等方法需要对整个骨干进行微调，计算开销极高

**冻结骨干上的方法精度不足**：后验概念瓶颈模型（Post-hoc CBMs）依赖文本概念监督，无法提供直接的空间定位；且精度通常不及全模型微调方法

**原型模型的可解释性有欺骗性**：其相似度计算不一定与人类认知一致

本文由此切入：与其去微调骨干或在事后硬贴解释，不如在完全冻结 DINOv2 的前提下挂一个轻量适配器，把它强大但纠缠的特征转换成稀疏、可空间定位、全局可解释的类表示——既不动骨干的能力，又把可解释性直接做进表示里。

## 方法详解

### 整体框架

DINO-QPM 想解决的是：在**完全冻结 DINOv2 骨干**的前提下，把它强大但高度纠缠的高维特征，转换成稀疏、可空间定位、全局可解释的类表示。整条管线只在骨干之上挂了一个轻量适配器。一张图先经冻结的 DINOv2 提取 patch 嵌入 $\boldsymbol{F}^{\text{froz}} \in \mathbb{R}^{N_p \times D}$；一个 MLP 把这些 patch 嵌入投射到任务特定的特征空间 $\boldsymbol{F} \in \mathbb{R}^{N_p \times N_f}$；接着对空间维度做平均池化得到特征向量 $\boldsymbol{f} = \text{AvgPool}(\boldsymbol{F}) \in \mathbb{R}^{N_f}$；最后由 BLDD（二值低维决策层）用一个二值稀疏矩阵把特征分配给各类、完成分类。

贯穿全程的一个关键取舍是**丢弃 CLS token、只用 patch 嵌入**——这违反分类文献的默认做法，但正因为如此，每个特征都对应一张可还原回原图分辨率的空间特征图，可解释性才有了空间落点。

### 关键设计

**1. MLP 特征变换：把纠缠特征解耦到任务空间。**

BLDD 决策层本身只是一个二值稀疏矩阵，没有任何表示变换能力，无法把 DINOv2 那 $D$ 维纠缠表示直接拿来稀疏分配。于是这里在决策层上游放一个 MLP，专门负责把 $D$ 维 patch 表示映射成 $N_f$ 个任务特定特征，承担起全部的"解耦"工作。它的重要性在稀疏模型上被放大：消融显示 MLP 层数对稠密模型几乎没影响，但对稀疏 QPM 至关重要——配上多层 MLP 后，稀疏 QPM 反而能比稠密模型高出近 10%，说明特征事先被解耦得越干净，后面极端稀疏的约束才越扛得住。

**2. 二次规划特征选择：用数学规划替代学习来挑特征。**

极端稀疏（全局只留 $N_f^* = 50$ 个特征、每类只分到 $N_f^c = 5$ 个）下，如果让特征选择也跟着梯度学，很容易选到一堆彼此冗余、又偏向全局的特征。本文转而把"挑哪些特征、分给哪个类"写成一个二次规划（QP）一次性解出来，同时优化三个目标：最大化每个类与其被分配特征激活之间的相关性（保证有判别力）、最小化被选特征两两之间的相似性（保证多样、对比性强）、并最大化偏置项以优先选局部化的特征（保证空间可定位）。这样选出的特征集天然兼顾紧凑、对比与可定位，而不是靠学习去碰运气。

**3. 平均池化实现空间定位：让每个特征都还原成一张显著性图。**

标准做法用 CLS token 聚合，但 CLS 是骨干内部预聚合出来的不透明全局表示，无法回答"模型在看哪里"。这里改用 AvgPool：特征向量 $\boldsymbol{f}$ 的每一维都是对应特征图在空间上的平均，于是反过来，任何一个特征图都能直接上采样回原始图像分辨率，当作该特征的显著性图来看。代价几乎为零——在带 register token 的 DINOv2 上，仅用 patch 嵌入就达到 88.3% 准确率，还反超了用 CLS 的 87.6%，等于把可解释性白送了。

**4. 特征图稀疏损失 $\mathcal{L}_{\text{L1-FM}}$：把激活逼回物体区域。**

即便有了空间特征图，激活也可能散落在背景上、显得杂乱。本文对特征图本身（而非只对特征向量）施加 L1 正则 $\mathcal{L}_{\text{L1-FM}}$，迫使每个特征的激活收缩到与分类真正相关的物体区域、压掉背景噪声和空间散射。一个意外收获是：这个损失不只提升了 Plausibility（解释与人类标注的吻合度），还显著提升了分类准确率——说明"逼模型只看物体"这件事本身对分类就是有益的，准确率与 Plausibility 在这里高度正相关，而非互相牺牲。

### 损失函数 / 训练策略

训练分三阶段。先做**稠密训练**：在全部 $N_f$ 个特征上用总损失训练，
$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{L1-FV}} \mathcal{L}_{\text{L1-FV}} + \lambda_{\text{L1-FM}} \mathcal{L}_{\text{L1-FM}}$$
其中 $\mathcal{L}_{\text{CE}}$ 为交叉熵，$\mathcal{L}_{\text{div}}$ 是特征多样性损失，$\mathcal{L}_{\text{L1-FV}}$ / $\mathcal{L}_{\text{L1-FM}}$ 分别对特征向量和特征图做 L1 稀疏约束。然后做一次性的**二次规划**：解上面那个 QP，得到特征选择向量 $\boldsymbol{s}$ 与稀疏权重 $\boldsymbol{W}^{\text{sparse}}$。最后**稀疏微调**：固定 $\boldsymbol{W}^{\text{sparse}}$ 的稀疏结构，只在被选中的 50 个特征上继续微调，收敛到最终的可解释分类器。

## 实验关键数据

### 主实验

| 方法 | CUB Acc↑ | CARS Acc↑ | CUB Plausib.↑ | Contrast.↑ |
|------|---------|----------|--------------|-----------|
| DINOv2 CLS 线性探测 | 87.9 | 91.7 | 42.6 | 59.2 |
| Dense $\boldsymbol{F}^{\text{froz}}$ | 78.1 | 92.9 | 32.7 | 84.5 |
| ResNet50 QPM | 82.9 | 92.1 | 82.9 | 93.6 |
| DINO-SLDD | 84.6 | 92.9 | 78.0 | 93.0 |
| DINO-QSENN | 85.4 | 93.3 | 86.0 | 94.4 |
| **DINO-QPM (本文)** | **88.3** | **94.0** | **95.0** | **100.0** |

DINO-QPM 在准确率上超越不可解释的 DINOv2 线性探测（88.3 vs 87.9），同时 Plausibility 从 42.6 跃升至 95.0。

### 消融实验

| 配置 | CUB Acc(%) | 说明 |
|------|-----------|------|
| CLS + 无 register | 87.3 | CLS token 无空间定位 |
| CLS + register | 87.6 | register 帮助 CLS |
| Patch + 无 register | 83.3 | 无 register 时 patch 表示差 |
| **Patch + register** | **88.3** | register tokens 至关重要 |

| 骨干大小 | CUB Acc(%) | Patch Contextualization↑ |
|---------|-----------|------------------------|
| DINO ViT-B/16 | 37.1 | 8.9 |
| DINOv2 ViT-S/14 Reg | 83.4 | 42.9 |
| DINOv2 ViT-B/14 Reg | 88.3 | 43.9 |
| DINOv2 ViT-L/14 Reg | 86.5 | 2.2 |

### 关键发现

1. **Register token 是关键**：没有 register 时，patch 嵌入的空间信息质量差，准确率下降约 5%。register 使 patch 不再承担全局上下文存储的"异常 token"角色
2. **Plausibility 与准确率高度正相关**：L1-FM 损失同时提升两者，说明迫使模型关注物体区域对分类本身有益
3. **紧凑性 vs 准确率的 trade-off 很小**：将每类特征从 5 减少到 4（Compact 版本），准确率几乎不变
4. **有趣的鸟类分类案例**：模型区分 Brewer's Blackbird 和 Rusty Blackbird 时自动定位到喙部，与鸟类学专家的鉴定依据完全一致

## 亮点与洞察

1. **不可解释的模型不一定更准**：DINO-QPM 用 50 个特征、每类仅 5 个特征的极端稀疏约束，反而超越了使用 768 个特征的线性探测
2. **反直觉的架构选择**：丢弃 CLS token（分类文献的标准选择）反而更好，因为直接从局部证据构建的全局表示比内部预聚合的不透明表示更可解释、也更有效
3. **训练效率极高**：由于骨干完全冻结，可预计算 patch 嵌入，每 epoch 训练仅 6 秒
4. **Plausibility 指标设计合理**：引入膨胀掩码处理 patch 边界效应，避免对精确轮廓上的激活不公平惩罚

## 局限与展望

1. 仅在细粒度分类（CUB-2011、Stanford Cars）上验证，通用图像分类场景待测试
2. ViT-L 骨干效果反而变差，可能需要针对不同骨干大小调整适配器设计
3. 二次规划特征选择是一次性的，不随训练动态调整，可能限制最优特征组合的发现
4. 目前每类固定分配 5 个特征，不同复杂度的类可能需要不同数量的特征

## 相关工作与启发

- **QPM / ChiQPM**：端到端训练的二次规划可解释模型，本文将其迁移到冻结骨干上
- **Post-hoc CBM**：通过文本概念进行可解释分类，但依赖外部语言模型且缺乏空间定位
- **ProtoViT / Zhu et al.**：基于原型的可解释方法，但需要微调骨干进行原型聚类
- 启发：冻结骨干 + 轻量适配器 = 高效可解释方案，这一范式值得在更多视觉任务上探索

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 QPM 迁移到冻结视觉基础模型上是自然但有效的创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 消融充分，指标设计严谨，跨骨干验证完整
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导清晰，可视化出色（鸟类特征定位案例尤为精彩）
- 价值: ⭐⭐⭐⭐ — 为冻结基础模型的可解释分类提供了强有力的工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](../../CVPR2025/interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[CVPR 2026\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](language_models_can_explain_visual_features_via_steering.md)
- [\[ICML 2025\] Foundation Molecular Grammar: Multi-Modal Foundation Models Induce Interpretable Molecular Grammar](../../ICML2025/interpretability/foundation_molecular_grammar_multi-modal_foundation_models_induce_interpretable_.md)

</div>

<!-- RELATED:END -->
