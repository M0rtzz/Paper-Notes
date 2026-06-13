---
title: >-
  [论文解读] Training the Untrainable: Introducing Inductive Bias via Representational Alignment
description: >-
  [NeurIPS 2025][归纳偏置] 提出Guidance方法，通过逐层表征对齐（CKA）将一个网络（guide）的架构归纳偏置迁移到另一个原本"不可训练"的网络（target），从而使FCN能做图像分类、RNN逼近Transformer的语言建模性能。 神经网络的架构选择至关重要——CNN革新了视觉…
tags:
  - "NeurIPS 2025"
  - "归纳偏置"
  - "表征对齐"
  - "CKA"
  - "架构先验"
  - "知识蒸馏"
---

# Training the Untrainable: Introducing Inductive Bias via Representational Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2410.20035](https://arxiv.org/abs/2410.20035)  
**代码**: [GitHub](https://untrainable-networks.github.io)  
**领域**: 深度学习 / 架构归纳偏置  
**关键词**: 归纳偏置, 表征对齐, CKA, 架构先验, 知识蒸馏

## 一句话总结

提出Guidance方法，通过逐层表征对齐（CKA）将一个网络（guide）的架构归纳偏置迁移到另一个原本"不可训练"的网络（target），从而使FCN能做图像分类、RNN逼近Transformer的语言建模性能。

## 研究背景与动机

神经网络的架构选择至关重要——CNN革新了视觉，Transformer重塑了NLP。然而，**架构设计本质上是一门"黑魔法"**：我们很少真正理解不同架构编码了怎样的归纳偏置。例如，残差连接的确切作用至今仍有争论。

这导致了一些实际问题：
- **FCN在图像分类上立即过拟合**：缺乏局部感受野等空间先验
- **深层CNN（无残差连接）梯度消失**：无法有效训练
- **RNN在长序列任务上性能饱和**：受限于梯度消失和有限的上下文整合能力
- **Transformer在需要全序列推理的形式语言任务上失败**：如奇偶性判断

传统答案是"换一个更好的架构"，但这依赖于对归纳偏置的深入理解。作者提出了一个根本性问题：**能否在不改变架构的情况下，将一个架构的归纳偏置"注入"到另一个架构中？**

近期研究发现，不同架构的网络在内部表征上惊人地相似（Han et al.），这暗示了表征层面的偏置迁移可能是可行的。

## 方法详解

### 整体框架

Guidance方法在目标网络（target）的原始损失之上增加一个逐层表征对齐项，使target的中间层激活向固定的引导网络（guide）看齐。Guide可以是已训练的（迁移架构先验+知识）或随机初始化的（仅迁移架构先验）。

总损失函数为：

$$\mathcal{L}(\theta^T) = \mathcal{L}_T(\theta^T) + \sum_{i \in I} \bar{\mathcal{M}}(\mathbf{A}_{i^T}^T(\theta^T), \mathbf{A}_{i^G}^G(\theta^G))$$

其中 $\bar{\mathcal{M}}$ 是表征不相似度量（CKA的补），$I$ 是guide-target层对应关系，$\theta^G$ 始终冻结。

### 关键设计

1. **CKA作为表征距离函数**：选用线性Centered Kernel Alignment（CKA），它基于二阶统计量（样本间的成对距离矩阵），能捕获架构特有的结构信息。例如，CNN的局部感受野导致相邻像素对应的unit有相关输入，这种相关性反映在距离矩阵中，可以通过CKA迁移到FCN的没有局部相关性的层中。权重共享（同一卷积核在所有空间位置应用）也会被类似地编码和迁移。

2. **逐层映射策略**：将guide的层均匀分布到target的层上。例如，ResNet-18有18个卷积层，50层FCN则每隔2-3个线性层对应一个ResNet层。对于stacked RNN/Transformer，从堆叠的中间层提取特征。实验发现使用更多层对应会产生更强的对齐信号。

3. **Guide可以是未训练的**：这是最令人惊讶的发现——随机初始化的guide（完全不能完成任务）也能带来显著性能提升，证明**架构本身就编码了有用的归纳偏置**，独立于学到的参数。

### 训练策略

- 所有网络使用交叉熵损失，Adam/AdamW优化器
- 固定batch size 256（CKA计算受样本数影响）
- 仔细调参基线学习率（5个值sweep），选最低验证损失
- 每个设置训练100 epochs，5个随机种子计算误差棒

## 实验关键数据

### 主实验：序列建模

| 实验 | Copy-Paste准确率↑ | 奇偶性准确率↑ | 语言建模(小)困惑度↓ | 语言建模(大)困惑度↓ |
|---|---|---|---|---|
| RNN (基线) | 14.35±0.01 | 100 | 69.19±1.89 | 89.13±2.00 |
| Transformer (基线) | 96.98 | 71.98±3.16 | 34.15 | 33.10 |
| Transformer→RNN | **23.27±1.02** | — | **40.01±1.54** | **36.91±1.04** |
| 未训练Transformer→RNN | **42.56±1.51** | — | 59.61±2.33 | 47.17±2.50 |
| RNN→Transformer | — | **78.49±2.16** | — | — |

### 主实验：图像分类（ImageNet Top-5）

| 实验 | 验证准确率↑ |
|---|---|
| Deep FCN (基线) | 1.65±1.21 |
| ResNet-18→Deep FCN | 7.50±1.51 |
| 未训练ResNet-18→Deep FCN | **13.10±0.72** |
| Wide FCN (基线) | 34.09±0.91 |
| ResNet-18→Wide FCN | **43.01±0.92** |
| Deep ConvNet (基线) | 70.02±1.52 |
| ResNet-50→Deep ConvNet | **78.91±2.16** |

### 消融实验

| 消融配置 | 关键指标 | 说明 |
|---|---|---|
| Guidance仅做初始化(300步) | 性能与持续guidance相当 | 暗示存在更好的FCN初始化方案 |
| Guidance vs 蒸馏 | Guidance全面优于蒸馏 | 中间层对齐远强于输出对齐 |
| 已训练guide vs 随机guide | 多数情况下随机guide更优 | 架构先验比学到的知识更关键 |
| Error consistency分析 | Guided FCN继承guide的决策模式 | ResNet-guided FCN与ViT-guided FCN的错误一致性关系复制了ResNet与ViT的关系 |

### 关键发现

- **未训练guide常优于已训练guide**：在Copy-Paste、Deep FCN等任务上，随机初始化的guide表现更好。这证明架构本身（而非学到的权重）是性能提升的主要来源
- **RNN+Guidance在语言建模上大幅缩小与Transformer的差距**：困惑度从~70降至35-40，接近Transformer的34
- **Guidance-only初始化即可防止过拟合**：仅用300步对齐随机ResNet，之后正常训练FCN，过拟合完全消失
- **Deep ConvNet（无残差）仅在已训练ResNet guide下获益**：暗示残差连接必须先训练才能影响表征空间

## 亮点与洞察

- **架构先验可以与训练先验分离**：通过对比已训练和未训练guide的效果，提供了研究不同架构编码何种偏置的实证工具
- **Guidance本质是信用分配的辅助**：逐层对齐帮助梯度下降更好地调整深层网络早期层的权重
- **错误一致性实验**特别精彩：guided FCN不是泛泛地变好，而是继承了guide的特定决策风格
- 挑战了"某些架构天然不适合某些任务"的传统观念

## 局限与展望

- **覆盖广度优先于单一任务最佳性能**：没有在任何一个任务上做到SOTA，超参调优和长时间训练可能进一步提升
- **仅使用CKA一种距离函数**：其他表征距离（如CCA, RSA）可能有不同效果
- **层对应策略简单**：均匀分布映射可能不是最优的，更复杂的网络对可能需要更精细的映射
- **计算开销**：每个minibatch需要在guide和target上前向传播并计算CKA，增加了训练成本
- **FCN虽然不再过拟合但绝对性能仍低**：需要更多工作才能使FCN在图像分类上真正实用

## 相关工作与启发

- 与知识蒸馏的区别：蒸馏匹配输出logit，guidance匹配中间层表征；蒸馏需要训练好的teacher，guidance可用随机guide
- 与NAS/架构搜索互补：guidance提供了一种"软"方式来注入架构偏置，而非硬编码
- 对理解大模型（如inference-time scaling）的架构选择有潜在启示

## 评分

- 新颖性: ⭐⭐⭐⭐ 用随机网络的表征对齐来迁移架构先验是一个新颖且反直觉的发现
- 实验充分度: ⭐⭐⭐⭐ 涵盖图像分类、序列建模4个任务，多种架构组合，但每个都没有做到最深入
- 写作质量: ⭐⭐⭐⭐ 实验设计系统，但限于覆盖面过广，每个实验的分析深度有限
- 价值: ⭐⭐⭐⭐ 提供了研究架构归纳偏置的有力实证工具，但FCN的实际效用仍待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] From Lists to Emojis: How Format Bias Affects Model Alignment](../../ACL2025/others/from_lists_to_emojis_how_format_bias_affects_model_alignment.md)
- [\[ACL 2025\] Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](../../ACL2025/others/aligned_but_blind_implicit_bias.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Graph Alignment via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)

</div>

<!-- RELATED:END -->
