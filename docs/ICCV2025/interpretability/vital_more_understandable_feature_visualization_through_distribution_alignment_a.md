---
title: >-
  [论文解读] VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow
description: >-
  [ICCV 2025][特征可视化] 提出VITAL方法，通过将特征可视化重新定义为真实图像特征分布对齐问题（而非传统的激活最大化），并结合相关性评分过滤无关特征，生成对人类更易理解的神经元可视化结果。
tags:
  - ICCV 2025
  - 特征可视化
  - 可解释AI
  - 分布匹配
  - 层级相关传播
  - 机械可解释性
---

# VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow

**会议**: ICCV 2025  
**arXiv**: [2503.22399](https://arxiv.org/abs/2503.22399)  
**代码**: [GitHub](https://github.com/VITAL-Framework)  
**领域**: 其他  
**关键词**: 特征可视化, 可解释AI, 分布匹配, 层级相关传播, 机械可解释性

## 一句话总结

提出VITAL方法，通过将特征可视化重新定义为真实图像特征分布对齐问题（而非传统的激活最大化），并结合相关性评分过滤无关特征，生成对人类更易理解的神经元可视化结果。

## 研究背景与动机

特征可视化（Feature Visualization, FV）是理解神经网络内部学到什么信息的重要工具，通过生成能强烈激活特定神经元的图像来解释网络行为。在安全关键领域（如医疗），理解网络的决策过程尤为重要。

然而，现有FV方法存在严重的**可理解性问题**：

**重复模式(Repetitive Patterns)**: 激活最大化倾向于在图像中重复出现相同模式以反复刺激目标神经元，导致"像万花筒一样"的视觉效果

**伪影(Artifacts)**: 生成的图像包含不自然的颜色、纹理等伪影

**无关特征(Irrelevant Features)**: 可视化中混入了与目标神经元无关的背景特征（如鸟喙检测器的可视化中出现草地）

这些问题在现代大型架构（ResNet-50、ViT等）上尤为严重，严重制约了FV作为理解工具的实用价值。

核心洞察：**激活最大化本身就是问题的根源**——它鼓励任何能提高激活值的模式（包括不自然的、重复的），而不关心生成的图像是否自然。如果改为要求生成图像的中间层特征分布与真实图像一致，就自然地抑制了重复模式（因为真实图像中不会出现极端重复）和伪影（因为它们偏离了真实数据流形）。

## 方法详解

### 整体框架

给定目标神经元 $f_i^{(l)}$，VITAL不再寻找最大化其激活的图像，而是寻找一张图像 $x^*$，使其在网络各层的特征分布与参考真实图像 $x' \in \mathcal{X}_{ref}$ 的特征分布对齐：

$$\forall l' < l, x' \in \mathcal{X}_{ref}: \text{dist}(A^{(l')}(x^*)) \approx \text{dist}(A^{(l')}(x'))$$

其中 $A^{(l')}(x) = f^{(l')}(x) \in \mathbb{R}^{C_l \times D}$ 是第 $l'$ 层的激活，$D$ 为展平后的空间维度。

### 关键设计

1. **Sort-Matching分布匹配**: 关键难题在于如何高效地匹配两个经验分布并支持反向传播。VITAL采用sort-matching方法：计算排序索引 $\pi$ 和 $\pi'$ 分别对生成图像和参考图像的特征向量排序，然后通过反排序索引 $\bar{\pi}$ 对齐两个排序后的分布，计算MSE损失：

$$\text{MSE}(z, z^r) = \frac{1}{|z|} \sum_{i=1}^{|z|} (z_i - z_i^r)^2$$

由于 $z$ 是 $x^*$ 的函数，可以通过该损失反向传播优化 $x^*$。对多个参考图像，先平均其排序后的特征向量作为代表原型。这种方法的巧妙之处在于：排序操作虽不可微，但排序索引是离散的固定映射，真正参与梯度计算的是连续的特征值。

2. **相关性评分融合（LRP加权）**: 仅做分布匹配仍会引入无关特征（如鸟类图像中的草地背景）。VITAL引入层级相关传播(Layer-wise Relevance Propagation, LRP)来判断每个中间特征对目标神经元的**相关程度**，然后用加权后的激活进行分布匹配：

$$A^{(l')}(x) \odot R_n^{(l')}(x)$$

其中 $R_n$ 是目标神经元 $n$ 的LRP相关性评分，$\odot$ 为Hadamard积。这确保只有与目标神经元真正相关的特征参与分布匹配，有效消除了"共激活但不相关"的背景特征。

3. **参考图像选择**: 对类别神经元，直接使用该类别的随机训练图像。对中间神经元，选取激活最高的Top-k图像patch（来自不同图像），裁剪并调整大小后作为参考集 $\mathcal{X}_{ref}$。

4. **透明度图与辅助正则化**: 使用梯度累积生成透明度图，仅展示优化过程中被网络关注的区域。同时施加辅助正则化：

$$\mathcal{L}_{\text{VITAL}}(x^*, \mathcal{X}_{ref}) = \mathcal{L}_{\text{SM}}(x^*, \mathcal{X}_{ref}) + \alpha_{\text{TV}} \mathcal{L}_{\text{TV}}(x^*) + \alpha_{\ell_2} \ell_2(x^*)$$

### 损失函数 / 训练策略

- 主损失：Sort-matching分布匹配损失（跨多层累加）
- 辅助损失：Total Variance (TV) 正则化 + $\ell_2$ 范数正则化
- 层选择：经消融验证，仅对齐ResNet50的第一个和最后一个block输出即可获得高质量图像
- 运行时间：约40秒/图像（含参考图像分布计算），与MACO的23-28秒和DeepInversion的1-3分钟可比

## 实验关键数据

### 主实验

在ImageNet预训练模型上的定量评估（分类准确率、FID、CLIP零样本预测）：

| 方法 | 架构 | Acc.↑ | FID↓ | CLIP Top1↑ | CLIP Top5↑ |
|------|------|-------|------|-----------|-----------|
| MACO | ResNet50 | 29.43 | 360.74 | 12.87 | 29.73 |
| DeepInv | ResNet50 | 100.00 | 35.76 | 29.90 | 55.20 |
| **VITAL** | **ResNet50** | **99.90** | **58.79** | **66.62** | **92.56** |
| MACO | ViT-L-16 | 44.33 | 946.96 | 3.93 | 10.57 |
| **VITAL** | **ViT-L-16** | **99.80** | **126.29** | **68.17** | **92.80** |
| MACO | ConvNeXt | 66.07 | 62.55 | 7.20 | 19.77 |
| **VITAL** | **ConvNeXt** | **99.97** | **3.92** | **63.53** | **90.30** |

### 消融实验

人类用户研究（58名参与者，三部分评估）：

| 评估任务 | MACO | Fourier | DeepInv | VITAL | 说明 |
|---------|------|---------|---------|-------|------|
| (a) 类可视化+类别名(1-5分) | ~2.0中位数 | ~1.5 | ~3.0 | **~4.0** | 给定类名评分可视化质量 |
| (b) 内部神经元(1-5分) | ~2.5 | ~2.0 | N/A | **~4.0** | 评价FV与参考图像的匹配度 |
| (c) 自由标注(相似度) | ~0.35中位 | ~0.30 | ~0.40 | **~0.60** | 无提示下标注FV内容 |

### 关键发现

- VITAL在CLIP零样本预测上接近真实图像的准确率（ResNet50上66.62% vs 69.11%），远超所有基线
- 在ConvNeXt-base上FID仅3.92，比MACO的62.55低一个数量级
- 人类用户研究中VITAL在"高分"（4-5分）可视化数量上比其他方法多出一个数量级
- tSNE嵌入分析表明VITAL生成的图像落在真实图像聚类的中心，而其他方法要么聚集在远离真实数据的点上，要么偏离聚类中心
- 对ViT架构的泛化能力明显优于其他方法

## 亮点与洞察

- **范式转换**：从"最大化激活"到"对齐分布"，是特征可视化领域的方法论创新
- **LRP加权的分布匹配**精确解决了"共激活≠相关"的问题，是理论和实践上的双重贡献
- 方法对架构完全不可知，能无缝扩展到CNN和ViT
- 小电路可视化（Small Circuits）展示了VITAL在机械可解释性中的应用潜力——为电路分析中的"where"补充了"what"

## 局限与展望

- 生成图像仍达不到照片级真实感，更像"莫奈风格的画作"
- 对复杂空间排列的可视化仍有困难
- 对中间神经元的可视化较慢（2-3分钟），因为需要额外的LRP反向传播
- 参考图像的选择策略可能引入偏差，值得进一步研究鲁棒性
- 可解释性评估缺乏标准化基准，人类研究的设计和规模有待完善

## 相关工作与启发

- 在机械可解释性(MI)框架下，VITAL补充了电路分析的"what"维度
- DeepInversion也利用特征统计但以batch norm统计量为目标，VITAL直接匹配经验分布更灵活
- MACO在频域操作，VITAL在特征域操作，两者思路正交
- 可启发扩展到CLIP等多模态模型的可视化分析

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 从激活最大化到分布对齐的范式转换，LRP融合的设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ 五种架构、多种定量指标、两项人类用户研究
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，方法阐述严谨，图示丰富
- **价值**: ⭐⭐⭐⭐⭐ 对可解释AI社区有重要影响，开辟了特征可视化的新方向

<!-- RELATED:START -->

## 相关论文

- [Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [Unsupervised Feature Selection Through Group Discovery](../../AAAI2026/interpretability/unsupervised_feature_selection_through_group_discovery.md)
- [ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](../../NeurIPS2025/interpretability/out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)
- [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)

<!-- RELATED:END -->
