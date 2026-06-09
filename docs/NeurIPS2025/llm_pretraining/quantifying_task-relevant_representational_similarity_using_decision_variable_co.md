---
title: >-
  [论文解读] Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation
description: >-
  [NeurIPS 2025][预训练][决策变量相关] 本文提出基于决策变量相关（DVC）的新方法来衡量两个神经表征在分类任务上的逐试次一致性，发现深度网络在 ImageNet 上准确率越高反而与猴脑 V4/IT 的 DVC 越低，对抗训练和大规模数据集预训练也无法缩小这一差距。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "决策变量相关"
  - "脑-模型对齐"
  - "表征相似性"
  - "信号检测理论"
  - "视觉皮层"
---

# Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation

**会议**: NeurIPS 2025  
**arXiv**: [2506.02164](https://arxiv.org/abs/2506.02164)  
**代码**: [github.com/wei-bbc-lab/DVC](https://github.com/wei-bbc-lab/DVC)  
**领域**: 计算神经科学 / 表征相似性分析  
**关键词**: 决策变量相关, 脑-模型对齐, 表征相似性, 信号检测理论, 视觉皮层

## 一句话总结

本文提出基于决策变量相关（DVC）的新方法来衡量两个神经表征在分类任务上的逐试次一致性，发现深度网络在 ImageNet 上准确率越高反而与猴脑 V4/IT 的 DVC 越低，对抗训练和大规模数据集预训练也无法缩小这一差距。

## 研究背景与动机

如何比较深度网络与大脑的表征是计算神经科学的核心问题。现有方法可分为两类：

**表征相似性方法**（RSA、CKA、CCA）：衡量表征的几何结构相似性，但对任务无关的维度也敏感。

**行为一致性方法**（Cohen's Kappa、error consistency）：关注逐图像的决策一致性，但容易受准确率差异和解码器偏差的混淆。

一个引人注目的矛盾是：一些研究认为深度网络与大脑表征高度相似，另一些则认为相似性有限且已趋于饱和。这部分源于评估方法本身的局限性。

本文的动机是提出一种**原则性的、关注任务相关特征的、对准确率和决策偏差不敏感**的相似性度量，统一表征层和行为层分析的优势。

## 方法详解

### 整体框架

DVC 方法建立在信号检测理论（SDT）的推广之上：

1. 对于二元分类任务，每个观察者（大脑区域或网络层）使用一个连续的**决策变量（DV）**来做选择。
2. 对于两个观察者，可以计算它们在同一组图像上的决策变量之间的相关性——即 DVC。
3. DVC 捕获的是**两个观察者在该分类任务上的编码和解码策略的相似性**。

关键创新在于：不从行为选择数据推断 DVC，而是直接从高维神经表征中推断 DVC。

### 关键设计

**从神经表征解码决策变量**

对于每对类别（如猫 vs 狗）：
1. 使用线性判别分析（LDA）找到最大化类别分离的轴
2. 将高维表征投影到该轴上，得到每张图像的决策变量值
3. 为解决高维不稳定性问题，先用 PCA 将表征降至相同维度（25维），再使用 LDA

**噪声校正的分半程序**

测量噪声会低估 DVC。为此设计了分半校正：
- 将每个观察者的表征分为两半：$\text{DV}_{A1}, \text{DV}_{A2}$ 和 $\text{DV}_{B1}, \text{DV}_{B2}$
- 噪声校正的 Pearson 相关为：

$$\rho_{\text{corrected}} = \frac{r_{\text{cross}}}{r_{\text{self}}}$$

其中 $r_{\text{cross}}$ 是跨观察者相关的几何均值，$r_{\text{self}}$ 是观察者内分半可靠性的几何均值。这消除了独立加性对称噪声引入的衰减偏差。

**多类别扩展**

给定 $N > 2$ 个类别，对每对类别分别计算 DVC，最终报告所有类对 DVC 值的平均。

### 损失函数 / 训练策略

本文不涉及模型训练，是一种分析方法。核心评估采用 Pearson 相关系数作为 DVC 的度量。

## 实验关键数据

### 主实验

使用公开的猴脑 V4/IT 数据集（2只猴子，各100个神经元，8类物体各400张图，共3200张图像）评估了43个 ImageNet-1k 预训练模型。

| 比较对象 | 平均 DVC | 说明 |
|---------|----------|------|
| 猴子-猴子（V4+IT） | 0.57 | 跨猴脑一致性高 |
| 猴子-猴子（仅V4） | 0.63 | V4 区域一致性最高 |
| 猴子-猴子（仅IT） | 0.41 | IT 区域一致性略低 |
| 模型-猴子（43个模型平均） | 0.29 ± 0.05 | 显著低于猴子间 |
| 对抗训练模型-猴子（9个模型） | 0.27 ± 0.02 | 略有下降 |
| 大数据集模型-猴子（13个模型） | 0.24 ± 0.05 | 进一步下降 |

**模型准确率 vs DVC 与猴脑的关系**：

| Pearson 相关 | p 值 | 结论 |
|-------------|------|------|
| -0.70 | 2.28e-07 | ImageNet 准确率越高，与 V4/IT 的 DVC 越低 |

### 消融实验

**模型-模型 DVC 分析**

| 比较类型 | DVC | p 值 |
|---------|-----|------|
| 同族模型对 | 显著更高 | 1.33e-56 |
| 不同族模型对 | 较低 | — |
| 对抗训练模型之间 | 0.69 ± 0.09 | 高度一致 |
| 对抗训练 vs 标准模型 | 显著降低 | 5.20e-37 |

**与 Cohen's Kappa 的对比分析**

| 度量 | 模型-猴脑 | 模型-模型 | 猴子-猴子 |
|------|---------|---------|---------|
| DVC（本文方法） | 0.29 ± 0.05 | 适中，不极端高 | 0.57 |
| Cohen's Kappa（交叉验证LR） | 0.13 ± 0.04 | 0.23 ± 0.07 | 0.22 |
| Cohen's Kappa（原始解码器） | 极低 | 极高 | — |

Cohen's Kappa 的高模型间一致性源于解码器引入的决策偏差。当替换为交叉验证逻辑回归后，Cohen's Kappa 结果与 DVC 基本一致。

### 关键发现

1. **负相关现象**：ImageNet 准确率越高的网络，与猴脑 V4/IT 在任务相关维度上的一致性越低（$r = -0.70$）。这与 BrainScore 等早期报告的正相关趋势相矛盾。
2. **对抗训练悖论**：对抗训练使模型之间高度一致（DVC = 0.69），但与大脑的一致性并未提升，甚至略有下降。
3. **大数据集无帮助**：ImageNet-21k、JFT-300M 等大规模预训练模型的 DVC 更低（0.24），而非更高。
4. **Cohen's Kappa 的陷阱**：该度量对准确率差异和决策偏差高度敏感，可能误导研究者认为模型间高度一致而模型-大脑间极低。

## 亮点与洞察

- DVC 是一种原则性的度量，在信号检测理论框架下自然地解耦了准确率和逐试次一致性，这是现有方法（RSA、Cohen's Kappa）难以做到的。
- 分半噪声校正方法简洁有效，无需对噪声结构做过强假设。
- 实验结果挑战了"更好的网络 = 更像大脑"的流行假说，揭示了深度网络和灵长类视觉表征之间可能存在的**根本性策略差异**。
- 对 Cohen's Kappa 文献的系统性再分析是方法论贡献的重要组成部分。

## 局限与展望

1. 仅使用两只猴子的数据，样本量有限，后续需要更大规模的神经记录数据验证。
2. 高维特征空间的降维可能丢失信息，尽管作者做了多种维度设置的鲁棒性验证。
3. 猴子与人类的视觉行为可能不同，结论向人类推广需谨慎。
4. 噪声校正假设加性对称噪声，对更一般噪声条件的 DVC 恢复仍是开放问题。
5. 实验仅涉及物体分类任务，未涵盖更复杂的视觉任务。

## 相关工作与启发

- **BrainScore**（Schrimpf et al.）：报告准确率与脑对齐正相关但趋于饱和，DVC 揭示了不同的趋势。
- **Error Consistency**（Geirhos et al.）：基于 Cohen's Kappa 的行为一致性分析，本文指出其受解码器偏差影响。
- **Platonic Representation Hypothesis**（Huh et al.）：认为深度学习系统趋向学习共同表征，但本文的 DVC 结果更为微妙。
- 未来方向：结合行为层和表征层的 DVC 分析；使用生态学相关数据集训练；在训练中引入内部噪声模型。

## 评分

- **创新性**: ★★★★☆（将 SDT 推广到高维表征比较，理论动机清晰）
- **实验设计**: ★★★★★（多层面分析、与 Cohen's Kappa 的系统对比、鲁棒性验证充分）
- **实用性**: ★★★★☆（开源代码，可直接应用于脑-模型对齐研究）
- **发现的冲击力**: ★★★★★（负相关、对抗训练悖论等发现颠覆常识）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](../../ICML2025/llm_pretraining/dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)
- [\[ICML 2025\] When Can In-Context Learning Generalize Out of Task Distribution?](../../ICML2025/llm_pretraining/when_can_in-context_learning_generalize_out_of_task_distribution.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)

</div>

<!-- RELATED:END -->
