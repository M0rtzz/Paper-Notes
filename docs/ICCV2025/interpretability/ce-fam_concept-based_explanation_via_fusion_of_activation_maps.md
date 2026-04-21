---
title: >-
  [论文解读] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps
description: >-
  [ICCV 2025][概念解释] 提出CE-FAM概念解释方法，通过训练与图像分类器共享激活图的分支网络来模拟VLM嵌入，实现概念预测→概念区域（激活图加权和）→概念贡献（对分类分数影响）的一一对应，并提出新的NRA评估指标，在零样本概念推理上超越现有方法。
tags:
  - ICCV 2025
  - 概念解释
  - 激活图融合
  - Grad-CAM
  - VLM知识迁移
  - 可解释性
---

# CE-FAM: Concept-Based Explanation via Fusion of Activation Maps

**会议**: ICCV 2025  
**arXiv**: [2509.23849](https://arxiv.org/abs/2509.23849)  
**代码**: 无  
**领域**: 多模态VLM / 可解释AI  
**关键词**: 概念解释, 激活图融合, Grad-CAM, VLM知识迁移, 可解释性

## 一句话总结

提出CE-FAM概念解释方法，通过训练与图像分类器共享激活图的分支网络来模拟VLM嵌入，实现概念预测→概念区域（激活图加权和）→概念贡献（对分类分数影响）的一一对应，并提出新的NRA评估指标，在零样本概念推理上超越现有方法。

## 研究背景与动机

可解释AI (XAI)中，现有方法存在三个层级的解释需求，但鲜有方法同时满足：

**是什么概念 (What)**：模型学到了哪些人类可理解的概念？

**在哪里 (Where)**：这些概念对应图像中哪些区域？

**如何贡献 (How)**：每个概念对最终预测的贡献有多大？

- **显著性图方法**（Grad-CAM等）：只能高亮重要区域，解释什么让给用户自行理解
- **概念瓶颈模型**（CBM、TCAV）：能量化概念贡献，但无法定位概念区域
- **Dissection方法**（CLIP-Dissect、WWW）：将概念与单个神经元关联，但遇到多对多映射问题——一个概念可能与多个神经元相关，且最优对应因样本而异

核心洞察：用单个激活图表示概念有本质局限，应通过**激活图的加权融合**来表示概念区域。

## 方法详解

### 整体框架

CE-FAM的工作流程：
1. 训练阶段：分支网络学习将分类器的多层激活图嵌入映射到VLM（CLIP）的图像嵌入空间
2. 概念预测：将映射后的嵌入与概念文本嵌入计算相似度
3. 概念区域：对概念预测分数反向传播，获得激活图权重，生成概念区域图
4. 概念贡献：通过遮蔽重要通道测量分类分数的下降量

### 关键设计

1. **多层概念学习 (Multi-Layer Concept Learning)**

   与仅使用最后一层嵌入的传统方法不同，CE-FAM利用CNN各层的嵌入向量捕获从低级到高级的全部特征：

   $$\mathbf{z}^l = \text{AvgPool}(A^l)$$
   $$\mathbf{z}_{\text{cat}} = \text{Concat}(\mathbf{z}^1, \mathbf{z}^2, \ldots, \mathbf{z}^L)$$

   通过translator函数 $h$（简单MLP）将 $\mathbf{z}_{\text{cat}}$ 投影到CLIP嵌入空间。训练损失：

   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{emb}} + \lambda \mathcal{L}_{\text{sim}}$$

   其中 $\mathcal{L}_{\text{emb}} = \text{MSE}(h(\mathbf{z}_{\text{cat}}) - E_{\text{image}}(\mathbf{x}))$ 模拟VLM嵌入，$\mathcal{L}_{\text{sim}} = \text{MSE}(S^t - S_{\text{VLM}}^t)$ 匹配概念预测分数。

   关键发现：即使不使用 $\mathcal{L}_{\text{sim}}$（即不需要预定义概念集），性能仍超过现有方法——这意味着方法支持对概念标签的**零样本推理**。

2. **概念区域表示 (Concept Region via Activation Map Fusion)**

   扩展Grad-CAM思想，对每个概念 $t$ 在每层 $l$ 生成区域图：

   $$R_t^l = \text{ReLU}\left(\sum_k \beta_k^t A_k^l\right)$$
   $$\beta_k^t = \frac{1}{\gamma} \sum_i \sum_j \frac{\partial S^t}{\partial A_k^l(i,j)}$$

   用概念预测分数 $S^t$ 的梯度作为权重（而非类别预测分数 $p^c$），生成概念特异的区域图。

   **层选择**：通过关联度评分选择最相关层——逐个遮蔽Top-K重要通道，测量 $S^t$ 下降量的AUC，AUC最大的层最相关。

3. **概念贡献量化 (Concept Contribution)**

   不限于最后一层，可量化任意层的概念贡献：
   - 计算方式与关联度评分相同，但目标改为类别预测分数 $p^c$
   - 逐步遮蔽最重要通道，记录 $p^c$ 的下降曲线AUC
   - 负贡献（遮蔽后分数上升）也有意义，表示该概念对预测有负面影响

4. **NRA评估指标 (Normalized Region Accuracy)**

   现有指标的问题：IoU受阈值选择影响，VEA (AUC)在大mask区域时即使随机结果也会得高分。

   NRA通过归一化消除样本分布影响：
   $$\text{NRA} = \frac{\text{AUC} - \text{AUC}_{\text{low}}}{\text{AUC}_{\text{high}} - \text{AUC}_{\text{low}}}$$

   其中 $\text{AUC}_{\text{high}}$ 为理想（GT分割）下的AUC，$\text{AUC}_{\text{low}}$ 为随机区域的AUC。NRA衡量预测区域在随机和理想之间的相对位置。

### 损失函数 / 训练策略

- 数据集：ImageNet用于训练概念学习
- 优化器：SGD，初始LR 0.1，warmup到0.2
- 最大150 epochs + early stopping（验证loss连续4 epoch不降则LR×0.1）
- $\lambda = 0.001$

## 实验关键数据

### 主实验

Broden数据集上概念区域评估（ResNet50分类器 + CLIP VLM）：

| 方法 | EPG(Object) | EPG(Avg) | NRA(Object) | NRA(Avg) | Hit Rate(Object) | Hit Rate(Avg) |
|------|-------------|----------|-------------|----------|-------------------|---------------|
| CLIP-Dissect | 0.197 | 0.146 | 0.327 | 0.334 | 0.215 | 0.199 |
| WWW | 0.179 | 0.117 | 0.322 | 0.278 | 0.154 | 0.114 |
| **CE-FAM** | **0.233** | **0.154** | **0.459** | **0.361** | **0.436** | **0.247** |

CE-FAM在NRA上比最佳基线高8%，Hit Rate（NRA>0.5的比例）高13%。

ImageNet-S数据集上（ViT-B/16分类器）：

| 方法 | EPG | NRA | Hit Rate |
|------|-----|-----|----------|
| CLIP-Dissect | 0.047 | 0.105 | 0.013 |
| WWW | 0.076 | 0.232 | 0.017 |
| **CE-FAM** | **0.138** | **0.273** | **0.193** |

在ViT上优势尤为明显，因为单通道激活图在ViT中噪声更大。

### 消融实验

不同条件下概念区域评估（ResNet50）：

| VLM | Sim Loss | 多层 | EPG Avg | NRA Avg | Hit Rate Avg |
|------|----------|------|---------|---------|--------------|
| CLIP | - | - | 0.152 | 0.338 | 0.209 |
| CLIP | ✓ | - | 0.156 | 0.348 | 0.234 |
| CLIP | ✓ | ✓ | 0.154 | 0.361 | 0.247 |
| SigLIP | - | - | 0.151 | 0.383 | 0.283 |
| SigLIP | ✓ | ✓ | **0.157** | **0.388** | **0.295** |

### 关键发现

- **激活图融合远优于单通道关联**：Hit Rate从11.4%（WWW）提升到24.7%，说明概念不应由单个神经元表示
- **多层特征互补**：使用多层嵌入比仅用最后一层提高NRA 2.3%和Hit Rate 3.8%，低级特征对颜色等概念重要
- **零样本概念推理有效**：不使用 $\mathcal{L}_{\text{sim}}$ 时仍超越需要概念标签的现有方法
- **VLM选择影响显著**：SigLIP比CLIP表现更好，2D表示更优秀的VLM将直接提升概念学习效果
- **训练效率高**：仅需几个epoch就超越现有方法的性能
- **误分类分析**：通过概念贡献可解释误分类原因（如将indigo bunting误判为goldfinch因yellow概念有负贡献）

## 亮点与洞察

1. **首次建立概念标签-区域-贡献的一一对应**：完成了概念解释的"What-Where-How"三位一体
2. **通用框架**：适用于任何使用激活图的图像分类器（CNN和ViT均可）
3. **NRA指标设计合理**：通过归一化消除了样本分布对评价的影响
4. **零样本能力**：无需概念标注数据集，仅依赖VLM知识即可处理任意概念

## 局限与展望

- 概念表达能力受限于VLM的性能：CLIP倾向于被图像中突出特征主导，难以预测细粒度概念
- 概念集选择敏感：过大的概念集会引入无关概念噪声
- 概念贡献缺乏定量评估的ground truth，验证困难
- 方法的计算成本随层数和概念数增加，规模化应用需优化

## 相关工作与启发

- 将Grad-CAM从"类别解释"扩展到"概念解释"的思路自然优雅
- "概念=激活图加权融合"是对Net2Vec多通道表示理念的有效延续
- NRA指标可推广到其他需要评估区域准确度的XAI任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 概念区域通过激活图融合表示，一一对应框架新颖
- **实验充分度**: ⭐⭐⭐⭐ Broden和ImageNet-S两数据集+多分类器+多VLM+消融+定性分析
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，评估指标设计有理有据
- **实用价值**: ⭐⭐⭐⭐ 框架通用，可用于模型调试和误分类诊断

<!-- RELATED:START -->

## 相关论文

- [Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](../../ACL2025/interpretability/separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)
- [When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)
- [Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)

<!-- RELATED:END -->
