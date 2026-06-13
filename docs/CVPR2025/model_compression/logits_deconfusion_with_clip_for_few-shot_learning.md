---
title: >-
  [论文解读] Logits DeConfusion with CLIP for Few-Shot Learning
description: >-
  [CVPR 2025][模型压缩][few-shot learning] 发现 CLIP 在下游任务中 logits 存在严重的类间混淆问题，提出 Logits DeConfusion（LDC）方法，通过多层级 Adapter 融合（MAF）增强特征表示，结合类间去混淆模块（ICD）以残差结构学习并消除混淆模式，在 11 个基准上取得 SOTA。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "few-shot learning"
  - "CLIP"
  - "logits deconfusion"
  - "inter-class confusion"
  - "adapter fusion"
  - "residual learning"
---

# Logits DeConfusion with CLIP for Few-Shot Learning

**会议**: CVPR 2025  
**arXiv**: [2504.12104](https://arxiv.org/abs/2504.12104)  
**代码**: [LiShuo1001/LDC](https://github.com/LiShuo1001/LDC)  
**领域**: 模型压缩  
**关键词**: few-shot learning, CLIP, logits deconfusion, inter-class confusion, adapter fusion, residual learning

## 一句话总结

发现 CLIP 在下游任务中 logits 存在严重的类间混淆问题，提出 Logits DeConfusion（LDC）方法，通过多层级 Adapter 融合（MAF）增强特征表示，结合类间去混淆模块（ICD）以残差结构学习并消除混淆模式，在 11 个基准上取得 SOTA。

## 研究背景与动机

**领域现状**: CLIP 通过大规模图文对比学习建立了强大的视觉语言对齐能力，在 zero-shot 和 few-shot 学习中表现出色。后续方法如 CoOp、Tip-Adapter 等通过 prompt 学习或 adapter 提升适应性。

**现有痛点**: CLIP 的预训练策略是对比学习而非直接优化分类边界，因此在下游任务中 logits 存在显著的类间混淆——不同类别的预测值难以准确区分。当类别相似度高或域差异大时问题尤为严重。

**核心矛盾**: CLIP 的预训练数据分布与下游任务的域差异导致分类边界模糊，且少样本设置下难以学习可靠分类器。

**本文切入角度**: 不修改 CLIP 的特征表示，而是直接在 logits 层面建模和消除混淆模式。

**核心 idea**: 将类间混淆视为可学习的噪声项 $\Delta s$，通过残差结构 $\hat{s}_i = s_i^{ZS} - \Delta s(x_i)$ 消除。

## 方法详解

### 整体框架

1. **ZS-CLIP**: 冻结的 CLIP 生成 zero-shot logits $s_i^{ZS}$
2. **MAF（多层级 Adapter 融合）**: 从图像编码器 4 个层级提取特征，融合为增强特征 $z_i^e$，再通过 MLP 生成 MAF logits $s_i^{MAF}$
3. **ICD（类间去混淆）**: 利用 $z_i^e$ 作为先验，从 $s_i^{ZS}$ 中学习混淆模式并以残差方式消除，输出 ICD logits $s_i^{ICD}$
4. **ALF（自适应 Logits 融合）**: 自适应权重 $\alpha$ 加权融合 MAF 和 ICD logits 得到最终预测

### 关键设计

**1. 多层级 Adapter 融合（MAF）**
- **功能**: 从 CLIP 图像编码器的 4 个不同层级提取特征 $f_i^1, f_i^2, f_i^3, f_i^4$，分别经过独立 Adapter 变换后融合为统一表示 $z_i^e$。
- **核心思路**: 提供两种融合机制——加权融合（WF: 权重 0.1:0.2:0.3:0.4）和可学习融合（LF: 拼接后 Adapter 降维）。融合特征经冻结的 Projector（ResNet 用 attention pooling，ViT 用线性投影）得到增强特征。
- **设计动机**: 低层特征包含细节信息，高层特征包含语义信息；多层融合使得在少样本情况下获得更全面的特征表示，提升泛化能力。

**2. 类间去混淆模块（ICD）**
- **功能**: 通过三个级联 Adapter 学习 logits 中的类间混淆模式并用残差消除: $s_i^{ICD} = s_i^{ZS} + \mathcal{E}_{A_3}^{ICD}(\mathcal{E}_{A_1}^{ICD}(s_i^{ZS}) + \mathcal{E}_{A_2}^{ICD}(z_i^e))$。
- **核心思路**: $\mathcal{E}_{A_1}$ 从 zero-shot logits 中提取混淆线索，$\mathcal{E}_{A_2}$ 从增强视觉特征中提取混淆先验，两路相加后经 $\mathcal{E}_{A_3}$ 联合学习混淆模式，最终残差结构移除混淆。
- **设计动机**: 实验观察发现 CLIP 对每个类别存在固定的类间混淆模式；视觉特征提供"应当是什么类别"的先验信息，指导混淆模式的精确学习。

**3. 自适应 Logits 融合（ALF）**
- **功能**: 使用 $\alpha$ Generator 从增强特征 $z_i^e$ 生成自适应权重 $\alpha$，融合两路 logits: $s_i^{ALF} = \alpha \cdot s_i^{MAF} + (1-\alpha) \cdot s_i^{ICD}$。
- **核心思路**: 对于不同样本动态调整视觉特征 logits 和去混淆 logits 的占比。
- **设计动机**: MAF logits 基于纯视觉特征，ICD logits 基于文本对齐的去混淆 logits，两者互补但最优权重因样本而异。

### 损失函数 / 训练策略

- 三路交叉熵损失: $\mathcal{L}_{CE} = \mathcal{L}_{CE}^{MAF} + \mathcal{L}_{CE}^{ICD} + \mathcal{L}_{CE}^{ALF}$
- 两路相似性正则: $\mathcal{L}_{Sim} = \|s_i^{MAF} - s_i^{ZS}\|_1 + \|s_i^{ICD} - s_i^{ZS}\|_1$（防止过度去混淆）
- 总损失: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Sim}$，$\lambda = 1.0$
- AdamW 优化器，初始学习率 0.001，50 epochs，batch size 64
- 输入 224×224，random resized crop + horizontal flip

## 实验关键数据

### 主实验——11 数据集平均准确率

| 方法 | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp | 59.80 | 62.21 | 66.84 | 70.05 | 73.45 |
| Tip-Adapter-F | 64.55 | 66.79 | 69.76 | 72.59 | 75.69 |
| APE | 65.13 | 67.19 | 69.47 | 71.58 | 73.36 |
| Proto-CLIP-F | 61.84 | 65.96 | 68.29 | 73.13 | 76.18 |
| **LDC（本文）** | **65.71** | **67.92** | **71.17** | **75.79** | **79.78** |
| 增益 vs 次优 | +0.58 | +0.73 | +1.41 | +2.66 | +3.60 |

### ImageNet 单数据集

| 方法 | 1-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|
| Tip-Adapter-F | 61.32 | 62.52 | 64.00 | 65.51 |
| FAR | 60.80 | 62.40 | 64.30 | 66.39 |
| **LDC（本文）** | 60.48 | 62.47 | **64.44** | **66.63** |

### 关键发现

1. **随 shot 数增加优势扩大**：11 数据集平均增益从 1-shot 的 +0.58% 增长到 16-shot 的 +3.60%，说明更多样本让去混淆模块学到更精确的混淆模式。
2. **在 ImageNet 上 1-shot 不如 APE 但 8-shot 后超越**：反映了去混淆模块需要一定数量样本才能可靠估计混淆模式。
3. **类间混淆是 CLIP FSL 的主要瓶颈**：实验可视化表明消除混淆后 logits 的类别区分度显著提升。
4. **训练极其高效**：11 个数据集 16-shot 设置总训练+测试仅需 37 分钟（单 4090D）。

## 亮点与洞察

- 问题定义精准：从 logits 混淆矩阵中发现了可复现的类间混淆模式，并用残差学习优雅解决
- 多层特征融合为少样本场景提供了更鲁棒的表示基础
- L1 相似性正则防止过度去混淆的设计考虑周全
- 自适应融合权重使方法对不同样本具有灵活性
- 整体方法轻量高效，仅需训练少量 Adapter 参数

## 局限与展望

- 在极端 1-shot 场景下优势较小，混淆模式估计不够稳定
- 仅在 ResNet-50 backbone 上做了主要实验，ViT backbone 的探索不够充分
- $\lambda$ 和融合权重预设值的鲁棒性未做系统分析
- 未分析在类别数非常多（如 1000+）时 ICD 的扩展性
- 未探索在开放词汇场景（新出现类别）下的适应性

## 相关工作与启发

- Tip-Adapter 用 cache 模型实现训练自由适应但未处理混淆问题，本文从互补角度入手
- CoOp/CoCoOp 通过 prompt 学习适应下游任务，方向正交于 logit 层面的去混淆
- 启发：其他视觉语言模型（如 BLIP、SigLIP）也可能存在类似的 logit 混淆问题

## 评分

- **新颖性**: ⭐⭐⭐⭐ 问题发现有价值，残差去混淆框架新颖；但核心模块（Adapter + 残差）设计较常规
- **实验充分度**: ⭐⭐⭐⭐ 11 数据集 + OOD 泛化 + 多 shot 设置，覆盖全面
- **写作质量**: ⭐⭐⭐ 技术描述清晰但部分表述冗余
- **价值**: ⭐⭐⭐⭐ 揭示了 CLIP FSL 的关键瓶颈并提供了有效解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/model_compression/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ECCV 2024\] Improving Zero-Shot Generalization for CLIP with Variational Adapter](../../ECCV2024/model_compression/improving_zero-shot_generalization_for_clip_with_variational_adapter.md)
- [\[CVPR 2025\] InsTaG: Learning Personalized 3D Talking Head from Few-Second Video](instag_learning_personalized_3d_talking_head_from_few-second_video.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/model_compression/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)

</div>

<!-- RELATED:END -->
