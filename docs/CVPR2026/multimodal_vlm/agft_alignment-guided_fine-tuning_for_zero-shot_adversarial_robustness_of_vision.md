---
title: >-
  [论文解读] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models
description: >-
  [CVPR 2026][多模态][对抗鲁棒性] AGFT 提出了一种对齐引导的微调框架，通过文本引导的对抗训练和分布一致性校准，在增强 VLM 零样本对抗鲁棒性的同时保持预训练的跨模态语义结构，在 15 个零样本基准上平均鲁棒准确率达到 46.57%，超越 SOTA 3.1 个百分点。
tags:
  - CVPR 2026
  - 多模态
  - 对抗鲁棒性
  - 视觉语言模型
  - 零样本泛化
  - 对齐引导
  - 分布一致性校准
---

# AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.29410](https://arxiv.org/abs/2603.29410)  
**代码**: [GitHub](https://github.com/YuboCui/AGFT) (有)  
**领域**: Multimodal VLM / Adversarial Robustness  
**关键词**: 对抗鲁棒性, 视觉语言模型, 零样本泛化, 对齐引导, 分布一致性校准

## 一句话总结
AGFT 提出了一种对齐引导的微调框架，通过文本引导的对抗训练和分布一致性校准，在增强 VLM 零样本对抗鲁棒性的同时保持预训练的跨模态语义结构，在 15 个零样本基准上平均鲁棒准确率达到 46.57%，超越 SOTA 3.1 个百分点。

## 研究背景与动机
**领域现状**：CLIP 等 VLM 展现强零样本能力，但对对抗扰动极度脆弱（零样本条件下 CLIP 鲁棒准确率仅 6.24%）。

**现有痛点**：
   - 现有方法（TeCoA、GLADIATOR）采用分类引导的对抗微调，使用硬标签监督推动特征向目标类聚拢；
   - 这种方式**破坏了预训练的跨模态对齐结构**，图像与文本之间的细粒度语义对应被扭曲，零样本泛化能力下降。

**核心矛盾**：增强对抗鲁棒性需要修改视觉特征空间，但这些修改会破坏 CLIP 赖以泛化的跨模态语义结构。如何在"鲁棒"和"对齐"之间取得平衡？

**本文切入角度**：不将 VLM 当作分类器微调，而是保持其作为跨模态对齐模型的本质——用原始模型的概率预测作为软监督，引导对抗特征对齐到文本嵌入。

**核心 idea**：用软对齐分布替代硬标签 + 温度校准消除置信度尺度失配 = 保持跨模态结构的对抗训练。

## 方法详解

### 整体框架
输入：ImageNet 训练集 → PGD 生成对抗样本 $\mathbf{x}_{adv}$ → 冻结预训练 CLIP 计算软预测分布 $\mathbf{p}_{rob}$（温度校准后）→ 微调图像编码器使对抗样本的预测匹配 $\mathbf{p}_{rob}$ → 在 15 个零样本数据集上评估。

### 关键设计
1. **文本引导的对抗训练（Text-Guided Adversarial Training）**：

    - **功能**：用预训练 CLIP 的软概率分布（而非硬标签）作为对抗训练目标。
    - **核心思路**：$p_{orig}^{i,j} = \frac{\exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^j)) / \tau)}{\sum_k \exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^k)) / \tau)}$，对抗训练损失用 KL 散度形式：$L = -\mathbb{E}_{i,j}[p_{rob}^{i,j} \log \frac{\exp(\cos(f_\theta(x_{adv}^i), f_\phi(t^j))/\tau)}{\sum_k ...}]$
    - **设计动机**：硬标签只关注正确类别，忽视了图像与其他文本之间的相对相似度关系。软分布保留了这些关系，使微调后的特征空间保持与原始 CLIP 一致的语义结构。

2. **分布一致性校准（Distribution Consistency Calibration）**：

    - **功能**：通过温度缩放调整目标分布，消除置信度尺度与语义结构的纠缠。
    - **核心思路**：$p_{rob}^{i,j} = \frac{\exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^j)) / (\tau/\gamma))}{\sum_k \exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^k)) / (\tau/\gamma))}$，其中 $\gamma \in (0,1]$，增大有效温度使分布更平滑。
    - **设计动机**：直接使用 $p_{orig}$ 作为目标会强制鲁棒模型继承预训练模型的置信度尺度（logits 的绝对大小），而这个尺度可能与鲁棒特征空间不匹配。温度缩放分离了"相对语义关系"和"置信度尺度"，只保留前者作为监督信号。

3. **最终目标函数**：$\min \mathbb{E}_{\mathbf{x} \in \mathcal{D}}[\max_{\mathbf{x}_{adv} \in B(\mathbf{x}, \epsilon)} L(\mathbf{x}_{adv}, \mathbf{t}, \mathbf{p}_{rob}, \tau)]$

    - 内层最大化：PGD 生成对抗样本
    - 外层最小化：使对抗样本预测匹配校准后的软分布

### 损失函数 / 训练策略
- 仅微调图像编码器（全参数），文本编码器冻结
- SGD，lr=$4 \times 10^{-4}$，余弦衰减，10 epochs
- 对抗训练使用 2-step PGD，$\epsilon \in \{1/255, 2/255, 4/255\}$
- 超参数 $\gamma = 0.4$, $\tau = 1/180$

## 实验关键数据

### 主实验（PGD-20, $\epsilon=1/255$ 零样本鲁棒准确率）

| 方法 | Caltech101 | CIFAR10 | Food101 | ImageNet | STL10 | 15数据集平均 |
|------|-----------|---------|---------|----------|-------|------------|
| CLIP（无防御） | 21.27 | 10.31 | 4.06 | 1.13 | 33.10 | 6.24 |
| TeCoA | 71.83 | 59.85 | 29.01 | 41.29 | 83.33 | 38.51 |
| GLADIATOR | 73.34 | 67.89 | 34.92 | 44.53 | 86.53 | 43.46 |
| **AGFT** | **82.23** | **71.72** | **44.76** | 44.95 | **88.52** | **46.57** |

### 零样本干净准确率

| 方法 | 15数据集平均（干净） | 15数据集平均（鲁棒） | 说明 |
|------|-------------------|------|------|
| CLIP | 66.20 | 6.24 | 干净强但极不鲁棒 |
| TeCoA | 56.93 | 38.51 | 干净下降严重 |
| GLADIATOR | 60.34 | 43.46 | 较好平衡 |
| **AGFT** | **61.35** | **46.57** | 干净与鲁棒均最优 |

### 关键发现
- AGFT 在鲁棒性和干净准确率上同时优于所有基线，说明保持对齐结构确实带来双赢
- 在 StanfordCars（+12.6%）和 Food101（+9.8%）等细粒度数据集上提升最显著
- 在 C&W 和 AutoAttack 等更强攻击下同样保持优势

## 亮点与洞察
- **核心洞察深刻**：指出分类引导微调破坏跨模态对齐是 ZSAR 性能上限的瓶颈
- 温度校准的分析角度新颖——将"语义结构"和"置信度尺度"解耦
- 方法极为简洁：本质上只是改变了对抗训练的目标分布

## 局限与展望
- 仅使用 ViT-B/32 验证，更大架构（ViT-L）效果待验证
- 温度参数 $\gamma$ 需要调参，对不同域可能需要调整
- 在 EuroSAT 等域特定数据集上，鲁棒准确率仍较低（16.25%）

## 相关工作与启发
- 与知识蒸馏思路相近，但目标是保持结构而非压缩
- 温度校准启发来自标签平滑和知识蒸馏中的温度技巧

## 评分
- 新颖性: ⭐⭐⭐⭐ 对齐引导替代分类引导的思路清晰且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 15个数据集×多种攻击×多个基线，极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导和方法阐述逻辑严密
- 价值: ⭐⭐⭐⭐ 对 VLM 对抗鲁棒性研究有重要启发

<!-- RELATED:START -->

## 相关论文

- [TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/multimodal_vlm/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)

<!-- RELATED:END -->
