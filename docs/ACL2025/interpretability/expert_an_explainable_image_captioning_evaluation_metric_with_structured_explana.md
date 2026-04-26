---
title: >-
  [论文解读] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations
description: >-
  [ACL 2025][图像描述评估] 本文提出 EXPERT，一种基于 VLM 微调的无参考图像描述评估指标，通过构建大规模结构化解释数据集并设计两阶段评估模板，在多个基准数据集上达到 SOTA 的同时，提供基于流畅度、相关性、描述性三个维度的高质量结构化解释。
tags:
  - ACL 2025
  - 图像描述评估
  - 可解释评估指标
  - 结构化解释
  - VLM微调
  - 参考无关评估
---

# EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations

**会议**: ACL 2025  
**arXiv**: [2506.24016](https://arxiv.org/abs/2506.24016)  
**代码**: [有](https://github.com/hjkim811/EXPERT)  
**领域**: 多模态VLM  
**关键词**: 图像描述评估, 可解释评估指标, 结构化解释, VLM微调, 参考无关评估

## 一句话总结

本文提出 EXPERT，一种基于 VLM 微调的无参考图像描述评估指标，通过构建大规模结构化解释数据集并设计两阶段评估模板，在多个基准数据集上达到 SOTA 的同时，提供基于流畅度、相关性、描述性三个维度的高质量结构化解释。

## 研究背景与动机

图像描述（Image Captioning）的自动评估对于衡量和改进模型至关重要。近年来，可解释评估指标受到关注——它们不仅给出数值分数，还提供文本解释。然而现有可解释指标存在两个关键不足：

**解释缺乏标准化准则**：现有指标（如 FLEUR）生成的解释没有统一的评估维度和格式，导致内容和结构不一致

**解释质量未经验证**：此前的研究缺乏对生成解释质量的系统评估——即使分数准确，解释可能包含错误或不相关信息

核心问题：如何构建一个**既准确评分又能提供高质量、可解释反馈**的图像描述评估指标？

## 方法详解

### 整体框架

EXPERT 的构建分为三个阶段：

1. **数据集构建**：扩展现有人工评判数据集（Polaris、Nebula），为每个图像-描述对添加结构化解释
2. **模板设计**：设计两阶段评估模板（先评分后解释）
3. **监督微调**：在 LLaVA-1.5 (13B) 上进行 SFT

### 关键设计

#### 1. 结构化解释数据集构建

扩展两个数据集获得大规模训练数据：

| 数据集 | 原数据集 | 解释数量 |
|--------|---------|---------|
| Polaris-exp | Polaris (Wada et al., 2024) | 16,014 |
| Nebula-exp | Nebula (Matsuda et al., 2024) | 26,152 |
| **总计** | | **42,166** |

每条解释基于三个标准化维度：
- **流畅度（Fluency）**：描述是否流畅、自然、语法正确
- **相关性（Relevance）**：描述是否正确表述了视觉内容、与图像紧密相关
- **描述性（Descriptiveness）**：描述是否精准、信息丰富，涵盖图像重要细节

解释使用 GPT-4o 生成，并通过人工评估验证质量。

**数据集质量验证（表1）**：

| 评估维度 | 平均分（4分制） | 标准差 |
|---------|--------------|--------|
| 一致性（Consistency） | 3.72 | 0.52 |
| 事实性（Factuality） | 3.84 | 0.39 |
| 信息量（Informativeness） | 3.72 | 0.45 |

4 名英语母语标注者对 100 个均匀采样的解释进行评估，确认质量很高。

#### 2. 两阶段评估模板

模板采用**先评分后解释**的顺序（scoring-explanation order），已被先前研究证明有效：

**第一阶段 - 评分**：
- 查询：要求对图像-描述对打分
- 回应：来自数据集的人工评分
- **分数分箱（Score Binning）**：将分数四舍五入到最近的 0.10 倍数，简化数值表示

**第二阶段 - 解释**：
- 查询：要求基于三个维度提供简要解释 + 每个维度的描述 + 预定义输出格式
- 回应：数据集中对应的结构化解释

模板的关键设计点：
- 两阶段使用相同的维度描述，保持一致性
- 预定义输出格式确保解释结构统一
- 评分阶段使用数据集中的人工分数，解释阶段使用 GPT-4o 生成的解释

#### 3. 监督微调（SFT）

Base model: **LLaVA-1.5 (13B)**

数据处理：
- 合并 Polaris-exp 和 Nebula-exp 的训练集
- Polaris 中同一图像-描述对有多名标注者分数 → 取均值
- 跨数据集重复对 → 合并取均值
- 转换为两阶段评估模板格式进行训练

#### 4. 推理时的分数平滑

使用 Score Smoothing 获取更精细的分数：

$$s = \sum_{j=1}^{2} 10^{-j} \sum_{i=0}^{9} i \times p(i, j)$$

其中 $p(i,j)$ 是第 $j$ 位小数处生成数字 $i$ 的概率。相比贪心解码（直接取概率最高的数字），分数平滑利用了概率分布信息，产生更连续的分数。

### 损失函数 / 训练策略

- 标准 SFT 损失（交叉熵）
- 基于 LLaVA-1.5 (13B) 的全参数微调
- 采用贪心解码确保确定性和可复现性
- 分数平滑在推理阶段应用，不影响训练

## 实验关键数据

### 主实验

**在多个人工评判基准上的性能（表2摘要）**：

| 指标 | Flickr8k-EX (τc) | Flickr8k-CF (τb) | COMPOSITE (τc) | Polaris (τc) | Nebula (τc) |
|------|-----------------|-----------------|---------------|------------|------------|
| CLIPScore | 51.2 | 34.4 | 53.8 | 52.3 | 46.9 |
| PAC-S | 54.3 | 36.0 | 55.7 | 52.5 | 47.2 |
| FLEUR | 53.0 | 38.6 | 63.5 | 58.3 | 51.7 |
| HICE-S | 56.4 | 37.2 | 57.9 | - | - |
| **EXPERT** | **56.7** | **39.3** | **65.0** | **61.1** | **54.9** |
| GPT-4o | 54.3 | 39.3 | 65.9 | 58.2 | 54.3 |

EXPERT 在所有无参考指标中取得 SOTA（除 Pascal-50S）。甚至超越了许多需要参考描述的指标（如 CLIPScore、PAC-S、FLEUR 等）。

**与 GPT-4o 的对比**：EXPERT 在大多数数据集上持平或优于 GPT-4o，展示了 13B 模型经过精心微调可以匹敌超大规模模型。

### 消融实验

**解释质量人工评估（图4）**：

| 指标 | 一致性 | 事实性 | 信息量 |
|------|--------|--------|--------|
| FLEUR | ~2.5 | ~2.8 | ~2.3 |
| EXPERT_{w/o SFT} | ~2.3 | ~2.5 | ~2.1 |
| **EXPERT** | **~3.4** | **~3.5** | **~3.2** |

关键发现：
- EXPERT 在所有维度上大幅领先，差异在 0.01 显著性水平下统计显著
- **仅有标准化维度不够**（EXPERT_{w/o SFT} 不优于 FLEUR），必须结合高质量解释的监督训练

**定性分析（图3）**：
- 示例1：描述提到「三只狗」但图中只有一只。FLEUR 忽略了描述遗漏飞盘的细节，EXPERT 准确指出缺乏对狗追飞盘动作的描述
- 示例2：描述有语法不完整。FLEUR 错误地将描述解读为提到「蓝色床」，EXPERT 正确识别语法不完整性

### 关键发现

1. **小模型可超越 GPT-4o**：13B 的 EXPERT 在评估任务上匹敌或超越 GPT-4o，原因在于 SFT 对齐了人工偏好 + 有完整的 token 概率用于分数平滑
2. **跨数据集泛化性好**：在 Polaris/Nebula（训练集来源）之外的数据集上也表现优异，表明人工评估偏好跨数据集有一定一致性
3. **结构化解释 + 监督训练缺一不可**：没有训练的结构化提示不比 FLEUR 好，但训练后提升巨大
4. **最常见错误类型**：对缺乏细节的描述过度惩罚

## 亮点与洞察

1. **首次系统评估可解释指标的解释质量**：之前的工作仅关注评分准确性，本文首次将解释质量本身作为评估对象
2. **数据构建策略高效**：利用 GPT-4o 生成解释 + 人工验证质量，以较低成本获得 42K+ 高质量训练数据
3. **分数分箱 + 分数平滑的组合**：训练时简化数值表示，推理时利用概率分布恢复精度，是一个巧妙的工程设计
4. **实用的三维度评估框架**：流畅度、相关性、描述性覆盖了图像描述评估的核心方面，具有良好的可解释性

## 局限与展望

1. **推理速度慢**：生成结构化解释需要大量输出 token，显著增加推理时间
2. **过度惩罚倾向**：最常见错误是对缺乏细节的描述给予过低分数
3. **仅支持英语**：基于 LLaVA-1.5 的微调限制了多语言能力
4. **解释数据依赖 GPT-4o**：如果 GPT-4o 的评判存在系统性偏差，会传递到 EXPERT
5. **可探索方向**：更大的基础模型（如 LLaVA-Next）、更多评估维度（如创意性、幽默性）、多语言扩展

## 相关工作与启发

- **FLEUR (Lee et al., 2024)**：唯一的可解释无参考指标前序工作，使用 scoring-explanation 顺序但缺乏标准化维度
- **Polaris (Wada et al., 2024) / Nebula (Matsuda et al., 2024)**：本文扩展的人工评判数据集
- **CLIPScore (Hessel et al., 2021)**：基于 CLIP 的无参考指标基线
- **HICE-S (Zeng et al., 2024)**：层次化图像描述评估，在 Pascal-50S 上优于 EXPERT
- 启发：「评分 + 结构化解释」的监督微调范式可推广到其他生成任务的评估（如摘要评估、对话评估、代码评估）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 实用性 | 4.5 |
| 实验完整度 | 5 |
| 写作清晰度 | 4.5 |
| 总评 | 4.5 |

实验异常扎实——6 个基准数据集、20+ 个基线对比、人工评估验证解释质量。数据构建和模板设计都很精巧。作为评估指标论文，兼顾了准确性和可解释性两个维度，具有很高的实用价值。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Why Does It Look There? Structured Explanations for Image Classification](../../CVPR2025/interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[NeurIPS 2025\] ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation](../../NeurIPS2025/interpretability/arecho_autoregressive_evaluation_via_chain-based_hypothesis_optimization_for_spe.md)
- [\[ACL 2026\] TabReX: Tabular Referenceless eXplainable Evaluation](../../ACL2026/interpretability/tabrex_tabular_referenceless_explainable_evaluation.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](shortcut_neuron_eval.md)
- [\[ACL 2025\] The Anatomy of Evidence: An Investigation Into Explainable ICD Coding](the_anatomy_of_evidence_an_investigation_into_explainable_icd_coding.md)

<!-- RELATED:END -->
