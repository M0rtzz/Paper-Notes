---
title: >-
  [论文解读] Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification
description: >-
  [ACL 2025][intent classification] 提出动态标签名称精炼方法，在检索式 ICL 意图分类中，利用 LLM 根据检索到的示例动态生成更具区分性的意图标签名称（如 "Verify PAN" → "Verify PAN card details"），有效降低语义相似意图间的混淆，在 6 个数据集上一致提升 2.07%-7.51% 准确率。
tags:
  - ACL 2025
  - intent classification
  - few-shot learning
  - label refinement
  - in-context learning
  - LLM
  - semantic disambiguation
---

# Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification

**会议**: ACL 2025  
**arXiv**: [2412.15603](https://arxiv.org/abs/2412.15603)  
**代码**: 无  
**领域**: 对话系统 / NLP  
**关键词**: intent classification, few-shot learning, label refinement, in-context learning, LLM, semantic disambiguation

## 一句话总结

提出动态标签名称精炼方法，在检索式 ICL 意图分类中，利用 LLM 根据检索到的示例动态生成更具区分性的意图标签名称（如 "Verify PAN" → "Verify PAN card details"），有效降低语义相似意图间的混淆，在 6 个数据集上一致提升 2.07%-7.51% 准确率。

## 研究背景与动机

**领域现状**：对话意图分类是任务导向对话系统的核心组件。基于检索的 ICL（In-Context Learning）方法已成为 few-shot 意图分类的主流：为每个测试查询检索相似示例，构造 prompt 让 LLM 进行分类。

**现有痛点**：（1）实际系统中意图类别可达数十甚至数百个，且许多意图之间语义高度重叠——如银行客服中 "Verify PAN"、"Bank verification details"、"Account not verified" 三个意图的嵌入余弦相似度高达 0.91；（2）检索到的示例往往跨不同意图类别存在显著语义重叠，使模型更加困惑；（3）CoT（思维链）方法在需要推理时会被相似标签名称误导，甚至降低性能。

**核心矛盾**：标签名称本身不够有区分性。当多个意图标签在嵌入空间中高度相似时，无论是直接分类还是 CoT 推理都难以准确区分。

**本文目标** 通过动态精炼意图标签名称来降低标签间语义重叠，从而提升分类准确率。

**切入角度**：利用 LLM 分析检索到的示例组之间的语义关系，动态生成更具描述性和区分性的标签名称。

**核心 idea**：让 LLM 先精炼意图标签（使名字更具区分性），再用精炼后的标签做 ICL 分类——用"改名"来解决"分不清"的问题。

## 方法详解

### 整体框架

方法分三步：（1）检索——用 SentenceTransformer 为每个测试查询检索 top-20 最相似的训练示例，按意图分组；（2）标签精炼——用 LLM 分析这些示例组，判断是否需要精炼标签名称，并生成更具描述性的标签；（3）分类——用精炼后的标签和示例构造 ICL prompt，进行最终意图分类。

### 关键设计

1. **基于检索的示例选择**:

    - 功能：为每个测试输入检索语义相似的训练示例
    - 核心思路：使用预训练 SentenceTransformer（Reimers 2019）编码查询和训练集，按余弦相似度检索 top-20 最相似示例。示例按原始意图分组，为后续标签精炼提供上下文
    - 设计动机：检索的示例不仅用于 ICL prompt 构造，更重要的是提供了意图间语义关系的"证据"——正是这些相似但不同意图的示例揭示了标签混淆的根源

2. **动态标签精炼（核心贡献）**:

    - 功能：利用 LLM 分析每组意图的示例，决定是否精炼标签并生成新标签
    - 核心思路：为每个检索到的意图组设计专门的 prompt，让 LLM 评估标签与示例之间的语义关系。模型判断原标签是否足够描述性——若不够则生成增强版本，否则保留原标签。例如 "verify_pan" → "verify_pan_card_details"，使其与 "bank_verification_details" 产生更大语义距离
    - 设计动机：动态精炼（而非静态预定义）因为不同测试查询检索到的示例组不同，需要针对当前上下文中的混淆点来调整标签。实验证实动态精炼显著优于静态精炼

3. **精炼后的 ICL 分类**:

    - 功能：用精炼后的标签替代原始标签进行最终分类
    - 核心思路：构造包含精炼标签示例、测试查询和分类指令的 prompt。示例按相似度从低到高排列（Milios et al. 2023）。由同一个 LLM 完成标签精炼和最终分类，确保精炼的语义理解与分类决策的一致性
    - 设计动机：两步过程让同一 LLM 先理解意图间差异再分类，比一步到位的分类方法更有效

## 实验关键数据

### 主实验（Llama3-8b-inst）

| 数据集 | Raw（基线） | CoT | Refined（精炼） | 提升 |
|--------|-----------|-----|-----------------|------|
| HWU64 | 88.10 | 87.17 (-0.93) | **89.03** | +0.93 |
| BANKING77 | 85.88 | 85.48 (-0.40) | **87.95** | +2.07 |
| CLINC150 | 95.03 | 95.13 (+0.10) | **95.51** | +0.48 |
| CUREKART | 89.76 | 89.76 (+0.00) | **91.94** | +2.18 |
| POWERPLAY11 | 70.87 | 67.00 (-3.87) | **76.10** | +5.23 |
| SOFMATTRESS | 82.61 | 81.10 (-1.51) | **85.40** | +2.79 |

### 消融实验

| 配置 | 说明 | 结论 |
|------|------|------|
| 原始标签相似度 vs 精炼标签相似度 (Llama3-8b) | 0.86 → 0.74 | 精炼显著降低标签间语义重叠 |
| 动态精炼 vs 静态精炼 | 动态一致优于静态 | 上下文相关的精炼更有效 |
| 小模型精炼 + 大模型分类 | 交叉配置下性能下降 | 精炼和分类应由同一模型完成 |

### 跨模型一致性

| 数据集 | Qwen2.5-7b 提升 | Qwen2.5-1.5b 提升 |
|--------|----------------|-------------------|
| BANKING77 | +1.62 | +4.03 |
| SOFMATTRESS | +3.61 | +7.51 |
| POWERPLAY11 | +3.56 | +2.26 |

### 关键发现
- 标签精炼在所有 6 个数据集和 3 个模型上一致提升性能，而 CoT 在 4/6 数据集上反而降低性能
- 提升在高语义重叠的数据集上最显著：POWERPLAY11（+5.23%）和 SOFMATTRESS（+2.79%/7.51%），这些数据集的意图标签间语义最相似
- 小模型受益更大：Qwen2.5-1.5b 在 SOFMATTRESS 上提升 7.51%（vs Llama3-8b 的 2.79%），说明标签精炼有效弥补小模型的语义理解不足
- 精炼后标签的嵌入相似度一致降低（Llama3: 0.86→0.74），验证了方法确实在语义空间中拉开了意图间距离
- 交叉配置实验表明：精炼和分类应由同一模型完成——不同模型对标签精炼的理解不同

## 亮点与洞察
- 方法极其简洁但效果显著——仅需在标准 ICL 流程中插入一个"标签精炼"步骤，无需微调、无需额外数据、无需架构修改。CoT 反而有害（平均 -1.2%），而标签精炼一致有效
- 精炼标签的嵌入相似度分析提供了直觉上令人信服的解释——降低了标签空间的"拥挤度"，让分类器有更多的"呼吸空间"

## 局限与展望
- 每个测试样本都需要额外调用 LLM 进行标签精炼，增加了推理延迟和成本
- 仅在 10-shot 设置下实验，未验证在更多 shot 或 zero-shot 下的效果
- 精炼质量依赖于 LLM 的能力——若 LLM 对领域理解不足，精炼可能反而误导
- 未与基于训练的 few-shot 方法（如 prototypical networks）做比较

## 相关工作与启发
- **vs Milios et al. 2023**: 基于检索的 ICL 意图分类的基础工作，本文在其框架上增加标签精炼步骤
- **vs CoT (Wei et al. 2023)**: CoT 在意图分类中反而有害（被相似标签误导推理链），标签精炼直接解决标签层面的混淆而非推理层面
- **vs Sung et al. 2023; Cho et al. 2024**: 关注 few-shot 意图分类的挑战，但从模型架构角度解决，本文从数据表示（标签名称）角度切入

## 评分
- 新颖性: ⭐⭐⭐⭐ "改名"的思路简单直观但有效，切入角度新颖——从标签命名而非模型能力角度解决分类混淆
- 实验充分度: ⭐⭐⭐⭐ 6 个数据集 + 3 个模型 + 多组消融，一致性好；但缺少与训练-based 方法的对比
- 写作质量: ⭐⭐⭐⭐ 问题导向清晰，Figure 1 的示例直观易懂
- 价值: ⭐⭐⭐⭐ 即插即用的标签精炼策略，可直接应用于各种 ICL 分类场景

<!-- RELATED:START -->

## 相关论文

- [Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](../../ICCV2025/others/is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](../../CVPR2026/others/dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)
- [Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)
- [ConceptCarve: Dynamic Realization of Evidence](conceptcarve_dynamic_realization_of_evidence.md)

<!-- RELATED:END -->
