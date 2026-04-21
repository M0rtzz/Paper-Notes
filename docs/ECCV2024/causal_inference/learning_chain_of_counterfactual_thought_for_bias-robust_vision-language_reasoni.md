---
title: >-
  [论文解读] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning
description: >-
  [ECCV 2024][反事实推理] 本文提出了反事实偏差鲁棒推理数据集（CoBRa）和反事实思维链方法（CoCT），通过构造编辑后的知识图谱和图像内容来评估和缓解大型视觉语言模型（LVLM）中的知识偏差，使模型能够逐步推理而非依赖偏见知识，在需要知识偏差下推理的任务上显著优于现有方法。
tags:
  - ECCV 2024
  - 反事实推理
  - 知识偏差
  - 视觉语言推理
  - VQA
  - 大型视觉语言模型
---

# Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning

**会议**: ECCV 2024  
**论文**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1309_ECCV_2024_paper.php)
**代码**: [GitHub](https://github.com/SuperJohnZhang/CoBRa)  
**领域**: 多模态VLM / 因果推理  
**关键词**: 反事实推理, 知识偏差, 视觉语言推理, VQA, 大型视觉语言模型

## 一句话总结

本文提出了反事实偏差鲁棒推理数据集（CoBRa）和反事实思维链方法（CoCT），通过构造编辑后的知识图谱和图像内容来评估和缓解大型视觉语言模型（LVLM）中的知识偏差，使模型能够逐步推理而非依赖偏见知识，在需要知识偏差下推理的任务上显著优于现有方法。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLM，如 GPT-4V、LLaVA）在 VQA、图像理解等任务上取得了巨大成功。这些模型在海量数据上预训练，积累了丰富的世界知识。然而，这种从训练数据中继承的知识也带来了隐含的偏见。

**现有痛点**：LVLM 对训练数据中的知识偏差（knowledge bias）高度敏感。当面对与训练分布不一致的反事实场景时——例如一只紫色的香蕉或一个穿着太空服的中世纪骑士——模型倾向于依赖预训练中学到的偏见知识（"香蕉是黄色的"）而非基于图像内容进行推理。这限制了模型在新场景下的泛化能力和实际应用的可靠性。

**核心矛盾**：模型需要知识来理解世界，但过度依赖知识又导致偏差。核心问题是如何让模型在保留有用知识的同时，能够在遇到与先验知识冲突的场景时，优先基于视觉证据进行推理。

**本文目标** (1) 如何系统地评估 LVLM 的知识偏差？(2) 如何教会模型在反事实场景下进行鲁棒的推理？

**切入角度**：作者从反事实思维的角度切入——构造"如果世界是另一个样子"的场景。通过编辑知识图谱（改变事实关系）和图像内容（视觉反事实），创建一个要求模型放弃偏见、基于当前证据推理的数据集。然后，教会模型一种"反事实思维链"——先识别与先验知识的冲突，再基于当前证据逐步推理。

**核心 idea**：构造反事实VQA数据集暴露知识偏差，用反事实思维链（CoCT）教会LVLM基于证据而非偏见进行推理。

## 方法详解

### 整体框架

LANP 包含数据集构建和方法两部分。数据集部分（CoBRa）通过编辑知识图谱和图像内容，生成包含反事实场景的 VQA 样本，每个样本附带详细的推理过程标注。方法部分（CoCT）包含两个核心步骤：(1) 训练一个翻译语言模型（TLM）来学习反事实推理过程；(2) 利用 TLM 生成的反事实推理示例作为 in-context 示例，引导 LVLM 进行偏差鲁棒推理。

### 关键设计

1. **CoBRa 反事实偏差鲁棒推理数据集**:

    - 功能：提供系统评估和训练 LVLM 应对知识偏差的数据
    - 核心思路：从已有的知识图谱（KG）出发，选择常见的事实三元组（如"香蕉-颜色-黄色"），然后进行编辑——将"黄色"替换为"紫色"。相应地，使用图像编辑工具修改图像使其与新知识一致。每个样本包含：(a) 原始图像和问题；(b) 编辑后的图像和知识图谱；(c) 完整的推理过程标注，解释如何从编辑后的证据得出正确答案。数据集规模约 64K 样本，涵盖 14K+ 唯一条目。
    - 设计动机：现有 VQA 数据集不专门考察知识偏差。通过显式构造反事实场景，可以准确衡量模型的偏差程度。详细的推理过程标注为学习鲁棒推理策略提供了基础。

2. **反事实思维链（Chain of Counterfactual Thought, CoCT）**:

    - 功能：教会 LVLM 在遇到知识冲突时进行逐步推理
    - 核心思路：CoCT 是一种结构化的推理模式，包含几个关键步骤——(a) 识别问题涉及的知识领域；(b) 检查视觉证据是否与先验知识冲突；(c) 如果冲突，明确放弃先验，基于当前证据建立新的推理链；(d) 逐步推导得出答案。CoCT 的推理链通过 CoBRa 数据集中的标注学习。在推理时，CoCT 作为 few-shot in-context 示例提供给 LVLM，引导模型遵循同样的推理模式。
    - 设计动机：标准的 Chain-of-Thought 提示不包含"检查偏差-放弃先验"的步骤，因此无法有效应对知识冲突。CoCT 显式地教模型"先质疑再推理"的策略。

3. **翻译语言模型（Translation Language Model, TLM）**:

    - 功能：学习将普通推理过程"翻译"为反事实推理过程
    - 核心思路：TLM 在 CoBRa 数据集上训练，输入为原始场景的推理过程 + 编辑后的知识图谱/图像，输出为反事实场景的推理过程。TLM 本质上学习了"如何将一个推理链适配到新的知识设定下"这一能力。训练好的 TLM 可以为新的反事实场景自动生成推理示例，这些示例再作为 LVLM 的 in-context 示例。训练基于 masked language modeling (MLM) 和 translation language modeling (TLM) 的联合目标。
    - 设计动机：手工编写反事实推理示例成本高且覆盖面有限。TLM 自动化了这一过程，可以为任意新场景生成适配的推理链。

### 损失函数 / 训练策略

TLM 使用 MLM + TLM 联合训练目标。MLM 在单语（原始推理或反事实推理）上预测被遮蔽的词，TLM 在双语对齐对（原始推理-反事实推理）上学习跨推理链的翻译。推理时使用 few-shot prompting，无需额外微调 LVLM。

## 实验关键数据

### 主实验

| 模型 | 方法 | CoBRa准确率(%) | 标准VQA准确率(%) | 偏差影响 |
|------|------|---------------|-----------------|---------|
| InstructBLIP | 标准推理 | 38.2 | 72.1 | 严重偏差 |
| InstructBLIP | CoT | 41.5 | 71.8 | 小改善 |
| InstructBLIP | CoCT | 52.3 | 71.5 | 显著改善 |
| LLaVA | 标准推理 | 42.7 | 74.3 | 严重偏差 |
| LLaVA | CoCT | 56.8 | 73.9 | 显著改善 |

### 消融实验

| 配置 | CoBRa准确率 | 说明 |
|------|------------|------|
| CoCT (完整) | 52.3 | 完整方法 |
| w/o TLM（手工示例） | 48.1 | TLM自动生成效果更好 |
| w/o 知识图谱编辑 | 44.6 | 仅图像编辑不足以暴露偏差 |
| w/o 推理过程标注 | 43.2 | 推理链标注是核心 |
| 标准 CoT | 41.5 | 缺乏反事实推理步骤 |

### 关键发现

- 所有测试的 LVLM 在 CoBRa 上都表现出严重的知识偏差，准确率远低于标准 VQA
- CoCT 在不损害标准 VQA 性能的前提下显著提升了反事实推理能力，说明两种能力不冲突
- TLM 生成的推理示例质量高于手工编写，因为 TLM 能自适应地生成与新场景更匹配的推理链
- 知识图谱编辑和图像编辑的组合效果优于单独使用

## 亮点与洞察

- **反事实思维链（CoCT）** 是对标准 CoT 的有意义扩展——增加了"检测冲突-放弃先验"的元认知步骤。这个设计反映了人类在处理反直觉信息时的认知过程，直觉上很合理。可以迁移到任何需要模型"怀疑自己知识"的场景。
- **TLM 作为推理链适配器** 的思路很新颖——不是直接训练LVLM，而是训练一个辅助模型来生成启发式示例。这种"教练模型"的理念可以推广到其他需要定制化 prompt 的任务。
- CoBRa 数据集本身也是重要贡献，为评估 LVLM 的鲁棒性提供了新的维度。

## 局限与展望

- CoBRa 主要依赖知识图谱进行编辑，受限于 KG 的覆盖范围和编辑操作的多样性
- 图像编辑质量可能影响反事实场景的自然度，不自然的编辑可能引入额外偏差
- CoCT 作为 in-context learning 方法，效果受限于上下文窗口长度和示例数量
- 未探索将 CoCT 直接用于 LVLM 微调，而非仅作为 prompt 策略
- 仅在 VQA 任务上验证，未扩展到其他视觉语言任务（如图像描述、视觉推理等）

## 相关工作与启发

- **vs VQA-CP (对比VQA偏差数据集)**: VQA-CP 侧重于语言偏差（如答案分布偏移），CoBRa 侧重于知识偏差（已知事实的反事实修改），两者互补
- **vs Debiasing方法 (如因果推理去偏)**: 传统去偏方法修改模型架构或训练过程，CoCT 通过推理策略实现，不需要重训练模型
- **vs Chain-of-Thought**: 标准 CoT 不包含偏差检测步骤，CoCT 增加了元认知层，更适合知识冲突场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 反事实推理链和CoBRa数据集构建思路新颖，方向有前景
- 实验充分度: ⭐⭐⭐⭐ 多个LVLM上验证，消融实验合理，但数据集规模可以更大
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，反事实思维的概念阐述深入
- 价值: ⭐⭐⭐⭐ 知识偏差是LVLM的重要问题，CoBRa为该方向提供了标准化评测工具

<!-- RELATED:START -->

## 相关论文

- [Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [Understanding Physical Dynamics with Counterfactual World Modeling](understanding_physical_dynamics_with_counterfactual_world_modeling.md)

<!-- RELATED:END -->
