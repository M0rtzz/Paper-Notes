---
title: >-
  [论文解读] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition
description: >-
  [AAAI 2026][不完整多模态] 提出 Cross-modal Prompting (ComP) 方法，通过渐进式提示生成+跨模态知识传播+动态调度器来解决不完整多模态情感识别中的模态不平衡问题，在 4 个数据集、 7 种缺失率下均达到 SOTA。
tags:
  - AAAI 2026
  - 不完整多模态
  - 情感识别
  - 提示学习
  - 模态平衡
  - 知识传播
---

# Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition

**会议**: AAAI 2026  
**arXiv**: [2512.11239](https://arxiv.org/abs/2512.11239)  
**代码**: [GitHub](https://github.com/WenjueHE/2026-AAAI-ComP)  
**领域**: 多模态VLM  
**关键词**: 不完整多模态, 情感识别, 提示学习, 模态平衡, 知识传播

## 一句话总结
提出 Cross-modal Prompting (ComP) 方法，通过渐进式提示生成+跨模态知识传播+动态调度器来解决不完整多模态情感识别中的模态不平衡问题，在 4 个数据集、 7 种缺失率下均达到 SOTA。

## 研究背景与动机

**领域现状**：多模态情感识别（MER）利用音频、文本、视频等多源信息，但实际场景中常出现模态缺失（背景噪声导致音频不可用、语音识别失败等）。

**现有痛点**：
   - **模态性能差距**：不同模态的识别能力差异巨大
   - **模态欠优化**：部分模态在多模态共同训练后性能反而下降（比单模态训练还差）
   - 现有 IMER 方法要么恢复缺失数据（消耗大、不一定有用），要么学统一表示但忽略模态不平衡

**核心矛盾**：如何在模态缺失的情况下，既处理不完整性又解决模态不平衡？

**本文目标** 通过跨模态提示学习同时解决缺失处理和模态平衡。

**切入角度**：不恢复缺失数据，而是将每个模态的信息压缩为提示（prompt），传播到其他模态来增强任务相关的一致性信息，同时通过动态调度达到平衡。

**核心 idea**：用渐进式提示生成压缩跨模态一致性信息，通过知识传播模块增强各模态的任务相关特征，增强而非恢复。

## 方法详解

### 整体框架
两阶段训练：
1. **第一阶段**：各模态独立训练编码器 + 分类器
2. **第二阶段**：提示生成→知识传播→多模态协作融合

### 关键设计

1. **渐进式提示生成（PG）**:

    - 功能：将每个模态的特征压缩为代表性且一致的提示
    - 核心思路：全局输入压缩为少量原型，配合动态梯度调制器避免原型被简单样本主导，再与上下文特征融合得到低维提示
    - 设计动机：提示应包含跨模态一致的情感信息，而非模态特有的噪声

2. **跨模态知识传播（KP）**:

    - 功能：将其他模态的提示传入当前模态，增强任务相关特征
    - 核心思路：将模态特征 $\mathbf{Z}_l^u$ 与两个跨模态提示 $\mathbf{P}_l^{vu}, \mathbf{P}_l^{wu}$ 拼接→线性投影压缩→多头自注意力增强→投影回原始空间
    - 巧妙之处：缺失实例在 MSA 中被 mask，但通过跨模态提示的传播自然地“重建”了缺失数据的信息

3. **多模态协作模块（Coordinator）**:

    - 动态重新加权各模态的输出，作为平衡策略的补充

### 损失函数 / 训练策略
$\mathcal{L} = \sum_{u} \mathcal{L}_{enc}(\mathbf{Z}^u, \hat{\mathbf{X}}^u) + \sum_{u} \mathcal{L}_{task}(\mathbf{Y}^u, \mathbf{Y})$

## 实验关键数据

### 主实验（IEMOCAP 4-class，不同缺失率）

| 方法 | 0.1 ACC | 0.3 ACC | 0.5 ACC | 0.7 ACC |
|------|---------|---------|---------|--------|
| GCNet | 74.82 | 74.49 | 72.67 | 71.00 |
| MoMKE | 76.70 | 73.47 | 69.73 | 66.52 |
| SDR-GNN | 78.48 | 78.22 | 75.47 | 70.52 |
| **ComP** | **80.66** | **78.37** | **75.62** | **73.41** |

ComP 在所有缺失率下均达到最优，在高缺失率（0.7）下优势更明显（+2.89%）。

### 消融/关键发现
- 在基线方法中，Video 和 Text 模态在多模态共同训练后性能下降（模态欠优化），而 ComP 使所有模态都从多模态学习中受益
- 在 4 个数据集、7 种缺失率、与 7 个 SOTA 方法对比中全面领先
- 梯度调制器对提示质量很关键——避免简单样本主导原型学习

## 亮点与洞察
- **“增强而非恢复”的思路**很巧妙：不恢复缺失数据，而是通过跨模态提示增强现有模态——更高效且自然解决缺失问题
- **模态不平衡问题的可视化分析**（Fig.1）直观展示了问题和解决效果
- **知识传播过程中自然重建缺失数据**——不需要额外的缺失处理模块

## 局限与展望
- 提示生成和知识传播增加了模型复杂度
- 仅在情感识别任务上验证，其他不完整多模态任务有待探索
- 动态梯度调制器的设计动机和理论分析可以更深入

## 相关工作与启发
- **vs MMIN**: MMIN 用级联自编码器恢复缺失数据，消耗大且恢复信息不一定有用；ComP 用提示增强更高效
- **vs MoMKE**: MoMKE 用 MoE 的思路而非平衡各模态，在高缺失率下性能下降严重；ComP 的稳定性更强
- **vs MMPareto**: MMPareto 用 Pareto 效率平衡梯度，但未考虑缺失场景；ComP 将平衡和缺失处理统一

## 评分
- 新颖性: ⭐⭐⭐⭐ 提示学习+模态平衡+缺失处理的统一框架很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集×7种缺失率×7个基线，非常全面
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 对不完整多模态场景提供了实用的解决方案

<!-- RELATED:START -->

## 相关论文

- [Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)

<!-- RELATED:END -->
