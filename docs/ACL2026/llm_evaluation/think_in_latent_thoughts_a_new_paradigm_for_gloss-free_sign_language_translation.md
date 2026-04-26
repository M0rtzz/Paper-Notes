---
title: >-
  [论文解读] Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation
description: >-
  [ACL 2026][手语翻译] 提出 SignThought，一种推理驱动的无注释手语翻译框架：引入可学习的潜在思维槽作为视频和文本之间的显式中间语义层，通过"先规划后定位"的双流解码器实现语义规划与视觉证据检索的解耦，在多个基准上超越现有无注释方法。
tags:
  - ACL 2026
  - 手语翻译
  - 无注释翻译
  - 潜在思维链
  - 跨模态推理
  - 双流解码器
---

# Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation

**会议**: ACL 2026  
**arXiv**: [2604.15301](https://arxiv.org/abs/2604.15301)  
**代码**: [GitHub](https://github.com/fletcherjiang/SignThought)  
**领域**: 手语翻译  
**关键词**: 手语翻译, 无注释翻译, 潜在思维链, 跨模态推理, 双流解码器

## 一句话总结

提出 SignThought，一种推理驱动的无注释手语翻译框架：引入可学习的潜在思维槽作为视频和文本之间的显式中间语义层，通过"先规划后定位"的双流解码器实现语义规划与视觉证据检索的解耦，在多个基准上超越现有无注释方法。

## 研究背景与动机

**领域现状**：手语翻译从基于 gloss 的级联方法逐步发展到无 gloss 的端到端视频到文本方法。

**现有痛点**：现有模型隐含假设手语视频片段可直接映射到口语词汇，但手语中大量含义是通过分类词、空间语法和运动调节动态生成的（productive forms），不存在固定词汇对应。

**核心矛盾**：SLT 本质上是跨模态推理问题而非简单对齐——含义分散在连续视频流中，需要跨时间推理才能正确理解。

**本文目标**：引入显式的中间语义表示（潜在思维链），在视频编码和文本解码之间建立可追溯的推理桥梁。

**切入角度**：类比 CoT，但在连续潜在空间而非离散文本空间中实现——用可学习的思维槽从视频中蒸馏语义。

**核心 idea**：K 个有序潜在思维槽通过因果自注意力+Sinkhorn路由交叉注意力迭代提取语义，形成有向思维链；双流解码器先查询思维链规划语义，再回到视频检索证据。

## 方法详解

### 整体框架

三部分：(1) 手语编码器→帧级证据 $\mathbf{E}$；(2) 潜在 CoT 模块→有序思维链 $\mathbf{C}$（分段→Sinkhorn绑定→路由检索）；(3) 双流解码器（思维注意力→视频注意力）。

### 关键设计

1. **潜在思维链模块**:

    - 功能：从密集视频特征中蒸馏有序语义推理状态
    - 核心思路：K 个可学习思维槽通过 L 层迭代细化。每层先做因果自注意力（思维 k 只看 1..k），再通过 Sinkhorn 归一化将视频分段分配给各思维，最后路由交叉注意力提取细粒度证据
    - 设计动机：因果约束提供有向结构，Sinkhorn 防止注意力退化

2. **"先规划后定位"双流解码器**:

    - 功能：分离语义决策和证据检索
    - 核心思路：每层：自注意力→思维交叉注意力（规划）→视频交叉注意力（定位）。思维注意力权重通过路由矩阵转换为帧级先验引导视频注意力
    - 设计动机：直接从所有帧检索会导致注意力扩散，先确定语义意图再检索更可控

3. **大规模无 gloss 数据集**:

    - 功能：提供更强上下文依赖的训练/评估数据
    - 核心思路：构建并开源新数据集，含更多 productive forms
    - 设计动机：现有数据集上下文依赖弱

### 损失函数 / 训练策略

标准序列级交叉熵 + 思维连续性损失。端到端训练，仅需句子级标注。

## 实验关键数据

### 主实验

| 方法 | PHOENIX14T B@4 | ROUGE |
|------|---------------|-------|
| SLTUNET | 28.47 | 52.11 |
| SignThought | **31.2** | **54.8** |

### 消融实验

| 配置 | B@4 Δ | 说明 |
|------|-------|------|
| 去除思维链 | -2.5 | 直接解码退化 |
| 去除因果约束 | -1.3 | 思维无序 |
| 去除 Sinkhorn | -1.8 | 注意力退化 |
| 去除双流 | -2.1 | 规划定位耦合 |

### 关键发现

- 潜在思维链比直接解码提升 2-3 BLEU
- 思维链可作为可追溯锚点对齐文本与视频时间区域

## 亮点与洞察

- "手语翻译是推理而非对齐"改变了领域建模范式
- 潜在思维链作为跨模态接口可迁移到其他连续-离散跨模态任务
- Sinkhorn 防止注意力退化的技巧在其他 slot attention 场景也适用

## 局限与展望

- 思维槽数 K 需预设，不同视频长度最优 K 可能不同
- 仅在受限数据集验证
- 编码器使用 Inception 特征，更强编码器可能进一步提升

## 相关工作与启发

- **vs Gloss-based**: 需昂贵标注，本文完全无 gloss
- **vs 直接视频到文本**: 缺乏中间推理层，性能受限

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将潜在 CoT 引入手语翻译
- 实验充分度: ⭐⭐⭐⭐ 多基准+充分消融
- 写作质量: ⭐⭐⭐⭐ 动机充分但符号密集
- 价值: ⭐⭐⭐⭐⭐ 对手语翻译和跨模态推理都有重要贡献

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](../../ICML2025/llm_evaluation/latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] Language Model as Planner and Formalizer under Constraints](language_model_as_planner_and_formalizer_under_constraints.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)

<!-- RELATED:END -->
