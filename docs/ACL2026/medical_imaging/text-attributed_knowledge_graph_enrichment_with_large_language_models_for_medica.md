---
title: >-
  [论文解读] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation
description: >-
  [ACL 2026][医学图像][医学概念表示] 本文提出 CoMed，一种 LLM 赋能的图学习框架，通过结合 EHR 统计证据和类型约束 LLM 推理构建全局医学知识图谱，再用 LLM 生成节点描述和边理由丰富为文本属性图…
tags:
  - "ACL 2026"
  - "医学图像"
  - "医学概念表示"
  - "知识图谱"
  - "图神经网络"
  - "电子健康记录"
  - "文本属性图"
---

# Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation

**会议**: ACL 2026  
**arXiv**: [2604.13331](https://arxiv.org/abs/2604.13331)  
**代码**: 无  
**领域**: 医学图像 / 图学习  
**关键词**: 医学概念表示, 知识图谱, LLM-GNN联合学习, 电子健康记录, 文本属性图

## 一句话总结

本文提出 CoMed，一种 LLM 赋能的图学习框架，通过结合 EHR 统计证据和类型约束 LLM 推理构建全局医学知识图谱，再用 LLM 生成节点描述和边理由丰富为文本属性图，最终联合训练 LoRA 微调的 LLaMA 编码器和异构 GNN 学习统一的医学概念嵌入，在 MIMIC-III/IV 上显著提升诊断预测性能。

## 研究背景与动机

**领域现状**：EHR 挖掘中学习高质量的医学概念表示（诊断/药物/手术代码的嵌入）是临床预测的基础。现有方法主要利用医学本体的层级结构（如 ICD 的父子关系）或有限的跨类型语义（如 UMLS）来构建知识图谱指导表示学习。

**现有痛点**：(1) 现有本体中跨类型依赖关系（如诊断-药物治疗关系、药物-手术关联）大量缺失或不完整；(2) 丰富的临床语义通常以文本形式存在但难以与 KG 结构集成；(3) 无约束的 LLM 提示可能产生看似合理但无支撑的边，且输出不一致。

**核心矛盾**：LLM 编码了广泛的生物医学知识，但用于临床建模的 KG 推断必须保持证据基础、类型感知和全局一致性——需要在 LLM 的语义丰富性与 EHR 的实证支撑之间取得平衡。

**本文目标**：构建一个临床可解释且有实证支撑的异构 KG，并学习融合文本语义和图结构的统一医学概念嵌入。

**切入角度**：先从 EHR 中提取统计显著的代码对作为候选关系，再用 LLM 在类型约束和证据条件下推断语义关系类型——"统计过滤 + LLM 推断"的双保险策略。

**核心 idea**：EHR 统计证据提供实证基础，LLM 提供语义解释和关系类型——两者互补构建 KG，然后通过 LLM-GNN 联合学习融合文本和结构信息。

## 方法详解

### 整体框架

CoMed 分四步：(1) 从 EHR 中提取共现和时序转移统计，保留统计显著的代码对；(2) 用类型约束的 LLM 提示为每对代码推断有向关系类型、置信度和理由；(3) 用 LLM 生成节点描述和边元数据丰富 KG；(4) 联合训练 LoRA 微调 LLaMA-1B 编码器和异构 GNN 学习概念嵌入。

### 关键设计

1. **EHR 统计证据提取与过滤**:

    - 功能：从数据中发现有实证支撑的候选关系
    - 核心思路：对每对代码计算三种统计量——平滑条件概率、PMI 关联度和卡方独立性检验 p 值。同时计算院内共现和跨次就诊时序转移两种设置。过滤掉低支持度、低关联和非显著（p>0.05）的代码对
    - 设计动机：纯 LLM 推断容易幻觉，统计过滤确保每条候选边在目标 EHR 数据集中有实际观测支撑——关系不仅"临床合理"还"在本数据集中确实存在"

2. **类型约束的 LLM 关系推断**:

    - 功能：为统计显著的代码对推断语义关系类型
    - 核心思路：为每种代码类型组合（dx-dx、rx-dx、px-dx 等）预定义候选关系池（如 causes、treats、diagnostic_of 等）。结构化 prompt 包含代码标识、频率、8 项统计指标和指标说明。LLM 返回关系标签、有向三元组、置信度分数和 50-60 词的临床推理
    - 设计动机：类型约束防止生成语义不合理的关系（如诊断"治疗"诊断）；证据条件让 LLM 综合临床知识和统计信号。临床专家审计 50 条边的平均评分 4.84/5，验证了高质量

3. **LLM-GNN 联合学习（CoMed）**:

    - 功能：融合文本语义和图结构学习统一概念嵌入
    - 核心思路：LoRA 微调的 LLaMA-1B 编码节点描述为文本嵌入，经类型特定线性投影到 GNN 空间。异构 GNN 在 KG 上进行关系感知消息传递，输出最终概念嵌入。端到端联合训练，使用两阶段 LoRA 更新调度——早期"最少更新优先"确保覆盖，后期混合低频和高频代码
    - 设计动机：GNN 擅长聚合图结构但不解释长文本；LLM 编码语义但不利用全局关系约束——联合学习让两者互补。两阶段调度解决了 mini-batch 训练中罕见代码更新不足的问题

### 损失函数 / 训练策略

使用多标签交叉熵损失训练下一次就诊诊断预测任务。CoMed 作为即插即用的概念编码器集成到标准 EHR 模型中端到端训练。

## 实验关键数据

### 主实验

**MIMIC-III 诊断预测性能对比**

| 方法 | AUPRC | F1 | Acc@15 |
|------|-------|-----|--------|
| Base Transformer | 41.00 | 33.16 | 47.20 |
| GRAM | 41.70 | 34.60 | 48.60 |
| LINKO | 44.91 | 38.20 | 52.30 |
| GraphCare | 43.35 | 35.46 | 52.76 |
| **CoMed** | **47.21** | **42.28** | **54.20** |

### 消融实验

**即插即用分析（CoMed 集成到不同 backbone）**

| Backbone | 无 CoMed | 有 CoMed | 提升 |
|----------|---------|---------|------|
| Transformer | 41.00 | 47.21 | +6.21 |
| RETAIN | ~40 | ~46 | +6 |
| GRAM | 41.70 | ~47 | +5 |

### 关键发现

- CoMed 在 MIMIC-III 上 AUPRC 从 41.00 提升到 47.21（+6.21），在所有 baseline 中排名第一
- 对罕见诊断标签（0-25% 频率）提升尤为显著——从 40.60 到 47.67（+7.07），因为 KG 关系帮助罕见概念借用关联概念的信息
- CoMed 作为即插即用概念编码器在多个 backbone 上都一致提升
- 临床专家对 LLM 推断边的评分 4.84±0.29/5，验证了 KG 的临床有效性
- MIMIC-IV 上同样有一致提升，证明跨数据集泛化性

## 亮点与洞察

- "统计过滤 + LLM 推断"的双保险 KG 构建策略确保了关系的实证支撑和语义合理性的双重保障
- 两阶段 LoRA 更新调度巧妙解决了医学代码长尾分布导致的训练不均衡问题
- 对罕见诊断的大幅提升具有重要临床意义——罕见疾病往往是最难预测也最需要关注的

## 局限与展望

- LLM 生成的节点描述和关系推理可能包含细微的幻觉或偏差
- 仅在诊断预测任务上评估，未验证在药物推荐、再入院预测等任务上的效果
- KG 构建依赖目标数据集的统计量，不同医院的 EHR 可能产生不同的 KG
- LLaMA-1B 的文本编码能力有限，更大的 LLM 可能带来更好的嵌入

## 相关工作与启发

- **vs GRAM**: GRAM 仅用 ICD 层级结构，CoMed 引入跨类型关系和文本语义——AUPRC +5.51
- **vs GraphCare**: 后者用外部医学 KG 但不与 EHR 数据对齐，CoMed 通过统计过滤确保实证支撑
- **vs LINKO**: 后者用链接预测构建 KG 但不融合文本语义，CoMed 的 LLM-GNN 联合学习更全面

## 评分

- 新颖性: ⭐⭐⭐⭐ EHR 统计 + LLM 推断的 KG 构建思路和 LLM-GNN 联合学习框架新颖
- 实验充分度: ⭐⭐⭐⭐⭐ MIMIC-III/IV × 多 baseline + 即插即用分析 + 临床专家验证
- 写作质量: ⭐⭐⭐⭐ 方法流程清晰，每步设计有明确动机
- 价值: ⭐⭐⭐⭐⭐ 即插即用概念编码器对 EHR 研究社区价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](../../AAAI2026/medical_imaging/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)

</div>

<!-- RELATED:END -->
