---
title: >-
  [论文解读] RePanda: Pandas-powered Tabular Verification and Reasoning
description: >-
  [ACL 2025][表格事实验证] 提出 RePanda，将自然语言声明翻译为可执行的 pandas 查询来实现表格事实验证，在 TabFact 上达到 84.09% 准确率，在 OOD 的 WikiFact 上无需额外微调达 84.72%，同时以仅 7B 参数的模型逼近 671B DeepSeek-Chat 的零样本性能，并扩展至表格问答任务取得 75.1% 准确率。
tags:
  - "ACL 2025"
  - "表格事实验证"
  - "pandas查询"
  - "可解释推理"
  - "知识蒸馏"
  - "执行型推理"
  - "OOD泛化"
---

# RePanda: Pandas-powered Tabular Verification and Reasoning

**会议**: ACL 2025  
**arXiv**: [2503.11921](https://arxiv.org/abs/2503.11921)  
**代码**: [datasets/AtoosaChegini/PanTabFact](https://huggingface.co/datasets/AtoosaChegini/PanTabFact)  
**作者**: Atoosa Malemir Chegini, Keivan Rezaei, Hamid Eghbalzadeh, Soheil Feizi  
**机构**: University of Maryland, AI at Meta  
**领域**: 其他  
**关键词**: 表格事实验证, pandas查询, 可解释推理, 知识蒸馏, 执行型推理, OOD泛化

## 一句话总结

提出 RePanda，将自然语言声明翻译为可执行的 pandas 查询来实现表格事实验证，在 TabFact 上达到 84.09% 准确率，在 OOD 的 WikiFact 上无需额外微调达 84.72%，同时以仅 7B 参数的模型逼近 671B DeepSeek-Chat 的零样本性能，并扩展至表格问答任务取得 75.1% 准确率。

## 研究背景与动机

**领域现状**：表格事实验证要求模型在结构化数据上进行数值推理、过滤和比较。现有方法如 TAPAS、TAPEX、PASTA 通过表格感知预训练提升表格理解能力，但存在两大核心问题。

**核心问题**：
   - **结构理解不足**：LLM 以顺序文本方式预训练，将表格展平为序列后会丢失行列间的结构关系；TAPAS/TAPEX 虽有表格位置编码，但在聚合、多行比较等复杂操作上仍力不从心
   - **可解释性缺失**：现有方法作为黑盒分类器，仅输出 True/False 而无法展示推理过程，在法律审计、金融等高风险场景下无法验证决策依据

**关键洞察**：pandas 查询天然为表格操作设计（过滤、计数、聚合等），将声明翻译为 pandas 查询既能提供透明推理步骤，又能通过执行获得可验证的结果

## 方法详解

### 整体框架

RePanda 的核心思路：将事实验证重新定义为**结构化表示学习任务**——模型学会生成可执行的 pandas 查询而非直接分类。

形式化定义：给定表格 𝒯 和声明 s，模型 f_θ 输出 pandas 查询 q_s = f_θ(s, 𝒯)，执行 q_s 在 𝒯 上产生验证结果。

### 数据集构建

**PanTabFact（事实验证数据集）**：
- 基于 TabFact 训练集构建
- 使用 DeepSeek-Chat（671B）为每个声明-表格对生成对应的 pandas 查询
- 每个查询编码逻辑操作步骤（过滤、聚合、比较等）

**PanWiki（问答数据集）**：
- 基于 WikiTableQuestions 构建
- 将每个问题转化为 pandas 查询，执行后获得答案
- 仅 **1200** 个训练样本

### 错误纠正流水线

三阶段自动纠正确保数据质量：

1. **逻辑纠正**：验证 pandas 查询执行结果是否与期望答案一致；不一致则将原始查询和期望结果交给模型重新生成（仅用于训练数据创建）
2. **语法纠正**：迭代修复运行时错误——将错误信息反馈给模型进行修正（训练和推理时均使用）
3. **过滤**：移除执行失败或结果不匹配的样本

### 模型训练

- **基础模型**：DeepSeek-coder-7B-instruct-v1.5
- **训练方式**：自回归生成 pandas 查询
- **损失函数**：负对数似然
  $$\mathcal{L} = -\sum_{t=1}^{T} \log P(q_t | q_{<t}, s, \mathcal{T}; \theta)$$
- **训练细节**：AdamW 优化器，学习率 2e-4，cosine 调度，4 epochs，batch size 4
- 事实验证查询输出布尔值，问答查询输出具体答案

### OOD 泛化设计

- 构建 WikiFact：将 WikiTableQuestions 的问答对转化为事实验证声明
- 不做额外微调，直接用 PanTabFact 训练的模型进行评估
- 测试跨表格结构和领域迁移的能力

## 实验

### 主实验结果

**In-Distribution（TabFact 测试集，Table 1）**：

| 方法 | 准确率 |
|------|--------|
| RePanda（事实验证） | **84.09%** |
| Finetuned-Direct | 67.85% |
| ZeroShot-Pandas | 51.82% |
| ZeroShot-Direct | 50.76% |

RePanda 比直接分类微调高 **16.24%**，比零样本高 **33.33%**。

**OOD 泛化（WikiFact，Table 2）**：

| 方法 | 准确率 |
|------|--------|
| RePanda（事实验证） | **84.72%** |
| Finetuned-Direct | 74.10% |
| ZeroShot-Pandas | 59.92% |
| ZeroShot-Direct | 53.20% |

无需额外微调，OOD 准确率（84.72%）甚至略高于 in-distribution（84.09%）。

### 与 SOTA 方法对比（WikiFact 平衡数据集，Table 3）

在 300 原始（全正例）+ 300 修改（全反例）的平衡数据上：

| 方法 | All False | All True | Overall |
|------|-----------|----------|---------|
| RePanda | **88.33%** | **85.67%** | **87.00%** |
| TAPEX | 41.00% | 59.33% | 50.16% |
| TAPAS | 55.00% | 65.33% | 60.16% |
| PASTA | 47.67% | 51.67% | 49.67% |

RePanda 整体超出 TAPAS 约 **27 个百分点**，证明真正学到了判别能力而非简单偏向正类。

### 与 DeepSeek-Chat (671B) 零样本对比

| 数据集 | RePanda (7B) | DeepSeek-Chat (671B) |
|--------|-------------|---------------------|
| TabFact | **84.09%** | 82.62% |
| WikiFact | 84.72% | 85.39% |

7B 模型在 TabFact 上甚至超过 671B 模型，成功实现结构化推理能力的知识蒸馏。

### 表格问答结果（WikiTableQuestions, Table 4）

| 方法 | 准确率 | 训练数据量 |
|------|--------|-----------|
| TabLaP | 76.6% | 大规模 |
| SynTQA (GPT) | 74.4% | 大规模 |
| **RePanda** | **75.1%** | **1,200** |
| Mix SC | 73.6% | 大规模 |
| Chain-of-Table | 67.31% | 大规模 |

仅用 1200 个训练样本即达到了与 SOTA 方法竞争的准确率。

### 消融实验：错误纠正的影响（Table 5）

| 数据集 | 无纠正 | 有纠正 | 提升 |
|--------|--------|--------|------|
| TabFact | 78.02% | 84.09% | +6.07% |
| WikiFact | 74.43% | 84.72% | +10.29% |
| WQA | 67.59% | 75.1% | +7.51% |

错误纠正在所有任务上都带来显著提升，OOD 场景提升最大（+10.29%）。

## 亮点与洞察

1. **执行型推理的可解释性优势**：生成的 pandas 查询本身就是推理过程，用户可逐步验证逻辑是否正确——这在黑盒模型中不可能实现
2. **惊人的 OOD 泛化能力**：在完全未见过的 WikiFact 表格上达到 84.72%，证明模型学到的是通用的表格推理模式而非特定数据分布
3. **高效知识蒸馏**：7B 模型通过结构化推理学习成功逼近甚至超越 671B 模型，说明中间表示（pandas 查询）是比直接分类更高效的知识传递媒介
4. **极低数据需求**：表格 QA 仅需 1200 样本即达 SOTA 水平，显示执行框架对训练数据的利用效率极高
5. **自动错误纠正的重要性**：推理时的语法纠正（4 轮迭代修复）贡献了约 6-10 个百分点的性能提升

## 局限性

1. **仅支持单表推理**：所有实验基于单表格场景，未验证多表连接、跨表推理等更复杂的结构化推理能力
2. **依赖 DeepSeek-Chat 生成训练数据**：PanTabFact 和 PanWiki 的查询均由 671B 模型生成，数据构建成本较高
3. **语法纠正增加推理延迟**：推理时最多需 4 轮迭代修复语法错误，对延迟敏感的在线场景可能不适用
4. **pandas 查询的表达力边界**：极端复杂的推理（涉及统计建模、模式发现等）可能超出 pandas 查询的表达能力
5. **评估范围有限**：仅在 TabFact 和 WikiTableQuestions 两个基准上验证，更广泛的表格推理基准（如 HybridQA、SQA 等）未测试

## 相关工作

- **表格事实验证**：Table-BERT (Chen et al., 2019)、TAPAS (Herzig et al., 2020)、TAPEX (Liu et al., 2021)、PASTA (Gu et al., 2022)
- **结构化表示学习**：ProgVGAT (Yang et al., 2020)、ReasTAP (Zhao et al., 2022)、StructGPT (Jiang et al., 2023)
- **表格问答**：Chain-of-Table (Wang et al., 2024b)、TabLaP (Wang et al., 2024a)、SynTQA (Zhang et al., 2024)
- **工具增强推理**：ReAct (Yao et al., 2023)、Chain-of-Thought (Wei et al., 2022)

## 评分

⭐⭐⭐⭐ — 方法清晰实用、可解释性强、OOD 泛化能力出色、数据效率极高，但仅支持单表格推理是明显短板，且错误纠正的多轮迭代增加了推理成本。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)
- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [\[ACL 2025\] Temporal Reasoning for Timeline Summarisation in Social Media](temporal_reasoning_for_timeline_summarisation_in_social_media.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ACL 2025\] Commonsense Reasoning in Arab Culture](commonsense_arab_culture.md)

</div>

<!-- RELATED:END -->
