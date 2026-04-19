---
title: >-
  [论文解读] Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts
description: >-
  [AAAI 2026][多模态][科学声明验证] 本文系统研究了多模态LLM在使用表格和图表作为证据验证科学声明时的鲁棒性，通过扩展SciTabAlign和ChartMimic两个数据集构建了表格-图表对齐的评估基准，发现12个多模态LLM在表格证据上的表现一致优于图表证据，而人类在两种格式上表现一致，揭示了当前模型在图表理解方面的关键短板。
tags:
  - AAAI 2026
  - 多模态
  - 科学声明验证
  - 表格vs图表
  - 多模态鲁棒性
  - 格式敏感性
  - 证据格式
---

# Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts

**会议**: AAAI 2026  
**arXiv**: [2511.10075](https://arxiv.org/abs/2511.10075)  
**代码**: [https://github.com/Alab-NII/tables-vs-charts](https://github.com/Alab-NII/tables-vs-charts)  
**领域**: 多模态VLM  
**关键词**: 科学声明验证, 表格vs图表, 多模态鲁棒性, 格式敏感性, 证据格式

## 一句话总结
本文系统研究了多模态LLM在使用表格和图表作为证据验证科学声明时的鲁棒性，通过扩展SciTabAlign和ChartMimic两个数据集构建了表格-图表对齐的评估基准，发现12个多模态LLM在表格证据上的表现一致优于图表证据，而人类在两种格式上表现一致，揭示了当前模型在图表理解方面的关键短板。

## 研究背景与动机

### 领域现状
科学声明验证 (Scientific Claim Verification) 要求模型判断给定的声明是否被提供的证据所支持。随着生成式AI时代科学论文数量激增，自动化审稿辅助系统的需求日益迫切。实验结果通常以表格或图表形式呈现，是论文的核心组成部分。

### 现有痛点

**现有数据集格式单一**：大多数声明验证数据集的证据只有一种格式——要么全是文本表格（如SciTab），要么全是图表（如MuSciClaims）。即使SciVer包含多种模态，其中表格也以图像而非结构化文本呈现

**缺乏对齐评估**：没有数据集能提供表达**相同信息**的表格和图表配对，因此无法公平评估模型在不同格式上的表现

**模型偏见未知**：多模态LLM是否在处理语义等价但格式不同的证据时表现一致？这个关键问题尚未被系统研究

### 核心矛盾
一个可靠的审稿辅助系统必须能够准确验证声明，无论支持证据以何种格式呈现。如果LLM在某一格式上表现优异但在另一格式上表现糟糕，将导致有偏或不完整的评估。

### 切入角度
构建表格-图表语义对齐的数据集，在表格独立、图表独立和混合输入三种设置下系统比较模型表现，同时纳入人类评估作为上界参照。

## 方法详解

### 整体框架
本文的方法论核心是**数据集构建+系统评估**：
1. 扩展两个现有数据集为"增强版"（SciTabAlign+ 和 ChartMimic+）
2. 在三种输入设置下评估12个多模态LLM
3. 进行人类评估验证任务难度的合理性

### 关键设计

#### 1. **SciTabAlign+ 数据集构建**
- **来源**：SciTabAlign（SciTab的扩展版，已移除模糊案例）
- **数据清洗**：移除HTML/括号标签、标准化数值，从136个表格中保留70个（162条声明）
- **四种图表类型生成**：
    - **基础柱状图**：不同颜色表示不同柱
    - **符号柱状图**：用"/"或"-"符号替代颜色
    - **折线图**：连接数据点
    - **交换图**：将x轴的方法和指标互换
- **设计动机**：多样化的图表类型可以测试模型对不同视觉编码方式的理解能力
- 最终：372条表格声明 + 648条图表声明（162×4种图表）

#### 2. **ChartMimic+ 数据集构建**
- **来源**：ChartMimic（chart2code任务数据集）的Direct Mimic子集
- 筛选70个折线图和80个柱状图
- 利用Python代码自动提取底层表格数据
- 由4名NLP研究者验证、编辑表格并编写支持/反驳声明
- 最终：152条声明（52个柱状图+24个折线图）
- **设计动机**：ChartMimic来自真实学术论文，图表质量高，配合代码可精确提取底层数据

#### 3. **评估设置**
- **三种输入模式**：表格独立（Table-only）、图表独立（Chart-only）、表格+图表组合
- **公平性处理**：图表设置中将声明中的"Table X"替换为"Figure X"
- **评估方法**：使用零样本CoT prompting，macro-F1为主指标
- **12个模型**：涵盖4个家族——InternVL3 (1B-38B)、Qwen-VL 2.5 (3B-72B)、LLaVA-v1.6 (7B-34B)、Llama-3.2 (11B)

### 人类评估
- 与模型使用相同的随机样本
- 验证任务本身的难度 vs 模型能力的差距

## 实验关键数据

### 主实验（SciTabAlign+ Macro-F1）

| 模型 | Table-only (全部) | Table (162) | Chart Avg | Chart+Table |
|------|------------------|-------------|-----------|-------------|
| Qwen2.5-VL-72B | **88.5** | 86.3 | 68.5 | 88.0 |
| InternVL3-38B | 80.7 | 82.4 | 62.5 | **88.8** |
| Qwen2.5-VL-32B | 84.6 | 86.2 | 67.6 | 86.2 |
| InternVL3-14B | 81.5 | 81.1 | 62.4 | 84.9 |
| InternVL3-8B | 69.9 | 70.4 | 55.7 | 70.2 |
| Qwen2.5-VL-7B | 75.7 | 80.0 | 58.3 | 75.9 |
| Qwen2.5-VL-3B | 52.7 | 53.6 | 39.9 | 50.4 |
| InternVL3-1B | 31.1 | 32.6 | 23.3 | 34.1 |
| LLaVA-v1.6-34B | 60.2 | 56.7 | 33.4 | 37.1 |

### 消融实验（不同图表类型的影响，12模型平均）

| 图表类型 | 平均Macro-F1 | 说明 |
|----------|-------------|------|
| 基础柱状图 | **53.0** | 最容易 |
| 折线图 | 51.9 | 次优 |
| 交换图 | 51.3 | 中等 |
| 符号柱状图 | 50.4 | 最难 |

### 表格 vs 图表性能差距（Top-5模型）

| 模型 | Table (162) | Chart Avg | 差距 |
|------|-------------|-----------|------|
| LLaVA-v1.6-34B | 56.7 | 33.4 | **23.3** |
| Qwen2.5-VL-7B | 80.0 | 58.3 | **21.7** |
| InternVL3-38B | 82.4 | 62.5 | **19.9** |
| InternVL3-14B | 81.1 | 62.4 | **18.7** |
| Qwen2.5-VL-32B | 86.2 | 67.6 | **18.6** |

### 关键发现
1. **表格一致优于图表**：12个模型中11个在表格设置下表现更好，唯一例外LLaVA-v1.6-Mistral-7B两者持平（57.6 vs 57.7）
2. **性能差距巨大**：最大差距达23.3%（LLaVA-v1.6-34B），说明模型强烈依赖结构化文本输入
3. **人类不受格式影响**：人类评估者在表格和图表两种格式上均保持优异表现，证明困难来自模型而非任务
4. **小模型跨模态泛化差**：8B以下模型在表格和图表任务的表现相关性很弱
5. **组合输入不一定更好**：对部分模型（如Qwen2.5-VL-3B、LLaVA-v1.6-34B），添加图表反而降低了性能
6. **图表+表格一致优于纯图表**：所有12个模型的组合输入都优于图表独立输入

## 亮点与洞察
- **问题定义精准**：明确区分"格式鲁棒性"和一般的多模态能力，通过语义等价的证据对照实验设计非常巧妙
- **符号柱状图的发现**：用符号替代颜色后性能进一步下降，说明模型对非标准视觉编码的理解更差
- **LLaVA-v1.6-34B的异常模式**：大模型反而在图表理解上表现最差（33.4%），且组合输入也无法提升，暗示某些架构设计可能不利于视觉推理
- **实际应用启示**：在构建自动化审稿系统时，应优先将图表转换为结构化表格再进行推理，而非直接处理图像

## 局限与展望
- 数据规模较小：SciTabAlign+仅162条声明匹配图表，ChartMimic+仅152条
- 仅关注柱状图和折线图，未涵盖热力图、散点图、箱线图等更多图表类型
- 未探索多图表/子图场景
- 仅使用零样本CoT，未测试few-shot或fine-tuning设置
- 图表由程序自动生成，可能与实际论文中手工美化的图表在复杂度上有差异
- 声明仅涉及二分类（支持/反驳），未探索"信息不足"等更复杂标签
- 缺乏对闭源大模型（GPT-4V、Gemini等）的评估

## 相关工作与启发
- SciTab (Lu et al., 2023)：科学声明验证数据集，使用真实论文声明，本文的数据基础
- ChartMimic (Yang et al., 2025)：chart2code数据集，提供高质量图表+代码配对
- SciVer (Wang et al., 2025)：多模态科学验证数据集，但表格以图像形式呈现
- ChartQA (Masry et al., 2022)、CharXiv (Wang et al., 2024)：图表问答任务，关注内容提取而非声明验证
- 启发：格式鲁棒性是多模态模型的基本要求。未来模型应加强对图表等视觉化数据的理解训练

## 评分
- 新颖性: ⭐⭐⭐⭐ （问题定义新颖，格式鲁棒性视角独特）
- 实验充分度: ⭐⭐⭐⭐ （12模型+3设置+4图表类型+人类评估，但数据规模小）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，图表说明性强，逻辑链完整）
- 价值: ⭐⭐⭐⭐ （揭示了多模态LLM的关键短板，对自动审稿系统建设有直接指导意义）

<!-- RELATED:START -->

## 相关论文

- [WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks](phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)
- [URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [Exploring LLMs for Scientific Information Extraction using the SciEx Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)

<!-- RELATED:END -->
