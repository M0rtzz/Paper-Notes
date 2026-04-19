---
title: >-
  [论文解读] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?
description: >-
  [ACL2025][LLM/NLP][场景图] 构建 TSG Bench（120 场景、2041 描述、4289 文本场景图），系统评估 11 个 LLM 在场景图理解（SGDS/SGQA）和生成（SA-SGG/MA-SGG）四类任务上的能力，发现最强模型在理解上接近人类但生成任务仍有 15-17% 的差距。
tags:
  - ACL2025
  - LLM/NLP
  - 场景图
  - LLM评估
  - 结构化推理
  - benchmark
  - 多模态理解
---

# LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?

**会议**: ACL 2025  
**arXiv**: [2505.19510](https://arxiv.org/abs/2505.19510)  
**代码**: [GitHub](https://github.com/docworlds/tsg-bench)  
**领域**: LLM/NLP  
**关键词**: 场景图, 文本场景图, LLM评估, 结构化推理, 空间关系

## 一句话总结
构建 TSG Bench（120 场景、2041 描述、4289 文本场景图），系统评估 11 个 LLM 在场景图理解（SGDS/SGQA）和生成（SA-SGG/MA-SGG）四类任务上的能力，发现最强模型在理解上接近人类但生成任务仍有 15-17% 的差距。

## 研究背景与动机
**领域现状**：场景图（Scene Graph）是视觉理解中广泛使用的结构化表示，用节点表示实体、边表示关系、属性描述状态。传统场景图研究聚焦视觉场景图生成（Visual SGG），依赖图像输入和目标检测。近年来文本场景图（Textual Scene Graph, TSG）——从文本描述直接构建场景图——成为新兴方向。

**现有痛点**：(a) 缺乏系统评估 LLM 处理场景图能力的基准；(b) 现有场景图数据集（如 Visual Genome）聚焦视觉模态，文本场景图数据集规模小且场景单一；(c) 不清楚 LLM 在"理解已有场景图"和"从零生成场景图"两种模式下的能力差距。

**核心矛盾**：LLM 具备强大的语言理解和生成能力，但场景图需要精确的实体识别、关系推理、属性绑定和跨场景一致性——这些结构化推理能力是否在 LLM 中存在？

**本文目标**：构建全面的文本场景图基准 TSG Bench，设计覆盖理解→生成的四类任务，并系统评估 11 个主流 LLM 的场景图处理能力。

**切入角度**：将场景图任务拆解为"理解"和"生成"两大类、进一步细分为四个子任务，从简单到复杂逐步测试 LLM 的能力边界。

**核心 idea**：LLM 在场景图理解上已接近人类水平（>90%），但在生成上仍存在显著差距——尤其是在隐式动作推理、数值推理和多场景分解等能力上。

## 方法详解

### 整体框架
TSG Bench 包含三个层次：
- **数据构建**：120 个真实场景 → 2041 条自然语言描述 → 4289 个标注场景图（每条描述对应 1-3 个场景图，覆盖不同粒度）
- **任务设计**：4 类任务覆盖理解→生成的完整频谱
- **评估体系**：自动评估指标 + 人工评估校准

### 关键设计

1. **数据集构建（TSG Bench）**：

    - 120 个场景覆盖日常生活、户外活动、工作场景等多样化情境
    - 每个场景由人工撰写多条描述（简短/详细/多步骤），总计 2041 条
    - 场景图标注采用三元组格式 (subject, relation, object)，附带属性标注
    - 关键标注原则：保留隐式动作（如"她把杯子递给他"隐含"拿起"动作）、保留数值信息（如"三个孩子"）、保留时序信息（多步骤场景的状态变化）

2. **场景图理解任务**：

    - **SGDS（Scene Graph Description Selection）**：给定场景图，从多个候选描述中选择最匹配的。测试 LLM 将结构化图映射回自然语言的能力。干扰项设计：替换实体、修改关系、增删三元组
    - **SGQA（Scene Graph Question Answering）**：给定场景图，回答关于实体/关系/属性的问题。包括存在性判断（"图中是否有猫？"）、关系查询（"谁在桌子旁边？"）、计数（"有几个人？"）、属性查询（"杯子是什么颜色？"）

3. **场景图生成任务**：

    - **SA-SGG（Single Action Scene Graph Generation）**：给定单一动作描述，生成完整场景图。测试基本的结构化信息提取能力
    - **MA-SGG（Multi-Action Scene Graph Generation）**：给定包含多个动作/状态变化的复杂描述，生成多个关联场景图。测试场景分解、时序推理和跨图一致性

4. **评估指标**：

    - 理解任务：准确率（SGDS 为选择题准确率，SGQA 为答案匹配率）
    - 生成任务：三元组精确率/召回率/F1（精确匹配 + 模糊匹配两种模式）、实体覆盖率、关系正确率
    - 人工评估：完整性、正确性、一致性三个维度的 1-5 评分

## 实验关键数据

### 主实验——理解任务

| 模型 | SGDS 准确率 | SGQA 准确率 | 平均 |
|------|-----------|-----------|------|
| Claude-3.5-Sonnet | **98.4%** | **90.6%** | **94.5%** |
| GPT-4o | 96.7% | 88.2% | 92.5% |
| GPT-4o-mini | 94.1% | 84.7% | 89.4% |
| Gemini-1.5-Pro | 95.3% | 86.9% | 91.1% |
| Llama-3-70B | 89.2% | 78.4% | 83.8% |
| Llama-3-8B | 81.6% | 69.3% | 75.5% |
| Qwen-2-72B | 91.8% | 82.1% | 87.0% |
| Human | 99.1% | 95.3% | 97.2% |

### 主实验——生成任务

| 模型 | SA-SGG F1 | MA-SGG F1 | 平均 |
|------|----------|----------|------|
| Claude-3.5-Sonnet | **68.4%** | **58.8%** | **63.6%** |
| GPT-4o | 65.2% | 55.4% | 60.3% |
| GPT-4o-mini | 59.8% | 48.7% | 54.3% |
| Gemini-1.5-Pro | 63.1% | 53.2% | 58.2% |
| Llama-3-70B | 54.6% | 42.3% | 48.5% |
| Llama-3-8B | 41.2% | 31.8% | 36.5% |
| Human | **82.5%** | **75.6%** | **79.1%** |

### 消融实验——Few-shot ICL 效果

| 模型 | MA-SGG 0-shot | MA-SGG 5-shot | MA-SGG 10-shot | 提升 |
|------|-------------|--------------|---------------|------|
| GPT-4o | 55.4% | 63.1% | 68.2% | +12.8% |
| Claude-3.5-Sonnet | 58.8% | 67.5% | 71.9% | +13.1% |
| Llama-3-70B | 42.3% | 51.8% | 55.6% | +13.3% |

### 关键发现
1. **理解 >> 生成**：所有模型在理解任务上表现远好于生成任务（平均差距 25-30%），理解已接近人类但生成仍有 15-17% 差距
2. **规模效应显著**：70B 模型比 8B 模型在生成任务上高 12-15%，但理解任务差距较小（~8%）
3. **三大失败模式**：(a) 隐式动作遗漏——模型倾向只提取显式描述的动作，丢失隐含的前置/后置动作；(b) 数值推理不足——"三个孩子在玩"生成时常丢失数量信息；(c) 多场景分解困难——MA-SGG 中模型难以确定场景边界
4. **10-shot ICL 显著有效**：在最难的 MA-SGG 任务上，10-shot 示例平均带来 ~13% 的提升，说明场景图格式可通过示例有效教给 LLM

## 亮点与洞察
- **理解-生成不对称性的量化**：首次用精确数据量化了 LLM 在结构化表示上"理解容易、生成难"的现象，这与代码生成领域的发现一致
- **TSG Bench 的数据质量**：120 场景 × 多粒度描述 × 多层次标注，数据设计精致——干扰项构造、隐式动作标注、多场景分解标准都经过仔细设计
- **ICL 的有效性**：10-shot 示例能弥补约一半的人机差距，暗示场景图生成更多是"格式学习"问题而非"能力缺失"
- **人类基线设定合理**：人类在理解上 97%、生成上 79%（不是 100%），说明任务本身有真实的挑战性

## 局限与展望
- **纯文本场景图**：未与视觉模态结合——VLM 从图像直接生成场景图的能力对比是自然的下一步
- **场景类型偏向日常**：120 场景主要是日常生活，缺少专业领域（医学影像描述、工程场景、科学实验描述等）
- **评估指标粗糙**：三元组精确匹配+模糊匹配的二元划分过于简单——"猫坐在桌上"和"猫在桌旁"的语义相似度应有连续量化
- **多模型对比数量有限**：11 个 LLM 已不少，但缺少 DeepSeek、Mistral 等新兴开源模型
- **未探索微调**：所有实验均为 zero/few-shot，若在 TSG 数据上微调小模型能否接近大模型水平？

## 相关工作与启发
- **与 Visual SGG 的关系**：传统 Visual SGG（如 Motif-Net, VCTree）从图像生成场景图，本文完全在文本空间操作——两者互补，可为多模态场景图研究提供桥梁
- **与结构化输出研究的联系**：场景图生成本质上是"从非结构化文本提取结构化表示"——与信息抽取（IE）、知识图谱构建（KG Construction）共享核心挑战
- **启发**：场景图可以作为 LLM 结构化推理能力的通用探针——比纯文本 benchmark 更能暴露模型在实体绑定和关系推理上的弱点

## 评分
- 新颖性: ⭐⭐⭐⭐ 文本场景图评估基准是新颖切入点，四类任务设计覆盖理解→生成频谱
- 实验充分度: ⭐⭐⭐⭐ 11 个模型 × 4 类任务 × 消融/人类基线，实验设计全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务定义明确，数据和指标描述详细
- 价值: ⭐⭐⭐⭐ TSG Bench 作为评估工具有实用价值，四类任务的设计可复用于其他结构化表示研究

**会议**: ACL2025  
**arXiv**: [2505.19510](https://arxiv.org/abs/2505.19510)  
**代码**: [GitHub](https://github.com/docworlds/tsg-bench)  
**领域**: llm_nlp  
**关键词**: 场景图, LLM评估, 结构化推理, benchmark, 多模态理解

## 一句话总结

<!-- RELATED:START -->

## 相关论文

- [AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](aad-llm_neural_attention-driven_auditory_scene_understanding.md)
- [Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)
- [Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'](can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)

<!-- RELATED:END -->
