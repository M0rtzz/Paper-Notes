---
title: >-
  [论文解读] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases
description: >-
  [ICLR 2026 (Gen2 Workshop)][医学图像][Text-to-SQL] 提出 BiomedSQL，首个专门评估 Text-to-SQL 系统在生物医学知识库上科学推理能力的基准，包含 68,000 个问题/SQL/答案三元组，揭示当前最强模型（GPT-o3-mini 62.6%）与领域专家（90%）之间仍有巨大差距。
tags:
  - ICLR 2026 (Gen2 Workshop)
  - 医学图像
  - Text-to-SQL
  - 生物医学知识库
  - 科学推理
  - BigQuery
  - LLM评估
---

# BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases

**会议**: ICLR 2026 (Gen2 Workshop)  
**arXiv**: [2505.20321](https://arxiv.org/abs/2505.20321)  
**代码**: [https://github.com/NIH-CARD/biomedsql](https://github.com/NIH-CARD/biomedsql)  
**领域**: 生物医学 NLP / Text-to-SQL  
**关键词**: Text-to-SQL, 生物医学知识库, 科学推理, BigQuery, LLM评估

## 一句话总结
提出 BiomedSQL，首个专门评估 Text-to-SQL 系统在生物医学知识库上科学推理能力的基准，包含 68,000 个问题/SQL/答案三元组，揭示当前最强模型（GPT-o3-mini 62.6%）与领域专家（90%）之间仍有巨大差距。

## 研究背景与动机

现代生物医学研究日益依赖大规模结构化数据库，研究人员需要频繁查询电子健康记录、高通量实验数据和人群规模研究。自然语言接口（尤其是 Text-to-SQL 系统）有望让非技术研究人员也能访问这些资源。

**现有痛点**：当前的 Text-to-SQL 系统将查询生成视为"语法翻译"任务，将问题结构映射到 SQL 模板，缺乏对领域知识的深入理解。在生物医学场景下，这种抽象会失效——领域专家常问"哪些 SNP 与阿尔茨海默病显著相关？"或"哪些已批准药物靶向帕金森病中上调的基因？"，这些问题隐含了统计阈值（如 GWAS 显著性 p < 5×10⁻⁸）、药物审批流程和跨模态因果推理等领域特定知识。

**核心矛盾**：通用 Text-to-SQL 基准（如 Spider、BIRD）不评估科学推理能力；EHR 导向基准（如 EHRSQL）侧重时间逻辑和患者检索，而非科学发现所需的推理。

**切入角度**：构建首个专门评估生物医学领域 Text-to-SQL 科学推理能力的大规模基准。

## 方法详解

### 整体框架
输入：自然语言的生物医学问题 + 数据库 schema 信息。输出：LLM 生成 SQL 查询 → 执行获取结果 → 生成自然语言回答。评估同时覆盖 SQL 执行准确性和自然语言回答质量。

### 关键设计

1. **关系数据库构建**：

    - 整合 10 个核心表，来源包括 OpenTargets Platform（基因-疾病-药物关联）、ChEMBL（生物活性分子和药理学数据）
    - 纳入阿尔茨海默病和帕金森病的 GWAS 统计摘要数据（含 p 值、rsID、等位基因频率等）
    - 整合 omicSynth 的因果推理数据（基于孟德尔随机化的多组学生物标志物）
    - 所有数据以 Parquet 格式上传到 Google BigQuery

2. **SQL 标注与扩增**：

    - 领域专家手工编写 40 个种子问题的 gold-standard SQL 查询
    - 通过模板化和实体替换将 40 个查询自动扩展为 68,000 个 QA 对
    - 所有生成的 SQL 均在 BigQuery 上执行获取 ground-truth 结果

3. **BMSQL 多步骤 Agent**：

    - 自定义的迭代式 Text-to-SQL 架构，模拟专家查询过程
    - 第一步：Schema 分析，识别相关表和列
    - 第二步：生成初始 SQL 查询
    - 第三步：如有语法错误则修正（最多3次重试）
    - 第四步：应用统计阈值过滤（如 p 值显著性）
    - 第五步：基于两组执行结果生成自然语言回答
    - 可选执行额外推理步骤（inference-time compute）

### 科学推理分类
三大推理类别：
1. **操作化隐含科学惯例**：需推断 GWAS 显著性阈值、效应方向性等
2. **整合缺失的上下文知识**：需理解药物审批状态、临床试验阶段等
3. **执行复杂多跳推理**：需跨多表串联关系操作

### 评估指标
- Execution Accuracy (EX)：SQL 执行结果精确匹配率
- Jaccard Index (JAC)：结果集交并比
- Syntax Error Rate (SER)：语法错误率
- BioScore（LLM-as-judge）：Response Quality Rate (RQR) + Safety Rate (SR)

## 实验关键数据

### 主实验

| 模型 | EX↑ | JAC↑ | RQR↑ | SR↑ | SER↓ |
|------|-----|------|------|-----|------|
| 领域专家 | 90.0 | 90.0 | 95.0 | - | - |
| GPT-o3-mini | 53.5 | 60.4 | 73.3 | 29.4 | 0.0 |
| GPT-4o | 46.9 | 54.7 | 71.2 | 26.1 | 1.3 |
| Claude-3.7-sonnet | - | - | - | 43.0 | - |
| Qwen-2.5-Coder-32B | 40.8 | - | - | - | - |
| **BMSQL-GPT-o3-mini** | **62.6** | **69.2** | **83.1** | 38.0 | 2.6 |
| **BMSQL-Gemini** | - | - | **84.6** | - | - |

### 交互范式实验

| 方法 | EX↑ | JAC↑ | RQR↑ |
|------|-----|------|------|
| ReAct-GPT-o3-mini | 56.2 | 64.8 | 73.6 |
| Index-GPT-o3-mini | - | - | - (最高SR 66.9%) |
| BMSQL-GPT-o3-mini | 62.6 | 69.2 | 83.1 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Combo prompt（3-rows+3-shot+stat-instruct） | EX +5.5%, RQR +4.5% | 但 token 消耗增加近 3 倍 |
| 1-pass vs 3-pass（推理时间计算） | EX 62.6→61.7, RQR 83.1→85.5 | 增加推理步数收益甚微 |
| 单独添加表格行示例 | 可忽略改进 | schema 理解 > 内容记忆 |

### 关键发现
- 最强模型 BMSQL-GPT-o3-mini 的 EX 仅 62.6%，距专家基准 90% 有约 30% 差距
- Join、Similarity Search 和 Multi-Filter 类查询最具挑战性
- 增加推理步数带来的改进极为有限，主要修正语法错误而非重构查询逻辑
- schema-level 理解比记忆原始数据行更重要
- 小模型 Qwen-2.5-Coder 在部分指标上超过参数量远大于它的 LLaMA 模型

## 亮点与洞察
- 首个专注于生物医学领域科学推理的 Text-to-SQL 基准，填补重要空白
- 68,000 个问题规模庞大，且每个问题都需要隐含的领域知识推理
- 多维评估体系设计合理（SQL 执行指标 + 自然语言回答质量）
- BMSQL 的多阶段设计模拟专家查询流程，效果显著优于单步方法
- 揭示了当前 LLM 在操作化领域特定科学惯例方面的重大不足
- 使用 BigQuery 模拟真实生产环境，增加了实际部署的相关性

## 局限与展望
- Gold SQL 并非唯一正确答案，可能存在多个语义等价的 SQL 表达
- 未评估 DIN-SQL、DAIL-SQL 等通用 Text-to-SQL 系统（与 BigQuery 方言不兼容）
- 依赖 BigQuery 云特定方言，限制了与其他基准的直接可比性
- 数据集通过模板扩展，可能引入系统性偏差
- 领域覆盖主要限于神经退行性疾病（阿尔茨海默和帕金森）

## 相关工作与启发
- 与 Spider、BIRD 等通用 Text-to-SQL 基准互补，专注科学推理维度
- 与 EHRSQL、MIMICSQL 等临床数据基准不同，面向科学知识发现而非患者检索
- 与 SciFact、EntailmentBank 等科学推理基准相关，但评估 SQL 生成能力
- 启发：可将类似方法推广到其他领域（如材料科学、环境科学）的知识库查询

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](../../NeurIPS2025/medical_imaging/cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)
- [Augmenting Representations with Scientific Papers](augmenting_representations_with_scientific_papers.md)
- [Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](../../ACL2025/medical_imaging/urca_biomedical_evidence_extraction.md)
- [Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)

<!-- RELATED:END -->
