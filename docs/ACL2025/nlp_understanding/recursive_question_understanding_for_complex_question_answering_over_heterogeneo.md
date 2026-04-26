---
title: >-
  [论文解读] Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data
description: >-
  [ACL2025][NLP理解][个人数据问答] 提出 ReQAP 方法，通过递归问题分解构建可执行算子树，在结构化+非结构化的异构个人数据上实现复杂问答，支持端侧轻量部署。
tags:
  - ACL2025
  - NLP理解
  - 个人数据问答
  - 异构数据
  - 问题分解
  - 算子树
  - 端侧部署
---

# Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data

**会议**: ACL2025  
**arXiv**: [2505.11900](https://arxiv.org/abs/2505.11900)  
**作者**: Philipp Christmann, Gerhard Weikum (Max Planck Institute for Informatics)
**代码**: [reqap.mpi-inf.mpg.de](https://reqap.mpi-inf.mpg.de/)  
**领域**: nlp_understanding  
**关键词**: 个人数据问答, 异构数据, 问题分解, 算子树, 端侧部署

## 一句话总结

提出 ReQAP 方法，通过递归问题分解构建可执行算子树，在结构化+非结构化的异构个人数据上实现复杂问答，支持端侧轻量部署。

## 研究背景与动机

- **个人数据管理需求日益增长**：用户设备每天产生海量数据（日历、健身记录、购物记录、流媒体历史等），用户需要对这些异构数据进行便捷查询
- **数据隐私为核心约束**：个人敏感数据要求全部处理在本地设备完成，严格限制算力和内存资源
- **现有方法的两大范式各有缺陷**：
    - **Verbalization (RAG)**：将所有数据文本化喂给 LLM，但当相关事件数达数百条时超出上下文窗口，且 LLM 难以处理聚合/分组等复杂操作
    - **Translation (NL2SQL/CodeGen)**：将问题翻译为 SQL 查询，但无法处理非结构化文本（如邮件正文、社交媒体帖子），且依赖完整 schema
- **核心挑战**：如何兼顾异构数据（结构化 + 非结构化）的复杂分析型问答能力与端侧轻量化部署

## 方法详解

### 整体框架：ReQAP

ReQAP 弥合 verbalization 与 translation 两大范式，分为两个阶段：

**阶段一：QUD (Question Understanding and Decomposition)**

- 对用户问题进行**递归分解**，生成可执行的算子树
- 关键创新：不是一次性生成完整算子树（one-shot 容易出错），而是多次调用 LLM，每次生成部分算子树，将子问题留给后续递归调用
- 训练流程：先用大模型 (GPT-4o) 通过 ICL (8 个 few-shot) 生成 (问题, 算子树) 对，再蒸馏到 1B 小模型 (LLaMA-3.2-1B) 用于端侧推理

**阶段二：OTX (Operator Tree Execution)**

- 自底向上执行 QUD 生成的算子树，核心算子包括：

| 算子 | 功能 |
|------|------|
| **RETRIEVE** | 从个人数据源检索匹配事件（核心算子） |
| **EXTRACT** | 从事件中提取指定 key-value 对（核心算子） |
| JOIN | 两组事件列表按条件连接 |
| GROUP_BY | 按指定 key 分组 |
| FILTER | 按条件过滤事件 |
| MAP / APPLY | 应用函数变换 |
| ARGMIN/MAX, SUM, AVG | 聚合操作 |

### RETRIEVE 算子（五步流水线）

1. **SPLADE 初筛**：稀疏检索获取候选事件池（追求高召回）
2. **模式发现**：识别候选池中的高频 key-value 模式
3. **模式分类**：Cross-encoder 将模式分为全部相关、全部无关、部分相关
4. **事件分类**：对部分相关的事件逐条二分类
5. **去重**：时间重叠的事件合并（避免日历+邮件+社交媒体同一事件重复计数）

### EXTRACT 算子

- 使用小规模 seq2seq 模型 (BART-base) 从非结构化文本中提取语义信息
- 例：从邮件正文中 "pizza oven" 提取 cuisine="Italian"
- 效率优化：对 >=70% 输入能直接映射的 key 建立冻结映射，避免重复推理

### 数据模型

所有数据源统一为**按时间排序的事件列表**，每个事件是 key-value 字典，涵盖日历、邮件、社交媒体、健身、购物、流媒体等。

## 实验关键数据

### PerQA 基准测试集

作者构建了 PerQA 数据集：20 个虚构人物、每人约 40K 事件、3,567 个复杂问题。

**主实验结果 (Table 3)：PerQA 测试集 Hit@1 / Rlx-Hit@1**

| 方法 | GPT-4o (>100B) | LLaMA-3.3 (70B) | SFT (1B) |
|------|---------------|-----------------|----------|
| RAG | 0.149 / 0.20 | 0.123 / 0.18 | 0.029 / 0.06 |
| CodeGen | 0.319 / 0.44 | 0.239 / 0.33 | 0.315 / 0.47 |
| **ReQAP** | **0.386 / 0.52** | **0.322 / 0.46** | **0.380 / 0.53** |

- ReQAP 各变体均显著优于基线 (McNemar test, p<0.05)
- 1B SFT 版本性能接近 GPT-4o（Hit@1: 0.380 vs 0.386），Rlx-Hit@1 甚至最优 (0.53)

**消融实验 (Table 5)：PerQA dev 集**

| 变体 | Hit@1 | Rlx-Hit@1 |
|------|-------|-----------|
| ReQAP (SFT) | **0.396** | **0.54** |
| w/o 递归分解 (one-shot) | 0.356 | 0.50 |
| w/o Cross-encoder (SPLADE-only) | 0.269 | 0.36 |
| w/o EXTRACT (仅 key 匹配) | 0.138 | 0.23 |

EXTRACT 算子的移除导致性能下降最大（-65%），证明从非结构化文本提取信息是关键能力。

**不同复杂度问题的表现 (Table 4, GPT-4o)**

| 问题类型 | RAG | CodeGen | ReQAP |
|----------|-----|---------|-------|
| Ordering | 0.167 | 0.440 | **0.529** |
| Grouping | 0.172 | 0.444 | **0.537** |
| Temporal | 0.129 | 0.290 | **0.417** |
| Aggregation | 0.130 | 0.228 | **0.296** |
| Join | 0.073 | 0.176 | **0.236** |
| Multi-source | 0.196 | 0.237 | **0.365** |

### 用户研究

20 名本科生使用自己的真实数据（Docker 离线部署），28% 回答完全正确，41% 几近正确。94% 的用户问题可映射到 PerQA 中同构的算子树，说明基准设计具有代表性。

## 亮点

- **递归分解思路巧妙**：避免 one-shot 生成复杂算子树的错误积累，通过多步递归让每步生成简单子树
- **端侧可部署**：1B 模型即可达到与 GPT-4o 相当的性能，全链路设计考虑了算力约束
- **RETRIEVE 五步流水线**：兼顾高召回与效率，整源裁剪+模式分类大幅减少计算量
- **EXTRACT 算子弥合结构化/非结构化鸿沟**：从邮件、帖子等文本中在线提取结构化字段
- **完整的隐私保护方案**：用户研究中 Docker 断网部署，数据完全不离开设备

## 局限与展望

- **数据类型有限**：目前仅支持日历/邮件/流媒体等有限类别，未覆盖照片、位置轨迹等多模态数据
- **QUD 仍是主要瓶颈**：错误分析显示 50% 的失败源于算子树生成错误
- **评测局限**：仅在合成数据 PerQA 和小规模用户研究(20人)上评估，缺乏大规模真实场景验证
- **聚合类问题仍有较大提升空间**：Hit@1 最高不到 40%，join 类问题仅 23.6%

## 与相关工作的对比

- **TimelineQA** (Tan et al., 2023)：最接近的先驱工作，但模板有限(42个)、数据多样性不足；ReQAP 在其基准上也取得显著优势 (SFT: 0.313 vs 0.135)
- **Text2SQL** (Li et al., 2024; Liu et al., 2024b)：纯结构化场景表现好，但无法处理非结构化文本
- **RAG 类方法** (Oguz et al., 2022; Badaro et al., 2023)：少量证据时有效，但无法应对大规模聚合和数值运算
- **问题分解方法** (Jia et al., 2024; Saeed et al., 2024)：针对特定场景设计，未泛化到异构数据问答

## 评分

- 新颖性: ⭐⭐⭐⭐ (递归分解 + 算子树 + RETRIEVE/EXTRACT 融合两大范式)
- 实验充分度: ⭐⭐⭐⭐ (自建基准 + 消融 + 多模型规模对比 + 用户研究，但缺乏更多外部数据集)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，running example 贯穿始终，图示直观)
- 价值: ⭐⭐⭐⭐ (个人数据问答是实际刚需，端侧部署方案有工业价值)

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] On Synthesizing Data for Context Attribution in Question Answering](on_synthesizing_data_for_context_attribution_in_question_answering.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)

<!-- RELATED:END -->
