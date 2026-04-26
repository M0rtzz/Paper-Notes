---
title: >-
  [论文解读] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables
description: >-
  [ICLR 2026][模型压缩][多跳推理] 提出 SPARTA，一个端到端自动构建大规模表格-文本多跳问答基准的框架，通过参考事实数据库、来源引导的修复和现实结构约束生成高质量嵌套 SQL 查询，SOTA 模型在 SPARTA 上 F1 下降超过 30 分。
tags:
  - ICLR 2026
  - 模型压缩
  - 多跳推理
  - 表格-文本问答
  - 基准构建
  - SQL
  - 跨模态推理
---

# SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables

**会议**: ICLR 2026  
**arXiv**: [2602.23286](https://arxiv.org/abs/2602.23286)  
**代码**: [github.com/pshlego/SPARTA](https://github.com/pshlego/SPARTA)  
**领域**: 模型压缩 / NLP 基准  
**关键词**: 多跳推理, 表格-文本问答, 基准构建, SQL, 跨模态推理

## 一句话总结

提出 SPARTA，一个端到端自动构建大规模表格-文本多跳问答基准的框架，通过参考事实数据库、来源引导的修复和现实结构约束生成高质量嵌套 SQL 查询，SOTA 模型在 SPARTA 上 F1 下降超过 30 分。

## 研究背景与动机

- **现有基准的三大缺陷**：
  1. **问题类型有限、推理浅**：多数基准仅需 ≤2 跳推理，不支持聚合、分组等高级操作
  2. **标注噪声严重**：审计 HybridQA 100 样本发现 21% 包含错误（冗余模态 52.4%、答案不完整 23.8%、错误/无法回答 23.8%）
  3. **仅依赖小规模 web 表格**：平均 ~15 行，远不及真实数据库的千行规模
- 手动标注复杂性限制了基准的规模和质量，需要自动化方法

## 方法详解

### 整体框架：三阶段流水线

1. **参考事实数据库构建** — 将源表格与接地表格合并为统一的关系数据库
2. **查询生成** — LLM 生成嵌套 SQL 查询，深度匹配目标跳数
3. **问题自然语言化** — 将验证通过的 SQL 转化为流畅的自然语言问题

### 参考事实数据库

- **源表格 $\mathcal{S}_T$**：保留原始关系表（如 NBA 薪资、奖项、选秀等 6 张 Kaggle 公开表）
- **接地表格 $\mathcal{G}_T$**：将非结构化文本分解为原子事实元组，存入 SQL 可查询的关系表
- 两种接地方法：(1) 利用已验证语料如 ROTOWIRE；(2) 基于模板的表到文本转换
- 共享实体属性（如 PLAYER_NAME）通过主外键约束保证联接可达性

### 查询生成的关键技术

#### 后序遍历 + 现实结构约束

- 将嵌套 SQL 建模为查询图 $G=(V,E)$，节点=查询块，边=嵌套谓词
- 采用**后序遍历**构建：先生成验证通过的叶子查询，再递归封装为高层查询
- 优于自顶向下或广度优先：后序保证每个中间块在构建时即可执行验证

#### 来源引导的修复 (Provenance-based Refinement)

当查询返回空结果时：
1. 逆序剥离谓词直到返回非空结果
2. 从非空结果中采样元组
3. 运行 why-not provenance 工具定位阻塞谓词
4. 将诊断报告反馈给 LLM 重写问题子句

### 问题自然语言化

- 使用 AST-ICL（SOTA SQL-to-text 模型）将 SQL 转为流畅自然语言
- 3 名 CS 研究生进行轻量级验证，标注效率为 HybridQA 的 **4 倍**

### 领域无关性设计

框架可扩展到任意领域：指定源表格和接地表格，应用表到文本生成后，查询生成流水线不变。已扩展到电影和医疗领域。

## 实验关键数据

### 基准对比

| 基准 | 表格规模 | 问题生成 | 分组/Having | >3-Hop | 标注错误率 |
|------|---------|---------|------------|--------|-----------|
| HybridQA | 4.4列×15.7行 | 手动 | ✗ | ✗ | 21% |
| OTT-QA | 4.4列×15.7行 | 手动 | ✗ | ✗ | 21% |
| TAT-QA | 4.0列×9.4行 | 手动 | ✗ | ✗ | 30% |
| **SPARTA (NBA)** | **12.2列×3280行** | **自动+轻量验证** | **✓** | **✓** | **0%** |

### SOTA 模型在 SPARTA 上的表现

| 模型 | HybridQA F1 | SPARTA F1 | 下降 |
|------|------------|-----------|------|
| 最优现有模型 | >70 | <40 | >30↓ |
| OTT-QA 最优模型 | >50 | <20 | >30↓ |

### 消融：查询生成策略

| 方法 | 执行成功率 | 查询多样性 |
|------|----------|-----------|
| One-Shot (无检查) | 低 | 低 |
| Post-Order (无 Provenance) | 中 | 中 |
| **Post-Order + Provenance** | **高** | **高** |

### 关键发现

1. SOTA 模型（GPT-4、Claude 等）在 SPARTA 上 F1 大幅下降，暴露跨模态推理根本弱点
2. 后序遍历 + 来源修复的组合显著提高了查询执行率和多样性
3. 轻量级人工验证仅需 HybridQA 1/4 的标注时间
4. 框架成功扩展到电影和医疗域，验证了领域无关性

## 亮点与洞察

- **从根本上重新设计 Table-Text QA 基准**：SQL-centric 流水线解决了规模、噪声和逻辑深度三个核心问题
- **Provenance 修复是关键创新**：将数据库技术（why-not provenance）引入 NLP 基准构建
- **高暴露性**：SOTA 模型 F1 骤降 30+ 分，清晰指向现有跨模态推理能力的根本不足
- **可复现可扩展**：代码、数据、模型全开源，方便后续研究

## 局限性

- 接地表格的原子事实提取依赖特定语料（如 ROTOWIRE），扩展到新领域需人工设计模板
- 自然语言化依赖 LLM，可能引入细微语义偏差
- 仅评估了提取式和生成式 QA 模型，尚未测试 Agent / Tool-augmented 方法

## 相关工作

- Table-Text QA 基准：HybridQA, OTT-QA, TAT-QA, FinQA, MultiHiertt 等
- 合成基准生成：ERBench, TDBench 等（多为单模态或浅层）
- PEEL：模板化 NL-嵌套 SQL 对生成

## 评分

- **新颖性**: ⭐⭐⭐⭐ — SQL-centric 的自动基准构建思路新颖
- **技术深度**: ⭐⭐⭐⭐ — Provenance 修复和后序遍历约束设计精巧
- **实验充分性**: ⭐⭐⭐⭐ — 多领域、多模型、消融全面
- **实用性**: ⭐⭐⭐⭐⭐ — 直接暴露 SOTA 的根本弱点，对社区价值高

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ICLR 2026\] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)
- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](../../ACL2025/model_compression/stun_moe_pruning.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](../../ACL2025/model_compression/rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)

<!-- RELATED:END -->
