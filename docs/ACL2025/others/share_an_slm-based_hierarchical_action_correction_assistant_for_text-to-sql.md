---
title: >-
  [论文解读] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL
description: >-
  [ACL 2025][Text-to-SQL] 提出 SHARE 框架，通过三个专门化小语言模型 (SLM, <8B) 的顺序管道协作，将声明式 SQL 转换为步骤化动作轨迹以暴露推理路径，再从 Schema 和逻辑两个维度分阶段纠正错误，实现高效低成本的 Text-to-SQL 自纠错。
tags:
  - ACL 2025
  - Text-to-SQL
  - 自纠正
  - 小语言模型
  - 动作轨迹
  - 层次化自演化
---

# SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL

**会议**: ACL 2025  
**arXiv**: [2506.00391](https://arxiv.org/abs/2506.00391)  
**代码**: [有](https://github.com/quge2023/SHARE)  
**领域**: NLP / Text-to-SQL  
**关键词**: Text-to-SQL, 自纠正, 小语言模型, 动作轨迹, 层次化自演化

## 一句话总结

提出 SHARE 框架，通过三个专门化小语言模型 (SLM, <8B) 的顺序管道协作，将声明式 SQL 转换为步骤化动作轨迹以暴露推理路径，再从 Schema 和逻辑两个维度分阶段纠正错误，实现高效低成本的 Text-to-SQL 自纠错。

## 研究背景与动机

Text-to-SQL 自纠正面临两个关键瓶颈：

**计算开销高**：传统自纠正依赖 LLM 的递归自调用（如 GPT-4o 多次调用），计算成本呈乘法级增长

**错误定位困难**：LLM 难以对声明式 SQL 进行有效的错误检测和修正——SQL 是声明式的，隐藏了底层推理路径。模型无法"看到"自己生成 SQL 时的推理过程，因此难以精确定位错误

核心洞察：如果把 SQL 从声明式形式拆解为步骤化的"动作轨迹"（类似 pandas 操作链），就能暴露推理路径，使错误更容易定位和修正。而且这种分析可以交给更小、更便宜的专用模型完成。

## 方法详解

### 整体框架

SHARE 由三个 LoRA 微调的 SLM 组成，按顺序执行：

1. BAM（Base Action Model）：SQL → 动作轨迹
2. SAM（Schema Augmentation Model）：修正 Schema 级错误
3. LOM（Logic Optimization Model）：修正逻辑级错误

### 关键设计

1. **Base Action Model (BAM)**：将 SQL 转换为动作轨迹

    - 功能：接收初始 SQL，输出类似 pandas API 的步骤化操作序列
    - 核心思路：将 `SELECT .. FROM .. WHERE ..` 分解为 `[where(...), select(...)]` 等操作序列
    - 训练数据：用 GPT-4o 将标注 SQL 转换为动作轨迹的 (SQL, trajectory) 对，通过可逆性验证过滤（轨迹能还原回 SQL 的才保留），收集 13k 高质量训练对
    - 数据增强：定义 ADD、DELETE、SUBSTITUTE 三种扰动类型，在正确轨迹上注入错误，生成更多训练样本

2. **Schema Augmentation Model (SAM)**：修正 Schema 链接错误

    - 功能：对动作轨迹中的表名、列名等 Schema 元素进行纠正
    - 核心思路：两阶段训练——(1) 学会识别和掩码 Schema 元素 (2) 学会根据数据库信息和用户问题重新填充正确的 Schema
    - 设计动机：Schema 链接是 Text-to-SQL 的关键瓶颈，错误的表/列映射会传播到整个查询

3. **Logic Optimization Model (LOM)**：修正推理逻辑错误

    - 功能：在 Schema 正确的前提下，修正动作轨迹中的逻辑推理错误
    - 核心思路：输入数据库信息、用户问题和 SAM 输出的轨迹，输出逻辑正确的轨迹
    - 训练数据来源：(1) 错误初始 SQL 对应的错误轨迹 (2) 对正确轨迹施加 ADD/DELETE/SUBSTITUTE 扰动生成的人工错误轨迹，共 15k 样本

4. **层次化自演化策略（Hierarchical Self-Evolve）**：BAM 已经学会了从 SQL 到轨迹的转换能力，SAM 和 LOM 的训练数据可以借助 BAM 少样本生成，而非每次都调用昂贵的 GPT-4o。这大幅降低了训练数据构建成本。

### 推理流程

```
初始 SQL s' → BAM → 动作轨迹 t'
t' → SAM → Schema 修正后的 t_f'
t_f' + 用户问题 + 数据库 → LOM → 最终修正轨迹 t
t → 作为反馈引导 LLM 重新生成 SQL
```

### 损失函数 / 训练策略

- 所有模型使用 LLaMA-Factory 框架进行 LoRA 微调
- 在 4×A100 80GB 上训练
- 支持 Llama-3.1-8B 和 Phi-3-Mini-3.8B 两个骨干模型

## 实验关键数据

### 主实验（GPT-4o 作为生成器）

| 方法 | BIRD Total | SPIDER Total |
|------|-----------|-------------|
| GPT-4o 基线 | 55.87 | 77.10 |
| + Self-Correction | 55.28 | 75.90 |
| + Self-Consistency | 58.75 | 81.80 |
| + MAGIC (SOTA) | 59.53 | 85.66 |
| **+ SHARE-3.8B** | 60.89 | 84.00 |
| **+ SHARE-8B** | **64.14** | **85.90** |

SHARE-8B 相比基线提升 14.80% (BIRD) 和 11.41% (SPIDER)。

### 跨生成器泛化

| 生成器 | 基线 | + SHARE-8B | 相对提升 |
|--------|------|-----------|---------|
| Claude-3.5-Sonnet | 49.41 | 63.56 | +28.64% |
| GPT-4o-mini | 49.09 | 59.64 | +21.49% |
| Llama-3.1-70B | 53.91 | 61.93 | +14.88% |
| Qwen-Coder-32B | 58.03 | 61.73 | +6.38% |

### 消融实验

| 设置 | BIRD Total | 变化 |
|------|-----------|------|
| SHARE-8B 完整 | 64.14 | - |
| w/o Schema Aug (SAM) | 60.02 | ↓ 4.08 |
| w/o Logic Opt (LOM) | 56.98 | ↓ 7.16 |
| w/o 层次化策略 | 60.55 | ↓ 3.59 |
| w/o 错误扰动 | 61.38 | ↓ 2.76 |

### 计算成本

| 方法 | 每 1K 实例成本 | EX (BIRD) |
|------|---------------|-----------|
| MAC-Refiner | $20.18 | 58.74 |
| MAGIC | $37.99 | 59.53 |
| **SHARE-8B** | **$2.57** | **64.14** |

### 关键发现

1. 直接自纠正（无反馈）反而降低 GPT-4o 性能，因为模型缺乏可靠的自我评估机制
2. SHARE 的成本仅为最经济基线的 1/10，但效果最佳
3. LOM 移除后性能下降最大（7.16%），说明逻辑纠正比 Schema 纠正更关键
4. SHARE 在 MySQL 和 PostgreSQL 方言上也有效，无需额外训练
5. 使用开源教师模型（Llama-3.1-70B）替代 GPT-4o 构建数据效果相当

## 亮点与洞察

1. **大小模型协作范式**：首次在 Text-to-SQL 领域实现 SLM 辅助 LLM 纠错，大幅降低成本
2. **声明式→过程式转换**：将不透明的 SQL 转换为透明的动作轨迹是关键创新，使错误定位变得可行
3. **两阶段纠正的解耦设计**：将 Schema 和逻辑错误分离处理，各个击破
4. **数据效率高**：50% 训练数据即可超越 SOTA 基线 MAGIC
5. **跨方言泛化**：学习的是底层推理路径纠正，不依赖特定 SQL 方言

## 局限与展望

1. 当前仅在 SQLite、MySQL、PostgreSQL 上验证，更多 SQL 方言（如 Oracle、SQL Server）未测试
2. 动作轨迹的设计基于 pandas-like API，可能无法完全覆盖所有 SQL 操作类型
3. BAM 的训练仍依赖 GPT-4o 进行初始数据标注，虽然后续 SAM/LOM 可自演化
4. 对极端复杂的嵌套子查询，动作轨迹表示可能变得冗长

## 相关工作与启发

- Self-Debugging：基于执行反馈的迭代修正，但需要直接数据库访问权限
- MAGIC：基于 ICL 的强自纠正方法，但成本是 SHARE 的 15 倍
- Action Model：本文借鉴了将任务分解为动作轨迹的思路并创造性地应用到 SQL 纠错

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ SQL → 动作轨迹的转换 + 三模型管道的设计极具创新性
- **实验充分度**: ⭐⭐⭐⭐⭐ 4 个基准、多个生成器、跨方言测试、成本分析、低资源实验、开源教师模型实验，非常全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ 成本降低 10x 且效果提升 14.8%，极具实用价值

<!-- RELATED:START -->

## 相关论文

- [STaR-SQL: Self-Taught Reasoner for Text-to-SQL](star-sql_self-taught_reasoner_for_text-to-sql.md)
- [Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)
- [ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)
- [Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction](cadreview_automatically_reviewing_cad_programs_with_error_detection_and_correcti.md)

<!-- RELATED:END -->
