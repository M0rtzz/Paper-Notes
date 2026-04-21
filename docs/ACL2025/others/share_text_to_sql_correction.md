---
title: >-
  [论文解读] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL
description: >-
  [ACL 2025] 提出 SHARE 框架，用三个 <8B 参数的专用小语言模型（SLM）组成顺序管道，将声明式 SQL 转换为可暴露推理路径的步进动作轨迹，再分阶段修正 schema 链接错误与逻辑推理错误，以极低成本实现 LLM 的 Text-to-SQL 自纠正。
tags:
  - ACL 2025
---

# SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL

**会议**: ACL 2025  
**arXiv**: [2506.00391](https://arxiv.org/abs/2506.00391)  
**代码**: [GitHub](https://github.com/quge2023/SHARE)  
**领域**: 其他

## 一句话总结

提出 SHARE 框架，用三个 <8B 参数的专用小语言模型（SLM）组成顺序管道，将声明式 SQL 转换为可暴露推理路径的步进动作轨迹，再分阶段修正 schema 链接错误与逻辑推理错误，以极低成本实现 LLM 的 Text-to-SQL 自纠正。

## 研究背景与动机

1. **LLM 自纠正成本高昂**: 现有 self-correction 方法依赖对 LLM 的递归调用，每轮都需完整推理，计算开销呈乘法增长；例如 Multiple-Prompt 方法每千条查询成本高达 $62.86
2. **执行反馈不可靠且受限**: Self-debugging 依赖数据库执行反馈，但 SQLite 等主流引擎返回信息过于简略，难以精确定位语义错误；且在数据隐私敏感场景中，直接访问数据库受到严格限制
3. **声明式 SQL 的黑盒困境**: LLM 直接对声明式 SQL 进行纠正时无法展示底层推理过程，存在 self-enhancement bias —— GPT-4o 原生 self-correction 反而使 BIRD 准确率从 55.87% 下降到 55.28%
4. **Prompt 工程依赖严重**: 现有高性能方法（如 MAGIC）依赖精心设计的多阶段 prompt，人力成本高且难以泛化到其他生成器模型或 SQL 方言

## 方法详解

### 整体框架

SHARE 采用 assistant-based 自纠正范式：任意生成器 LLM 产生初始 SQL 后，三个 LoRA 微调的专用 SLM（均 <8B 参数）组成顺序管道进行纠正，最终将修正后的动作轨迹作为反馈引导 LLM 重新生成 SQL。

### 关键设计

**1. Base Action Model (BAM) — 声明式到过程式的推理暴露**

BAM 将声明式 SQL 查询拆解为 pandas-like API 的步进动作轨迹（如 `where → groupby → orderby → select`），使隐含的推理路径显式化。训练数据通过 GPT-4o 蒸馏构建 13K 条高质量 query-trajectory 对，并采用可逆性验证（轨迹必须能还原回原始 SQL）来过滤幻觉样本。BAM 同时充当后续模型的数据工厂，通过层级自演化策略为 SAM 和 LOM 自动合成训练数据，避免重复调用教师 LLM。

**2. Schema Augmentation Model (SAM) — 两阶段 Schema 链接修正**

SAM 专注于检测和修正动作轨迹中的 schema 链接错误（表名、列名误匹配等）。采用两阶段训练：Phase 1 学习在轨迹中用 `[MASK]` 精确标记所有 schema 元素位置；Phase 2 根据数据库 schema、用户问题和初始 SQL 的 schema 列表，将 mask 位置填充为正确的 schema 链接。这种"先定位后修正"的解耦设计使模型能独立优化每个子任务。

**3. Logic Optimization Model (LOM) — 动作扰动增强的逻辑修正**

LOM 处理逻辑推理错误（操作顺序、条件逻辑、聚合方式等）。为扩充训练数据，提出动作扰动策略：对正确轨迹施加 ADD（插入冗余动作）、DELETE（删除必要动作）、SUBSTITUTE（替换动作类型或参数）三种扰动，模拟真实错误模式。最终收集 15K 条错误-正确轨迹对进行训练。

### 训练与推理

| 阶段 | 模型 | 数据来源 | 数据量 | 训练方式 |
|------|------|---------|--------|---------|
| Stage 1 | BAM | GPT-4o 蒸馏 + 可逆性验证 | 13K | LoRA 微调 |
| Stage 2 | SAM | BAM 层级自演化 + schema masking | 13K | LoRA 两阶段微调 |
| Stage 3 | LOM | BAM 自演化 + 动作扰动增强 | 15K | LoRA 微调 |

推理时三个模型顺序执行：BAM 生成动作轨迹 → SAM 修正 schema → LOM 修正逻辑 → 将修正轨迹反馈给 LLM 重新生成 SQL，全程单轮交互。

## 实验结果

### 主实验：GPT-4o 生成器 + 单轮修正

| 方法 | 外部反馈 | BIRD EX(%) | SPIDER EX(%) | 每千条成本 |
|------|---------|-----------|-------------|-----------|
| GPT-4o (baseline) | — | 55.87 | 77.10 | — |
| Self-Correction | ✗ | 55.28 ↓ | 75.90 | — |
| Self-Consistency | ✗ | 58.75 | 81.80 | — |
| Multiple-Prompt | ✗ | 58.80 | 81.50 | $62.86 |
| Self-Debugging | ✓ | 58.28 | 81.20 | — |
| MAC-Refiner | ✓ | 58.74 | 80.40 | $20.18 |
| MAGIC | ✗ | 59.53 | 85.66 | $37.99 |
| +SHARE-3.8B | ✗ | 60.89 | 84.00 | — |
| **+SHARE-8B** | **✗** | **64.14** | **85.90** | **$2.57** |

SHARE-8B 在 BIRD 上实现 14.80% 的相对提升，且推理成本仅为最经济基线的 1/10。

### 跨模型泛化与消融

| 实验维度 | 设置 | BIRD EX(%) | 变化 |
|---------|------|-----------|------|
| **跨模型** | Claude-3.5-S → +SHARE-8B | 49.41 → 63.56 | +28.64% |
| | GPT-4o-mini → +SHARE-8B | 49.09 → 59.64 | +21.49% |
| | Llama-3.1-70B → +SHARE-8B | 53.91 → 61.93 | +14.88% |
| | DS-Coder-6.7B → +SHARE-8B | 34.57 → 51.24 | +48.22% |
| **鲁棒性** | DK 数据集 +SHARE-8B | 64.10 → 75.30 | +11.20% |
| | Realistic 数据集 +SHARE-8B | 73.40 → 81.50 | +8.10% |
| **消融** | w/o SAM (Schema Aug) | 60.02 | ↓ 4.08 |
| | w/o LOM (Logic Opt) | 56.98 | ↓ 7.16 |
| | w/o 层级自演化 | 60.55 | ↓ 3.59 |
| | w/o 动作扰动 | 61.38 | ↓ 2.76 |
| **数据效率** | 50% 训练数据 | 60.71 | 已超 MAGIC |
| **开源教师** | SHARE-llama (Llama-70B 教师) | 65.19 | 超 SHARE-gpt |

### 关键发现

1. GPT-4o 原生 self-correction 导致性能下降（55.87→55.28），验证了 LLM 在声明式 SQL 上的 self-enhancement bias
2. SHARE 单轮即可实现 BIRD +14.80%、SPIDER +11.41%，推理成本仅 $2.57/千条
3. 仅 50% 训练数据即超过 SOTA（MAGIC），层级自演化策略大幅提升数据效率
4. 跨模型（闭源/开源）、跨 SQL 方言（MySQL/PostgreSQL）均有效泛化，非拟合特定错误模式
5. 使用开源 Llama-70B 替代 GPT-4o 作为教师模型，SHARE-llama 性能与 SHARE-gpt 持平甚至更优

## 亮点与不足

**亮点**:

- 核心创新在于"声明式→过程式"的范式转换，将 SQL 黑盒纠正转为动作轨迹白盒调试，错误定位精度大幅提升
- SLM 辅助 LLM 的协作范式极具性价比，推理成本仅为 MAGIC 的 6.8%
- 层级自演化策略解耦了训练数据构建对教师 LLM 的依赖，训练阶段成本也仅为 MAGIC 的 14.7%

**不足**:

- 仅验证单轮纠正，多轮迭代修正场景尚未探索
- 预定义的 pandas-like action space 可能无法覆盖所有 SQL 方言的复杂特性
- 对数学推理类错误（Mathematical Delusion）修正效果有限（仅 ↓1.63%），受限于生成器模型的数学能力

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|:---------:|------|
| 新颖性 | 7 | 声明式→过程式转换 + 三模型顺序管道的设计有新意，但动作模型和 LoRA 微调均为成熟技术 |
| 有效性 | 9 | 四个基准全面提升，跨模型/跨方言泛化性强，50%数据即超SOTA，消融完整 |
| 工程价值 | 8 | 推理成本极低（$2.57/千条），可即插即用辅助任意 LLM，开源教师模型方案进一步降低门槛 |
| 可复现性 | 8 | 代码已开源，训练细节充分，超参数和 prompt 均在附录给出 |
- 目前仅评估在 SQLite 相关 benchmark 上，其他数据库系统的效果未知

## 相关工作

- Self-debugging: 基于执行反馈的迭代修正（Zhong et al., 2023; Li & Xie, 2024）
- Self-correction: 无需执行反馈的自主纠正（Liu & Tan, 2024; Askari et al., 2024）
- Action model: 将任务分解为过程式动作轨迹（Zhang et al., 2024）
- MAGIC: 当前 SOTA 的 text-to-SQL 自纠正方法

## 评分

- **新颖性**: ★★★★☆ — 动作轨迹转换加模块化纠正的设计很有创意
- **技术深度**: ★★★★☆ — 三阶段管道设计和自演化训练策略设计精细
- **实验充分性**: ★★★★★ — 4个benchmark + 多个生成器 + 低资源分析 + 跨方言测试
- **实用价值**: ★★★★☆ — 低成本辅助纠正在隐私受限的实际场景中非常有价值

<!-- RELATED:START -->

## 相关论文

- [Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [STaR-SQL: Self-Taught Reasoner for Text-to-SQL](star-sql_self-taught_reasoner_for_text-to-sql.md)
- [ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development](comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)

<!-- RELATED:END -->
