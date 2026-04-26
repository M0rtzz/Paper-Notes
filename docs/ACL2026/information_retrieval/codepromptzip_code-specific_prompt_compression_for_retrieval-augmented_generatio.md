---
title: >-
  [论文解读] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs
description: >-
  [ACL 2026][代码提示压缩] 提出 CodePromptZip，首个面向代码的提示压缩框架，通过类型感知优先级排序构建训练数据并训练带 copy 机制的小模型压缩器，在三个编码任务上分别比最佳基线提升 23.4%、28.7% 和 8.7%。
tags:
  - ACL 2026
  - 代码提示压缩
  - RAG
  - 类型感知优先级
  - copy机制
  - 编码任务
---

# CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs

**会议**: ACL 2026  
**arXiv**: [2502.14925](https://arxiv.org/abs/2502.14925)  
**代码**: 无  
**领域**: information_retrieval  
**关键词**: 代码提示压缩, RAG, 类型感知优先级, copy机制, 编码任务

## 一句话总结

提出 CodePromptZip，首个面向代码的提示压缩框架，通过类型感知优先级排序构建训练数据并训练带 copy 机制的小模型压缩器，在三个编码任务上分别比最佳基线提升 23.4%、28.7% 和 8.7%。

## 研究背景与动机

**领域现状**：RAG 通过检索相关代码示例增强 LLM 在编码任务上的表现，但检索到的代码往往长达数万 token，受限于 LLM 上下文窗口和 API 调用成本。

**现有痛点**：现有提示压缩技术（LLMLingua、RECOMP 等）都针对自然语言设计，忽略了代码的独特特征——不同 token 类型（如 Identifier、Symbol、Invocation）对生成质量的影响差异巨大。

**核心矛盾**：自然语言压缩方法用启发式信息熵或知识蒸馏来判断 token 重要性，但这些指标未考虑代码的类型结构信息，压缩效果不理想。

**本文目标**：设计首个代码专用的提示压缩框架，能在指定压缩率下最大程度保留对下游任务有用的代码信息。

**切入角度**：利用程序分析将代码 token 按类型分类，通过消融分析建立类型级别的移除优先级，再用此指导训练数据构建和压缩模型学习。

**核心 idea**：不同类型的代码 token 对任务的影响不同，按影响从小到大的优先级顺序移除 token，并训练带 copy 机制的 CodeT5 模型来学习这种压缩策略。

## 方法详解

### 整体框架

分为训练和推理两阶段。训练阶段：(1) 类型感知优先级排序——用 JavaParser 做 AST 分析，消融各 token 类型对任务性能的影响，建立移除优先级；(2) 贪心算法按优先级构建压缩训练样本；(3) 训练 copy 增强的 CodeT5 压缩器。推理阶段：压缩器接受原始代码和目标压缩率，输出压缩代码嵌入 RAG 提示。

### 关键设计

1. **类型感知优先级排序 (Type-aware Priority Ranking)**:
    - 功能：确定不同类型代码 token 的移除优先级
    - 核心思路：将 token 分为 Symbol、Signature、Invocation、Identifier、Structure 五类，逐类消融后测量 Priority(T) = 压缩率/性能退化率，优先级高的类型可被优先移除
    - 设计动机：观察到优先级层级是跨模型一致但任务特异的（如 Invocation 在 Assertion Generation 中优先级最高，但在 Code Suggestion 中最低）

2. **Copy 增强的 CodeT5 压缩器**:
    - 功能：学习在指定压缩率下生成压缩代码
    - 核心思路：在 CodeT5 编码器-解码器架构上加入 copy module——计算 p_gen 决定从词表生成还是从源序列复制，最终输出分布为 P(y) = p_gen·P_vocab + (1-p_gen)·P_copy
    - 设计动机：代码压缩本质是提取性任务（输出完全来自输入），copy 机制天然契合；同时可处理不可解析的代码片段

3. **灵活压缩率控制**:
    - 功能：支持用户指定任意目标压缩率
    - 核心思路：扩展词表引入 <Ratio> 等特殊 token，在输入中显式编码目标压缩率，让模型自适应学习不同压缩级别
    - 设计动机：不同场景对压缩率的需求不同，需要一个可控的压缩器而非固定压缩

### 损失函数 / 训练策略

使用交叉熵损失训练，AdamW 优化器，batch=16，lr=5e-5，warmup=1000 步，训练 10 epoch。训练数据由 Algorithm 1（优先级驱动贪心算法）按不同压缩率自动构建。

## 实验关键数据

### 主实验

| 方法 | Assertion (EM%) | Bugs2Fix (CB%) | Code Suggestion (CB%) |
|------|-----------------|----------------|----------------------|
| w/o retrieval | 23.9 | 41.7 | 14.2 |
| LLMLingua | 33.8 | 41.9 | 21.8 |
| LongLLMLingua | 34.1 | 42.1 | 21.2 |
| LLMLingua-2 | 21.2 | 48.1 | 21.7 |
| RECOMP | 23.4 | 45.3 | 21.0 |
| CodePromptZip (w/o Copy) | 40.9 | 56.7 | 20.5 |
| **CodePromptZip** | **42.1** | **61.9** | **23.7** |
| Oracle (AST) | 46.2 | 66.8 | 23.8 |
| w/o compression | 50.5 | 81.4 | 24.7 |

*τ_code=0.3, 1-shot, 使用 GPT-3.5-turbo*

### 消融实验

| 组件 | Assertion (EM%) | Bugs2Fix (CB%) | Code Suggestion (CB%) |
|------|-----------------|----------------|----------------------|
| CodePromptZip w/o Copy | 40.9 | 56.7 | 20.5 |
| CodePromptZip (full) | 42.1 (+1.2) | 61.9 (+5.2) | 23.7 (+3.2) |

**压缩率控制**：CodePromptZip 的实际压缩率与指定压缩率紧密对齐，而无 copy 机制的版本控制能力显著变差。

### 关键发现

- 在三个任务上分别比最佳基线提升 23.4%（42.1 vs 34.1）、28.7%（61.9 vs 48.1）、8.7%（23.7 vs 21.8）
- Copy 机制对压缩率控制至关重要，同时在三个任务上均带来性能提升
- 权衡分析显示：在固定 token 预算下，使用更少的低压缩率示例优于更多的高压缩率示例
- 跨模型泛化：在 CodeLlama-13B 和 Gemini-1.0-Pro 上同样优于所有基线
- 不可解析代码：移除末尾 1-3% token 后性能仅微降（42.1% → 42.0%/41.7%），证明了学习型方法的鲁棒性

## 亮点与洞察

- 首次提出代码专用的提示压缩问题和解决方案，填补了 NL 压缩和代码压缩之间的空白
- 类型感知优先级排序的发现具有启发性：移除优先级跨模型一致但跨任务不同，说明代码 token 重要性是任务驱动而非模型驱动的
- 基于学习的方法优雅解决了 Oracle（需 AST 解析）无法处理不完整代码的限制
- 压缩率可控性设计使框架可适应不同成本/质量权衡需求

## 局限与展望

- 目前仅支持 Java 代码（依赖 JavaParser 构建训练数据），其他语言的泛化性未验证
- 压缩器基于 CodeT5-775M，模型大小和推理延迟需要考量
- 消融分析中的优先级排序需要对每个新任务重做，自动化/自适应的优先级发现是未来方向
- 未考虑多语言代码混合场景

## 相关工作与启发

- 对 LLMLingua 系列的改进方向明确：信息熵指标不适合代码，需要利用代码的类型结构信息
- RECOMP 用 GPT-3.5 蒸馏的方式成本高且压缩率不可控，CodePromptZip 的优先级驱动方法更高效可控
- 对 RAG 系统优化的启示：不同模态的检索内容应使用专门的压缩策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个代码专用提示压缩框架，类型感知优先级排序思路新颖
- 实验充分度: ⭐⭐⭐⭐ 三个任务、多个基线、跨模型泛化、不可解析代码测试
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法论述完整，实验结论明确

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](../../ACL2025/information_retrieval/exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)

<!-- RELATED:END -->
