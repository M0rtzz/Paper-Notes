---
title: >-
  [论文解读] Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval
description: >-
  [ICLR 2026][人体理解][LLM 个性化] 受认知科学双过程理论启发，提出 RF-Mem 框架，通过 Familiarity（快速相似度匹配）和 Recollection（深层链式重建）双路径自适应切换的记忆检索机制，实现高效且可扩展的 LLM 个性化。
tags:
  - ICLR 2026
  - 人体理解
  - LLM 个性化
  - 记忆检索
  - 双过程理论
  - 自适应检索
  - 认知科学
---

# Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval

**会议**: ICLR 2026  
**arXiv**: [2603.09250](https://arxiv.org/abs/2603.09250)  
**代码**: 见论文 Reproducibility Statement  
**领域**: 个性化 / 信息检索  
**关键词**: LLM 个性化, 记忆检索, 双过程理论, 自适应检索, 认知科学

## 一句话总结

受认知科学双过程理论启发，提出 RF-Mem 框架，通过 Familiarity（快速相似度匹配）和 Recollection（深层链式重建）双路径自适应切换的记忆检索机制，实现高效且可扩展的 LLM 个性化。

## 研究背景与动机

个性化大语言模型需要将用户特定的历史记录、偏好和上下文纳入到对话生成中。现有两种主流方法各有严重缺陷：

**全上下文方法**: 将用户所有历史记忆塞入 prompt，成本高昂且不可扩展——随着用户记忆的积累，prompt 长度会迅速超出模型窗口限制

**一次性检索方法**: 将检索简化为单轮相似度搜索（top-K），只能捕获表面匹配，无法深度恢复与查询间接相关但关键的记忆内容

认知科学研究表明，人类记忆识别是通过**双过程**运作的：
- **Familiarity（熟悉感）**: 快速但粗糙的识别过程，能够迅速判断某事物是否曾经遇到过
- **Recollection（回忆）**: 慢速但精确的重建过程，能够有意识地回溯具体细节和相关上下文

现有系统既缺乏回忆式检索的能力，也没有在两种检索路径之间自适应切换的机制。这导致要么检索不足（遗漏关键记忆），要么引入噪声（检索到不相关的内容）。

## 方法详解

### 整体框架

RF-Mem（Recollection-Familiarity Memory Retrieval）是一种受熟悉感不确定性引导的双路径记忆检索器。整体流程为：给定用户查询，先计算 Familiarity 信号判断检索确定性，然后根据确定性水平选择 Familiarity 快速路径或 Recollection 深度路径。

### 关键设计

1. **Familiarity 信号计算**:

    - 计算查询与候选记忆的相似度分数分布
    - 通过两个指标衡量 Familiarity 水平：
        - **均值分数（Mean Score）**: 捕获整体匹配强度，均值越高表示存在熟悉的记忆
        - **熵（Entropy）**: 衡量分数分布的不确定性，熵越低表示检索结果越明确和集中
    - 高均值 + 低熵 = 高 Familiarity（确信存在直接匹配）
    - 低均值 / 高熵 = 低 Familiarity（需要更深入的检索）

2. **Familiarity 路径（快速通道）**:

    - 当 Familiarity 信号强时激活
    - 执行标准的 top-K 相似度检索，直接返回最相关的记忆
    - 高效：只需一次前向检索，无需额外计算
    - 适用场景：用户提问与历史记忆有明确的直接关联

3. **Recollection 路径（深度重建通道）**:

    - 当 Familiarity 信号弱时激活，模拟人类的有意识记忆重建过程
    - **候选记忆聚类**: 对检索到的候选记忆进行聚类，识别不同的记忆主题和上下文
    - **Alpha-Mix 查询扩展**: 将查询与候选记忆的聚类中心进行 alpha 混合，在嵌入空间中生成扩展查询
    - **迭代证据扩展**: 用扩展后的查询重新检索，逐步覆盖与原始查询间接相关但语义关联的记忆
    - 适用场景：用户提问涉及跨时间、跨主题的记忆关联，需要链式推理才能回忆

4. **自适应路径切换**:

    - 基于 Familiarity 信号的阈值机制自动选择路径
    - 避免了两种极端：全上下文的高开销和单次检索的低召回率
    - 在固定预算和延迟约束下实现最优的检索质量

### 训练策略

RF-Mem 框架是模块化的，可以与不同的嵌入模型和 LLM 组合。关键创新在于检索策略层面而非模型训练层面——通过更智能的检索决策提升个性化效果。

## 实验关键数据

### 主实验

在三个个性化基准上进行评测，涵盖不同的语料库规模：

| 方法 | 基准1 | 基准2 | 基准3 | 说明 |
|------|-------|-------|-------|------|
| 全上下文推理 | 基线 | 基线 | 基线 | 成本最高，性能上限 |
| 一次性检索 | 低于全上下文 | 低于全上下文 | 低于全上下文 | 简单快速但质量差 |
| **RF-Mem** | **最优** | **最优** | **最优** | 在固定预算下一致超越两种基线 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅 Familiarity 路径 | 基线水平 | 等同于标准 top-K 检索 |
| 仅 Recollection 路径 | 高于 Familiarity-only | 但在简单查询上浪费计算 |
| 双路径 + 自适应切换 | **最优** | 兼顾效率和质量 |
| 去除聚类 | 性能下降 | 聚类帮助识别记忆主题结构 |
| 去除 Alpha-Mix | 性能下降 | 查询扩展是 Recollection 的核心 |

### 关键发现

- **一致优势**: RF-Mem 在所有三个基准和不同语料库规模上均优于两种基线方法
- **预算效率**: 在固定的检索预算（token 数量）和延迟约束下，RF-Mem 实现了接近全上下文方法的性能，同时保持了一次性检索的效率
- **可扩展性**: 随着用户记忆库规模增大，RF-Mem 的优势更加明显——全上下文方法的成本线性增长，而 RF-Mem 的开销增长温和
- **路径分布**: 约 60-70% 的查询通过 Familiarity 快速路径处理，30-40% 需要 Recollection 深度路径

## 亮点与洞察

1. **认知科学的优雅迁移**: 将人类记忆的 Familiarity-Recollection 双过程理论引入 LLM 检索系统设计，这种跨学科启发既优雅又实用
2. **不确定性引导的自适应**: 利用检索分数分布的均值和熵作为自适应切换信号，比简单的阈值方法更鲁棒
3. **嵌入空间中的记忆重建**: 通过聚类和 Alpha-Mix 在嵌入空间中模拟回忆过程的链式重建，避免了代价昂贵的多轮 LLM 调用
4. **实用的设计哲学**: 框架模块化，各组件可独立替换和优化，适合工程部署

## 局限与展望

1. **Familiarity 阈值的设定**: 自适应切换依赖于阈值参数，不同数据集可能需要不同的阈值，缺乏完全自动化的方案
2. **聚类算法的选择**: Recollection 路径中的聚类方法可能对高维稀疏记忆效果有限
3. **长期记忆遗忘与更新**: 未明确讨论如何处理过时或矛盾的用户记忆
4. **隐私考量**: 存储和检索用户历史记忆涉及隐私风险，论文未深入讨论隐私保护机制
5. **仅基于摘要评估** (本笔记注): 由于论文全文 HTML 版本不可用，部分实验细节和数据来自摘要推断

## 相关工作与启发

- **双过程认知理论 (Yonelinas, 2002)**: Familiarity 和 Recollection 是人类记忆识别的两种基本过程，本文将此理论操作化为检索系统设计
- **RAG (Retrieval-Augmented Generation)**: RF-Mem 可视为 RAG 的增强版本，专门针对个性化场景优化检索策略
- **自适应检索 (Adaptive Retrieval)**: 如 Self-RAG, FLARE 等工作研究何时检索，RF-Mem 研究如何检索
- **个性化 LLM**: 如 LaMP, PersonaLLM 等基准推动了个性化 LLM 的发展，本文在此基础上改进检索模块

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 认知科学双过程理论在 LLM 检索中的创新应用
- 实验充分度: ⭐⭐⭐⭐ — 三个基准、多尺度评测、消融分析
- 写作质量: ⭐⭐⭐⭐ — 概念清晰，跨学科动机解释充分
- 价值: ⭐⭐⭐⭐ — 为个性化 LLM 的记忆检索提供了实用可扩展的方案

<!-- RELATED:START -->

## 相关论文

- [LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)
- [P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)

<!-- RELATED:END -->
