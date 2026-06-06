---
title: >-
  [论文解读] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization
description: >-
  [ACL 2026][LLM安全][文本匿名化] 提出自适应文本匿名化框架，通过进化式提示优化自动为LLM发现任务特定的匿名化指令，在多个隐私-效用权衡场景中超越手工设计的策略，且可在开源模型上运行。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "文本匿名化"
  - "隐私保护"
  - "提示优化"
  - "进化算法"
  - "隐私-效用权衡"
---

# Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization

**会议**: ACL 2026  
**arXiv**: [2602.20743](https://arxiv.org/abs/2602.20743)  
**代码**: [https://github.com/gabrielloiseau/adaptive-text-anonymization](https://github.com/gabrielloiseau/adaptive-text-anonymization)  
**领域**: AI Safety  
**关键词**: 文本匿名化, 隐私保护, 提示优化, 进化算法, 隐私-效用权衡

## 一句话总结

提出自适应文本匿名化框架，通过进化式提示优化自动为LLM发现任务特定的匿名化指令，在多个隐私-效用权衡场景中超越手工设计的策略，且可在开源模型上运行。

## 研究背景与动机

**领域现状**：文本匿名化是实现敏感数据共享和分析的基础技术。目前的方法主要分为传统的序列标注（检测并掩码PII实体）和基于LLM的对抗协作管线（如AF方法中使用攻击者LLM引导匿名化决策）。

**现有痛点**：现有LLM匿名化管线存在三大限制：（1）固定权衡范式——每个场景手动设计一个策略，无法灵活适应新需求；（2）依赖人工提示工程，主观、费力且效果欠佳；（3）大多依赖闭源API模型（如GPT-4/5），处理敏感数据通过外部API本身就与隐私目标矛盾。

**核心矛盾**：匿名化本质上是高度上下文依赖的——医疗报告和社交评论的匿名化策略截然不同，不存在"一刀切"的方案，但现有方法无法自适应地调整策略。

**本文目标**：设计一个自适应框架，能够（1）自动发现针对特定隐私-效用需求的匿名化提示，（2）在开源模型上运行，（3）在单次优化中发现多个Pareto最优策略。

**切入角度**：将匿名化问题重新定义为"字符串发现"问题——不修改模型参数，而是搜索最优的自然语言指令来引导模型行为。

**核心 idea**：利用进化式提示优化算法（GEPA）自动搜索匿名化提示空间，从一个通用种子提示出发进化出任务适应的指令，实现自适应的隐私-效用权衡。

## 方法详解

### 整体框架

输入为待匿名化文本和隐私-效用任务规格 $(p, u)$，输出为匿名化后的文本。框架通过进化提示优化在固定计算预算内搜索最佳匿名化指令 $\Pi^*$，整个过程分为三个阶段：初始化、基础反馈热启动、丰富反馈精炼。

### 关键设计

1. **两阶段GEPA进化优化**:

    - 功能：从通用种子提示出发，通过进化搜索发现任务特定的匿名化指令
    - 核心思路：维护一个提示池 $P$，每轮迭代中通过Pareto排序选择高性能且多样的提示，proposer agent分析执行轨迹和反馈后提出变异，新候选在验证集上评估后通过Pareto剪枝纳入池中。Stage 2使用简单的标量聚合反馈 $\mu$，当性能停滞时进入Stage 3
    - 设计动机：进化搜索天然支持多目标优化（隐私vs效用），能在单次运行中发现多个Pareto最优解，而非收敛到单一固定权衡点

2. **丰富反馈生成机制**:

    - 功能：将粗粒度标量反馈升级为包含自然语言解释的结构化反馈
    - 核心思路：由专门的rich feedback agent（独立LLM）将聚合指标 $\mu$ 分解为 $\mu_{rich}$，为proposer提供可解释的、结构化的改进信号，使其能做出更大幅度的定向行为更新
    - 设计动机：标量反馈过于粗糙，proposer难以理解"哪里不好、如何改进"。丰富反馈使得在剩余预算内用更少的评估实现更精准的提示优化

3. **自适应验证采样**:

    - 功能：在精炼阶段使用采样子集评估候选提示以节省计算预算
    - 核心思路：采用轮询策略优先选择被评估次数少的样本作为 $D'_{valid} \subset D_{valid}$（采样比例 $\alpha=0.3$），最终选择时使用完整验证集确保公平
    - 设计动机：每次在全验证集上评估消耗大量预算，采样可在保持覆盖多样性的同时提高预算利用效率

### 损失函数 / 训练策略

不涉及梯度训练。优化目标是隐私得分和效用得分的聚合（如平均值），通过Pareto选择实现多目标权衡。进化预算 $B=1500$ 次LLM前向传播，早停耐心 $n=5$。

## 实验关键数据

### 主实验

| 基准 | 方法 | 隐私↑ | 效用↑ |
|------|------|-------|-------|
| DB-Bio | Optimized Qwen3 | 65.5 | 100 |
| DB-Bio | AF (GPT-5) | 78.0 | 92.1 |
| TAB | Optimized Qwen3 | 92.3 | 56.2 |
| TAB | AF (GPT-5) | 59.9 | 42.5 |
| PUPA | Optimized Qwen3 | 98.0 | 79.3 |
| PUPA | AF (GPT-5) | 94.2 | 46.0 |
| MedQA | Optimized Qwen3 | 24.6 | 45.9 |
| MedQA | AF (GPT-5) | 24.4 | 45.8 |

### 消融实验

| 配置 | 隐私-效用表现 | 说明 |
|------|-------------|------|
| Seed Prompt | 基线 | 通用种子提示，无优化 |
| Task-Specific Prompt | 中等 | 人工设计的任务特定提示 |
| Optimized Prompt | 最优 | 自动优化后的提示 |
| OpenPII (实体检测) | 高效用低隐私 | 仅检测PII实体，隐私保护不足 |
| DP-Prompt ($\epsilon=100$) | 高隐私低效用 | 差分隐私噪声严重破坏效用 |

### 关键发现
- 优化后的开源Qwen3-30B在多数任务上与GPT-5基线竞争力相当甚至更优，尤其在效用保持方面
- 不同模型表现出不同的优化特征：Mistral倾向激进隐私提升（可能牺牲效用），Gemma保守改进，Qwen最鲁棒
- 单次优化运行可发现多个Pareto最优策略，覆盖隐私优先到效用优先的完整频谱

## 亮点与洞察
- 将匿名化问题转化为"字符串搜索"问题是一个巧妙的抽象，每个Pareto解只是一个自然语言字符串，存储和部署成本极低
- 进化优化天然支持多目标发现，单次运行就能找到多个不同权衡点，这比传统方法每个权衡点需要单独设计策略高效得多
- 丰富反馈机制的思路——将标量指标分解为结构化自然语言解释——可迁移到任何需要LLM自我改进的场景

## 局限与展望
- 隐私和效用指标的评估仍依赖闭源LLM（如Gemini-2.5-flash），与完全本地化部署的目标存在矛盾
- 每个任务仍需少量标注数据（111训练+111验证），非完全零样本
- 未考虑推理型模型（如CoT模型）的匿名化能力，可能是互补方向

## 相关工作与启发
- **vs AF (Staab et al.)**: AF使用固定的对抗协作策略，依赖GPT-5，本文用进化优化自动搜索策略，且可在开源模型上运行
- **vs DP-Prompt**: 差分隐私方法提供理论保证但严重损害效用，本文在实际隐私-效用权衡上远优于DP-Prompt

## 评分
- 新颖性: ⭐⭐⭐⭐ 将匿名化重新定义为提示优化问题，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、3个开源模型、多个基线和消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 对敏感数据处理场景有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](../../CVPR2026/llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
