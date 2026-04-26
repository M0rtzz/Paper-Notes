---
title: >-
  [论文解读] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models
description: >-
  [ACL 2026][科学可行性评估] 构建控制知识框架系统研究LLM在科学可行性评估中如何利用实验描述和结果证据，发现提供结果证据比实验描述更可靠，部分实验信息常导致性能低于仅用参数知识的基线，揭示了LLM推理的脆弱性。
tags:
  - ACL 2026
  - 科学可行性评估
  - 控制知识框架
  - 证据鲁棒性
  - 实验vs结果
  - LLM推理
---

# Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.18786](https://arxiv.org/abs/2604.18786)  
**代码**: https://github.com/mohammadi-ali/scify  
**领域**: 科学推理 / LLM评估  
**关键词**: 科学可行性评估, 控制知识框架, 证据鲁棒性, 实验vs结果, LLM推理

## 一句话总结

构建控制知识框架系统研究LLM在科学可行性评估中如何利用实验描述和结果证据，发现提供结果证据比实验描述更可靠，部分实验信息常导致性能低于仅用参数知识的基线，揭示了LLM推理的脆弱性。

## 研究背景与动机

**领域现状**：LLM被越来越多地用于科学工作流程（文献综述、假设生成、实验规划），但其执行基本科学任务——科学可行性评估——的能力尚不清楚。可行性评估要求判断一个声明是否符合既有知识，以及实验证据能否支持或反驳它。

**现有痛点**：现有工作要么聚焦于假设生成而非评估，要么将模型内部知识与检索混合使用而未隔离各自的贡献，要么在非科学场景中检验外部知识的依从性。三个关键问题未被回答：（RQ1）LLM能否仅用参数知识评估可行性？（RQ2）提供实验/结果上下文如何改变判断？（RQ3）这些判断在信息不完整时多鲁棒？

**核心矛盾**：直觉上更多证据应该帮助判断，但部分/噪声证据可能反而误导——LLM是否能优雅地处理不完整信息？

**本文目标**：通过系统控制实验和结果的可见性，理解证据类型对LLM可行性判断的影响。

**切入角度**：设计4种知识条件（仅假设/+实验/+结果/+两者）和稳定性分析（渐进移除部分证据）。

**核心 idea**：结果证据通常比实验描述更可靠，部分证据常导致脆弱崩溃而非优雅退化。

## 方法详解

### 整体框架

给定科学假设h，在4种控制知识条件下评估LLM的可行性判断：H（仅假设）、H+E（+实验描述）、H+O（+结果摘要）、H+E+O（+两者）。通过参数 $k_1, k_2 \in \{0, 0.5, 1.0\}$ 控制实验和结果的可见比例，每个配置5次随机采样取平均。

### 关键设计

1. **控制知识框架（Controlled Knowledge Framework）**:

    - 功能：隔离不同类型证据对LLM可行性判断的影响
    - 核心思路：保持预测任务完全相同（输出可行/不可行+理由），仅改变伴随假设的上下文：$x \in \{\emptyset, \mathcal{E}^*, \mathcal{O}^*, (\mathcal{E}^*, \mathcal{O}^*)\}$。实验描述和结果从源论文中提取而非检索，确保证据质量。不同条件的预测差异完全反映证据影响而非任务变化
    - 设计动机：此前工作混合了多种信息源，无法区分哪种类型的证据真正有用

2. **稳定性分析（Stability Analysis）**:

    - 功能：测试LLM判断在证据不完整时的退化模式
    - 核心思路：渐进移除实验和/或结果的比例（$k_1, k_2$从1.0降到0.5再到0），观察性能是单调退化（优雅）还是非单调崩溃（脆弱）。定义"低于基线率"——部分证据条件下性能低于零证据(H)基线的比例
    - 设计动机：真实科学推理常基于不完整证据，如果部分证据反而误导模型，说明模型在做表面对齐而非深层推理

3. **多维评估体系**:

    - 功能：全面评估可行性判断的准确性和解释质量
    - 核心思路：评估准确率、macro-F1、MCC（类别不平衡下更有信息量），以及解释与参考解释的ROUGE词汇重叠（仅作诊断信号）。覆盖5个前沿LLM（GPT-5.1, GPT-4o, Gemini-2.5-Pro/Flash, Grok-4.1-fast），在两个数据集上测试
    - 设计动机：MCC在不平衡分类中比准确率更可靠，多模型评估确保发现的跨平台一致性

### 损失函数 / 训练策略

纯评估研究，使用零样本提示。所有模型使用相同的任务指令。

## 实验关键数据

### 主实验

GPT-5.1在MoF数据集上的表现：

| 条件 | Accuracy | F1_macro | MCC |
|------|----------|---------|-----|
| H（仅假设）| 0.68 | 0.67 | 0.42 |
| H+E（100%实验）| 0.70 | 0.69 | 0.44 |
| H+O（100%结果）| 0.66 | 0.66 | 0.33 |
| H+E+O（全部）| 0.66 | 0.66 | 0.33 |

### 消融实验

在Reasons数据集上（GPT-5.1）：

| 条件 | Accuracy | 说明 |
|------|----------|------|
| H | 0.84 | 参数知识基线 |
| H+E (50%) | 0.85 | 轻微提升 |
| H+O (100%) | 0.92 | 结果证据强 |
| H+E+O (100%) | 0.93 | 最优 |
| H+E (50%) + H+O (50%) | 0.90 | 部分证据有用 |

### 关键发现

- 结果证据（outcomes）通常比实验描述（experiments）更能改善可行性判断——在Reasons数据集上，H+O一致优于H+E
- 实验描述可能是"脆弱的"：部分实验信息（$k_1=0.5$）在多个模型上导致性能低于仅假设的基线，表明模型在做表面特征匹配而非真正理解实验设计
- 退化常是非单调的——$k_1=0.5$ 的表现可能比 $k_1=0$ 更差——说明模型不是在做"有多少信息用多少"的推理
- Gemini-2.5-Pro在实验描述条件下表现最不稳定（从0.67降到0.48），暴露了严重的表面对齐问题
- 即使是最强的GPT-5.1，提供完整实验+结果也不一定比仅提供结果更好（MoF数据集上MCC相同或更低）

## 亮点与洞察

- "部分证据反而有害"是一个深刻且令人警醒的发现：它揭示了LLM科学推理的根本脆弱性——模型更像是在做pattern matching而非真正理解实验的逻辑结构。这对将LLM用于科学评审和决策有重要警示。
- 控制知识框架的实验设计非常优雅：通过保持任务恒定、仅变化上下文，实现了干净的因果推断。这个方法论可以迁移到其他评估LLM知识利用能力的研究。
- "结果>实验"的发现意味着LLM更擅长处理陈述性知识（"发生了什么"）而非程序性知识（"怎么做的"）——这与LLM训练数据的性质一致。

## 局限与展望

- 仅使用零样本评估，微调或少样本设置可能产生不同结果
- 可行性判断被简化为二分类，真实科学可行性通常是一个频谱
- 实验和结果的提取质量可能影响结论——如果提取不完整本身就可能导致"脆弱性"
- 解释质量仅用ROUGE词汇重叠评估，无法真正衡量科学推理的逻辑正确性
- 仅测试了商业API模型，开源模型的表现可能不同

## 相关工作与启发

- **vs Qi et al. (2023) / Yang et al. (2024)**: 聚焦假设生成而非评估；本文填补了可行性判断的空白
- **vs Jansen et al. (2025)**: 混合内部知识和检索，未隔离各自贡献；本文的控制框架实现了干净的分离
- **vs Mohammadi et al. (2025)**: 研究LLM对外部知识的依从性但在非科学场景；本文专注于科学推理中的证据利用

## 评分
- 新颖性: ⭐⭐⭐⭐ 控制知识框架+稳定性分析的实验设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5模型×2数据集×9证据条件×5随机种子
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，实验设计严谨
- 价值: ⭐⭐⭐⭐ 对LLM科学推理能力的理解有重要推进

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](../../NeurIPS2025/interpretability/the_trilemma_of_truth_in_large_language_models.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](../../ICLR2026/interpretability/universal_properties_of_activation_sparsity_in_modern_large_language_models.md)

<!-- RELATED:END -->
