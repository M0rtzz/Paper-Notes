---
title: >-
  [论文解读] Position-aware Automatic Circuit Discovery
description: >-
  [ACL 2025][电路发现] 提出位置感知的边归因修补方法（PEAP）和数据集 Schema 机制，解决了自动电路发现中忽略位置信息导致的抵消效应和重要性高估问题，实现了更小且更忠实的电路发现。
tags:
  - ACL 2025
  - 电路发现
  - 位置感知
  - 机械可解释性
  - 归因修补
  - Schema
---

# Position-aware Automatic Circuit Discovery

**会议**: ACL 2025  
**arXiv**: [2502.04577](https://arxiv.org/abs/2502.04577)  
**代码**: [https://github.com/technion-cs-nlp/PEAP](https://github.com/technion-cs-nlp/PEAP)  
**领域**: 其他  
**关键词**: 电路发现, 位置感知, 机械可解释性, 归因修补, Schema

## 一句话总结

提出位置感知的边归因修补方法（PEAP）和数据集 Schema 机制，解决了自动电路发现中忽略位置信息导致的抵消效应和重要性高估问题，实现了更小且更忠实的电路发现。

## 研究背景与动机

电路分析是理解语言模型内部机制的核心方法——通过找到执行特定任务的最小计算子图来揭示模型的工作原理。然而，现有的**自动**电路发现方法（如 EAP）存在一个关键盲点：**忽略了位置信息**。

具体而言，位置无关的方法存在两大问题：

**抵消效应（低召回）**：一个组件在不同位置的重要性得分可能符号相反。跨位置求和时，正负抵消导致实际重要的边被遗漏。实验证实，在 IOI 任务中 top-1% 的边排名差异高达 17.1%。

**重要性高估（低精度）**：不考虑位置时，方法倾向于选择在多个位置有小影响的边，而忽略在少数位置有大影响的边。同样在 top-1% 中差异达 17.5%。

手动电路发现（如 IOI 电路、Greater-Than 电路）可以区分位置，但不可扩展且容易引入人类偏见。本文的目标是：**在自动化的同时保持位置敏感性**。

## 方法详解

### 整体框架

方法分为两个核心贡献：
1. **PEAP**（Position-aware Edge Attribution Patching）：将边归因修补扩展到跨位置边
2. **Schema**：定义语义标签来处理变长输入的位置对齐问题

### 关键设计

1. **位置感知边归因修补（PEAP）**：

    - 原始 EAP 仅处理同一位置的边。PEAP 扩展到**跨位置的注意力边**
    - 对于注意力头 $h^i_t$ 在位置 $t$，它通过 value、key、query 三种边与其他位置的节点相连
    - 通过分别修补 $v_{t'}$、$k_{t'}$、$q_t$，计算每种边的归因分数
    - 使用一阶线性近似：$M(x|e=e_{x'}) - M(x) \approx (z^*_{h^i_t} - z_{h^i_t})^\top \nabla_{z_{h^i_t}} M(x)$
    - 关键：分别评估每个位置的边重要性，而非跨位置聚合

2. **数据集 Schema**：

    - **问题**：现实数据集的输入长度不一，无法直接跨样本对齐位置
    - **方案**：定义语义 span（如 "Subject"、"Year" 等），将不同长度的样本映射到统一的抽象计算图
    - 映射函数 $f_\mathcal{S}^x$：将抽象图的边映射到具体样本计算图中的边集合
    - Schema 级别的归因分数：对映射到同一抽象边的所有具体边分数求和，再跨样本平均

3. **自动化 Schema 生成流水线**：

    - 使用 LLM（Claude 3.5 Sonnet）自动生成 Schema：采样多个子集分别生成，再统一为最终版本
    - **显著性增强**：使用 input×gradient 计算各 token 位置对目标度量的重要性，生成重要性掩码
    - 将掩码提供给 LLM，使其在设计 Schema 时考虑模型实际的计算模式
    - Schema 应用也交由 LLM 自动完成，有效率≥90% 被视为成功

### 损失函数 / 训练策略

这是一种推理时的分析方法，不涉及模型训练。关键的超参数是电路大小的选择——通过贪心算法逐步添加重要性最高的边来构建电路。

## 实验关键数据

### 主实验

在 Greater-Than 任务（GPT2-small）上，PEAP 发现的电路在相同忠实度下比非位置性电路小数个数量级。

**Hard Faithfulness 比较（多任务、多模型）**：

| 任务 | 模型 | PEAP+Schema | 非位置性 | 改进 |
|------|------|------------|---------|------|
| Greater-Than | GPT2-small | 更小电路达到同等忠实度 | 大电路才能达到忠实 | 显著 |
| IOI | GPT2-small | LLM+Mask≈人工Schema | 非位置性差距大 | 显著 |
| IOI | Llama-3-8B | LLM Schema略优于人工 | - | 显著 |
| Winobias | Llama-3-8B | Mask增强一致改善 | - | 显著 |

### 消融实验

**Schema 生成方法比较**：

| 方法 | 特点 | 忠实度 |
|------|------|--------|
| 人工设计 Schema | 金标准 | 高 |
| LLM + Mask | 自动 + 显著性引导 | ≈人工 |
| 仅 LLM | 纯自动 | 略低于 LLM+Mask |
| 无 Schema（EAP） | 非位置性 | 最低 |

**抵消与高估效应量化（IOI，GPT2-small）**：

| K% | 抵消差异 | 控制差异 | 高估差异 | 控制差异 |
|----|---------|---------|---------|---------|
| 1% | 17.1% | 3.9% | 17.5% | 3.6% |
| 5% | 13.4% | 2.4% | 14.6% | 2.1% |
| 10% | 12.1% | 2.3% | 12.4% | 2.2% |

### 关键发现

- 位置感知电路在所有任务和模型上都实现了更好的忠实度-电路大小权衡
- LLM 自动生成的 Schema + 显著性掩码可以匹配甚至超越人工设计的 Schema
- 在 Llama-3-8B 上，LLM 生成的 Schema 甚至优于为 GPT2-small 设计的人工 Schema——说明 Schema 应当针对具体模型定制
- 抵消效应可以在**单个样本内跨位置**发生，而非仅在样本间

## 亮点与洞察

- **理论直觉清晰**：抵消和高估两个问题的形式化非常直观，用简单的图示（Figure 2）就让读者理解了忽略位置的危害
- **全自动化流水线**：从 Schema 生成到应用再到电路发现，整个过程可以完全自动化，大大降低了机械可解释性研究的人力成本
- **模型感知的 Schema 设计**：通过 input×gradient 显著性分数让 LLM "看到" 模型的计算模式，是一种优雅的 AI 辅助可解释性方法
- **发现了一个反直觉的现象**：为 GPT2-small 精心设计的人工 Schema 迁移到 Llama-3-8B 后效果反而不如 LLM 自动生成的 Schema

## 局限与展望

- Schema 要求所有样本中的 span 按相同顺序出现，限制了对更自由文本格式的适用性
- 什么样的 Schema 是"好"的 Schema 缺乏先验原则，目前只能通过下游忠实度来事后评估
- LLM 自动应用 Schema 也有失败率，需要过滤无效应用
- 仅在 GPT2-small 和 Llama-3-8B 上实验，更大模型上的可扩展性有待验证
- 本文只用了 Claude 3.5 Sonnet，其他 LLM（Llama-3-70B、GPT-4o）在 Schema 应用上未能达标

## 相关工作与启发

- 直接改进了 EAP（Syed et al., 2023）的方法论，将位置维度引入自动电路发现
- 与手动电路发现工作（Wang et al., 2023 的 IOI 电路；Hanna et al., 2024 的 Greater-Than 电路）互补——PEAP 实现了自动化的位置感知
- Schema 的概念与 IOI 数据集中的角色标注（IO、S1 等）异曲同工，但将其推广为系统化方法
- 显著性引导的 Schema 生成是 LLM-as-agent 用于自动化可解释性的有趣实例

## 评分

- 新颖性: ⭐⭐⭐⭐ 位置感知电路发现是自然但重要的改进，Schema + 自动化流水线增加了实用价值
- 实验充分度: ⭐⭐⭐⭐ 三个任务、两个模型、多种 Schema 生成方法的完整比较
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机阐述清晰，图示精心设计，方法层层推进
- 价值: ⭐⭐⭐⭐ 为机械可解释性研究提供了更精确的工具和自动化流水线，有望成为标准实践

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICML 2025\] Position: We Need An Algorithmic Understanding of Generative AI](../../ICML2025/interpretability/position_we_need_an_algorithmic_understanding_of_generative_ai.md)
- [\[NeurIPS 2025\] OrdShap: Feature Position Importance for Sequential Black-Box Models](../../NeurIPS2025/interpretability/ordshap_feature_position_importance_for_sequential_black-box_models.md)

<!-- RELATED:END -->
