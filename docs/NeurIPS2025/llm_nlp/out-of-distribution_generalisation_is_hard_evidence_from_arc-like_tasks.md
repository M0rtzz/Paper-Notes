---
title: >-
  [论文解读] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks
description: >-
  [NeurIPS 2025][LLM/NLP][OOD泛化] 通过构建具有明确OOD度量的ARC类任务，证明标准神经网络(MLP/CNN/Transformer)无法实现组合OOD泛化，即使设计具有正确归纳偏置的架构达到近乎完美的OOD性能，也可能学到错误的组合特征。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - OOD泛化
  - 组合泛化
  - ARC任务
  - 归纳偏置
  - 特征学习
---

# Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2505.09716](https://arxiv.org/abs/2505.09716)  
**代码**: 无  
**领域**: LLM/NLP / 泛化理论  
**关键词**: OOD泛化, 组合泛化, ARC任务, 归纳偏置, 特征学习

## 一句话总结

通过构建具有明确OOD度量的ARC类任务，证明标准神经网络(MLP/CNN/Transformer)无法实现组合OOD泛化，即使设计具有正确归纳偏置的架构达到近乎完美的OOD性能，也可能学到错误的组合特征。

## 研究背景与动机

分布外 (OOD) 泛化被认为是人类和动物智能的标志性能力。通过组合实现OOD泛化要求系统：
1. 发现环境不变的输入-输出映射属性
2. 将其迁移到新颖输入上

然而，现有研究存在关键问题：
- 仅在OOD设置上测试不足以证明算法学到了组合结构
- 还需确认所识别的特征确实是组合性的
- 当一个系统在OOD测试中表现良好时，我们无法确定它是否真正学到了正确的组合规则

作者的核心论点：**验证OOD泛化需要同时验证性能和特征正确性**。

## 方法详解

### 整体框架

1. 设计两个具有明确定义OOD度量的任务
2. 测试三种标准网络（MLP、CNN、Transformer）的OOD能力
3. 设计两种具有正确归纳偏置的新架构
4. 分析即使OOD性能完美时特征学习是否正确

### 关键设计

**任务设计**：
- 借鉴ARC (Abstraction and Reasoning Corpus) 的思路
- 任务具有可分解的组合结构
- OOD度量清晰定义：输入特征的新组合是否能正确处理

**任务1 - 几何变换任务**：
- 输入：具有特定几何属性的网格图案
- 输出：经过确定性变换的图案
- OOD设置：训练集和测试集的几何属性组合不同
- 关键特征：变换可以分解为独立的子操作

**任务2 - 颜色-形状组合任务**：
- 输入涉及颜色和形状的独立变化
- 正确的组合泛化要求分别学习颜色和形状的不变特征

**两种新架构**：
- 架构A：嵌入几何不变性的偏置，使网络能够分别处理各维度的变换
- 架构B：嵌入组合分解的偏置，强制网络将特征空间分解为独立子空间

### 损失函数 / 训练策略

- 标准交叉熵/MSE损失
- 各架构使用相同的训练数据和优化策略
- 训练-测试分割严格控制OOD程度

## 实验关键数据

### 主实验

标准网络的OOD泛化测试：

| 网络 | 任务1 ID准确率 | 任务1 OOD准确率 | 任务2 ID准确率 | 任务2 OOD准确率 |
|------|-------------|--------------|-------------|--------------|
| MLP | ~100% | ~0% | ~100% | ~0% |
| CNN | ~100% | ~0% | ~100% | ~0% |
| Transformer | ~100% | ~0% | ~100% | ~0% |

具有正确偏置的新架构：

| 架构 | 任务1 OOD准确率 | 任务2 OOD准确率 | 特征正确性 |
|------|--------------|--------------|----------|
| 偏置架构A | ~100% | ~100% | 部分不正确 |
| 偏置架构B | ~100% | ~100% | 部分不正确 |

### 消融实验

| 实验条件 | OOD性能 | 学到正确组合特征 |
|---------|--------|--------------|
| 无归纳偏置 | 失败 | 否 |
| 弱归纳偏置 | 部分成功 | 不确定 |
| 强归纳偏置 | 成功 | 不保证 |
| 完美归纳偏置 | 近完美 | 仍可能不正确 |

### 关键发现

1. **标准网络全面失败**：MLP、CNN、Transformer在两个任务上的OOD准确率均接近零
2. **正确偏置≠正确特征**：即使嵌入正确的归纳偏置实现了近完美OOD性能，网络仍可能学到不正确的组合特征
3. **OOD性能≠组合泛化**：高OOD性能并不能证明算法确实学到了底层的组合结构
4. **验证特征正确性的必要性**：仅测试OOD性能不足以确认组合泛化能力

## 亮点与洞察

1. **方法论贡献**：提出了评估组合泛化的新标准——不仅要看性能，还要验证特征正确性
2. **反直觉发现**：近完美的OOD性能可以通过"错误"的方式达成，即不依赖于正确的组合特征
3. **对ARC的启示**：解释了为什么当前AI系统在ARC任务上表现不佳——缺乏真正的组合推理能力
4. **理论与实践差距**：归纳偏置虽然必要但不充分，强调了理解模型学到什么的重要性

## 局限与展望

1. 任务设计相对简单，与真实ARC任务相比复杂度有限
2. 仅测试了三种标准架构和两种自定义架构
3. 特征正确性的验证方法较为手工化，缺乏自动化评估手段
4. 未探索更大规模模型或预训练模型的组合泛化能力
5. 未考虑in-context learning等新范式对OOD泛化的影响

## 相关工作与启发

- **ARC Challenge**：François Chollet提出的抽象推理基准
- **组合泛化**：SCAN、COGS等语义解析中的组合泛化测试
- **系统性泛化**：Lake & Baroni关于神经网络系统性泛化的讨论
- 启发：简单任务上的深入分析往往比复杂任务上的表面测试更有价值

## 评分

- 新颖性：⭐⭐⭐⭐ (方法论贡献显著)
- 技术深度：⭐⭐⭐⭐ (分析严谨)
- 实验充分性：⭐⭐⭐ (任务可更丰富)
- 实用价值：⭐⭐⭐ (主要是理论启示)

<!-- RELATED:START -->

## 相关论文

- [Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](../../ICML2025/llm_nlp/adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)
- [Zero-Shot Belief: A Hard Problem for LLMs](../../ACL2025/llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)
- [Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](../../ICML2025/llm_nlp/product_of_experts_with_llms_boosting_performance_on_arc_is_a_matter_of_perspect.md)
- [HumT DumT: Measuring and Controlling Human-like Language in LLMs](../../ACL2025/llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](../../ACL2025/llm_nlp/llm_braces_straightening.md)

<!-- RELATED:END -->
