---
title: >-
  [论文解读] Revealing Multimodal Causality with Large Language Models
description: >-
  [NeurIPS 2025][多模态因果发现] 提出 MLLM-CD 框架，首次实现从多模态非结构化数据（文本+图像）中进行因果发现，通过对比因子发现识别因果变量、统计方法推断因果结构、迭代多模态反事实推理消除结构歧义。
tags:
  - NeurIPS 2025
  - 多模态因果发现
  - 大语言模型
  - 对比因子发现
  - 反事实推理
  - 非结构化数据
---

# Revealing Multimodal Causality with Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.17784](https://arxiv.org/abs/2509.17784)  
**代码**: [GitHub](https://github.com/JinLi-i/MLLM-CD)  
**领域**: 因果推断  
**关键词**: 多模态因果发现, 大语言模型, 对比因子发现, 反事实推理, 非结构化数据

## 一句话总结

提出 MLLM-CD 框架，首次实现从多模态非结构化数据（文本+图像）中进行因果发现，通过对比因子发现识别因果变量、统计方法推断因果结构、迭代多模态反事实推理消除结构歧义。

## 研究背景与动机

因果发现旨在从数据中推断因果结构，是科学进步的基础。传统因果发现方法依赖预定义的结构化变量，无法直接处理非结构化数据（文本、图像等）。随着多模态数据的激增（如医疗诊断中的临床笔记 + 医学影像 + 检验结果），从非结构化多模态数据中发现因果关系变得尤为迫切。

虽然 LLM 在文本因果发现上取得了进展（如 COAT），但将其扩展到多模态场景面临两大挑战：(1) **跨模态交互的因子发现困难**：因果变量可能嵌入在不同模态中，仅在跨模态交互中才可被识别（如"较小的苹果得分低"需要同时理解图像和文本）；(2) **结构歧义处理不足**：纯观测数据下多个因果结构可能产生相同的统计依赖，多模态设置中变量更多使歧义更严重。

COAT 的简单多模态扩展只能发现少量因果因子，且推断的因果边保持无向，远不足以解决多模态因果发现问题。

## 方法详解

### 整体框架

MLLM-CD 包含三个核心模块，以迭代方式运行：(1) 对比因子发现（CFD）模块利用 MLLM 从对比样本对中识别多模态因果变量；(2) 统计因果结构发现模块（如 FCI 算法）推断因果关系；(3) 迭代多模态反事实推理（MCR）模块通过生成反事实样本消除结构歧义并迭代优化。

### 关键设计

**1. 对比因子发现（CFD）模块**

- **功能**: 从多模态非结构化数据中识别完整的因果变量集合
- **核心思路**: 分为模态内和模态间两种对比探索。**模态内对比**：在每种模态中选择语义距离最大的 top-$K$ 样本对 $\mathcal{P}_i$，让 MLLM 分析差异中隐含的变量。**模态间对比**：构建跨模态错配最大的样本对，错配分数 $s(a,b) = (1 - \text{sim}(\mathbf{e}_{ai}, \mathbf{e}_{bj})) + |y_i - y_j|$，让 MLLM 识别跨模态依赖中隐藏的变量。最后通过提示 MLLM 合并去重，并为每个样本标注变量值
- **设计动机**: 单纯依赖 MLLM 的通用知识只能发现最显著的因子（如味道、香气），而对比信号可以揭示隐含但重要的因子（如营养成分）

**2. 迭代多模态反事实推理（MCR）模块**

- **功能**: 通过反事实样本生成消除因果结构中的歧义（如无向边）
- **核心思路**: 对不确定关系中的变量 $V_a$ 进行反事实干预——让 MLLM 预测假设 $V_a$ 取不同值时其他变量如何变化，并生成对应的多模态反事实样本。对生成的反事实样本执行两重验证：(1) **语义合理性**：确保反事实样本与原始样本的嵌入相似度 $\geq \tau_{\text{sem}}$；(2) **因果一致性**：验证干预变量的非后代节点变化比例 $R_{\text{indep}} \leq \tau_{\text{causal}}$。通过验证的样本加入数据集进行下一轮因果发现
- **设计动机**: 纯观测数据导致的马尔可夫等价类问题只能通过引入干预/反事实数据来解决。MLLM 的世界知识提供了超越观测数据的反事实证据

**3. 统计因果结构发现**

- **功能**: 从结构化数据中推断因果 DAG
- **核心思路**: 使用 FCI 算法处理可能存在未观测混淆因子的场景，将 CFD 输出的结构化数据 $\mathcal{D}_S^{(t)}$ 和变量集 $\mathbf{V}^{(t)} \cup \{Y\}$ 输入得到因果图 $\mathcal{G}^{(t)}$
- **设计动机**: 统计方法提供了因果推断的理论严谨性保证，MLLM 的推理能力作为补充而非替代

### 损失函数 / 训练策略

本文无模型训练。使用 GPT-4o、Gemini 2.0、LLaMA 4 Maverick 和 Grok-2v 四种 MLLM。对比探索使用 CLIP 提取语义表示，反事实图像使用 Stable Diffusion 3.5 或 Gemini 2.0 生成。

## 实验关键数据

### 主实验：MAG 数据集（Gemini 2.0）

| 方法 | NF ↑ | AF ↑ | ESHD ↓ |
|------|------|------|--------|
| META | 0.67 | 0.51 | 18.67 |
| COAT | 0.51 | 0.37 | 16.00 |
| Pairwise | - | 0.51 | 30.00 |
| **MLLM-CD** | **0.87** | **0.60** | **14.00** |

### 消融实验（Gemini 2.0）

| 变体 | MAG NF | MAG AF | MAG ESHD | Lung NF | Lung AF | Lung ESHD |
|------|--------|--------|----------|---------|---------|-----------|
| w/o Both | 0.54 | 0.41 | 16.33 | 0.55 | 0.13 | 9.67 |
| w/o CFD | 0.73 | 0.47 | 15.00 | 0.62 | 0.36 | 8.00 |
| w/o CR | 0.81 | 0.52 | 15.67 | 0.94 | 0.38 | 5.33 |
| **MLLM-CD** | **0.87** | **0.60** | **14.00** | **0.97** | **0.87** | **4.67** |

### 关键发现

1. **因子发现大幅领先**: MLLM-CD 的平均 NF 达到 0.89（跨 4 个 MLLM），远超 COAT 的 0.53 和 META 的 0.52
2. **结构发现显著改善**: 平均 ESHD 从 COAT 的 16.42 降至 13.42
3. **CFD 和 MCR 互补**: CFD 主要提升因子识别完整性，MCR 主要提升因果结构精度
4. **MCR 在小数据集上效果更显著**: Lung Cancer 数据集上 MCR 将 AF 从 0.38 提升至 0.87
5. **跨 MLLM 一致有效**: 在 GPT-4o、Gemini 2.0、LLaMA 4、Grok-2v 上均表现最优

## 亮点与洞察

- 首个面向多模态非结构化数据的因果发现框架，显著拓展了因果发现的适用范围
- 对比因子发现的模态内/模态间双重探索策略设计精妙，有效解决了隐含变量识别问题
- 反事实推理模块的双重验证（语义 + 因果一致性）机制巧妙地平衡了 MLLM 知识注入与统计严谨性
- 建立了首个多模态非结构化因果发现的基准数据集（MAG + Lung Cancer）

## 局限与展望

- 基准数据集规模较小（MAG 200 样本，Lung Cancer 60 样本），可扩展性有待验证
- MLLM 处理的模态范围受限于其自身能力，传感器数据、基因组数据等无法直接处理
- 因果图的 ground truth 依赖领域专家知识
- MLLM 可能存在幻觉和训练数据偏见，影响反事实推理质量
- 未来计划开发更大规模的基准、拓展模态范围、并研究不确定性量化

## 相关工作与启发

- **LLM 因果发现**: COAT 首次实现从非结构化数据的 LLM-driven CD，但仅限文本模态
- **因果表示学习**: 从低级观测数据提取高级表示和因果依赖，但实际应用仍有挑战
- **启发**: MLLM 的世界知识可以作为超越观测数据的反事实证据源，为因果发现提供了全新的方法论视角。统计方法与 LLM 推理的结合代表了因果发现的重要发展方向

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个多模态非结构化因果发现框架，对比因子发现和反事实推理模块均有原创性
- **实验充分度**: ⭐⭐⭐⭐ 合成 + 真实数据集 + 4 个 MLLM + 全面消融 + 采样策略分析
- **写作质量**: ⭐⭐⭐⭐ 问题定义严谨，方法描述详尽
- **价值**: ⭐⭐⭐⭐⭐ 开辟了多模态因果发现的新方向，对医疗诊断等领域具有重要应用前景

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)

<!-- RELATED:END -->
