---
title: >-
  [论文解读] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding
description: >-
  [ICLR 2026][Transformer] 通过对训练梯度的前导项近似分析，推导出Transformer在训练早期阶段各权重矩阵的闭式表达——均可分解为三种基函数（bigram、token-interchangeability、context mapping）的简单组合——从而揭示Transformer如何从自然语言数据中学习"bird"↔"flew"这类语义关联，且理论预测与真实LLM的学到权重高度吻合。
tags:
  - ICLR 2026
  - Transformer
  - 训练动态
  - 梯度前导项
  - 语义关联
  - 闭式权重表达
---

# How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding

**会议**: ICLR 2026  
**arXiv**: [2601.19208](https://arxiv.org/abs/2601.19208)  
**代码**: 无  
**领域**: LLM / NLP (Mechanistic Interpretability)  
**关键词**: Transformer可解释性, 训练动态, 梯度前导项, 语义关联, 闭式权重表达

## 一句话总结
通过对训练梯度的前导项近似分析，推导出Transformer在训练早期阶段各权重矩阵的闭式表达——均可分解为三种基函数（bigram、token-interchangeability、context mapping）的简单组合——从而揭示Transformer如何从自然语言数据中学习"bird"↔"flew"这类语义关联，且理论预测与真实LLM的学到权重高度吻合。

## 研究背景与动机
"bird"和"flew"之间的语义关联是语言建模的基础——模型需要超越记忆，实现泛化和连贯文本生成。理解这些关联如何在语言模型中被学习和表示，对于连接深度学习与语言学理论、建立大型语言模型的机制性基础至关重要。

**当前问题**：
- 现有Transformer可解释性研究主要分为两条路线：(1) 分析训练好的模型的内部表示（如attention可视化、probing）；(2) 简化模型/任务的理论分析（如单头attention在合成数据上的分析）
- **关键gap**：缺乏对Transformer如何在**训练过程中**从真实自然语言数据中逐步学习语义关联的**机制性理解**
- 已有的基于简化假设的理论分析（如线性attention、单层模型）难以直接解释多层、多头、真实训练的Transformer

**本文切入点**：从**训练动态**的视角出发，利用梯度的**前导项近似**，推导出可解析、可验证的权重闭式表达。这种方法不假设简化的架构，而是通过数学分析梯度更新中最主要的贡献项来表征早期训练阶段模型权重的形成过程。

**核心洞察**：Transformer的每组权重都可以表达为三种反映文本语料统计特性的基函数的简单组合，每种基函数对应一种语义关联的学习机制。

## 方法详解

### 整体框架
- 输入/对象：基于attention的语言模型（Transformer），在自然语言语料上做标准自回归或MLM训练
- 分析工具：对每步梯度更新进行Taylor展开式的前导项（leading-term）分析
- 输出：Transformer各权重矩阵（Query $W_Q$、Key $W_K$、Value $W_V$、Embedding等）在训练早期阶段的闭式表达

### 关键设计

1. **梯度前导项近似**：

    - 对梯度进行分解，识别出对权重更新贡献最大的主导项
    - 忽略高阶项后，梯度的前导项可以用语料库的统计量来表达
    - 这一近似在训练早期（权重尚小时）特别精确

2. **三种基函数（Basis Functions）**：
   推导结果显示，Transformer的每组权重都可以表达为以下三种基函数的组合：
   
    - **Bigram映射**：捕捉相邻token对的共现统计。例如"the"→"cat"的高频共现导致对应权重增大。这是最直接的序列统计——当token $A$ 频繁出现在token $B$ 之前时，模型学习到$A \rightarrow B$的关联
   
    - **Token-interchangeability映射**：捕捉"可互换性"——在相似上下文中出现的token之间的关系。例如，"dog"和"cat"虽然不是相邻共现的token，但它们在"The ___ sat on the mat"这类上下文中具有互换性。这反映了分布式语义假说（distributional semantics）
   
    - **Context映射**：捕捉更广泛的上下文模式——特定上下文如何影响对后续token的预测。这一基函数编码了"给定一个上下文窗口，什么token最可能跟随"的统计规律

3. **闭式权重表达**：

    - 每个权重矩阵（$W_Q$, $W_K$, $W_V$等）在训练早期可以写成上述三种基函数的线性组合
    - 组合系数取决于架构细节（层数、头数）和训练超参数
    - 这些闭式表达揭示了Transformer各组件的**功能性分工**：
        - Query-Key权重主要依赖bigram和token-interchangeability → 决定"attend to what"
        - Value权重主要依赖context mapping → 决定"传递什么信息"

### 理论贡献的层次
1. **描述性**：给出了权重的数学表达形式
2. **解释性**：说明了每个组件如何基于语料库统计捕获语义关联
3. **预测性**：理论表达与真实训练的LLM权重的定量对比

## 实验关键数据

### 理论 vs 真实权重对比

| 验证维度 | 结果 | 说明 |
|---------|------|------|
| 权重近似精度 | 高度吻合 | 理论闭式表达与实际训练权重的模式高度一致 |
| 在真实LLM上验证 | 成功 | 不仅在小模型上，在实际规模的LLM上也验证了理论预测 |
| 定性分析 | 可解释 | 闭式表达能够解释模型学到的具体语义关联模式 |

### 消融实验

| 分析维度 | 关键发现 | 说明 |
|---------|---------|------|
| 训练早期 vs 后期 | 早期近似更精确 | 符合前导项近似的理论预期 |
| 不同基函数贡献 | 三者均有显著贡献 | 移除任一基函数都会显著降低近似质量 |
| 不同权重矩阵 | 功能性分化 | Q/K权重更依赖bigram，V权重更依赖context |
| 不同层的行为 | 层间差异 | 浅层更偏向bigram，深层更偏向context |

### 关键发现
- **Transformer权重的"可分解性"**：复杂的权重矩阵可以分解为仅3种基于语料库统计的简单基函数的组合，这大幅简化了对Transformer学习机制的理解
- **语义关联的涌现机制**：语义关联不是被直接编码的，而是通过bigram共现统计、分布式可互换性和上下文模式的交互作用涌现出来的
- **功能性分化**：Q/K矩阵和V矩阵在语义关联编码中扮演本质不同的角色，这与直觉一致但首次有了理论支撑
- **理论预测的实用性**：闭式表达在实际LLM上得到验证，说明这不仅是理论上的优美结果，而是具有实际解释力的工具

## 亮点与洞察
- **从训练动态视角理解Transformer**：不同于多数可解释性工作分析"训练好的模型做了什么"，本文分析"模型是如何学到这些的"——这提供了更根本的理解
- **三基函数分解的简洁性**：将高维复杂的权重矩阵归结为3种有清晰统计语义的基函数，既优美又有解释力，是连接深度学习与计算语言学的桥梁
- **分布式语义假说的理论证据**：token-interchangeability基函数直接对应"在相似上下文中出现的词具有相似语义"的语言学假说，本文从训练动态的纯数学分析中自然导出了这一结论
- **定量验证而非仅定性描述**：不仅仅是理论推导，还在真实LLM上做了系统性的定量验证

## 局限与展望
- 前导项近似在训练后期（权重增大后）精度下降，对完全训练后的模型解释力有限
- 分析主要集中在训练早期阶段，训练中后期的非线性效应和特征复杂化未被充分捕捉
- 三种基函数的分解是否适用于更大规模的模型（数十亿参数级别）需要进一步验证
- 仅分析了语义关联这一特定能力，Transformer的其他能力（如推理、计划）是否也可以用类似框架分析？
- 闭式表达的实际应用场景（如指导模型初始化、架构设计或知识编辑）尚未被充分探索
- 对多层Transformer中层间交互和残差连接的影响分析有待深入

## 相关工作与启发
- **Mechanistic Interpretability**（Olah et al., 2020）：电路级别的Transformer可解释性分析
- **Induction Heads**（Olsson et al., 2022）：attention head级别的特定计算模式
- **Transformer training dynamics**（Li et al., 2023等）：分析Transformer训练过程的动态特性
- **Distributional Semantics**（Harris, 1954; Firth, 1957）：词的语义由其分布上下文决定——本文为这一经典假说提供了从神经网络训练动态角度的理论支撑
- **Feature Learning Theory**：近年来关于神经网络如何学习特征的理论工作（lazy training vs feature learning regime）
- 启发：梯度前导项分析是一种强大但被低估的分析工具，可能广泛应用于理解其他神经网络架构和任务。将训练动态与语料库统计直接联系起来的方法论，为"数据如何塑造模型"提供了新的分析框架

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （梯度前导项分析 + 三基函数分解 = 全新理论框架）
- 实验充分度: ⭐⭐⭐⭐ （在真实LLM上验证，但可扩展到更大模型）
- 写作质量: ⭐⭐⭐⭐ （理论深度高，数学推导严谨）
- 价值: ⭐⭐⭐⭐⭐ （对Transformer机制性理解的重要理论贡献，连接深度学习与语言学）

<!-- RELATED:START -->

## 相关论文

- [How Do Transformers Learn Implicit Reasoning?](../../NeurIPS2025/interpretability/how_do_transformers_learn_implicit_reasoning.md)
- [Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance](stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)
- [Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)
- [Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)

<!-- RELATED:END -->
