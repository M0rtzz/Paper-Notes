---
title: >-
  [论文解读] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations
description: >-
  [NLP理解] 通过在 T5 编码器-解码器模型中简单插入一层 Poincaré 双曲层，以最小改动将欧几里得嵌入映射到双曲空间进行多跳推理，实验在四个数据集上一致性地超越欧几里得对照组，并证明了基于 δ-hyperbolicity 初始化曲率的有效性以及双曲空间在层次结构更强的数据集上优势更明显。
tags:
  - NLP理解
---

# Multi-Hop Reasoning for Question Answering with Hyperbolic Representations

- **会议**: ACL 2025
- **arXiv**: [2507.03612](https://arxiv.org/abs/2507.03612)
- **代码**: 未公开（使用了开源的 [2WikiMultiHop 评估代码](https://github.com/Alab-NII/2wikimultihop) 和 Poincaré 层实现）
- **领域**: NLP Understanding / 多跳推理 / 双曲表示学习
- **关键词**: 多跳推理, 双曲空间, Poincaré Ball, 知识图谱, 问答系统, 曲率初始化

## 一句话总结

通过在 T5 编码器-解码器模型中简单插入一层 Poincaré 双曲层，以最小改动将欧几里得嵌入映射到双曲空间进行多跳推理，实验在四个数据集上一致性地超越欧几里得对照组，并证明了基于 δ-hyperbolicity 初始化曲率的有效性以及双曲空间在层次结构更强的数据集上优势更明显。

## 背景与动机

多跳推理（Multi-Hop Reasoning）要求模型整合多条跨步证据来得出答案。例如"Cloudburst 的作曲者来自哪个国家？"需要先找到作曲者（Eric Whitacre），再查其国籍（American）。这种推理本质上是在知识图谱的层次/树状结构上做路径遍历。

传统语言模型使用欧几里得空间表示，虽然能在一定程度上捕获层次结构，但双曲空间在建模树状和图状结构时天然具有优势——体积随距离呈指数增长，更适合编码层次关系。然而，现有双曲推理工作存在两个主要问题：

1. **缺乏受控对比**：已有工作在引入双曲几何的同时引入了大量架构改动和额外参数，无法区分性能提升来自几何特性还是模型改变
2. **架构改动过大**：需要完全重新设计模型架构，增加复杂度和计算开销

本文的核心贡献在于提供了一个**严格受控**的对比实验，仅添加一个双曲层（与等参数量欧几里得层对比），隔离几何空间本身的影响。

## 方法详解

### 设计一：基于 PaTH 框架的两阶段软提示推理

方法基于 PaTH（Prompt-Aided T5 for multi-Hop reasoning）框架，包含两个核心阶段：

**阶段一：知识集成（Knowledge Integration）**

使用知识图谱中的三元组 $(e_1, r_1, e_2)$ 微调 T5 模型，让模型内化实体-关系结构。仅使用与 2-hop 问题相关的子图三元组。

**阶段二：软提示调优（Soft Prompt Tuning）**

训练两种软提示：
- **解析提示（Parsing Prompt）**：将自然语言问题解析为不完整路径序列 $(e_1, r_1, r_2, \ldots, r_n)$
- **跳跃提示（Hopping Prompt）**：给定不完整序列，使用知识图谱上的随机游走训练模型预测完整路径 $(e_1, r_1, e_2, r_2, \ldots, r_{n-1}, e_n)$

### 设计二：Poincaré Ball 双曲层的集成

模型在 T5 编码器输出后插入一层双曲处理流程（见 Figure 2）：

$$\text{T5 Encoder} \xrightarrow{\text{exp}_0^c} \text{Poincaré Ball} \xrightarrow{\text{Poincaré Layer}} \text{Poincaré Ball} \xrightarrow{\text{log}_0^c} \text{T5 Decoder}$$

**具体步骤**：

1. **指数映射**：将欧几里得嵌入 $v$ 映射到 Poincaré 球上

$$\exp_0^c(v) = \frac{\tanh(\|v\| \cdot \sqrt{c})}{\|v\| \cdot \sqrt{c}} \cdot v$$

2. **Poincaré 线性层**：在双曲空间中执行操作，使用双曲多项式逻辑回归的变换公式

$$v_k(x) = \frac{2}{\sqrt{c}} \|z_k\| \sinh^{-1}\left(\lambda_x^c \left\langle \sqrt{c}x, \frac{z_k}{\|z_k\|} \right\rangle \cosh(2\sqrt{c}r_k) - (\lambda_x^c - 1)\sinh(2\sqrt{c}r_k)\right)$$

其中 $\lambda_x^c = \frac{2}{1 - c\|x\|^2}$ 为保形因子，$Z = \{z_k\}$ 和 $r = \{r_k\}$ 分别为可训练权重和偏置。

3. **对数映射**：将双曲嵌入映射回欧几里得空间以兼容 T5 解码器

$$\log_0^c(y) = \frac{\tanh^{-1}(\|y\| \cdot \sqrt{c})}{\|y\| \cdot \sqrt{c}} \cdot y$$

### 设计三：基于 δ-hyperbolicity 的曲率初始化

通过 Gromov 乘积计算数据集的 δ-hyperbolicity，使用相对指标 $\delta_{rel}(X) = \frac{2\delta(X)}{\text{diam}(X)}$ 消除尺度影响。基于此初始化曲率参数：

$$c(X) = \left(\frac{0.144}{\delta_{rel}(X)}\right)^2$$

该曲率在训练过程中可学习更新，实验采样 1500 个点重复 5 次估计。

## 实验结果

### 实验设置

- 模型：T5-Large（770M 参数），冻结主体，仅训练附加层和软提示
- 优化器：AdaFactor，学习率 0.001，batch size 64
- 评估指标：Exact Match (EM)
- 数据集：4 个闭卷 QA 数据集（仅使用 2-hop 问题）

### 表 1：数据集统计信息

| 数据集 | 节点数 | 边数 | 关系数 | 训练/验证/测试 |
|:---:|:---:|:---:|:---:|:---:|
| 2WikiMultiHopQA | 97,298 | 95,116 | 29 | 72,760/8,085/6,768 |
| MetaQA | 31,374 | 58,974 | 9 | 47,108/5,951/5,942 |
| MLPQ | 51,402 | 53,327 | 72 | 57,283/7,160/7,161 |
| PQ | 1,056 | 1,211 | 13 | 1,698/210/191 |

### 表 2：Hopping Prompt 阶段的 EM 得分（%）

| 划分 | 模型 | 2WikiHop | MetaQA | MLPQ | PQ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Dev | Euclidean | 44.36 | 22.92 | 81.03 | 18.28 |
| Dev | **Hyperbolic** | **46.93** | **28.33** | **82.60** | **29.03** |
| Test | Euclidean | 14.88 | 19.76 | 72.10 | 11.90 |
| Test | **Hyperbolic** | **15.20** | **25.40** | **74.58** | **23.21** |

### 表 3：不同阶段使用双曲/欧几里得层的 Test EM 得分（%）

| Parsing | Hopping | 2WikiHop | MetaQA | MLPQ | PQ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Euclidean | Euclidean | 13.39 | 19.20 | 72.59 | 12.04 |
| Hyperbolic | Euclidean | 13.56 | 19.08 | 72.74 | 12.04 |
| Euclidean | **Hyperbolic** | **13.40** | **24.74** | **73.48** | **23.04** |
| Hyperbolic | Hyperbolic | 13.65 | 24.72 | 73.40 | 22.51 |

**关键发现**：主要增益来自 Hopping 阶段使用双曲层，因为该阶段直接依赖知识图谱的层次结构。

## 关键发现

1. **双曲一致优于欧几里得**：在所有 4 个数据集上，双曲层在 Hopping 阶段均超越等参数量的欧几里得层，PQ 数据集上提升最大（dev: +10.75%, test: +11.31%）
2. **曲率初始化至关重要**：基于 δ-hyperbolicity 初始化的曲率显著优于随机初始化；曲率过大（如 c=10）会导致性能崩溃（2WikiHop EM 降至 2.21%，MetaQA 降至 0.22%）
3. **层次结构越强，增益越大**：MLPQ 因 80% 节点出度为 1（接近线性），提升最小（+1.57%）；MetaQA 和 PQ 层次结构更复杂，提升更显著
4. **双曲空间扩展距离**：在 2WikiHop、MetaQA、PQ 上几乎 100% 的情况下双曲测地距离大于欧几里得距离，更利于路径消歧
5. **计算开销可忽略**：双曲层引入的推理时间和内存增加几乎为零
6. **无需软提示也有效**：去掉软提示后双曲层仍一致优于欧几里得层（表 6）

## 亮点

- **实验设计严谨**：唯一变量为几何空间类型，等参数量对比，是双曲多跳推理领域首个真正受控的对比研究
- **集成极简**：仅添加一个双曲层 + 指数/对数映射，不改变 T5 主体架构，冻结编码器/解码器参数
- **δ-hyperbolicity 驱动的曲率初始化**：将数据结构特性与模型几何对齐，提供了有理论依据的超参选择方法
- **多角度分析**：从性能、计算效率、嵌入距离、数据集难度等多维度解读，解释了"为什么"而非仅展示"是什么"

## 局限性

1. **仅限闭卷 QA**：未测试开放域或检索增强设置，信息受限于模型已训练知识
2. **冻结模型**：仅微调附加层（~1M 参数），当主模型全量微调（数十亿参数）时，单层双曲层的影响可能被稀释
3. **仅限编码器-解码器架构**：未扩展到 decoder-only 模型（如 GPT 系列），泛化性存疑
4. **仅测试 2-hop**：未验证更长推理链（3-hop 及以上）的表现
5. **数据集规模有限**：PQ 仅 1908 个问题，统计可靠性可能受影响

## 相关工作

- **基于路径的多跳推理**：Lao et al. (2011) 使用预定义规则在知识库上推理
- **神经嵌入方法**：TransE (Bordes et al., 2013)、RotatE (Sun et al.) 等将实体/关系向量化
- **图神经网络推理**：R-GCN (Schlichtkrull et al., 2018)、GAT (Veličković et al., 2018) 在多跳关系结构上传播信息
- **双曲知识图谱嵌入**：Poincaré Embedding (Nickel & Kiela, 2017)、MuRP (Balažević et al., 2019)、ATTH (Chami et al., 2020) 利用双曲空间建模层次关系
- **双曲图神经网络**：HGCN (Chami et al., 2019)、DeepHGCN (Liu et al., 2024) 在双曲空间中做消息传递
- **PaTH 框架**：Misra et al. (2023) 提出软提示 + 随机游走的 T5 多跳推理基线

## 评分

- **创新性**: ⭐⭐⭐ — 方法本身不复杂（仅加一层），但实验设计思路（隔离几何空间的影响）严谨且填补了对照实验的空白
- **有效性**: ⭐⭐⭐⭐ — 在四个数据集上一致性优于基线，消融实验全面，多角度分析有说服力
- **实用性**: ⭐⭐⭐ — 集成简单、开销小，但局限于特定架构和闭卷设置，工程落地场景较窄
- **推荐阅读**: ⭐⭐⭐⭐ — 适合关注几何深度学习 × NLP 交叉方向的研究者，对理解双曲空间在推理中的作用有很好的参考价值

<!-- RELATED:START -->

## 相关论文

- [BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)
- [iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data](recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)

<!-- RELATED:END -->
