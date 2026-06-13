---
title: >-
  [论文解读] OASIS: Order-Augmented Strategy for Improved Code Search
description: >-
  [ACL 2025][代码智能][代码搜索] 提出OASIS方法，通过为负样本对引入基于序的相似度标签来捕捉代码语义中的细微差异，结合InfoNCE和CoSENT双重损失函数训练代码嵌入模型，在CoSQA、AdvTest和CodeSearchNet三个基准的NL2Code和Code2Code搜索任务上全面超越现有SOTA。
tags:
  - "ACL 2025"
  - "代码智能"
  - "代码搜索"
  - "代码嵌入"
  - "对比学习"
  - "序标签"
  - "负样本对"
  - "相似度精调"
---

# OASIS: Order-Augmented Strategy for Improved Code Search

**会议**: ACL 2025  
**arXiv**: [2503.08161](https://arxiv.org/abs/2503.08161)  
**代码**: [HuggingFace](https://huggingface.co/Kwaipilot/OASIS-code-embedding-1.5B)  
**领域**: 其他（代码搜索/代码嵌入）  
**关键词**: 代码搜索, 代码嵌入, 对比学习, 序标签, 负样本对, 相似度精调

## 一句话总结

提出OASIS方法，通过为负样本对引入基于序的相似度标签来捕捉代码语义中的细微差异，结合InfoNCE和CoSENT双重损失函数训练代码嵌入模型，在CoSQA、AdvTest和CodeSearchNet三个基准的NL2Code和Code2Code搜索任务上全面超越现有SOTA。

## 研究背景与动机

代码搜索任务旨在根据自然语言查询检索最匹配的代码片段，是代码相关LLM应用（如RAG、代码补全等）的核心基础。当前主流方法使用对比学习训练代码嵌入模型，通过InfoNCE损失函数拉近正样本对，推远负样本对。

然而，现有方法存在关键缺陷：

**过度依赖正负样本的"主要差异"**：当前训练仅关注正负样本对之间的大差异，忽略了负样本对之间的细微差异。由于代码上下文的稀疏性，即便微小的改动也可能导致功能和语义上的显著变化。

**表面语义匹配的陷阱**：仅关注正负差异容易导致模型学到表面特征。例如，给定一个NL查询，模型可能将其与存在高词汇重叠但功能不同的代码匹配，而忽略词汇重叠低但语义正确的代码。

**代码领域相似度标注的困难**：与文本嵌入领域已有STS（语义文本相似度）数据集不同，代码的相似度标注因上下文稀疏性而更加困难，导致该领域发展滞后。

OASIS的核心思想是：**利用负样本对之间的细微差异（通过序标签捕捉）来学习更深层的代码语义**，而非仅依赖正负样本之间的粗粒度区分。

## 方法详解

### 整体框架

OASIS包含三个核心步骤：（1）Docstring生成与相似度标注；（2）相似度精调；（3）混合损失训练。数据来源于GitHub开源代码仓库，共合成5300万NL-Code对，覆盖9种编程语言。

### 关键设计

#### 1. Docstring生成与相似度标注

- **仓库级程序分析**：对每个函数提取其调用者和被调用者信息，连同代码本身一起构建prompt，利用LLM为代码生成高质量docstring
- **仓库内负样本构造**：对于给定的docstring A，从同一仓库内随机选取K个其他函数的代码片段，构成负样本对。同一仓库内的代码往往语义相近甚至词汇重叠，天然构成了高质量的"硬负样本"
- **相似度标签计算**：使用另一个嵌入模型（Text-Embedding-3-Large）计算每个负样本对的相似度分数 $sim \in [0,1)$，作为序标签提供额外的训练信号

#### 2. 相似度精调（Similarity Refinement）

初始相似度标签可能存在误标注，需要进一步精调：

- **GMM阈值方法**：用高斯混合模型（GMM）拟合所有样本对的相似度分布（呈双峰分布），取两个分布的交点作为正负分界阈值 $s^*$。若负样本对的相似度超过 $s^*$ 或超过对应正样本对的相似度，则该负样本对可能被误标注
- **AST编辑距离方法**：解析代码的抽象语法树（AST），选择AST编辑距离与两树节点之和之比低的候选对——即结构高度相似但词汇层面不相似的配对
- **LLM二选一判断**：对候选配对，使用LLM判断"候选代码是否也满足docstring的描述"，若是，则将该负样本对的相似度正向调整 $\Delta s$（通过网格搜索确定最优值）

#### 3. 混合损失训练

OASIS采用两个互补的损失函数：

**InfoNCE损失**——传统的对比学习目标，关注batch内正负样本对的整体区分：

$$\mathcal{L}_{ibn} = -\sum_b \sum_{i=1}^m \log \frac{\exp(\cos(h_i, h_i^+) / \tau)}{\sum_{j=1}^N \exp(\cos(h_i, h_j) / \tau)}$$

**CoSENT损失**——基于序的优化目标，关注样本对相似度的相对排序关系：

$$\mathcal{L}_{cos} = \log \left[1 + \sum_{s_{ij} > s_{mn}} \exp\left(\frac{\cos_{nm} - \cos_{ij}}{\tau}\right)\right]$$

CoSENT不强求模型预测精确的相似度值，而是确保预测的相似度排序与标签排序一致。当标签显示对 $(i,j)$ 的相似度高于 $(m,n)$ 但模型的预测相反时，将产生损失。

**总损失**为两者的加权组合：$L = w_1 \cdot L_{ibn} + w_2 \cdot L_{cos}$

### 信息论视角

InfoNCE关注整体嵌入的"粗粒度"区分（正vs负），CoSENT聚焦于嵌入之间的"细粒度"相对关系。两者互补：InfoNCE建立全局嵌入空间结构，CoSENT在此基础上精调局部排序关系。

## 实验关键数据

### 主实验——NL2Code搜索（MRR@1000）

| 方法 | CoSQA | AdvTest | CSN Python | CSN Java | CSN JS | CSN PHP | CSN Go | CSN Ruby | CSN Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI-ada-002 | 44.23 | 38.08 | 68.02 | 71.49 | 67.50 | 60.62 | 85.63 | 74.72 | 71.33 |
| Text-Embed-3-Large | 55.38 | 46.84 | 70.84 | 72.92 | 68.13 | 59.59 | 87.64 | 75.25 | 72.40 |
| CodeSage-large | 47.53 | 52.67 | 70.77 | 70.21 | 69.50 | 61.33 | 83.71 | 71.92 | 71.24 |
| **OASIS** | **55.77** | **57.27** | **73.69** | **73.97** | **69.80** | **63.84** | **88.21** | **75.47** | **74.16** |

### 主实验——Code2Code搜索（MAP）

| 方法 | Python | Java | JS | TS | C# | C | Ruby | PHP | Go | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Text-Embed-3-Large | 41.51 | 25.75 | 22.40 | 22.45 | 11.56 | 32.82 | 41.70 | 43.47 | 21.57 | 29.25 |
| CodeSage-large | 46.70 | 33.13 | 37.16 | 41.18 | 16.81 | 32.89 | 54.12 | 52.13 | 32.48 | 38.51 |
| **OASIS** | **66.27** | **37.26** | **47.71** | **51.15** | **22.18** | **49.38** | **58.60** | **64.06** | **34.18** | **47.87** |

Code2Code任务上OASIS相比CodeSage-large提升了24.31%相对（9.36%绝对），远超NL2Code任务的提升幅度。

### 消融实验（NL2Code平均MRR@1000）

| 配置 | MRR |
|------|:---:|
| **完整OASIS** | **69.75** |
| w/o 相似度精调 | 69.15 |
| 仅order目标（CoSENT） | 67.33 |
| 仅对比目标（InfoNCE） | 65.49 |
| 仅AST候选对策略 | 69.46 |
| 仅阈值候选对策略 | 69.26 |

### 困难样本子集——CSN Python（MRR@1000）

| 方法 | MRR |
|------|:---:|
| CodeSage-Large | 45.67 |
| Text-Embedding-3-Large | 45.78 |
| **OASIS** | **51.13** |

### 关键发现

1. **序标签比对比损失贡献更大**：仅用CoSENT（67.33）优于仅用InfoNCE（65.49），表明负样本对间的序关系信息对代码语义学习至关重要
2. **Code2Code提升远大于NL2Code**：Code2Code任务更难、更依赖细粒度语义理解，OASIS在此类任务上的优势更明显
3. **超越标注模型**：OASIS的相似度标签由Text-Embedding-3-Large生成，但最终性能超越了该模型本身，证明方法的有效性不依赖于标注模型的能力上限
4. **困难样本优势显著**：在所有模型都表现不佳的困难子集上，OASIS的MRR提升超过5%，说明序标签帮助模型学到了更本质的语义特征
5. **两种精调策略互补**：GMM阈值方法关注数值层面的异常标注，AST方法关注结构层面的相似性，组合使用效果最佳

## 亮点与洞察

1. **从"区分正负"到"排序负样本"的范式转变**：这是第一个在代码嵌入中探索负样本对间细微差异的工作。不同于硬负样本挖掘（关注单个难负样本），OASIS通过序标签建立负样本之间的全序关系
2. **仓库级数据增强的巧妙设计**：同仓库内的代码天然具有高相似度但功能不同的特性，无需人工标注即可获得海量高质量负样本对
3. **程序分析与LLM互补**：用AST结构信息捕捉代码层面的相似性，用LLM判断语义层面的等价性，两者分别从不同维度精调标注质量
4. **可视化证据充分**：MDS降维可视化表明，OASIS的嵌入空间中query与target code的距离更近，不同query的检索空间重叠更少

## 局限与展望

- 初始相似度标签依赖外部嵌入模型（Text-Embedding-3-Large），标注质量受限于该模型
- 仅在函数级代码上验证，未探索文件级或项目级代码搜索场景
- GMM阈值假设相似度分布为双峰高斯分布，对多峰分布或偏态分布的适用性未验证
- 数据合成管线中LLM生成docstring和LLM判断等价性的成本较高，规模化生产可能受限

## 相关工作与启发

- **CodeSage**：使用对比学习和大规模预训练的代码嵌入先驱，OASIS在其基础上引入序标签实现进一步提升
- **CoSENT**：文本嵌入中利用STS标签进行序优化的工作，OASIS将其迁移到代码领域并解决了代码相似度标注困难的问题
- **对NLP嵌入的启发**：序标签思路可扩展到其他稀疏上下文的嵌入任务，如SQL查询嵌入、配置文件嵌入等

## 评分

- **新颖性**: ⭐⭐⭐⭐ 序标签+双重损失的思路新颖，首次在代码嵌入中系统探索负样本对间的细粒度差异
- **实验充分度**: ⭐⭐⭐⭐ NL2Code和Code2Code两类任务×3个基准+详尽消融+困难子集分析+可视化
- **写作质量**: ⭐⭐⭐⭐ 方法和动机阐述清晰，Figure 1的例子直观说明了问题
- **价值**: ⭐⭐⭐⭐ 实用价值高，53M训练数据和1.5B模型权重已开源，可直接使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoRet: Improved Retriever for Code Editing](coret_improved_retriever_for_code_editing.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](../../AAAI2026/code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)

</div>

<!-- RELATED:END -->
