---
title: >-
  [论文解读] VeriTrail: Closed-Domain Hallucination Detection with Traceability
description: >-
  [ICLR 2026][hallucination detection] 提出 VeriTrail，首个面向多步生成（MGS）过程的闭域幻觉检测方法，通过将生成过程建模为 DAG 并沿图逐层验证 claim，实现了幻觉检测+溯源（provenance）+错误定位（error localization）的完整可追溯性，在两个新数据集上显著优于所有基线。
tags:
  - ICLR 2026
  - hallucination detection
  - faithfulness evaluation
  - traceability
  - multi-step generation
  - DAG
---

# VeriTrail: Closed-Domain Hallucination Detection with Traceability

**会议**: ICLR 2026  
**arXiv**: [2505.21786](https://arxiv.org/abs/2505.21786)  
**代码**: 无（数据集将发布于 https://aka.ms/veritrail-datasets）  
**领域**: LLM/NLP  
**关键词**: hallucination detection, faithfulness evaluation, traceability, multi-step generation, DAG

## 一句话总结
提出 VeriTrail，首个面向多步生成（MGS）过程的闭域幻觉检测方法，通过将生成过程建模为 DAG 并沿图逐层验证 claim，实现了幻觉检测+溯源（provenance）+错误定位（error localization）的完整可追溯性，在两个新数据集上显著优于所有基线。

## 研究背景与动机
语言模型在基于源文档生成内容时常产生不忠于原文的内容（闭域幻觉）。随着多步生成过程（MGS，如层次化摘要、GraphRAG）的广泛应用，幻觉风险被放大——每一步都可能引入或传播错误。

**现有方法的核心痛点**：
1. 现有幻觉检测方法将最终输出与源文档直接对比，**不区分中间输出和最终输出**
2. 对 SGS（单步生成）这没问题，但对 MGS 来说，只检测"有无幻觉"远远不够——还需要知道**幻觉在哪里产生的**（error localization）和**忠实内容是如何推导来的**（provenance）
3. 简单方法（逐一对比中间输出）在中间输出数量巨大时（>100K 个）计算上不可行，且无法处理多个中间输出**联合**支持一个 claim 的情况

**切入角度**：将生成过程建模为 DAG（有向无环图），设计逐层回溯的验证算法，在检测幻觉的同时自动构建证据链和定位错误阶段。

## 方法详解

### 整体框架
将生成过程建模为 DAG $G=(V,E)$：
- **节点** $v \in V$：表示文本片段（源文档 / 中间输出 / 最终输出）
- **边** $(u,v) \in E$：表示 $u$ 被用作生成 $v$ 的输入
- **根节点** $V_0$：源文档，**终端节点** $v^*$：最终输出

VeriTrail 对最终输出提取的每个 claim 独立执行以下迭代过程，从终端节点向根节点回溯。

### 关键设计
1. **Sub-Claim 分解**：

    - 使用 Claimify 将复合 claim 分解为独立可验证的子声明
    - 例："公司X在2020年收购了两家医疗创业公司" → (1) 收购了两家创业公司 (2) 在2020年 (3) 属于医疗扩张战略
    - 子声明作为后续验证的上下文保留，但不直接验证

2. **Evidence Selection（证据选择）**：

    - 取终端节点的源节点 $src(v^*)$，将其切分为句子并分配唯一 ID
    - LLM 选择强烈支持或反驳 claim 的句子 ID
    - 关键：返回的 ID 必须与程序化分配的 ID 匹配，**保证选中的句子不会是幻觉**
    - 支持并行处理以降低延迟

3. **Verdict Generation（判定生成）**：

    - 基于选中的证据句子，LLM 给出三级判定：Fully Supported / Not Fully Supported / Inconclusive
    - 避免冗余上下文：根节点保留全文，中间节点使用 Evidence Selection 生成的摘要
    - 超出上下文限制时自动重跑 Evidence Selection 进行压缩

4. **Candidate Node Selection & 终止条件**：

    - Fully Supported/Inconclusive → 继续验证证据来源节点
    - Not Fully Supported → 扩大搜索范围（验证所有检查过的节点的源节点），降低假阳性
    - 终止条件：(1) 仅剩已验证的根节点 (2) 无候选节点 (3) 连续 $q$ 次 Not Fully Supported
    - 超参 $q$ 控制检测严格度：$q=1$ 偏向高 recall，$q=3$ 偏向高 precision

### 可追溯性输出
- **Provenance**：对 Fully Supported 的 claim，返回从终端到根节点的完整证据链
- **Error Localization**：对 Not Fully Supported 的 claim，定位幻觉最可能被引入的 DAG 阶段

## 实验关键数据

### 主实验（幻觉检测性能）
在 FABLES+（书籍层次化摘要，734 claims）和 DiverseSumm+（GraphRAG 新闻问答，560 claims）上：

| 方法 | Macro F1 (F) | Macro F1 (D) | Bal. Acc (F) | Bal. Acc (D) |
|------|:---:|:---:|:---:|:---:|
| **VeriTrail (q=3)** | **84.5** | **79.5** | **83.6** | 76.3 |
| **VeriTrail (q=1)** | 74.0 | 76.6 | 84.6 | **83.0** |
| RAG (k=15) | 69.6 | 75.1 | 76.5 | 74.0 |
| Bespoke-MiniCheck-7B | 62.2 | 72.1 | 69.0 | 69.4 |
| Gemini 1.5 Pro | 61.1 | 49.8 | 60.8 | 57.6 |
| GPT-4.1 Mini | 60.7 | 62.9 | 58.2 | 61.5 |
| AlignScore | 59.6 | 60.4 | 67.5 | 62.7 |
| INFUSE | 40.5 | 20.0 | 59.5 | 50.1 |

- VeriTrail 在两个数据集上的 Macro F1 均**领先最强基线 5-15 个百分点**
- 长上下文 LLM（Gemini 1.5 Pro, GPT-4.1 Mini）表现不佳，说明直接塞入长文本并非好策略

### 消融实验（组件贡献）
论文在附录 E.1 进行了详细消融：

| 组件 | 移除后 Macro F1 变化 (FABLES+) |
|------|:---:|
| 去掉 sub-claim 分解 | 下降 |
| 去掉 DAG 结构（仅验证终端 vs 源文档） | 显著下降（退化为 RAG） |
| 去掉 Not Fully Supported 扩展搜索 | 假阳性增加 |
| q=1 vs q=3 | q=3 提升 precision，q=1 提升 recall |

### 关键发现
1. **MGS 专属价值**：在 FABLES+ 中（层次化摘要，超过 100K 中间输出），VeriTrail 的 DAG 回溯机制发挥关键作用
2. **成本效率**：尽管验证负担显著更大，VeriTrail 的总成本仍在合理范围内
3. **错误阶段分布**：分析显示不同阶段的幻觉引入概率不同，为 MGS 流程优化提供了指引

## 亮点与洞察
- **首次系统处理 MGS 幻觉检测**：填补了"多步生成过程缺乏可追溯性"这一重要空白
- **DAG 建模的优雅性**：将异构生成过程统一为 DAG 表示，算法通用性强
- **证据 ID 的工程巧思**：通过 programmatic ID 分配 + LLM 返回 ID 匹配，巧妙地消除了"用 LLM 验证 LLM"中的二次幻觉风险
- **q 参数提供灵活性**：用户可根据场景（高召回 vs 高精度）调整检测行为

## 局限与展望
- 仅评估了两种 MGS 过程（层次化摘要 + GraphRAG），通用性有待在更多场景验证
- 依赖 LLM 作为 evaluator，本身可能引入系统性偏差
- 对 Inconclusive 判定的处理较粗糙（实验中直接排除）
- 数据集规模相对较小（~1300 claims），统计显著性有限
- 未对比 reference-free 方法（如基于注意力图的方法）

## 相关工作与启发
- 与 NLI 方法（AlignScore, INFUSE）互补：VeriTrail 解决了它们无法处理大规模 MGS 的局限
- 对 RAG 系统的实际启示：当 RAG 后接多步处理时，应考虑全链路验证
- 对 LLM agent 系统的启发：agent 的多步推理链也可建模为 DAG 并用类似方法验证
- 与 LangGraph 等工作的区别：VeriTrail 的节点是文本片段而非生成步骤，粒度更细

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 MGS 可追溯幻觉检测方法，DAG 建模思路优雅，但核心验证仍是"用LLM查LLM"
- 实验充分度: ⭐⭐⭐⭐ 两个新数据集 + 丰富的基线对比 + 消融 + 成本分析，但数据规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 框架定义严谨，算法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 解决了实际且重要的问题，对 MGS 流程的质量保障有直接工程价值

<!-- RELATED:START -->

## 相关论文

- [Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)
- [Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [Self-Destructive Language Model](self-destructive_language_model.md)
- [Lifelong Learning with Behavior Consolidation for Vehicle Routing](lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)

<!-- RELATED:END -->
