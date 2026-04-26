---
title: >-
  [论文解读] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs
description: >-
  [AAAI 2026][幻觉缓解] 系统研究三种幻觉缓解方法（CoVe、DoLa、RAG）对LLM创造力的影响，发现它们对发散性创造力有截然相反的效果——CoVe增强、DoLa抑制、RAG无影响——而收敛性创造力基本不受影响，这一规律跨模型家族和参数规模一致成立。
tags:
  - AAAI 2026
  - 幻觉缓解
  - 创造力
  - LLM
  - 发散性思维
  - 收敛性思维
---

# Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs

**会议**: AAAI 2026  
**arXiv**: [2512.11509](https://arxiv.org/abs/2512.11509)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 幻觉缓解, 创造力, LLM, 发散性思维, 收敛性思维

## 一句话总结
系统研究三种幻觉缓解方法（CoVe、DoLa、RAG）对LLM创造力的影响，发现它们对发散性创造力有截然相反的效果——CoVe增强、DoLa抑制、RAG无影响——而收敛性创造力基本不受影响，这一规律跨模型家族和参数规模一致成立。

## 研究背景与动机

大语言模型在自然语言理解和推理上展现了卓越能力，但长期受幻觉问题困扰。学术界投入了大量精力开发幻觉缓解方法，尤其在AI辅助科学发现等高风险领域。然而，一个关键问题被完全忽视了：

**抑制幻觉是否会同时损害模型的创造力？**

这个问题之所以重要，是因为**创造力通常涉及建立非常规关联**。科学家有时通过将看似无关的概念联系起来生成有用假设——这种联想在模型中可能类似于被标记为"幻觉"的那种关联跳跃。按照心理学家Guilford（1950）的定义，创造力分为：

- **收敛性思维（Convergent thinking）**：在约束内正确解决问题
- **发散性思维（Divergent thinking）**：生成多样化的不同想法

如果幻觉缓解方法无差别地抑制了模型的"自由联想"能力，那么在追求事实准确性的同时，可能会牺牲科学发现所必需的创造性假设生成能力。这对于AI4Science领域有重大意义。

**本文的核心假设**：作者最初假设幻觉缓解方法会普遍抑制创造力，但实验结果揭示了更微妙的规律——不同方法对创造力有截然相反的影响。

## 方法详解

### 整体框架

本文采用实验驱动的研究框架（如图1所示）：

1. **选择三种代表性幻觉缓解方法**：CoVe、DoLa、RAG
2. **在两个创造力基准上评估**：NeoCoder（编程创造力）和CS4（故事创造力）
3. **跨模型家族和规模测试**：LLaMA（1B/8B/70B）、Qwen-Coder 7B、Mistral 7B
4. **分别衡量收敛性和发散性创造力**

### 关键设计

#### 1. **三种幻觉缓解方法**

**(a) CoVe（Chain of Verification）**：
- 多阶段推理的结构化方法，分四步：起草初始答案→生成验证问题→回答验证问题→生成精炼答案
- 通过让模型批判性评估自身输出来提高事实一致性
- 使用AutoGen多智能体框架实现

**(b) DoLa（Decoding by Contrasting Layers）**：
- 对比高层和低层预测的解码方法
- 使用Jensen-Shannon散度动态选择与最终层差异最大的早期层
- 从高层logits中减去低层logits，增强跨层学习到的token，抑制低层不可靠token

**(c) RAG（Retrieval-Augmented Generation）**：
- 检索增强生成，从外部知识库检索相关信息后再生成
- 使用ColBERTv2作为检索骨干，从CodeRAG-bench检索文档
- 仅用于NeoCoder基准（CS4缺乏检索语料库）

#### 2. **创造力评估基准**

**(a) NeoCoder**：
- 199个CodeForces编程问题，每题约30个人类解答
- 约束逐步增加（最多5个约束），模拟科学实验中的固定法则
- **收敛性**：在满足所有约束的同时正确解决问题的比例
- **发散性**：使用超出人类解答模式的新颖edit技术的比例

$$\text{NeoCoder-Divergent}(\text{LLM}, t) = \frac{1}{|\mathcal{Y}_t|}\sum_{y_i^t \in \mathcal{Y}_t} \frac{|\mathcal{T}_t^i \setminus \widehat{\mathcal{T}}^i|}{|\mathcal{T}_t^i|}$$

**(b) CS4**：
- 50条指令，每条扩展到39个累积约束，共250个唯一提示
- 评估维度：约束满足度、一致性、多样性（Dist-N）、QUC质量分

#### 3. **DoLa的深入研究：线性探针分析**

为了理解DoLa为何抑制创造力，使用受ITI（Inference-Time Intervention）启发的线性探针分析各层与创造力的相关性：
- 在每个注意力头上训练线性探针，预测模型是否会生成发散性创造输出
- 将头级相关性聚合到层级
- **发现**：早期层与创造力的相关性显著强于后期层（如图4所示）

基于此发现，提出**反转DoLa的创造力增强方法**：放大创造力相关层（前5层），抑制反相关层（后5层），成功提升发散性创造力而不损害收敛性创造力。

### 损失函数 / 训练策略

本文是纯实验研究，不涉及训练。评估使用百分比改进量化影响：

$$\text{Difference from Base}(\%) = \frac{M_{\text{method}} - M_{\text{baseline}}}{M_{\text{baseline}}} \times 100$$

所有生成独立运行三次取平均值。需要LLM-as-a-Judge时使用GPT-5-mini。

## 实验关键数据

### 主实验

**发散性创造力的百分比变化（相对baseline）**：

| 方法 | LLaMA-1B | LLaMA-8B | LLaMA-70B | Qwen-Coder-7B | Mistral-7B |
|------|---------|---------|----------|--------------|-----------|
| CoVe (NeoCoder) | +12.5% | +2~4% | +2~4% | +2~4% | -3%~+2% |
| DoLa (NeoCoder) | -2.5%~-1% | -1%~-0.5% | -1%~-0.5% | -1%~-0.5% | -2.5%~+3% |
| RAG (NeoCoder) | -3%~+1.5% | ~-5% | ~+3% | -0.5%~+1.5% | -2.5%~+2.5% |
| CoVe (CS4) | +5~8% | +5~8% | N/A | N/A | ~+2% |
| DoLa (CS4) | ~-8% | -3%~-2% | N/A | N/A | -3%~-2% |

**关键趋势**：
- CoVe：**一致增强**发散性创造力（大多数模型+设置）
- DoLa：**一致抑制**发散性创造力（所有模型+设置）
- RAG：**基本无影响**（模型间波动正负抵消）

### 消融实验

**层级创造力相关性分析**（图4）：

| 层级位置 | 与创造力的相关性 | 说明 |
|---------|--------------|------|
| 早期层（1-8层） | 高相关 | 创造力相关层主要集中在此 |
| 中间层（9-20层） | 中等相关 | 过渡区域 |
| 后期层（21-32层） | 低/反相关 | 反创造力层集中在此 |

**反转DoLa效果**（图5）：

| 配置 | 发散性创造力 | 收敛性创造力 | 说明 |
|------|-----------|-----------|------|
| Baseline | 基准 | 基准 | 无干预 |
| DoLa (原始) | 下降 | 不变 | 对比早期层→抑制创造力 |
| 反转DoLa（放大创造力层） | **上升** | **不变** | 发散性和收敛性创造力可解耦 |

### 关键发现

1. **CoVe增强发散性创造力**：验证过程促使模型更广泛地探索解空间，类似人类中"通过提问增强创造力"的认知机制
2. **DoLa抑制发散性创造力**：因为它对比的早期层恰好编码了更多探索性/发散性表征，减去它们等于压制了创造力
3. **RAG无显著影响**：可能是因为检索质量问题——CodeForces叙事场景与技术教程文档之间的语义不匹配
4. **收敛性创造力基本不受影响**：三种方法都未显著改变模型正确解决约束问题的能力
5. **这些趋势跨模型家族（LLaMA/Qwen/Mistral）和规模（1B-70B）一致成立**：说明创造力-幻觉关系是LLM的固有特征
6. **发散性和收敛性创造力可能是解耦的**：可以增强一个而不损害另一个

## 亮点与洞察

1. **研究问题极具洞察力**：幻觉与创造力的关系是一个被严重忽视但至关重要的问题，尤其对AI4Science
2. **实验设计严谨**：跨模型、跨规模、跨基准的全面评估，三次独立运行取平均
3. **线性探针分析**为理解LLM创造力的层级分布提供了新视角，发现早期层对创造力至关重要
4. **反转DoLa实验**巧妙地证明了发散性与收敛性创造力可以解耦，具有实际应用价值
5. **对AI4Science的实际指导意义明确**：选择CoVe而非DoLa来维持创造力

## 局限与展望

1. 编程和故事生成仅是科学假设生成的代理任务，未来需要开发专门的科学创造力评估框架
2. 未对CoVe为何增强创造力进行消融研究，具体机制不明
3. RAG的无效果可能源于检索质量差而非RAG本身的局限，未测试替代检索策略
4. 反转DoLa的创造力增强方法仅作为初步结果展示，缺乏深入分析和大规模验证
5. 将幻觉和创造力完全对立可能过于简化——两者之间可能存在更微妙的互作关系

## 相关工作与启发

- **CoVe** (Dhuliawala et al., 2023)：链式验证方法，意外地发现增强创造力
- **DoLa** (Chuang et al., 2024)：层对比解码，对事实性提升有效但损害创造力
- **ITI** (Li et al., 2024)：推理时干预，本文借鉴其线性探针方法分析层级特性
- **NeoCoder** (Lu et al., 2025)：约束驱动的编程创造力基准
- 启发：LLM的"创造力"和"事实性"可能分布在不同的网络层中，这为设计更精细的解码策略提供了线索；未来的幻觉缓解方法应考虑创造力保持

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（首次系统研究幻觉缓解对创造力的影响）
- 实验充分度: ⭐⭐⭐⭐（跨模型/规模/基准全面评估，但缺少深入消融）
- 写作质量: ⭐⭐⭐⭐⭐（逻辑清晰，实验组织出色）
- 价值: ⭐⭐⭐⭐（对AI4Science有重要指导意义，但应用范围可能有限）

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](../../ACL2025/information_retrieval/drag_distilling_rag_slm.md)
- [\[AAAI 2026\] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)
- [\[ACL 2025\] Removal of Hallucination on Hallucination: Debate-Augmented RAG](../../ACL2025/information_retrieval/removal_of_hallucination_on_hallucination_debate-augmented_rag.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)

<!-- RELATED:END -->
