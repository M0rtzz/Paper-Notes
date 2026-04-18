---
title: >-
  [论文解读] Deliberation on Priors: Trustworthy Reasoning of LLMs on Knowledge Graphs
description: >-
  [NeurIPS 2025][图学习][KGQA] 提出 Deliberation over Priors（DP）框架，通过渐进式知识蒸馏（SFT + KTO 偏好优化）提升关系路径生成的忠实度，并通过约束引导的内省-回溯机制保障推理可靠性，在 ComplexWebQuestions 上 H@1 提升 16.5%，且 LLM 调用次数仅为 2.9 次（ToG 需 22.6 次）。
tags:
  - NeurIPS 2025
  - 图学习
  - KGQA
  - 约束推理
  - 知识蒸馏
  - KTO偏好优化
  - 内省回溯
---

# Deliberation on Priors: Trustworthy Reasoning of LLMs on Knowledge Graphs

**会议**: NeurIPS 2025  
**arXiv**: [2505.15210](https://arxiv.org/abs/2505.15210)  
**代码**: https://github.com/mira-ai-lab/Deliberation-on-Priors  
**领域**: 图学习 / 知识图谱问答  
**关键词**: KGQA, 约束推理, 知识蒸馏, KTO偏好优化, 内省回溯

## 一句话总结

提出 Deliberation over Priors（DP）框架，通过渐进式知识蒸馏（SFT + KTO 偏好优化）提升关系路径生成的忠实度，并通过约束引导的内省-回溯机制保障推理可靠性，在 ComplexWebQuestions 上 H@1 提升 16.5%，且 LLM 调用次数仅为 2.9 次（ToG 需 22.6 次）。

## 研究背景与动机

**领域现状**：知识图谱增强的检索增强生成（KG-RAG）旨在用外部知识缓解 LLM 幻觉。现有方法分两类：端到端检索（按语义相似度取 top-K 三元组）和逐步推理（ToG、PoG、DoG 等将问题分解为子问题依次检索）。

**现有痛点**：
   - 未充分利用 KG 的 **结构先验**：关系路径蕴含的多跳结构模式被忽视，LLM 只做表面语义匹配
   - 未利用 KG 的 **约束先验**：问题中隐含的类型约束、多实体约束、时间约束等未被显式提取和验证
   - **LLM 调用次数过多**：ToG 平均每个问题需 22.6 次 LLM 调用，PoG 需 13.3 次，部署成本高

**核心矛盾**：如何在保证推理忠实度（路径结构正确）和可靠性（约束满足）的同时，最小化 LLM 交互次数？

**本文目标**：设计框架同时优化路径生成质量和约束验证能力，并大幅降低推理成本。

**切入角度**：(1) 用 Dijkstra 最短路径提取多路径弱监督 + KTO 偏好优化训练路径生成器→忠实度；(2) 从问题中提取 5 种预定义约束 + 内省回溯→可靠性。

**核心 idea**：多路径弱监督蒸馏 + KTO 偏好优化训练忠实的路径生成器，约束引导的内省-回溯保证路径正确性，实现高效可信赖的 KG 推理。

## 方法详解

### 整体框架

DP 由四个模块组成，分离线和在线两个阶段：

- **离线阶段**：Distillation 模块——用 SFT + KTO 渐进蒸馏训练路径生成器
- **在线阶段**：Planning（生成候选关系路径）→ Instantiation（在 KG 中实例化）→ Introspection（约束验证与回溯）

### 关键设计

#### 1. 渐进式知识蒸馏（Distillation）

**目标**：训练 LLM 生成忠实的关系路径（relation path）。

**弱监督信号构造**：给定问题 q 和主题实体 e_s，在 KG 中提取 k-hop 子图，用 Dijkstra 算法找从 e_s 到答案实体 e_t 的 **所有最短关系路径**，构成一对多（1:n）映射。相比 RoG 的一对一映射，1:n 在路径生成 F1 上提升 17.4%（WebQSP）和 21.3%（CWQ）。

**SFT 阶段**：在弱监督的 question→relation path 映射上微调 LLaMA3.1-8B-Instruct（LoRA, rank=16, alpha=32），训练 2 个 epoch，最大化条件对数似然。

**KTO 偏好优化阶段**：构造偏好数据——正例为弱监督路径，负例通过三种扰动合成：
- Path Truncation：删除末跳关系，生成不完整路径
- Entity-Path Swapping：在同一问题的不同主题实体间交换路径
- Relation Deletion：随机删除某个主题实体的关系路径

负正比为 3:1，正因为这种严重不平衡，选择 KTO 而非 DPO。KTO 基于 Kahneman-Tversky 前景理论的效用函数，不需要成对偏好数据，对类别不平衡更鲁棒，训练 1 个 epoch。

#### 2. 规划与实例化（Planning & Instantiation）

**规划**：经蒸馏训练的路径生成器以 zero-shot 方式为每个主题实体生成多个候选关系路径。多实体问题的候选路径合并为统一候选池，增加语义多样性。LLM 按语义相关性从候选池中选择最佳关系路径。

**实例化**：将抽象关系路径在 KG 中具体化。例如 {directed, won_award} 实例化为 "Ang Lee →directed→ Crouching Tiger →won_award→ Compound Value Type 2"。一条关系路径可能有多个实例化结果。

#### 3. 约束引导的内省-回溯（Introspection）

**约束定义**：预定义 5 种约束类型，源自 Bao et al. (COLING 2016)：

| 约束类型 | 说明 | 示例 |
|---------|------|------|
| Type | 答案需满足特定类型 | "What **city** did Esther live in?" |
| Multi-Entity | 答案需同时满足多实体条件 | "Which team owned by **Malcolm Glazer** has **Tim Howard** playing for it?" |
| Explicit Time | 明确指定时间范围 | "Who was the governor of Arizona **in 2009**..." |
| Implicit Time | 隐含时间约束 | "...when **Andrew Jackson was the president**" |
| Ordinal | 包含排序规则 | "...that **last** won the World Series when?" |

统计显示三个数据集中 Type 约束最常见，大多数问题包含至少一种约束。

**内省流程**：LLM 从问题中提取约束 → 验证实例化路径是否满足约束 → 满足则生成回答 → 不满足则反馈具体违反信息并回溯选择替代路径 → 迭代直到满足或候选耗尽。

### 训练细节

- 基座模型：LLaMA3.1-8B-Instruct
- SFT：Adam, lr=5e-5, warmup ratio 0.1, 2 epochs
- KTO：1 epoch, 效用感知偏好损失
- LoRA：rank=16, alpha=32
- 在线推理 LLM：可灵活替换（LLaMA3.1-8B / ChatGPT / GPT-4 / GPT-4o / GPT-4.1）

## 实验关键数据

### 主实验（三基准 SOTA）

| 数据集 | 指标 | DP (LLaMA3.1-8B) | DP (GPT-4.1) | 先前最优 |
|--------|------|-------------------|--------------|----------|
| WebQSP | H@1 | 82.8% | 86.7% | 83.8% (LightPROF) |
| WebQSP | F1 | 75.7% | 80.1% | 73.5% (GNN-RAG) |
| CWQ | H@1 | 61.1% | **75.8%** | 62.8% (GNN-RAG) |
| CWQ | F1 | 58.5% | **69.4%** | 60.4% (GNN-RAG) |
| MetaQA | H@1 | 87.4% | 95.5% | 99.2% (TransferNet) |

CWQ 上 H@1 比 LightPROF 提升 16.5%。DP 与 GPT-4.1 集成在 CWQ 上达到最佳。

### 效率对比

| 方法 | LLM 调用次数 (CWQ) | 总 Token (CWQ) |
|------|-------------------|----------------|
| ToG | 22.6 | 9,669 |
| PoG | 13.3 | 8,156 |
| DoG | 12.7 | 8,683 |
| **DP** | **2.9** | **3,115** |

DP 仅需 2.9 次 LLM 调用，相比 ToG 减少 87%，总 token 减少 68%。

### 消融实验（GPT-4.1 为在线 LLM）

| 配置 | WebQSP H@1 | CWQ H@1 |
|------|-----------|---------|
| 完整 DP | 86.7% | 75.8% |
| w/o KTO | 84.7% (−2.0) | 74.6% (−1.2) |
| w/o 内省回溯 | 82.0% **(−4.7)** | 70.8% **(−5.0)** |
| w/o 约束预定义(用LLM自动生成) | 83.4% (−3.3) | 74.4% (−1.4) |
| w/o 违反反馈 | 83.0% (−3.7) | 73.2% (−2.6) |
| Vanilla GPT-4.1 (无KG增强) | 71.0% (−15.7) | 53.0% (−22.8) |

**内省回溯是最关键组件**：移除后 H@1 下降最大（4.7-5.0%）。

### 先验知识影响

| 路径映射方式 | WebQSP F1 | CWQ F1 |
|------------|-----------|--------|
| 1:1 (单路径) | 59.3% | 49.8% |
| **1:n (多路径)** | **76.7%** | **71.1%** |

多路径弱监督比单路径 SFT 在 F1 上分别提升 +17.4% 和 +21.3%。

### 回溯分析

GPT-4.1 在 CWQ 上平均回溯 0.42 次/问题，GPT-3.5 仅 0.10 次——更强的指令跟随能力导致更严格的约束检查和更多回溯，这与其更好的最终性能一致。

## 亮点与洞察

- **结构先验的充分利用**：多路径弱监督 + 约束验证充分挖掘了 KG 的结构信息，将 KG 从被动的知识库升格为主动的推理引导
- **KTO 处理不平衡偏好数据**：3:1 负正比下 KTO 比 DPO 更适合，因为 KTO 不需要成对偏好，基于前景理论的效用函数天然适配不平衡场景
- **效率-质量双赢**：路径级抽象（先生成关系路径再实例化）大幅缩减搜索空间，仅需 2.9 次 LLM 调用就超越需 22.6 次的 ToG
- **约束的双重作用**：约束既用于筛选候选路径（内省），又用于指导回溯（反馈），形成闭环推理验证

## 局限与展望

- 5 种预定义约束依赖人工设计，迁移到垂直领域需重新定义
- 评估在 500 个子采样测试集上进行，完整测试集结果未报告
- 仅在 Freebase 子集上验证，DBpedia/Wikidata 等其他 KG 的兼容性未测试
- MetaQA 上性能低于一些监督学习方法（TransferNet 99.2% vs DP 95.5%），在简单 KG 上优势不明显

## 相关工作与启发

- **vs ToG/PoG/DoG**：DP 的路径级抽象比逐步实体搜索高效得多，将推理决策上移到关系路径层面
- **vs RoG**：RoG 也训练路径生成器但用单路径监督且无约束验证，DP 的多路径+内省是关键改进
- **vs GNN-RAG**：GNN-RAG 用 GNN 做检索但 F1 低于 DP，说明 LLM 层面的结构感知更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 多路径蒸馏 + KTO 偏好优化 + 约束内省的组合新颖，效率提升显著
- 实验充分度: ⭐⭐⭐⭐ 三基准 + 五种 LLM 集成 + 详细消融 + 效率与回溯分析
- 写作质量: ⭐⭐⭐⭐ 框架清晰，四模块分工明确，消融逻辑严密
- 价值: ⭐⭐⭐⭐ 推理效率 87% 提升对实际部署有重大意义，约束引导的内省机制可泛化

## 一句话总结
