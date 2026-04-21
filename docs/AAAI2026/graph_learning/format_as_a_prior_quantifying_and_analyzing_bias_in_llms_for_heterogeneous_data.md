---
title: >-
  [论文解读] Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data
description: >-
  [AAAI 2026][图学习][格式偏差] 首次系统研究 LLM 在处理异构格式数据（文本/表格/信息框/知识图谱）时的格式偏差问题，通过三阶段实验揭示偏差的存在性、数据层面驱动因素和注意力机制层面的内部成因，并验证了注意力重平衡干预的有效性。
tags:
  - AAAI 2026
  - 图学习
  - 格式偏差
  - LLM
  - 异构数据
  - 知识图谱
  - 注意力机制
  - 信息丰富度
  - 结构质量
---

# Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data

**会议**: AAAI 2026  
**arXiv**: [2508.15793](https://arxiv.org/abs/2508.15793)  
**代码**: [github.com/NLPGM/Format-as-a-prior](https://github.com/NLPGM/Format-as-a-prior)  
**领域**: 图学习 / LLM 与异构数据  
**关键词**: 格式偏差, LLM, 异构数据, 知识图谱, 注意力机制, 信息丰富度, 结构质量  

## 一句话总结

首次系统研究 LLM 在处理异构格式数据（文本/表格/信息框/知识图谱）时的格式偏差问题，通过三阶段实验揭示偏差的存在性、数据层面驱动因素和注意力机制层面的内部成因，并验证了注意力重平衡干预的有效性。

## 研究背景与动机

LLM 在实际应用中需要处理来自多种格式的外部知识——非结构化文本、半结构化信息框（infobox）、结构化表格和知识图谱（KG）。**这些不同格式的知识互为补充**，能否公正整合它们对知识密集型应用至关重要。

然而，当 LLM 面对异构格式输入时，可能对某些格式有系统性偏好，导致：
- 忽略非偏好格式中的关键信息
- 推理错误和下游任务风险增大
- 例如：临床决策中过度依赖文本描述而忽略表格中的异常指标

先前研究已探讨 LLM 的多模态偏差、实体流行度偏差、事件时效性偏差等，但 **格式本身作为偏差来源从未被系统研究**。

核心三问：
1. 格式偏差是否系统性存在？
2. 哪些数据层面因素驱动偏差？
3. LLM 内部什么机制产生偏差？

## 方法详解

### 整体框架

三阶段递进式实验研究：
- **第一阶段**：偏差的存在性和方向性（10 个 LLM × 6 种格式对 = 60 组实验）
- **第二阶段**：数据层面因素分析（信息丰富度/结构质量/格式类型）
- **第三阶段**：注意力机制分析 + 轻量级干预实验

### 关键设计一：冲突场景构建与评估协议

基于 ConflictBank 采样 4000 条事实声明，每条配 3 个反事实，构建 12000 个矛盾对。将两方证据随机转换为不同格式（text/table/infobox/KG），使用 GPT-4o-mini 进行格式转换。

两大混淆因素控制：
- **内部知识过滤**：每条声明在零样本下测试 16 次，仅保留模型完全无法回答的样本
- **证据顺序随机化**：消除输入顺序偏差

两个核心度量：
- **DCR（双覆盖率）**= Both / (Pref-A + Pref-B + Both)——衡量偏差是否存在
- **FPR（格式偏好率）**= Pref-A / (Pref-A + Pref-B)——衡量偏差的方向

### 关键设计二：数据层面三因素分析

**因素 1：信息丰富度**
- 同构设定：同格式内比较 4/8/12 条目，LLM 一致偏好条目更多的变体
- 异构设定：结构化数据 vs 文本，随条目增加 LLM 对结构化格式偏好增强
- 结论：LLM 将"更多信息"等同于"更高证据价值"

**因素 2：结构质量**
- 对格式定义符号（括号、冒号、分隔符）引入受控损坏（概率 0.45/0.9）
- 同构设定：LLM 一致偏好完整格式，且偏好在中等损坏后即饱和
- 异构设定：损坏程度增加时 LLM 对结构化格式偏好急剧下降
- 结论：结构完整性充当"可信度信号"

**因素 3：格式类型**
- 固定内容（相同条目数），仅改变表现格式
- 结果偏好层次：Table > KG > Infobox（相对于 Text）

| 度量 | Infobox vs Text | Table vs Text | KG vs Text |
|------|-----------------|---------------|------------|
| FPR | 0.235 | 0.398 | 0.336 |

### 关键设计三：注意力机制分析与干预

分析 Qwen3-8B、Mistral-7B、LLaMA-3.1-8B 三个模型：

**注意力 vs 偏差存在**：注意力差距与 DCR 呈负相关（Spearman ρ = -0.31/-0.37/-0.54），注意力越不平衡，越难同时识别两方信息。

**注意力 vs 偏差方向**：82.35% 的单侧响应中，模型偏好注意力较少的那方——注意力多并不意味着被选为答案。

**注意力重平衡干预**：对两段证据的注意力总量进行归一化：

$$a'_j = \frac{\bar{m}}{m_k + \varepsilon} \cdot a_j, \quad j \in I_k$$

### 损失函数

本文为实证分析研究，不涉及训练损失函数的设计。干预方法在推理时直接修改注意力分布。

## 实验

### 主实验：格式偏差存在性（FPR 热力图）

| 格式对 (A vs B) | 偏好方向 | FPR 范围 | 统计显著 |
|-----------------|----------|----------|----------|
| Text vs Table | 偏好 Text | 0.55-0.75 | 多数 * |
| Text vs Infobox | 偏好 Text | 0.65-0.85 | 全部 * |
| Text vs KG | 偏好 Text | 0.45-0.65 | 部分 * |
| KG vs Table | 偏好 KG | 0.55-0.80 | 多数 * |
| KG vs Infobox | 偏好 KG | 0.65-0.85 | 全部 * |
| Table vs Infobox | 偏好 Table | 0.50-0.65 | 部分 * |

偏好层次：**Text ≈ KG > Table > Infobox**，在 10 个 LLM、6 个模型家族中一致。

### 消融/干预实验

| 条件 | DCR 变化 | FPR 变化 |
|------|----------|----------|
| 异构输入（基线） | 3-24% | 显著偏向 |
| 同构输入（text-text） | **28-53%** ↑↑ | 趋于 0.5 |
| 注意力重平衡（异构） | **显著提升** | **无显著变化** |

### 关键发现

1. **格式偏差系统性存在**：DCR 低至 3-24%（异构）vs 28-53%（同构），格式异构性本身即可独立损害多源信息整合
2. **偏差无随模型增大而改善的趋势**：Qwen3 系列中 8B→32B 的 DCR 无明显提升
3. **三因素均显著影响偏差**：信息丰富度越高越被偏好、结构越完整越被信任、格式类型存在内在偏好层级
4. **偏差存在性可干预、方向性难干预**：注意力重平衡有效提升 DCR 但不改变 FPR，方向性偏好可能根植于预训练
5. **下游任务影响**：同质→异质输入转换后 HotpotQA 准确率下降 9%、MuSiQue 下降 12%；注意力干预后分别提升 6.5%/9.5%

## 亮点

- **首次系统研究格式偏差**——概念新颖，填补了 LLM 偏差研究的重要空白
- 实验规模大且严谨：10 个 LLM × 6 模型家族 × 6 格式对 + 三阶段递进
- **混淆因素控制严格**：内部知识过滤（16 次零样本测试）+ 顺序随机化
- 从"存在 → 因素 → 机制 → 干预"的完整研究闭环
- 提出三个可操作的缓解方向：数据预处理、推理时干预、格式平衡训练

## 局限性

- 格式转换依赖 GPT-4o-mini，转换质量可能引入额外偏差
- 仅考虑 4 种格式（text/table/infobox/KG），未覆盖 JSON、XML、代码等格式
- 注意力干预仅在 8B 级别模型上验证，更大模型（70B+）的效果未知
- 冲突场景构造依赖 ConflictBank 中的事实性声明，泛化到开放域推理需进一步验证
- 方向性偏差的根因（预训练数据分布？tokenizer？）未做深入因果分析

## 相关工作

- **异构推理基准**：COMPMIX (Christmann et al. 2024) 要求跨格式推理；CompMix-IR 提供统一检索框架
- **LLM 知识冲突**：ConflictBank (Su et al. 2024) 评估 LLM 对冲突信息的处理；Jin et al. (2024) 研究参数知识与上下文证据的拉锯
- **多模态偏差**：Zhu et al. (2024) 研究视觉-语言模型中的跨模态知识冲突；Zhang et al. (2025) 评估和引导多模态偏好
- **偏差缓解**：冲突感知解码 (Yuan et al. 2024)、注意力剪枝 (Jin et al. 2024b)、神经元重加权 (Shi et al. 2024)

## 评分

⭐⭐⭐⭐ (4/5)

问题定义新颖且重要，实验设计系统全面，混淆因素控制严格。三阶段递进式分析逻辑清晰，从现象到因素到机制再到干预形成完整闭环。主要扣分在于注意力干预方法相对简单，以及缺乏对偏差方向性根因的深入分析。对图学习社区来说，本文揭示了 KG 等结构化知识在与 LLM 交互时可能被系统性低估，具有重要的实践警示意义。

<!-- RELATED:START -->

## 相关论文

- [Spiking Heterogeneous Graph Attention Networks](spiking_heterogeneous_graph_attention_networks.md)
- [Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing](assessing_llms_for_serendipity_discovery_in_knowledge_graphs_a_case_for_drug_rep.md)
- [NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)

<!-- RELATED:END -->
