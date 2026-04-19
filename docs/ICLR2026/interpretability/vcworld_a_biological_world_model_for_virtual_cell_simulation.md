---
title: >-
  [论文解读] VCWorld: A Biological World Model for Virtual Cell Simulation
description: >-
  [ICLR2026][Virtual Cell] 提出 VCWorld，一个细胞级白盒模拟器，整合结构化生物知识图谱与大语言模型的迭代推理能力，以数据高效的方式模拟药物扰动引发的信号级联，生成可解释的逐步预测和显式机制假说，在药物扰动基准上达到 SOTA。
tags:
  - ICLR2026
  - Virtual Cell
  - world model
  - LLM Reasoning
  - Signaling Cascade
  - Drug Perturbation
---

# VCWorld: A Biological World Model for Virtual Cell Simulation

**会议**: ICLR2026  
**arXiv**: [2512.00306](https://arxiv.org/abs/2512.00306)  
**代码**: 无  
**领域**: 可解释性 / AI for Science  
**关键词**: Virtual Cell, world model, LLM Reasoning, Signaling Cascade, Drug Perturbation

## 一句话总结

提出 VCWorld，一个细胞级白盒模拟器，整合结构化生物知识图谱与大语言模型的迭代推理能力，以数据高效的方式模拟药物扰动引发的信号级联，生成可解释的逐步预测和显式机制假说，在药物扰动基准上达到 SOTA。

## 研究背景与动机

**领域现状**：虚拟细胞建模（Virtual Cell Modeling）是计算生物学的前沿方向，目标是预测细胞在各种扰动（药物处理、基因敲除等）下的响应。这对药物发现、疾病机制理解和精准医疗至关重要。近年来，以 scGPT、GEARS 为代表的深度学习模型通过大规模单细胞 RNA-seq 数据学习基因表达与扰动之间的映射关系，取得了一定进展。

**现有痛点**：（1）**数据依赖过重**——现有模型严重依赖大规模高质量单细胞数据集，但此类数据采集成本高、覆盖范围有限；（2）**泛化能力受限**——数据质量、覆盖范围和批次效应（batch effects）三重因素制约模型在新细胞类型、新扰动条件下的泛化性能；（3）**黑盒问题**——端到端训练的模型只输出基因表达预测值，无法提供扰动如何在细胞内传播的机制解释。

**核心矛盾**：科学研究对可解释性和机制一致性的需求与深度学习模型"黑盒"特性之间的根本冲突。缺乏机制解释的预测结果在科学研究中难以获得认可，无法真正推动生物学认知。即使预测数值准确，研究者也无法从中提取可验证的生物学假说。

**本文方案**：VCWorld 跳出"数据驱动端到端拟合"的范式，转而将结构化生物学知识（如蛋白质相互作用网络、信号通路图谱）与 LLM 在生物医学文献上训练获得的先验知识相结合。模型不再学习 $\text{扰动} \to \text{基因表达}$ 的黑盒映射，而是显式模拟扰动从靶点蛋白到下游基因表达的信号级联传播过程，每一步推理都产生可追溯的机制路径。

## 方法详解

### 整体框架

VCWorld 将虚拟细胞模拟建模为一个**生物世界模型（Biological World Model）**。核心流程：

1. **输入**：药物/扰动信息 + 从公开数据库提取的结构化生物知识图谱
2. **推理**：LLM 驱动的迭代推理，模拟扰动在细胞信号网络中的级联传播
3. **输出**：逐步的可解释基因表达预测 + 显式的信号通路机制假说

与传统方法 $f_\theta(\text{perturbation}, \text{cell\_type}) \to \Delta \text{gene\_expression}$ 的端到端范式形成鲜明对比，VCWorld 的推理链条为：

$$\text{Drug} \xrightarrow{\text{靶点识别}} \text{Target Protein} \xrightarrow{\text{通路传播}} \text{Signaling Cascade} \xrightarrow{\text{转录调控}} \text{Gene Expression Change}$$

### 关键设计一：结构化生物知识整合

VCWorld 的"白盒"特性建立在结构化知识基础之上：

- **知识来源**：从 KEGG、Reactome、STRING 等公开数据库提取蛋白质-蛋白质相互作用（PPI）网络、信号通路拓扑、基因调控关系等多层次生物学知识
- **知识表示**：将这些关系结构化为 LLM 可处理的格式（文本化的通路描述或图谱表示），使 LLM 能够在推理时查询和利用这些先验约束
- **设计动机**：不是简单地用数据训练端到端模型，而是将数十年积累的生物学知识显式注入推理过程，使得每个预测步骤都有明确的生物学依据

### 关键设计二：LLM 驱动的迭代信号级联推理

VCWorld 的核心推理引擎利用 LLM 模拟扰动在细胞内的传播：

- **迭代推理过程**：模型逐步推断信号传导链条——药物与靶点蛋白结合 → 激活/抑制下游信号通路 → 影响转录因子活性 → 导致特定基因表达上调/下调
- **LLM 作为推理引擎**：LLM 在海量生物医学文献上训练后隐含了丰富的分子生物学知识，VCWorld 利用这种隐式知识来"补全"知识图谱中缺失的交互关系，并评估每条信号传导路径的合理性
- **可追溯的机制路径**：每一步推理都产生明确的因果假说（如"Drug X 抑制 Protein A → Protein A 无法磷酸化 Protein B → 通路 C 被阻断 → 基因 D 下调"），为下游实验验证提供直接线索

### 关键设计三：数据高效的工作模式

VCWorld 的知识驱动范式使其对训练数据的需求大幅降低：

- 传统方法需要大规模匹配的（扰动条件, 基因表达变化）数据对来训练端到端映射
- VCWorld 的核心推理能力来自结构化知识图谱 + LLM 先验知识，训练数据主要用于校准和验证
- 这一特性使 VCWorld 在数据稀缺的场景（罕见细胞类型、新型药物扰动）中仍能保持有效预测

## 实验关键数据

### 药物扰动基准测试

| 方法 | 类型 | 核心特点 | 预测精度 |
|------|------|---------|---------|
| scGPT | 数据驱动 | 大规模预训练+微调 | 基线水平 |
| GEARS | 数据驱动 | 图神经网络建模基因关系 | 中等 |
| 多源信息融合方法 | 数据驱动 | 整合多组学数据 | 改善有限 |
| **VCWorld (本文)** | **知识+LLM 推理** | **白盒，可解释** | **SOTA** |

VCWorld 在药物扰动预测基准上达到最先进性能，同时是唯一提供完整机制解释的方法。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 去除结构化知识 | 性能显著下降 | 仅靠 LLM 内部知识推理不够可靠 |
| 去除迭代推理 | 性能下降 | 单步预测丢失信号级联的逐步传播信息 |
| 去除 LLM 推理 | 性能大幅下降 | 纯知识图谱无法处理知识缺口 |
| 完整 VCWorld | **最优** | 结构化知识 + LLM 推理的协同效应 |

### 关键发现

1. **机制一致性**：VCWorld 推断的信号传导路径与已发表的生物学文献证据高度一致，验证了推理过程的生物学合理性
2. **可解释性优势**：每个预测附带完整的信号级联路径，研究者可逐步审查推理逻辑并定位潜在错误
3. **数据效率**：在有限训练数据下的表现优于依赖大规模数据集的数据驱动基线方法

## 亮点与洞察

- **白盒模拟器的理念**突破了 AI for Science 中"预测精度至上"的现状——在科学研究中，一个能给出合理机制解释的中等精度预测往往比一个无法解释的高精度预测更有价值
- **LLM 作为"生物推理引擎"**是一个巧妙的设计——LLM 在 PubMed 等海量生物医学文献上训练后，隐式编码了大量分子间关系和生物学原理，VCWorld 将这种隐式知识转化为显式的推理能力
- **"世界模型"视角**将细胞响应预测从统计拟合上升为因果模拟——给定初始扰动条件，模型能够"预演"细胞的动态响应过程
- **跨领域方法论启发**：将 LLM 推理能力与领域知识图谱结合的范式可推广到材料科学、化学反应预测等其他科学领域

## 局限与展望

- **LLM 幻觉风险**：LLM 可能生成看似合理但生物学上错误的推理链条，需要额外的校验机制来过滤不可靠推断
- **知识图谱覆盖不完整**：KEGG/Reactome 等数据库仍有大量未知的信号传导关系，在知识空白区域模型performance会下降
- **推理效率**：迭代调用 LLM 进行逐步推理的计算成本显著高于端到端前向传播
- **扰动类型覆盖**：当前主要聚焦药物扰动验证，基因敲除（gene knockout）、过表达（overexpression）等其他扰动类型的泛化效果有待验证
- **单细胞层面异质性**：同一细胞类型内部存在显著的细胞间异质性，当前框架对此建模有限

## 相关工作与启发

- **vs scGPT / GEARS**：端到端数据驱动方法，预测精度依赖数据规模，无法提供机制解释；VCWorld 以知识+推理换取可解释性和数据效率
- **vs Virtual Cell Initiative (CZI)**：Chan Zuckerberg Initiative 推动的虚拟细胞研究项目，VCWorld 从"白盒世界模型"角度提供了一种 complementary 的技术路线
- **vs GeneGPT / BioGPT**：LLM 在生物学中的早期应用侧重知识问答，VCWorld 进一步将 LLM 用于结构化的因果推理和动态模拟
- **启发**：LLM 推理 + 领域知识图谱的"白盒世界模型"范式有望在其他知识密集型科学领域（如药物化学、材料设计）中复现

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 白盒生物世界模型概念新颖，LLM 推理与知识图谱结合的思路在虚拟细胞领域属首创
- 实验充分度: ⭐⭐⭐⭐ 药物扰动基准全面，机制验证有说服力
- 写作质量: ⭐⭐⭐⭐ 概念清晰，跨领域读者友好
- 价值: ⭐⭐⭐⭐⭐ 对 AI for Science 和可解释 AI 均有重要方向性启发

<!-- RELATED:START -->

## 相关论文

- [scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](../../NeurIPS2025/interpretability/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [NIMO: a Nonlinear Interpretable MOdel](nimo_a_nonlinear_interpretable_model.md)
- [Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)
- [Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)

<!-- RELATED:END -->
