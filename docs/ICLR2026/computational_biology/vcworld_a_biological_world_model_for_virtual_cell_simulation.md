---
title: >-
  [论文解读] VCWorld: A Biological World Model for Virtual Cell Simulation
description: >-
  [ICLR2026][计算生物][Virtual Cell] 提出 VCWorld，一个细胞级白盒模拟器，整合结构化生物知识图谱与大语言模型的迭代推理能力，以数据高效的方式模拟药物扰动引发的信号级联，生成可解释的逐步预测和显式机制假说，在药物扰动基准上达到 SOTA。
tags:
  - "ICLR2026"
  - "计算生物"
  - "Virtual Cell"
  - "world model"
  - "LLM Reasoning"
  - "Signaling Cascade"
  - "Drug Perturbation"
---

# VCWorld: A Biological World Model for Virtual Cell Simulation

**会议**: ICLR2026  
**arXiv**: [2512.00306](https://arxiv.org/abs/2512.00306)  
**代码**: 无  
**领域**: 计算生物
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

VCWorld 把虚拟细胞模拟当成一个**生物世界模型（Biological World Model）**：给它一个药物扰动和一张从公开数据库取来的结构化生物知识图谱，它不去拟合 $f_\theta(\text{perturbation}, \text{cell\_type}) \to \Delta \text{gene\_expression}$ 这样的黑盒映射，而是用 LLM 一步步"推演"扰动如何从靶点蛋白沿信号网络传播、最终改变基因表达。整条推理链是 $\text{Drug} \to \text{Target Protein} \to \text{Signaling Cascade} \to \text{Gene Expression Change}$，每一步都落地成一条可追溯、可被实验验证的机制路径，最终同时产出逐步的基因表达预测和显式的通路假说。

### 关键设计

**1. 结构化生物知识整合：让每一步推理都有生物学依据。**

数据驱动模型把几十年积累的通路知识全部压进参数里，预测时无从追溯；VCWorld 反过来把这些知识显式摆在推理现场。它从 KEGG、Reactome、STRING 等数据库抽取蛋白质-蛋白质相互作用（PPI）网络、信号通路拓扑和基因调控关系，再把这些关系整理成 LLM 能直接读的形式——文本化的通路描述或图谱表示，使模型在推断时可以随时查询"A 蛋白下游连着哪些通路"这类先验约束。这样每个预测步骤都锚定在一条已知的生物学关系上，而不是凭网络权重凭空给出一个数值，白盒特性正是建立在这一层显式知识之上。

**2. LLM 驱动的迭代信号级联推理：用文献先验补全知识缺口。**

知识图谱再全也有空白，纯靠图谱传播会在缺边处断链。VCWorld 让 LLM 充当推理引擎，逐步推断信号链条：药物与靶点蛋白结合，激活或抑制下游通路，改变转录因子活性，最终上调或下调特定基因。LLM 在海量生物医学文献上训练后隐含了大量分子间关系，VCWorld 借这种隐式知识"补"上图谱里缺失的交互、并评估每条候选路径是否合理。关键在于迭代而非单步——每一轮推理都把当前激活状态往下游推进一格，并显式吐出一条因果假说（例如"Drug X 抑制 Protein A → Protein A 无法磷酸化 Protein B → 通路 C 被阻断 → 基因 D 下调"），这串假说既是预测的中间产物，也直接给下游湿实验提供了可验证的线索。

**3. 数据高效的工作模式：核心能力来自知识而非数据规模。**

端到端方法要靠大规模匹配的（扰动条件, 基因表达变化）数据对才能学出映射，数据一稀缺就失灵。VCWorld 的推理能力主要来自结构化知识图谱加 LLM 先验，训练数据退居二线，只承担校准和验证的角色。正因为不把性能押在数据规模上，模型在罕见细胞类型、新型药物扰动这类数据稀缺场景里仍能给出有效预测——这也是知识驱动范式相对数据驱动范式最实际的优势。

### 一个完整示例

以一次药物扰动预测为例：输入是某靶向激酶抑制剂加上目标细胞类型，以及对应的知识图谱子图。VCWorld 先做靶点识别，从图谱中定位药物结合的靶点蛋白；随后进入迭代推理，第一格推断该靶点被抑制后无法磷酸化其下游底物，第二格沿 PPI 边推断相关信号通路因此被阻断，遇到图谱缺边时由 LLM 依据文献先验补上中间环节并判断合理性；逐格推进直到信号抵达转录调控层，推断出哪些转录因子活性改变、进而哪些基因被上调或下调。最终输出既包含逐基因的表达变化预测，也包含这一整条带因果标注的信号级联路径，研究者可以顺着每一格审查推理逻辑、定位可疑环节。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](../../ACL2026/computational_biology/aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](../../NeurIPS2025/computational_biology/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[ICCV 2025\] Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines](../../ICCV2025/computational_biology/integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)

</div>

<!-- RELATED:END -->
