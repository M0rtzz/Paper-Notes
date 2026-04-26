---
title: >-
  [论文解读] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling
description: >-
  [ACL 2026][医学图像][虚拟细胞建模] 提出 AROMA 框架，通过整合文本证据、知识图谱拓扑信息和蛋白质序列特征的多模态架构，结合两阶段训练策略（SFT + GRPO），实现了可解释且精确的基因扰动效应预测。
tags:
  - ACL 2026
  - 医学图像
  - 虚拟细胞建模
  - 基因扰动预测
  - 多模态融合
  - 知识图谱
  - 强化学习推理
---

# AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling

**会议**: ACL 2026  
**arXiv**: [2604.20263](https://arxiv.org/abs/2604.20263)  
**代码**: [github](https://github.com/blazerye/AROMA)  
**领域**: Medical Imaging / 生物信息学  
**关键词**: 虚拟细胞建模, 基因扰动预测, 多模态融合, 知识图谱, 强化学习推理

## 一句话总结

提出 AROMA 框架，通过整合文本证据、知识图谱拓扑信息和蛋白质序列特征的多模态架构，结合两阶段训练策略（SFT + GRPO），实现了可解释且精确的基因扰动效应预测。

## 研究背景与动机

**领域现状**：虚拟细胞建模旨在预测基因扰动后的分子状态变化，对生物机制研究至关重要。现有方法包括通用 LLM、领域微调语言模型、细胞基础模型以及检索增强方法。

**现有痛点**：(1) 通用 LLM 缺乏生物约束，自由形式推理不可靠；(2) 现有基础模型仅输出标签或差异表达分数，缺乏人类可解释的推理过程；(3) 检索增强方法的检索信号与调控拓扑弱对齐，未建模调控方向性和多步传播。

**核心矛盾**：基因扰动效应高度依赖上下文，且通过多步调控级联传播，单纯的文本相似度检索无法捕捉从扰动基因到靶标基因的机制性路径。

**本文目标**：构建一个既能准确预测又能提供可解释推理的基因扰动预测框架。

**切入角度**：将扰动预测锚定在结构化的、查询特定的生物学证据上，显式建模扰动基因与靶标基因之间的依赖关系。

**核心 idea**：结合知识图谱检索（提供拓扑结构证据）、图神经网络编码器（结构表示）、蛋白质序列编码器（分子表示），通过跨模态交互注意力机制建模扰动-靶标关系，再用两阶段训练优化预测与推理质量。

## 方法详解

### 整体框架

AROMA 包含三个阶段：(1) 数据阶段——构建 Gene-KG、Path-KG 和 PerturbReason 推理数据集；(2) 建模阶段——检索增强上下文化 + 多模态交互编码；(3) 训练阶段——多模态 SFT + GRPO 强化学习。

### 关键设计

1. **双知识图谱构建与检索增强上下文化**:

    - 功能：为扰动预测提供结构化生物学证据
    - 核心思路：Gene-KG 整合 STRING 和 CORUM 构建基因级关联网络（18k 节点、700k 边），Path-KG 整合 GO 和 Reactome 编码生物过程结构（80k 节点、400k 边）；检索包括基因功能描述、BFS 最短路径（≤3 条）和细胞系描述
    - 设计动机：两图互补——Gene-KG 提供基因间直接关联，Path-KG 提供更高层的通路结构，共同覆盖多层次证据

2. **多模态交互编码模块**:

    - 功能：显式建模扰动-靶标基因之间的跨模态交互
    - 核心思路：预训练两个 GAT 编码器分别编码 Gene-KG 和 Path-KG 子图，冻结 ESM-2 编码蛋白质序列；对每种模态使用跨注意力（扰动基因作 Query，靶标基因作 Key/Value），通过轻量投影器注入语言模型输入
    - 设计动机：不同于仅依赖文本的方法，结构和分子层面的表示能丰富扰动-靶标关系建模

3. **两阶段优化策略（SFT + GRPO）**:

    - 功能：先对齐多模态信息，再优化推理质量
    - 核心思路：第一阶段在 PerturbReason（498k+ 样本）上做多模态 SFT，冻结 GNN 和 ESM-2，LoRA 微调 LLM；第二阶段用 GRPO 强化学习，采样多条推理轨迹，奖励正确预测（+5.0）和格式规范（+0.5）
    - 设计动机：SFT 注入领域知识，GRPO 通过任务级反馈进一步优化推理过程的准确性和一致性

### 损失函数 / 训练策略

SFT 阶段使用标准自回归语言建模损失。GRPO 阶段为每个实例采样多条推理轨迹，奖励函数考虑：预测正确性（5.0/-1.0）、推理格式规范性（+0.5）、答案类别唯一性（+0.5），组内归一化计算优势值。

## 实验关键数据

### 主实验

| 方法 | K562 Avg | HepG2 Avg | Jurkat Avg | RPE1 Avg | 总平均 F1 |
|------|---------|---------|---------|---------|----------|
| DeepSeek-R1 | 0.32 | 0.34 | 0.33 | 0.31 | 0.33 |
| SUMMER | 0.58 | 0.67 | 0.65 | 0.67 | 0.64 |
| GAT | 0.59 | 0.67 | 0.63 | 0.65 | 0.64 |
| AROMA | **0.66** | **0.76** | **0.75** | **0.77** | **0.73** |

### 消融实验

| 配置 | 平均 F1 | 说明 |
|------|---------|------|
| 原始 Qwen3-8B | 0.26 | 缺乏领域知识 |
| + SFT | 0.65 | 领域知识注入关键 |
| + SFT + GRPO | 0.68 | 强化学习提升推理 |
| + RAG | 0.71 | 检索证据补充 |
| 全模块（AROMA） | 0.73 | 各组件协同增益 |

### 关键发现
- AROMA 在所有 4 个细胞系上一致超越所有基线方法，平均 F1 达 0.73，比最强基线 SUMMER 高出 9 个百分点
- 零样本泛化（RPE1）性能仅轻微下降（0.77 → 0.73），展示了强跨分布泛化能力
- 在低流行度和低连通性基因上的性能下降远小于去除检索和多模态模块的变体，说明增益来自联合建模而非记忆高频基因
- GRPO 采样轨迹数从 4 增加到 16 时性能稳步提升

## 亮点与洞察
- 首次在基因扰动预测中系统性地整合知识图谱拓扑、蛋白质序列和文本证据三种模态
- 双知识图谱的设计思路值得借鉴：Gene-KG 提供局部关联，Path-KG 提供全局通路结构
- 跨注意力机制以扰动基因为 Query、靶标基因为 Key/Value 的设计直觉清晰
- 构建的 PerturbReason 数据集（498k 样本）是重要的社区资源贡献

## 局限与展望
- 目前仅支持单基因扰动，无法处理多基因组合扰动或化学干预
- 每次推理仅预测单个靶标基因的表达变化，未扩展到同时预测多个下游基因
- 对知识图谱和外部文本资源的依赖意味着对缺乏注释的基因预测可能退化
- 未来可扩展到组合扰动和化学干预场景

## 相关工作与启发
- **vs SUMMER**: SUMMER 使用文本相似度检索，AROMA 进一步引入拓扑结构和蛋白质序列建模扰动-靶标交互
- **vs GEARS**: GEARS 注入图结构先验但缺乏可解释推理，AROMA 同时提供预测和推理路径
- **vs SynthPert/rBio-1**: 它们依赖合成推理轨迹训练，可能继承监督噪声；AROMA 通过 GRPO 直接从任务反馈优化

## 评分
- 新颖性: ⭐⭐⭐⭐ 多模态融合思路清晰，双知识图谱和交互编码设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多细胞系、零样本、消融、鲁棒性分析全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示精美
- 价值: ⭐⭐⭐⭐ 对虚拟细胞建模领域有重要推动，资源贡献价值高

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_imaging/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[NeurIPS 2025\] MuSLR: Multimodal Symbolic Logical Reasoning](../../NeurIPS2025/medical_imaging/muslr_multimodal_symbolic_logical_reasoning.md)
- [\[NeurIPS 2025\] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](../../NeurIPS2025/medical_imaging/multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)

<!-- RELATED:END -->
