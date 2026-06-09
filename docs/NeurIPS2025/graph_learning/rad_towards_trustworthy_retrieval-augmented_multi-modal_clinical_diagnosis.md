---
title: >-
  [论文解读] RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis
description: >-
  [NeurIPS 2025][图学习][检索增强诊断] 提出检索增强诊断框架RAD，通过从多源医学语料中检索疾病指南并注入多模态模型的特征提取和跨模态融合全流程，同时引入双轴可解释性评估体系，在四个不同解剖部位的数据集上达到SOTA。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "检索增强诊断"
  - "多模态融合"
  - "知识注入"
  - "可解释性"
  - "临床指南"
---

# RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis

**会议**: NeurIPS 2025  
**arXiv**: [2509.19980](https://arxiv.org/abs/2509.19980)  
**代码**: 暂无  
**领域**: 医学图像  
**关键词**: 检索增强诊断, 多模态融合, 知识注入, 可解释性, 临床指南

## 一句话总结

提出检索增强诊断框架RAD，通过从多源医学语料中检索疾病指南并注入多模态模型的特征提取和跨模态融合全流程，同时引入双轴可解释性评估体系，在四个不同解剖部位的数据集上达到SOTA。

## 研究背景与动机

当前AI驱动的医学研究主要通过知识图谱或大规模文本预训练将医学知识编码到模型参数中，但这些方法存在根本性局限：**知识是隐式编码的，缺少对下游特定任务所需的细粒度知识的显式注入**。

具体痛点包括：

**预训练阶段注入的局限**：PubMedBERT、KAD等在预训练时注入知识，但这些知识是"通用"的，无法灵活适配具体诊断任务。例如KAD在胸部X光预训练数据上表现好，但换到眼科/皮肤科就效果下降

**黑盒决策的不透明性**：临床诊断必须遵循循证原则（evidence-based），依赖标准化的诊断标准。黑盒神经网络的模糊决策机制阻碍了在临床环境中的部署

**缺乏可解释性量化评估**：现有多模态诊断模型缺少对可解释性的定量评估手段

本文的核心insight是：**有效的知识集成需要以任务为中心，在整个诊断流程（输入增强→特征提取→模态融合→决策）中全方位地与疾病级知识进行对齐，而不仅仅是在某一个环节**。

## 方法详解

### 整体框架

RAD包含三个协同组件：(1) 多源指南检索与精炼；(2) 指南增强的特征约束；(3) 双解码诊断网络。三者从输入层、特征层到决策层系统地注入外部疾病知识。

### 关键设计

1. **指南检索与精炼 (Guideline Retrieval & Refinement)**

   从"Wiki"、"Research"（PubMed）、"Guideline"（45K条临床实践指南）、"Book"（医学教材）四类源检索知识。

   对数据集中的 $m$ 种疾病，使用MedCPT（双编码器检索模型）计算疾病名称与语料的相似度：
    $\mathcal{C}_i = \underset{p_j \in P}{\text{Top-}k} \text{Sim}(e_i, p_j)$

   检索到的文档可能冗余或含噪声，使用Qwen2.5-72B进行自动摘要精炼：
    $g_i = \text{LLM}([\text{Prompt}, c_{i,1}, \cdots, c_{i,k}])$

   产出标准化、结构化的诊断指南，包含关联症状、影像特征、关键检查项等，并经人工验证。

2. **指南增强的对比损失 (GECL)**

   将视觉编码器输出 $\mathbf{V}_i$ 和文本编码器输出 $\mathbf{T}_i$ 与疾病指南原型 $\mathbf{G}'$ 在潜空间中对齐。

   对样本 $i$，将指南特征分为正集 $\mathbf{P}_i$（对应正标签的指南）和负集 $\mathbf{N}_i$，通过负采样获得子集 $\mathbf{Q}_i$，最终GECL损失为：
    $\mathcal{L}_{\text{GECL}} = \frac{1}{N}\sum_{i=1}^{N} \left(\mathcal{L}_{\text{SupCon}}(\mathbf{T}'_i, \mathbf{S}_i) + \alpha \mathcal{L}_{\text{SupCon}}(\mathbf{V}'_i, \mathbf{S}_i)\right) \cdot \mathbb{I}[|\mathbf{P}_i|>0]$

   设计动机：将样本特征动态拉向正类指南原型、推离负类原型，引导模型选择性关注与指南匹配的临床相关特征，同时提升性能和可解释性。

3. **双解码诊断网络 (Dual Diagnostic Network)**

   两个对称的Transformer解码器并行工作：
    - **指南分支**：以指南 $g$ 的编码为query，拼接的模态特征 $\mathbf{V}_i \oplus \mathbf{T}_i$ 为key/value，输出 $\hat{y}_i^{\text{guide}}$
    - **标签分支**：以疾病名称 $E$ 的编码为query，拼接的模态特征为key/value，输出 $\hat{y}_i^{\text{label}}$

   总训练损失：
    $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}^{\text{guide}}, y) + \mathcal{L}_{\text{BCE}}(\hat{y}^{\text{label}}, y) + \beta \mathcal{L}_{\text{GECL}}$

### 可解释性评估体系

- **文本指标 — Guideline Recall**：衡量模型对指南中提及的关键实验室指标的注意力召回率
- **视觉指标 — Visual Grounding IoU**：注意力图与专家标注病变区域的IoU重叠度

## 实验关键数据

### 主实验

| 数据集 | 方法 | F1 | AUC | mAP | Avg |
|-------|------|-----|-----|-----|-----|
| MIMIC-ICD53 (胸部) | RAD | **39.71** | **93.00** | **36.74** | **57.28** |
| | KAD | 36.32 | 91.95 | 33.54 | 54.19 |
| | HEALNet | 35.42 | 88.80 | 31.97 | 53.13 |
| FairVLMed (眼部) | RAD | **84.30** | **91.32** | **91.88** | **86.63** |
| | HEALNet | 81.80 | 89.60 | 90.45 | 84.39 |
| SkinCAP (皮肤) | RAD | **85.48** | **97.97** | **83.55** | **88.64** |
| | KAD | 82.06 | 97.80 | 80.40 | 86.15 |
| NACC (脑部) | RAD | **37.65** | **87.11** | **30.03** | **58.12** |
| | HEALNet | 35.91 | 85.04 | 26.13 | 55.67 |

### 消融实验

| 配置 | F1 | AUC | Avg | 说明 |
|-----|-----|-----|-----|------|
| 无GECL + 无Dual Decoder | 34.91 | 91.27 | 53.35 | 基线 |
| +GECL_vision | 37.43 | 92.53 | 54.79 | 视觉对齐有效 |
| +GECL_text | 37.75 | 92.91 | 55.52 | 文本对齐更强 |
| +GECL_both | 39.34 | 92.94 | 56.26 | 双模态对齐叠加 |
| +Dual Decoder only | 39.22 | 92.25 | 55.91 | 解码器贡献大 |
| **RAD (完整)** | **39.71** | **93.00** | **57.28** | 各组件互补 |

### 可解释性评估

| 指标 | w/o RAD | RAD | 提升 |
|-----|---------|-----|------|
| Guideline Recall (总体) | 24.76% | **65.62%** | +40.86% |
| Visual Grounding mIoU (Avg-D) | 15.98 | **19.72** | +3.74 |
| Visual Grounding mIoU (Avg-P) | 17.78 | **22.04** | +4.26 |

### 关键发现

1. RAD在四个不同解剖部位的数据集上一致超越所有基线，平均提升2.24%-3.09%，证明了**跨解剖区域的泛化能力**
2. KAD在胸部X光数据上表现好但在其他区域下降，验证了预训练阶段注入知识的局限性
3. Guideline Recall从24.76%提升至65.62%，**定量证明了知识注入确实引导模型关注指南推荐的关键指标**
4. 文本分支的GECL比视觉分支贡献更大（同模态对齐更容易），Dual Decoder移除带来最大性能退化

## 亮点与洞察

- "知识注入全流程"的设计理念非常清晰：从输入（指南检索）→特征（GECL约束）→决策（双解码器）全方位对齐
- 双轴可解释性评估体系（文本Recall + 视觉IoU）为多模态诊断模型的可解释性量化提供了有价值的方法论
- 构建了MIMIC-ICD53新数据集（对齐MIMIC-CXR和MIMIC-IV），覆盖3种模态和53种疾病
- 通过离线检索+LLM精炼获取指南，比在线RAG更稳定，适合判别式任务

## 局限与展望

- 指南需要人工验证，扩展到更多疾病时成本较高
- 仅在判别式任务上验证，未探索生成式任务（如报告生成）
- Dual Decoder增加了推理开销，两个解码器的冗余度可进一步分析
- 视觉Grounding评估依赖外部数据集(ChestX-Det)的标注，覆盖的疾病类型有限

## 相关工作与启发

- 与RAG方法的区别值得关注：RAD是离线检索+结构化注入用于判别任务，而标准RAG是在线检索增强生成
- GECL的设计思路可推广到其他需要将领域知识注入特征空间的场景
- 双轴可解释性评估可作为标准化的评估协议推广

## 评分

- 新颖性: ⭐⭐⭐⭐ 全流程知识注入框架设计系统，可解释性评估体系有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 四个解剖部位、详尽消融、可解释性定量评估
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，公式推导完整，实验分析深入
- 价值: ⭐⭐⭐⭐ 对多模态医学诊断的知识注入和可解释性有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](../../ACL2026/graph_learning/legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2025\] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](../../ACL2025/graph_learning/kg_rag_recommendation.md)

</div>

<!-- RELATED:END -->
