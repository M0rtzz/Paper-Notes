---
title: >-
  [论文解读] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction
description: >-
  [ACL 2026][医学图像][内部记忆检索] 本文提出K2K框架，将LLM的FFN参数空间视为可检索的知识库，通过LoRA注入临床知识、激活引导的探针构建精确检索、交叉注意力重排序自适应整合，实现了无需外部检索延迟的医疗预测SOTA。
tags:
  - "ACL 2026"
  - "医学图像"
  - "内部记忆检索"
  - "FFN键值记忆"
  - "医疗预测"
  - "知识注入"
  - "交叉注意力重排"
---

# Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction

**会议**: ACL 2026  
**arXiv**: [2604.07659](https://arxiv.org/abs/2604.07659)  
**代码**: [https://anonymous.4open.science/r/K2K-2390/](https://anonymous.4open.science/r/K2K-2390/)  
**领域**: 医学图像  
**关键词**: 内部记忆检索、FFN键值记忆、医疗预测、知识注入、交叉注意力重排

## 一句话总结
本文提出K2K框架，将LLM的FFN参数空间视为可检索的知识库，通过LoRA注入临床知识、激活引导的探针构建精确检索、交叉注意力重排序自适应整合，实现了无需外部检索延迟的医疗预测SOTA。

## 研究背景与动机

**领域现状**：LLM在医疗领域展现了显著潜力，但部署中面临幻觉和缺乏细粒度医学上下文的问题。RAG是主流的知识接地策略，现有方法从知识图谱、非结构化文档或自生成知识中检索。

**现有痛点**：传统RAG管线存在两个关键瓶颈——(1) 通过输入prompt注入外部知识会扩展上下文长度，增加推理成本并限制可扩展性；(2) 构建高质量检索器仍是难题，监督检索需要大量标注的查询-上下文对，结构化检索依赖昂贵的图搜索或过度简化的启发式。这些在时间敏感的临床环境中是不可接受的。

**核心矛盾**：需要既准确又快速地获取相关医学知识，但外部检索带来的延迟和复杂性与临床实时决策的需求冲突。已有研究表明FFN层隐式存储了事实知识（键值记忆解释），但直接用原始查询检索内部键不够准确——不同查询检索到的键高度相似，探针表示缺乏判别力。

**本文目标**：设计一种从LLM内部参数空间直接检索知识的框架，避免外部检索的延迟和复杂性。

**切入角度**：利用Geva et al.的FFN键值记忆解释——FFN权重矩阵 $W_1$ 的列作为"键"存储语义模式，$W_2$ 的行作为"值"存储对应知识。通过LoRA注入领域知识后，这些键就成为了一个可搜索的内部知识库。

**核心 idea**：将医学知识通过LoRA"写入"LLM参数空间，然后用激活引导的探针精确检索相关内部键，再通过交叉注意力动态整合。

## 方法详解

### 整体框架
K2K分三步：(1) 内部记忆构建——通过领域适配模型和LoRA注入分别构建文档级和图谱级记忆；(2) 激活引导的探针构建——识别关键token和稀缺异常特征来增强查询的判别力；(3) 交叉注意力重排——动态整合和重新加权多源内部知识。输入为纵向EHR诊断代码序列，输出为二分类预测（死亡率/再入院）。

### 关键设计

1. **内部记忆构建**:

    - 功能：将外部临床知识编码到LLM的FFN参数空间中，形成可检索的内部记忆。
    - 核心思路：文档级记忆利用领域适配LLM（如BioMistral）的FFN $W_1$ 矩阵作为键 $K_{\text{doc}}^l$；图谱级记忆将医学知识图谱的三元组线性化为文本（如"The relationship between [head] and [tail] is [relation]"），通过LoRA微调注入，LoRA的 $A_1 B_1$ 矩阵作为图谱键 $K_{\text{graph}}^l$。两种记忆提供互补的非结构化和结构化知识。
    - 设计动机：外部检索需要维护独立的检索系统和处理长上下文，而将知识编码到参数中消除了推理时的检索延迟。LoRA的低秩特性使知识注入高效且不损害模型原有能力。

2. **激活引导的探针构建**:

    - 功能：构建具有高判别力的查询探针，确保从内部记忆中精确检索相关知识。
    - 核心思路：对输入序列的隐状态 $H_w$，不使用简单的均值池化（会分散注意力），而是计算每个token的Mahalanobis距离的对角近似 $\phi_j^w \approx \sqrt{\sum_d \frac{(h_{j,d}^w - \bar{z}_d^w)^2}{\sigma_d^2}}$ 作为上下文激活权重。归一化后作为软注意力分布，加权聚合token向量得到增强的探针 $Q_w = \sum_j \alpha_j^w \cdot h_j^w$，强调语义锚点token。
    - 设计动机：标准均值池化导致不同查询产生高度相似的探针（缺乏判别力）。Mahalanobis距离考虑每个维度的方差，对低方差方向的偏差更敏感，能识别出真正重要的异常特征。

3. **交叉注意力重排**:

    - 功能：动态整合和重新加权从内部记忆检索到的多源知识，实现任务感知的知识选择。
    - 核心思路：将输入表示分割为多个窗口，每个窗口的增强探针 $Q_w^+$ 从文档和图谱记忆中检索top-k键。通过交叉注意力（CA）机制重排这些键，得到文档知识 $H_{\text{doc}}^w$ 和图谱知识 $H_{\text{graph}}^w$。将两者池化、拼接后与原始输入表示合并，通过MLP做最终预测。
    - 设计动机：检索到的内部键是潜在的、未接地的，缺乏显式来源。交叉注意力提供了一种自适应、任务感知的方式来动态选择和加权最相关的知识。

### 损失函数 / 训练策略
使用标准交叉熵损失进行分类。LoRA用于知识注入阶段。评估使用MIMIC-III和MIMIC-IV数据集，按患者ID分组划分防止数据泄漏。

## 实验关键数据

### 主实验

| 方法 | MIMIC-III Mort Avg | MIMIC-III Read Avg | MIMIC-IV Mort Avg | MIMIC-IV Read Avg |
|------|-------------------|-------------------|-------------------|-------------------|
| KARE (前SOTA) | 较低 | 较低 | 较低 | 较低 |
| Standard RAG | 中等 | 中等 | 中等 | 中等 |
| K2K (BioMistral) | 较高 | 较高 | 较高 | 较高 |
| K2K (Meditron3) | **最高** | **最高** | **最高** | **最高** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full K2K | 最优 | 完整框架 |
| w/o 图谱记忆 | 下降 | 结构化知识的贡献 |
| w/o 激活引导 | 下降 | 探针判别力的重要性 |
| w/o 交叉注意力重排 | 下降 | 动态整合的必要性 |
| 使用均值池化探针 | 显著下降 | 验证了Mahalanobis加权的优势 |

### 关键发现
- K2K在四个基准数据集上达到SOTA，且检索效率远高于KARE和prompt-based方法
- Meditron3-Qwen2.5-7B比BioMistral-7B表现更好，说明基础模型能力对内部记忆质量有重要影响
- 激活引导的探针比标准均值池化显著提升检索精度，验证了Mahalanobis距离的有效性
- 同时使用文档级和图谱级记忆优于单一来源，两者提供互补信息

## 亮点与洞察
- **参数即知识库**：将FFN的键值记忆解释从理论洞察转化为实用系统，直接在参数空间中检索知识，消除了外部检索的延迟。这个思路可以推广到任何需要快速知识访问的场景。
- **Mahalanobis距离增强探针**：考虑每个维度的方差来加权token的重要性，比简单的欧氏距离或均值池化更能识别语义锚点。这是一个通用的表示增强技巧。
- **LoRA作为知识注入工具**：不是用LoRA来微调任务性能，而是用它来"写入"新知识到参数空间。这种用法为LoRA开辟了新的应用方向。

## 局限与展望
- 内部键是潜在的、未接地的，缺乏可解释性——不知道检索到的键对应什么具体知识
- 依赖领域适配的基础模型（BioMistral/Meditron3），通用LLM的效果未知
- 仅在ICD代码序列的分类任务上验证，未测试生成性任务
- 知识图谱线性化的质量依赖于三元组的表述方式

## 相关工作与启发
- **vs KARE**：KARE结合文档检索和知识图谱最短路径，但图搜索成本高。K2K将所有知识编码到参数中，推理时零延迟
- **vs Standard RAG**：RAG扩展上下文长度增加推理成本，K2K通过内部检索避免上下文膨胀
- **vs RETRO**：RETRO也使用基于窗口的检索和交叉注意力，但从外部数据库检索。K2K将这种架构适配到内部参数检索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将FFN键值记忆解释转化为实用的内部检索系统是非常新颖的思路
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、多个基线、充分的消融实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，理论与实践结合好
- 价值: ⭐⭐⭐⭐ 为时间敏感的临床场景提供了低延迟的知识接地方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)
- [\[AAAI 2026\] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives](../../AAAI2026/medical_imaging/bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](../../AAAI2026/medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)

</div>

<!-- RELATED:END -->
