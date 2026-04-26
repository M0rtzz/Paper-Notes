---
title: >-
  [论文解读] M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations
description: >-
  [ACL 2025][图学习][多模态情感原因分析] 提出 M3HG 模型，通过构建多模态多类型节点异构图来显式建模对话中的情感与原因上下文，并在句间和句内两个尺度上融合语义信息，实现多模态对话中情感-原因三元组的端到端提取。同时构建了首个中文多场景 MECTEC 数据集 MECAD。
tags:
  - ACL 2025
  - 图学习
  - 多模态情感原因分析
  - 异构图注意力网络
  - 情感-原因三元组提取
  - 多尺度语义融合
  - 对话情感分析
---

# M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations

**会议**: ACL 2025  
**arXiv**: [2508.18740](https://arxiv.org/abs/2508.18740)  
**代码**: https://github.com/redifinition/M3HG  
**领域**: 图学习  
**关键词**: 多模态情感原因分析, 异构图注意力网络, 情感-原因三元组提取, 多尺度语义融合, 对话情感分析

## 一句话总结

提出 M3HG 模型，通过构建多模态多类型节点异构图来显式建模对话中的情感与原因上下文，并在句间和句内两个尺度上融合语义信息，实现多模态对话中情感-原因三元组的端到端提取。同时构建了首个中文多场景 MECTEC 数据集 MECAD。

## 研究背景与动机

**领域现状**: 多模态对话中的情感原因分析（MECTEC）是社交媒体分析中的重要任务，目标是从包含文本、音频、视频的对话中同时提取情感utterance、原因utterance和情感类别组成的三元组 $\mathcal{P}=\{(\boldsymbol{U}_j^e, \boldsymbol{U}_j^c, y_j^e)\}$。

**现有痛点**: 
   - 数据集匮乏：现有仅一个 ECF 数据集，来源单一（仅 Friends 剧集），场景多样性严重不足
   - 现有方法未显式建模情感和原因上下文，忽视了不同尺度（句间/句内）语义信息的融合
   - 无法有效处理原因utterance出现在情感utterance之后的情况

**核心矛盾**: 情感归因理论指出，情感与原因的关系由特定上下文（如文本中的情感词、音频中的语调、视频中的表情）揭示，但现有方法将多模态融合和上下文提取视为独立过程。

**本文目标**: 如何在多模态对话中显式捕获情感和原因上下文，同时有效融合不同尺度的语义信息来提取情感-原因三元组。

**切入角度**: 设计异构图，引入多类型节点（情感上下文节点、原因上下文节点、话语超级节点、对话超级节点）和多种边关系，在图上进行多尺度语义融合。

**核心 idea**: 用多类型节点异构图统一建模情感/原因上下文、多模态特征和多尺度语义关系。

## 方法详解

### 整体框架

M3HG 是端到端模型，包含四个核心组件：
1. 单模态特征提取
2. 异构图构建
3. 多尺度语义融合
4. 情感-原因分类

### 关键设计

1. **单模态特征提取**: 

    - 文本：SA-RoBERTa 提取 $\boldsymbol{E}^t \in \mathbb{R}^{n \times d_t}$，再用多头自注意力编码局部上下文得到 $\boldsymbol{H}^t$
    - 音频：Wav2Vec2 提取 $\boldsymbol{E}^a$，用 GRU + LayerNorm + FFN 编码：$\boldsymbol{H}^m = LN(\boldsymbol{E}^m + \boldsymbol{E}'^m + FFN(\boldsymbol{E}'^m))$
    - 视频：DenseNet 提取 $\boldsymbol{E}^v$，同样用 GRU 编码
    - 三个模态通过线性层映射到统一维度 $d_h$

2. **异构图构建**: 

    - **四种节点类型**: 情感上下文节点 $N^e$、原因上下文节点 $N^c$、话语超级节点 $SN^u=\{N^t, N^a, N^v\}$、对话超级节点 $SN^d$
    - **五种边关系**: 同说话人边 $r_{ss}$、不同说话人边 $r_{ds}$、全局连接边 $r_{gc}$、情感连接边 $r_{ec}$、原因连接边 $r_{cc}$
    - 全局连接边使得每个话语超级节点都与对话超级节点双向连接，**首次支持原因utterance出现在情感utterance之后的场景**

3. **多尺度语义融合（HGAT）**: 

    - **句内融合（Intra-utterance）**: 通过元路径 $\Phi_{intra}$ 在单个话语内部融合三模态语义信息到情感/原因上下文节点。注意力计算为：
    $\alpha_{ij}^\phi = \frac{\exp(\sigma(\boldsymbol{a}_\phi^T \cdot [\boldsymbol{H}'_i \| \boldsymbol{H}'_j]))}{\sum_{k \in \mathcal{N}_i^\phi} \exp(\sigma(\boldsymbol{a}_\phi^T \cdot [\boldsymbol{H}'_i \| \boldsymbol{H}'_k]))}$
    - **句间融合（Inter-utterance）**: 通过元路径 $\Phi_{inter}$ 在不同话语之间传播语义信息，经由对话超级节点 $SN^d$ 实现全局信息聚合
    - 融合后用语义注意力机制 + PFFN 更新节点特征

4. **情感-原因分类**: 

    - 情感节点 $\boldsymbol{Z}_i^e$ 送入 Emotion MLP 预测情感类别 $\hat{y}_i^e$
    - 原因节点 $\boldsymbol{Z}_i^c$ 送入 Cause MLP 预测原因指示 $\hat{y}_i^c$
    - 对于话语对 $(U_i, U_j)$，用 RBF 核函数计算相对位置编码 $RPE_{ij}$，拼接后送入 MLP 判断因果关系：
    $\hat{y}_{ij}^{ec} = \sigma(MLP(\boldsymbol{Z}_j^e \| \boldsymbol{Z}_i^c \| RPE_{ij}))$

### 损失函数/训练策略

- 使用 Focal Loss 处理类别不平衡：$\mathcal{L}^\beta = -\frac{1}{N^\beta}\sum_{i=1}^{N^\beta}\alpha^\beta(1-\hat{y}_i^\beta)^\gamma \log(\hat{y}_i^\beta)$
- 三个任务（情感预测、原因预测、情感-原因对预测）的损失联合优化

## 实验关键数据

### 主实验

| 数据集 | 方法 | 模态 | 6 Avg F1 | 4 Avg F1 |
|--------|------|------|----------|----------|
| ECF | HiLo (SOTA) | T,A,V | 33.04 | 35.81 |
| ECF | GPT-4o (5-shots) | T | 29.13 | 30.30 |
| ECF | **M3HG (T)** | T | 37.46 | 39.95 |
| ECF | **M3HG (T,A,V)** | T,A,V | **40.07** | **41.96** |
| MECAD | SHARK (SOTA) | T | 27.58 | 29.99 |
| MECAD | GPT-4o (5-shots) | T | 27.16 | 28.42 |
| MECAD | **M3HG (T,A,V)** | T,A,V | **32.82** | **34.59** |

### 消融实验

| 模态组合 | ECF 6 Avg | ECF 4 Avg | MECAD 6 Avg | MECAD 4 Avg |
|----------|-----------|-----------|-------------|-------------|
| T | 37.46 | 39.95 | 30.81 | 32.55 |
| T+A | 39.10 | 40.97 | 32.16 | 33.73 |
| T+V | 38.90 | 40.72 | 31.95 | 33.52 |
| T+A+V | 40.07 | 41.96 | 32.82 | 34.59 |

### 关键发现

- M3HG 仅用文本模态就超越所有基线（包括多模态基线），证明图结构设计的有效性
- 相比 SOTA 方法 HiLo，M3HG 在 ECF 上 6 Avg 提升 21.28%，4 Avg 提升 17.17%
- 在 Disgust 和 Fear 等样本稀少的类别上，M3HG 相比 GPT-4o 分别提升 31.36% 和 58.46%
- 每增加一个模态都带来性能提升，验证了多模态融合的必要性
- MECAD 数据集包含 989 段对话、10,516 条utterance，来自 56 部电视剧，Fleiss's Kappa 达到 0.6932

## 亮点与洞察

- **异构图设计精妙**: 四种节点 + 五种边关系的异构图，统一了多模态融合、上下文建模和情感-原因关联三个问题
- **首次处理"因后于果"场景**: 通过全局连接边使每个utterance都与对话节点相连，打破了前向依赖的限制
- **多尺度融合**: 句内融合捕获单个utterance内多模态情感/原因线索，句间融合传播对话级上下文，两者互补
- **数据集贡献**: MECAD 是首个中文多场景 MECTEC 数据集，显著提升了该领域数据多样性

## 局限与展望

- 未整合外部知识（如常识知识图谱），限制了情感预测和原因预测的准确性
- 无法处理过长对话，受限于语言模型的输入长度
- 多模态融合过程中可能存在误差传播，尤其当不同模态信息不一致时
- MECAD 数据集来自电视剧，与真实社交媒体对话仍有差距

## 相关工作与启发

- **情感原因分析**: RECCON → ConvECPE → ECF → MECAD，从文本到多模态的演进
- **异构图注意力网络 HGAT** (Wang et al., 2019) 的元路径机制为多类型节点间的信息传播提供了灵活框架
- **Focal Loss** 有效缓解了情感类别不均衡问题
- 可以考虑引入 LLM 进行情感原因分析的 few-shot 或 zero-shot 推理

## 评分

- **新颖性**: ⭐⭐⭐⭐ (异构图设计新颖，四种节点+五种边完整建模了 MECTEC 任务的多个方面)
- **实验充分度**: ⭐⭐⭐⭐ (两个数据集上全面对比，模态消融、GPT-4o 对比均有)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，图示表意准确)
- **价值**: ⭐⭐⭐⭐ (MECAD 数据集 + M3HG 模型双重贡献，推动了多模态情感原因分析领域)

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs](multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs.md)
- [\[CVPR 2025\] DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition](../../CVPR2025/graph_learning/dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition.md)
- [\[NeurIPS 2025\] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](../../NeurIPS2025/graph_learning/heterogeneous_swarms_jointly_optimizing_model_roles_and_weights_for_multi-llm_sy.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[ICML 2025\] GlycanAA: Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training](../../ICML2025/graph_learning/modeling_all-atom_glycan_structures_via_hierarchical_message_passing_and_multi-s.md)

<!-- RELATED:END -->
