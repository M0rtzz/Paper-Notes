---
title: >-
  [论文解读] Mario: Multimodal Graph Reasoning with Large Language Models
description: >-
  [CVPR 2026][图学习][多模态图] 提出 Mario，针对多模态图（MMG）上的 LLM 推理，通过图条件视觉语言模型（GVLM）实现拓扑感知的跨模态对齐，再用模态自适应提示路由器（MAPR）为每个节点选择最优模态配置，在节点分类和链接预测上达到 SOTA。
tags:
  - CVPR 2026
  - 图学习
  - 多模态图
  - LLM推理
  - 视觉语言对齐
  - 模态自适应路由
  - 指令微调
---

# Mario: Multimodal Graph Reasoning with Large Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.05181](https://arxiv.org/abs/2603.05181)  
**代码**: 即将公开  
**领域**: 图学习  
**关键词**: 多模态图, LLM推理, 视觉语言对齐, 模态自适应路由, 指令微调

## 一句话总结

提出 Mario，针对多模态图（MMG）上的 LLM 推理，通过图条件视觉语言模型（GVLM）实现拓扑感知的跨模态对齐，再用模态自适应提示路由器（MAPR）为每个节点选择最优模态配置，在节点分类和链接预测上达到 SOTA。

## 研究背景与动机

现有多模态 LLM 处理独立图文对，忽略了现实中多模态数据间的关系结构。多模态图（MMG）中每个节点有文本+图像属性、边提供结构先验。直接用 VLM（如 CLIP）编码再送图模型存在两个挑战：

**C1 弱跨模态一致性**：节点的图文不一定语义同步，邻居信息可以消歧但被忽略。CLIP 冻结时的跨模态余弦相似度低，加入图拓扑后提升 68%。

**C2 异质模态偏好**：不同节点的信息量在不同模态上不同。约 30% 节点只能被某种特定模态配置正确分类。一刀切的提示模板浪费信息。

### 开放问题

> 能否设计一个统一框架，在 LLM 推理中同时解决 MMG 上的跨模态不一致和异质模态偏好？

## 方法详解

### 整体框架

**Stage 1（GVLM）**：双塔编码器 + 拓扑感知多模态混合器 → 图条件对比学习 → 结构感知跨模态一致表示。
**Stage 2（模态自适应指令微调）**：构建文本/图像/多模态三种提示模板 → MAPR 路由器选择最优模板 → LLM 推理。

### 关键设计

1. **拓扑感知多模态混合器**：在每个编码层，从全图收集各节点的 CLS 表示，通过带图结构位置偏置的多头注意力聚合邻居信息，再将结构感知的 CLS 重新注入 token 序列替代原 CLS。逐层迭代实现结构+模态的深度融合。

2. **图条件对比学习**：对结构感知的文本/图像 CLS 嵌入做双向 InfoNCE：
    $\mathcal{L}_{\text{S1}} = -\frac{1}{|\mathcal{B}|}\sum_v [\log\frac{e^{s(v,v)/\tau}}{\sum_u e^{s(v,u)/\tau}} + \log\frac{e^{s(v,v)/\tau}}{\sum_u e^{s(u,v)/\tau}}]$

3. **模态自适应提示路由器 (MAPR)**：

    - 为每个节点构建3种提示：$\mathcal{S}_v^{\text{txt}}$（仅文本token）、$\mathcal{S}_v^{\text{vis}}$（仅图像token）、$\mathcal{S}_v^{\text{mm}}$（双模态token）
    - 路由器输入：$[\mathbf{h}_v^{\text{text}}; \mathbf{h}_v^{\text{image}}; \phi^{(1)}(v); \phi^{(2)}(v); \log d_v]$
    - MLP 输出3类路由概率 $\mathbf{p}_v = \text{softmax}(\mathbf{s}_v)$
    - 用性能后验 $\mathbf{q}_v = \text{softmax}(-[\ell_v^{(\text{txt})}, \ell_v^{(\text{vis})}, \ell_v^{(\text{mm})}])$ 作为教师信号
    - 损失：$\mathcal{L}_{\text{S2}} = \frac{1}{|B|}\sum_v [\sum_k q_v^{(k)} \ell_v^{(k)} + \lambda \text{KL}(\mathbf{q}_v \| \mathbf{p}_v)]$

### 损失函数 / 训练策略

Stage 1 用对比损失训练编码器，Stage 2 用性能加权的 LM 损失 + KL 正则微调 LLM 和路由器。推理时路由器选最优模态模板。

## 实验关键数据

### 主实验（节点分类准确率 %）

| 方法 | Movies | Reddit | CDs | Arts |
|------|--------|--------|-----|------|
| GCN(text) | 43.8 | 84.3 | 51.4 | 76.9 |
| GATv2(text) | 48.7 | 85.6 | 54.7 | 80.4 |
| **Mario** | **53.6+** | **95.3+** | **63.4+** | **92.1+** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无图条件VLM（CLIP冻结） | 低一致性 | 跨模态不对齐 |
| 节点级微调（无拓扑） | 部分改善 | 缺邻居信息 |
| +GVLM（阶段1） | 显著提升 | 拓扑+模态双感知 |
| +MAPR（阶段2） | **最优** | 模态自适应选择 |

### Mix-Training 设置（节点分类准确率 %）

| 方法 | 模态 | Movies | Reddit | CDs | Arts |
|------|------|--------|--------|-----|------|
| SAGE | Text | 46.85 | 89.96 | 53.24 | 87.46 |
| LLaGA | Text | 47.80 | 91.14 | 51.33 | 74.02 |
| LLaGA-A | Text+Image | 50.61 | 92.94 | 56.29 | 88.83 |
| Graph4MM | Text+Image | 51.07 | 92.89 | 55.53 | 89.32 |
| **Mario-8B** | **Text+Image** | **53.63** | **95.30** | **63.43** | **92.13** |

### 关键发现

- 图拓扑引入后跨模态一致性提升 68%（vs CLIP 冻结）
- ~30% 节点有明确的单模态偏好
- 零样本迁移最高提升 1.6 倍

## 亮点与洞察

- **两个挑战识别精准**：弱一致性和异质偏好是 MMG 推理的真实瓶颈，Venn 图分析直观有力
- **MAPR 路由机制优雅**：用 LLM 损失作为性能信号驱动路由学习，训练时软路由、推理时硬路由零开销
- **Stage 1 的 GVLM 是新范式**：拓扑感知的视觉语言模型，Transformer 层内交替执行图注意力和 token 注意力
- **零样本迁移强**：在未见过的 MMG 上实现最高 1.6× 增益，说明学到的模态路由策略具有泛化性
- **统一框架**：同一架构处理节点分类和链接预测两种任务，通用性好

## 局限性

- 两阶段训练增加复杂性，Stage 2 训练时每个样本需三次 LLM 前向传播
- 混合器的注意力复杂度 $\mathcal{O}(|\mathcal{V}_s|^2 d)$，对大规模图需节点采样
- 当前仅处理文本+图像双模态图，未扩展到音频、视频等模态
- 图拓扑偏置 $\mathbf{B}_h$ 依赖最短路径预计算，对动态图不友好
- MLaGA 用 Q-Former 融合后再送 LLM，Graph4MM 处理缺失模态——Mario 在完整模态场景更优，但缺失模态场景未测试

## 评分

⭐⭐⭐⭐⭐ (5/5)

GVLM + MAPR 双重创新，四数据集 × 两任务 × 三模态设置实验全面覆盖，零样本迁移验证泛化力，是多模态图 + LLM 推理方向的重要开拓性工作。

<!-- RELATED:START -->

## 相关论文

- [Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](../../ACL2025/graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [Table as a Modality for Large Language Models](../../NeurIPS2025/graph_learning/table_as_a_modality_for_large_language_models.md)
- [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](../../ICML2025/graph_learning/from_rag_to_memory_non-parametric_continual_learning_for_large_language_models.md)

<!-- RELATED:END -->
