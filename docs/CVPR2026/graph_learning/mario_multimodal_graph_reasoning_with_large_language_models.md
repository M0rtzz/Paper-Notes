---
title: >-
  [论文解读] Mario: Multimodal Graph Reasoning with Large Language Models
description: >-
  [CVPR 2026][图学习][多模态图] 提出 Mario，针对多模态图（MMG）上的 LLM 推理，通过图条件视觉语言模型（GVLM）实现拓扑感知的跨模态对齐，再用模态自适应提示路由器（MAPR）为每个节点选择最优模态配置，在节点分类和链接预测上达到 SOTA。
tags:
  - "CVPR 2026"
  - "图学习"
  - "多模态图"
  - "LLM推理"
  - "视觉语言对齐"
  - "模态自适应路由"
  - "指令微调"
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

Mario 处理的是多模态图（MMG）：每个节点带文本 + 图像属性，边提供结构先验。它要同时治两个病——节点的图文不一定语义同步（弱跨模态一致性），以及不同节点偏好不同模态（异质模态偏好）。整体两阶段：Stage 1 训练一个图条件视觉语言模型（GVLM），用双塔编码器加拓扑感知多模态混合器做图条件对比学习，产出结构感知、跨模态一致的表示；Stage 2 为每个节点构建文本/图像/双模态三种提示模板，用模态自适应提示路由器（MAPR）挑最优模板再送 LLM 推理。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["多模态图 MMG<br/>节点带文本+图像属性，边给结构先验"] --> B
    subgraph S1["Stage 1：图条件视觉语言模型 GVLM"]
        direction TB
        B["双塔编码器<br/>文本塔 / 图像塔分别编码"] --> C["拓扑感知多模态混合器<br/>逐层聚合邻居 CLS（带图位置偏置）再注入替换"]
        C --> D["图条件对比学习<br/>对结构感知图文 CLS 做双向 InfoNCE 对齐"]
    end
    D --> E["结构感知、跨模态一致的节点表示"]
    E --> F
    subgraph S2["Stage 2：模态自适应图指令微调"]
        direction TB
        F["构建三种提示模板<br/>仅文本 / 仅图像 / 双模态"] --> G["模态自适应提示路由器 MAPR<br/>吃图文表示 + 两跳结构 + 度数 → 路由概率"]
        G -->|"训练：软路由按性能后验加权"| H["LLM 指令微调"]
        G -->|"推理：硬路由选概率最大模板"| H
    end
    H --> I["节点分类 / 链接预测"]
```

### 关键设计

**1. 拓扑感知多模态混合器：让邻居信息进来消除图文歧义**

CLIP 这类 VLM 冻结编码时，单个节点的图文跨模态相似度很低，而邻居本可以帮它消歧却被忽略了。混合器的做法是在每个编码层从全图收集各节点的 CLS 表示，用带图结构位置偏置的多头注意力聚合邻居信息，再把这份结构感知的 CLS 重新注入 token 序列、替换掉原来的 CLS，逐层迭代实现结构与模态的深度融合。正是这一步把图拓扑灌进表示，让跨模态一致性相比冻结 CLIP 提升了 68%。

**2. 图条件对比学习：把"结构感知后的图文"对齐到一起**

光有混合器还需要一个训练目标来真正拉近图文。Mario 对结构感知后的文本/图像 CLS 嵌入做双向 InfoNCE：

$$\mathcal{L}_{\text{S1}} = -\frac{1}{|\mathcal{B}|}\sum_v \Big[\log\frac{e^{s(v,v)/\tau}}{\sum_u e^{s(v,u)/\tau}} + \log\frac{e^{s(v,v)/\tau}}{\sum_u e^{s(u,v)/\tau}}\Big]$$

同一节点的图文互为正样本、其余为负样本，双向对称地拉近正对、推开负对，得到的就是带拓扑约束的跨模态一致表示，供第二阶段使用。

**3. 模态自适应提示路由器（MAPR）：每个节点用自己最吃得开的模态**

约 30% 的节点只能在某种特定模态配置下被正确分类，一刀切的提示模板白白浪费信息。MAPR 为每个节点准备三种提示——仅文本 $\mathcal{S}_v^{\text{txt}}$、仅图像 $\mathcal{S}_v^{\text{vis}}$、双模态 $\mathcal{S}_v^{\text{mm}}$，路由器吃进 $[\mathbf{h}_v^{\text{text}}; \mathbf{h}_v^{\text{image}}; \phi^{(1)}(v); \phi^{(2)}(v); \log d_v]$（图文表示 + 两跳结构特征 + 度数），经 MLP 输出三类路由概率 $\mathbf{p}_v = \text{softmax}(\mathbf{s}_v)$。训练时用实际表现当老师：把三种模板各自的损失取负、softmax 成性能后验 $\mathbf{q}_v = \text{softmax}(-[\ell_v^{(\text{txt})}, \ell_v^{(\text{vis})}, \ell_v^{(\text{mm})}])$，让路由概率去逼近它——

$$\mathcal{L}_{\text{S2}} = \frac{1}{|B|}\sum_v \Big[\sum_k q_v^{(k)} \ell_v^{(k)} + \lambda \, \text{KL}(\mathbf{q}_v \| \mathbf{p}_v)\Big]$$

训练时软路由（按概率加权三种损失）、推理时硬路由（直接选概率最大的模板），既学得稳又在推理时零额外开销。

### 损失函数 / 训练策略

Stage 1 用对比损失训练编码器，Stage 2 用性能加权的 LM 损失 + KL 正则同时微调 LLM 和路由器；推理时路由器直接选最优模态模板。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](../../ACL2025/graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)

</div>

<!-- RELATED:END -->
