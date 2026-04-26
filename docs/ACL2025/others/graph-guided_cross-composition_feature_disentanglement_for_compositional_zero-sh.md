---
title: >-
  [论文解读] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning
description: >-
  [ACL 2025][compositional zero-shot learning] DCDA 提出图引导的跨组合特征解耦方案，通过双适配器（L-Adapter 用于文本端 GNN 特征聚合、V-Adapter 用于视觉端跨注意力解耦）注入冻结 CLIP，在组合零样本学习任务上显著超越现有方法。
tags:
  - ACL 2025
  - compositional zero-shot learning
  - CLIP adapter
  - feature disentanglement
  - 图神经网络
  - 注意力机制
---

# Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning

**会议**: ACL 2025  
**arXiv**: [2408.09786](https://arxiv.org/abs/2408.09786)  
**代码**: [有](https://github.com/zhurunkai/DCDA)  
**领域**: 视觉语言 / 零样本学习  
**关键词**: compositional zero-shot learning, CLIP adapter, feature disentanglement, graph neural network, cross-attention

## 一句话总结

DCDA 提出图引导的跨组合特征解耦方案，通过双适配器（L-Adapter 用于文本端 GNN 特征聚合、V-Adapter 用于视觉端跨注意力解耦）注入冻结 CLIP，在组合零样本学习任务上显著超越现有方法。

## 研究背景与动机

组合零样本学习（CZSL）的目标是：模型训练时只见过部分属性-物体组合（如 red tomato、green apple），测试时需要识别从未见过的新组合（如 green tomato）。这要求模型能够将属性和物体的视觉特征"解耦"，并重新组合用于泛化。

当前基于 CLIP 的 CZSL 方法面临的核心挑战是**原语（primitive）的视觉纠缠问题**：

**属性与物体在图像中高度纠缠**：例如 "red" 在 red tomato 中渗透到所有像素，无法简单分离

**同一原语在不同组合中表现差异大**：如 "red" 在 tomato、wine、car 上的色调、空间分布截然不同（Figure 1）

**现有方法忽视了跨组合多样性**：无论是基于文本提示的方法（CSP、DFSP）还是视觉适配器方法（CAILA、Troika），都仅在单一组合内学习解耦特征，没有利用"共享同一属性的不同组合"之间的关系

作者通过在 MIT-States 数据集上对 CAILA 进行 t-SNE 可视化（Figure 2）发现：解耦后的属性特征（如 "broken"）仍然广泛散布、与其他属性聚类重叠，导致未见组合的泛化能力不足。

**核心洞察**：有效的原语解耦需要**跨组合特征聚合**——利用多个共享同一原语的组合来约束解耦特征的跨组合一致性。

## 方法详解

### 整体框架

DCDA 在冻结的 CLIP 文本编码器和图像编码器中分别插入 L-Adapter 和 V-Adapter。L-Adapter 利用组合图和 GNN 在文本端聚合跨组合的原语特征；V-Adapter 利用跨注意力机制在视觉端从共享原语的图像对中提取不变特征。两类适配器均仅插入最后 3 层 Transformer block。

### 关键设计

1. **L-Adapter（文本端图引导特征聚合）**：

    - **组合图构建**：构建三部图，包含所有属性节点、物体节点和组合节点。每个组合 $c=(a,o)$ 与其属性 $a$ 和物体 $o$ 形成三角形连接
    - **节点特征初始化**：为属性、物体、组合分别设计独立提示（"a photo of [attribute] object"、"a photo of [object]"、"a photo of [attribute] [object]"），取 CLIP 文本编码器输出的 [EOT] token 嵌入作为初始特征。这样文本原语特征天然与组合特征解耦
    - **GNN 特征传播**：在图上运行 K 层 GNN，每层对属性/物体/组合节点分别执行邻域聚合（AGG）和更新（CON），使每个原语节点融合所有共享它的组合的特征

   设计动机：通过图结构显式建模属性-物体-组合之间的共享关系，让文本端的原语特征能够一般化到所有相关组合。

2. **V-Adapter（视觉端跨注意力解耦）**：

    - **跨注意力机制**：对于目标图像 $x_{(a,o)}$，采样一个共享属性 $a$ 但物体不同的辅助图像 $x_{(a,o')}$，通过交叉注意力（辅助图像为 query，目标图像为 key/value）提取目标图像中与属性 $a$ 更相关的特征
    - **原语相关性引导的采样策略（PRG）**：为解决属性邻居组合数量不平衡的问题，构建物体相关性矩阵 $A^{\text{obj}} = (A^{\text{att-obj}})^T A^{\text{att-obj}}$，选择与目标物体最相关和最不相关的 top-n 物体组成代表性辅助组合，再加权随机采样。物体相关性由共同出现的属性数量决定

   设计动机：视觉端无法像文本端那样为每个原语建立独立提示（属性和物体在图像中纠缠），因此通过共享原语的图像对之间的交叉注意力来"过滤"出跨组合共有的视觉模式。

3. **适配器集成策略**：

    - L-Adapter 插入在 Transformer block 的自注意力层后和前馈层后（位置②③）
    - V-Adapter 插入在整个 Transformer block 之后（位置①）
    - 均仅在最后 3 层插入，保留 CLIP 底层的通用特征
    - 每个适配器有跳跃连接：输出 = 适配器输出 + 原始输入
    - 提示中的词嵌入设为可训练

### 损失函数 / 训练策略

兼容性得分融合三个维度的匹配：

$$s(x_i, c_i) = \alpha[\hat{h}_i^v \cdot \hat{h}_{c_i}^t] + \beta[\hat{h}_{i \to A}^v \cdot \hat{h}_{a_i}^t] + \gamma[\hat{h}_{i \to O}^v \cdot \hat{h}_{o_i}^t]$$

第一项为组合级匹配，第二项为属性级匹配，第三项为物体级匹配。$\alpha, \beta, \gamma$ 为可学习参数。

训练使用标准交叉熵损失，以温度 $\tau$ 缩放的 softmax 在所有 seen 组合上归一化。仅训练适配器参数和可训练 token 嵌入，CLIP 原始参数完全冻结。

## 实验关键数据

### 主实验——闭世界设置（表格）

| 方法 | MIT-States AUC | UT-Zappos AUC | C-GQA AUC |
|------|---------------|---------------|-----------|
| CLIP (无微调) | 11.0 | 5.0 | 1.4 |
| CSP | 19.4 | 33.0 | 6.2 |
| CAILA | 23.4 | 44.1 | **9.9** |
| Troika | 22.1 | 41.7 | 9.2 |
| DCDA[RD] | 16.2 | 40.1 | 8.5 |
| DCDA[PRG] | 26.9 | 43.0 | 8.9 |
| **DCDA[PRG+N]** | **27.0** | **44.2** | 9.4 |

DCDA[PRG+N] 在 MIT-States 上 AUC 较 CAILA 提升 3.6%，在 UT-Zappos 上提升 0.1%。

### 消融实验（表格）

| 配置 | MIT-States S | U | H | AUC |
|------|-------------|-----|-----|-----|
| 完整模型 (DCDA[PRG]) | 57.3 | 55.1 | 43.2 | 26.9 |
| 移除 L-Adapter | 55.9 | 54.7 | 42.2 | 26.1 |
| 移除 V-Adapter | 44.9 | 46.9 | 33.8 | 17.1 |
| L-Adapter 无跨组合信息 | 57.5 | 54.6 | 43.0 | 26.7 |
| V-Adapter 无跨组合信息 | 44.5 | 46.2 | 33.7 | 17.0 |
| L&V 均无跨组合信息 | 44.2 | 46.1 | 33.6 | 16.7 |

### 关键发现

1. **V-Adapter 比 L-Adapter 更关键**：移除 V-Adapter 导致 AUC 从 26.9 暴跌至 17.1（-36%），而移除 L-Adapter 仅降至 26.1。这证实了视觉原语比文本原语更纠缠，视觉端解耦更重要
2. **PRG 采样策略至关重要**：DCDA[PRG] vs DCDA[RD]，在 MIT-States AUC 上从 16.2 提升至 26.9（+66%），验证了有策略地选择辅助组合远胜于随机采样
3. **跨组合信息在 V-Adapter 中是核心**：不使用跨组合信息的 V-Adapter（17.0 AUC）甚至不如完全移除 V-Adapter（17.1 AUC），说明无跨组合约束的解耦会产生误导性的特征子空间
4. **适配器仅需插入最后 3 层**：加入更多层（如 6 层）反而过拟合

## 亮点与洞察

- **跨组合特征解耦**是本文的核心创新：不是在单个组合内做解耦，而是利用多个共享原语的组合相互约束，学到跨组合一致的原语特征
- **双适配器架构设计精巧**：文本端用图传播（天然适合离散的标签结构），视觉端用交叉注意力（适合处理连续的纠缠视觉特征），针对不同模态特性选择不同机制
- **参数效率高**：仅在最后 3 层插入适配器，可训练参数远少于全层方案（CAILA/Troika），但性能更好
- t-SNE 可视化（Figure 2 右）清晰展示 DCDA 解耦后属性聚类更紧密、类间分离更好

## 局限与展望

1. 组合图的构建和更新开销随属性/物体数量增长，大规模应用可能面临效率瓶颈
2. 极端长尾分布（罕见原语仅有少量组合）下的跨组合监督可能不足
3. C-GQA 数据集上表现未达最优，可能与低质量图像和更大组合空间有关
4. 推理时无法获知测试图像的真实原语标签，只能用自身作为辅助图像，V-Adapter 的效果可能打折
5. 未探索动态图构建机制以处理开放词汇的原语发现

## 相关工作与启发

- CGE (Naeem 2021) 首次将图卷积网络引入 CZSL，但从头学习对齐而非利用 CLIP
- CAILA (Zheng 2024) 和 Troika (Huang 2024) 也在 CLIP 中插入适配器，但未利用跨组合信息
- 本文展示了组合图在 CZSL 中的巨大潜力：不仅是 L-Adapter 的基础结构，还通过共现矩阵指导 V-Adapter 的采样策略

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 跨组合解耦框架和双适配器设计是显著创新，尤其 PRG 采样策略非常巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三数据集、闭/开世界、多变体对比、适配器消融、插入位置/深度消融，非常系统
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清晰，图示设计用心，但数学符号和公式可以更紧凑
- **价值**: ⭐⭐⭐⭐ — 对 CZSL 领域有显著推动，双适配器思路可迁移到其他视觉语言任务

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](../../ICLR2026/others/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[ACL 2025\] ALGEN: Few-Shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment](algen_few-shot_inversion_attacks_on_textual_embeddings_via_cross-model_alignment.md)
- [\[ACL 2025\] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)
- [\[ACL 2025\] Synthia: Novel Concept Design with Affordance Composition](synthia_novel_concept_design_with_affordance_composition.md)

<!-- RELATED:END -->
