---
title: >-
  [论文解读] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment
description: >-
  [AAAI 2026][图学习][多模态实体对齐] 提出 MyGram，通过模态感知图卷积扩散（MGD）模块捕获模态内的深层结构上下文信息，并引入基于Gram矩阵行列式的全局分布对齐损失（Gram Loss），在高维空间中强制跨模态语义一致性，实现更鲁棒的多模态实体对齐。 问题背景 多模态知识图谱（MMKG）通过整合文本、图…
tags:
  - "AAAI 2026"
  - "图学习"
  - "多模态实体对齐"
  - "知识图谱"
  - "Gram矩阵"
  - "图扩散学习"
  - "Transformer"
---

# MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment

**会议**: AAAI 2026  
**arXiv**: [2601.11885](https://arxiv.org/abs/2601.11885)  
**代码**: [https://github.com/HubuKG/MyGram](https://github.com/HubuKG/MyGram)  
**领域**: 图学习 / 知识图谱  
**关键词**: 多模态实体对齐, 知识图谱, Gram矩阵, 图扩散学习, Transformer

## 一句话总结

提出 MyGram，通过模态感知图卷积扩散（MGD）模块捕获模态内的深层结构上下文信息，并引入基于Gram矩阵行列式的全局分布对齐损失（Gram Loss），在高维空间中强制跨模态语义一致性，实现更鲁棒的多模态实体对齐。

## 研究背景与动机

### 问题背景
多模态知识图谱（MMKG）通过整合文本、图像等多种模态增强实体的语义表示。然而，不同来源的MMKG对同一现实实体往往具有不一致的表示。**多模态实体对齐（MMEA）** 旨在识别不同MMKG中指代同一现实对象的等价实体，是知识融合的核心任务。

### 现有方法的两大挑战

**对比学习的局限性**：现有方法主要采用模态内对比学习框架，通过优化正负实体对的特征距离来对齐。但这些方法**忽视了全局特征空间中模态间的分布差异**，仅关注局部点对点对齐，无法保证跨模态特征的全局一致性

**浅层特征干扰**：现有方法忽视了每种模态内的**结构上下文信息**，导致模型难以区分外观相似但本质不同的实体。典型案例：Anne Hathaway 和 Kirsten Dunst 在视觉和属性特征上高度相似，会对对齐产生干扰，但利用结构信息仍可实现准确对齐

### 核心思路
- 通过图扩散学习获取富含结构上下文的模态特征（解决浅层特征干扰）
- 通过Gram矩阵构建的高维平行体体积作为几何指标约束跨模态分布一致性（解决全局对齐缺失）

## 方法详解

### 整体框架

MyGram 由三大模块组成：
1. **多模态特征提取**：从不同模态独立提取单模态嵌入
2. **模态感知扩散学习**：通过图卷积扩散获取富含结构上下文的模态特征
3. **多模态训练与学习**：使用 Gram Loss 建立等价实体间的对齐

### 关键设计

#### 1. **多模态特征提取**：为每种模态独立构建嵌入

- **结构模态**：使用关系反射图注意力网络（RRGAT）聚合邻居，保留关系结构信息：$\mathbf{h}_g = RRGAT(\omega, \mathbf{M}_g, x_g)$
- **关系/属性/视觉模态**：通过线性变换投影到共享特征空间：$\mathbf{h}_m = \mathbf{W}_m x_m + b_m, \quad m \in \{r, a, v\}$
    - 属性和关系使用词袋特征表示
    - 视觉使用预训练图像编码器（VGG-16）提取特征

#### 2. **模态感知图卷积扩散（MGD）模块**：捕获深层结构上下文

**设计动机**：传统方法忽视邻居实体的模态信息，仅使用浅层特征容易被相似但不同的实体干扰。MGD对每种模态独立进行多跳邻域信息聚合。

**图卷积扩散过程**：
- 构建带自环的归一化邻接矩阵：$\hat{A} = D^{-1/2}(A+I)D^{-1/2}$
- 迭代传播（$k$ 轮），带残差连接防止过平滑：

$$\mathbf{H}_m^{(l)} = \beta \cdot \hat{A}\mathbf{H}_m^{(l-1)} + \alpha \cdot \mathbf{H}_m^{(0)}, \quad (l=1,2,...,k)$$

- 最终输出经归一化和Dropout：$\mathbf{H}_m = \text{Dropout}\left(\frac{1}{\gamma}\mathbf{H}_m^{(k)}\right)$，其中 $\gamma = \beta^k + \alpha \sum_{c=0}^{k-1}\beta^c$ 防止梯度爆炸

**Transformer自注意力融合**：
- 对扩散后的模态特征应用多头交叉注意力：

$$head_m^i = \beta_m^{(i)} V_m^{(i)}, \quad \beta_m = \text{softmax}\left(\frac{Q_m^\top K_m}{\sqrt{d_h}}\right)$$

- 计算跨模态权重进行自适应融合：

$$\omega_m = \frac{\exp\left(\sum_{j \in M}\sum_{i=0}^{N_h}\beta_{m,j}^{(i)} / \sqrt{|M| \times N_h}\right)}{\sum_{k \in M} \exp\left(\sum_{k \in M}\sum_{i=0}^{N_h}\beta_{m,k}^{(i)} / \sqrt{|M| \times N_h}\right)}$$

- 联合嵌入：$\mathbf{H}_o = \mathbf{H}_g \oplus_{m \in M}[\omega_m \mathbf{H}_m]$

#### 3. **Gram-based 全局分布对齐**：基于高维体积的几何约束

**核心思想**：利用多模态向量构成的4维平行体的体积作为跨模态一致性的几何指标。体积越小，表示嵌入位于更紧凑的子空间，跨模态语义一致性越强。

**具体实现**：
- 先通过相似度矩阵选择 Top-K 候选实体
- 用源实体结构特征和目标实体的视觉/属性/关系特征构建多模态矩阵：$\mathcal{M} = [\tilde{\mathbf{H}}_g^s, \tilde{\mathbf{H}}_v^t, \tilde{\mathbf{H}}_a^t, \tilde{\mathbf{H}}_r^t] \in \mathbb{R}^{d_h \times 4}$
- 计算 Gram 矩阵 $G = \mathcal{M}^\top \mathcal{M} \in \mathbb{R}^{4 \times 4}$
- 4维平行体体积：$Vol = \sqrt{|\det(G)| + \epsilon}$
- Gram Loss（稀疏对比损失）：

$$\mathcal{L}_{Gram} = -\frac{1}{M}\sum_{m=1}^{M} \log \frac{\exp(-Vol^{(m,p)}/\tau)}{\sum_{k=1}^{K} \exp(-Vol^{(m,k)}/\tau)}$$

**与传统方法的区别**：传统方法优化点对点特征距离（局部），Gram Loss约束多模态向量在高维空间的整体几何关系（全局），促进跨模态语义结构一致性。

### 损失函数 / 训练策略

总损失为InfoNCE对比损失 + 加权Gram损失：

$$\mathcal{L}_{total} = \mathcal{L}_{InfoNCE} + \lambda \mathcal{L}_{Gram}$$

其中InfoNCE最大化真实对齐实体对的相似度并分离负样本：

$$\mathcal{L}_{InfoNCE} = \sum_{(e_i,e_j) \in \mathcal{S}} -\log \frac{\exp(\text{sim}(e_i, e_j)/\mathcal{T})}{\sum_{e_k \in \mathcal{N}_i^{neg}} \exp(\text{sim}(e_i, e_k)/\mathcal{T})}$$

## 实验关键数据

### 实验设置
- **数据集**：
    - 跨知识图谱：FB15K-DB15K、FB15K-YG15K（种子比例：20%/50%/80%）
    - 双语：DBP15K（ZH-EN、JA-EN、FR-EN，种子比例30%）
- **指标**：Hits@1、Hits@10、MRR
- **图像特征**：VGG-16，$d_v = 4096$
- **隐层维度**：300，自注意力头数5，Transformer中间层400

### 主实验

| 数据集 | 指标 | MyGram | 次优方法 | 提升 |
|--------|------|--------|---------|------|
| FBDB15K (80%) | Hit@1 | **0.842** | IBMEA: 0.821 | +2.6% |
| FBDB15K (80%) | MRR | **0.879** | SimDiff: 0.865 | +1.6% |
| FBYG15K (80%) | Hit@1 | **0.783** | PMF: 0.756 | +3.6% |
| FBYG15K (20%) | Hit@1 | **0.629** | SimDiff: 0.530 | **+18.7%** |
| DBP15K ZH-EN | Hit@1 | **0.833** | DESAlign: 0.810 | +2.8% |
| DBP15K JA-EN | Hit@1 | **0.836** | DESAlign: 0.811 | +3.1% |
| DBP15K FR-EN | Hit@1 | **0.869** | DESAlign: 0.826 | **+5.2%** |

在 FBDB15K 上 Hit@1 最大提升 4.8%，FBYG15K 上 9.9%，DBP15K 上 4.3%。

### 消融实验

| 配置 | FBDB15K MRR | FBDB15K Hit@1 | FBYG15K MRR | FBYG15K Hit@1 | 说明 |
|------|-------------|---------------|-------------|---------------|------|
| MyGram (full) | 0.879 | 0.842 | 0.836 | 0.783 | 完整模型 |
| w/o Relation | 0.842 | 0.822 | 0.811 | 0.761 | **关系模态移除影响最大** |
| w/o Attributes | 0.859 | 0.834 | 0.818 | 0.768 | 属性贡献显著 |
| w/o Image | 0.851 | 0.829 | 0.824 | 0.772 | 视觉信息有帮助 |
| w/o MGD | 显著下降 | 显著下降 | 显著下降 | 显著下降 | 模态扩散模块关键 |
| w/o Gram | 下降 | 下降 | 下降 | 下降 | Gram Loss有效 |

### 关键发现

1. **关系模态最重要**：移除关系信息导致最大性能下降，说明结构信息在多模态实体对齐中起核心作用
2. **低资源场景优势显著**：在5%-30%种子比例的低资源实验中，MyGram始终保持对MEAformer和SimDiff的优势
3. **案例研究**：在"上海/Shanghai"实体对齐中，MEAformer和PMF对正确实体排名较低，MyGram能准确匹配，证明其捕获深层信息的能力
4. **MGD模块贡献大于Gram Loss**：模态图卷积扩散对性能的影响更为显著

## 亮点与洞察

1. **几何视角的全局对齐**：Gram矩阵行列式作为多模态一致性指标非常巧妙——体积为零意味着所有模态向量线性相关（完美一致），体积越大越不一致。这比逐对比较提供了更全局的约束
2. **图扩散 + Transformer的互补性**：图扩散捕获局部结构上下文，Transformer捕获跨模态全局依赖，两者互补
3. **模态权重的自适应计算**：通过注意力分数的归一化来确定各模态权重，比手动设定更灵活
4. **实验覆盖全面**：5大数据集、9组对比实验、低资源分析、案例研究

## 局限与展望

1. **使用VGG-16作为图像编码器**相对过时，可尝试CLIP等更强的多模态编码器
2. **4维平行体是硬编码设计**，如果模态数量变化需要重新设计
3. **隐层维度固定300**可能限制了模型对复杂语义的建模能力
4. **未探索文本模态的深度表示**（仅用词袋特征表示属性和关系）
5. **可改进方向**：引入LLM增强文本理解（论文已提到）、探索动态知识图谱场景中的增量对齐

## 相关工作与启发

- **SimDiff**使用扩散增强对齐，**IBMEA**用信息瓶颈抑制虚假线索，但都停留在局部对齐层面
- Gram矩阵的体积约束思想可以迁移到**多模态检索、跨模态生成质量评估**等场景
- 图扩散学习的模态感知设计思路可应用于**多模态推荐**和**社交网络分析**

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Gram Loss的几何视角新颖，但MGD部分较常规
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5数据集、9组实验、模态消融+组件消融+低资源+案例
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，但公式符号偶有不一致
- **实用价值**: ⭐⭐⭐⭐ — 开源代码，方法可行，但特征提取器偏老旧

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](../../ACL2026/graph_learning/ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ICLR 2026\] Relational Graph Transformer](../../ICLR2026/graph_learning/relational_graph_transformer.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)

</div>

<!-- RELATED:END -->
