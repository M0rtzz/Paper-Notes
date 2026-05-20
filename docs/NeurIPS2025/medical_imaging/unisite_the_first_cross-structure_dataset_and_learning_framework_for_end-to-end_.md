---
title: >-
  [论文解读] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection
description: >-
  [NeurIPS 2025][医学图像][蛋白质结合位点检测] 提出首个以UniProt（唯一蛋白质）为中心的配体结合位点数据集UniSite-DS，以及首个端到端的结合位点检测框架UniSite，通过集合预测损失和双射匹配直接预测多个可能重叠的结合位点，同时引入IoU-based AP作为更准确的评估指标。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "蛋白质结合位点检测"
  - "端到端集合预测"
  - "UniProt中心数据集"
  - "匈牙利匹配"
  - "IoU评估指标"
---

# UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection

**会议**: NeurIPS 2025  
**arXiv**: [2506.03237](https://arxiv.org/abs/2506.03237)  
**代码**: [GitHub](https://github.com/quanlin-wu/unisite)  
**领域**: 医学图像 / 结构生物学  
**关键词**: 蛋白质结合位点检测, 端到端集合预测, UniProt中心数据集, 匈牙利匹配, IoU评估指标

## 一句话总结

提出首个以UniProt（唯一蛋白质）为中心的配体结合位点数据集UniSite-DS，以及首个端到端的结合位点检测框架UniSite，通过集合预测损失和双射匹配直接预测多个可能重叠的结合位点，同时引入IoU-based AP作为更准确的评估指标。

## 研究背景与动机

蛋白质配体结合位点检测是基于结构的药物设计中的基础步骤。然而现有方法和数据集面临三个关键问题：

**PDB中心的统计偏差**：现有数据集以单个蛋白质-配体复合物结构为中心，忽略了同一蛋白质在不同复合物中可能存在多种不同的结合位点。例如同一个UniProt ID (Q8WS26) 在PDBbind2020中只记录了1个结合位点，而在UniSite-DS中整合了13个代表性PDB条目的17个独特结合位点。这种数据构建方式引入了严重的统计偏差。

**不连续的工作流**：大多数方法采用先语义分割生成二值掩码、再聚类为离散结合位点的流程。这种碎片化管线高度依赖后处理方法（如聚类算法），限制了端到端优化，且难以处理重叠的结合位点。

**评估指标的局限性**：传统的DCC（预测中心与真实中心距离）和DCA指标存在两个根本缺陷：(a) 完全忽略结合位点的形状、大小和残基组成等关键结构属性；(b) 缺乏预测与真实值之间的适当匹配，可能导致重复计数。定量分析显示约20%的蛋白质在评估时受到重复计数的影响。

## 方法详解

### 整体框架

UniSite将结合位点检测形式化为集合预测任务：给定蛋白质序列 $S$（长度 $L$），目标是识别一组结合位点 $\{m_i^{gt}\}_{i=1}^{N_{gt}}$，每个结合位点由二值掩码 $m_i^{gt} \in \{0,1\}^L$ 表示。整体架构包括编码器、Transformer解码器和分割模块三部分，实现单次前向传播直接预测 $N$ 个可能重叠的结合位点。

### 关键设计

1. **UniSite-DS数据集构建**：利用AHoJ工具系统搜索PDB中所有蛋白质-配体相互作用，经过严格筛选（分辨率 < 2.5Å、晶体学方法、配体原子数 ≥ 5等），通过UniProt唯一标识符将同一蛋白质的所有结合位点映射到其序列上，并使用NMS（IoM阈值0.7、IoU阈值0.5）去除冗余。最终得到11,510个有效UniProt ID，其中3,670个含有多个结合位点，多位点数据量是现有数据集的4.81倍。

2. **基于双射匹配的集合预测损失**：模型单次前向推理固定大小的 $N$ 个预测 $z = \{(p_i, m_i)\}_{i=1}^N$。通过匈牙利算法计算预测集与真实集之间的最优匹配：

$$\hat{\sigma} = \arg\min_\sigma \sum_i^N \mathcal{L}_{\text{match}}(z_i^{gt}, z_{\sigma(i)})$$

匹配代价综合考虑分类概率和掩码相似度：$\mathcal{L}_{\text{match}} = -\mathbf{1}_{\{c_i^{gt}\neq\emptyset\}} \log p_{\sigma(i)}(c_i^{gt}) + \mathbf{1}_{\{c_i^{gt}\neq\emptyset\}} \mathcal{L}_{\text{mask}}$。设计动机来源于DETR的目标检测范式，通过双射匹配避免重复预测，实现真正的端到端检测。

3. **双路径编码器（序列+结构）**：

    - **序列编码器**：输入包括21种氨基酸的可学习嵌入、正弦位置编码和预训练ESM-2嵌入，经3层MLP生成初始特征后通过Transformer编码器层捕获残基间交互。
    - **结构编码器（可选）**：使用GearNet-Edge（E(3)-不变GNN），将蛋白质结构表示为残基级关系图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$，通过关系图卷积更新节点特征：$h_i^{(l)} = h_i^{(l-1)} + \text{ReLU}(\text{BN}(\sum_{r} W_r \sum_{j \in \mathcal{N}_r(i)} h_j^{(l-1)}))$。

4. **分割模块**：$N$ 个site query通过MLP转化为掩码嵌入 $\mathcal{E}_{\text{mask}} \in \mathbb{R}^{N \times d_{\text{model}}}$，与残基级特征 $\mathcal{F}$ 做点积后sigmoid得到掩码预测：$m_i[j] = \text{sigmoid}(\mathcal{E}_{\text{mask}}[i,:] \cdot \mathcal{F}[j,:]^T)$。

### 损失函数 / 训练策略

总损失结合分类损失和掩码损失：$\mathcal{L}_{\text{mask\&cls}} = \lambda_{\text{cls}} \sum_i^N -\log p_{\hat{\sigma}(i)}(c_i^{gt}) + \mathbf{1}_{\{c_i^{gt}\neq\emptyset\}} \mathcal{L}_{\text{mask}}$，其中掩码损失 $\mathcal{L}_{\text{mask}} = 5.0 \cdot \mathcal{L}_{\text{bce}} + 5.0 \cdot \mathcal{L}_{\text{dice}}$，分类损失权重 $\lambda_{\text{cls}} = 2.0$。对非结合类别的分类损失降权10倍缓解类不平衡。在每个解码器层后共享权重的分割模块进行中间监督。AdamW优化器，学习率 $10^{-4}$，权重衰减0.05。

IoU-based AP评估指标：先按置信度排序预测，将每个真实位点与IoU超过阈值的最高分预测匹配（一对一约束），计算插值精度-召回曲线下面积。

## 实验关键数据

### 主实验（UniSite-DS）

| 方法 | 类型 | AP₀.₃ ↑ | AP₀.₅ ↑ |
|------|------|---------|---------|
| Fpocket | 几何 | 0.1836 | 0.1017 |
| P2Rank | 机器学习 | 0.5056 | 0.2157 |
| DeepPocket | CNN | 0.4273 | 0.2334 |
| GrASP | GNN | 0.4469 | 0.2848 |
| VN-EGNN | GNN | 0.1621 | 0.0705 |
| **UniSite-1D** | **本文** | **0.5121** | **0.3033** |
| **UniSite-3D** | **本文** | **0.5603** | **0.3835** |

### 消融实验

| 配置 | AP₀.₃ | AP₀.₅ | 说明 |
|------|-------|-------|------|
| 序列相似度 < 0.9 | 0.5603 | 0.3835 | 默认设置 |
| 序列相似度 < 0.7 | 0.5579 | 0.3734 | 泛化能力良好 |
| 序列相似度 < 0.5 | 0.4677 | 0.2801 | 远缘蛋白质性能下降 |
| 16 queries | 0.5515 | 0.3795 | query数量影响不大 |
| 32 queries（默认） | 0.5603 | 0.3835 | — |
| 64 queries | 0.5615 | 0.3867 | 边际提升 |

### 关键发现

- UniSite-1D仅使用序列信息就超越了所有基线方法，证明了无结构结合位点检测的可行性
- VN-EGNN在DCC/DCA传统指标上表现良好，但在AP指标下表现极差（仅预测中心，丢失结构信息）
- IoU-based AP与传统指标排名一致但区分度更强：在HOLO4K-sc上DeepPocket和GrASP在DCA指标差异<0.01，但在AP₀.₃差异>0.10
- 约20%蛋白质在DCC/DCA评估时存在重复计数问题

## 亮点与洞察

- **数据视角的创新**：从PDB-centric到UniProt-centric，揭示了之前数据集的根本性统计偏差，这一思路可推广到其他生物分子数据集构建
- **DETR范式的迁移**：成功将目标检测中成熟的集合预测框架迁移到蛋白质结合位点检测，证明通用架构在生物学任务中的有效性
- **评估指标的反思**：通过严谨分析揭示DCC/DCA的根本缺陷，IoU-based AP的引入对领域有长远影响

## 局限与展望

- 数据集构建仍需人工检查来移除不合理条目，未来可开发自动化修复方法
- 模型设计未使用特化的特征工程，引入蛋白质表面特征等可能进一步提升性能
- 当前仅考虑氨基酸残基级别的结合位点，原子级别的精细预测有待探索

## 相关工作与启发

- 延续了DETR → MaskFormer的集合预测范式到生物学领域
- ESM-2预训练嵌入的使用体现了蛋白质语言模型在下游任务的价值
- 对药物发现中的虚拟筛选和从头分子设计具有直接应用价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个UniProt-centric数据集和端到端检测框架，三个贡献（数据集、方法、评估指标）均具开创性
- 实验充分度: ⭐⭐⭐⭐ 在自建数据集和经典数据集上全面评测，消融实验完整
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，三个issue的分析逻辑严谨
- 价值: ⭐⭐⭐⭐⭐ 数据集、方法和评估指标对结构生物学和药物发现领域具有长远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)
- [\[ICML 2025\] Protriever: End-to-End Differentiable Protein Homology Search for Fitness Prediction](../../ICML2025/medical_imaging/protriever_end-to-end_differentiable_protein_homology_search_for_fitness_predict.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[ACL 2026\] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](../../ACL2026/medical_imaging/decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)
- [\[NeurIPS 2025\] Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs](faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with.md)

</div>

<!-- RELATED:END -->
