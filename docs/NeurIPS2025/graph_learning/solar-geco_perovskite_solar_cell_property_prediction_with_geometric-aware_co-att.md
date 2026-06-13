---
title: >-
  [论文解读] Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention
description: >-
  [NeurIPS 2025][图学习][钙钛矿太阳能电池] 提出Solar-GECO多模态框架，将钙钛矿吸收层的3D晶体结构通过几何GNN编码、器件其他层通过LLM文本嵌入编码，经共注意力融合后预测光电转换效率(PCE)及其不确定性，MAE从3.066降至2.936。 钙钛矿太阳能电池效率取决于多个层之间的耦合行为（吸收层、…
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "钙钛矿太阳能电池"
  - "几何图神经网络"
  - "多模态融合"
  - "共注意力机制"
  - "不确定性量化"
---

# Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention

**会议**: NeurIPS 2025  
**arXiv**: [2511.19263](https://arxiv.org/abs/2511.19263)  
**代码**: 暂无  
**领域**: 图学习  
**关键词**: 钙钛矿太阳能电池, 几何图神经网络, 多模态融合, 共注意力机制, 不确定性量化

## 一句话总结

提出Solar-GECO多模态框架，将钙钛矿吸收层的3D晶体结构通过几何GNN编码、器件其他层通过LLM文本嵌入编码，经共注意力融合后预测光电转换效率(PCE)及其不确定性，MAE从3.066降至2.936。

## 研究背景与动机

钙钛矿太阳能电池效率取决于多个层之间的耦合行为（吸收层、电子传输层ETL、空穴传输层HTL、背电极、基底），而非单一材料的本征性质。这带来两个核心问题：

**组合空间爆炸**：每层可选多种材料和工艺参数，组合数指数增长，传统实验筛选不可行

**现有ML方法的局限**：
   - 单材料性质预测（bandgap等）无法反映器件级的层间交互
   - 已有器件级方法（如Semantic GNN）仅用文本嵌入表示钙钛矿层，忽略了其至关重要的晶体几何结构

**关键洞察**：钙钛矿吸收层的3D原子排列（键长、角度、对称性）对器件效率有直接影响，但文本表示无法捕获这些信息。需要一种融合晶体几何特征与器件架构上下文的方法。

## 方法详解

### 整体框架

Solar-GECO是三阶段流水线：(1) 双模态特征提取——CGCNN处理晶体图、MaterialsBERT处理器件文本；(2) 共注意力融合模块——自注意力+交叉注意力迭代精炼；(3) 概率回归头——预测PCE均值和方差。

### 关键设计

1. **晶体图编码器（CGCNN）**：将钙钛矿吸收层的原子结构建模为图 $G = (\mathcal{V}, \mathcal{E})$

   节点 $\mathcal{V}$ 对应晶胞中的原子，边 $\mathcal{E}$ 基于截断半径内的原子间距建立。节点特征包含电负性、原子量等，边特征为原子间距离。预训练的CGCNN经图卷积迭代更新原子特征向量，输出 $\mathbf{H}_{\text{graph}} \in \mathbb{R}^{N \times d_{\text{node}}}$。
   
   设计动机：几何GNN尊重E(3)对称性，对旋转、平移和反射不变/等变，是物理系统的正确归纳偏置。

2. **器件文本编码器（MaterialsBERT）**：用预训练材料科学语言模型编码各功能层的化学描述

   对基底、ETL、HTL、背电极四层的文本字符串分别编码，提取[CLS]标记，堆叠为 $\mathbf{H}_{\text{text}} \in \mathbb{R}^{4 \times d_{\text{bert}}}$。
   
   设计动机：其余层无现成的晶体结构数据，文本嵌入能利用MaterialsBERT在大规模材料语料上学到的语义知识。

3. **共注意力融合模块**：自注意力+交叉注意力交替进行 $L$ 层

   **模态内自注意力**：先让每个模态内部元素相互关注
    $\mathbf{H}'_{\text{graph}}^{(l)} = \gamma\left(\mathbf{H}_{\text{graph}}^{(l-1)} + \text{MultiHead}(\mathbf{H}_{\text{graph}}^{(l-1)}, \mathbf{H}_{\text{graph}}^{(l-1)}, \mathbf{H}_{\text{graph}}^{(l-1)})\right)$

   **模态间交叉注意力**：原子表示query器件上下文，器件表示query晶体结构
    $\mathbf{H}_{\text{graph}}^{(l)} = \gamma\left(\mathbf{H}'_{\text{graph}}^{(l)} + \text{MultiHead}(\mathbf{H}'_{\text{graph}}^{(l)}, \mathbf{H}'_{\text{text}}^{(l)}, \mathbf{H}'_{\text{text}}^{(l)})\right)$

   设计动机：交叉注意力使模型学到哪些原子对特定器件层最相关，以及哪些层受特定晶体结构特征影响最大，实现双向信息流。

### 损失函数 / 训练策略

采用高斯负对数似然损失预测PCE的均值 $\mu(x)$ 和标准差 $\sigma(x)$：

$$\mathcal{L} = \frac{1}{2B} \sum_{i=1}^{B} \left(\log(\sigma_i^2) + \frac{(y_i - \mu_i)^2}{\sigma_i^2}\right)$$

第一项 $\log(\sigma^2)$ 防止模型预测过大方差来"逃避"精度要求；第二项为方差加权MSE。使用AdamW优化器，余弦学习率调度，冻结CGCNN编码器参数。

## 实验关键数据

### 主实验

| 模型 | R² Score↑ | MAE↓ | Spearman's ρ↑ |
|------|-----------|------|---------------|
| CrabNet | 0.2090 | 3.3655 | 0.3807 |
| BERT+MLP | 0.3863 | 3.0436 | 0.5944 |
| CGCNN+BERT+MLP | 0.4009 | 3.0111 | 0.6109 |
| Semantic GNN | 0.3907 | 3.0668 | 0.5943 |
| LLM Co-attention | 0.4048 | 2.9812 | 0.6120 |
| **Solar-GECO** | **0.4179** | **2.9361** | **0.6192** |

所有与Solar-GECO的差异均通过t检验达到统计显著(p<0.05)。

### 消融实验

| 消融项 | R² Score↑ | MAE↓ | 说明 |
|--------|-----------|------|------|
| MatSciBERT替换MaterialsBERT | 0.421 | 2.924 | 差异微弱 |
| CHGNet替换CGCNN | 0.394 | 3.032 | 几何GNN选择影响明显 |
| 门控注意力替换标准注意力 | 0.372 | 3.108 | 复杂机制过拟合 |
| MSE损失替换Gaussian NLL | 0.415 | 2.922 | NLL提升R²但MAE相当 |

### 不确定性校准

| 指标 | 值 | 说明 |
|------|----|------|
| PICP (95%置信) | 0.9593 | 与名义值95%仅差0.93% |
| 校准曲线 | 理论线在95%CI内 | σ预测良好校准 |

### 关键发现

1. CrabNet(仅成分)效果最差，证明多尺度器件预测不能只看组成
2. Solar-GECO vs CGCNN+BERT+MLP：共注意力融合相比简单拼接将MAE从3.011降至2.936，R²提升4.2%
3. 低PCE器件预测偏差更大，可能与训练数据中高效器件的主导地位有关
4. Group split（未见过的材料组合）下Solar-GECO仍优于基线，MAE从3.274降至3.127

## 亮点与洞察

1. **多模态设计精准**：对有晶体结构的层用几何GNN、对仅有文本描述的层用LLM，各取所长
2. **不确定性量化**：Gaussian NLL损失同时预测均值和方差，PICP校准精确，对实验筛选有实际指导价值
3. **跨尺度建模**：从原子级（晶体结构）到器件级（层间交互）的端到端建模

## 局限与展望

- 晶体结构依赖Materials Project，限制了钙钛矿配方多样性（465→34种）
- 未纳入制造工艺参数（退火温度、沉积方法等），而这些对效率有重要影响
- 低PCE区域预测偏差大，可通过自适应采样或重要性加权改善
- 共注意力层数和头数的选择缺乏理论指导

## 相关工作与启发

- **材料性质预测**: CGCNN, SchNet, Matformer
- **器件级预测**: Semantic Device Graphs
- **多模态融合**: Multimodal Transformer, 交叉注意力
- **不确定性建模**: Gaussian NLL, 混合密度网络

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将几何GNN晶体编码与器件级文本信息通过共注意力融合预测PCE
- 实验充分度: ⭐⭐⭐⭐ — 5个基线+4项消融+不确定性校准+group split分析
- 写作质量: ⭐⭐⭐⭐ — 方法动机清晰，图示信息量大
- 价值: ⭐⭐⭐⭐ — 可加速钙钛矿器件筛选，材料科学+AI交叉的实用贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](../../ICML2026/graph_learning/unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)

</div>

<!-- RELATED:END -->
