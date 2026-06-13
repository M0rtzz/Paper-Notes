---
title: >-
  [论文解读] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction
description: >-
  [AAAI 2026][医学图像][空间组学] 提出空间多组学框架GROVER，通过KAN-GCN编码器捕获非线性空间-特征依赖、spot-feature-pair对比学习对齐异构模态、以及自适应混合专家（MoE）动态路由过滤低质量信号，在四个真实空间组学数据集上实现了优于现有方法的聚类性能。 空间分辨转录组学和空间蛋白质组…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "空间组学"
  - "多模态融合"
  - "图卷积网络"
  - "混合专家模型"
  - "对比学习"
---

# GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction

**会议**: AAAI 2026  
**arXiv**: [2511.11730](https://arxiv.org/abs/2511.11730)  
**代码**: [GitHub](https://github.com/Xubin-s-Lab/GROVER)  
**领域**: 计算生物学 / 空间多组学融合  
**关键词**: 空间组学, 多模态融合, 图卷积网络, 混合专家模型, 对比学习

## 一句话总结

提出空间多组学框架GROVER，通过KAN-GCN编码器捕获非线性空间-特征依赖、spot-feature-pair对比学习对齐异构模态、以及自适应混合专家（MoE）动态路由过滤低质量信号，在四个真实空间组学数据集上实现了优于现有方法的聚类性能。

## 研究背景与动机

空间分辨转录组学和空间蛋白质组学分别被Nature评为2021年和2024年的年度方法，这些技术将单细胞分析扩展到空间维度，提供了前所未有的组织结构洞察。然而，空间多模态组学分析面临一个核心挑战：如何有效整合来自不同模态（转录组、蛋白质组、组织学图像）的特征，生成一致的低维表示用于下游分析（如空间域识别、细胞类型注释等）。

现有方法存在三个关键痛点：（1）大多数方法仅整合转录组和蛋白质组数据，忽略了组织学图像提供的关键形态学上下文。SpatialGlue、PRAGA等方法都只处理双模态。虽然MISO尝试引入组织学图像，但采用简单的外积交互，效果有限。（2）现有框架在所有空间位置对所有模态一视同仁，忽略了数据质量的巨大差异。实际中空间组学数据常受技术噪声（如单细胞测序中的dropout事件）和生物/实验伪影（如组织切片误差）的影响，不同spot的不同模态信噪比差异很大。（3）组学数据和组织学图像之间存在显著的语义鸿沟，且图像patch和测序spot之间是多对多映射关系，精确跨模态对齐极为困难。

本文的核心方案：设计一个三层策略——用KAN-GCN提取空间感知的模态表示，用spot级别的对比学习桥接模态语义鸿沟，用自适应MoE在每个spot动态选择可靠的模态信号。

## 方法详解

### 整体框架

GROVER处理三种模态输入：转录组（RNA）、蛋白质组（ADT）和组织学图像。对每种模态构建两个图：基于空间坐标的空间图 $\mathcal{G}_S$ 和基于特征相似度的模态特征图 $\mathcal{G}_F^{(m)}$。通过KAN-GCN编码后用注意力加权融合为统一表示，再经过spot级对比学习对齐跨模态语义，最后由自适应MoE动态融合，输出统一嵌入 $Z$ 用于聚类等下游分析。

### 关键设计

1. **基于KAN的图卷积编码器（KAN-GCN）**:

    - 功能：替代标准GCN中的固定线性变换，增强图卷积的表达能力
    - 核心思路：传统GCN的层传播为 $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$，其中 $W^{(l)}$ 是固定线性变换。GROVER用Kolmogorov-Arnold Network（KAN）替换这个线性变换，每一层包含可训练的单变量函数矩阵 $\varphi_{q,p}^{(l)}$，变换后的特征计算为 $\mathcal{F}^{(l)}(H^{(l)})_{i,q} = \sum_{p=1}^{d_l}\varphi_{q,p}^{(l)}(H^{(l)}_{i,p})$，节点更新变为 $H^{(l+1)} = \sigma(\hat{A}\cdot\mathcal{F}^{(l)}(H^{(l)}))$
    - 设计动机：KAN网络能以核函数方式学习非线性变换，比标准GCN的线性投影更能捕获空间组学数据中复杂的非线性模式（如基因表达与空间位置之间的非线性关系）

2. **空间-特征注意力融合**:

    - 功能：为每种模态自适应地融合空间图和特征图的编码
    - 核心思路：对spot $i$ 的模态 $m$，空间图编码 $e_i^S$ 和特征图编码 $e_i^{F(m)}$ 分别通过共享线性变换计算注意力分数 $e_i^{(t)} = \mathbf{q}^\top \tanh(\mathbf{W}e_i^{(t)} + \mathbf{b})$，然后softmax归一化后加权求和：$\tilde{e}_i^{(m)} = \alpha_i^{(S)}e_i^S + \alpha_i^{(F)}e_i^{F(m)}$
    - 设计动机：空间图编码物理邻近关系，特征图编码功能相似性，两者对不同spot的重要性不同，需要自适应权重

3. **Spot-Feature-Pair对比学习**:

    - 功能：在模态融合前对齐不同模态的语义表示
    - 核心思路：采用带掩码的双向InfoNCE损失。首先计算每个模态内部的余弦相似度矩阵 $S_{i,j}^{(m)}$，构建二元掩码 $M^{(m)}$ 排除高度相似的负样本（避免将生物学上相似但标签不同的spot作为困难负例），然后对RNA-ADT、RNA-Image、ADT-Image三个模态对分别计算双向对比损失：$\mathcal{L}_{contrast}^{m_1,m_2} = \frac{1}{2}(\ell_{masked}(\tilde{E}^{(m_1)}, \tilde{E}^{(m_2)}, M^{(m_1)}) + \ell_{masked}(\tilde{E}^{(m_2)}, \tilde{E}^{(m_1)}, M^{(m_2)}))$
    - 设计动机：不同模态（尤其是组学和图像）的数据分布和语义差距巨大。掩码机制解决了空间组学特有的问题——相邻spot可能在生物学上高度相似，如果作为负例会产生大量假负例干扰训练

4. **自适应混合专家（Self-adaptive MoE）**:

    - 功能：在每个spot级别动态调整各模态的贡献权重
    - 核心思路：三个模态的对齐嵌入取平均得到门控输入 $x_i = \frac{1}{3}(\hat{e}_i^{(R)} + \hat{e}_i^{(A)} + \hat{e}_i^{(I)})$，门控网络输出归一化权重 $\beta_i^{(m)}$。关键创新是引入阈值过滤：当 $\beta_i^{(m)} < \gamma$ 时将该模态权重置零，$\gamma=0.3$。过滤后重新归一化，每个模态通过专用FFN专家处理后加权求和：$z_i = \sum_m s_i^{(m)} \cdot h_i^{(m)}$。极端情况下所有模态都低于阈值时，使用置信度最高的单一模态
    - 设计动机：空间组学数据中不同spot的各模态质量波动很大——某些spot可能因测序dropout导致RNA信号极差，或因切片误差导致图像失真。MoE可以在这些spot自动降低不可靠模态的影响

### 损失函数 / 训练策略

总体训练目标为三个模态的重建损失加上跨模态对比损失的加权和：$\mathcal{L}_{total} = \sum_{m}\mathcal{L}_{rec}^{(m)} + \lambda\sum_{m_i \neq m_j}\mathcal{L}_{contrast}^{m_i, m_j}$，其中 $\lambda=2$。重建损失通过图解码器利用空间邻接结构从融合嵌入 $Z$ 恢复各模态特征。模型在双NVIDIA RTX A5000 GPU上训练，300 epochs内收敛。

## 实验关键数据

### 主实验（四个数据集上的聚类性能）

| 数据集 | 指标 | GROVER | MISO | SpatialGlue | COSMOS |
|--------|------|--------|------|-------------|--------|
| Human Tonsil | ARI (%) | **45.2±7.8** | 41.3±6.7 | 43.3±6.7 | 19.8±6.7 |
| Human Tonsil | SC (%) | **31.6±3.9** | 7.0±1.6 | 23.8±3.2 | 20.0±0.7 |
| Human Tonsil | CHI | **2494.4±285.5** | 244.4±14.6 | 1063.6±123.6 | 937.4±99.6 |
| Human Breast Cancer | ARI (%) | **44.1±10.7** | 37.5±3.0 | 43.0±6.9 | 25.6±2.2 |
| Human Breast Cancer | SC (%) | **36.3±7.7** | 11.0±0.6 | 20.2±0.8 | 24.8±0.8 |
| Human Glioblastoma | ARI (%) | 40.8±6.6 | **43.5±6.9** | 40.1±7.6 | 32.0±6.9 |
| Human Glioblastoma | NMI (%) | **53.9±4.1** | 49.2±2.2 | 53.8±7.3 | 48.6±4.3 |
| Tonsil Add-on | ARI (%) | **46.5±5.6** | 44.6±11.9 | 45.3±7.3 | 24.6±4.3 |
| Tonsil Add-on | SC (%) | **38.2±1.2** | 8.3±0.5 | 21.4±1.1 | 18.4±2.5 |

### 消融实验（Human Tonsil with Add-on Antibodies）

| 配置 | ARI (%) | NMI (%) | SC (%) | 说明 |
|------|---------|---------|--------|------|
| GROVER (完整) | 46.5±5.6 | 59.0±4.8 | 38.2±1.2 | 全部组件启用 |
| w/o MoE | 42.5±4.3 | 56.8±3.0 | 21.8±1.2 | 移除专家路由，用简单求和替代，ARI降4.0% |
| w/o $\mathcal{L}_{contrast}$ | 45.5±7.2 | 57.8±4.3 | 21.6±2.6 | 移除对比损失，SC降16.6% |
| w/o KAN-GCN | 42.7±6.7 | 55.9±5.2 | 52.6±1.1 | 用标准GCN替代，监督指标下降但SC反而上升 |

### 关键发现
- GROVER在SC（轮廓系数）上的优势最为显著——在Human Tonsil上比MISO高出24.6个百分点，说明其融合嵌入的聚类结构更清晰
- 双模态方法SpatialGlue在部分数据集上竟然优于三模态方法MISO，说明简单地增加模态并均匀融合反而可能引入噪声，佐证了自适应融合的必要性
- MoE移除后性能显著下降，验证了按spot动态加权的重要性
- KAN-GCN替换为标准GCN后，无监督指标（SC、DBI）反而改善但监督指标（ARI、NMI）下降——说明KAN的非线性建模在有标签评价下更有优势，但可能导致嵌入空间不够"光滑"
- $\gamma=0.3$（接近专家数的倒数 $1/3$）是最佳阈值——过低则无法过滤噪声模态，过高则过度依赖单模态丢失互补信息

## 亮点与洞察
- 将KAN引入GCN是一个新颖的组合，用可训练的非线性函数替代固定线性变换，本质上让每条边的消息传递都有独立的非线性变换能力
- spot级别的MoE设计精确匹配了空间组学的数据特点——不同空间位置的数据质量差异巨大，全局统一权重不合理
- 对比学习中的掩码机制巧妙地处理了空间数据特有的"生物学相似假负例"问题
- 框架高度模块化，可无缝替换任何SOTA病理学基础模型（如OmiCLIP），具有良好的可扩展性

## 局限与展望
- 实验仅在聚类任务上验证，未展示在生存预测、细胞类型注释等更多下游任务上的表现（尽管标题提到"cancer survival prediction"，但论文实验部分并未涉及生存分析）
- KAN-GCN在无监督指标上不如标准GCN，暗示其可能有过拟合风险或嵌入空间不够规整
- 四个数据集全部来自10x Genomics平台，跨平台泛化能力未经验证
- 当前仅处理三种模态（RNA、ADT、Image），对更多组学类型（如表观组学、代谢组学）的扩展需要进一步验证
- MoE的gate网络相对简单（单层线性），更复杂的门控可能带来进一步提升

## 相关工作与启发
- SpatialGlue采用双注意力GNN融合转录组和蛋白质组，是当前双模态SOTA
- MISO通过外积交互引入组织学图像，但均匀融合策略限制了性能
- MoE思想源自经典的混合专家模型，在NLP（如Switch Transformer）中广泛使用，本文将其引入空间组学融合是一个很好的跨领域迁移
- KAN（Kolmogorov-Arnold Network）是最近热门的网络结构，本文是较早将其用于GCN中的工作之一

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[CVPR 2025\] Distilled Prompt Learning for Incomplete Multimodal Survival Prediction](../../CVPR2025/medical_imaging/distilled_prompt_learning_for_incomplete_multimodal_survival_prediction.md)
- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)

</div>

<!-- RELATED:END -->
