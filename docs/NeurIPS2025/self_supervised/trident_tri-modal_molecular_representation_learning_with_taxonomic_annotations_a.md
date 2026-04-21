---
title: >-
  [论文解读] TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships
description: >-
  [NeurIPS 2025][自监督学习][分子属性预测] 提出 TRIDENT 三模态分子表示学习框架，引入层次分类标注（HTA）作为第三模态，结合体积对比损失做全局三模态对齐和功能团-文本局部对齐，通过动量机制动态平衡两者，在 18 个分子属性预测任务上达到 SOTA。
tags:
  - NeurIPS 2025
  - 自监督学习
  - 分子属性预测
  - 三模态对齐
  - 层次分类标注
  - 体积对比损失
  - 局部对齐
---

# TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships

**会议**: NeurIPS 2025  
**arXiv**: [2506.21028](https://arxiv.org/abs/2506.21028)  
**代码**: [GitHub](https://github.com/uta-smile/TRIDENT)  
**领域**: 多模态VLM / 分子表示学习  
**关键词**: 分子属性预测, 三模态对齐, 层次分类标注, 体积对比损失, 局部对齐

## 一句话总结

提出 TRIDENT 三模态分子表示学习框架，引入层次分类标注（HTA）作为第三模态，结合体积对比损失做全局三模态对齐和功能团-文本局部对齐，通过动量机制动态平衡两者，在 18 个分子属性预测任务上达到 SOTA。

## 研究背景与动机

分子表示学习旨在将化学结构映射为可计算的特征向量，是药物发现、虚拟筛选等领域的核心技术。多模态学习已被证明能通过整合结构、文本、功能信息来提升表示质量。

**现有方法的三大痛点**：

**忽略精细化分类标注**：现有方法只使用通用的功能描述文本，不同分类体系（如 LOTUS Tree 关注天然产物分类，MeSH Tree 关注医学功能）对同一分子有不同侧重，被粗暴忽略

**对齐方式局限**：现有方法依赖以单一模态为锚点的成对对齐（如 SMILES-Text），无法捕捉三模态间的高阶交互关系

**忽视局部对应关系**：大多数方法只做分子级全局对齐，忽略了功能团（如羟基、芳香环）与对应文本描述之间的细粒度关联

**核心 idea**：引入 HTA（Hierarchical Taxonomic Annotation）作为第三模态，使用体积对比损失实现三模态几何感知对齐，同时用局部对齐模块捕捉功能团与子文本的对应关系。

## 方法详解

### 整体框架

输入三模态数据 <SMILES, Text, HTA>。SMILES 通过分子编码器 $E_m$ 编码，文本描述和 HTA 共享文本编码器 $E_t$。三模态嵌入通过各自 MLP 投影到共享空间，再做全局对齐和局部对齐。

### 关键设计

1. **层次分类标注（HTA）模态构建**

    - 从 PubChem 获取分子信息，跨 32 种分类体系（LOTUS Tree、MeSH Tree 等）提取多层级功能标注
    - 使用 GPT-4o 将结构化标注合成为高保真、人类可读的 HTA 文本描述
    - 最终构建 47,269 个 <SMILES, Text, HTA> 三元组数据集
    - HTA 与传统描述互补：HTA 提供 32 个角度的多面信息（如化学来源、天然产物分类、医学功能），传统描述更直接突出核心特征

2. **基于体积的全局三模态对齐**

    - 不使用传统成对余弦相似度，而是计算三个归一化嵌入 $(m, t, h)$ 张成的平行体体积：
      $\text{Vol}(m,t,h) = \sqrt{1 - \langle m,t\rangle^2 - \langle m,h\rangle^2 - \langle t,h\rangle^2 + 2\langle m,t\rangle\langle t,h\rangle\langle h,m\rangle}$
    - 匹配三元组体积应小（三模态趋同），不匹配三元组体积应大
    - 双向损失：$\mathcal{L}_{M2TH}$（分子检索文本+HTA）和 $\mathcal{L}_{TH2M}$（文本+HTA检索分子），取均值

3. **功能团-文本局部对齐**

    - 使用 RDKit 从 SMILES 中提取显著功能团（85 种，如羟基、胺基、羧基、芳香体系）
    - 人工+GPT-4o 为每个功能团撰写高质量文本描述
    - 分别编码功能团和文本描述为嵌入，max-pooling 后做双向对比损失
    - 双向损失 $\mathcal{L}_{FG2T}$ + $\mathcal{L}_{T2FG}$

4. **动量机制动态平衡**

    - 整体损失：$\mathcal{L} = \alpha \mathcal{L}_g + (1-\alpha) \mathcal{L}_l$
    - $\alpha$ 通过指数移动平均动态更新：$\alpha_t = \beta \alpha_{t-1} + (1-\beta) \frac{\mathcal{L}_g^{(t)}}{\mathcal{L}_g^{(t)} + \mathcal{L}_l^{(t)}}$，动量参数 $\beta = 0.9$
    - 自动聚焦于当前损失更高的对齐目标

### 损失函数 / 训练策略

- 全局损失采用体积对比 + softmax 温度 $\tau$（可学习）
- 局部损失采用标准对比损失，共享温度参数
- Scaffold split 评估，三种随机种子取均值和标准差

## 实验关键数据

### 主实验（MoleculeNet 分类 ROC-AUC%）

| 方法 | BBBP | Tox21 | ToxCast | Sider | ClinTox | MUV | HIV | Bace | 平均 |
|------|------|-------|---------|-------|---------|-----|-----|------|------|
| MoleculeSTM | 70.75 | 75.71 | 65.17 | 63.70 | 86.60 | 65.69 | 77.02 | 81.99 | 73.33 |
| Atomas | 73.72 | 77.88 | 66.94 | 64.40 | 93.16 | 76.30 | 80.55 | 83.14 | 77.01 |
| **TRIDENT (M-M)** | **73.95** | **79.36** | **67.80** | 63.64 | 95.41 | **83.51** | **81.63** | 82.39 | **78.46** |

### 消融实验

| 配置 | Tox21 | ToxCast | BBBP | Bace |
|------|-------|---------|------|------|
| 完整 TRIDENT | **79.36** | **67.80** | **73.95** | **82.39** |
| w/o HTA | 下降明显 | 下降 | 下降 | 下降 |
| w/o 局部对齐 | 下降 | 下降 | 下降 | 下降 |
| w/o 体积损失（用标准对比） | 显著不稳定 | 下降 | 下降 | 下降 |
| 直接求和(Sum) vs 动量 | 77.79 vs 79.36 | 66.73 vs 67.80 | 72.15 vs 73.95 | 81.42 vs 82.39 |

### 关键发现

- **HTA 贡献最大**：去掉 HTA 后性能显著下滑，32 种分类体系的多视角标注为模型提供了传统文本无法给出的结构化语义
- **体积损失优于标准对比损失**：标准对比损失在三模态场景下不稳定，体积损失能捕捉高阶几何关系
- **动量机制优于固定权重**：动态平衡能在训练不同阶段自适应分配全局/局部优化力度

## 亮点与洞察

- **HTA 是关键创新**：将分子的多分类体系标注作为独立模态引入，比单一功能描述信息量大幅增加
- **体积对比损失的巧妙应用**：将 GRAM 的音视频文本三模态对齐框架首次扩展到分子领域，处理结构差异更大的模态组合
- **功能团级局部对齐**：85 个功能团配高质量文本描述的策划数据集，是可复用的宝贵资源
- **GPT-4o 辅助数据构建**：用 LLM 合成 + 领域专家审核的方式构建高质量标注，是现实可行的路线

## 局限与展望

- 分子毒性等属性不仅取决于分子结构，还与靶点和代谢物相关，当前框架未纳入这些因素
- 数据集规模偏小（47,269 个三元组），扩展到更大化学空间的扩展性待验证
- HTA 构建依赖 PubChem 的分类体系，对 PubChem 中信息不全的分子可能效果受限
- 功能团检测依赖 RDKit 的预定义模式，可能遗漏新型功能团

## 相关工作与启发

- **vs MoleculeSTM**：MoleculeSTM 仅做 SMILES-Text 双模态对比，TRIDENT 引入 HTA 第三模态和局部对齐，ROC-AUC 平均提升 5 个百分点
- **vs Atomas**：Atomas 有局部对齐但用静态注意力，TRIDENT 用动量机制和功能团级对齐效果更好
- **vs GRAM**：GRAM 首提体积对比损失用于音视频文本，TRIDENT 首次将其扩展到分子领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 三模态框架和HTA模态是新贡献，但各个技术组件（体积损失、LoRA）来自现有工作
- 实验充分度: ⭐⭐⭐⭐⭐ 18个下游任务、完整消融、多编码器配置
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，数据构建过程透明
- 价值: ⭐⭐⭐⭐ 对分子表示学习领域有明确推动，HTA数据集本身也有独立价值

<!-- RELATED:START -->

## 相关论文

- [SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](../../ECCV2024/self_supervised/scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions](../../ICLR2026/self_supervised/enhancing_molecular_property_predictions_by_learning_from_bond_modelling_and_int.md)
- [Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning](connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)

<!-- RELATED:END -->
