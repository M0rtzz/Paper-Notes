---
title: >-
  [论文解读] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data
description: >-
  [AAAI 2026][多任务学习] 提出MultiTab-Net——首个面向表格数据的多任务Transformer架构，通过多任务掩码注意力机制缓解任务竞争，在推荐、人口普查、物理等多个领域的数据集上显著超越现有MLP-based多任务模型和单任务Transformer模型。
tags:
  - AAAI 2026
  - 多任务学习
  - Transformer
  - 表格数据
  - 掩码注意力
  - 合成数据基准
---

# MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data

**会议**: AAAI 2026  
**arXiv**: [2511.09970](https://arxiv.org/abs/2511.09970)  
**代码**: [Armanfard-Lab/MultiTab](https://github.com/Armanfard-Lab/MultiTab)  
**领域**: 推荐系统 / 表格数据  
**关键词**: 多任务学习, Transformer, 表格数据, 掩码注意力, 合成数据基准

## 一句话总结

提出MultiTab-Net——首个面向表格数据的多任务Transformer架构，通过多任务掩码注意力机制缓解任务竞争，在推荐、人口普查、物理等多个领域的数据集上显著超越现有MLP-based多任务模型和单任务Transformer模型。

## 研究背景与动机

### 表格数据+多任务学习的空白

表格数据是世界上最丰富的数据类型，广泛应用于金融、医疗、电商等领域。许多场景天然需要同时预测多个相关目标：

**医疗**：同一份病历可以预测糖尿病和高血压风险

**电商**：不仅预测点击，还预测加购和购买

**推荐系统**：同时优化CTR和CVR

**多任务学习（MTL）**通过共享表示来利用任务间的相关性、提高泛化和效率。但现有的表格MTL工作存在两个问题：

**范围狭窄**：主要集中在大规模推荐系统（MMoE、PLE、STEM），对更广泛的表格场景探索不足

**骨干网络局限**：几乎都基于MLP，难以捕捉复杂的特征交互，且在数据丰富时扩展性差

### 为什么需要Transformer？

- MLP隐式地通过全连接层学习特征交互，缺乏显式的交互建模机制
- Transformer通过自注意力可以动态地建模特征间和样本间的依赖关系
- 在NLP和CV中，Transformer在大数据场景下的优势已被广泛验证
- 表格领域的Transformer（FT-T、SAINT）已显示出对特征间和样本间关系建模的能力

### 核心挑战：任务竞争

将Transformer扩展到多任务设置时，引入的任务token改变了注意力矩阵的结构。任务token之间的自由交互可能导致**"跷跷板现象"（seesaw phenomenon）**——主导任务抢占共享容量，损害整体性能。

## 方法详解

### 整体框架

MultiTab-Net的架构建立在SAINT的基础上，核心创新点：
1. **多token设计**：为每个任务分配独立的任务token（而非共享一个CLS token）
2. **多任务掩码注意力**：在特征间注意力中限制任务token的交互，缓解竞争

输入处理流程：
- $d$个特征 + $t$个任务token → 各自通过embedding网络映射为维度$e$的向量
- 拼接为$x\in\mathbb{R}^{(d+t)\times e}$ → 通过$N$个encoder块
- 每个encoder块包含：特征间注意力（Inter-Feature）+ 样本间注意力（Inter-Sample）+ FFN
- 最终$t$个任务token分别通过任务专属MLP输出预测

### 关键设计

#### 1. 多任务掩码注意力（Multitask Masked Attention）

**功能**：在特征间注意力中选择性地屏蔽某些token间的交互。

**核心思路**：在注意力矩阵$A_i$的预激活分数上加掩码$M_A$：

$$A_i = \text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}} + M_A\right)$$

$M_A$在被屏蔽位置加$-\infty$，softmax后变为0。

**三种候选掩码方案**：

| 方案 | 含义 | 效果 |
|------|------|------|
| F↛T | 特征token对任务token的注意力被屏蔽 | 不稳定，有时反降 |
| T↛T | 不同任务token之间的注意力被屏蔽 | **最佳**，一致最优 |
| F↛T & T↛T | 两者组合 | 次优 |

**设计动机**：T↛T屏蔽阻止任务token之间直接影响彼此，从而缓解任务竞争。但保留T→F（任务token可以注意特征）是必要的，因为任务需要从特征中提取信息。F→T被屏蔽后效果不稳定，说明特征token也需要感知任务上下文。

#### 2. 多token机制 vs 单token

**功能**：为每个任务分配独立的可学习任务token，而非使用BERT风格的共享CLS token。

**核心优势**（从消融实验中确认）：
- 任务数=2时，单/多token差异不大
- 任务数=8时（Higgs数据集），多token + T↛T掩码的$\Delta_m$=1.23%，而单token最优配置的$\Delta_m$=-6.35%
- 单个共享token无法充分捕捉多任务的任务特定信息

#### 3. MultiTab-Bench：合成多任务数据集生成器

**功能**：生成具有可调任务相关性、任务复杂度和任务数量的合成多任务表格数据。

**核心思路**：通过特征分解构造权重矩阵。给定期望的相关矩阵$\mathbf{P}$，进行特征分解$\mathbf{P}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$，构造权重矩阵$\mathbf{W}=\mathbf{Q}\boldsymbol{\Lambda}^{1/2}\mathbf{U}^T$，其中$\mathbf{U}$由正交单位向量构成。可以证明不同任务的权重向量的余弦相似度恰好等于$P_{ij}$。

标签生成：$y_i=\sum_{k=1}^{d_i}(\mathbf{w}_i^T\mathbf{x})^k+\epsilon_i$

**相比MMoE的合成数据**：
- 支持**任意数量**的任务（MMoE仅限2个）
- 支持不同任务使用不同**多项式度**（控制相对难度）
- 支持**任务特定噪声**

### 损失函数 / 训练策略

- 分类任务使用二元/多类交叉熵，回归任务使用RMSE
- 使用Adam优化器 + weight decay
- 早停策略：监控分类任务的平均AUC + 回归任务的平均EV
- 所有结果基于5个随机种子取平均

## 实验关键数据

### 主实验

| 模型 | AliExpress $\Delta_m$↑ | ACS Income $\Delta_m$↑ | Higgs $\Delta_m$↑ |
|------|----------------------|----------------------|-------------------|
| STL (基线) | 0.0000 | 0.0000 | 0.0000 |
| MTL | 0.1129 | 0.0612 | -0.6531 |
| MMoE | 0.0873 | 0.0893 | -0.3525 |
| PLE | 0.2778 | 0.0892 | -0.0314 |
| STEM | 0.1763 | 0.0725 | 0.0571 |
| SAINT | 0.1146 | 0.0948 | -1.6514 |
| **MultiTab-Net** | **0.5512** | **0.1064** | **1.2337** |

MultiTab-Net在所有数据集上都取得了最高的多任务增益。特别是在8任务Higgs数据集上，大部分模型的$\Delta_m$为负（多任务反而不如单任务），而MultiTab-Net取得了1.23%的正增益。

### 消融实验

| 配置 | AliExpress $\Delta_m$ | ACS Income $\Delta_m$ | Higgs $\Delta_m$ |
|------|----------------------|----------------------|-------------------|
| 单token, 无掩码 | 0.2669 | 0.0893 | -6.3491 |
| 多token, 无掩码 | 0.2579 | 0.0783 | 1.1182 |
| 多token, F↛T | 0.3698 | 0.0951 | 0.9626 |
| 多token, T↛T | **0.5512** | **0.1064** | **1.2337** |
| 多token, F↛T & T↛T | 0.2975 | 0.1007 | 1.0197 |

**关键发现**：T↛T掩码在所有数据集上一致最优。

### 计算效率

| 模型 | AliExpress (Params/FLOPs M) | ACS Income | Higgs |
|------|---------------------------|------------|-------|
| SAINT（最近的单任务Transformer） | 3.62/9.70 | 0.49/1.35 | 5.50/15.02 |
| STEM（最新MTL模型） | 1.55/3.11 | 0.69/1.29 | 1.25/2.51 |
| **MultiTab-Net** | **1.80/4.85** | **0.28/0.77** | **0.70/1.90** |

相比SAINT，MultiTab-Net在ACS Income和Higgs上分别实现约2×和8×的效率提升（约等于任务数）。

### 关键发现

1. 多token机制的优势随任务数增加而增大（2任务时差异小，8任务时差异巨大）
2. 合成数据实验验证了MultiTab-Net在不同任务相关性、任务复杂度和任务数量下的一致优势
3. 非均匀任务复杂度设置下，MultiTab-Net的优势更加明显

## 亮点与洞察

1. **填补空白**：首个表格数据的多任务Transformer，将注意力的优势带入MTL+tabular的交叉领域
2. **掩码设计简洁有效**：T↛T掩码的直觉很清楚——阻止任务互相干扰——且几乎不增加计算开销
3. **MultiTab-Bench有实用价值**：支持任意任务数、可调相关性和难度，为MTL研究提供了标准化的合成基准
4. **从STEM中汲取灵感**：STEM用stop-gradient约束后向传播中的跨任务更新，MultiTab-Net则在前向传播的注意力层面实现更直接的隔离

## 局限与展望

1. **数据集规模和多样性有限**：仅3个公开数据集，且任务类型主要是分类+回归，缺少如排序等更多样的任务
2. **与XGBoost的比较不够公平**：XGBoost的多任务支持有限，仅在输出类型相同时使用multioutput变体
3. **掩码策略是静态的**：T↛T一视同仁地屏蔽所有任务间交互，未考虑有些任务间的信息共享可能有益
4. **未探索动态掩码**：可根据任务相关性学习adaptive masking
5. **扩展性未验证**：任务数>8时的表现未知

## 相关工作与启发

- **MMoE (Ma et al. 2018)**：多专家门控混合架构，开创了表格MTL方向
- **PLE (Tang et al. 2020)**：共享+任务专属专家，缓解跷跷板效应
- **STEM (Su et al. 2024)**：stop-gradient约束，直接启发了MultiTab-Net的掩码设计
- **SAINT (Somepalli et al. 2021)**：样本间注意力，MultiTab-Net的架构基础
- 启发：**在注意力层面进行任务隔离**可能是多任务学习中的通用范式，值得在CV、NLP等领域推广

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个表格多任务Transformer，掩码设计虽简单但有效）
- 实验充分度: ⭐⭐⭐⭐ （3公开数据集+合成数据，但可以更多样）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，MultiTab-Bench部分的数学推导扎实）
- 价值: ⭐⭐⭐⭐ （填补空白，开源代码，合成基准有独立价值）

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](../../ICML2025/recommender/simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[AAAI 2026\] Probabilistic Hash Embeddings for Online Learning of Categorical Features](probabilistic_hash_embeddings_for_online_learning_of_categorical_features.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](../../ACL2026/recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)

<!-- RELATED:END -->
