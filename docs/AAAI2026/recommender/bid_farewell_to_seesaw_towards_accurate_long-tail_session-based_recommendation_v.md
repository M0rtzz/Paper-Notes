---
title: >-
  [论文解读] Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents
description: >-
  [AAAI 2026 Oral][推荐系统][会话推荐] 提出HID框架，通过属性感知的谱聚类构建混合意图来区分会话相关与无关的尾部物品，并设计针对长尾和准确性的双约束损失（ICLoss），实现长尾推荐与准确性的"双赢"，打破传统方法中两者此消彼长的"跷跷板"困境。 问题背景 会话推荐（Session-based Recom…
tags:
  - "AAAI 2026 Oral"
  - "推荐系统"
  - "会话推荐"
  - "长尾分布"
  - "混合意图"
  - "谱聚类"
  - "对比学习"
---

# Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.08378](https://arxiv.org/abs/2511.08378)  
**作者**: Xiao Wang, Ke Qin, Dongyang Zhang, Xiurui Xie, Shuang Liang  
**代码**: 未公开  
**领域**: 推荐系统  
**关键词**: 会话推荐, 长尾分布, 混合意图, 谱聚类, 对比学习  

## 一句话总结

提出HID框架，通过属性感知的谱聚类构建混合意图来区分会话相关与无关的尾部物品，并设计针对长尾和准确性的双约束损失（ICLoss），实现长尾推荐与准确性的"双赢"，打破传统方法中两者此消彼长的"跷跷板"困境。

## 研究背景与动机

### 问题背景
会话推荐（Session-based Recommendation, SBR）旨在根据匿名用户的短期交互序列预测其下一个交互物品。在实际推荐场景中，物品的曝光频率呈现严重的长尾分布——少量头部物品占据绝大多数交互，大量尾部物品极少被推荐。这种不平衡导致推荐系统倾向于反复推荐头部物品，形成恶性循环，大幅降低推荐多样性。

### 已有工作的两类技术路线
现有长尾SBR方法分为两大类：（1）**增强类方法**（如LOAM、MelT、LLM-ESR），通过增强尾部物品的嵌入表示或在生成会话嵌入时强调尾部物品；（2）**重排序类方法**（如TailNet、CSBR、LAP-SR），根据交互会话预测头尾物品分布并直接调整最终排名结果。

### 核心痛点
这两类方法存在两个共性缺陷：

**无差别强调尾部物品引入噪声**：并非所有尾部物品都与当前会话相关。例如一个以"图书"为主题的会话中，"服装"类的尾部物品虽属于长尾，但对该会话而言是噪声。已有方法不区分相关/无关的尾部物品，一律提升尾部曝光，导致推荐准确性下降。

**缺乏显式长尾监督信号**：大多数方法仍依赖交叉熵损失间接优化，增强和重排策略往往与交叉熵优化目标冲突，导致长尾性能与准确性之间的"跷跷板效应"——提升一方必然损害另一方。

在Tmall数据集上，GRU4Rec加上TailNet等长尾方法后，tCov@20有所提升但HR@20明显下降，清晰展示了这种跷跷板现象。

### 本文的核心思路
解决上述矛盾需要回答两个问题：如何有效区分噪声？如何同时为长尾和准确性提供显式监督？HID的回答是：通过**混合意图建模**捕获用户高层偏好，将物品划分为会话目标意图和噪声意图；然后通过**双约束损失**分别约束目标意图内的头尾一致性（促长尾）和会话与噪声意图的距离（保准确）。

## 方法详解

### 整体架构
HID是一个模型无关的即插即用框架，可接入任意SBR模型（如GRU4Rec、STAMP、SRGNN、GCE-GNN）。它由两个核心模块组成：混合意图学习模块和意图约束损失。

### 模块一：混合意图学习（Hybrid Intent Learning）

传统意图挖掘方法仅考虑单个会话内的时序关系，容易受噪声干扰且忽略跨会话的意图一致性。HID提出属性感知的谱聚类，分三步构建全局混合意图：

**Step 1 — 初步意图单元**：利用物品的属性信息（如商品类别、音乐流派）作为初步意图单元。共享同一属性的物品被归入同一初步意图组$c'_i$。

**Step 2 — 初步意图图构建**：将每个会话中的物品ID替换为属性ID，遍历所有会话统计属性间的共现频率，构建初步意图图$\mathcal{G}=(\mathcal{P}, \mathcal{E}, \mathcal{W})$。图的节点是属性，边权重为共现频率。例如"食物"与"锅具"在购物会话中频繁共现，它们之间的边权重就很高。

**Step 3 — 谱聚类生成混合意图**：对初步意图图计算归一化拉普拉斯矩阵$L = I - D^{-1/2}WD^{-1/2}$，取最小的$q$个特征值对应的特征向量，然后对特征向量矩阵的行向量执行K-means聚类，将属性重新归入$n$个簇。同一簇中的属性合并形成一个混合意图。例如"食物"和"锅具"可能被聚合为"烹饪"这一混合意图。混合意图的嵌入通过其包含的所有物品嵌入的均值池化获得：$\mathbf{c}_i = \frac{1}{|c_i|}\sum_{v_j \in c_i} \mathbf{v}_j$。

值得注意的是，整个混合意图构建过程可以**离线预计算**，训练和推理时仅需查表检索，不增加在线开销。

### 目标意图与噪声意图的定义

对于会话$S^u$，给定下一物品$v_{l+1}^u$（训练时已知的ground truth）：
- **目标意图** $\mathcal{C}^u$：包含$v_{l+1}^u$的混合意图集合
- **噪声意图** $\hat{\mathcal{C}}^u$：同一batch中其他会话的目标意图中不在$\mathcal{C}^u$内的部分

目标意图和噪声意图仅在训练阶段的ICLoss中作为监督信号使用，不存在数据泄露风险。

### 模块二：意图约束损失（Intent Constraint Loss）

在获取混合意图嵌入后，HID对会话嵌入和意图嵌入先做$L_2$归一化，投影到单位超球面上确保度量空间一致性。然后施加两个约束：

**约束一：长尾约束（Constraint for Long-tail）**

核心思想：最小化会话到目标意图内所有物品的距离方差。头部物品嵌入通常离会话更近，尾部物品更远；约束方差后迫使两者的距离趋于一致，使尾部物品获得与头部相当的推荐概率。

$$\min\ \mathcal{L}_l = \text{Var}_{v_i \in \mathcal{C}^u}[d(\mathbf{S}^u, \mathbf{v}_i)]$$

直接计算方差的复杂度为$O(Nd)$。论文通过Theorem 1证明：最小化该方差在优化等价意义上等同于最小化会话嵌入到目标意图中心的距离$d(\mathbf{S}^u, \mathbf{c}^u)$，复杂度降至$O(d)$。

**约束二：准确性约束（Constraint for Accuracy）**

核心思想：最大化会话到噪声意图的平均距离，同时约束方差避免极端情况。

$$\max\ \mathcal{L}_a = \mathbb{E}_{c^v \in \hat{\mathcal{C}}^u} d(\mathbf{S}^u, \mathbf{c}^v), \quad \text{s.t.}\ \text{Var}_{c^v \in \hat{\mathcal{C}}^u}(d(\mathbf{S}^u, \mathbf{c}^v)) < \eta$$

**统一损失函数**

两个约束被统一为一个类InfoNCE形式的损失。论文通过Theorem 2证明该损失近似等价于带固定margin=2的(N-1)-Triplet Loss。为了适应不同场景，引入灵活系数$\sigma$替换固定margin，并将方差硬约束转化为惩罚项$p^u$。最终ICLoss为：

$$\mathcal{L}_c = -\sum_{S^u \in \mathcal{B}} \log \frac{\mathbf{X}}{(1+\lambda p^u)(\mathbf{X}+\mathbf{Y})}$$

其中$\mathbf{X} = \exp(\cos(\mathbf{S}^u, \mathbf{c}^u)/\sigma)$，$\mathbf{Y} = \sum_{c^v \in \hat{\mathcal{C}}^u}\exp(\cos(\mathbf{S}^u, \mathbf{c}^v)/\sigma)$。

总训练损失为：$\mathcal{L} = \mathcal{L}_p + \epsilon \mathcal{L}_c$，其中$\mathcal{L}_p$为原始SBR模型的交叉熵损失。

## 实验关键数据

### 表1：主实验结果 — STAMP和GRU4Rec基线（HR@20 / tCov@20）

| 方法 | Tmall HR↑ | Tmall tCov↑ | Diginetica HR↑ | Diginetica tCov↑ | Retailrocket HR↑ | Retailrocket tCov↑ |
|------|-----------|-------------|----------------|------------------|-------------------|---------------------|
| STAMP (base) | 26.10 | 69.46 | 50.15 | 90.71 | 50.54 | 53.70 |
| + TailNet | 20.61 ↓ | 71.33 ↑ | 45.39 ↓ | 91.23 ↑ | 47.00 ↓ | 51.56 ↓ |
| + LOAM | 24.31 ↓ | 71.68 ↑ | 46.19 ↓ | 89.96 ↓ | 50.27 ↓ | 55.67 ↑ |
| + LAP-SR | 25.21 ↓ | 72.11 ↑ | 49.87 ↓ | 91.32 ↑ | 49.59 ↓ | 55.32 ↑ |
| **+ HID** | **28.26 ↑** | **73.65 ↑** | **50.39 ↑** | **93.05 ↑** | **52.38 ↑** | **56.02 ↑** |
| GRU4Rec (base) | 19.69 | 49.60 | 50.23 | 84.97 | 45.01 | 69.98 |
| **+ HID** | **25.13 ↑** | **63.21 ↑** | **52.23 ↑** | **90.73 ↑** | **48.89 ↑** | **73.21 ↑** |

关键发现：已有长尾方法几乎全部以牺牲准确性为代价换取长尾提升（跷跷板效应），而HID在所有4个基线×3个数据集组合上同时提升了准确性和长尾性能。GRU4Rec+HID相比base的Tmall HR@20提升27.6%（19.69→25.13），tCov@20提升27.4%（49.60→63.21）。

### 表2：消融实验（STAMP + SRGNN，HR@20 / tCov@20）

| 变体 | Tmall HR | Tmall tCov | Diginetica HR | Diginetica tCov | Retailrocket HR | Retailrocket tCov |
|------|----------|------------|---------------|-----------------|-----------------|-------------------|
| STAMP+HID | **28.26** | **73.65** | **50.39** | **93.05** | **52.38** | **56.02** |
| w/o HI | 27.43 | 69.29 | 50.17 | 91.96 | 51.75 | 55.31 |
| w/o FC | 26.77 | 70.20 | 49.76 | 92.15 | 50.89 | 55.67 |
| SRGNN+HID | **28.38** | **66.40** | **52.09** | **96.02** | **53.45** | **55.75** |
| w/o HI | 27.48 | 61.00 | 51.96 | 92.94 | 53.10 | 54.01 |
| w/o FC | 27.36 | 62.92 | 51.16 | 93.56 | 52.80 | 55.11 |

去除混合意图（HI）对多样性影响更大，去除灵活系数（FC）对准确性影响更大。Tmall上混合意图的影响尤为显著（tCov: 73.65→69.29），因为Tmall会话更长，意图漂移更频繁，精确的目标意图建模更关键。

## 亮点与洞察

- **概念创新**：首次明确提出"跷跷板效应"的根因是对尾部物品的无差别强调引入了会话无关噪声，并通过意图建模提供了解决思路
- **理论扎实**：Theorem 1将$O(Nd)$的方差最小化等价归约为$O(d)$的距离最小化；Theorem 2将统一损失近似为带margin的Triplet Loss，理论推导完整
- **即插即用**：HID可无缝接入任意SBR模型（序列型、图型均可），且混合意图构建可离线完成，不增加推理开销
- **双赢验证充分**：在4个基线模型×3个数据集×6个指标的所有组合上，HID均实现了准确性和长尾性能的同步提升，p值普遍<0.001

## 局限与展望

- **依赖物品属性信息**：混合意图构建需要物品的类别属性作为初步意图单元，在缺乏属性元数据的场景下适用性受限（尽管附录中有使用语义聚类替代的实验）
- **谱聚类簇数需调参**：最优簇数$n$因数据集而异（Tmall=4, Diginetica=3），缺乏自适应选取策略
- **仅验证表格型SBR**：未在基于Transformer的现代SBR模型（如SASRec、BERT4Rec）上验证
- **目标意图依赖ground truth**：训练时目标意图通过下一物品$v_{l+1}^u$定义，这限制了方法在半监督或弱监督场景的扩展

## 与相关工作的对比

- **TailNet / CSBR / LAP-SR（重排序类）**：直接修改排名结果，缺乏对噪声的判别，HID通过意图建模从根源区分会话相关/无关物品
- **LOAM / MelT / LLM-ESR（增强类）**：增强尾部嵌入同样不区分噪声，HID的双约束损失在提升尾部覆盖的同时显式约束噪声远离会话
- **意图建模工作（ICL, MISAR, STP）**：现有意图挖掘仅从单个会话的滑动窗口或局部子图提取，忽略跨会话一致性；HID的混合意图通过全局属性共现关系构建，更稳健
- **对比学习推荐（CL4SRec等）**：利用正负样本对学习表示，但不针对长尾问题；HID的ICLoss本质上是一种面向长尾的对比学习变体

## 评分

- 新颖性: ⭐⭐⭐⭐ — 混合意图+双约束的组合思路新颖，对跷跷板效应的归因分析有说服力
- 实验充分度: ⭐⭐⭐⭐ — 4基线×3数据集的全面验证，消融和超参分析完整，统计检验严谨
- 写作质量: ⭐⭐⭐⭐ — 问题动机阐述清晰，理论推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 提出的即插即用框架实用性强，对长尾推荐领域有明确的方法论贡献
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](../../ACL2026/recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)

</div>

<!-- RELATED:END -->
