---
title: >-
  [论文解读] SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation
description: >-
  [NeurIPS 2025][图学习][旅行推荐] 提出SPOT-Trip框架，首次系统研究异地旅行推荐问题，通过知识图谱增强的静态偏好学习、神经ODE驱动的动态偏好学习以及偏好融合模块，在两个真实数据集上最高提升17.01%。
tags:
  - NeurIPS 2025
  - 图学习
  - 旅行推荐
  - 知识图谱
  - 神经ODE
  - 时间点过程
  - 静态-动态偏好融合
---

# SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation

**会议**: NeurIPS 2025  
**arXiv**: [2506.01705](https://arxiv.org/abs/2506.01705)  
**代码**: 暂无  
**领域**: 图学习  
**关键词**: 旅行推荐, 知识图谱, 神经ODE, 时间点过程, 静态-动态偏好融合

## 一句话总结

提出SPOT-Trip框架，首次系统研究异地旅行推荐问题，通过知识图谱增强的静态偏好学习、神经ODE驱动的动态偏好学习以及偏好融合模块，在两个真实数据集上最高提升17.01%。

## 研究背景与动机

当用户从家乡出发去陌生城市旅行时，需要推荐一条包含多个中间景点的行程。这个问题面临三大挑战：

**数据稀疏**：用户在异地通常只有极少甚至没有历史签到记录，传统推荐模型无法有效学习

**偏好二元性**：用户偏好包含稳定的静态兴趣（长期行为模式）和随情境变化的动态偏好（时间、位置敏感），现有方法将二者混为一谈

**偏好转移**：家乡签到偏好与异地行为可能存在漂移（interest drift），如何有效迁移家乡知识是关键

**核心创新点**：首次将异地旅行推荐形式化为"给定起点、终点和停留数，推荐中间POI序列"的任务，并显式解耦静态/动态偏好。

## 方法详解

### 整体框架

SPOT-Trip包含三大模块：知识增强静态偏好学习（KSPL）、基于ODE的动态偏好学习（ODPL）、静态-动态偏好融合。三个模块通过联合损失端到端训练。

### 关键设计

1. **知识增强静态偏好学习（KSPL）**

   构建POI属性知识图谱 $\mathcal{G}_k = (v, r, e)$，编码POI与属性实体之间的语义关系（如"Balboa Park → Located in → San Diego"）。通过关系感知注意力聚合机制生成增强的POI嵌入：

   $$\bar{\mathbf{v}} = \mathbf{v} + \sum_{e \in \mathcal{N}_v} \alpha(e, r_{e,v}, v) \mathbf{e}$$

   其中注意力权重 $\alpha$ 编码实体-关系特定的语义关联。静态偏好对齐通过MLP将家乡偏好映射到异地：

   $$\bar{\mathbf{P}}^o = \phi(\mathbf{W}_S \bar{\mathbf{u}}^h + \mathbf{b}_S)$$

   用欧氏距离损失桥接推断偏好与真实异地偏好：$\mathcal{L}_S = \sum_u \|\bar{\mathbf{P}}^o - \bar{\mathbf{u}}^o\|_2^2$

   设计动机：KG提供的外部语义知识可以缓解异地数据稀疏，同时静态/空间信息分开处理避免冲突交互。

2. **基于ODE的动态偏好学习（ODPL）**

   使用神经ODE建模用户动态偏好在连续时间上的漂移：

   $$\frac{d\tilde{\mathbf{p}}_t^o}{dt} = f(\tilde{\mathbf{p}}_{t_1}^o)$$

   结合时间点过程描述每个行为发生的瞬时概率，通过非齐次泊松过程的强度函数 $\lambda(\cdot)$ 刻画：

   $$t_n \sim \text{NHPP}(\lambda(\tilde{\mathbf{p}}_t^o))$$

   使用变分推断估计潜在初始状态的后验，通过Transformer编码器聚合家乡行为序列获得变分参数。动态学习损失基于ELBO：

   $$\mathcal{L}_D = -\sum_{u \in \mathcal{U}} \left[\underbrace{\text{重构行为对数似然}}_{(i)} + \underbrace{\text{时间点过程对数似然}}_{(ii)} - \underbrace{\text{KL散度}}_{(iii)}\right]$$

   设计动机：签到数据的时间采样不规则，神经ODE天然处理连续时间建模，点过程可量化行为发生概率。

3. **静态-动态偏好融合**

   将查询表示、静态偏好和动态偏好序列拼接，通过非线性预测头计算每个POI的logit：

   $$z_{u,n,i} = \phi(\mathbf{W}_R [\mathbf{Q}^o \| \bar{\mathbf{P}}^o \| \tilde{\mathbf{P}}^o] + \mathbf{b}_R)$$

### 损失函数 / 训练策略

联合优化三个损失：

$$\mathcal{L} = \beta_1 \mathcal{L}_S + \beta_2 \mathcal{L}_D + \beta_3 \mathcal{L}_R$$

推荐损失 $\mathcal{L}_R$ 为交叉熵。推荐阶段使用Top-p采样生成中间POI序列，用归一化位置代替实际时间戳。

## 实验关键数据

### 主实验

| 方法 | Foursquare F₁ | Foursquare PairsF₁ | Yelp F₁ | Yelp PairsF₁ |
|------|-------------|-------------------|---------|-------------|
| AR-Trip | 0.0304 | 0.0045 | 0.0307 | 0.0153 |
| Base+KDDC | 0.0375 | 0.0079 | 0.0341 | 0.0156 |
| Base+CNN-ODE | 0.0367 | 0.0094 | 0.0326 | 0.0168 |
| **SPOT-Trip** | **0.0400** | **0.0109** | **0.0399** | **0.0190** |
| 提升幅度 | +6.67% | +15.96% | +17.01% | +13.90% |

### 消融实验

| 变体 | Yelp F₁ | Yelp PairsF₁ | 说明 |
|------|---------|-------------|------|
| SPOT-Trip | 0.0399 | 0.0190 | 完整模型 |
| w/o KS | ↓10.02% | ↓21.79% | 去除知识增强静态偏好 |
| w/o OD | 性能下降 | 性能下降 | 去除ODE动态偏好 |
| w/o SI | 显著下降 | 显著下降 | 去除空间信息 |

### 数据稀疏实验

| 数据比例 | SPOT-Trip F₁ (Yelp) | Base+KDDC F₁ | Base+CNN-ODE F₁ |
|---------|--------------------|--------------| ----------------|
| 40% | 0.0327 | 0.0321 | 0.0313 |
| 60% | 0.0343 | 0.0331 | 0.0327 |
| 80% | 0.0384 | 0.0338 | 0.0333 |
| 100% | 0.0399 | 0.0341 | 0.0326 |

### 关键发现

1. 结合家乡信息的方法普遍优于不使用家乡信息的旅行推荐基线，验证了偏好迁移的必要性
2. KSPL模块去除后PairsF₁急剧下降21.79%，说明语义知识图谱对序列推荐至关重要
3. 即使只有40%家乡数据，SPOT-Trip仍优于基线用100%数据的效果
4. 仅给定起点或终点时，模型仍能生成合理推荐（如预测的终点对应火车站），展示了内在的旅行模式理解能力

## 亮点与洞察

1. **问题定义新颖**：首次系统化定义异地旅行推荐为给定起终点生成中间POI序列
2. **偏好分离设计**：静态偏好用KG语义建模，动态偏好用神经ODE连续时间建模，避免混合冲突
3. **TransE交替训练**：KG嵌入与推荐模型交替优化，增强多关系语义空间

## 局限与展望

- 知识图谱构建依赖数据集自带的属性信息，对属性不丰富的场景适用性有限
- 神经ODE在推荐阶段缺乏实际时间戳，使用归一化位置作为代理可能产生偏差
- Top-p采样策略在序列生成中可能引入不确定性

## 相关工作与启发

- **异地推荐**: KDDC, CNN-ODE, PPROC
- **旅行推荐**: GraphTrip, AR-Trip, MatTrip
- **神经ODE应用**: Neural ODE在时空任务中的连续建模优势
- **知识增强推荐**: KG-based POI推荐

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次定义异地旅行推荐问题并提出完整框架
- 实验充分度: ⭐⭐⭐⭐ — 两个数据集、9个基线、消融+稀疏+参数敏感性分析
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，模块设计逻辑递进
- 价值: ⭐⭐⭐⭐ — 旅行推荐的实际应用价值高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)
- [\[ACL 2025\] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](../../ACL2025/graph_learning/kg_rag_recommendation.md)
- [\[CVPR 2025\] Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](../../CVPR2025/graph_learning/unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)
- [\[ICML 2025\] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](../../ICML2025/graph_learning/toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)

<!-- RELATED:END -->
