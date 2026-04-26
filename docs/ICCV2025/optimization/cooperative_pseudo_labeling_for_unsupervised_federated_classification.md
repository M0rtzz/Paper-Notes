---
title: >-
  [论文解读] Cooperative Pseudo Labeling for Unsupervised Federated Classification
description: >-
  [ICCV 2025][优化][无监督联邦学习] FedCoPL 首次将无监督联邦学习扩展到分类任务，通过协作伪标签策略（全局分配伪标签确保类别平衡）和部分 prompt 聚合协议（仅聚合视觉 prompt、保留文本 prompt 本地化）有效应对 CLIP 固有偏差和标签偏移挑战。
tags:
  - ICCV 2025
  - 优化
  - 无监督联邦学习
  - 伪标签
  - CLIP
  - 提示学习
  - 标签偏移
---

# Cooperative Pseudo Labeling for Unsupervised Federated Classification

**会议**: ICCV 2025  
**arXiv**: [2510.10100](https://arxiv.org/abs/2510.10100)  
**代码**: [https://github.com/krumpguo/FedCoPL](https://github.com/krumpguo/FedCoPL)  
**领域**: 优化 / 联邦学习  
**关键词**: 无监督联邦学习, 伪标签, CLIP, prompt tuning, 标签偏移

## 一句话总结

FedCoPL 首次将无监督联邦学习扩展到分类任务，通过协作伪标签策略（全局分配伪标签确保类别平衡）和部分 prompt 聚合协议（仅聚合视觉 prompt、保留文本 prompt 本地化）有效应对 CLIP 固有偏差和标签偏移挑战。

## 研究背景与动机

联邦学习（FL）是保护数据隐私的分布式学习范式。传统 FL 依赖各客户端有标注数据，但标注成本高昂且需要领域专家。无监督联邦学习（UFL）虽已有研究，但**此前的 UFL 方法仅聚焦于表示学习和聚类任务，无法直接用于分类**。

CLIP 等视觉语言模型的出现改变了这一格局——其强大的零样本分类能力降低了对标注数据的依赖。然而，将 CLIP 直接应用于 UFL 框架面临两个核心挑战：

### 挑战 1：CLIP 固有偏差 + 未知标签分布

CLIP 在 web-scale 数据上预训练，对某些类别存在偏好。直接用 CLIP 生成伪标签会导致：
- 偏好类别被过度选择，即使伪标签不正确
- 各客户端的真实标签分布未知（只有无标签数据）
- 全局训练严重失衡——某些类别被大量训练，某些被忽略

**例子**：假设客户端真实标签分布为 [0.1, 0.1, 0.8]，CLIP 生成的伪标签分布可能是 [0.6, 0.05, 0.35]——精度低且不能代表真实分布。

### 挑战 2：标签偏移导致聚合冲突

联邦场景下各客户端的标签分布不同（label skew），导致模型更新方向存在差异。简单平均聚合 prompt 参数会产生冲突，降低性能。

**切入角度**：
1. 让客户端估计本地伪标签分布 → 服务端全局重新分配 → 实现跨类别平衡
2. 利用视觉/文本 prompt 的特性差异——视觉 prompt 跨客户端相似（学通用图像表示），文本 prompt 差异大（编码类别特定知识）→ 只聚合视觉 prompt

## 方法详解

### 整体框架

FedCoPL 包含两大核心策略：
1. **协作伪标签（CoPL）**：客户端估计分布 → 上传服务端 → 全局调整 → 下发各客户端
2. **部分 prompt 聚合**：视觉 prompt 上传聚合，文本 prompt 本地保留

### 关键设计

#### 1. 协作伪标签策略（CoPL）

**第一步：本地分布估计**

每个客户端 $k$ 通过置信度和熵双重过滤，从无标签集 $D_k$ 中筛选出估计集：

$$D_k^{est} = \{(x, \hat{y}) \mid \max_c p_c(x) > \tau_1, \; Ent(p(x)) < \tau_2\}$$

其中 $\tau_1$ 和 $\tau_2$ 分别取置信度和熵的中位数。

**设计动机**：高置信度确保伪标签可靠，低熵确保预测不模糊。使用中位数作为动态阈值避免手动调参。

**第二步：统计上传**

客户端计算各类别的伪标签数量得到估计分布 $U_k^{est}$（C 维向量），上传至服务端。注意：上传的仅是统计信息，非原始数据，保护隐私。

**第三步：全局分配**

服务端按比例将 $M = \frac{\sum_k \sum_c u_{k,c}}{C}$（每类全局预算）分配给各客户端：

$$\tilde{u}_{k,c} = \left\lceil \frac{u_{k,c}}{\sum_{i \in K} u_{i,c}} \cdot M \right\rceil$$

**关键洞察**：使用比例分配而非绝对值——即使估计分布有噪声，比例关系仍能保持代表性。

**第四步：本地选择**

客户端 $k$ 按调整后的容量 $\tilde{U}_k$，为每类选择预测概率最高的样本构建训练集 $\tilde{D}_k$。

每隔 Q 轮（默认 Q=5）用最新模型重新执行伪标签生成-估计-分配-选择流程。

#### 2. 部分 prompt 聚合协议

作者通过实验发现了一个重要现象：**文本 prompt 在客户端间的差异显著大于视觉 prompt**。

用 drift diversity 和 cosine distance 两个指标度量发现：
- 视觉 prompt 差异小 → 学习的是通用图像表示
- 文本 prompt 差异大 → 编码了与本地标签分布相关的类别特定知识

基于此，提出：
- **视觉 prompt $P_k^v$**：上传至服务端做加权平均聚合
  $$P^v = \sum_{k=1}^K \frac{\tilde{n}_k}{\sum_i \tilde{n}_i} \cdot P_k^v$$
- **文本 prompt $P_k^t$**：保留在本地，实现个性化

聚合后的视觉 prompt 下发给各客户端作为下一轮的初始化。

**额外收益**：只传输视觉 prompt 还减少了 50% 的通信开销。

### 损失函数 / 训练策略

总目标函数：
$$\min_{P^v, \{P_k^t\}} \sum_{k=1}^K \mathbb{E}_{(x, \hat{y}) \in \tilde{D}_k} \ell_{ce}(g(P^v, P_k^t; x), \hat{y})$$

- 冻结 CLIP 骨干（ViT-B/32 或 ViT-B/16），仅训练 prompt
- SGD 优化器，学习率 0.1，余弦退火
- 20 轮通信，每轮 10 个 epoch 本地训练

## 实验关键数据

### 主实验

**Dirichlet-based 标签偏移（β=0.1），ViT/B-32 骨干**：

| 方法 | PL | DTD | RESISC45 | CUB | UCF101 | CIFAR10 | CIFAR100 | 平均 |
|------|-----|-----|----------|-----|--------|---------|---------|------|
| CLIP 零样本 | - | 43.24 | 54.51 | 51.28 | 61.00 | 86.93 | 64.17 | 60.19 |
| PromptFL + FPL | FPL | 45.79 | 59.76 | 47.29 | 64.39 | 87.51 | 63.26 | 61.33 |
| pFedPrompt + CPL | CPL | 44.22 | 61.76 | 47.23 | 65.59 | 90.26 | 65.63 | 62.45 |
| FedOPT + CPL | CPL | 37.43 | 51.10 | 46.61 | 58.93 | 91.08 | 57.87 | 57.17 |
| **FedCoPL** | **CoPL** | **60.89** | **75.76** | **56.09** | **73.20** | **95.38** | **73.59** | **72.49** |

FedCoPL 平均精度 **72.49%**，比最佳基线高约 **10%**。特别是在 RESISC45（75.76% vs 65.95%）和 UCF101（73.20% vs 65.59%）上优势巨大。

### 消融实验

| 置信度过滤 | 熵过滤 | 全局分配 | DTD | RESISC45 | CUB | UCF101 |
|----------|--------|---------|-----|----------|-----|--------|
| ✗ | ✗ | ✗ | 45.79 | 59.76 | 47.29 | 64.39 |
| ✗ | ✗ | ✓ | 46.35 | 69.60 | 51.98 | 65.31 |
| ✓ | ✓ | ✗ | (未单独评估) | - | - | - |
| **✓** | **✓** | **✓** | **60.89** | **75.76** | **56.09** | **73.20** |

**prompt 聚合策略消融**：

| 聚合文本 | 聚合视觉 | DTD | RESISC45 | CUB | UCF101 |
|---------|---------|-----|----------|-----|--------|
| ✗ | ✗（两者都保留本地） | 50.45 | 61.72 | 49.08 | 66.02 |
| ✓ | ✓（两者都聚合） | 47.34 | 61.68 | 49.08 | 64.97 |
| ✓ | ✗（仅聚合文本） | 55.31 | 67.12 | 51.76 | 69.53 |
| ✗ | **✓（仅聚合视觉）** | **60.89** | **75.76** | **56.09** | **73.20** |

仅聚合视觉 prompt 的策略效果最佳，验证了视觉/文本 prompt 特性差异的假设。

### 关键发现

1. **全局分配是核心**：全局分配策略在 RESISC45 上单独贡献约 +10%，解决了 CLIP 偏差导致的类别不平衡
2. **CoPL 可即插即用**：将 CoPL 替换到其他联邦框架中，FedOPT + CoPL 在 RESISC45 上提升 **17.09%**
3. **客户端数量增长时优势更大**：当客户端从 5 增至 50，基线方法性能大幅下滑，FedCoPL 保持稳定
4. **伪标签精度**：CoPL 的伪标签精度在 CIFAR10 上达到 **96.07%**（vs FPL 91.02%，CPL 92.18%）

## 亮点与洞察

- **首次将 UFL 扩展到分类**：开辟了新的研究方向，利用 VLM 的零样本能力弥补无标签的缺陷
- **比例分配优于绝对值**：即使估计不准确，比例信息仍能保持代表性——这是一个适用于更多场景的通用策略
- **隐私增强**：仅上传伪标签分布统计（而非精确类别分布），从 CLIP 预测派生，无法逆推真实数据分布
- **视觉/文本 prompt 差异的发现**：为联邦 prompt learning 提供了实证依据和设计原则

## 局限与展望

- 假设全局标签分布均匀（uniform）——在实际场景中可能不成立
- 未提供收敛性、隐私和公平性的理论分析（论文中提到作为未来工作）
- $\tau_1$、$\tau_2$ 使用中位数的默认值可能非最优（虽然实验表明对此不敏感）
- 尚未探索更大规模 VLM（如 GPT-4V）或更复杂任务（如分割、检测）

## 相关工作与启发

- **UPL / FPL** [Huang et al., 2022; Menghini et al., 2023]：中心化无监督 prompt learning 基线
- **CPL** [Zhang et al., 2024]：生成多候选伪标签提升精度
- **FedOPT** [Li et al., 2024]：用最优传输融合全局和本地文本 prompt
- **Orchestra** [Lubana et al., 2022]：无监督联邦表示学习中的全局聚类方法
- 启发：CLIP 的模态分离特性（视觉通用 vs 文本专用）可能在更多联邦场景中被利用

## 评分

- 新颖性：⭐⭐⭐⭐ — 首次将 UFL 扩展到分类，问题定义有开创性
- 技术深度：⭐⭐⭐⭐ — 全局分配策略和部分聚合协议设计合理
- 实验充分度：⭐⭐⭐⭐⭐ — 6 个数据集、两种偏移类型、多种基线、全面消融
- 实用性：⭐⭐⭐⭐ — 无需标签、通信高效、隐私保护，适合实际联邦场景

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit](learning_interpretable_queries_for_explainable_image_classification_with_informa.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](../../NeurIPS2025/optimization/deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](../../ICML2025/optimization/sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](../../NeurIPS2025/optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)

<!-- RELATED:END -->
