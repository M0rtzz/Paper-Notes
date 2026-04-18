---
title: >-
  [论文解读] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data
description: >-
  [ICCV 2025][优化][联邦学习] 提出 FED-PRIME，一个面向多模态数据模态缺失场景的联邦 Prompt-Tuning 框架，通过 inter-client 和 intra-client 两组 prompt 分别捕获跨客户端可对齐的缺失模式和客户端内特有的缺失模式，并通过聚类-对齐机制进行服务端聚合，在多种缺失数据设置下大幅超越现有基线。
tags:
  - ICCV 2025
  - 优化
  - 联邦学习
  - 提示学习
  - 多模态
  - 缺失模态
  - 异构数据
---

# Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data

**会议**: ICCV 2025  
**arXiv**: [2602.07081](https://arxiv.org/abs/2602.07081)  
**代码**: [github.com/hangpt01/FedPrime](https://github.com/hangpt01/FedPrime)  
**领域**: 联邦学习 / 多模态优化  
**关键词**: 联邦学习, Prompt-Tuning, 多模态, 缺失模态, 异构数据

## 一句话总结

提出 FED-PRIME，一个面向多模态数据模态缺失场景的联邦 Prompt-Tuning 框架，通过 inter-client 和 intra-client 两组 prompt 分别捕获跨客户端可对齐的缺失模式和客户端内特有的缺失模式，并通过聚类-对齐机制进行服务端聚合，在多种缺失数据设置下大幅超越现有基线。

## 研究背景与动机

### 领域现状

大型预训练模型的微调（Fine-tuning）已成为主流范式。Prompt-Tuning 作为参数高效的微调方法，通过在输入前添加可学习的 prompt token 来适配下游任务。联邦学习（FL）允许多设备在不共享数据的情况下协同训练模型。

### 现有痛点

**联邦 Prompt-Tuning 仅支持单模态**：现有方法假设所有客户端的数据模态相同，无法处理多模态场景

**多模态联邦学习不利用预训练模型**：现有多模态 FL 方法（FedMSplit、FedMAC 等）使用定制架构，不能通过微调预训练的多模态基础模型（如 CLIP、ViLT）来获益

**模态缺失的双重异构性**：
   - **Intra-heterogeneity（客户端内异构）**：单个数据集内不同样本有不同的缺失模态
   - **Inter-heterogeneity（客户端间异构）**：不同客户端有不同的模态缺失分布模式

**朴素聚合失效**：不同客户端的 prompt 可能因偏向不同的缺失模式而无法直接平均，简单 FedAvg 会将 prompt 坍缩为低信息量的表示

### 核心矛盾

多模态联邦学习中，各客户端的模态缺失模式不同，导致学到的 prompt 编码了不同的信息模式。直接聚合这些异构的 prompt 会导致信息冲突和性能退化。需要一种能识别、对齐、聚合编码了相似缺失模式的 prompt 的机制。

## 方法详解

### 整体框架

FED-PRIME 基于预训练的 ViLT 模型，每个客户端维护两组可学习的 prompt 集合（inter-client 和 intra-client），通过输入自适应机制选择最相关的 prompt 子集。服务端对 intra-client prompt 使用标准 FedAvg 聚合，对 inter-client prompt 使用基于聚类的对齐-聚合机制。

### 关键设计

#### 1. **双 Prompt 集合设计**

- **功能**：将微调知识分解为两组 prompt，分别编码不同类型的缺失模式信息
- **核心思路**：

**Inter-client prompts** $\mathbf{w}_p^{inter} = \{\mathbf{p}_1^{inter}, \ldots, \mathbf{p}_\tau^{inter}\}$：编码输入级别的缺失数据分布模式，可跨客户端对齐和聚合

**Intra-client prompts** $\mathbf{w}_p^{intra} = \{\mathbf{p}_1^{intra}, \ldots, \mathbf{p}_\tau^{intra}\}$：编码与输入无关的缺失模态模式（如仅缺图像 vs 仅缺文本），可直接 FedAvg 聚合

- **设计动机**：聚合机制反向约束了知识编码方式。如果与输入级别模式相关的知识错误地编入 intra-prompt，会被 FedAvg 平均掉；如果通用知识错误地编入 inter-prompt，会浪费其表达带宽。这种分离设计通过隐式的梯度信号自动实现知识的正确分配

#### 2. **输入自适应 Prompt 检索**

- **功能**：为每个输入样本从两组 prompt 中分别选择最相关的 $\kappa$ 个 prompt 作为微调指令
- **核心思路**：学习 key 函数 $k(\mathbf{p})$ 和 query 函数 $q(\mathbf{x}(M))$，通过余弦距离 $d(\mathbf{x}(M), \mathbf{p}) = \cos(q(\mathbf{x}(M)), k(\mathbf{p}))$ 衡量相关性。局部损失函数加入正则项：

$$L'_t(\mathbf{w}) = \sum_{s=1}^m \ell(F(\mathbf{x}(M_{t,s}); \mathbf{w}'), z_{t,s}) + \sum_{s=1}^m r(\mathbf{x}(M_{t,s}), \mathbf{w}'_p)$$

其中 $r(\mathbf{x}(M), \mathbf{w}'_p) = \sum_{\mathbf{p} \in \mathbf{w}'_p} d(\mathbf{x}(M), \mathbf{p})$ 惩罚选中的 prompt 与输入的距离

- **设计动机**：不同样本有不同的缺失模式，需要不同的 prompt 指令。正则项确保 prompt 不会被过载——每个 prompt 只负责与其"邻近"的样本模式，从而实现知识的蒸馏和分离

#### 3. **服务端聚类-对齐聚合**

- **功能**：识别跨客户端中编码了相似缺失模式的 inter-client prompt，将它们聚类并合并为更综合的 prompt
- **核心思路**：将对齐问题形式化为带约束的聚类优化任务：

$$\min_{\boldsymbol{\alpha}, \boldsymbol{\theta}, \gamma} G(\boldsymbol{\alpha}, \boldsymbol{\theta}, \gamma) + R(\boldsymbol{\alpha}, \zeta)$$

其中 $\alpha_t^{p,q} \in \{0,1\}$ 表示客户端 $t$ 的第 $p$ 个 prompt 是否匹配到第 $q$ 个聚类，$\boldsymbol{\theta}_q$ 为聚类中心（即聚合后的 prompt）。约束确保同一客户端的 prompt 不被分到同一聚类。$R(\boldsymbol{\alpha}, \zeta)$ 通过可学习的流行度函数 $U(\boldsymbol{\theta}_q; \zeta)$ 优先更新更通用的 prompt。使用 Hungarian 算法求解离散优化子问题。

- **设计动机**：同一位置的 inter-client prompt 在不同客户端可能编码了完全不同的缺失模式（因为某些模式在某些客户端不存在），朴素的位置对齐会导致不兼容 prompt 的混合。聚类机制按语义相似性对齐而非按位置对齐

### 损失函数 / 训练策略

- 主模型：冻结的 ViLT + 可学习的 prompt 集合 + 分类头
- 客户端更新：最小化 $L'_t(\mathbf{w})$（含正则项的局部损失）
- 服务端聚合：inter-prompt 通过聚类-对齐算法聚合，intra-prompt 通过 FedAvg 聚合
- 交替优化：(1) 固定 $\boldsymbol{\alpha}$ 优化 $(\boldsymbol{\theta}, \zeta, \gamma)$；(2) 固定 $(\boldsymbol{\theta}, \zeta, \gamma)$ 通过 Hungarian 算法求解 $\boldsymbol{\alpha}$

## 实验关键数据

### 主实验

UPMC Food-101 数据集（分类准确率 %）：

| 训练场景 | 方法 | Test(~Train) | Test(Miss Both) | Test(Full Modal) | Test(Text only) | Test(Image only) |
|---------|------|-------------|----------------|-----------------|----------------|-----------------|
| Miss Text | FEDAVG-P | 15.71 | 14.90 | 21.56 | 16.91 | 15.36 |
| Miss Text | FED-INTER | 54.82 | 48.87 | 59.17 | 35.13 | 56.59 |
| Miss Text | **FED-PRIME** | **78.88** | **80.38** | **92.12** | **73.01** | **76.83** |
| Miss Image | FEDAVG-P | 17.35 | 15.12 | 16.84 | 18.12 | 14.81 |
| Miss Image | FED-INTER | 77.96 | 64.62 | 82.08 | 77.69 | 37.56 |
| Miss Image | **FED-PRIME** | **90.55** | **79.12** | **92.89** | **90.18** | **54.14** |
| Miss Both | FEDAVG-P | 14.57 | - | 17.17 | 16.40 | 13.24 |
| Miss Both | FED-INTER | 56.32 | - | 69.57 | 45.15 | 59.30 |
| Miss Both | **FED-PRIME** | **84.44** | - | **93.64** | **87.95** | **72.41** |

FED-PRIME 相对第二名的提升幅度在 Food-101 上为 1.73%~107.83%，在 MM-IMDB 上为 4.41%~69.65%。

### 消融实验

| 方法 | 组成 | Food-101 Miss Text (Full Modal) | MM-IMDB Miss Text (Full Modal) |
|------|------|-------------------------------|-------------------------------|
| FEDAVG-P | 无 prompt 分离 | 21.56 | 30.78 |
| FED-INTRA | 仅 intra-prompt | 62.06 | 12.55 |
| FED-INTER | 仅 inter-prompt | 59.17 | 18.67 |
| **FED-PRIME** | **两者结合** | **92.12** | **37.67** |

鲁棒性（Miss Both, Food-101, 不同缺失率 η）：
| 缺失率 η | FED-PRIME | FEDAVG-P | Centralized-P |
|---------|-----------|---------|--------------|
| 0.00 | ~93% | ~90% | ~93% |
| 0.25 | ~88% | ~60% | ~85% |
| 0.50 | ~85% | ~45% | ~80% |
| 0.75 | ~82% | ~30% | ~75% |
| 1.00 | ~80% | ~15% | ~70% |

### 关键发现

- **双 prompt 设计缺一不可**：仅用 FED-INTER 或 FED-INTRA 都远不如完整的 FED-PRIME，验证了 inter/intra 异构性需要分别处理
- **对齐机制至关重要**：无对齐的 FedAvg prompt-tuning 在高缺失率下性能急剧退化（从 ~90% 降至 ~15%），而 FED-PRIME 保持在 80% 以上
- **FED-PRIME 接近集中式上界**：在高缺失率下，FED-PRIME 甚至超过了集中式 Centralized-P（两者都使用 prompt-tuning）
- **收敛更快更稳定**：FED-PRIME 的训练/测试 loss 收敛速度显著快于 FED-INTER 和 FED-INTRA
- **有趣的 Miss Text 实验**：70% 文本缺失训练后，在 Text Only 测试上仍表现良好，说明 prompt 对齐能有效恢复缺失模态的信息

## 亮点与洞察

1. **问题定义的系统性**：清晰区分了 intra-heterogeneity 和 inter-heterogeneity，并为每种设计了对应的 prompt 集合和聚合策略
2. **隐式知识分离机制**：聚合机制反向约束编码方式——这是一个优雅的设计哲学，让模型自动学习如何将不同类型的知识分配到不同的 prompt
3. **聚类-对齐的形式化**：将 prompt 对齐问题转化为带约束的聚类优化，流行度函数 $U(\boldsymbol{\theta}_q; \zeta)$ 进一步区分通用和专用 prompt
4. **全面的实验设置**：3种训练缺失场景 × 5种测试场景 = 15组实验，覆盖面广

## 局限与展望

1. **仅在双模态（图像+文本）上验证**：三模态及以上场景的可扩展性未知
2. **ViLT 模型固有的文本偏向**：实验发现 Image Only 测试性能总是较差，可能源于 ViLT 预训练的文本中心性
3. **仅选择 8 类样本**：从原始数据集中筛选频次最高的 8 类，可能低估了大规模类别下的挑战
4. **Hungarian 算法的扩展性**：聚类数为 $n \times \tau$ 时，$O(n^3\tau^3)$ 复杂度可能限制大规模部署
5. **缺失模式为随机模拟**：真实世界中的模态缺失可能有更复杂的结构（如与地理位置相关）
6. **未与 CLIP 等更强基础模型结合**：ViLT 已不是最先进的多模态模型

## 相关工作与启发

- Missing Prompt-Tuning（Lee et al.）是集中式场景下为每种缺失模态子集学习专用 prompt 的方法，FED-PRIME 将其扩展到联邦场景
- FedMSplit 和 FedMAC 处理多模态联邦学习，但不利用预训练基础模型
- 聚类-对齐思想可以推广到其他联邦学习中的异构性对齐问题

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次连接联邦学习和多模态 Prompt-Tuning，双 prompt + 聚类对齐设计有创新
- **实验充分度**: ⭐⭐⭐⭐ — 15 组训练-测试场景组合全面，但仅 2 个数据集稍显不足
- **写作质量**: ⭐⭐⭐⭐ — 问题形式化清晰，但符号和公式较多，可读性有提升空间
- **价值**: ⭐⭐⭐⭐ — 填补了多模态缺失数据的联邦 Prompt-Tuning 空白，实际应用场景广泛
