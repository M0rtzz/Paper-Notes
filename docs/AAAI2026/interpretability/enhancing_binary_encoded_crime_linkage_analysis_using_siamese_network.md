---
title: >-
  [论文解读] Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network
description: >-
  [AAAI 2026][犯罪关联分析] 提出基于 **Siamese Autoencoder** 的犯罪关联分析框架，通过 **decoder 阶段融合地理时间特征** 和 **领域专家驱动的数据降维策略**，在英国 NCA 的真实 ViCLAS 数据库上实现了 AUC 提升最高 9%，为高维稀疏二进制编码犯罪数据提供了有效的机器学习解决方案。
tags:
  - AAAI 2026
  - 犯罪关联分析
  - 孪生网络
  - 自编码器
  - 地理时间特征融合
  - ViCLAS数据库
---

# Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network

**会议**: AAAI 2026  
**arXiv**: [2511.07651](https://arxiv.org/abs/2511.07651)  
**代码**: [https://github.com/AlberTgarY/CrimeLinkageSiamese](https://github.com/AlberTgarY/CrimeLinkageSiamese)  
**领域**: 其他  
**关键词**: 犯罪关联分析, 孪生网络, 自编码器, 地理时间特征融合, ViCLAS数据库

## 一句话总结

提出基于 **Siamese Autoencoder** 的犯罪关联分析框架，通过 **decoder 阶段融合地理时间特征** 和 **领域专家驱动的数据降维策略**，在英国 NCA 的真实 ViCLAS 数据库上实现了 AUC 提升最高 9%，为高维稀疏二进制编码犯罪数据提供了有效的机器学习解决方案。

## 研究背景与动机

### 问题场景

**犯罪关联（Crime Linkage, CL）** 是现代执法的关键任务，旨在通过分析犯罪者的 **作案手法（Modus Operandi, MO）** 来识别系列犯罪。准确的 CL 有助于优化侦查资源配置、加强公共安全，其理论基础是 **行为一致性**（同一犯罪者在不同犯罪中行为相似）和 **行为独特性**（不同犯罪者的行为模式可区分）。

### 现有方法的局限

**传统统计方法的限制**：逻辑回归假设线性特征关系、决策树强加刚性层级结构，都不适合捕获犯罪行为中的非线性关联。

**高维稀疏数据挑战**：真实犯罪数据库（如 ViCLAS）包含 446 维二进制编码特征，约 91% 为零值，信号极其稀疏。

**地理时间信息利用不足**：现有方法要么忽略地理时间信息，要么简单拼接到输入层，2 维的地理时间信号在 217-446 维行为特征中占比不足 1%，被严重稀释。

**数据集规模受限**：此前主要研究集中于小规模、地理范围有限的数据集，未在像 ViCLAS 这样的大型真实数据库上验证。

### 核心创新动机
- **信号稀释问题**：将地理时间特征从输入层移到 **decoder 阶段融合**，使其在行为抽象已完成后再调制潜在表示，避免被高维行为特征淹没。
- **领域知识驱动降维**：与 NCA 领域专家合作设计 5 种数据降维映射策略，在减少维度的同时保留行为语义。

## 方法详解

### 整体框架

框架包含三个核心组件：
1. **Siamese Autoencoder 网络结构**：双分支共享权重，encoder 压缩 + decoder 重构 + decoder 阶段地理时间融合
2. **对比+重构联合损失**：聚拢同源犯罪、分离不同犯罪
3. **推理管道**：基于潜在空间距离计算相似度概率分数

### 关键设计

#### 1. **Siamese Autoencoder 网络结构**

- **Encoder**：两层线性层 + ReLU（446→128→8），将输入压缩到 8 维潜在空间
- **Decoder**：镜像结构（8→128→446），重构原始特征空间
- **地理时间融合**：在 decoder 第一层输出后，将对数变换后的空间-时间特征通过线性层加性融合
- **参数量**：共 21,740 参数（vs. Naive Siamese 的 22,981 参数），参数量更少但性能更优
- **设计动机**：
    - 自编码器的重构约束确保潜在表示保留结构信息
    - 8 维瓶颈层强迫网络学习最紧凑的行为表示

#### 2. **Decoder 阶段地理时间融合**

- **核心思路**：地理时间数据本质上反映**成对关系**（两个犯罪之间的距离和时间间隔），将其在 encode 个体行为特征之后融合更符合侦查逻辑
- **具体做法**：
    - 对空间距离和时间间隔取对数变换
    - 通过一个线性层将 2D 地理时间特征映射到 128 维
    - 与 decoder 第一层输出**加性融合**
- **优势对比**：
    - 输入层拼接：2D 信号在 446D 中占 <1%，几乎无影响
    - Decoder 融合：行为抽象完成后再引入，信号放大效果显著
- **实验验证**：跨网络变体一致提升 0.86-3.29% AUC（Table 4）

#### 3. **领域专家驱动的数据降维策略**

设计了 5 种降维策略（Table 1）：

| 策略 | 剩余特征数 | 降维率 | 设计者 |
|------|-----------|--------|--------|
| No Map | 446（原始） | 0% | — |
| Map 1 | 282 | 36.8% | NCA 分析师（20年+经验） |
| Map 2 | 384 | 13.9% | Map 1 + 法医心理学家咨询 |
| Map 3 | 266 | 40.4% | Map 1 和 Map 2 的混合 |
| Map 4 | 217 | 51.3% | 法医心理学专家 |
| Map 5 | 286 | 35.9% | Map 4 的改良（减少抽象度） |

- **降维方法**：将语义相似的二进制特征合并到更抽象的类别（如"购物中心停车场"和"体育场停车场"合并为"停车场"）
- **设计动机**：分析师在实践中通常通过主题匹配而非精确行为匹配来关联犯罪

### 损失函数 / 训练策略

**联合损失**：$\mathcal{L} = \alpha \mathcal{L}_{\text{contrast}} + \beta \mathcal{L}_{\text{recon}}$，$\alpha=1.0$, $\beta=0.2$

**对比损失**：使用混合欧几里得-曼哈顿距离度量
$$\mathcal{L}_{\text{contrast}} = \mathbb{E}[y \cdot d^2 + (1-y) \cdot \max(m-d, 0)^2]$$
其中 $y$ 表示是否关联（1=关联），$m=5$ 为间隔参数。

**重构损失**：基于余弦相似度
$$\mathcal{L}_{\text{recon}} = \mathbb{E}\left[\frac{v_1^\top \hat{v_1}}{\|v_1\| \|\hat{v_1}\|} + \frac{v_2^\top \hat{v_2}}{\|v_2\| \|\hat{v_2}\|}\right]$$

**推理阶段**：$S_{ij} = \exp(-D_{ij}/\beta)$，$\beta = m/1.5$，将潜在空间距离转换为 $(0, 1]$ 的概率分数。

**训练细节**：Adam 优化器，lr=0.001，batch_size=128，2 epochs，Cosine Annealing 学习率调度，5-fold 交叉验证。

## 实验关键数据

### 数据集
- **Single Victim-Offender-Scene Series**：1,482 个案例 / 493 个系列 / 2014年记录，无地理时间数据
- **Multiple Victim-Offender-Scene Series**：22,282 犯罪（1990-2021），446 特征，11,970 用于分析（这是已发表研究中最大的 ViCLAS 数据集）

### 主实验

**小数据集（Single Victim-Offender-Scene）**：

| 方法 | AUC | TP Fixed FP |
|------|-----|-------------|
| Ours | 85 ± 1.98 | 77.73 ± 4.32 |
| Logistic Regression | 86 ± 2.14 | 77.19 ± 5.98 |
| PCA | 82 ± 4.02 | 64.97 ± 4.00 |

**大数据集（Multiple Victim-Offender-Scene, 使用 Map 5）**：

| 方法 | AUC | TP Fixed FP | AUPRC |
|------|-----|-------------|-------|
| **Ours (map 5)** | **84 ± 2.86** | **79.38 ± 2.56** | **15.43** |
| Naive Siamese (map 5) | 83 ± 2.72 | 76.20 ± 2.69 | 15.09 |
| Logistic Regression | 75 ± 2.97 | 70.43 ± 2.12 | 10.24 |
| Ours (无 map) | 77 ± 2.11 | 68.31 ± 1.92 | 13.32 |

**相对提升**：相比 LR，分别提升 12.0%（AUC）、12.71%（TPFP）、50.68%（AUPRC）。

### 消融实验

**架构选择影响（RQ3, Table 4 部分）**：

| 配置 | AUC (%) | 说明 |
|------|---------|------|
| MLP, 无 skip, 2+2, Decoder | **77.29** | 最优配置 |
| MLP, 无 skip, 2+2, Concat | 76.43 | Decoder 融合 +0.86% |
| MLP, 有 skip, 2+2, Decoder | 67.07 | Skip connection 有害 |
| 1D CNN, 无 skip, 2+2, Decoder | 61.74 | MLP 显著优于 CNN |
| SIREN, 无 skip, 2+2, Decoder | 58.28 | 周期激活函数不适合 |
| MLP, 无 skip, 1+1 | 52.49 | 太浅 |
| MLP, 无 skip, 4+4 | 63.57 | 太深过拟合 |

**核心发现**：
- 2+2 层深度最优，更浅/更深都会退化
- 不使用 skip connections 反而更好（+6.55%），可能是因为直接特征传播干扰了细微犯罪模式的抽象

### 关键发现

1. **Decoder 融合一致优于输入层拼接**：MLP +0.86%、1D CNN +3.29%、SIREN +3.09%
2. **适度的数据降维提升性能**：Map 5（35.9% 降维率）取得最佳 AUC 84%，过度抽象或不抽象都不如
3. **时间 OOD 挑战**：2021-2025 年的后 COVID 数据显示性能明显下降，需要定期重新训练
4. **系统可减少 80% 人工审查**：在保留超半数真实犯罪关联的同时，大幅减少假阳性

## 亮点与洞察

1. **真实数据验证**：使用英国 NCA 的 ViCLAS 数据库（迄今公开研究中最大的性犯罪数据集），而非合成数据，增强了研究的实际意义。
2. **Decoder 融合思路的普适性**：当辅助信息维度远低于主要特征时，延迟融合（late fusion）的策略值得在其他领域借鉴。
3. **领域专家驱动的特征工程**：5 种降维策略来自不同专业背景的专家，展示了 ML 与领域知识结合的最佳实践。
4. **伦理考量周全**：有详细的部署安全保障、偏见审计计划和人在环设计。

## 局限与展望

1. **二进制编码丢失信息**：将复杂犯罪行为简化为 0/1 编码可能丢失重要的频次和程度信息。
2. **时间分布漂移**：模型在 post-2021 数据上性能显著下降，需要持续再训练机制。
3. **数据机密性**：ViCLAS 数据不可公开获取，可复现性受限。
4. **仅 8 维瓶颈**：可能过于紧凑，探索不同瓶颈维度对性能影响的分析有限。
5. **仅限性犯罪**：未评估在盗窃、抢劫等其他犯罪类型上的泛化能力。

## 相关工作与启发

- **Solomon et al. (2020)**：将 Siamese 网络用于入室盗窃关联（40 维 TF-IDF），本文扩展到 446 维二进制编码并增加重构约束。
- **Tonkin et al. (2017)**：混合欧几里得-曼哈顿距离度量的来源。
- **启发**：对于高维稀疏数据 + 低维辅助信息的场景，信号稀释是一个被低估的问题，延迟融合策略值得系统性研究。

## 评分

- 新颖性: ⭐⭐⭐ — 核心技术（Siamese + autoencoder）并不新颖，主要贡献在应用和融合策略
- 实验充分度: ⭐⭐⭐⭐ — 真实数据、多种消融、时间OOD测试，但缺少其他犯罪类型
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，研究问题驱动，伦理讨论详尽
- 价值: ⭐⭐⭐⭐ — 对执法领域有直接实际价值，方法论贡献一般

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[AAAI 2026\] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)
- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)

<!-- RELATED:END -->
