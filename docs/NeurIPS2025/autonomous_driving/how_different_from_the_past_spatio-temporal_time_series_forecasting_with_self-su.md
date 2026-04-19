---
title: >-
  [论文解读] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning
description: >-
  [NeurIPS 2025][自动驾驶][时空预测] 提出 ST-SSDL 框架，通过自监督偏差学习（SSDL）捕捉当前输入与历史模式之间的动态偏差，利用可学习原型离散化隐空间并以对比损失+偏差损失实现相对距离一致性，在六个时空基准上取得 SOTA。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 时空预测
  - 自监督学习
  - 偏差建模
  - 原型学习
  - 对比学习
---

# How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.04908](https://arxiv.org/abs/2510.04908)  
**代码**: [GitHub](https://github.com/Jimmy-7664/ST-SSDL)  
**领域**: 自动驾驶  
**关键词**: 时空预测, 自监督学习, 偏差建模, 原型学习, 对比学习

## 一句话总结

提出 ST-SSDL 框架，通过自监督偏差学习（SSDL）捕捉当前输入与历史模式之间的动态偏差，利用可学习原型离散化隐空间并以对比损失+偏差损失实现相对距离一致性，在六个时空基准上取得 SOTA。

## 研究背景与动机

时空预测（如交通流量、能源需求）是城市计算的核心任务。现有方法虽在准确率上有所提升，但普遍忽略了一个关键信号：**当前观测与历史模式之间的动态偏差**。在真实交通系统中，政策干预、特殊事件或外部事故常导致当前时序与历史状态产生显著偏差，这些偏差蕴含对未来行为的重要预测信号。

现有问题：
1. 部分方法（如使用固定时间偏移）只能捕捉静态的历史上下文，无法建模动态偏差
2. 简单的阈值方法将偏差视为二值事件，而实际偏差是连续变化的
3. 在高维隐空间中，如何量化物理空间偏差到隐空间偏差的映射关系，是核心挑战

核心观察：市中心传感器通常呈现高方差（偏差大），而乡村道路较稳定（偏差小），偏差程度随时空上下文动态变化。

## 方法详解

### 整体框架

ST-SSDL 由三个核心部分组成：
1. **历史锚点构建**：以历史平均值作为自监督锚点
2. **自监督空间离散化**：利用可学习原型将连续隐空间离散化
3. **自监督偏差量化**：通过偏差损失强制物理空间和隐空间的相对距离一致性

整体架构采用编码器-解码器结构，基于 GCRU（Graph Convolution Recurrent Unit）建模时空依赖。

### 关键设计

#### 1. 历史作为自监督锚点

将训练序列按周分段（利用时空数据的周期性），计算各时刻的历史加权平均 $\bar{X}^w = \frac{1}{S}\sum_{s=1}^{S}X_s^w$。对当前输入 $X^c$，检索时间戳对齐的历史锚点 $X^a$。二者通过共享编码器得到隐表示 $H^c, H^a \in \mathbb{R}^{N \times h}$。

#### 2. 自监督空间离散化（原型学习 + 对比损失）

引入 $M=20$ 个可学习原型 $\mathbf{P}_1, \ldots, \mathbf{P}_M \in \mathbb{R}^{M \times d}$（$d=64$），通过 query-prototype 注意力机制实现离散化：

$$\alpha_i = \frac{\exp(Q \cdot \mathbf{P}_i^\top / \sqrt{d})}{\sum_{j=1}^{M} \exp(Q \cdot \mathbf{P}_j^\top / \sqrt{d})}$$

对注意力分数排序后，选最相关原型为正样本 $\mathcal{P}^c$、次相关为负样本 $\mathcal{N}^c$，施加 triplet 对比损失：

$$\mathcal{L}_{Con} = \max(\|\widetilde{\nabla}(Q^c) - \mathcal{P}^c\|_2^2 - \|\widetilde{\nabla}(Q^c) - \mathcal{N}^c\|_2^2 + \delta, 0)$$

其中 $\widetilde{\nabla}$ 为 stop-gradient 操作，防止模型进入所有表示坍缩到同一原型的懒惰模式。

#### 3. 自监督偏差量化（偏差损失）

核心思想是**相对距离一致性**：物理空间中接近（远离）的 current-history 对，在隐空间中也应保持接近（远离），即 $D_1 > D_2 \Rightarrow \widetilde{D}_1 > \widetilde{D}_2$。

以各自最近原型作为代理，计算偏差损失：

$$\mathcal{L}_{Dev} = \|\widetilde{\nabla}(\|Q^c - Q^a\|_1) - \|\mathcal{P}^c - \mathcal{P}^a\|_1\|_1$$

stop-gradient 使 $\|Q^c - Q^a\|_1$ 近似物理空间距离 $D$，$\|\mathcal{P}^c - \mathcal{P}^a\|_1$ 代表隐空间距离 $\widetilde{D}$。

#### 4. GCRU 编码器-解码器

编码器：基于 Chebyshev 图卷积的 GRU 单元，图卷积 $Z \star_{\mathcal{G}} \Theta = \sum_{k=0}^{K} \tilde{\mathcal{A}}^k Z W_k$。

解码器：利用编码器输出和原型增强表示生成自适应邻接矩阵 $\tilde{\mathcal{A}} = \text{Softmax}(\text{ReLU}(H' \cdot H'^\top))$，其中 $H' = W[H^c | V^c | H^a | V^a] + b$。

### 损失函数 / 训练策略

联合训练目标：

$$\mathcal{L} = \mathcal{L}_{MAE} + \lambda_{Con} \cdot \mathcal{L}_{Con} + \lambda_{Dev} \cdot \mathcal{L}_{Dev}$$

- $\mathcal{L}_{MAE}$：预测与真值的 MAE 损失
- $\lambda_{Con}, \lambda_{Dev}$：超参数控制各损失贡献
- 优化器：Adam，初始学习率 0.001
- 架构：1层编码器 + 1层解码器，隐维度 128/64/32（依数据集而定）
- 输入/预测窗口：均为1小时（12个时间步）

## 实验关键数据

### 主实验

在6个交通数据集上与13个基线对比：

| 数据集 | 指标 | MegaCRN | STDN | **ST-SSDL** |
|--------|------|---------|------|-------------|
| METRLA (60min) | MAE | 3.48 | 3.57 | **3.37** |
| METRLA (60min) | RMSE | 7.31 | 7.80 | **7.17** |
| PEMSBAY (15min) | MAE | 1.26 | 1.36 | **1.26** |
| PEMSBAY (15min) | RMSE | 2.71 | 2.96 | **2.65** |
| PEMSD7(M) (15min) | MAE | 2.05 | 2.17 | **2.02** |
| PEMSD7(M) (15min) | RMSE | 3.88 | 4.17 | **3.83** |

在所有6个数据集、所有预测步长（15/30/60 min）上，ST-SSDL 均取得最佳或并列最佳。

### 消融实验

| 变体 | METRLA MAE | PEMSBAY MAE |
|------|------------|-------------|
| 完整 ST-SSDL | **3.37** | **1.86** |
| 去掉 $\mathcal{L}_{Con}$ | 3.42 | 1.89 |
| 去掉 $\mathcal{L}_{Dev}$ | 3.44 | 1.90 |
| 去掉原型模块 | 3.48 | 1.91 |
| 去掉历史锚点 | 3.46 | 1.90 |

### 关键发现

1. 对比损失和偏差损失均有独立贡献，联合使用效果最佳
2. 原型数量 $M=20$ 为最优平衡点，过多/过少都导致性能下降
3. 可视化显示：高偏差输入被映射到原型空间中离锚点更远的位置，证实模型成功量化动态偏差
4. 复杂度分析：SSDL 模块仅增加 $\mathcal{O}(NMd)$ 复杂度，不构成瓶颈

## 亮点与洞察

1. **首次提出时空数据偏差建模**：开创性地识别并解决了"当前与历史偏差量化"这一被忽视的问题
2. **相对距离一致性**思想优雅：不要求精确量化绝对偏差，而是保持偏差的相对排序，鲁棒且实用
3. **全自监督设计**：不需要额外标签，历史平均值天然提供锚点，具有很强的通用性
4. **即插即用**：SSDL 可视为一种通用的隐空间正则化技术，理论上可应用于其他时空模型

## 局限与展望

1. 历史锚点以周为周期简单平均，对于非周期性或突变模式的建模能力有限
2. 原型数量 $M$ 固定，无法自适应不同数据集的复杂度
3. 仅在交通数据集上验证，方法在气候、能源等其他时空场景的泛化性待验证
4. 编码器-解码器骨干（GCRU）相对传统，结合 Transformer/Mamba 骨干可能进一步提升

## 相关工作与启发

- **时空预测**：STGCN、DCRNN、Graph WaveNet 等关注图建模，ST-SSDL 在此基础上增加偏差感知
- **自监督学习**：借鉴视觉对比学习（SimCLR、MoCo）的思想，首次将偏差量化纳入自监督框架
- **原型学习**：将 VQ-VAE 式的原型离散化与对比学习结合，用于时空偏差的结构化表示

## 评分

- 新颖性: ⭐⭐⭐⭐ — 偏差建模思路新颖，相对距离一致性设计优雅
- 实验充分度: ⭐⭐⭐⭐ — 6个数据集 + 充分消融 + 可视化分析
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 提出的 SSDL 模块具有广泛的即插即用潜力

<!-- RELATED:START -->

## 相关论文

- [DBLoss: Decomposition-based Loss Function for Time Series Forecasting](dbloss_decomposition-based_loss_function_for_time_series_forecasting.md)
- [Self-Supervised Learning of Graph Representations for Network Intrusion Detection](self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)
- [ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)
- [Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](../../ECCV2024/autonomous_driving/equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)

<!-- RELATED:END -->
