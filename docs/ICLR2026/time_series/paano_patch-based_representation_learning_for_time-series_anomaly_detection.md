---
title: >-
  [论文解读] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection
description: >-
  [ICLR 2026][时间序列][时间序列异常检测] 提出 PaAno，一种基于 patch 级表示学习的轻量时间序列异常检测方法，使用 1D-CNN 编码器 + triplet loss + pretext loss 学习 patch 嵌入空间，通过与记忆库中正常 patch 的距离计算异常分数，在 TSB-AD 基准上全面 SOTA，且仅需 0.3M 参数和数秒推理。
tags:
  - ICLR 2026
  - 时间序列
  - 时间序列异常检测
  - Patch表示学习
  - 轻量级CNN
  - 记忆库
  - 度量学习
---

# PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection

**会议**: ICLR 2026  
**arXiv**: [2602.01359](https://arxiv.org/abs/2602.01359)  
**代码**: [有](https://github.com/jinnnju/PaAno)  
**领域**: 时间序列  
**关键词**: 时间序列异常检测, Patch表示学习, 轻量级CNN, 记忆库, 度量学习

## 一句话总结

提出 PaAno，一种基于 patch 级表示学习的轻量时间序列异常检测方法，使用 1D-CNN 编码器 + triplet loss + pretext loss 学习 patch 嵌入空间，通过与记忆库中正常 patch 的距离计算异常分数，在 TSB-AD 基准上全面 SOTA，且仅需 0.3M 参数和数秒推理。

## 研究背景与动机

时间序列异常检测在工业监控、金融交易、医疗健康等领域至关重要。近年来 Transformer 和 Foundation Model（如 AnomalyTransformer、MOMENT、TimesFM 等）逐步占据主导，但：

**现有方法的问题**：

**幻象式进步（illusion of progress）**：Sarfraz et al. 和 Liu & Paparrizos 揭示——在严格评估协议下（去除 point adjustment、避免阈值调优），复杂大模型并未显著超越简单方法

**计算成本高**：Transformer 和 Foundation Model 参数量大（0.5M-210M），运行时间长（几十秒到上千秒），不适合实时和资源受限场景

**局部性稀释**：全局自注意力机制对局部上下文不敏感，而异常检测恰恰依赖短区间内的局部时间模式

**PaAno 的定位**：

- 采用 **表示学习** 范式（而非预测或重构），这在异常检测中相对欠探索
- 引入局部性归纳偏置（locality inductive bias）：借鉴视觉异常检测中 PatchCore 等 patch-based 方法的成功经验
- 核心思想：正常时间序列具有重复的局部模式，异常打破这些短程规律

## 方法详解

### 整体框架

PaAno 分为训练和推理两个阶段：

**训练阶段**：(1) 从训练序列提取重叠的固定长度 patch，(2) 用 1D-CNN 编码器将 patch 映射到嵌入空间，(3) 通过 triplet loss + pretext loss 训练编码器，(4) 构建正常 patch 的记忆库。

**推理阶段**：(1) 提取查询时间步周围的 patch，(2) 计算每个 patch 与记忆库中最近邻的距离作为 patch 级异常分数，(3) 聚合为时间步级异常分数。

### 关键设计

**1. Patch 提取与归一化**

给定训练序列 $\mathbf{X} = (\mathbf{x}_1, \ldots, \mathbf{x}_N)$，使用窗口大小 $w$、步长1的滑动窗口提取 patch 集合 $\mathcal{P} = \{\mathbf{p}_t\}_{t=1}^{N-w+1}$。每个 patch 进行实例归一化（零均值、单位方差），提升对分布漂移和体制变化的鲁棒性。

**2. 模型架构（三组件）**

- **Patch 编码器 $f_\theta$**：4层 1D-CNN + 全局平均池化，输出64维嵌入向量 $\mathbf{h}$
- **投影头 $g_\theta$**：两层 MLP（256维），将 $\mathbf{h}$ 投影为 $\mathbf{z}$ 用于度量学习
- **分类头 $c_\theta$**：单层 MLP，预测两个 patch 是否时间连续

训练后仅保留编码器 $f_\theta$ 用于推理。

**3. Triplet Loss（主损失）**

对每个锚定 patch $\mathbf{p}_i$：

- **正样本** $\mathbf{p}_i^+$：锚定 patch 在 $r$ 步内随机移位（非零移位），保证时间模式相似
- **负样本** $\mathbf{p}_i^-$：minibatch 中与 $\mathbf{p}_i$ 余弦距离最远的 patch（最远负样本策略）

$$\mathcal{L}_{\text{triplet}} = \frac{1}{M} \sum_{i=1}^{M} \max(0, \text{dist}(\mathbf{z}_i, \mathbf{z}_i^+) - \text{dist}(\mathbf{z}_i, \mathbf{z}_i^-) + \delta)$$

目标：小位移的 patch 嵌入接近（对微小时序偏移鲁棒），不同模式的 patch 远离（对异常敏感）。

**4. Pretext Loss（辅助损失）**

受视觉异常检测中预测 patch 空间关系的启发，设计时序版本——预测两个 patch 是否时间上连续：

$$\mathcal{L}_{\text{pretext}} = \frac{1}{M} \sum_{i=1}^{M} \left[ -\log c_\theta(\mathbf{h}_i, \mathbf{h}_i^{\text{pre}}) - \frac{1}{U} \sum_{j=1}^{U} \log(1 - c_\theta(\mathbf{h}_i, \mathbf{h}_{i,j}^{\text{rand}})) \right]$$

其中 $\mathbf{p}_i^{\text{pre}}$ 是前 $w$ 步的 patch（时间上连续），$\mathbf{p}_{i,j}^{\text{rand}}$ 是随机 patch。仅在训练初期应用（前20次迭代线性衰减至0），帮助尽快建立结构化嵌入空间。

**5. 记忆库构建与压缩**

全量记忆库 $\mathcal{M} = \{f_\theta(\mathbf{p}_t) \mid \mathbf{p}_t \in \mathcal{P}\}$。为减少存储和计算：

- K-means 聚类为 $K$ 个簇
- 每簇选最近质心的向量作为代表
- 压缩后记忆库 $\hat{\mathcal{M}}$ 仅为原始的10%

**6. 异常分数计算**

查询时间步 $t_*$ 的 $w$ 个包含它的 patch，每个 patch 的异常分数为 $k$ 近邻距离均值：

$$S(\mathbf{p}_t) = \frac{1}{k} \sum_{i=1}^{k} \text{dist}(f_\theta(\mathbf{p}_t), \mathbf{m}_t^{(i)})$$

最终异常分数为覆盖 $t_*$ 的所有 patch 分数的平均值。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{triplet}} + \lambda \cdot \mathcal{L}_{\text{pretext}}$$

- $\lambda$ 在前20次迭代从1线性衰减至0
- AdamW 优化器，weight decay $10^{-4}$
- 仅训练200次迭代，minibatch 512
- 10次不同随机种子取平均

## 实验关键数据

### 主实验

在 TSB-AD 基准上评估，包括530条单变量序列（TSB-AD-U）和180条多变量序列（TSB-AD-M），与48个基线对比。

**单变量异常检测（TSB-AD-U）**：

| 方法 | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | 参数量 | 运行时间 |
|------|--------|---------|--------|---------|--------|---------|
| **PaAno** | **0.52** | **0.89** | **0.46** | **0.86** | 0.3M | 6.9s |
| KAN-AD | 0.43 | 0.82 | 0.41 | 0.80 | <0.1M | 12.1s |
| (Sub)-PCA | 0.42 | 0.76 | 0.37 | 0.71 | - | 1.5s |
| MOMENT (FT) | 0.39 | 0.76 | 0.30 | 0.69 | 109.6M | 43.6s |
| TimesFM | 0.30 | 0.74 | 0.28 | 0.67 | 203.5M | 83.8s |
| AnomalyTrans. | 0.12 | 0.56 | 0.08 | 0.50 | 4.8M | 48.9s |

**多变量异常检测（TSB-AD-M）**：

| 方法 | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | 参数量 | 运行时间 |
|------|--------|---------|--------|---------|--------|---------|
| **PaAno** | **0.43** | **0.79** | **0.38** | **0.76** | 0.3M | 12.8s |
| KAN-AD | 0.41 | 0.75 | 0.38 | 0.73 | <0.1M | 31.9s |
| DeepAnT | 0.31 | 0.76 | 0.32 | 0.73 | <0.1M | 9.5s |
| PCA | 0.31 | 0.74 | 0.31 | 0.70 | - | 0.1s |
| CATCH | 0.30 | 0.73 | 0.24 | 0.67 | 210.8M | 40.1s |

### 消融实验

论文在附录提供了详细消融，关键点：

- Triplet loss 是核心贡献（去掉后 VUS-PR 显著下降）
- Pretext loss 的早期应用加速了嵌入空间结构化
- 记忆库压缩（10%）几乎不损失性能
- 超参数鲁棒性高，不需要精细调优

### 关键发现

- PaAno 在所有6项评测指标上均为第一名（单变量和多变量）
- 参数量仅 0.3M，远小于 Transformer 方法（4.8M-210M）
- 运行时间 6.9-12.8s，而 Foundation Model 需 42-1221s
- 传统 PCA 和 KShapeAD 在严格评估下表现出乎意料地好，验证了"幻象式进步"现象
- AnomalyTransformer、DCdetector 等 Transformer 方法在严格评估下排名极低（20+）

## 亮点与洞察

1. **小即是美**：0.3M 参数的 1D-CNN 碾压百万级 Transformer 和 Foundation Model，从根本上质疑了"越大越好"的假设
2. **局部性优先**：异常检测需要精细的局部感知，全局注意力反而稀释了关键信号
3. **视觉→时序的成功迁移**：patch-based 表示学习 + 记忆库的范式（类 PatchCore）在时序领域同样有效
4. **严格评估的重要性**：去除 point adjustment 和阈值调优后，方法排名大幅变化

## 局限与展望

- 半监督设定假设训练数据全为正常，无法处理已知少量标注异常的场景
- patch 大小 $w$ 为固定超参，不同类型异常可能需要不同窗口
- 记忆库方案在超长训练序列上可能面临存储挑战（虽然压缩到10%）
- 仅评估了异常检测任务，其 patch 嵌入能否迁移到分类/预测等任务值得探索

## 相关工作与启发

- 受视觉异常检测（PaDiM、PatchCore、SPADE）启发，将 patch-level 表示+记忆库范式迁移到时序
- 与 预测/重构 方法形成互补：PaAno 属于 **表示学习** 范式，不需要重构原始信号
- 启发：简单但针对性强的方法（局部 patch + 度量学习）在特定任务上可能比通用大模型更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ （视觉→时序的 patch 表示学习迁移，设计简洁有效）
- 实验充分度: ⭐⭐⭐⭐⭐ （48个基线、710条时序、严格评估协议、多项指标）
- 写作质量: ⭐⭐⭐⭐ （动机论证充分，对评估问题的讨论有深度）
- 价值: ⭐⭐⭐⭐⭐ （以极低成本达到 SOTA，实用性极高，有望成为工业界首选方案）

<!-- RELATED:START -->

## 相关论文

- [Contextual and Seasonal LSTMs for Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)
- [GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](../../ICML2025/time_series/kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)
- [Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](../../ICML2025/time_series/causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)

<!-- RELATED:END -->
