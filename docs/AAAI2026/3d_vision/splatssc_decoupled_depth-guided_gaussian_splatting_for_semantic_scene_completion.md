---
title: >-
  [论文解读] SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion
description: >-
  [AAAI 2026][3D视觉][语义场景补全] 提出 SplatSSC，通过深度引导的高斯基元初始化策略和解耦高斯聚合器（DGA），解决目标中心（object-centric）范式中随机初始化低效和离群基元产生浮点伪影的问题，在 Occ-ScanNet 上IoU提升6.3%、mIoU提升4.1%，同时延迟和内存成本降低超过9.3%。
tags:
  - AAAI 2026
  - 3D视觉
  - 语义场景补全
  - 3D高斯
  - 深度引导
  - 解耦聚合
  - 室内场景理解
---

# SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion

**会议**: AAAI 2026  
**arXiv**: [2508.02261](https://arxiv.org/abs/2508.02261)  
**代码**: [GitHub](https://github.com/Made-Gpt/SplatSSC)  
**领域**: 3D视觉  
**关键词**: 语义场景补全, 3D高斯溅射, 深度引导, 解耦聚合, 物体中心表示

## 一句话总结

提出SplatSSC，一种基于深度引导初始化和解耦高斯聚合器（DGA）的单目3D语义场景补全框架，通过紧凑的高斯基元初始化和鲁棒的几何-语义解耦聚合，在Occ-ScanNet上以更少基元获得SOTA性能。

## 研究背景与动机

### 问题定义

单目3D语义场景补全（SSC）旨在从单张图像推断场景的稠密几何和语义描述。该任务在具身智能和自动驾驶中至关重要。近年来，物体中心范式（以GaussianFormer为代表）使用灵活的3D高斯基元表示场景，在性能与效率之间取得了新平衡。

### 核心动机

物体中心范式在纯视觉设置（单目输入）下面临一个**基础性挑战**：如何仅从单目线索高效初始化和可靠监督3D基元。

现有方法采用的**随机初始化策略**导致两个关键问题：

**1. 低效的基元初始化**：大量基元被浪费在空白或未知空间的表示上。例如，GaussianFormer-2使用19200个基元，其中大量基元对应空区域，造成计算资源浪费。

**2. 离群基元的脆弱聚合**：现有的Gaussian-to-voxel聚合策略（GaussianFormer和GaussianFormer-2的PGS）**缺乏有效的拒绝机制**。离群基元会将虚假语义扩散到远处的体素上，产生"floaters"伪影。

### 对PGS（概率高斯叠加）缺陷的深入分析

作者对GaussianFormer-2的PGS进行了严格的数学分析，发现其根本缺陷：

- PGS将学习到的不透明度 $\mathbf{a}_i$ 用作GMM中的先验概率
- 对于一个**孤立的离群基元** $G_n$，其附近任意点 $\mathbf{x}^f$ 处，其他远处基元的似然 $p(\mathbf{x}^f|G_m)$ 趋近于零
- 这导致后验概率 $p(G_n|\mathbf{x}^f)$ **退化为1**，无论不透明度 $\mathbf{a}_n$ 多低：

$$p(G_n|\mathbf{x}^f) \approx \frac{p(\mathbf{x}^f|G_n)\mathbf{a}_n}{p(\mathbf{x}^f|G_n)\mathbf{a}_n + 0} = 1$$

- 结果：低置信度的离群基元仍会对附近体素产生完全的语义影响，产生floaters

## 方法详解

### 整体框架

SplatSSC的流程：
1. 图像编码器（EfficientNet-B7 + FPN）提取多尺度图像特征
2. 冻结的Depth-Anything-V2提取深度特征
3. GMF模块融合图像和深度特征
4. 深度分支生成精细化深度图 → 引导初始化**紧凑的高斯基元集**（仅1200个）
5. 多阶段编码器迭代精炼基元
6. DGA将精炼基元转换为最终的语义体素网格

### 关键设计

#### 1. 组级多尺度融合模块（GMF）与组交叉注意力（GCA）

**功能**：高效融合多尺度图像特征和深度特征，生成高质量的几何先验。

**GCA的具体设计**：
1. 从深度特征 $\mathcal{F}_d$ 和多尺度图像特征 $\mathcal{F}_{rgb}$ 中采样特征
2. 将特征沿通道维度分为 $G$ 组，每组维度 $D_g = D/G$
3. Query来自深度特征，Key/Value来自各尺度图像特征：

$$Q_g = (\mathcal{F}_d^s W_q)^g, \quad K_g^l = (f_{rgb}^{s,l} W_k)^g, \quad V_g^l = (f_{rgb}^{s,l} W_v)^g$$

4. 用轻量级线性投影取代标准点积注意力（借鉴Deformable Attention）：

$$A_g^l = \mathbb{S}_l(W_a(Q_g + K_g^l))$$

5. 最终融合：$\mathcal{F}_d' = \mathbb{C}_g(\sum_{l=1}^{L} A_g^l \circ V_g^l) W_o$

**效率分析**：标准交叉注意力复杂度 $\mathcal{O}(LN^2D)$，GCA降为 $\mathcal{O}(ND^2(L+2)/G)$，特别适合长序列。

**设计动机**：
- $W_a$ 在不同组和尺度间共享，显著减少参数和计算量
- 利用深度估计器的潜在特征（而非仅用深度图值），获取更丰富的初始化信息
- 为高斯基元提供同时包含**位置（where）和语义（what）**信息的初始嵌入

#### 2. 解耦高斯聚合器（DGA）

**功能**：将高斯基元到体素的聚合分解为几何占用预测和条件语义分布两条独立路径，鲁棒处理离群基元。

**几何占用预测**：

$$\alpha'(\mathbf{x}) = 1 - \prod_{i \in \mathcal{N}(\mathbf{x})} (1 - \alpha(\mathbf{x}; G_i) \cdot \mathbf{a}_i)$$

关键区别：将不透明度 $\mathbf{a}_i$ 直接乘入高斯核，作为**存在置信度**。低置信度离群基元的不透明度直接压制其占用贡献。

**条件语义分布**（假设位置已被占用）：

$$e^k(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_i) \cdot \tilde{\mathbf{c}}_i^k}{\sum_{j \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_j)}$$

关键区别：语义预测**完全独立于不透明度**，仅依赖几何邻近度和softmax归一化的语义属性。

**概率融合**：

$$\hat{\mathbf{y}}_x^k = \alpha'(\mathbf{x}) \cdot e^k(\mathbf{x}), \quad \hat{\mathbf{y}}_x^{empty} = 1 - \alpha'(\mathbf{x})$$

**设计动机**：当一个低不透明度的离群基元存在时：
- $\alpha'(\mathbf{x})$ 会很低（因为 $\mathbf{a}_i$ 直接参与占用计算）
- 即使 $e^k(\mathbf{x})$ 可能被该离群基元主导，但乘以低 $\alpha'(\mathbf{x})$ 后影响被有效抑制
- 无需复杂的启发式规则即可消除floaters

#### 3. 概率尺度损失（Probability Scale Loss）

**功能**：为所有编码器层提供几何辅助监督，稳定端到端训练。

$$\mathcal{L}_{scal}^{prob} = \frac{1}{2} \sum_{i=1}^{n-1} \frac{i}{n} \cdot \mathcal{L}_{scal}^{geo,i} + \mathcal{L}_{scal}^{geo,n}$$

**设计动机**：引入线性加权调度——对早期层施加较弱约束，对深层施加较强一致性要求，匹配逐层精炼的特性。

### 损失函数 / 训练策略

**两阶段训练**：

**Stage 1 — 深度分支预训练**：
$$\mathcal{L}_d = \lambda_1 \mathcal{L}_{huber}^{depth} + \lambda_2 \mathcal{L}_{huber}^{pts} + \lambda_3 \mathcal{L}_{grad}$$
- 10个epoch，2张RTX 3090

**Stage 2 — 端到端SplatSSC训练**：
$$\mathcal{L}_{ssc} = \mathcal{L}_{sem} + \lambda_4 \mathcal{L}_{scal}^{prob}$$
- 移除 $\mathcal{L}_d$，避免过度约束
- $\mathcal{L}_{sem} = \lambda_5 \mathcal{L}_{focal} + \lambda_6 \mathcal{L}_{lovasz}$
- 10个epoch（full），20个epoch（mini），4张RTX 4090

损失权重：$\lambda_1=10, \lambda_2=20, \lambda_3=\lambda_4=0.5, \lambda_5=100, \lambda_6=2$

## 实验关键数据

### 主实验

**Occ-ScanNet数据集上的语义场景补全结果**：

| 方法 | 输入 | IoU↑ | mIoU↑ |
|---|---|---|---|
| TPVFormer | RGB | 33.39 | 24.94 |
| MonoScene | RGB | 41.60 | 24.62 |
| GaussianFormer | RGB | 40.91 | 29.93 |
| SurroundOcc | RGB | 42.52 | 30.83 |
| EmbodiedOcc | RGB | 53.95 | 45.48 |
| EmbodiedOcc++ | RGB | 54.90 | 46.20 |
| RoboOcc | RGB | 56.48 | 47.67 |
| **SplatSSC** | **RGB** | **62.83** | **51.83** |

**提升幅度**：IoU和mIoU分别超过RoboOcc达**6.35%和4.16%**。

### 消融实验

**网络组件消融（Occ-ScanNet-mini）**：

| GMF | 聚合器 | IoU↑ | mIoU↑ | 说明 |
|---|---|---|---|---|
| ✗ | GF.agg | 11.64 | 12.62 | 两者都差，近乎失效 |
| ✗ | GF2.agg | 27.54 | 17.27 | GF2改善有限 |
| ✗ | DGA | 48.85 | 36.91 | DGA即使无GMF也表现优秀 |
| ✓ | GF.agg | 16.63 | 10.45 | GMF+GF.agg仍然失败 |
| ✓ | GF2.agg | 57.70 | 45.13 | GMF大幅提升GF2 |
| ✓ | DGA | **60.61** | **48.01** | GMF+DGA最优 |

**高斯参数消融**：

| 基元数量 | 尺度范围 | IoU↑ | mIoU↑ | 说明 |
|---|---|---|---|---|
| 19200 | [0.01, 0.08] | 62.77 | 47.69 | 数量多但mIoU反而更低 |
| 4800 | [0.01, 0.08] | 62.23 | 47.20 | 中等数量 |
| **1200** | **[0.01, 0.16]** | **61.47** | **48.87** | 最少基元，最高mIoU |
| 1200 | [0.01, 0.32] | 57.09 | 42.38 | 尺度范围过大反而退化 |

**深度分支消融**：

| 配置 | $\delta_1$↑ | RMSE↓ | 说明 |
|---|---|---|---|
| DAv2 (无GMF) | 0.075 | 50.31 | 冻结深度模型直接用效果极差 |
| DAv2 + GMF | 0.981 | 4.94 | GMF提升 $\delta_1$ 达0.906 |
| FT-DAv2 + GMF | **0.993** | **2.98** | 微调+GMF最优 |

### 关键发现

1. **GF.agg在稀疏设置下几乎失效**（10.45% mIoU），证明floaters是稀疏溅射的关键瓶颈
2. **仅1200个基元即可达到最高mIoU**，比19200个基元反而更好——证明了深度引导初始化的优越性
3. **GMF对深度质量有决定性影响**：冻结DAv2的 $\delta_1$ 仅0.075，加GMF后提升至0.981
4. **两阶段训练优于端到端**：端到端训练的 $\mathcal{L}_{ssc}$ 更大、mIoU更低
5. **效率提升明显**：推理延迟降低9.32%，内存降低9.64%

## 亮点与洞察

1. **对PGS缺陷的严谨数学分析**非常有说服力——通过极限分析证明了不透明度在后验归一化中被抵消
2. **解耦思路优雅**：几何用不透明度门控，语义纯靠几何邻近，两者通过概率乘法融合
3. **少即是多**：1200个基元超过19200个，深度引导的精确初始化远比大量随机初始化有效
4. **GMF的巨大提升**：将 $\delta_1$ 从0.075提升至0.981，说明利用深度估计器的特征而非仅用深度值的重要性
5. **概率尺度损失的线性加权调度**符合逐层精炼的直觉

## 局限与展望

1. **超参数敏感性**：batch size < 4时性能急剧下降（mIoU从48.87降至36.09）
2. **单帧架构限制**：目前逐帧预测，无法直接扩展到全局场景感知（基元累积导致内存增长）
3. 依赖冻结的预训练Depth-Anything模型，上限受限
4. 仅在室内场景（ScanNet）上验证，室外自动驾驶场景的适用性待验证
5. 下采样的30×40深度网格分辨率可能限制细粒度几何的恢复

## 相关工作与启发

- **GaussianFormer/GaussianFormer-2**：物体中心3D占用预测的开创者和改进者，本文的直接对比和改进对象
- **EmbodiedOcc/EmbodiedOcc++**：将物体中心范式应用于室内场景，本文基于其框架改进
- **RoboOcc**：通过不透明度线索和平面约束增强稳定性，是本文在Occ-ScanNet上的主要对比方法
- **Depth-Anything-V2**：提供强大的深度先验，本文进一步利用其潜在特征
- **VoxFormer**：建议先稀疏再致密的两阶段方法，与本文的稀疏初始化理念相通

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — PGS缺陷分析深刻，DGA设计优雅，1200基元超过19200令人印象深刻
- **实验充分度**: ⭐⭐⭐⭐⭐ — 双数据集、全面消融（组件/参数/损失/深度/效率）、与8+方法对比
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题分析严谨，数学推导清晰，框架一目了然
- **价值**: ⭐⭐⭐⭐⭐ — IoU/mIoU分别提升6.3%/4.1%，实际部署友好（更少基元=更高效率）
# SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion

**会议**: AAAI 2026  
**arXiv**: [2508.02261](https://arxiv.org/abs/2508.02261)  
**代码**: [GitHub](https://github.com/Made-Gpt/SplatSSC)  
**领域**: 3D视觉  
**关键词**: 语义场景补全, 3D高斯, 深度引导, 解耦聚合, 室内场景理解

## 一句话总结

提出 SplatSSC，通过深度引导的高斯基元初始化策略和解耦高斯聚合器（DGA），解决目标中心（object-centric）范式中随机初始化低效和离群基元产生浮点伪影的问题，在 Occ-ScanNet 上IoU提升6.3%、mIoU提升4.1%，同时延迟和内存成本降低超过9.3%。

## 研究背景与动机

单目3D语义场景补全（SSC）旨在从单张图像推断完整的3D几何和语义描述。近期目标中心范式（以GaussianFormer为代表）使用3D高斯基元表示场景，取得了效率和性能的突破。但该范式存在两个根本性问题：

**问题一：低效的基元初始化**
- 为了在无几何线索的情况下覆盖整个3D空间，现有方法在3D体积中随机分布大量基元
- 大部分基元浪费在表示空旷或未知空间上，造成严重冗余
- 例如GaussianFormer使用19200个基元，其中大量是无效的

**问题二：离群基元的脆弱聚合**
- 高斯到体素的溅射策略（GaussianFormer、GaussianFormer-2）缺乏有效的离群排斥机制
- 孤立的离群基元会将错误语义溅射到远处体素上，产生"floaters"
- GaussianFormer-2的概率高斯叠加（PGS）存在设计缺陷：不透明度 $\mathbf{a}_i$ 在后验概率归一化中被抵消，使低置信度离群基元仍产生高占用值

作者对PGS的缺陷给出了严谨的数学分析：对于孤立离群基元 $G_n$，其邻域点 $\mathbf{x}^f$ 处其他基元的似然趋近于0，导致后验概率坍缩为1：
$$p(G_n|\mathbf{x}^f) \approx \frac{p(\mathbf{x}^f|G_n)\mathbf{a}_n}{p(\mathbf{x}^f|G_n)\mathbf{a}_n + 0} = 1$$

即使 $\mathbf{a}_n$ 很低，归一化后也被抵消，语义期望退化为该离群基元的语义标签。

## 方法详解

### 整体框架

SplatSSC 包含以下主要组件：
1. **图像编码器**（EfficientNet + FPN）提取多尺度图像特征
2. **冻结的 Depth-Anything-V2** 提取深度特征
3. **深度分支**：GMF模块融合图像和深度特征，输出精化深度图
4. **Lifter**：基于深度先验初始化稀疏高斯基元
5. **多阶段编码器**：迭代精化高斯属性
6. **DGA**：解耦几何和语义，将高斯基元聚合为语义体素网格

### 关键设计

#### 1. **深度分支与 GMF 模块（Group-wise Multi-scale Fusion）**：高效的多模态融合

**GCA层（Group Cross-Attention）**：
- 将深度特征和多尺度图像特征在预定义参考点处采样
- 沿通道维度分为 $G$ 组，每组特征维度为 $D_g = D/G$
- Query来自深度特征，Key和Value来自各尺度图像特征
- 使用轻量级线性投影替代标准点积注意力：

$$A_g^l = \mathbb{S}_l(W_a(Q_g + K_g^l))$$

$$\mathcal{F}_d' = \mathbb{C}_g(\sum_{l=1}^{L} A_g^l \circ V_g^l) W_o$$

**效率分析**：标准交叉注意力复杂度为 $\mathcal{O}(LN^2D)$，GCA降低为 $\mathcal{O}(ND^2(L+2)/G)$，权重矩阵 $W_a$ 跨组和尺度共享，大幅减少参数量。

**数据意义**：GMF将冻结的Depth-Anything-V2的 $\delta_1$ 指标从0.075提升到0.981（提升0.906），对微调版提升到0.993。

#### 2. **解耦高斯聚合器（DGA）**：消除floaters的关键

DGA将语义占用预测分解为两条独立路径：

**几何占用预测（Geometric Occupancy Prediction）**：
$$\alpha'(x) = 1 - \prod_{i \in \mathcal{N}(\mathbf{x})} (1 - \alpha(\mathbf{x}; G_i) \cdot \mathbf{a}_i)$$

关键区别：每个基元的影响被其学习到的不透明度 $\mathbf{a}_i$ 调制。低置信度的离群基元自然被抑制。

**条件语义分布（Conditional Semantic Distribution）**：
$$e^k(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_i) \cdot \tilde{\mathbf{c}}_i^k}{\sum_{j \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_j)}$$

语义预测**不使用不透明度**，仅依赖几何邻近度和归一化语义权重。

**概率融合**：
$$\hat{\mathbf{y}}_x^k = \alpha'(\mathbf{x}) \cdot e^k(\mathbf{x}), \quad \hat{\mathbf{y}}_x^{empty} = 1 - \alpha'(\mathbf{x})$$

这是一个优雅的门控机制：低占用概率直接抑制任何错误的语义预测，无需额外的启发式规则即可消除floaters。

#### 3. **概率尺度损失（Probability Scale Loss）**：渐进式几何监督

扩展 MonoScene 的几何感知尺度损失，适用于所有 $n$ 个编码器层的占用概率预测：

$$\mathcal{L}_{scal}^{prob} = \frac{1}{2}\sum_{i=1}^{n-1} \frac{i}{n} \cdot \mathcal{L}_{scal}^{geo,i} + \mathcal{L}_{scal}^{geo,n}$$

线性权重调度：对早期层施加较弱约束，逐渐在深层强化一致性。

### 损失函数 / 训练策略

**两阶段训练**：

Stage 1：深度分支预训练
$$\mathcal{L}_d = 10 \mathcal{L}_{\text{huber}}^{\text{depth}} + 20 \mathcal{L}_{\text{huber}}^{\text{pts}} + 0.5 \mathcal{L}_{\text{grad}}$$

Stage 2：端到端SSC训练
$$\mathcal{L}_{ssc} = 100 \mathcal{L}_{\text{focal}} + 2 \mathcal{L}_{\text{lovasz}} + 0.5 \mathcal{L}_{scal}^{prob}$$

注意：Stage 2 中移除了深度损失 $\mathcal{L}_d$，避免模型被初始深度预测过度约束。Depth-Anything-V2 全程冻结。

## 实验关键数据

### 主实验（Occ-ScanNet）

| 方法 | 输入 | IoU↑ | mIoU↑ | 说明 |
|------|------|------|-------|------|
| TPVFormer | RGB | 33.39 | 24.94 | Transformer基线 |
| GaussianFormer | RGB | 40.91 | 29.93 | 目标中心范式开创 |
| MonoScene | RGB | 41.60 | 24.62 | 密集2D→3D提升 |
| EmbodiedOcc | RGB | 53.95 | 45.48 | 之前的代表性方法 |
| EmbodiedOcc++ | RGB | 54.90 | 46.20 | 增强版 |
| RoboOcc | RGB | 56.48 | 47.67 | 之前SOTA |
| **SplatSSC** | RGB | **62.83** | **51.83** | 大幅领先 |

IoU提升6.35%（绝对值），mIoU提升4.16%。在所有语义类别上均有一致提升。

### 消融实验

**组件消融**：

| GMF | 聚合器 | IoU↑ | mIoU↑ | 说明 |
|-----|--------|------|-------|------|
| ✗ | GF.agg | 11.64 | 12.62 | 无GMF+原始聚合近乎失败 |
| ✗ | GF2.agg | 27.54 | 17.27 | 无GMF+PGS聚合 |
| ✗ | DGA | 48.85 | 36.91 | 无GMF但有DGA仍有效 |
| ✓ | GF.agg | 16.63 | 10.45 | GMF+原始聚合仍差 |
| ✓ | GF2.agg | 57.70 | 45.13 | GMF+PGS |
| ✓ | **DGA** | **60.61** | **48.01** | 完整方法最优 |

**高斯参数消融**：

| 基元数量 | 尺度范围 | 内存(MiB) | 延迟(ms) | IoU | mIoU |
|---------|---------|-----------|----------|-----|------|
| 19200 | [0.01,0.08] | 3.122 | 135.18 | 62.77 | 47.69 |
| 4800 | [0.01,0.08] | 3.158 | 123.27 | 62.23 | 47.20 |
| **1200** | **[0.01,0.16]** | **3.112** | **115.56** | **61.47** | **48.87** |
| 19200 | [0.01,0.32] | 14.380 | 134.51 | OOM | — |

仅用1200个基元即可达到最高mIoU，比19200个基元更优且大幅减少计算量。

### 关键发现

1. DGA比GF2.agg在IoU和mIoU上均高出约2.8%，证明floaters是稀疏溅射的关键瓶颈
2. GMF模块对性能影响巨大——去除后即使有DGA也会降低11%+
3. 仅1200个基元+适中尺度范围[0.01,0.16]为最优配置
4. 显式深度损失在Stage 2反而有害，概率尺度损失更合适
5. 效率优势显著：相比EmbodiedOcc延迟降低9.32%，内存降低9.64%

## 亮点与洞察

1. **PGS缺陷的精确数学分析**：对GaussianFormer-2聚合器中不透明度被归一化抵消的问题给出了严谨证明，非常有说服力
2. **解耦设计的优雅性**：将几何占用和语义预测完全分解，不透明度仅在几何路径中起门控作用，是一个自然且原则性的解决方案
3. **少即是多**：1200个深度引导的基元优于19200个随机基元，直观展示了初始化质量的重要性
4. **GCA的效率设计**：组共享注意力权重实现了参数和计算的大幅节省
5. **概率尺度损失的渐进设计**：对不同编码器层施加不同权重的监督，适应了逐层精化的特点

## 局限与展望

1. 目前仅在室内场景（Occ-ScanNet）上评估，未验证室外场景（如nuScenes）
2. 依赖冻结的Depth-Anything-V2，深度先验的质量上限受限
3. 1200个基元可能不够应对大规模或极其复杂的场景
4. 未讨论时序一致性（连续帧间的占用预测一致性）
5. GCA中的分组数 $G$ 等超参的敏感性未详细分析

## 相关工作与启发

- **GaussianFormer/GaussianFormer-2**：目标中心SSC范式的开创者，SplatSSC指出其聚合器缺陷
- **EmbodiedOcc/EmbodiedOcc++**：将目标中心范式引入室内场景的代表工作
- **VoxFormer**：稀疏到密集的Transformer方法，首次引入几何先验来生成提议
- **Depth-Anything-V2**：强大的单目深度估计器，提供了深度特征和深度先验
- **启发**：深度引导的稀疏初始化+解耦聚合的设计思路可推广到其他目标中心3D感知任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — PGS缺陷分析和DGA解耦设计新颖且有深度
- **实验充分度**: ⭐⭐⭐⭐ — 详尽的消融实验，但数据集限于室内
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题分析严谨，数学推导清晰，方法动机充分
- **实用价值**: ⭐⭐⭐⭐ — 室内3D场景理解的重要进展，对具身智能有直接价值

<!-- RELATED:START -->

## 相关论文

- [Monocular Semantic Scene Completion via Masked Recurrent Networks](../../ICCV2025/3d_vision/monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](../../ICCV2025/3d_vision/disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [Global-Aware Monocular Semantic Scene Completion with State Space Models](../../ICCV2025/3d_vision/global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)

<!-- RELATED:END -->
