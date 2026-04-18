---
title: >-
  [论文解读] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation
description: >-
  [ICCV 2025][图像分割][无监督动作分割] 提出闭环最优传输（CLOT）框架，通过三级循环特征学习（帧嵌入→段嵌入→交叉注意力精化帧嵌入）联合求解三个OT问题，在帧级和段级表征之间建立显式反馈循环，显著提升无监督动作分割的边界检测和聚类质量。
tags:
  - ICCV 2025
  - 图像分割
  - 无监督动作分割
  - 最优传输
  - 闭环学习
  - 编码器-解码器
  - Sliced Wasserstein距离
---

# CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.03539](https://arxiv.org/abs/2507.03539)  
**代码**: https://github.com/elenabbbuenob/CLOT  
**领域**: 视频理解/动作分割  
**关键词**: 无监督动作分割, 最优传输, 闭环学习, 编码器-解码器, Sliced Wasserstein距离

## 一句话总结

提出闭环最优传输（CLOT）框架，通过三级循环特征学习（帧嵌入→段嵌入→交叉注意力精化帧嵌入）联合求解三个OT问题，在帧级和段级表征之间建立显式反馈循环，显著提升无监督动作分割的边界检测和聚类质量。

## 研究背景与动机

无监督动作分割旨在无标注条件下对视频帧进行动作类别标注，在体育、监控和机器人等领域有重要应用。现有方法可分为两类：

**经典流水线方法**（CTE、VTE等）：先学习帧表征再聚类，缺乏表征学习与聚类之间的反馈。

**OT方法**（TOT、ASOT等）：利用最优传输联合学习动作表征和伪标签，通过自训练实现反馈。

其中**ASOT**表现最佳，它不假设动作顺序，通过Gromov-Wasserstein OT在帧和动作标签之间获得时序一致的分割。但ASOT存在两个问题：(a) 隐含的段长度先验使其难以检测短时间动作；(b) 缺乏**帧级与段级表征之间的显式反馈**，导致学到的聚类可能与真实段边界不对齐。

另一方面，**HVQ**通过层次向量量化改善了短动作检测，但其固定码本缺乏OT方法的反馈机制，泛化能力弱。CLOT的设计目标就是融合两者优势：保留OT的反馈能力，同时加强段级一致性。

## 方法详解

### 整体框架

CLOT包含三级架构（参见原文Fig. 2）：
- **第一级**：MLP编码器+特征调度机制 → 帧嵌入 $F$ + 伪标签 $\mathbf{T}$（通过OT_1求解）
- **第二级**：并行解码器 → 段嵌入 $S$ + 伪标签 $\mathbf{T}_S$（通过OT_2求解）
- **第三级**：帧-段交叉注意力 → 精化帧嵌入 $F_R$ + 伪标签 $\mathbf{T}_R$（通过OT_3求解）

三级形成闭环：帧→段→精化帧，每级都有OT约束生成伪标签，实现多层次循环优化。

### 关键设计

1. **特征调度编码器（Feature Dispatching Encoder）**: MLP将输入帧特征 $X \in \mathbb{R}^{N \times D}$ 映射为帧嵌入 $F \in \mathbb{R}^{N \times d}$。特征调度机制基于可学习的相似度函数 $\phi(A,F) = \sigma(\beta + \alpha \cdot \frac{A \cdot F}{\|A\|\|F\|})$，其中 $A$ 为可学习的动作聚类嵌入。每帧根据注意力权重分配到最相关的聚类并更新表征：$f_i' = f_i + \frac{1}{K}\sum_{k=0}^K \phi(A_k, f_i) \cdot A_k$。这使帧嵌入能动态适应当前学到的聚类结构，产生更有组织的表征。

2. **并行解码器（Parallel Decoder）**: 采用基于查询的注意力机制（灵感来自DETR），使用可学习查询 $Q \in \mathbb{R}^{K' \times d_{dec}}$ 作为段原型，通过多头交叉注意力和自注意力将帧嵌入解码为段嵌入 $S \in \mathbb{R}^{K' \times d}$（$K' \leq K$）。与自回归解码器不同，并行解码同时预测所有段，避免误差累积。

3. **交叉注意力精化（Cross-Attention Refinement）**: 将段级结构信息注入帧嵌入：$F_R = F + \text{softmax}(\frac{FS^\top}{\tau \cdot \sqrt{d}})S$。这是闭环的关键——让帧表征根据段嵌入进行结构化调整，确保帧级细节与时序分割过程对齐。

4. **Sliced Wasserstein距离**: 作为对余弦距离的补充引入代价矩阵。SW距离通过随机投影将高维分布映射到一维子空间上计算：$\text{SWD}_p(x_i, a_j) = (\frac{1}{M}\sum_{m=1}^M d(R_{\theta_m\#}x_i, R_{\theta_m\#}a_j))^{1/p}$。代价矩阵定义为 $\mathbf{C}_{ij}^{sw} = 1 + \text{SWD}(x_i, a_j) - \mathbf{C}_{i,j}^k$，将SW距离与视觉代价结合。使用 $p=1$ 可获得闭式解，计算高效。

### 损失函数 / 训练策略

OT采用非平衡公式，融合KOT和GW两个子问题：$\min_\mathbf{T} \alpha \mathcal{F}_{GW} + (1-\alpha) \mathcal{F}_{KOT} - \lambda KL(\mathbf{T}^\top \mathbf{1}_n \| \nu)$，其中KL散度罚项允许灵活的非均匀标签分配。训练目标为三级交叉熵损失之和：$\mathcal{L}_{train} = \mathcal{L}(\mathbf{T}, \mathbf{P}) + \mathcal{L}(\mathbf{T}_S, \mathbf{P}_S) + \mathcal{L}(\mathbf{T}_R, \mathbf{P}_R)$，其中 $P_{ij} = \text{softmax}(FA^\top / \tau)_{ij}$。

## 实验关键数据

### 主实验（Activity-level，Hungarian匹配）

| 数据集 | 指标 | ASOT | HVQ | **CLOT** | 提升 |
|--------|------|------|-----|---------|------|
| Breakfast | MoF | 56.1 | 54.4 | **60.1** | +4.0 |
| Breakfast | F1 | 38.3 | 39.7 | **40.1** | +0.4 |
| YTI | MoF | 52.9 | 50.3 | **54.4** | +1.5 |
| 50Salads(Eval) | F1 | 53.6 | - | **63.2** | +9.6 |
| 50Salads(Eval) | mIoU | 30.1 | - | **38.8** | +8.7 |
| DA | F1 | 68.0 | - | **72.6** | +4.6 |

### 消融实验

| 配置 | Breakfast MoF | 50Salads(Eval) F1 | DA F1 | 说明 |
|------|-------------|------------------|-------|------|
| **CLOT** | **60.1** | **63.2** | **72.6** | 完整模型 |
| w/o SWD | 59.8 | 52.7 | 62.3 | 去掉SW距离，F1大幅下降 |
| w/o FD | 51.9 | 51.9 | 68.3 | 去掉特征调度，MoF降8% |
| w/o Decoder | 58.0 | 52.7 | 68.4 | 无段级OT，F1明显下降 |
| w/o Refinement | 59.7 | 52.6 | 68.2 | 无交叉注意力精化，闭环断裂 |

### 关键发现

- **闭环精化的贡献最为关键**：去掉Decoder或Refinement后，50Salads的F1从63.2降至52.7，说明段级反馈对边界检测至关重要。
- **特征调度对Breakfast贡献最大**（MoF降8.2），因为Breakfast的活动种类多、时序复杂，结构化表征尤为重要。
- **SW距离在50Salads和DA上效果显著**（F1分别降10.5和10.3），验证了其在高维空间中更鲁棒的距离度量。
- Video-level评估中，CLOT在Breakfast上MoF达66.3（ASOT为63.3），在50Salads(Eval)上F1达69.7（ASOT为58.9），提升更加显著。

## 亮点与洞察

- 闭环设计的"帧→段→精化帧"三级循环是核心贡献：让不同粒度的表征相互增强，而非单向传播。
- 特征调度机制实现了"软聚类引导的表征更新"，相比硬聚类更灵活。
- 使用Sliced Wasserstein距离替代余弦距离来构建OT代价矩阵，利用了SW距离在保持分布几何结构上的优势。
- 并行解码器的设计避免了自回归解码器的误差累积问题。

## 局限与展望

- 需要预先指定动作类别数 $K$，在实际应用中可能难以确定。
- 与ASOT一样，计算OT本身有一定开销，扩展到超长视频可能受限。
- Video-level和Activity-level的结果不总一致，说明跨视频泛化仍有提升空间。
- 代码结构中Decoder的查询数 $K'$ 的选择策略未充分讨论。

## 相关工作与启发

- 在ASOT的非平衡GW-OT框架上扩展，添加了段级OT和精化OT。
- 并行解码器借鉴了动作预测领域（FUTR3D等）的DETR式设计。
- SW距离已在点云处理、颜色迁移等领域验证过，本文首次将其应用于动作分割的OT代价矩阵。
- 与TOT、UFSA相比不需要动作顺序先验，更通用。

## 评分

- 新颖性: ⭐⭐⭐⭐ 闭环OT的多级循环设计新颖，三个OT问题的编排合理
- 实验充分度: ⭐⭐⭐⭐ 四个基准、两种评估协议、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Fig.2的架构图信息量大
- 价值: ⭐⭐⭐⭐ 对无监督动作分割有明显推动，在多数数据集上创SOTA
