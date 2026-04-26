---
title: >-
  [论文解读] Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction
description: >-
  [AAAI 2026][自动驾驶][动态图学习] 提出 GLFormer，一个轻量级的无注意力 Transformer 框架用于动态图链接预测，用基于交互顺序和时间间隔的自适应令牌混合器替代自注意力，配合层次化聚合机制扩展时间感受野，在 6 个基准上取得了与 Transformer 基线持平或更优的性能，同时大幅降低计算复杂度。
tags:
  - AAAI 2026
  - 自动驾驶
  - 动态图学习
  - 链接预测
  - 注意力机制替代
  - 自适应令牌混合
  - 层次化聚合
---

# Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction

**会议**: AAAI 2026  
**arXiv**: [2511.12442](https://arxiv.org/abs/2511.12442)  
**代码**: 无  
**领域**: Autonomous Driving / Graph Learning  
**关键词**: 动态图学习, 链接预测, 注意力机制替代, 自适应令牌混合, 层次化聚合

## 一句话总结
提出 GLFormer，一个轻量级的无注意力 Transformer 框架用于动态图链接预测，用基于交互顺序和时间间隔的自适应令牌混合器替代自注意力，配合层次化聚合机制扩展时间感受野，在 6 个基准上取得了与 Transformer 基线持平或更优的性能，同时大幅降低计算复杂度。

## 研究背景与动机

动态图学习在交通系统、社交网络、推荐系统等领域至关重要。核心任务之一是动态链接预测——预测未来某个时间点两个节点之间是否会产生交互。

当前方法普遍采用 Transformer 架构来捕获交互序列中的长程时间依赖关系。典型流程是：先通过时间感知随机游走或记忆网络提取结构信息，再用 Transformer 学习历史交互序列中的时间依赖。然而，自注意力机制的**计算复杂度与序列长度呈二次方关系**，在高频或大规模图上难以扩展。此外，注意力机制不加区分地聚合所有成对交互，可能放大噪声并降低泛化能力。

关键观察：受计算机视觉领域 MetaFormer 等工作启发，Transformer 的成功可能更多归功于其架构设计（残差连接、FFN 等），而非自注意力本身。作者通过控制实验验证了这一猜想——**在五种 Transformer 基线上用 pooling 或 MLP 替代自注意力，在四个数据集上性能往往持平**。

这引出了核心问题：**能否设计更简单、无注意力的架构用于动态图，保持表征能力的同时大幅降低计算开销？**

## 方法详解

### 整体框架
GLFormer 的 pipeline：
1. 嵌入层：利用现有动态图方法（TGN/TGAT/DyGFormer 等）获取邻居初始嵌入
2. 自适应令牌混合器：替代自注意力，基于交互顺序和时间间隔进行局部聚合
3. 通道混合器：标准 FFN，学习通道间依赖
4. 层次化聚合：多层堆叠扩展时间感受野
5. 链接预测：MLP 解码器基于节点对的时间表征预测链接概率

### 关键设计

1. **自适应令牌聚合模块（Adaptive Token Aggregation）**:

    - 功能：对每个邻居 $u_i$，从其最近的 $M$ 个邻居中加权聚合信息
    - 核心思路：聚合权重 $\alpha_p^i = \beta \mathbf{w}_p + (1 - \beta) \theta_p^i$ 融合两种因素
        - **顺序权重 $\mathbf{w}_p$**：可学习参数，捕捉交互顺序的重要性
        - **时间权重 $\theta_p^i$**：通过对时间间隔应用 softmax 计算，$\theta_p^i = \frac{\exp(-(t_i - t_{i-p}))}{\sum_q \exp(-(t_i - t_{i-q}))}$，时间距离越近权重越大
        - 可学习参数 $\beta$ 控制两者的融合比例
    - 设计动机：在动态图中，最近的邻居提供了最相关的交互模式，局部聚合比全局注意力更有效且更高效

2. **层次化聚合机制（Hierarchical Aggregation）**:

    - 功能：受膨胀因果卷积启发，随层数增加逐步扩展聚合的时间跨度
    - 核心思路：定义层级偏移集合 $\mathcal{R}_l = \{p \in \mathbb{Z} \mid s^{l-1} \leq p \leq s^l\}$，随着 $l$ 增大，聚合范围覆盖更远的历史交互
    - 第 $l$ 层的聚合：$\mathbf{H}_{i,:}^{(l)} = \sum_{p \in \mathcal{R}_l} (\alpha_p^i)^{(l)} \mathbf{H}_{\text{TA}, i-p}^{(l-1)}$
    - 超出序列边界的位置被因果掩码处理
    - 设计动机：在保持局部聚合低复杂度的同时，通过堆叠捕获长程时间依赖，类似不同尺度的卷积核

3. **复杂度优势**:

    - 功能：将每层复杂度从自注意力的 $O(N^2)$ 降低到 $O(NK_l)$
    - 总复杂度 $O(\sum_{l=1}^L NK_l)$，当 $K_l \ll N$ 时远低于二次方
    - 每层只需 $O(K_l)$ 个核参数，参数效率极高

### 损失函数 / 训练策略
- 二元交叉熵损失，1:1 负采样策略
- 正样本为实际交互 $(u_i, v_j, t)$，负样本从非交互节点中随机采样
- 预测概率 $\hat{y} = \sigma(\mathbf{K}_2(\text{ReLU}(\mathbf{K}_1([\mathbf{Z}_{u_i}; \mathbf{Z}_{v_j}]))))$

## 实验关键数据

### 主实验

**AP（Average Precision）指标，GLFormer 平均排名**:

| 骨干网络 | Vanilla (排名) | Pooling (排名) | MLP (排名) | GLFormer (排名) |
|----------|--------------|--------------|-----------|----------------|
| TGN | 3.17 | 2.83 | 2.17 | **1.67** |
| TCL | 3.33 | 3.50 | 2.00 | **1.17** |
| TGAT | 3.00 | 3.50 | 2.17 | **1.17** |
| CAWN | 2.83 | 3.00 | **1.33** | 2.50 |
| DyGFormer | 2.17 | 3.17 | 3.33 | **1.00** |

**具体数值（DyGFormer 骨干）**:

| 数据集 | Vanilla AP | GLFormer AP | 差异 |
|--------|-----------|-------------|------|
| Wikipedia | 99.03 | **99.03** | +0.00 |
| Reddit | 99.22 | **99.24** | +0.02 |
| MOOC | 87.52 | **87.87** | +0.35 |
| LastFM | 93.00 | **93.34** | +0.34 |
| SocialEvo | 94.73 | **94.76** | +0.03 |
| Enron | 92.47 | **92.62** | +0.15 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 令牌混合器类型 | | |
| Self-Attention (Vanilla) | 排名 2-3 | 原始 Transformer |
| Average Pooling | 排名 3-3.5 | 简单平均，通常最差 |
| MLP | 排名 1.3-2.3 | 非线性变换，较强 |
| **GLFormer** | 排名 **1-1.67** | 最优或次优 |
| 层次化聚合 | | |
| 单层固定窗口 | 性能较低 | 感受野有限 |
| 多层层次化 | 性能提升 | 捕获长程依赖 |
| AUC-ROC 指标 | | |
| GLFormer 在 TGN 上 | 排名 **1.67** | 一致优于原始注意力 |
| GLFormer 在 DyGFormer 上 | 排名 **1.00** | 6 个数据集全面最优 |

### 关键发现
- GLFormer 在大多数骨干网络和数据集上排名第一或第二，**无注意力架构确实能匹配甚至超越 Transformer 基线**
- 在 DyGFormer 骨干上效果最佳（AP 和 AUC-ROC 均排名 1.00），说明长序列建模场景下优势更明显
- CAWN 骨干上 MLP 变体更强（排名 1.33 vs GLFormer 2.50），说明不同骨干对令牌混合器的偏好不同
- 时间权重和顺序权重的融合通过可学习 $\beta$ 自适应平衡，消融显示两者缺一不可
- 计算效率显著：推理速度快于所有使用自注意力的基线

## 亮点与洞察
- **反直觉发现**：动态图学习中自注意力并非不可替代，简单的局部聚合加层次化堆叠即可匹配
- **时间感知的设计**：同时建模交互顺序（学习到的位置权重）和时间间隔（物理时间衰减），比通用注意力更适合时序场景
- **膨胀因果卷积的启发**：将 WaveNet 的层次化感受野扩展思想引入动态图
- **即插即用**：可直接替换现有方法的注意力模块（TGN/TGAT/DyGFormer 等）
- 复杂度分析清晰，理论和实验一致

## 局限与展望
- 在 CAWN 骨干上表现不如 MLP 变体，适应性因骨干而异
- 部分数据集上提升幅度较小（如 Wikipedia 仅 +0.00），高性能区间改进空间有限
- 层次化聚合的超参数（基数 $s$、层数 $L$）需要针对不同数据集调优
- 未在超大规模图（百万节点级）上验证可扩展性
- 未考虑边特征的影响，当前只利用了节点特征和时间戳

## 相关工作与启发
- MetaFormer/PoolFormer 的"注意力不是全部"思想在图学习中同样成立
- 局部聚合 + 层次化扩展的范式可推广到时间序列预测等领域
- 时间衰减权重 $\theta_p^i$ 的设计简单有效，可借鉴到其他时序建模任务
- 动态图学习方法的"嵌入 + 聚合"两阶段范式为架构探索提供了灵活的实验框架

## 评分
- 新颖性: ⭐⭐⭐⭐ — 问题重要、实验验证了反直觉假设，但技术手段相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ — 6 数据集、5 骨干网络、AP+AUC-ROC 双指标、完整消融
- 写作质量: ⭐⭐⭐⭐ — 动机论证严密，初步实验设计精巧
- 价值: ⭐⭐⭐⭐ — 为动态图学习提供了高效替代方案，实用性强

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)
- [\[AAAI 2026\] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)

<!-- RELATED:END -->
