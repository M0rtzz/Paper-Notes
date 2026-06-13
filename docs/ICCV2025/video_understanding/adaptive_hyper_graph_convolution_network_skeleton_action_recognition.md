---
title: >-
  [论文解读] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections
description: >-
  [ICCV 2025][视频理解][骨架动作识别] 本文提出 Hyper-GCN，通过自适应非均匀超图卷积和虚拟超节点（hyper joints）的设计，突破了传统 GCN 仅建模关节对之间二元关系的限制，实现了多关节协同语义的高效聚合，在 NTU-60/120 和 NW-UCLA 数据集上以最轻量的 GCN 设计达到了 SOTA 性能。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "骨架动作识别"
  - "超图卷积"
  - "虚拟连接"
  - "自适应拓扑"
  - "图卷积网络"
---

# Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections

**会议**: ICCV 2025  
**arXiv**: 无 (CVF Open Access)  
**代码**: [https://github.com/6UOOON9/Hyper-GCN](https://github.com/6UOOON9/Hyper-GCN)  
**领域**: 视频理解  
**关键词**: 骨架动作识别, 超图卷积, 虚拟连接, 自适应拓扑, 图卷积网络

## 一句话总结

本文提出 Hyper-GCN，通过自适应非均匀超图卷积和虚拟超节点（hyper joints）的设计，突破了传统 GCN 仅建模关节对之间二元关系的限制，实现了多关节协同语义的高效聚合，在 NTU-60/120 和 NW-UCLA 数据集上以最轻量的 GCN 设计达到了 SOTA 性能。

## 研究背景与动机

1. **领域现状**：骨架动作识别是视频理解的重要方向，现有方法主要分为 GCN 和 Transformer 两大范式。GCN 方法（如 CTR-GCN、BlockGCN）通过图拓扑建模骨架关节关系，参数高效但受限于二元连接；Transformer 方法（如 SkateFormer）通过注意力建模关节间关系，表达力强但计算开销大。

2. **现有痛点**：现有 GCN 方法依赖普通图（normal graph）的邻接矩阵来表示关节拓扑，邻接矩阵只能描述两个关节之间的二元连接关系。然而人体动作是由多个关节协同完成的（如"起跑"需要左手抬起和右腿前迈的协同），二元连接无法捕捉这种多关节协同关系。虽有少数工作尝试使用超图，但都依赖固定的手动设计，缺乏自适应能力。

3. **核心矛盾**：超图能表达多关节关系，但如何自适应地构建与动作相关的超图结构是关键难题。手动设计的固定超图无法适应不同动作类别的拓扑差异。同时，骨架关节数固定（如 NTU 中 25 个），每个关节既要存储局部上下文又要传递全局语义，信息容量有限。

4. **本文目标**：设计一种自适应的超图卷积网络，能根据输入数据动态构建最优的超图拓扑，并通过引入虚拟节点扩展信息传递的容量和路径。

5. **切入角度**：受傀儡戏（皮影、提线木偶）的启发——动作由操纵线（虚拟连接）控制真实关节来驱动。作者提出引入"超节点"（hyper joints）作为虚拟操纵点，通过超图连接到真实关节，既扩展了信息传递路径，又为每个真实关节分担了存储全局语义的压力。

6. **核心 idea**：自适应非均匀超图（A-NHG）通过特征空间中的 K 近邻动态构建超边，多头超图卷积（M-HGC）在不同通道子空间独立构建超图拓扑，虚拟超节点补充全局语义传递能力。

## 方法详解

### 整体框架

Hyper-GCN 由嵌入层和 9 个时空卷积层组成，分为 3 个阶段，每个阶段内使用密集连接。每层包含多头超图卷积（M-HGC）和多尺度时间卷积（MS-TC）两个模块。输入为骨架序列特征 $F_{in} \in \mathbb{R}^{C \times T \times V}$，输出为动作分类结果。

### 关键设计

1. **自适应非均匀超图构建 (A-NHG)**:

    - 功能：根据输入特征动态构建与当前动作相关的超图拓扑。
    - 核心思路：给定关节特征 $X \in \mathbb{R}^{N \times C}$，先用映射函数 $\Phi$ 投影到子空间 $X_H \in \mathbb{R}^{N \times C_h}$，然后计算关节间的欧氏距离矩阵 $m_{i,j} = \|v_i - v_j\|_2$。对每个关节 $i$，只保留距离最近的 $K$ 个关节构建超边，用 softmax 将距离转为概率：$h_{i,j} = \exp(-m_{i,j}) / \sum_{k \in set_i} \exp(-m_{i,k})$。这样构建的超图是非均匀的——每条超边包含的关节数不固定，能表达更分化的关联模式。
    - 设计动机：均匀超图（每条超边固定包含 $K$ 个关节）和非均匀超图（每个关节被限制最多属于 $K$ 条超边）的区别在于后者允许超边大小不同，能更灵活地匹配不同动作的拓扑需求。

2. **多头超图卷积 (M-HGC)**:

    - 功能：在不同通道子空间独立构建超图，捕获多尺度的语义关系。
    - 核心思路：将输入特征沿通道维度分成 8 个头，每个头独立构建超图并进行超图卷积。先做时间平均池化得到空间特征 $\bar{F}_{in}$，然后每个头用独立映射构建 A-NHG 获取关联矩阵 $H$，用 MLP 学习超边权重 $W$。卷积操作为 $F_{out} = \oplus_{k=1}^8 (\hat{A}^k + \alpha \cdot \hat{H}^k) F_{in}^k P^k$，其中 $\hat{A}$ 是物理拓扑，$\hat{H}$ 是超图拓扑，$\alpha$ 是可学习的融合权重。
    - 设计动机：不同通道子空间的特征可能反映关节间不同类型的语义关系（如位置关系、运动趋势等），多头设计允许捕获这种多样性。

3. **虚拟超节点 (Virtual Hyper Joints)**:

    - 功能：扩展信息传递路径，为真实关节分担全局语义的存储压力。
    - 核心思路：引入 $V_h$ 个可学习的超节点 $F_h \in \mathbb{R}^{C \times T \times V_h}$，形状与真实关节特征一致，跨帧共享。超节点手动连接到所有物理关节，参与空间超图卷积但不参与时间卷积。为防止超节点同质化，设计散度损失 $L_h = (\sum_{i,j} \text{ReLU}(c_{i,j}) - V_h) / (V_h(V_h - 1))$，其中 $c_{i,j}$ 为超节点间的余弦相似度。每层设置独立的超节点以适应不同深度的特征。
    - 设计动机：类比 Transformer 中的 CLS token，超节点作为全局信息的中继站，使真实关节能专注于存储局部邻域特征，而全局信息传递交给超节点完成。

### 损失函数 / 训练策略

- 总损失：$L = L_{CE} + \frac{1}{L}\sum_{l=1}^L L_h(C_l)$，交叉熵 + 各层散度损失
- 标签平滑交叉熵
- SGD 优化，Nesterov momentum=0.9，weight decay=0.0004
- 140 epochs，前 5 epochs warmup，lr=0.05，110 epoch 降到 0.005

## 实验关键数据

### 主实验

| 方法 | 类别 | 参数(M) | NTU60-XSub(%) | NTU120-XSub(%) | NW-UCLA(%) |
|------|------|---------|-------------|--------------|-----------|
| CTR-GCN | GCN | 1.5 | 92.4 | 88.9 | 96.5 |
| BlockGCN | GCN | 1.3 | 93.1 | 90.3 | 96.9 |
| HD-GCN | GCN | 1.7 | 93.0 | 89.8 | 96.9 |
| SkateFormer | Trans. | 2.0 | 93.5 | 89.8 | 98.3 |
| DST-HCN | HGCN | 3.5 | 92.3 | 88.8 | 96.6 |
| **Ours (Base)** | HGCN | **1.1** | **93.3** | **90.5** | 97.2 |
| **Ours (Large)** | HGCN | **2.3** | **93.7** | **90.9** | 97.6 |

### 消融实验

| 配置 | NTU120 XSub(%) | 说明 |
|------|---------------|------|
| Baseline (normal GCN) | 84.7 | 无超图 |
| + M-HGC (K=9, 非均匀) | 86.7 (+2.0) | 超图卷积贡献最大 |
| + 3 hyper joints w/o $L_h$ | 86.6 (-0.1) | 超节点无散度损失会同质化 |
| + 3 hyper joints w/ $L_h$ | **86.9** (+0.2) | 散度损失促进多样性 |

### 关键发现

- 非均匀超图在 K=9 时最优（86.7%），而均匀超图在 K=5 时最优（86.5%），非均匀超图更灵活
- 虚拟超节点在无超图卷积时提升有限（+0.2），但与超图卷积结合后效果显著（+2.2）
- 散度损失对 3+ 个超节点至关重要，否则超节点会趋同失效
- Hyper-GCN (Base) 仅 1.1M 参数即超越所有 GCN 方法，参数效率最高

## 亮点与洞察

- **超图表达多关节协同**：突破了普通图只能描述关节对之间二元关系的限制，超图的超边可以自然表达多关节协同语义（如"跑步"涉及双手双脚协同），这是比注意力机制更结构化的建模方式。
- **虚拟超节点的精妙设计**：类比皮影戏的操纵线，虚拟节点作为全局信息中继站，与真实关节通过超图连接，实现了局部和全局信息的优雅分离。这个思路可以迁移到任何图神经网络中。
- **极致的参数效率**：1.1M 参数即达 SOTA，比同级别的 GCN 方法（1.3-1.7M）更小，比 Transformer 方法（2.0-3.5M）小数倍。

## 局限与展望

- 超参数 $K$ 需要手动选择，虽然消融实验给出了指导，但不同数据集可能需要不同的 $K$
- 虚拟超节点的数量也需要手动设定，未来可考虑自适应确定
- 在 NW-UCLA 上未达到 SkateFormer 的 98.3%，可能在小数据集上泛化能力略弱

## 相关工作与启发

- **vs CTR-GCN**：CTR-GCN 学习通道级拓扑但仍是二元关系，本文通过超图扩展到多元关系
- **vs DST-HCN**：DST-HCN 用手动设计的固定超图，本文的 A-NHG 可根据输入自适应构建
- **vs SkateFormer**：SkateFormer 用注意力隐式建模关系，本文用超图显式建模，参数更少

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应超图+虚拟超节点的组合设计具有原创性
- 实验充分度: ⭐⭐⭐⭐ 三个数据集 + 详细消融 + 可视化分析
- 写作质量: ⭐⭐⭐⭐ 皮影戏类比形象直观，公式推导清晰
- 价值: ⭐⭐⭐⭐ 以最小参数量达到 SOTA，为骨架动作识别提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

</div>

<!-- RELATED:END -->
