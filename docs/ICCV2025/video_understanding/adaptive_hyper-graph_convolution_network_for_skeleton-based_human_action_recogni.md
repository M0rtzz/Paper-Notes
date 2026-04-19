---
title: >-
  [论文解读] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition
description: >-
  [ICCV 2025][视频理解][骨架动作识别] 提出 Hyper-GCN，通过**自适应非均匀超图**替代传统二元图来建模骨骼拓扑，并引入**虚拟超关节**（hyper joints）创建虚拟连接，使多关节协同关系得以直接建模，在 NTU-60/120 和 NW-UCLA 上以最轻量的 GCN 设计实现 SOTA（base 版仅 1.1M 参数、1.63 GFLOPs）。
tags:
  - ICCV 2025
  - 视频理解
  - 骨架动作识别
  - 超图卷积
  - 自适应拓扑
  - 虚拟连接
  - 图卷积网络
---

# Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: https://github.com/6UOOON9/Hyper-GCN  
**领域**: video_understanding  
**关键词**: 骨架动作识别, 超图卷积, 自适应拓扑, 虚拟连接, 图卷积网络

## 一句话总结

提出 Hyper-GCN，通过**自适应非均匀超图**替代传统二元图来建模骨骼拓扑，并引入**虚拟超关节**（hyper joints）创建虚拟连接，使多关节协同关系得以直接建模，在 NTU-60/120 和 NW-UCLA 上以最轻量的 GCN 设计实现 SOTA（base 版仅 1.1M 参数、1.63 GFLOPs）。

## 研究背景与动机

骨架动作识别的核心是如何建模关节之间的拓扑关系。现有方法主要分三类：

**GCN 方法**（ST-GCN, CTR-GCN 等）：用邻接矩阵表示二元关节连接，但**只能建模两个关节之间的关系**

**Transformer 方法**（SkateFormer 等）：用注意力图建模拓扑，效果好但参数量和 GFLOPs 显著更高

**已有超图方法**（Hyper-GNN, Selective-HCN）：尝试用超图建模多关节关系，但**依赖固定的手工设计**

**为什么二元连接不够？** 以"起跑"动作为例——它是左手抬起 + 右腿前迈的**协同组合**。这种多关节协作关系无法用两两之间的边来充分表达。在普通图中，信息需要经过 2 层卷积才能从一个关节传到 2-hop 邻居；而在超图中，一次卷积就能将信息传播到超边连接的所有关节，**扩展了感受野**。

**为什么固定超图不好？** 已有的超图方法（如 Hyper-GNN）手工定义超边，依赖于人类先验知识。但不同动作的关键关节组合完全不同——"喝水"关注头和手，"踢东西"关注腿和手的协同。因此需要**自适应地**为每种动作学习不同的超图拓扑。

**皮影戏/提线木偶的启发**：骨架就像木偶，而虚拟超关节就像操控木偶的"线"和"控制点"。真实关节承担着存储局部特征和传递全局语义的双重压力，引入虚拟关节可以分担全局信息传递的任务，让真实关节专注于局部特征表示。

## 方法详解

### 整体框架

Hyper-GCN 由以下模块组成：
- **嵌入层**：将原始坐标映射到特征空间
- **9 层时空卷积层**：每层包含 M-HGC（多头超图卷积）+ MS-TC（多尺度时间卷积）
- **3 个阶段**：通道数分别为 128/256/256（base）或 128/256/512（large），阶段内有密集连接
- **分类头**：全局平均池化 + FC

### 关键设计

#### 1. 自适应非均匀超图构建（A-NHG）

**核心思路**：不固定超边结构，而是根据关节特征在语义空间中的距离**自适应构建**超图。

给定关节特征 $X \in \mathbb{R}^{N \times C}$，首先通过映射函数 $\Phi$ 投影到子空间 $X_H \in \mathbb{R}^{N \times C_h}$，然后计算关节间的欧氏距离矩阵：

$$m_{i,j} = m_{j,i} = \|v_i - v_j\|_2$$

**为什么用欧氏距离而不是余弦相似度？** 欧氏距离可以更好地反映特征空间中关节的绝对位置关系，而映射到子空间后的距离计算也保留了原始特征的空间相关性。

对每个关节 $i$，只保留距离最近的 $K$ 个关节所对应的超边，通过 softmax 转换为概率：

$$h_{i,j} = \begin{cases} \frac{\exp(-m_{i,j})}{\sum_{k \in \text{set}_i} \exp(-m_{i,k})}, & j \in \text{set}_i \\ 0, & j \notin \text{set}_i \end{cases}$$

**为什么是"非均匀"？** 与均匀超图（每个超边包含固定 $K$ 个关节）不同，A-NHG 限制的是每个关节**最多被 $K$ 个超边包含**。这意味着不同超边包含的关节数量是不固定的，允许超边捕捉更多样化的关节组合模式。

#### 2. 多头超图卷积（M-HGC）

将特征沿通道维度分为 8 个头，每个头独立构建超图并执行卷积：

$$F_{out} = \bigoplus_{k=1}^{8} (\hat{A}^k + \alpha \cdot \hat{H}^k) F_{in}^k P^k$$

其中 $\hat{A}$ 是归一化物理邻接矩阵，$\hat{H}$ 是归一化超图关联矩阵，$\alpha$ 是可学习融合参数。

**为什么保留物理拓扑？** 超图虽然能学习高阶关系，但身体的物理连接（骨骼）仍然是动作的基础约束。通过加权融合物理拓扑和学习拓扑，模型可以在保持物理合理性的同时探索隐式关系。

**为什么 8 个头？** 不同通道组可能关注不同语义（位置、速度、姿态等），多头设计让每组通道有自己专属的超图拓扑。

超图权重通过 MLP 学习，使用 LeakyReLU + Tanh 激活，将权重限制在 $[-1, 1]$ 范围内——允许**抑制性连接**，这是正常 GCN 不具备的能力。

#### 3. 虚拟超关节与连接

引入可学习的超关节 $F_h \in \mathbb{R}^{C \times T \times V_h}$（$V_h$ 为超关节数量），与真实关节一起参与超图卷积：

- 超关节在时间维度上跨帧共享（对齐时间维度）
- 每层设独立超关节（匹配不同深度的特征层次）
- 超关节仅参与空间超图卷积，不参与时间卷积
- 手动将超关节连接到所有物理关节

**散度损失（Divergence Loss）**：防止超关节同质化，通过余弦矩阵度量超关节之间的差异：

$$\mathcal{L}_h(C) = \frac{\sum_{i=1}^{V_h} \sum_{j=1}^{V_h} \text{ReLU}(c_{i,j}) - V_h}{V_h(V_h - 1)}$$

$$\mathcal{L} = \mathcal{L}_{CE} + \frac{1}{L} \sum_{l=1}^{L} \mathcal{L}_h(C^l)$$

### 损失函数 / 训练策略

- **主损失**：标签平滑交叉熵损失 $\mathcal{L}_{CE}$
- **辅助损失**：散度损失 $\mathcal{L}_h$，按层数平均后与主损失相加
- **优化器**：SGD + Nesterov momentum (0.9)，weight decay 0.0004
- **训练周期**：140 epochs，前 5 epochs warm-up
- **学习率**：初始 0.05，epoch 110 降至 0.005，epoch 120 降至 0.0005
- **多流集成**：4-stream（关节 + 骨骼 + 关节运动 + 骨骼运动）
- **硬件**：单卡 RTX 3090

## 实验关键数据

### 主实验

与 SOTA 对比（4-stream 集成）：

| 方法 | 类别 | 参数(M) | GFLOPs | NTU60-XSub | NTU60-XView | NTU120-XSub | NTU120-XSet | NW-UCLA |
|------|------|---------|--------|------------|-------------|-------------|-------------|---------|
| CTR-GCN | GCN | 1.5 | 1.97 | 92.4 | 96.4 | 88.9 | 90.6 | 96.5 |
| InfoGCN | GCN | 1.6 | 1.84 | 92.7 | 96.9 | 89.4 | 90.7 | 96.6 |
| BlockGCN | GCN | 1.3 | 1.63 | 93.1 | 97.0 | 90.3 | 91.5 | 96.9 |
| SkateFormer | Trans | 2.0 | 3.62 | 93.5 | **97.8** | 89.8 | 91.4 | **98.3** |
| DST-HCN | HGCN | 3.5 | 2.93 | 92.3 | 96.8 | 88.8 | 90.7 | 96.6 |
| **Ours (B)** | HGCN | **1.1** | **1.63** | 93.3 | 97.4 | 90.5 | 91.7 | 97.2 |
| **Ours (L)** | HGCN | 2.3 | 2.88 | **93.7** | **97.8** | **90.9** | **92.0** | 97.6 |

关键结论：
- Base 版以**最少参数**（1.1M）和**最少 GFLOPs**（1.63）全面超越所有 GCN 和超图方法
- Large 版在 NTU120 上超越参数量最小的 Transformer 方法 SkateFormer

### 消融实验

A-NHG 超参数 $K$ 的影响（NTU120 X-Sub，单流关节）：

| K 值 | 均匀超图 (%) | 非均匀超图 (%) |
|------|-------------|---------------|
| Baseline | 84.7 | 84.7 |
| 3 | 86.4 (+1.7) | 86.0 (+1.3) |
| 5 | **86.5 (+1.9)** | 86.2 (+1.5) |
| 7 | 86.3 (+1.6) | 86.5 (+1.8) |
| 9 | 86.0 (+1.3) | **86.7 (+2.2)** |
| 11 | 85.9 (+1.2) | 86.4 (+1.7) |

虚拟超关节消融（NTU120 X-Sub）：

| 超关节数 | 无 M-HGC 无散度损失 | 无 M-HGC 有散度损失 | 有 M-HGC 无散度损失 | 有 M-HGC 有散度损失 |
|---------|-------------------|-------------------|-------------------|-------------------|
| 0 | 84.7 | 84.7 | - | - |
| 1 | 84.9 | 84.9 | 86.7 | 86.7 |
| 3 | 84.9 | **85.2** | 86.6 | **86.9** |
| 5 | 84.7 | 85.0 | 86.6 | 86.8 |

### 关键发现

1. **非均匀超图在 $K=9$ 时最优**：与均匀超图在 $K=5$ 时最优不同，说明非均匀结构能容忍更多连接而不引入过多噪声
2. **3 个超关节最优**：过多超关节引入冗余和歧义线索，反而降低性能
3. **散度损失对多超关节至关重要**：可视化显示无散度损失时超关节严重同质化（余弦矩阵趋近全 1）
4. **超图构建与动作语义对齐**：可视化显示"踢东西"时超边连接左腿和右手（协同运动部位），"站起来"时超边连接受力关节
5. **t-SNE 显示语义收敛**：Hyper-GCN 最后一层的关节特征高度聚类，说明信息交互更充分

## 亮点与洞察

1. **自适应超图是核心创新**：现有超图方法全部依赖固定结构，A-NHG 通过特征距离动态构建，实现了真正的数据驱动拓扑学习
2. **虚拟关节的"木偶提线"类比**：直观解释了为什么需要额外的信息载体来分担真实关节的压力，与 Transformer 中的 class token 也有异曲同工之妙
3. **极致的效率-性能平衡**：Base 版仅 1.1M/1.63G 就超越大量 GCN 方法，证明超图卷积的信息聚合效率确实更高
4. **散度损失的简洁有效**：用余弦相似度 + ReLU 截断的简单设计就解决了超关节同质化问题

## 局限与展望

1. **仅在标准数据集上验证**：NTU 和 NW-UCLA 主要是室内单人/双人动作，缺少复杂多人、户外场景的验证
2. **超关节数量固定**：当前手工设定为 3 个，能否根据动作复杂度自适应确定数量？
3. **仍然依赖多流集成**：4-stream 增加了整体系统复杂度和推理成本
4. **未与最新的 Mamba/状态空间模型对比**：骨架序列的时间建模可能有更高效的替代方案
5. **K 值对不同数据集的敏感性未分析**：当前仅在 NTU120 上做了 K 的消融

## 相关工作与启发

- **CTR-GCN (ICCV 2021)**：提出通道拓扑细化，本文在其基础上扩展到多关节超边
- **InfoGCN (CVPR 2022)**：信息瓶颈视角的 GCN，与 Hyper-GCN 的信息聚合效率互补
- **BlockGCN**：当前参数最少的 GCN SOTA，被 Hyper-GCN Base 以更少参数超越
- **SkateFormer (ECCV 2024)**：Transformer 路线的最新进展，性能强但计算量是 Hyper-GCN 的 2 倍多
- **超图神经网络 HGNN+ (TPAMI 2023)**：提供了超图卷积的通用框架，本文从中获得归一化方法的启发

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)
- [Heterogeneous Skeleton-Based Action Representation Learning](../../CVPR2025/video_understanding/heterogeneous_skeleton-based_action_representation_learning.md)
- [Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](../../NeurIPS2025/video_understanding/grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)

<!-- RELATED:END -->
