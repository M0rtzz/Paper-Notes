---
title: >-
  [论文解读] DAMSDet: Dynamic Adaptive Multispectral Detection Transformer
description: >-
  [ECCV 2024][目标检测][红外-可见光检测] DAMSDet 提出一种基于 DETR 架构的动态自适应红外-可见光目标检测方法，通过模态竞争 Query 选择（为每个目标动态选择主导模态特征作为初始 query）和多光谱可变形交叉注意力（在多语义层级上自适应采样和聚合双模态特征），同时解决互补信息融合和模态未对齐两大挑战，在 4 个公开数据集上显著超越 SOTA。
tags:
  - ECCV 2024
  - 目标检测
  - 红外-可见光检测
  - 多光谱融合
  - DETR
  - 模态竞争选择
  - 可变形交叉注意力
---

# DAMSDet: Dynamic Adaptive Multispectral Detection Transformer

**会议**: ECCV 2024  
**arXiv**: [2403.00326](https://arxiv.org/abs/2403.00326)  
**代码**: [GitHub](https://github.com/gjj45/DAMSDet)  
**领域**: 目标检测  
**关键词**: 红外-可见光检测, 多光谱融合, DETR, 模态竞争选择, 可变形交叉注意力

## 一句话总结

DAMSDet 提出一种基于 DETR 架构的动态自适应红外-可见光目标检测方法，通过模态竞争 Query 选择（为每个目标动态选择主导模态特征作为初始 query）和多光谱可变形交叉注意力（在多语义层级上自适应采样和聚合双模态特征），同时解决互补信息融合和模态未对齐两大挑战，在 4 个公开数据集上显著超越 SOTA。

## 研究背景与动机

**领域现状**：红外-可见光目标检测通过融合双模态互补信息实现全天候鲁棒检测。红外成像不受光照、烟雾影响但缺乏纹理细节；可见光提供丰富纹理和颜色但受低光环境制约。

**现有方法局限**：
   - **互补特征融合困难**：同一场景中不同目标的互补特性差异巨大——有的目标在可见光中完全消失（烟雾遮挡），有的在两个模态中都有部分信息。全局权重或区域级融合方法粒度不够
   - **模态未对齐**：红外和可见光图像通常存在配准偏差，即使手动配准后目标仍有位移。AR-CNN 需要额外的配对框标注来学习偏移，代价高昂
   - **现有融合方式粗糙**：一步式融合或整图特征融合难以充分挖掘复杂场景中的互补信息

**核心思路**：借鉴人类观察模式——先关注各模态中显著的目标，再逐步聚合两模态关键信息。用 DETR 级联解码结构逐层精炼。

## 方法详解

### 整体框架

DAMSDet 包含四个主要组件：
1. **双模态特定 CNN 骨干**：分别提取红外和可见光特征
2. **双模态特定 Efficient Encoder**：参考 RT-DETR 的高效编码器，结合 Transformer 和 CNN 降低计算复杂度
3. **模态竞争 Query 选择（MCQS）**：从双模态编码特征中竞争选择显著模态特征作为初始 query
4. **多光谱 Transformer 解码器**：通过多光谱可变形交叉注意力在多个语义层级上逐步精炼 query

### 关键设计

**1. 模态竞争 Query 选择（Modality Competitive Query Selection）**

- 将双模态编码特征序列拼接，通过线性投影层获取每个特征点的得分
- 选择 Top-K 最高分特征作为初始 query：$z = \text{Top-}K(\text{Linear}(\text{concat}(I, V)))$
- 每个 query 来自某一个模态（红外或可见光），代表该模态中的一个目标实例
- **竞争机制**：自动为每个目标选择信号更强的"主导模态"，避免早期引入干扰信息
- 配合 IoU-aware 分类损失进一步提升选择质量
- 冗余 query（双模态指向同一目标）通过 DETR 的一对一匹配和 self-attention 自然消除
- 训练时引入 Noise Query Learning 帮助学习最优模态匹配
- 可视化验证：不同目标确实被不同主导模态选择，与直觉一致

**2. 多光谱 Transformer 解码器**

每层解码器包含：
- **Multi-head Self-Attention**：获取上下文信息并减少冗余
- **多光谱可变形交叉注意力（MDCA）**：核心融合模块
- **4D 锚框约束**：使用 4D 参考点 $(x,y,w,h)$ 约束采样范围，逐层迭代精炼：$b_{q}^{d} = \sigma(MLP^d(z_q^d) + \sigma^{-1}(b_q^{d-1}))$

**3. 多光谱可变形交叉注意力（MDCA）**

- 将 Deformable DETR 的可变形注意力扩展到多模态形式
- 对双模态多语义层级特征图自适应稀疏采样和加权聚合
- $m \in \{1,2\}$ 分别表示可见光和红外，$l$ 索引语义层级，$k$ 索引采样点
- 每个模态**独立预测采样位置偏移**，天然适应模态未对齐
- 注意力权重在双模态、多语义层级、多采样点上归一化：$\sum_m \sum_l \sum_k A_{mhlqk} = 1$
- 偏移约束在参考框范围内，聚焦目标周围信息
- **可视化分析**：
    - 解码器深层倾向于关注红外模态的低级语义特征（基本轮廓）和可见光的高级语义特征（类别关系）
    - 烟雾遮挡目标主要聚焦红外；良好光照目标主要聚焦可见光
    - 采样点能自适应对齐到未对齐的目标位置

### 损失函数 / 训练策略

- 跟随 DETR-like 检测器的训练损失：$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{dn}$
- $\mathcal{L}_{cls}$：IoU-aware 分类损失（RT-DETR）
- $\mathcal{L}_{box}$：L1 损失 + GIoU 损失
- $\mathcal{L}_{dn}$：去噪训练损失（DN-DETR）
- 每层解码器均计算辅助优化损失
- 骨干使用 COCO 预训练权重，ResNet50，训练 20-50 epochs
- 设置：H=8 注意力头，K=4 采样点，N=300 queries

## 实验关键数据

### 主实验

**M3FD 数据集（最具挑战性，多场景多类别）**：

| 方法 | 骨干 | mAP50 | mAP75 | mAP |
|------|------|:---:|:---:|:---:|
| Yolov7 (RGB) | CSPDarknet53 | 69.0 | - | 42.7 |
| DINO (RGB) | ResNet50 | 73.3 | 48.2 | 46.3 |
| CFT | CSPDarknet53 | 68.2 | 44.6 | 42.5 |
| ICAFusion | CSPDarknet53 | 67.8 | 44.5 | 41.9 |
| **DAMSDet** | ResNet50 | **80.2** | **56.0** | **52.9** |

在 M3FD 上较 CFT 提升 12.0% mAP50、10.4% mAP。

**FLIR 数据集**：

| 方法 | mAP50 | mAP |
|------|:---:|:---:|
| ICAFusion | 79.2 | 41.4 |
| LRAF-Net | 80.5 | 42.8 |
| **DAMSDet** | **86.6** | **49.3** |

较最佳方法提升 6.1% mAP50、6.5% mAP。

### 消融实验

在 M3FD 上的模块消融：

| 配置 | mAP50 | mAP |
|------|:---:|:---:|
| 基线（双模态加法融合 + 标准 query） | 77.8 | 51.6 |
| + MCQS（模态竞争选择） | 78.9 | 52.3 |
| + MDCA（多光谱可变形交叉注意力） | 79.4 | 52.5 |
| + MCQS + MDCA + CQS | **80.2** | **52.9** |

- MCQS 贡献 +0.7% mAP：避免早期引入干扰模态信息
- MDCA 贡献 +0.9% mAP：细粒度多层级互补信息挖掘
- CQS（内容 query 选择）在多模态场景中提供更强先验

### 关键发现

- **单模态 vs 多模态差异大**：仅用红外在 M3FD 上 mAP=35.0，加入可见光融合后达到 52.9
- **级联结构有效**：不同解码层的采样位置和权重分布差异明显，验证了逐层精炼互补信息的必要性
- **模态竞争选择符合直觉**：可视化显示低光条件选红外、清晰条件选可见光
- **在小目标 VEDAI 上 mAP50 提升 5.6%**，但 mAP 略低于 CNN 方法

## 亮点与洞察

- **同时解决融合和对齐**：MDCA 在单个模块中统一处理互补信息融合和模态未对齐，比分离处理更高效
- **动态竞争替代全局融合**：为每个目标独立选择主导模态，避免一刀切的融合策略
- **多语义层级融合**：不同语义层级的互补特性也是动态变化的（红外→低级，可见光→高级）
- **无需额外配对标注**：不像 AR-CNN 需要双模态配对框标注来学习对齐

## 局限与展望

- 极端未对齐（目标超出 4D 参考点范围）时性能下降
- 在小目标检测上 Transformer 的框回归精度不如 CNN，mAP 指标有时不及 LRAF-Net
- 骨干采用 ResNet50，未探索更强骨干或更大 DETR 变体
- 可结合红外-可见光图像配准方法进一步提升

## 相关工作与启发

- **Deformable DETR**：MDCA 的直接基础，将可变形注意力扩展到双模态
- **RT-DETR**：借鉴其 Efficient Encoder 和 IoU-aware Query Selection
- **DINO**：借鉴其级联结构和 DN 训练策略
- **启发**：DETR 家族的 query 机制天然适合多模态场景——可以竞争性选择最优模态

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](layoutdetr_detection_transformer_is_a_good_multimodal_layout.md)
- [AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)
- [HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)

<!-- RELATED:END -->
