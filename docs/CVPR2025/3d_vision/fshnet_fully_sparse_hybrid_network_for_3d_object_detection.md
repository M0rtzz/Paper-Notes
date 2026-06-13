---
title: >-
  [论文解读] FSHNet: Fully Sparse Hybrid Network for 3D Object Detection
description: >-
  [CVPR 2025][3D视觉][3D目标检测] FSHNet 提出全稀疏混合网络，通过 SlotFormer（槽分区+线性注意力）建立全局范围的稀疏体素交互，配合动态稀疏标签分配和稀疏上采样模块，在 Waymo、nuScenes、Argoverse2 三大基准上超越现有稀疏和密集检测器。 领域现状：LiDAR 3D 目标…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D目标检测"
  - "稀疏检测器"
  - "线性注意力"
  - "动态标签分配"
  - "稀疏上采样"
---

# FSHNet: Fully Sparse Hybrid Network for 3D Object Detection

**会议**: CVPR 2025  
**arXiv**: [2506.03714](https://arxiv.org/abs/2506.03714)  
**代码**: [https://github.com/Say2L/FSHNet](https://github.com/Say2L/FSHNet)  
**领域**: 3D视觉  
**关键词**: 3D目标检测, 稀疏检测器, 线性注意力, 动态标签分配, 稀疏上采样

## 一句话总结
FSHNet 提出全稀疏混合网络，通过 SlotFormer（槽分区+线性注意力）建立全局范围的稀疏体素交互，配合动态稀疏标签分配和稀疏上采样模块，在 Waymo、nuScenes、Argoverse2 三大基准上超越现有稀疏和密集检测器。

## 研究背景与动机

**领域现状**：LiDAR 3D 目标检测分为密集检测器（将稀疏特征转为 2D 密集 BEV 再用 2D 检测 head）和稀疏检测器（全程只处理非空体素）。密集检测器浪费大量计算资源处理空体素，尤其在远距离检测时更严重。稀疏检测器（FSD、VoxelNeXt、SAFDNet）效率更高但性能仍有差距。

**现有痛点**：稀疏检测器面临两个核心挑战——(1) **长距离交互困难**：稀疏卷积只在非空体素上操作，两个远距离体素如果中间没有非空体素就完全无法交互，不像密集检测器可以通过中间空体素传递信息；(2) **中心特征缺失**：物体中心通常是空的（没有点云回波），尤其是大型物体，导致网络无法有效优化——许多方法依赖中心特征作为物体的代理。

**核心矛盾**：简单增大卷积核能扩大感受野但计算开销大；Transformer 有长距离交互能力但直接处理原始点云数量过大。现有稀疏检测器没有一个标准化的方案能同时解决特征提取能力弱和优化困难两个问题。

**本文目标** (1) 为稀疏检测器增加高效的全局交互能力；(2) 提供更多高质量正样本来优化检测网络；(3) 恢复下采样丢失的细粒度信息。

**切入角度**：不用 Transformer 处理原始点云（太慢），而是先用稀疏卷积高效提取和下采样特征，再在下采样后的少量体素上用注意力建立全局交互。用"槽"替代传统窗口分区来获得更大感受野。

**核心 idea**：稀疏卷积高效下采样 + SlotFormer 线性注意力全局交互 + 动态稀疏标签分配 + 稀疏上采样，四个组件互补构成完整的高性能稀疏检测器。

## 方法详解

### 整体框架
输入原始点云经 VFE 转为稀疏体素，稀疏卷积编码器逐步提取和下采样特征，SlotFormer 在下采样体素上建立全局交互，稀疏上采样模块恢复细粒度表示，最终动态稀疏检测头进行预测。有两个变体：FSHNet_light（VoxelNeXt 编码器 + 4层 SlotFormer）和 FSHNet_base（SAFDNet 编码器 + 8层 SlotFormer）。

### 关键设计

1. **SlotFormer（槽分区 + 线性注意力）**:

    - 功能：在全局场景级别建立稀疏体素间的长距离交互
    - 核心思路：将 BEV 场景沿 X 或 Y 轴切成若干"槽"（每个槽沿一个轴覆盖整个场景长度，另一轴宽度为 w）。在每个槽内用线性注意力处理所有非空体素：先计算 $Q=\phi(fW_q)$, $K=\phi(fW_k)$, $V=fW_v$（$\phi$ 为 ReLU），然后同一槽内的 K-V 乘积 $kv_j = \sum k_i^T \cdot v_i$ 求和聚合，每个体素再查询 $v_i' = q_i \cdot kv_{d_i} / (q_i \sum k_j^T)$。不同层交替 X/Y 方向的槽，形成全局交互
    - 设计动机：传统窗口分区受窗口大小限制，槽分区让一个方向上的感受野无限延伸。线性注意力将复杂度从 $O(N^2)$ 降到 $O(N)$，并天然支持变长输入（不同槽内的体素数量差异大）

2. **动态稀疏标签分配（Dynamic Sparse Label Assignment）**:

    - 功能：为每个 GT 框动态选择多个高质量正样本，改善网络优化
    - 核心思路：对每个 GT 框 $\mathbf{b}^t$，找距其中心最近的 $n$ 个体素作为候选集 $\mathcal{V}_b$。计算每个候选的选择代价 $c_i = \ell_{cls}(Pred(\nu_i), \mathbf{b}^t) + \lambda \ell_{reg}(Pred(\nu_i), \mathbf{b}^t)$，再用候选体素的 IoU 之和确定正样本数量 $k = \max(\lfloor\sum IoU\rfloor, 1)$，最终选代价最小的 top-k 体素作为正样本
    - 设计动机：现有稀疏检测器只分配离中心最近的 1 个体素作为正样本，排除了大量高质量候选，导致优化不充分。动态选择多个正样本且数量随检测质量自适应调整

3. **稀疏上采样模块（Sparse Upsampling）**:

    - 功能：恢复下采样过程中丢失的细粒度信息，提升小目标检测
    - 核心思路：先将体素坐标翻倍 $(x_i, y_i) \rightarrow (2x_i, 2y_i)$（等效于将体素大小减半），然后用一个 3×3 stride-1 的稀疏卷积扩散体素特征到邻近位置 $\mathcal{V}^{up} = SpConv(\mathcal{V}')$，实现体素精度恢复到原来的 1/4 并利用邻域特征补充细节
    - 设计动机：多步下采样虽然降低计算量但丢失小目标信息，稀疏上采样用卷积扩散既恢复分辨率又能缓解中心特征缺失问题（扩散到原来为空的位置）

### 损失函数 / 训练策略
- 分类损失：Focal Loss，正样本权重 1，候选负样本权重为预测与 GT 的 IoU，其余负样本权重 0
- 回归损失：Rotation-weighted IoU Loss
- Adam 优化器，学习率 0.003，从零开始端到端训练
- Waymo/AV2 训练 12 epochs，nuScenes 训练 36 epochs（不使用 CBGS）
- 2×A100 GPU，总 batch size 16

## 实验关键数据

### 主实验

**Waymo Open 验证集（LEVEL 2 mAP/mAPH）**:

| 方法 | 类型 | mAP/mAPH | Vehicle | Pedestrian | Cyclist |
|------|------|----------|---------|------------|---------|
| **FSHNet_base** | 稀疏 | **77.1/74.9** | 82.2/81.7 | 85.9/80.8 | 80.5/79.4 |
| SAFDNet | 稀疏 | 75.7/73.9 | 80.6/80.1 | 84.7/80.4 | 80.0/79.0 |
| ScatterFormer | 密集 | 75.7/73.8 | 81.0/80.5 | 84.5/79.9 | 79.9/78.9 |
| HEDNet | 密集 | 75.3/73.4 | 81.1/80.6 | 84.4/80.0 | 78.7/77.7 |
| DSVT | 密集 | 74.0/72.1 | 79.7/79.3 | 83.7/78.9 | 77.5/76.5 |

**nuScenes 验证集**:

| 方法 | 类型 | NDS | mAP |
|------|------|-----|-----|
| **FSHNet_base** | 稀疏 | **71.7** | **68.1** |
| DSVT | 密集 | 71.1 | 66.4 |
| SAFDNet | 稀疏 | 71.0 | 66.3 |

**Argoverse2 验证集（长距离检测 200m）**:

| 方法 | 类型 | mAP |
|------|------|-----|
| **FSHNet_base** | 稀疏 | **40.2** |
| SAFDNet | 稀疏 | 38.7 |
| HEDNet | 密集 | 37.1 |

### 消融实验

| SlotFormer | DSLA | SU | Vehicle | Pedestrian | Cyclist |
|:---:|:---:|:---:|---------|------------|---------|
| | | | 69.1/68.7 | 75.3/69.5 | 75.0/73.9 |
| ✓ | | | 70.3/69.9 | 75.9/70.5 | 76.2/75.1 |
| | ✓ | | 69.9/69.5 | 75.5/69.7 | 75.7/74.5 |
| | | ✓ | 69.3/68.8 | 76.6/71.0 | 75.2/74.1 |
| ✓ | ✓ | | 70.5/70.1 | 77.1/71.6 | 75.6/74.5 |
| ✓ | ✓ | ✓ | **72.5/72.0** | **77.9/72.6** | **77.2/76.1** |

**体素分区方式对比**:

| 分区方式 | Vehicle | Pedestrian | Cyclist |
|---------|---------|------------|---------|
| Slot + linear attn | **72.5/72.0** | **77.9/72.6** | **77.2/76.1** |
| Window + linear attn | 72.2/71.7 | 77.8/72.4 | 77.1/76.1 |
| Window + Set + linear attn | 72.0/71.5 | 77.7/72.3 | 77.1/76.0 |
| Window + Set + self-attn | 71.8/71.3 | 77.3/72.0 | 76.8/75.7 |

### 关键发现
- SlotFormer 对大物体（Vehicle、Cyclist）提升最大（+1.2/+1.2 AP），验证了全局交互对大物体检测的重要性
- 稀疏上采样对小物体（Pedestrian）贡献最大（+1.3 APH），体现了细粒度信息恢复的价值
- DSLA 在所有类别上都有一致提升，说明更多高质量正样本对稀疏检测器优化至关重要
- 槽分区优于窗口分区，线性注意力优于自注意力（更高效且效果更好）
- 三个组件完全互补——同时使用时提升远大于单独使用之和

## 亮点与洞察
- **槽分区设计**：将场景按一个轴方向切成无限长的条带，结合线性注意力实现 $O(N)$ 复杂度的全局交互——比窗口分区简单且感受野更大，交替 X/Y 方向后覆盖全场景
- **稀疏上采样的双重作用**：不仅恢复细粒度分辨率提升小目标检测，其卷积扩散还能在原来空的位置生成新体素，间接缓解中心特征缺失
- **动态正样本数量**：基于候选体素的 IoU 之和自适应确定 k 值——好的预测得到更多正样本形成正反馈，差的预测保守分配 1 个

## 局限与展望
- 仅在单帧设置下验证，未探索多帧时序融合的效果
- SlotFormer 的槽宽度 w 是手动设定的超参数（w=12），不同场景可能需要不同设置
- 线性注意力相比标准注意力在稀疏场景中损失的表达能力对最终检测精度的影响未深入分析
- 仅测试了 LiDAR-only 设置，与相机融合方案的结合有待探索

## 相关工作与启发
- **vs SAFDNet**: SAFDNet 用自适应特征扩散缓解中心特征缺失，但只在局部范围内扩散。FSHNet 的 SlotFormer 提供全局交互能力，稀疏上采样也能扩散生成新体素
- **vs DSVT/FlatFormer**: 它们使用窗口+排序的分区策略处理原始体素，受窗口大小限制。FSHNet 先用稀疏卷积下采样减少体素数量，再用槽分区做全局交互更高效
- **vs DCDet**: DCDet 的动态十字标签分配是为密集检测器设计的。FSHNet 的 DSLA 针对稀疏体素的特点，从最近候选体素中动态选择

## 评分
- 新颖性: ⭐⭐⭐⭐ SlotFormer 的槽分区思想新颖，与稀疏卷积的结合有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 三大数据集+验证集/测试集+详细消融+多种分区对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详实
- 价值: ⭐⭐⭐⭐ 为稀疏 3D 检测提供了新的标准化方案，有代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[CVPR 2025\] FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection](faster_focal_token_acquiring-and-scaling_transformer_for_long-term_3d_objection_.md)
- [\[CVPR 2025\] MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)
- [\[ECCV 2024\] Interactive 3D Object Detection with Prompts](../../ECCV2024/3d_vision/interactive_3d_object_detection_with_prompts.md)

</div>

<!-- RELATED:END -->
