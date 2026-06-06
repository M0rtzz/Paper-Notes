---
title: >-
  [论文解读] Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable
description: >-
  [AAAI 2026][3D视觉][3D 目标检测] 提出 GPQ（Gradually Pruning Queries），通过分类分数逐步裁剪 DETR 系 3D 检测器中大量冗余的 object queries，无需额外可学习参数，可直接在预训练 checkpoint 上微调完成…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3D 目标检测"
  - "DETR"
  - "剪枝"
  - "模型压缩"
  - "自动驾驶"
---

# Redundant Queries in DETR-Based 3D Detection: Unnecessary and Prunable

**会议**: AAAI 2026  
**arXiv**: [2412.02054](https://arxiv.org/abs/2412.02054)  
**代码**: 待确认  
**领域**: 3D视觉 / 3D目标检测  
**关键词**: 3D 目标检测, DETR, Query Pruning, 模型压缩, 自动驾驶

## 一句话总结

提出 GPQ（Gradually Pruning Queries），通过分类分数逐步裁剪 DETR 系 3D 检测器中大量冗余的 object queries，无需额外可学习参数，可直接在预训练 checkpoint 上微调完成，在边缘设备上最高实现 67.86% FLOPs 减少和 65.16% 推理时间下降。

## 研究背景与动机

### 问题背景

DETR 系列方法在 3D 目标检测中被广泛使用，其核心机制是通过预定义的 object queries 在 transformer 层中与图像特征交互来产生检测结果。然而，这些方法通常需要设置远超实际目标数量的 queries（如 900 个），而 nuScenes 等场景中待检测目标通常不超过 100 个。这导致正负样本比可达 8:1，大量 queries 在匈牙利匹配中被反复匹配为负样本，分类分数被持续压低。

### 核心观察

作者统计了 PETR、PETRv2、FocalPETR、StreamPETR 等方法在推理时各 query 被选为最终结果的频率，发现分布极度不均衡：少量 query 承担了绝大部分检测任务，而很多 query 几乎从未被选中为最终预测结果，甚至在 PETR 中有完全未被选中的 query。

### 现有方法的局限

传统 transformer 剪枝方法（如注意力头剪枝、token 剪枝）难以直接应用于 3D 检测：

- **剪枝目标不存在**：3D 检测中的注意力头是通过 reshape 实现的，修改数量不影响计算量
- **结构不一致**：3D 检测中 query 和 key 维度不等（$N_q \neq N_k$），注意力矩阵非方阵
- **数据量差异**：3D 检测产生的 token 数远多于 ViT（至少 4000 个 vs. 不到 200 个），token 剪枝开销过大

## 方法详解

### 核心思路

将每个 query 视为最小剪枝单元，以分类分数作为剪枝标准。分类分数最低的 query 贡献最小，优先被移除。

### GPQ 算法流程

1. **加载预训练 checkpoint**：从包含大量 queries 的已训练模型出发
2. **正常前向推理**：每次迭代后获取各 query 的分类分数
3. **定期剪枝**：每 $n$ 次迭代触发一次剪枝，选出分类分数最低的 query 并永久移除
4. **重复直至目标数量**：从初始 $N_q$ 逐步减少到 $N_q'$

整个过程不引入任何额外可学习参数，也不需要 learnable binary mask，可在几个 epoch 内完成。

### 理论分析：为什么剪枝有效

query 之间的独立性是关键。在 MLP 和 cross-attention 中，query 矩阵 $Q$ 只出现一次，按矩阵乘法的行独立性（$AB \equiv \text{Concat}_{i}(A_i B)$），删除某行不影响其他行的结果。唯一的影响来自 self-attention——因为 $Q$ 同时作为 query、key 和 value。但作者论证 self-attention 对图像特征的间接采样影响远小于 cross-attention 的直接交互，因此移除低贡献 query 产生的干扰很小。

### 为什么不直接用少量 queries 训练

作者可视化了参考点分布：从 900 剪枝到 300 的 query 仍保持聚集、有序的分布（继承了大规模训练的知识），而直接用 300 query 训练则分布散乱，表示能力较弱。GPQ 还可从一个 checkpoint 灵活生成不同 query 数量的模型版本。

## 实验

### 实验设置

- **数据集**：nuScenes（23000+ 样本，6 个环视相机，10 个类别）
- **检测器**：DETR3D、PETR、PETRv2、FocalPETR、StreamPETR、RayDN
- **评估指标**：mAP、NDS、各类误差指标（mATE/mASE/mAOE/mAVE/mAAE）、FPS、GFLOPs

### 主要结果（Table 2）

| 模型 | Backbone | Queries | mAP | NDS | FPS |
|------|----------|---------|-----|-----|-----|
| PETR | ResNet50 | 900/- | 31.74% | 0.3668 | 6.9 |
| PETR | ResNet50 | 300/-(从头训) | 31.19% | 0.3536 | 8.9 |
| PETR | ResNet50 | 900→300(GPQ) | **32.85%** | **0.3884** | 8.9 |
| PETR | ResNet50 | 900→150(GPQ) | 30.52% | 0.3671 | 9.3 |
| StreamPETR | ResNet50 | 900/- | 37.83% | 0.4734 | 16.1 |
| StreamPETR | ResNet50 | 300/-(从头训) | 33.62% | 0.4429 | 18.5 |
| StreamPETR | ResNet50 | 900→300(GPQ) | **39.42%** | **0.4941** | 18.7 |
| FocalPETR | ResNet50 | 900/- | 32.44% | 0.3752 | 16.4 |
| FocalPETR | ResNet50 | 900→300(GPQ) | **33.17%** | **0.3925** | 19.6 |

关键发现：PETR、FocalPETR、StreamPETR 通过 GPQ 将 900 剪枝到 300 后，mAP 甚至**超过**了用 900 queries 从头训练的基线。PETR 加速达 1.35x。

### 边缘设备部署结果（Table 3 - Jetson Nano B01）

| 模型 | Backbone | Queries | GFLOPs | 时间(ms) | FLOPs 减少 | 时间减少 |
|------|----------|---------|--------|----------|-----------|---------|
| StreamPETR | ResNet18 | 900 | 172.08 | 1520 | - | - |
| StreamPETR | ResNet18 | 900→300 | 123.90 | 916 | 28.00% | 39.74% |
| StreamPETR | ResNet18 | 900→150 | 112.51 | 791 | 34.62% | 47.96% |
| StreamPETR | w/o backbone | 900 | 87.78 | 1030 | - | - |
| StreamPETR | w/o backbone | 900→150 | 28.21 | 359 | **67.86%** | **65.16%** |

去除 backbone 后纯 transformer 部分加速更为显著，表明 GPQ 精确作用于计算瓶颈。

## 消融实验

- **剪枝标准**（Table 5）：按最高分类分数剪枝（GPQ-H）性能显著下降（mAP 34.34%），用匹配 cost 剪枝（GPQ-C）达 38.78%，而 GPQ 按最低分类分数剪枝最优（39.42%）
- **渐进 vs. 一步剪枝**：一次性剪掉 600 个 query（GPQ-1）mAP 仅 35.71%，远低于渐进策略的 39.42%，验证了逐步剪枝的必要性
- **与其他方法比较**（Table 4）：ToMe（token merging）在 3D 检测上反而变慢（相似度矩阵计算开销太大），GBC 可加速但会导致检测精度下降；GPQ 兼顾速度和精度
- **完全收敛模型**（Table 6）：对训练 90 epoch 的 StreamPETR 做 GPQ，300-query 仍优于从头训 300-query 90 epoch 的模型
- **训练同步剪枝**（Table 7）：GPQ 可在训练过程中同步执行，无需先完整训练再剪枝

## 亮点与创新

- **极简有效**：不引入任何可学习参数，仅靠分类分数排序+渐进删除，实现无损甚至涨点的 query 剪枝
- **即插即用**：作为微调步骤可直接接入任何 DETR 系检测器的预训练 checkpoint，一个 checkpoint 可灵活导出多个轻量版本
- **首次聚焦 query 冗余**：系统分析了 3D 检测中 query 选择频率的不均衡现象，填补了该方向的空白
- **边缘部署友好**：在 Jetson Nano 上验证了显著的实际加速效果

## 局限性

- 仅在 nuScenes 数据集上验证，未涉及 Waymo、KITTI 等其他 3D 检测基准
- 方法依赖分类分数作为剪枝准则，对于分类分数分布均匀的场景可能效果下降
- 未考虑 query 的空间分布——仅按分数剪枝可能导致某些空间区域覆盖不足
- 边缘设备实验使用随机 dummy 输入而非真实数据，实际推理加速可能受 I/O 等因素影响
- 对 2D 检测（ConditionalDETR）仅做了初步验证，通用性有待更广泛验证

## 相关工作

- **DETR 系 3D 检测器**：PETR、PETRv2、StreamPETR、FocalPETR、Far3D、DETR3D 等，均使用预定义 queries 与图像特征交互
- **Transformer 剪枝方法**：注意力头剪枝（Michel et al.）、层随机丢弃（Fan et al.）、ViT 稀疏性探索（Chen et al.）、宽度深度联合剪枝（ZipLM）、token 剪枝（EViT）等
- **Token Merging/Pruning**：ToMe（ICLR 2023）合并相似 token，但在 3D 检测中因 token 数量巨大导致开销过高
- **GBC**（ICCV 2025）：可加速但会导致检测精度下降

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |

总评：⭐⭐⭐⭐ — 方法极简但切中实际痛点，实验覆盖多种检测器和部署场景，对工业界部署 DETR 系检测器有直接参考价值。新颖性主要体现在"发现并系统验证 query 冗余"这一 observation，技术本身较为直观。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)
- [\[ECCV 2024\] SEED: A Simple and Effective 3D DETR in Point Clouds](../../ECCV2024/3d_vision/seed_a_simple_and_effective_3d_detr_in_point_clouds.md)
- [\[AAAI 2026\] Real-Time 3D Object Detection with Inference-Aligned Learning](real-time_3d_object_detection_with_inference-aligned_learning.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)

</div>

<!-- RELATED:END -->
