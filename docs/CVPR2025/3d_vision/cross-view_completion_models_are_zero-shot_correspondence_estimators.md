---
title: >-
  [论文解读] Cross-View Completion Models are Zero-shot Correspondence Estimators
description: >-
  [CVPR 2025][3D视觉][跨视图补全] 揭示跨视图补全（CVC）模型中交叉注意力图（cross-attention map）本质上学到了精确的稠密对应关系，提出ZeroCo在零样本匹配和学习型几何匹配中利用这一发现，显著超越基于编码器/解码器特征的常规用法。
tags:
  - CVPR 2025
  - 3D视觉
  - 跨视图补全
  - 零样本匹配
  - 交叉注意力
  - 代价体
  - 深度估计
---

# Cross-View Completion Models are Zero-shot Correspondence Estimators

**会议**: CVPR 2025  
**arXiv**: [2412.09072](https://arxiv.org/abs/2412.09072)  
**代码**: [cvlab-kaist.github.io/ZeroCo](https://cvlab-kaist.github.io/ZeroCo/)  
**领域**: 3D视觉 / 视觉对应  
**关键词**: 跨视图补全, 零样本匹配, 交叉注意力, 代价体, 深度估计

## 一句话总结

揭示跨视图补全（CVC）模型中交叉注意力图（cross-attention map）本质上学到了精确的稠密对应关系，提出ZeroCo在零样本匹配和学习型几何匹配中利用这一发现，显著超越基于编码器/解码器特征的常规用法。

## 研究背景与动机

- 跨视图补全（CroCo/CroCo-v2）已成为强大的自监督预训练任务，但其成功机制尚不清楚
- 现有利用CVC知识的方法（DUSt3R、MASt3R、CroCo-Flow）都只使用解码器特征作为下游表示
- 关键発现：CVC模型中**交叉注意力图比编码器/解码器特征更准确地捕获了几何对应关系**
- 从类比角度看：CVC的学习过程与自监督对应学习（光流、立体深度）高度相似——都是通过源视图特征重建目标视图

## 方法详解

### 整体框架

基于CroCo-v2预训练模型，直接利用其解码器中交叉注意力层的注意力图作为代价体，无需任何训练即可实现零样本稠密对应估计。进一步提出轻量级学习模块（代价聚合+上采样）构建学习型匹配和深度估计模型。

### 关键设计

1. **交叉注意力图作为代价体（ZeroCo零样本）**:
    - 功能：将CVC模型的交叉注意力图直接用于零样本对应估计
    - 核心思路：从原始和交换输入对分别提取softmax前的交叉注意力图 $C^l(i,j) = D_t^{l,Q}(i) \cdot D_s^{l,K}(j)$，跨层平均后融合 $C' = \frac{1}{L}\sum_l C^l + (\frac{1}{L}\sum_l C^l_{\text{swap}})^T$ 以强制互质性（reciprocity），最终通过soft-argmax得到flow
    - 设计动机：CVC的交叉注意力本质上在学习"如何从源视图warp特征到目标视图"，等价于学习对应关系

2. **Register Token修正**:
    - 功能：消除注意力图中由register token引起的伪影
    - 核心思路：将register token对应的注意力值替换为最小注意力值
    - 设计动机：Transformer中的register token导致快捷学习（shortcut learning），产生注意力集中在register而非正确位置的伪影

3. **学习型扩展（ZeroCo-flow / ZeroCo-depth）**:
    - 功能：在冻结的交叉注意力图上添加轻量级头实现细粒度匹配和深度估计
    - 核心思路：ZeroCo-flow在交叉注意力图上应用代价聚合 $\mathcal{T}_c$ 和沿目标轴的上采样 $\mathcal{U}$ 得到高分辨率flow；ZeroCo-depth将聚合后的注意力图输入DPT head预测深度
    - 设计动机：零样本交叉注意力图分辨率粗，添加浅层学习头解决细粒度问题

### 损失函数 / 训练策略

- ZeroCo零样本：无需训练，直接使用CroCo-v2预训练权重
- ZeroCo-flow：标准对应回归损失训练
- ZeroCo-finetuned：微调交叉注意力图本身
- ZeroCo-depth：重投影、外观一致性和平滑损失（自监督深度常用损失）

## 实验关键数据

### 主实验（零样本匹配 HPatches-240）

| 方法 | AEPE ↓ (I) | AEPE ↓ (V) | AEPE ↓ (Avg) |
|------|-----------|-----------|------------|
| DINOv2 (Correlation) | 18.81 | 36.60 | 28.08 |
| DIFT_SD (Correlation) | 15.89 | 40.34 | 29.06 |
| CroCo Encoder (Corr.) | 39.69 | 54.63 | 47.52 |
| CroCo Decoder (Corr.) | 32.38 | 54.84 | 44.63 |
| **ZeroCo (Cross-attn)** | **5.07** | **13.26** | **9.41** |

### 消融实验

| 配置 | HPatches AEPE ↓ | 说明 |
|------|----------------|------|
| 仅正向注意力 | 较高 | 缺少互质性约束 |
| 带register token | 有伪影 | 注意力集中异常 |
| 去register + 双向融合 | **最优** | 互质性+伪影消除 |
| 编码器特征相关性 | 47.52 | 几何信息被稀释 |
| 解码器特征相关性 | 44.63 | 比编码器好但远不如交叉注意力 |

### 关键发现

- 交叉注意力图在零样本设置下的AEPE仅为9.41，远优于DINOv2（28.08）和扩散模型特征（29.06）
- ETH3D零样本匹配平均AEPE从最佳基线的25.69降至12.72
- 以CroCo同样的编码器/解码器特征构建的相关性图表现很差，进一步证明几何知识主要编码在交叉注意力中
- 添加极轻量级头后在学习型匹配和多帧深度估计中也取得竞争性结果
- 对动态物体表现鲁棒（不依赖极线几何）

## 亮点与洞察

- 核心发现极具洞察力：交叉注意力 ≈ 隐式学到的代价体，CVC ≈ 自监督对应学习
- 挑战了现有做法（DUSt3R用解码器特征），证明交叉注意力才是真正承载几何知识的组件
- ZeroCo设计极简：仅互质性融合+register修正，无需训练即达SOTA级零样本匹配
- 类比清晰：CVC中的源→目标warp与光流/立体匹配中的源→目标对齐完全对应
- 为CVC预训练模型的使用方式提供了新范式

## 局限与展望

- 交叉注意力图分辨率受ViT patch size限制（通常16×16），细粒度匹配仍需额外学习头
- 零样本推理需要正向和交换两次前向传播
- 仅在CroCo-v2上验证，未探索其他CVC模型（如DUSt3R decoder的交叉注意力）
- 可探索交叉注意力图在其他3D任务（如位姿估计、3D重建）中的直接应用

## 相关工作与启发

- 揭示了DUSt3R/MASt3R成功的可能机制：交叉注意力中的对应先验
- 与DIFT等扩散模型特征匹配方法形成对比，CVC专为几何任务设计效果更好
- Register token修正与ViT中的artifact研究（Darcet et al.）相关
- 启发：预训练模型中可能存在比我们常用的更好的特征表示，需要深入分析内部机制

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 揭示了一个被忽略但极重要的发现
- 实验充分度: ⭐⭐⭐⭐⭐ 零样本+学习型+深度估计全面验证，可视化丰富
- 写作质量: ⭐⭐⭐⭐⭐ 类比驱动的叙事优雅，图表清晰
- 价值: ⭐⭐⭐⭐⭐ 改变CVC预训练模型的使用范式，影响深远

<!-- RELATED:START -->

## 相关论文

- [MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](genpc_zero-shot_point_cloud_completion_via_3d_generative_priors.md)
- [FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [Zero-Shot Multi-Object Scene Completion](../../ECCV2024/3d_vision/zero-shot_multi-object_scene_completion.md)
- [3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)

<!-- RELATED:END -->
