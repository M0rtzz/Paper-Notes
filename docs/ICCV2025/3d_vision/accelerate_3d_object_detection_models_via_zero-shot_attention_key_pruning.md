---
title: >-
  [论文解读] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning
description: >-
  [ICCV 2025][3D视觉][3D目标检测] 提出 tgGBC（trim keys gradually Guided By Classification scores），一种零样本运行时剪枝方法，利用分类分数与注意力图的乘积计算键重要性，逐层剪除不重要的键…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D目标检测"
  - "Transformer"
  - "注意力键剪枝"
  - "零样本加速"
  - "自动驾驶"
---

# Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning

**会议**: ICCV 2025  
**arXiv**: [2503.08101](https://arxiv.org/abs/2503.08101)  
**代码**: [https://github.com/iseri27/tg_gbc](https://github.com/iseri27/tg_gbc)  
**领域**: 3D视觉  
**关键词**: 3D目标检测, Transformer剪枝, 注意力键剪枝, 零样本加速, 自动驾驶

## 一句话总结
提出 tgGBC（trim keys gradually Guided By Classification scores），一种零样本运行时剪枝方法，利用分类分数与注意力图的乘积计算键重要性，逐层剪除不重要的键，在多个3D检测器上实现Transformer解码器近2×加速且性能损失<1%。

## 研究背景与动机
基于 Query 的3D目标检测方法（如 PETR、StreamPETR、ToC3D 等）使用多层 Transformer 解码器处理全景相机的密集特征，已成为 SOTA 范式。但密集方法的全局特征交互带来巨大计算开销，阻碍了在边缘设备上的部署。

现有加速方案的痛点：

**静态剪枝方法**（如 FastV、SparseViT）：需要额外运行模型进行参数搜索，且针对 ViT 分类模型设计，难以迁移到3D检测器

**运行时剪枝方法**（如 ToMe、Zero-TPrune）：ToMe 的相似度矩阵计算复杂度为 $O(N_k^2)$，对3D检测器的大token量（4224-24000）开销太大；Zero-TPrune 依赖方形注意力矩阵，但3D检测器中 Query 和 Key 数量不同，注意力矩阵非方阵

**本质矛盾**：3D检测的 token 数远超 ViT 模型（3D: 4224-24000 vs ViT: ~1024），现有方法直接迁移效率和效果都不理想

**核心洞察**：在3D检测器中，最终输出由分类分数最高的预测决定。因此，对高置信预测贡献小的键可以安全剪除。分类分数和注意力图都是 Transformer 解码器内部自然产生的，无需引入额外参数。

**核心 idea**：将分类分数扩展并与注意力图逐元素相乘，按列求和得到每个键的重要性分数，逐层剪除重要性最低的键。

## 方法详解

### 整体框架
tgGBC 模块插入在相邻 Transformer 解码器层之间，接收上一层的分类分数 $C \in \mathbb{R}^{N_q \times N_C}$ 和交叉注意力权重 $A \in \mathbb{R}^{N_q \times N_k}$，计算键重要性后剪除不重要的键。总共剪除 $r$ 个键，分 $n$ 层逐步进行，每层剪 $\lfloor r/n \rfloor$ 个。

### 关键设计

1. **键重要性计算**：

    - Step 1: 取每个 Query 的最大分类分数 $\hat{C}_i = \max_j C_{i,j}$
    - Step 2: 扩展为 $\tilde{C} \in \mathbb{R}^{N_q \times N_k}$（沿键维度重复）
    - Step 3: 逐元素乘 $S_0 = A \odot \tilde{C}$（将分类分数的质量传递给每个键）
    - Step 4: 选分类分数 top-k 的行得到 $S_1 \in \mathbb{R}^{k \times N_k}$
    - Step 5: 按列求和得到键重要性 $S_j = \sum_{i=1}^k (S_1)_{i,j}$
    - **设计动机**：不是所有 Query 都重要，只有高置信 Query 被选为最终预测。优先保留对这些高置信 Query 贡献大的键，避免"民主化"地给所有 Query 相同权重。

2. **逐层渐进剪枝**：不在单层一次性剪掉所有键，而是在前 $n$ 个 Transformer 层后各剪 $\lfloor r/n \rfloor$ 个键。

    - **设计动机**：渐进式剪枝让后续层有机会在已剪枝的键集上重新计算注意力，比一次性剪枝造成的信息损失更小。

3. **可行性保证**：注意力模块输出形状 $O \in \mathbb{R}^{N_q \times E}$ 不依赖 $N_k$，因此剪键不改变输出维度。只要保证 $K$ 和 $V$ 同步剪枝（3D检测器中 $V = K$），原有模型参数保持有效。

### 损失函数 / 训练策略
- **零训练**：tgGBC 不修改模型参数，不需要任何训练或微调
- 即插即用：仅在推理阶段添加剪枝层
- 额外计算开销极小：仅需一次矩阵逐元素乘法和列求和

## 实验关键数据

### 主实验

| 模型 | Backbone | $N_k$ | $r$ | mAP | NDS | Dec. Time (ms) | 加速比 |
|------|----------|-------|-----|-----|-----|----------------|--------|
| PETR | ResNet50 | 16896 | 0 | 31.74% | 0.367 | 47.09 | 1.00× |
| PETR+tgGBC | ResNet50 | 16896 | 12000 | 30.78% | 0.358 | 28.58 | **1.65×** |
| StreamPETR | VovNet | 24000 | 0 | 48.89% | 0.573 | 64.93 | 1.00× |
| StreamPETR+tgGBC | VovNet | 24000 | 21000 | 48.55% | 0.573 | 34.98 | **1.86×** |
| 3DPPE | VovNet | 6000 | 0 | 39.81% | 0.446 | 53.25 | 1.00× |
| 3DPPE+tgGBC | VovNet | 6000 | 3000 | 39.74% | 0.445 | 27.56 | **1.93×** |
| ToC3D | - | - | - | 最新模型 | - | - | **1.99×** |

在 ToC3D 上实现 Transformer 解码器 1.99× 加速，mAP 损失 < 1%。

### 消融实验

| 剪枝方法 | 模型 | mAP | 解码器耗时 |
|---------|------|-----|-----------|
| 无剪枝 | FocalPETR (VovNet) | 42.36% | 24.65ms |
| ToMe | FocalPETR (VovNet) | 41.82% | 22.35ms |
| **tgGBC** | FocalPETR (VovNet) | **42.38%** | **17.08ms** |
| 无剪枝 | StreamPETR (ResNet50) | 38.01% | 31.10ms |
| ToMe | StreamPETR (ResNet50) | 37.55% | 29.40ms |
| **tgGBC** | StreamPETR (ResNet50) | **37.93%** | **24.15ms** |

tgGBC 在 FocalPETR 上甚至提升了 mAP（42.38% vs 42.36%），表明适度的键剪枝可能起到正则化效果。

### 关键发现
- 在边缘设备（Orin）上部署 FocalPETR 和 StreamPETR 分别实现 1.18× 和 1.19× 推理加速
- tgGBC 可以在单层中剪除高达90%的键（$r$ 接近 $N_k$），这是 ToMe 做不到的（受二部匹配限制）
- 某些模型上 tgGBC 反而提升性能，说明冗余键实际上是"噪声"

## 亮点与洞察
- **真正的零样本**：不需要任何训练数据、参数搜索或微调，即插即用
- 利用模型已有的分类分数和注意力图（"免费"的信息），不引入额外参数
- 跨模型通用性好：在 PETR、FocalPETR、StreamPETR、3DPPE、MV2D、M-BEV、ToC3D 上均有效
- 与模型结构无关：只要使用密集全局注意力的 Transformer 解码器就可应用
- 在边缘设备上验证了实际部署效果

## 局限与展望
- 仅适用于使用全局注意力的密集方法，不适用于使用可变形注意力（Deformable Attention）或 Flash Attention 的稀疏方法
- 剪枝比例 $r$ 需要手动设定，未提供自动选择策略
- 仅在 nuScenes 数据集上评估，未在其他自动驾驶数据集（如 Waymo、Argoverse）上验证
- 时序模型（如 StreamPETR）的历史帧键可能也可以剪枝，但未深入探讨
- 自注意力中的 Query 剪枝未探索（当前只剪交叉注意力的键）

## 相关工作与启发
- **ToMe**：二部匹配合并 token，本文避免了 $O(N^2)$ 的相似度计算
- **Zero-TPrune**：Markov 链收敛作为剪枝准则，但要求方阵注意力图
- **3D检测器**：PETR 系列、OPEN、ToC3D 等 query-based 方法是主要应用对象
- **启发**：利用任务特定信号（分类分数）作为剪枝准则的思路，可推广到其他有类似分数信号的 Transformer 任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 巧妙利用分类分数加权注意力图进行键剪枝，方法简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 7个模型 × 多种配置，含边缘设备部署验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 即插即用的加速工具，对自动驾驶部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](../../ECCV2024/3d_vision/zero-shot_multi-object_scene_completion.md)
- [\[ICCV 2025\] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[CVPR 2025\] Cross-View Completion Models are Zero-shot Correspondence Estimators](../../CVPR2025/3d_vision/cross-view_completion_models_are_zero-shot_correspondence_estimators.md)

</div>

<!-- RELATED:END -->
