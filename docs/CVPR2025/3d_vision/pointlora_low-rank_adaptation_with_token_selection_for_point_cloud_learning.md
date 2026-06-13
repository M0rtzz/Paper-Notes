---
title: >-
  [论文解读] PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning
description: >-
  [CVPR 2025][3D视觉][点云学习] PointLoRA 将低秩适配 (LoRA) 与多尺度 token 选择结合，为点云预训练模型提供了一种简单高效的参数微调方案，仅用 3.43% 的可训练参数即达到与全量微调竞争的性能，在 ScanObjectNN、ModelNet40 和 ShapeNetPart 上均取得 SOTA 或接近 SOTA 的结果。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "点云学习"
  - "参数高效微调"
  - "LoRA"
  - "Token选择"
  - "预训练模型"
---

# PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning

**会议**: CVPR 2025  
**arXiv**: [2504.16023](https://arxiv.org/abs/2504.16023)  
**代码**: [https://github.com/songw-zju/PointLoRA](https://github.com/songw-zju/PointLoRA)  
**领域**: 3D视觉  
**关键词**: 点云学习, 参数高效微调, LoRA, Token选择, 预训练模型

## 一句话总结

PointLoRA 将低秩适配 (LoRA) 与多尺度 token 选择结合，为点云预训练模型提供了一种简单高效的参数微调方案，仅用 3.43% 的可训练参数即达到与全量微调竞争的性能，在 ScanObjectNN、ModelNet40 和 ShapeNetPart 上均取得 SOTA 或接近 SOTA 的结果。

## 研究背景与动机

**领域现状**：点云自监督预训练（如 Point-MAE、Point-BERT、ReCon）通过掩码重建或对比学习获得强大的表征，下游任务通常采用全量微调。参数高效微调 (PEFT) 方法开始被引入点云领域，如 IDPT（实例感知动态提示）、DAPT（提示+适配器）、PPT（位置提示调优）。

**现有痛点**：全量微调破坏预训练知识、存储多份权重成本高。现有 PEFT 方法依赖复杂的适配器和精巧的提示设计，可调参数量仍然偏多（如 IDPT 7.69%、DAPT 4.97%）。PPT 的位置提示将序列长度翻倍，显著增加计算量。

**核心矛盾**：点云数据同时包含全局结构信息和关键的局部几何特征。LoRA 作为 MLP 类结构擅长捕获全局特征（类似 PointNet 的全连接聚合），但缺乏局部信息提取能力——而局部细节对下游 3D 任务至关重要。

**本文目标**：设计一种更加简洁且参数更少的 PEFT 方法，同时捕获点云的全局和局部特征。

**切入角度**：LoRA 的两个低秩矩阵 $W_u \cdot W_d$ 本质上等价于 PointNet 的全连接层，天然适合点云全局特征提取。配合多尺度 token 选择补充局部信息，可以低成本地实现全局+局部互补。

**核心 idea**：在点云 transformer 的 QKV 投影和 FFN 层嵌入 LoRA，同时通过多尺度 token 选择提取关键局部特征作为提示，用共享的 Prompt MLP 编码后与 LoRA 输出融合。

## 方法详解

### 整体框架

输入点云 $P \in \mathbb{R}^{N \times 3}$ 经过 Point Tokenizer（FPS + kNN + mini-PointNet）生成 token 序列，与 CLS token 拼接后送入 L 层 Transformer。在每层 Transformer 的 QKV 投影和 FFN 中嵌入 LoRA 适配，同时从原始点云通过多尺度 token 选择模块提取局部提示 token，拼接到序列中。预训练参数全部冻结，仅训练 LoRA 矩阵、Mask Predictor 和共享 Prompt MLP。

### 关键设计

1. **Vanilla LoRA 基线在点云 Transformer 中的应用**:

    - 功能：在冻结预训练权重的基础上，通过低秩矩阵适配 QKV 和 FFN 层
    - 核心思路：将预训练权重 $W_p$ 更新为 $W_{\text{update}} = W_p + W_u \cdot W_d$，其中 $W_u \in \mathbb{R}^{d \times r}$, $W_d \in \mathbb{R}^{r \times d}$, $r \ll d$。训练时只更新 $W_u, W_d$，推理时合并为单一权重矩阵 $W_{\text{infer}} = W_p + \Delta W$ 无额外开销。LoRA 嵌入到参数最密集的 QKV 投影和 FFN 层中
    - 设计动机：QKV 和 FFN 占点云 transformer 参数量的绝大部分。LoRA 的 MLP 结构与 PointNet 原理天然契合——两个低秩矩阵相当于对无序点集做置换不变的全局特征提取

2. **多尺度 Token 选择模块**:

    - 功能：从原始点云中提取不同粒度的局部几何特征，并选择最有信息量的 token 作为提示
    - 核心思路：在 M 个不同尺度上执行 FPS（不同的中心点数 $N_1, ..., N_M$），对每个中心用 kNN 聚邻域后通过共享 mini-PointNet 编码为 token。Mask Predictor（两层 MLP + Sigmoid）为每个 token 打重要性分 $s^m = \text{Sigmoid}(\text{MLP}(T_p^m))$，通过 Top-K 选择保留最关键的 $N'_m$ 个 token。实验中使用 2 个尺度：(128 中心, 选 32 token) 和 (64 中心, 选 8 token)
    - 设计动机：不是所有局部区域都对下游任务有用，自适应选择可以在保持紧凑性的同时引入关键的局部先验。不同尺度捕获不同粒度的结构——较大尺度关注整体布局，较小尺度关注精细结构

3. **局部几何提示与 LoRA 融合**:

    - 功能：将选择的局部 token 与 LoRA 的全局适配无缝结合
    - 核心思路：选择的 $N_s$ 个 token $S_p$ 与原始输入拼接，在每个 LoRA 层中通过一个共享的 Prompt MLP（带 GELU 激活）编码后与 LoRA 输出相加：$O_{\text{update}} = \text{Prompt MLP}(T_{\text{input}}, S_p) + \Delta W \cdot (T_{\text{input}}, S_p)$。QKV 和 FFN 的 Prompt MLP 独立配置但跨所有层共享
    - 设计动机：跨层共享 MLP 进一步减少参数。将局部提示注入每个 LoRA 层而非仅在输入端，确保了每层决策都有局部信息参与

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{mask}}$，其中 $\mathcal{L}_{\text{task}}$ 是任务损失（分类或分割），$\mathcal{L}_{\text{mask}}$ 是 Mask Predictor 的正则化损失（二值交叉熵形式，鼓励分数向 0/1 极化）。$\lambda = 0.004$。使用 Point-MAE 预训练模型时训练 300 epochs，初始学习率 $5 \times 10^{-4}$，权重衰减 0.05。

## 实验关键数据

### 主实验

ScanObjectNN + ModelNet40（Point-MAE 预训练）：

| 方法 | 可调参数 | OBJ-BG | OBJ-ONLY | PB-T50-RS | ModelNet40 |
|------|----------|--------|----------|-----------|------------|
| Point-MAE (Full-FT) | 22.1M (100%) | 90.02 | 88.29 | 85.18 | 93.2 |
| IDPT | 1.7M (7.69%) | 91.22 | 90.02 | 84.94 | 93.3 |
| DAPT | 1.1M (4.97%) | 90.88 | 90.19 | 85.08 | 93.5 |
| PPT | 1.04M (4.57%) | 89.84 | 88.98 | 84.45 | 93.2 |
| **PointLoRA** | **0.77M (3.43%)** | 90.71 | 89.33 | **85.53** | 93.3 |

Few-shot Learning (ReCon 预训练)：

| 方法 | 5-way 10-shot | 5-way 20-shot | 10-way 10-shot | 10-way 20-shot |
|------|---------------|---------------|----------------|----------------|
| ReCon (Full-FT) | 97.3 | 98.9 | 93.3 | 95.8 |
| **ReCon+PointLoRA** | 96.9 | **98.8** | 92.7 | **95.8** |

### 消融实验

t-SNE 可视化（PB-T50-RS on ScanObjectNN）显示 PointLoRA 的聚类边界比全量微调更清晰，特别是在细粒度类别的分离上。

关键组件贡献（从论文中的 Overcoming Vanilla LoRA 分析）：
- 纯 LoRA → +多尺度 token (coarse) → +多尺度 token (fine) → +Mask Selection：逐步提升

### 关键发现

- 仅 3.43% 的参数即可在最难的 PB-T50-RS 上超过全量微调 0.35%（85.53 vs 85.18）
- LoRA 在点云上的分析：低秩矩阵等价于 PointNet 的全连接聚合，天然适合无序点集的置换不变特征提取
- 多尺度 token 选择有效补充了 LoRA 缺失的局部信息，且动态选择比使用全部 token 更好
- 推理时 LoRA 可合并回原权重，唯一新增开销仅为共享 Prompt MLP 和小型 Mask Predictor
- 方法可无缝集成到不同的预训练模型（Point-MAE 和 ReCon）

## 亮点与洞察

- 将 LoRA 与 PointNet 建立类比是一个有趣且有启发性的分析角度
- 设计极简但有效：仅 0.77M 参数就超越了使用复杂适配器和提示的方法
- 多尺度 token 选择作为局部几何提示是对 LoRA 全局特征提取的精准补充
- 跨层共享 Prompt MLP 的设计在参数效率和性能之间取得了良好平衡

## 局限与展望

- 在 OBJ-BG 和 OBJ-ONLY 上未超过 IDPT，说明某些场景下实例感知的动态提示可能更有效
- 仅验证了分类和分割任务，在 3D 检测、场景理解等更复杂任务上的效果未知
- token 选择的 Top-K 数量和尺度数作为超参数需要调优
- 未探索与其他 PEFT 方法（如 adapter）的组合可能性

## 相关工作与启发

- LoRA [20] 从 NLP 到 2D 视觉再到 3D 的成功迁移，说明低秩适配具有跨领域的通用性
- IDPT [72] 的实例感知提示和 DAPT [77] 的提示+适配器组合是直接对比对象
- 启发：在将 NLP/2D 方法迁移到 3D 时，需要分析 3D 数据的独特性（如无序性、局部几何重要性），针对性地设计互补模块

## 评分

- **新颖性**: 7/10 — LoRA 在点云上的应用不算全新，但多尺度 token 选择的互补设计有新意
- **实验充分度**: 8/10 — 三个数据集 + 两种预训练模型 + few-shot + 分割任务，覆盖全面
- **写作质量**: 8/10 — 动机清晰，图示直观（特别是 LoRA vs PointNet 的对比图），组织良好
- **价值**: 7/10 — 实用价值高（参数极少且推理无额外开销），但技术创新幅度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[CVPR 2025\] Parametric Point Cloud Completion for Polygonal Surface Reconstruction](parametric_point_cloud_completion_for_polygonal_surface_reconstruction.md)
- [\[CVPR 2025\] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)

</div>

<!-- RELATED:END -->
