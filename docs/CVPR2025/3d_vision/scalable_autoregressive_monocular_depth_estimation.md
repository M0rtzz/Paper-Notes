---
title: >-
  [论文解读] Scalable Autoregressive Monocular Depth Estimation
description: >-
  [CVPR 2025][3D视觉][单目深度估计] 提出深度自回归模型 DAR，通过分辨率自回归（从低到高分辨率逐步生成深度图）和粒度自回归（从粗到细递归细化深度区间）两个有序目标，将单目深度估计任务转化为自回归预测范式，模型可扩展至 2.0B 参数并在 KITTI 和 NYU Depth v2 上达到 SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 单目深度估计
  - 自回归模型
  - 多分辨率预测
  - 深度离散化
  - 模型可扩展性
---

# Scalable Autoregressive Monocular Depth Estimation

**会议**: CVPR 2025  
**arXiv**: [2411.11361](https://arxiv.org/abs/2411.11361)  
**代码**: 无  
**领域**: 3D Vision / Depth Estimation  
**关键词**: 单目深度估计, 自回归模型, 多分辨率预测, 深度离散化, 模型可扩展性

## 一句话总结

提出深度自回归模型 DAR，通过分辨率自回归（从低到高分辨率逐步生成深度图）和粒度自回归（从粗到细递归细化深度区间）两个有序目标，将单目深度估计任务转化为自回归预测范式，模型可扩展至 2.0B 参数并在 KITTI 和 NYU Depth v2 上达到 SOTA。

## 研究背景与动机

单目深度估计 (MDE) 是从单张 RGB 图像预测逐像素深度的任务，在自动驾驶、机器人和增强现实等领域有广泛应用。传统深度学习方法主要采用 encoder-decoder 架构，通过提取和融合低层与高层特征实现深度估计。

自回归 (AR) 模型在 NLP 和多模态生成任务中展现了出色的泛化能力和可扩展性，如 GPT-4 和 LLaVA。这自然引出一个问题：**能否为单目深度估计任务开发自回归模型？**

然而，自回归建模依赖于组织良好的序列数据格式，每一步的预测需与前一步逻辑关联。这种序列依赖关系在 MDE 中并不直观——深度图不具备天然的顺序预测目标。现有方法如 DORN 和 Ord2Seq 将 MDE 视为序数回归任务，通过离散化深度空间进行预测，但未充分挖掘深度估计中的双重有序性质。

本文的核心洞察是：MDE 任务存在两个天然的有序属性——**深度图分辨率**（从低到高）和**深度值粒度**（从粗到细），这两个属性可以被转化为自回归目标。

## 方法详解

### 整体框架

DAR 由四个核心组件构成：(1) Image Encoder：使用 ViT 提取 RGB 图像特征，聚合不同层的特征图到 $1/8$ 分辨率，得到 $1536 \times H/8 \times W/8$ 的 token map；(2) DAR Transformer：通过 patch-wise causal mask 逐步预测不同分辨率的 token map；(3) Multiway Tree Bins (MTBin)：将每个像素的深度范围递归细分为不同粒度的 bins；(4) Bins Injection：利用 bin 候选信息引导深度特征的建模。

整个模型将深度图预测建模为条件概率的连乘：$p(\tilde{D}_1, \tilde{D}_2, \ldots, \tilde{D}_K) = \prod_{k=1}^{K} p_\theta(\tilde{D}_k \mid \tilde{D}_1, \ldots, \tilde{D}_{k-1})$，最终输出最高分辨率的深度图 $\hat{D} = \tilde{D}_K$。

### 关键设计

**1. DAR Transformer 与 Patch-wise Causal Mask（分辨率自回归）**

- **功能**: 实现从低到高分辨率的逐步深度图生成
- **核心思路**: 在每一步 $k$，将上一步的 token map $r_{k-1}$ 上采样至下一分辨率作为输入 $y_{in}^k$，通过 Multi-headed Self-Attention (MSA) 和 Multi-headed Cross-Attention (MCA) 层生成输出 logits $y_{out}^k$。MSA 使用 patch-wise causal mask 确保当前 token map 只能与自身及前缀 token 交互，MCA 引入 RGB 图像特征作为条件控制
- **设计动机**: 与传统 encoder-decoder 中的特征融合不同，此设计将低层与高层特征融合过程重新表述为从低到高分辨率的自回归目标，使模型能利用之前所有步骤的深度预测来生成更高分辨率的深度图

**2. Multiway Tree Bins (MTBin)（粒度自回归）**

- **功能**: 递归细化深度区间，实现从粗到细的深度值预测
- **核心思路**: 假设步骤 $k-1$ 预测像素 $\mathbf{x}$ 的深度落在第 $t$ 个 bin 内，MTBin 将该 bin 扩展到相邻 bins（$[b_{k-1}^{t-1}, b_{k-1}^{t+2}]$）以容错，然后将扩展范围均匀分为 $N=16$ 个子 bin。最终深度由 bin 中心与 softmax 概率的线性组合给出：$\tilde{D}_k(\mathbf{x}) = \sum_{i=1}^{N} c_k^i \cdot p_k^i(\mathbf{x})$
- **设计动机**: 传统固定 bin 策略无法根据预测结果动态调整搜索范围。MTBin 像多叉树一样递归搜索更精细的深度值，每个像素的决策过程独立，逐步从粗到细。扩展到相邻 bin 的设计提供了误差容忍能力，避免了预测误差的级联放大

**3. Bins Injection（连接两个自回归目标）**

- **功能**: 将深度候选信息注入到 latent token map 中，连接分辨率与粒度两个自回归过程
- **核心思路**: 将深度候选值 $c^k$ 通过 $3 \times 3$ 卷积投影到特征空间得到 $f_{bin}^k$，再通过 ConvGRU 模块将 bin 特征与 DAR Transformer 的输出融合：$r_k = \text{ConvGRU}(y_{out}^k; f_{bin}^k)$
- **设计动机**: 仅靠分辨率方向的自回归无法感知深度值的粒度信息。Bins Injection 将粒度信息嵌入到 latent token 中，使得模型在生成更高分辨率深度图时能同时利用更精细的深度区间引导

### 损失函数 / 训练策略

- **损失函数**: 使用缩放的 Scale-Invariant Loss，对所有 $K$ 步预测的深度图统一上采样至 ground truth 尺寸后计算。$\mathcal{L} = \sum_{k=1}^{K} \alpha \sqrt{\frac{1}{|T|} \sum (g_k(x))^2 - \frac{\beta}{|T|^2} (\sum g_k(x))^2}$，其中 $g_k(x) = \log \tilde{D}_k(x) - \log D_{gt}(x)$，$\alpha=10$，$\beta=0.85$
- **训练策略**: 使用 AdamW 优化器，学习率从 $3 \times 10^{-5}$ 线性增加到 $5 \times 10^{-4}$ 后线性衰减，batch size 16，训练 25 个 epoch。DAR-Base 使用 8 块 A100 训练，每 epoch 约 30 分钟
- **模型配置**: 三个规模——DAR-Small (440M, 5层), DAR-Base (1B, 7层), DAR-Large (2B, 13层)，步数 $K=5$，每步 bin 数 $N=16$

## 实验关键数据

### 主实验

NYU Depth v2 室内数据集结果：

| 方法 | 模型大小 | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|---------|-----------|--------|-------------|
| Depth Anything | 343M | 0.063 | 0.235 | 0.975 |
| EcoDepth | 954M | 0.059 | 0.218 | 0.978 |
| DAR-Small | 440M | 0.059 | 0.217 | 0.979 |
| DAR-Base | 1.0B | 0.058 | 0.214 | 0.980 |
| **DAR-Large** | **2.0B** | **0.056** | **0.205** | **0.982** |

KITTI 室外数据集结果：

| 方法 | 模型大小 | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|---------|-----------|--------|-------------|
| Depth Anything | 343M | 0.046 | 1.896 | 0.982 |
| EcoDepth | 954M | 0.048 | 2.039 | 0.979 |
| DAR-Small | 440M | 0.046 | 1.839 | 0.984 |
| DAR-Base | 1.0B | 0.046 | 1.823 | 0.985 |
| **DAR-Large** | **2.0B** | **0.044** | **1.799** | **0.986** |

### 消融实验

在 NYU Depth v2 上的消融结果：

| 方法 | 参数量 | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|--------|-----------|--------|-------------|
| Baseline + Transformer | 420M | 0.063 | 0.229 | 0.976 |
| Baseline + MTBins + BI | 363M | 0.061 | 0.220 | 0.978 |
| Baseline + DAR | 440M | 0.059 | 0.217 | 0.979 |
| Baseline + DAR + Scale Up | 2.0B | 0.056 | 0.205 | 0.982 |

### 关键发现

1. **分辨率自回归和粒度自回归各自独立贡献性能提升**：仅添加 Transformer（分辨率目标）RMSE 从 baseline 降至 0.229，仅添加 MTBins+BI（粒度目标）降至 0.220，两者结合降至 0.217
2. **强可扩展性**：模型从 440M 扩展到 2.0B 时，RMSE 持续下降（0.217→0.205），展现出类似 LLM 的 scaling law
3. **零样本泛化能力**：仅在 NYU Depth v2 训练的 DAR 在 SUN RGB-D 上 RMSE 为 0.319，优于 Depth Anything (0.346)，后者使用了 61M 数据预训练
4. **KITTI 上 DAR-Large 的 RMSE 为 1.799，相比 Depth Anything 的 1.896 提升约 5%**

## 亮点与洞察

1. **核心创新在于发现 MDE 中的双重有序性**：将分辨率和深度粒度两个维度的有序性转化为自回归目标，这一洞察简洁而深刻
2. **MTBin 的误差容忍设计**非常实用：扩展到相邻 bin 避免了递归细化过程中的误差级联，这是整个粒度自回归能成功的关键
3. **为将深度估计能力整合到 GPT-4 等大模型中提供了可能路径**：DAR 的自回归范式与现有 LLM 架构天然兼容

## 局限与展望

1. 多步渐进式预测产生更平滑连续的深度图，但可能**模糊边界、降低锐度**
2. 自回归 Transformer 导致**参数量较高**（2.0B），计算成本大
3. 未来可通过**大模型蒸馏**或**轻量化 AR 基础模型**设计降低复杂度
4. 可探索与更强 encoder（如 DINOv2）结合，或在更大规模混合数据上预训练

## 相关工作与启发

- **Depth Anything**: 数据驱动的 SOTA 方法，用 62M 无标签数据自监督预训练。DAR 在监督设置下即超越其性能
- **VAR (Visual AutoRegressive)**: 提出 next-scale prediction 的图像生成范式，启发了 DAR 的分辨率自回归设计
- **Ord2Seq**: 将序数回归视为标签序列任务的自回归网络，启发了 DAR 的粒度自回归目标
- **DORN**: 首个将 MDE 转化为序数回归的工作，DAR 的 MTBin 策略是对固定 bin 策略的重要演进

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 MDE 重新建模为双重自回归任务的思路新颖，MTBin 设计实用
- **实验充分度**: ⭐⭐⭐⭐ — 在两个主流数据集上达到明确 SOTA，有零样本实验和消融分析，但缺少推理速度对比
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图示直观，数学表达规范
- **价值**: ⭐⭐⭐⭐ — 为 MDE 引入了自回归范式，展示了可扩展性，对将深度感知整合到大模型中有启发意义

<!-- RELATED:START -->

## 相关论文

- [Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](relative_pose_estimation_through_affine_corrections_of_monocular_depth_priors.md)
- [Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)

<!-- RELATED:END -->
