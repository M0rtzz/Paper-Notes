---
title: >-
  [论文解读] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching
description: >-
  [3D视觉] 深入分析单目先验融合中的三大问题（仿射不变性 vs 绝对深度的对齐、迭代更新中的局部最优、噪声视差对融合的干扰），提出二元局部排序图和全局配准模块，在 SceneFlow→Middlebury/Booster 泛化实验中将 bad2 错误减半甚至更多，且几乎不增加计算开销。
tags:
  - 3D视觉
---

# Diving into the Fusion of Monocular Priors for Generalized Stereo Matching

## 论文信息

- **会议**: ICCV 2025
- **arXiv**: 2505.14414
- **代码**: 未公开（基于 RAFT-Stereo 结构）
- **领域**: 3D 视觉 / 立体匹配
- **关键词**: stereo matching, monocular prior, vision foundation model, depth estimation, generalization

## 一句话总结

深入分析单目先验融合中的三大问题（仿射不变性 vs 绝对深度的对齐、迭代更新中的局部最优、噪声视差对融合的干扰），提出二元局部排序图和全局配准模块，在 SceneFlow→Middlebury/Booster 泛化实验中将 bad2 错误减半甚至更多，且几乎不增加计算开销。

## 研究背景与动机

立体匹配在遮挡、无纹理和非朗伯表面（反射/透明）等病态区域泛化能力差。融合单目先验可以纠正这些区域的匹配错误，但存在核心矛盾：

**有偏先验**：在小规模立体数据集上学到的单目先验受领域偏差影响，无法泛化

**无偏先验的可用性**：DepthAnything V2 等视觉基础模型可提供无偏单目先验，但如何高效融合是关键

作者深入分析融合过程，发现三个限制因素：

- **对齐问题**：单目深度是仿射不变的相对深度，与立体匹配的绝对视差存在天然鸿沟
- **局部最优**：在迭代更新结构中，视差更新的过度自信导致被困在局部最优解
- **噪声干扰**：前几次迭代的噪声视差误导直接深度融合

## 方法详解

### 整体框架

网络包含三个模块：
1. **单目编码器**：冻结的 DepthAnything V2 提取无偏单目先验（深度图 + 上下文特征）
2. **迭代局部融合**：使用二元局部排序图引导每次迭代的视差更新
3. **全局融合**：将单目深度配准到优化后的视差图

### 关键设计 1：二元局部排序图

将深度值转换为局部的远近关系，统一单目深度和双目视差的表示：

$$M_O(u,v) = \{\sigma(D(u', v') - D(u, v))\}$$

其中 $(u', v') \in \mathcal{N}_{(u,v)}$，$\sigma$ 为 sigmoid 函数。

核心优势：
- **鲁棒性**：将绝对值转为排序关系，对异常值噪声不敏感
- **统一性**：仿射不变的单目深度和绝对视差在排序关系上天然兼容
- **可解释性**：可视化显示随迭代推进，视差的排序图逐渐趋近单目深度的排序图

使用 LBP-like（局部二值模式）卷积块计算排序图，采用固定权重而非可学习权重（因为有限数据使可学习权重不可靠）。将单目和视差的排序图拼接，预测服从 Beta 分布的引导图 $G$。

视差更新通过引导图重加权，并随迭代逐步释放影响：

$$\tilde{\Delta}_d = \Delta_d (1 + G \cdot r \cdot t / T)$$

$$D_d^t = D_d^{t-1} + \tilde{\Delta}_d$$

### 关键设计 2：全局融合模块

将优化后的视差图建模为单目深度的带噪配准版本，学习像素级线性回归参数：

$$\tilde{D}_m = a \cdot D_m + b$$

$$a, b = \mathcal{F}(D_m, D_d^T)$$

其中 $\mathcal{F}$ 为卷积网络。通过置信度图融合配准后的单目深度和优化的视差：

$$D_f = c \cdot D_d^T + (1 - c) \cdot \tilde{D}_m$$

置信度 $c$ 由代价体积、GRU 隐状态和最后一次迭代的引导图共同预测。

### 损失函数

$$\mathcal{L} = \sum_{t=1}^T \gamma^{T+2-t} \|D_d^t - D_G\|_1 + \gamma \|\tilde{D}_m - D_G\|_1 + \|D_f - D_G\|_1$$

三阶段训练：先训不含全局融合的主体 → 冻结主体训配准参数 → 冻结主体训完整全局融合。

## 实验关键数据

### 主实验：SceneFlow → 真实世界泛化

| 方法 | 额外数据 | Middlebury EPE | Middlebury bad2 | ETH3D EPE | ETH3D bad2 |
|------|---------|----------------|-----------------|-----------|------------|
| RAFTStereo | ✗ | 1.92 | 12.60 | 0.36 | 3.30 |
| IGEV | ✗ | 2.63 | 11.93 | 0.33 | 4.00 |
| Mocha-Stereo | ✗ | 2.66 | 10.18 | 0.28 | 3.47 |
| Selective-IGEV | ✗ | 2.59 | 11.79 | 0.33 | 4.05 |
| NerfStereo | ✓ | 1.42 | 9.67 | 0.29 | 2.94 |
| **Ours** | **✗** | **1.15** | **8.39** | **0.25** | **1.88** |

- Middlebury 上 bad2 从 RAFT-Stereo 的 12.60 降至 8.39（约 33% 改进）
- ETH3D 上 bad2 从 3.30 降至 1.88（约 43% 改进）
- 不使用额外数据，性能超越使用 NeRF 额外数据的 NerfStereo

### Booster 数据集（透明/反射区域）

| 方法 | All EPE | All bad2 | Trans bad2 | Trans bad5 | NonTrans bad2 |
|------|---------|----------|-----------|------------|---------------|
| RAFTStereo | 4.18 | 17.64 | 67.69 | 47.40 | 13.13 |
| Mocha-Stereo | 3.88 | 16.82 | 66.44 | 45.73 | 12.31 |
| RAFT+ME | 2.40 | 11.44 | 64.84 | 43.95 | 6.96 |
| **Ours** | **2.26** | **11.02** | **59.83** | **38.44** | **6.98** |

透明区域 bad5 从 ~47% 降至 38.4%，约 10 个百分点改进。

### 消融实验

| 配置 | Middlebury EPE | Middlebury bad2 |
|------|----------------|-----------------|
| Baseline (RAFT-Stereo) | 2.11±0.16 | 14.12±0.64 |
| Baseline w/o mono feature | 1.83±0.11 | 12.45±0.86 |
| + Monocular Encoder (ME) | 1.42±0.01 | 9.81±0.18 |
| + ME + Iterative Direct Fusion | 1.41±0.04 | 10.34±0.19 |
| + ME + Iterative Local Fusion (ILF) | 1.20±0.08 | 9.06±0.70 |
| + ME + ILF + Global Fusion (GF) | **1.15±0.01** | **8.35±0.04** |

**关键发现**：
- 移除 RAFT-Stereo 原有的有偏单目特征反而提升性能（1.83 vs 2.11），证明有偏先验有害
- 单目编码器（ME）贡献最大的改进（EPE: 2.11→1.42）
- 直接融合（IDF）反而比不融合更差（bad2: 10.34 vs 9.81），验证了噪声干扰问题
- ILF（排序图引导）+ GF（全局配准）联合最优
- LBP-like 固定权重优于可学习卷积权重（有限数据→可学习权重不可靠）
- 时间开销：Baseline 0.32s → Ours 0.40s，仅增加 25%

## 亮点与洞察

1. **问题分析深入**：不只是提出方法，而是系统剖析了单目先验融合的三大瓶颈，每个组件都有明确的问题对应
2. **局部排序图的巧妙设计**：用排序关系统一仿射不变深度和绝对视差，既解决对齐问题又抗噪声
3. **固定权重 > 可学习权重**：在数据有限的泛化场景中，先验知识（手工设计的 LBP 卷积）比学习更可靠
4. **效率设计**：虽然引入了 VFM（DepthAnything V2），但冻结 + 适当分辨率处理使时间开销仅增加 0.08s
5. **全面的度量分析**：指出许多论文使用隐含假设（限制视差范围、只评估非遮挡区域），提供了更统一公平的对比

## 局限性

- 三阶段训练流程较复杂，需要分步冻结不同模块
- 依赖 DepthAnything V2 的质量，如果单目模型在特定场景失效则融合无效
- 全局融合的线性配准假设（$a \cdot D_m + b$）可能对非线性深度分布建模不足
- KITTI 上的改进相对有限（已较成熟的数据集），更大改进集中在 Middlebury/Booster

## 相关工作与启发

- 与 StereoAnywhere 的比较：同期工作，都融合 VFM 先验到立体匹配，但融合机制不同
- 配准思想（scale+shift 对齐）与 MiDaS/ZoeDepth 中的做法类似，但在迭代立体匹配中的应用是新的
- 二元局部排序图与 LBP 特征的联系：借用经典纹理描述子的思想处理深度排序问题
- 渐进释放引导（$t/T$ 衰减）的设计思路可推广到其他噪声引导的迭代优化场景

## 评分

⭐⭐⭐⭐⭐ — 分析功底极强的工作，三个问题-三个解决方案的对应关系清晰。实验严谨（均值+标准差报告），在最困难的透明/反射区域取得显著突破，且几乎不增计算开销。对立体匹配的泛化研究有重要参考价值。

<!-- RELATED:START -->

## 相关论文

- [RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

<!-- RELATED:END -->
