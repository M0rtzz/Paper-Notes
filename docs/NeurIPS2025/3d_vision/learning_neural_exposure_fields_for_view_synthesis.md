---
title: >-
  [论文解读] Learning Neural Exposure Fields for View Synthesis
description: >-
  [NeurIPS 2025][3D视觉][神经辐射场] 提出神经曝光场（NExF），通过学习每个 3D 点的最优曝光值（而非每张图像的曝光），实现 3D 一致的高质量视图合成，在高动态范围场景中相比 SOTA 方法 PSNR 提升 3.5+，速度快 50 倍。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 神经辐射场
  - 视图合成
  - 曝光补偿
  - 3D 一致性
  - 高动态范围
---

# Learning Neural Exposure Fields for View Synthesis

**会议**: NeurIPS 2025  
**arXiv**: [2510.08279](https://arxiv.org/abs/2510.08279)  
**代码**: https://m-niemeyer.github.io/nexf  
**领域**: 3D 视觉 / 视图合成  
**关键词**: 神经辐射场, 视图合成, 曝光补偿, 3D 一致性, 高动态范围

## 一句话总结

提出神经曝光场（NExF），通过学习每个 3D 点的最优曝光值（而非每张图像的曝光），实现 3D 一致的高质量视图合成，在高动态范围场景中相比 SOTA 方法 PSNR 提升 3.5+，速度快 50 倍。

## 研究背景与动机

**领域现状**：标准 NeRF 基准排除了曝光变异，但真实场景（如室内外混合、有窗房间）常含强曝光变化，导致重建质量严重下降。

**现有痛点**：HDRNeRF 需要专业 HDR 软件后处理，且仅在 2D 图像层做色调映射，不同视图的同一 3D 点颜色不一致。GLO 嵌入对小曝光变化鲁棒但大变化时失效。

**核心矛盾**：传统相机为每张图像选一个曝光（2D 操作），而理想的方式是为每个 3D 点学习最优曝光（3D 操作）。

**核心 idea**：将曝光从 2D 图像级别提升到 3D 点级别，保证同一 3D 点在所有视图中颜色一致。

## 方法详解

### 整体框架

在标准 NeRF 架构上增加两个组件：(1) 潜在曝光条件化——在 NeRF 瓶颈层（而非输入层）注入 log 曝光；(2) 神经曝光场——一个额外的 MLP 学习 3D 空间中的最优曝光值。

### 关键设计

1. **潜在曝光条件化（Section 3.1）**

    - 功能：在 NeRF 的瓶颈层而非输入层做 log 曝光条件化
    - 核心思路：$f_\theta(\mathbf{x}, \mathbf{d}, \Delta t) = f_\theta^{view}(f_\theta^{pos}(\mathbf{x}) + \ln \Delta t(\mathbf{r}), \mathbf{d})$，位置编码 $f_\theta^{pos}$ 已预测 log 辐射度，直接在中间层加入曝光更稳定
    - 设计动机：中间层条件化相比直接输入条件化性能提升 5%+，因为位置编码已经包含辐射度信息

2. **神经曝光场（Section 3.2）**

    - 功能：学习 3D 曝光值的神经场 $e_\phi: \mathbb{R}^3 \to \mathbb{R}$
    - 核心思路：全连接 MLP（4 层，维度 128），仅在颜色"好曝光且饱和"时更新曝光——好曝光权重 $w_{exp}(\mathbf{c}) = \prod_i \exp(-(c_i - 1/2)^2/\sigma_{exp})$；饱和度权重 $w_{sat}(\mathbf{c}) = \sqrt{\frac{1}{3}\sum_i (c_i - \bar{\mu}_c)^2}$
    - 设计动机：3D 一致性由设计保证——同一 3D 点的曝光值与视角无关；3D 光滑约束 $\|\Delta t_{diff}\|_2^2$ 保证相邻点曝光接近

3. **联合优化（Section 3.3）**

    - 功能：端到端联合训练 NeRF 参数 $\theta$ 和曝光场参数 $\phi$
    - 核心思路：按像素权重有选择地反向传播——仅当颜色好曝光且饱和时更新曝光，忽略欠曝过曝像素
    - 总损失：$\mathcal{L}(\theta, \phi) = \mathcal{L}_f(\theta) + \mathcal{L}_e(\phi)$

### 损失函数 / 训练策略

- NeRF 重建损失：标准 MSE
- 曝光场损失：加权 L2 + 3D 光滑正则化
- 权重函数：好曝光 × 饱和度，避免在过曝/欠曝区域学习错误的曝光值

## 实验关键数据

### 主实验（HDRNeRF 数据集）

| 方法 | 推理时间 | ID-PSNR↑ | OOD-PSNR↑ | ID-LPIPS↓ |
|------|--------|-------------|-------------|-------------|
| NeRF | 405min | 13.97 | 14.51 | 0.376 |
| ZipNeRF | 11min | 19.00 | 19.73 | 0.142 |
| NeRF-W | 437min | 29.83 | 29.22 | 0.047 |
| HDRNeRF | 542min | 39.07 | 37.53 | 0.026 |
| HDR-GS* | 34min | 41.10 | 36.33 | 0.011 |
| **NExF** | **11min** | **42.54** | **38.36** | **0.014** |

### 消融实验

| 配置 | PSNR(ID) | SSIM(ID) | LPIPS(ID) | 说明 |
|------|---------|---------|----------|------|
| w/o 视角 MLP | 33.85 | 0.928 | 0.104 | 基础 NeRF |
| w/o 潜在条件化 | 39.88 | 0.979 | 0.038 | 直接条件化 |
| **完整 NExF** | **42.54** | **0.988** | **0.014** | **最优** |

### 关键发现

- 相比 HDRNeRF：PSNR +3.5（ID）/ +2.2（OOD），速度快 50 倍（11 vs 542 分钟）
- 潜在层条件化贡献 +2.66 PSNR，视角感知 MLP 贡献 +8.69 PSNR
- 3D 一致性问题在定性实验中完全解决——不同视角下同一区域颜色一致
- 在 Eyeful Tower 真实数据集上，极端曝光（太暗/太亮）的泛化显著优于 baseline

## 亮点与洞察

- **理论优美**：2D 图像曝光 → 3D 点曝光的提升是自然且优雅的。这个思路可以迁移到白平衡、色彩校正等其他图像处理任务。
- **3D 一致性**：设计级解决，无需后处理对齐，从根本上避免了色调映射不一致问题。
- **速度与质量兼得**：50 倍加速的同时质量还有提升，得益于轻量级的曝光场 MLP。
- **实现简洁**：仅额外一个 4 层 MLP，任何 NeRF 变体都可以轻松集成。

## 局限与展望

- 假设大多数 3D 点在某个视图中曝光良好，极端均欠曝场景可能失效
- 仅处理静态 3D 场景，动态光照变化未涉及
- 未尝试与 3DGS 结合，GS 版本可能有不同考量
- 假设输入包含准确曝光值，实际相机 ISO/快门值可能有误差

## 相关工作与启发

- **vs HDRNeRF**：HDRNeRF 在 2D 层做色调映射，NExF 在 3D 层做曝光学习，从根本上避免了不一致
- **vs NeRF-W**：NeRF-W 用 GLO 嵌入处理外观变化，但对大曝光变化无效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 3D 曝光场的提出是优雅创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集，充分消融，定性定量完整
- 写作质量: ⭐⭐⭐⭐⭐ 清晰流畅，形式化完善
- 价值: ⭐⭐⭐⭐⭐ 处理真实场景混合照明，部署价值极高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[ICML 2025\] High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](../../ICCV2025/3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)

<!-- RELATED:END -->
