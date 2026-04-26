---
title: >-
  [论文解读] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution
description: >-
  [ICCV 2025][图像生成][超分辨率] 本文提出 PatchScaler，一种 Patch 级独立扩散超分管线，通过全局修复模块生成置信度图量化各区域重建难度，并将 Patch 分组为简单/中等/困难三组分配不同采样步数，搭配纹理提示检索机制，在 RealSR 上仅 0.23× ResShift 运行时间达到更优质量。
tags:
  - ICCV 2025
  - 图像生成
  - 超分辨率
  - Patch自适应采样
  - 扩散加速
  - 纹理提示
  - DiT
---

# PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution

**会议**: ICCV 2025  
**arXiv**: [2405.17158](https://arxiv.org/abs/2405.17158)  
**代码**: https://github.com/yongliuy/PatchScaler  
**领域**: 图像超分辨率 / 扩散模型  
**关键词**: 超分辨率, Patch自适应采样, 扩散加速, 纹理提示, DiT

## 一句话总结

本文提出 PatchScaler，一种 Patch 级独立扩散超分管线，通过全局修复模块生成置信度图量化各区域重建难度，并将 Patch 分组为简单/中等/困难三组分配不同采样步数，搭配纹理提示检索机制，在 RealSR 上仅 0.23× ResShift 运行时间达到更优质量。

## 研究背景与动机

**扩散模型 SR 的效率问题**：扩散模型大幅提升了超分的感知质量，但大量迭代采样导致推理效率低下，尤其处理高分辨率图像时计算开销巨大。

**统一采样的次优性**：现有加速方法（条件蒸馏、重定义扩散过程）统一减少所有区域的采样步数，忽略了不同区域重建难度的差异——结构简单的区域几步即可重建，而纹理丰富的区域需要更多步骤。

**文本提示的局限**：在 SR 任务中，文本提示与图像内容的对齐度远低于 T2I 任务，局部纹理恢复更需要视觉级条件信息而非文本描述。

**核心观察**：如图 1(a) 所示，简单 patch 仅需 2 步即可高质量重建，而复杂 patch 需要 15 步。

## 方法详解

### 整体架构

PatchScaler 分三个阶段：

1. **全局修复模块(GRM)**：移除退化并生成粗 HR 特征和置信度图
2. **Patch 自适应分组采样(PGS)**：按置信度分组设置不同采样配置
3. **Patch-DiT**：以纹理提示为条件精细化各组 Patch

### 关键设计一：全局修复模块与置信度图

GRM 同时输出粗 HR 特征 $\mathbf{y}_{HR}$ 和置信度图 $C$，训练目标：

$$L(\theta) = \|\mathbf{y}_{HR} - \mathbf{x}_{HR}\|_1^2 + \lambda(C\|\mathbf{y}_{HR} - \mathbf{x}_{HR}\|_2^2 - \eta\log(C))$$

低置信度区域表明 GRM 重建困难（需更多扩散步改进），高置信度则表明已足够好。

### 关键设计二：Patch 自适应分组采样 (PGS)

将粗 HR 特征切为 patch 并按平均置信度分组：

$$Qmap_{\mathbf{y}_{0,i}} = \begin{cases}\text{Simple}, & Avg(C\langle\mathbf{y}_{0,i}\rangle) \in (\gamma_1, 1] \\\text{Medium}, & Avg(C\langle\mathbf{y}_{0,i}\rangle) \in (\gamma_2, \gamma_1] \\\text{Hard}, & Avg(C\langle\mathbf{y}_{0,i}\rangle) \in [0, \gamma_2]\end{cases}$$

**快捷路径推导**：设 $\mathbf{x}_0 = \mathbf{y}_0 + \triangle\mathbf{x}_0$，当 GRM 已去除退化后 $\triangle\mathbf{x}_0$ 较小。找到适当中间时间步 $\tau$ 使得 $\sqrt{\bar{\alpha}_\tau}\triangle\mathbf{x}_0 \to 0$：

$$q(\mathbf{x}_\tau|\mathbf{y}_0) \approx \mathcal{N}(\mathbf{x}_\tau; \sqrt{\bar{\alpha}_\tau}\mathbf{y}_0, (1-\bar{\alpha}_\tau)\mathbf{I})$$

不同组设置不同 $(T_i, N_i)$：
- Simple: $T_1 < T_2 < T_3$, $N_1 < N_2 < N_3$
- 简单 patch 从更近的中间点出发，用更少步数完成

### 关键设计三：纹理提示

构建通用**参考纹理记忆库(RTM)**：
- 收集多样高质量纹理 patch 作为 RTM-value
- 用纹理分类器提取语义特征作为 RTM-key
- 推理时对目标 patch 提取 query，通过内积检索最相似的纹理 patch 作为条件

纹理提示为 Patch-DiT 提供局部纹理先验，替代不够精确的文本提示。

### Patch-DiT 架构

基于 DiT 构建，天然适合处理 token 序列形式的 patch 级特征。相比 U-Net 在低分辨率 patch 上效果更好。

## 实验

### RealSR 4× 定量对比

| 方法 | CLIPIQA↑ | MUSIQ↑ | NIQE↓ | 运行时间(s) |
|------|---------|--------|-------|-----------|
| Real-ESRGAN | 基线 | 基线 | 基线 | 快 |
| StableSR | 高 | 高 | - | 慢 |
| DiffBIR | 高 | 高 | - | 慢 |
| ResShift | 较高 | 较高 | - | 中等 |
| **PatchScaler** | **最优** | **最优** | **最优** | **0.23× ResShift** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 统一采样 vs PGS | PGS 质量相当但速度显著提升 |
| 无纹理提示 | 纹理丰富区域细节下降 |
| 文本提示 vs 纹理提示 | 纹理提示在 SR 任务中更有效 |
| 3组 vs 2组 vs 1组 | 3组最佳平衡 |

### 关键发现

- PatchScaler 在 512→2048 SR 任务上运行时间仅为 ResShift 的 0.23×
- 置信度图准确反映区域难度：复杂纹理→困难组，平坦区域→简单组
- 纹理提示比文本提示在 SR 中更有效——文本提示与局部纹理的对齐度自然较低
- 简单 patch 可跳过大部分扩散步骤而不损失质量，验证了自适应采样的合理性
- 对高分辨率图像加速效果更显著（patch 越多，简单 patch 比例通常越高）

## 亮点与洞察

1. **patch 级自适应采样**首次在 SR 扩散中实现，从根本上解决了统一采样的效率浪费
2. **置信度驱动分组**有理论支撑——$\triangle\mathbf{x}_0$ 小时可用更近的中间点出发
3. **纹理提示**巧妙替代了 SR 场景下不够精确的文本提示
4. **patch-independent 管线**天然支持并行计算和高分辨率扩展

## 局限性

- 需要预训练 GRM 并构建 RTM，额外训练开销
- patch 边界处理（如拼接伪影）需要额外注意
- 纹理检索质量依赖 RTM 的覆盖面

## 相关工作

- **扩散 SR**: StableSR, DiffBIR, ResShift
- **经典 SR**: Real-ESRGAN, BSRGAN, SwinIR
- **扩散加速**: 条件蒸馏, DDIM, DPM-Solver

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — Patch 自适应采样 + 纹理提示双创新
- 技术深度：⭐⭐⭐⭐ — 快捷路径理论推导完整
- 实验充分度：⭐⭐⭐⭐ — 多数据集、速度对比详尽
- 实用价值：⭐⭐⭐⭐⭐ — 0.23× 运行时间、高分辨率友好

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](../../CVPR2025/image_generation/uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](../../CVPR2025/image_generation/faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](../../CVPR2026/image_generation/vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)

<!-- RELATED:END -->
