---
title: >-
  [论文解读] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance
description: >-
  [ICCV 2025][图像生成][CFG蒸馏] 本文提出 TeEFusion，通过将 CFG 的引导幅度直接编码为条件/无条件文本嵌入的线性组合来替代双重前向传播，实现零额外参数的高效 CFG 蒸馏，同时兼容教师模型的复杂采样策略（如 Z-Sampling、W2SD），使学生模型推理速度达教师的 6 倍。
tags:
  - ICCV 2025
  - 图像生成
  - CFG蒸馏
  - 文本嵌入融合
  - 采样加速
  - Guidance Distillation
  - DiT
---

# TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance

**会议**: ICCV 2025  
**arXiv**: [2507.18192](https://arxiv.org/abs/2507.18192)  
**代码**: https://github.com/AIDC-AI/TeEFusion  
**领域**: 扩散模型 / 图像生成 / 蒸馏  
**关键词**: CFG蒸馏, 文本嵌入融合, 采样加速, Guidance Distillation, DiT

## 一句话总结

本文提出 TeEFusion，通过将 CFG 的引导幅度直接编码为条件/无条件文本嵌入的线性组合来替代双重前向传播，实现零额外参数的高效 CFG 蒸馏，同时兼容教师模型的复杂采样策略（如 Z-Sampling、W2SD），使学生模型推理速度达教师的 6 倍。

## 研究背景与动机

**CFG 的代价**：Classifier-Free Guidance 是当前 T2I 模型（SD3、FLUX）保证生成质量的关键技术，但需要两次前向传播（条件+无条件），推理开销翻倍。更复杂的采样策略（如 Z-Sampling+CFG 需 6 倍开销）进一步加剧了这一问题。

**现有蒸馏方法的不足**：
   - DistillCFG 等方法需要额外的 MLP 来编码引导尺度，增加了架构复杂度
   - 现有方法仅限于师生使用相同采样算法的场景，无法蒸馏复杂采样策略
   - 缺乏在大规模 SOTA 模型（如 SD3）上的充分验证

**关键发现**：在文本嵌入空间中对不同 prompt 的嵌入进行加减操作可以有效地融合或消除特定语义成分，这为将 CFG 的线性组合前移到嵌入空间提供了理论基础。

## 方法详解

### 核心思想

CFG 的标准公式为：

$$\tilde{\epsilon}_\theta(x_t, c) = (1+w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t)$$

需要两次前向传播。TeEFusion 将这一线性组合从模型输出空间前移到文本嵌入空间，仅需一次前向传播：

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \hat{c})$$

其中 $\hat{c} = (1+w)c - w\varnothing = c + w(c - \varnothing)$。

### 嵌入融合公式

在 DiT 模型中，时间步和文本的联合嵌入为 $z_{t,c} = \mathcal{G}(\psi(t)) + \mathcal{F}(c)$。TeEFusion 将引导尺度 $w$ 编码为：

$$\hat{z}_{t,c,\varnothing,w} = \mathcal{G}(\psi(t)) + \mathcal{F}(c) + \underbrace{\mathcal{G}(\psi(w))\mathcal{F}(c - \varnothing)}_{\text{extra term}}$$

关键设计要点：
- 使用正弦余弦时间嵌入 $\psi(w)$ 将标量 $w$ 投影到向量空间，避免 $\mathcal{O}(w^2)$ 方差问题
- 额外项中 $\mathcal{G}(\psi(w))$ 作为逐元素缩放因子作用于 $\mathcal{F}(c-\varnothing)$，精确对应 $w \cdot (c-\varnothing)$ 的语义
- **不引入任何新参数**：完全复用已有的 $\mathcal{G}$ 和 $\mathcal{F}$ 网络

### 蒸馏流程

$$L_{\text{distill}} = \|\epsilon_{\theta_S}(x_t, \hat{z}_{t,c,\varnothing,w}) - \tilde{\epsilon}_{\theta_T}(x_t, w, c)\|_2^2$$

- 学生模型以融合嵌入 $\hat{z}$ 为输入，单次前向传播
- 教师模型使用标准 CFG（甚至加上反射采样如 W2SD）作为训练目标
- 引导尺度 $w$ 每次迭代均匀采样自 $[w_{\min}, w_{\max}]$
- 支持教师使用任意复杂采样策略（Z-Sampling、W2SD）

## 实验

### HPS 美学评分对比

| 模型 | 方法 | 开销 | Anime | Concept-Art | Paintings | Photo |
|------|------|------|-------|-------------|-----------|-------|
| SD3 | CFG | 2× | 30.78 | 30.06 | 30.28 | 27.93 |
| SD3 | W2SD+CFG | 6× | 31.96 | 30.65 | 30.67 | 29.76 |
| SD3 | DistillCFG | 1× | 31.14 | 29.52 | 30.03 | 29.04 |
| SD3 | **TeEFusion** | **1×** | **32.37** | **30.88** | **30.74** | **29.84** |

### 消融实验与训练效率

| 组件 | 效果 |
|------|------|
| 移除 $\mathcal{G}(\psi(w))$ 缩放 | HPS 显著下降，引导尺度不可控 |
| 直接线性组合（不经 sin/cos 编码） | 大 $w$ 值时数值不稳定 |
| 训练收敛速度 | ~10K 步即可收敛，训练高效 |

### 关键发现

- TeEFusion 在所有美学维度上超越教师模型的 CFG 结果，同时只需 1/6 推理开销
- 在 DPG-Bench 目标组合和 prompt 跟随评估中同样领先
- 方法简洁——学生模型结构与原模型完全一致，便于部署
- 从 W2SD+CFG 教师蒸馏的学生甚至超越教师质量，验证了 test-time scaling 蒸馏的可行性

## 亮点与洞察

1. **极简设计**：零额外参数，仅计算一个额外项即可免去双重前向传播
2. **采样策略蒸馏**：首次将复杂反射采样策略蒸馏到简单 Euler 采样中
3. **实验验证扎实**：在工业级大模型（SD3、In-house T2I）上充分验证

## 局限性

- 文本嵌入线性融合假设在极端引导尺度下可能不成立
- 目前验证局限于 DiT 架构，对 U-Net 架构的适用性未探讨
- 蒸馏仍需教师模型生成训练目标，离线训练成本不可忽略

## 相关工作

- **引导蒸馏**: DistillCFG, DICE, Progressive Distillation
- **测试时缩放**: Z-Sampling, W2SD, RePaint
- **高级采样器**: DPM-Solver, DDIM

## 评分

- 新颖性：⭐⭐⭐⭐ — 嵌入空间融合思路简洁优雅
- 技术深度：⭐⭐⭐⭐ — 对 CFG 机制有深入分析
- 实验充分度：⭐⭐⭐⭐⭐ — 工业级模型验证
- 实用价值：⭐⭐⭐⭐⭐ — 6× 加速、零参数增加、易部署

<!-- RELATED:START -->

## 相关论文

- [DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [TCFG: Tangential Damping Classifier-Free Guidance](../../CVPR2025/image_generation/tcfg_tangential_damping_classifier-free_guidance.md)
- [Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [Classifier-Free Guidance inside the Attraction Basin May Cause Memorization](../../CVPR2025/image_generation/classifier-free_guidance_inside_the_attraction_basin_may_cause_memorization.md)

<!-- RELATED:END -->
