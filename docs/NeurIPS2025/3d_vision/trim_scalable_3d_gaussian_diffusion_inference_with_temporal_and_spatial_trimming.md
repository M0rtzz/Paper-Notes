---
title: >-
  [论文解读] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming
description: >-
  [NeurIPS 2025][3D视觉][3D生成] 提出TRIM（Trajectory Reduction and Instance Mask denoising），一种后训练框架，通过时间维度的轨迹预筛选和空间维度的背景token裁剪来加速3D高斯扩散模型推理，同时提升生成质量，在T3Bench文本生成3D和GSO图像生成3D任务上均优于DiffSplat等基线。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3D生成
  - 高斯扩散
  - 推理加速
  - 推理时缩放
  - 后训练优化
---

# TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming

**会议**: NeurIPS 2025  
**arXiv**: [2511.16642](https://arxiv.org/abs/2511.16642)  
**代码**: 有 (link in paper)  
**领域**: 3D视觉  
**关键词**: 3D生成, 高斯扩散, 推理加速, 推理时缩放, 后训练优化

## 一句话总结

提出TRIM（Trajectory Reduction and Instance Mask denoising），一种后训练框架，通过时间维度的轨迹预筛选和空间维度的背景token裁剪来加速3D高斯扩散模型推理，同时提升生成质量，在T3Bench文本生成3D和GSO图像生成3D任务上均优于DiffSplat等基线。

## 研究背景与动机

文本/图像到3D生成领域近期取得了显著进展，尤其是DiffSplat等方法通过复用2D图像扩散模型来生成3D高斯表示。然而，将2D扩散的多种后训练加速技术（蒸馏、压缩、推理时缩放等）迁移到3D面临两大核心困难：

**3D高斯表示的非结构化特性**：大量高斯原语散布在3D空间中，难以进行结构化压缩

**沉重的三阶段计算pipeline**：Recon-Gen-Render过程（重建→生成→渲染）比简单的2D去噪计算成本高得多

本文发现了两个关键的效率瓶颈：
- **轨迹层面**：增加采样轨迹数量（不同随机种子）能显著提升生成质量，但在3D扩散中每条轨迹的去噪代价极高
- **Token层面**：大量透明背景区域的高斯原语被不必要地去噪，浪费了计算资源

核心idea：在时间维度上通过轻量级选择器提前筛选高质量轨迹（只保留一条），在空间维度上通过实例mask检测并剔除背景token，双管齐下实现加速+提质。

## 方法详解

### 整体框架

TRIM是一个三阶段推理框架：(1) 第一阶段多条轨迹并行去噪到中间时间步，用Latent Selector选出最优轨迹，其余终止；(2) 第二阶段所选轨迹继续去噪，通过实例mask逐步裁剪背景token减少transformer计算量；(3) 第三阶段利用mask修正去噪后的高斯原语参数（将背景区域Gaussian不透明度置零），消除渲染伪影。

### 关键设计

1. **Trajectory Reduction（轨迹缩减）**: 训练一个轻量级Latent Selector来预测哪条去噪轨迹最终能生成高质量3D资产。采用离线知识蒸馏策略分两步：首先数据合成——用100个文本提示各生成64条轨迹，对最终渲染图像用CLIP评分得到{轨迹, 分数}数据对；然后训练选择器——将其建模为pairwise比较任务，给定两条轨迹的中间latent，预测哪条分数更高。选择器架构为单层CNN+两层MLP，简洁高效（额外延迟可忽略）。推理时采用tournament选拔策略：在去噪50%进度时，N条轨迹两两比较选出一条继续。这使总去噪步数从NT减少为NT-(N-1)t，且后续的VAE解码和渲染也减少N倍。

2. **Instance Mask Denoising（实例mask去噪）**: 无训练的背景检测与裁剪机制。利用latent特征图四角区域通常对应透明背景这一观察，聚合四角特征形成[REF]参考token，计算每个patch与[REF]的相似度，低相似度区域为前景实例。采用**渐进式mask扩展调度器**：将去噪过程分为4个阶段，从仅mask最外围2行/列的patch逐步扩展到4、6、8行/列（全网格），避免早期硬mask引起的伪影。检测到的背景token通过Token Merging合并为单个[BG] token，与前景token序列拼接后送入transformer，去噪后再将[BG] token填回原位。

3. **Post-denoising Correction（后去噪修正）**: 由于原模型训练时没有见过mask和[BG] token，[BG] token的去噪结果不是完全透明的背景。因此在最后一步利用mask将背景区域的高斯原语不透明度置零，消除渲染伪影。

### 损失函数 / 训练策略

Selector用BCE损失训练：L_BCE = -(y·log σ(ŷ) + (1-y)·log(1-σ(ŷ)))，其中y = 1(s₁>s₂)。AdamW优化器(lr=0.001, wd=0.01)，cosine调度，batch size 64，20个epoch。Instance Mask Denoising完全无训练（training-free），直接插入现有3D扩散模型。

## 实验关键数据

### 主实验

**Text-to-3D (T3Bench)**:

| 类别 | 指标 | TRIM (Ours) | DiffSplat | LGM | GVGEN |
|------|------|-------------|-----------|-----|-------|
| Single Object | CLIP Sim.% | **31.58** | 30.95 | 29.96 | 23.66 |
| Single Object | ImageReward | **0.12** | -0.49 | -0.72 | -2.15 |
| Single w/ Sur. | CLIP Sim.% | **31.48** | 30.20 | 27.79 | 22.65 |
| Single w/ Sur. | CLIP R-Prec.% | **88.25** | 80.75 | 55.00 | 26.75 |
| Multiple Objects | CLIP Sim.% | **30.11** | 29.46 | 27.07 | 21.48 |
| Multiple Objects | ImageReward | **-0.24** | -0.84 | -1.73 | -2.27 |

TRIM是唯一获得正ImageReward分数的方法(0.12)，表明生成质量达到人类偏好水平。

**Image-to-3D (GSO dataset)**:

| 指标 | TRIM (Ours) | DiffSplat | InstantMesh | LGM |
|------|-------------|-----------|-------------|-----|
| PSNR ↑ | **16.78** | 16.20 | 15.53 | 14.90 |
| SSIM ↑ | **0.82** | 0.79 | 0.77 | 0.71 |
| LPIPS ↓ | **0.17** | 0.19 | 0.22 | 0.25 |

### 消融实验

| 配置 | FLOPs(T) ↓ | 吞吐量(step/s) ↑ | 运行时间(s) ↓ | 说明 |
|------|-----------|------------------|-------------|------|
| DiffSplat (baseline) | 195.68 | 13.18 | 8.64 | - |
| + Instance Masking | 165.60 | 18.09 | - | FLOPs减少15.7% |
| + Trajectory Reduction | 110.07 | 13.18 | - | 减少总去噪步数 |
| + TRIM (both) | **106.31** | **18.09** | **~5** | 推理时间从8秒降至5秒 |

Selector架构消融表明单层CNN+两层MLP即为最优（pairwise accuracy 74.18%），更复杂架构不带来额外增益。

### 关键发现

- **轨迹多样性优于步数增加**：固定总去噪步数为80，TRIM用10步×8轨迹+选择 vs DiffSplat用80步×1轨迹，TRIM的CLIP和ImageReward均远优，且后者在高步数时反而出现语义漂移和过度平滑
- **Selector在50%进度时效果最佳**：过早应用noise太大无法区分，过晚则效率增益减小
- **轨迹缩减略微降低输出多样性**：将输出分布从低质量-高方差移向高质量-低方差，符合设计预期
- **Instance Masking主要减少运行时间**，Trajectory Reduction主要提升质量，两者互补

## 亮点与洞察

- 首次将推理时缩放(inference-time scaling)引入3D扩散模型领域，发现轨迹多样性比去噪步数更有效
- Latent Selector的离线蒸馏+pairwise比较训练范式简洁有效，仅用100个提示×64轨迹即可训得可靠选择器
- Instance Mask的corner-reference attention检测方法无需训练即可即插即用，巧妙利用了3D生成中四角为背景的先验
- 渐进式mask扩展避免了早期硬裁剪的伪影，是一个实用的工程设计
- 后处理阶段用mask置零背景Gaussian不透明度，简单但有效地解决了无训练插入带来的伪影

## 局限与展望

- 当前3D扩散pipeline严重依赖repurposed 2D backbone，导致空间裁剪只能在去噪transformer block中应用，无法延伸至完整生成pipeline
- Selector需要在合成数据上训练，对于新backbone需要重新合成数据和训练
- Instance Mask依赖"四角为背景"的假设，对物体偏离中心的情况可能失效
- 仅在DiffSplat一个3D扩散模型上验证，虽声称model-agnostic但未在其他模型上展示

## 相关工作与启发

- 2D领域的推理时缩放（SANA-1.5、Inference Scaling for Diffusion等）启发了轨迹多样性策略
- Token Merging (ToMe) 是2D领域的token裁剪方法，但需要retraining，TRIM的方法是training-free
- DiffSplat和Gaussian Atlas是将2D扩散模型复用于3D高斯生成的代表工作
- DINO中[CLS] token对patch attention的可视化启发了corner-reference attention的设计

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](../../ICCV2025/3d_vision/spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)
- [7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](../../ICCV2025/3d_vision/7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)
- [MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)
- [Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](../../ICCV2025/3d_vision/towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)

<!-- RELATED:END -->
