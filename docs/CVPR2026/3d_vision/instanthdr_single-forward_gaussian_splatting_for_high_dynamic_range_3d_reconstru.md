---
title: >-
  [论文解读] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction
description: >-
  [CVPR 2026][3D视觉][HDR新视角合成] 提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模实现多曝光融合，配合元网络学习场景自适应色调映射器，在单次前向传播中从未校准的多曝光 LDR 图像重建 HDR 3D 场景，比优化方法快 ~700×（前馈）/ ~20×（后优化）。
tags:
  - CVPR 2026
  - 3D视觉
  - HDR新视角合成
  - 3D高斯泼溅
  - 前馈重建
  - 色调映射
  - 多曝光融合
---

# InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2603.11298](https://arxiv.org/abs/2603.11298)  
**代码**: 待公开（代码、模型、数据集将在review后发布）  
**领域**: 3D视觉  
**关键词**: HDR新视角合成, 3D高斯泼溅, 前馈重建, 色调映射, 多曝光融合

## 一句话总结

提出 InstantHDR，首个前馈式 HDR 新视角合成方法，通过几何引导的外观建模实现多曝光融合，配合元网络学习场景自适应色调映射器，在单次前向传播中从未校准的多曝光 LDR 图像重建 HDR 3D 场景，比优化方法快 ~700×（前馈）/ ~20×（后优化）。

## 研究背景与动机

HDR 新视角合成旨在从不同曝光的多视角 LDR 图像重建 HDR 场景。现有方法存在关键瓶颈：

- **优化式方法**（HDR-GS, GaussianHDR）：依赖精确相机位姿、密集初始点云，每场景优化耗时 15-30 分钟，且在稀疏视角下因 SfM 点云崩塌而失败
- **前馈式方法**（AnySplat, VGGT）：速度快但忽略 HDR 问题，假设外观曝光不变，直接融合多曝光输入会产生严重鬼影伪影
- **数据瓶颈**: 公开 HDR 场景数据集极度稀缺（最多十几个场景），远不够支撑前馈模型的大规模预训练

将前馈范式引入 HDR 重建面临四大挑战：曝光导致的外观不一致、像素级几何对齐困难、不同相机响应函数差异、HDR 数据稀缺。

## 方法详解

### 整体框架

双分支架构：(1) **几何分支**（冻结）——用预训练交替注意力 Transformer 估计深度图和相机位姿；(2) **外观分支**（可训练）——归一化曝光、融合跨视角辐照度、恢复像素级细节。两分支输出由 Gaussian Head 合并生成 HDR 3D 高斯，再通过 MetaNet 预测色调映射参数渲染可控曝光的 LDR 图像。

### 关键设计

1. **曝光归一化 $F_E$**: 定义相对对数曝光 $\tilde{\ell}_v = \ell_v - \bar{\ell}$，通过正弦位置编码得到曝光嵌入 $\mathbf{e}_v$。用 FiLM 层从曝光嵌入和外观特征均值预测逐视角仿射参数 $(\gamma_v, \beta_v)$，对外观 token 做调制：$\hat{\boldsymbol{t}}_v^A = \boldsymbol{t}_v^A \odot (1+\gamma_v) + \beta_v$，将所有视图对齐到共享辐照度水平。设计动机：消除亮度变化是跨视角融合的前提。

2. **几何引导的跨视角注意力 $F_A$**: 发现冻结几何编码器的全局注意力图已编码可靠的跨视角几何对应关系——在极端曝光变化（0.5s-32s）下仍能准确匹配叶子、杯子、门框等元素。直接复用第14层的 Q、K 矩阵引导外观融合：
$$\tilde{\boldsymbol{t}}_v^A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)\hat{\boldsymbol{t}}_v^A$$
设计动机：多曝光图像捕获互补信息（亮曝光揭示阴影，暗曝光保留高光），融合需要几何对应关系。

3. **MetaNet 场景自适应色调映射**: 元网络 $F_M$ 以 LDR 特征、曝光嵌入和预测的 HDR 高斯为输入，通过卷积编码器+全局池化预测色调映射器 $g_{\boldsymbol{\theta}}$（两层 MLP）的所有权重和偏置 $\boldsymbol{\theta}$。色调映射公式为 $\mathbf{L}_v(\ell) = g_{\boldsymbol{\theta}}(\log \mathbf{H}_v + (\ell - \bar{\ell})\cdot\log 2)$，支持任意目标曝光渲染。设计动机：不同相机使用不同色彩变换（AgX, Filmic等），学习统一映射器无法泛化，需要场景自适应。

### 损失函数 / 训练策略

- 总损失: $\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_g \mathcal{L}_g$
    - RGB 损失: MSE + $\lambda_{\text{perc}} \cdot \mathcal{L}_{\text{perc}}$（感知损失），$\lambda_{\text{perc}}=0.05$
    - 几何一致性损失: 冻结DPT头深度与渲染深度对齐，仅在置信度 top-30% 像素上监督，$\lambda_g=0.1$
- 无3D或HDR监督，仅用多视角LDR图像+已知曝光时间端到端训练
- 训练: 30K iterations, 8×A6000 GPU, ~2天, AdamW (lr=2e-4), bf16精度
- **HDR-Pretrain 数据集**: 168个 Blender 渲染室内场景，源自 HSSD，含多种灯光+3种色调映射算子(AgX/Filmic/Standard)，每场景35视角 × 5曝光

### 后优化（可选）

前馈输出后可做轻量后优化：剪枝低不透明度高斯(σ<0.01)，用 MSE+SSIM 损失微调 1K iterations，~30-40秒/场景。

## 实验关键数据

### 主实验

| 数据集 | 指标 | InstantHDR | 之前SOTA | 提升 |
|--------|------|----------|----------|------|
| HDR-NeRF Real (4-view, 后优化) | PSNR | **22.16 dB** | 19.26 (GaussianHDR) | +2.90 dB |
| HDR-NeRF Real (4-view, 后优化) | SSIM | **0.762** | 0.691 (GaussianHDR) | +0.071 |
| HDR-NeRF Real (4-view, 后优化) | Time | **32 s** | 1833 s (GaussianHDR) | **~57×** faster |
| HDR-NeRF Real (18-view, 后优化) | PSNR | 29.19 | **29.36** (GaussianHDR) | -0.17 (接近) |
| HDR-NeRF Syn (8-view, zero-shot) | PSNR | **22.58** | 14.51 (AnySplat) | +8.07 dB |
| HDR-NeRF Syn (8-view, 后优化) | PSNR | 32.75 | **34.49** (GaussianHDR) | -1.74 |
| HDR Syn (8-view, 后优化) | PSNR(μ-law) | **27.55** | 27.69 (HDR-GS) | 接近 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| InstantHDR (full) | 18.95 | 0.724 | 0.269 | 完整模型 |
| w/o MetaNet | 16.32 | 0.699 | 0.289 | 无法适应不同CRF，训练不稳定 |
| w/o Exposure Norm | 13.72 | 0.693 | 0.278 | **降幅最大**，亮度不一致破坏融合 |
| w/o Cross-view Attn | 17.63 | 0.702 | 0.277 | 平滑表面出现鬼影 |
| w/o Upsampling | 19.20 | 0.718 | 0.386 | 保留粗结构但丢失细节（LPIPS大幅下降） |

### 关键发现

- 曝光归一化是最关键模块——移除后 PSNR 下降 5.23 dB
- 前馈模型在稀疏视角下大幅优于优化方法（4-view: +2.90 dB vs GaussianHDR），因为前馈基础模型提供了丰富的几何先验
- Zero-shot HDR 输出亮度偏高（极端辐射值难以单次前向准确预测），1K 后优化可大幅缓解
- 冻结几何编码器的注意力图在极端曝光变化下仍能可靠匹配，这一发现对多曝光融合有通用启发

## 亮点与洞察

- **范式突破**: 首个前馈式 HDR NVS 方法，将 HDR 重建时间从分钟级降到秒级
- **几何引导外观融合设计优雅**: 直接复用冻结几何编码器的注意力图作为跨视角对应，零额外计算开销
- **MetaNet 预测色调映射参数**: 将通常需要逐场景优化的 CRF 学习变为前馈预测，区别于固定/可学习参数的传统做法
- **HDR-Pretrain 数据集填补空白**: 168场景远超现有 HDR 数据集（最大仅14场景），推动前馈 HDR 研究

## 局限与展望

- 在密集视角（18-view）下仍略逊于 GaussianHDR（-0.17 dB），可能受限于单分支色调映射
- Zero-shot HDR 输出亮度偏高，前馈模型预测极端辐射值仍是开放问题
- 合成→真实的域差距需要在 HDR-Plenoxels 真实场景上微调才能评估真实数据
- HDR-Pretrain 仅168个室内场景，多样性仍有扩展空间（室外、动态场景等）

## 相关工作与启发

- AnySplat / VGGT 等前馈几何方法的注意力图蕴含丰富的跨视角对应信息，这一发现可推广到其他需要多视角融合的任务
- FiLM 条件化用于曝光归一化是简洁有效的做法，可启发其他需要条件化的场景（如光照/天气适应）
- MetaNet 预测网络权重的元学习思路可推广到其他需要场景自适应的渲染参数预测

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个前馈HDR NVS，几何引导外观融合+MetaNet色调映射设计新颖
- 实验充分度: ⭐⭐⭐⭐ 多设置(4/8/18-view, zero-shot/后优化, LDR/HDR)全面对比+消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，pipeline图直观，公式符号一致
- 价值: ⭐⭐⭐⭐⭐ 将HDR 3D重建从分钟级压缩到秒级，实用价值极高，数据集有广泛复用价值

<!-- RELATED:START -->

## 相关论文

- [AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)

<!-- RELATED:END -->
