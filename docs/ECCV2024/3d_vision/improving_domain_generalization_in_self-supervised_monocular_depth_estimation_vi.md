---
title: >-
  [论文解读] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training
description: >-
  [ECCV 2024][3D视觉][自监督单目深度估计] 提出 SCAT 框架，通过缩放深度网络（SDN）降低 UNet 跳跃连接对扰动的敏感性，并引入冲突梯度手术（CGS）解决对抗增强导致的双重优化冲突，首次将对抗数据增强成功应用于自监督单目深度估计以提升跨域泛化能力。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "自监督单目深度估计"
  - "域泛化"
  - "对抗训练"
  - "梯度冲突"
  - "UNet"
---

# Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training

**会议**: ECCV 2024  
**arXiv**: [2411.02149](https://arxiv.org/abs/2411.02149)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 自监督单目深度估计, 域泛化, 对抗训练, 梯度冲突, UNet

## 一句话总结

提出 SCAT 框架，通过缩放深度网络（SDN）降低 UNet 跳跃连接对扰动的敏感性，并引入冲突梯度手术（CGS）解决对抗增强导致的双重优化冲突，首次将对抗数据增强成功应用于自监督单目深度估计以提升跨域泛化能力。

## 研究背景与动机

**领域现状**: 自监督单目深度估计（MDE）通过图像重建损失学习深度，无需标注数据，已在 KITTI 等驾驶场景中取得显著进展。代表方法包括 MonoDepth2、CADepth、MonoVit 等。

**现有痛点**: 现实世界中存在大量域偏移（如雾天、雨天、夜间），训练好的模型在未见场景中泛化能力严重不足。现有离线数据增强方法（如 Robust-Depth）需要预设目标分布，无法应对多样化的真实场景。

**核心矛盾**: 对抗数据增强（ADA）在有监督任务中已被验证能有效提升泛化，但直接应用于自监督 MDE 会导致严重性能退化甚至训练崩溃。这种矛盾的根源尚未被充分分析。

**本文目标**: 分析对抗增强在自监督 MDE 中失败的原因，并设计一种通用对抗训练框架使其稳定有效。

**切入角度**: 从两个角度切入——(i) UNet 长跳跃连接（LSC）对噪声的内在敏感性；(ii) 对抗增强导致的梯度方向冲突（双重优化冲突）。

**核心 idea**: 通过缩放跳跃连接系数抑制扰动放大 + 渐进式冲突梯度手术，实现稳定对抗训练。

## 方法详解

### 整体框架

SCAT（Stabilized Conflict-optimization Adversarial Training）是一个模型无关的对抗训练框架，包含三个核心组件：

- **对抗噪声生成器** $g_\phi$：学习生成最大化混淆深度网络的噪声 $\delta = g_\phi(z)$
- **缩放深度网络（SDN）**：调节 UNet 中 LSC 的缩放系数 $\kappa_i$ 以稳定训练
- **冲突梯度手术（CGS）**：从历史对抗生成器缓冲区中采样多个生成器，渐进式融合对抗梯度

训练采用 min-max 博弈形式：

$$\min_\theta \max_\phi \mathbb{E}_{\tilde{I}_{t'}, I_t} \mathbb{E}_{\delta \sim p_\phi(\delta)} \left[ \mathcal{L}(f_\theta(\tilde{I}_{t'}), I_t) \right]$$

关键设计是对抗增强仅施加于输入图像，重建目标仍为未增强的原始帧 $I_t$，避免错误引导。

### 关键设计

1. **缩放深度网络（SDN）**: 标准 UNet 的 LSC 公式为 $f_i(x) = b_{i+1} \circ [a_{i+1} \circ x + f_{i+1}(a_{i+1} \circ x)]$，SDN 引入缩放系数 $\kappa_i$：

$$f_i(x) = b_{i+1} \circ [\kappa_{i+1} \cdot a_{i+1} \circ x + f_{i+1}(a_{i+1} \circ x)]$$

理论分析证明，对于扰动 $\epsilon_\delta$，深度网络输出误差的上界为：

$$\|f_\theta(I_t^{\epsilon_\delta}) - f_\theta(I_t)\|_2 \leq \epsilon_\delta \left[\sum_{i=1}^{N} \kappa_i M_0^i + c_0\right]$$

标准 UNet（$\kappa_i = 1$）的上界为 $\mathcal{O}(NM_0^N)$，当 $N$ 较大时急剧增长。通过设置 $\kappa < 1$（默认 0.7），可以有效降低扰动敏感性，同时保持深度估计精度。

设计动机：跳跃连接虽然能融合多尺度特征、保留低级细节，但也为对抗噪声提供了直接传播通道，缩放系数抑制了这一放大效应。

2. **冲突梯度手术（CGS）**: 定义原始数据梯度 $g_i$ 与对抗数据梯度 $g_{mix_i}$ 的夹角 $\theta_i$，当 $\cos\theta_{ij} < 0$ 时梯度冲突。CGS 的核心目标：

$$\mathbb{E}[\cos(\theta)] > 0, \quad \cos(\theta_i) = \frac{g_i \cdot g_{mix_i}}{|g_i| |g_{mix_i}|}$$

具体实现：维护一个历史对抗生成器缓冲区 $\mathcal{B}$，每次训练从中随机采样 $j$ 个历史生成器，利用它们生成的多样化对抗样本进行渐进式梯度融合，而非仅用当前最强对抗器。

设计动机：单一对抗生成器持续强化后产生过度正则化，导致对抗梯度与原始梯度方向相反。通过混合多个历史时期的生成器，对抗强度被稀释，梯度余弦分布从负偏态转为正偏态。

3. **对抗训练流程**: 每个 epoch 包含：(1) 从缓冲区采样历史生成器产生对抗噪声 $\delta^{1:j}$；(2) SDN 分别处理干净图像和对抗图像产生深度图；(3) 用干净目标 $I_t$ 同时约束两个分支的重投影损失；(4) 将当前生成器存入缓冲区。

### 损失函数 / 训练策略

总体损失由两部分组成：

- **自监督约束** $\mathcal{L}_p$：对干净和对抗图像的重投影误差

$$\mathcal{L}_p = \sum_{t'} pe(I_t, I_{t' \to t}) + pe(I_t, \tilde{I}_{t' \to t})$$

- **对抗损失** $\mathcal{L}_{AD}$：用于优化对抗生成器

$$\mathcal{L}_{AD} = \sum_{t'} pe(I_t, \tilde{I}_{t' \to t})$$

其中 $pe$ 是加权 L1 和 SSIM 组合：$pe(I_a, I_b) = \frac{\alpha}{2}(1 - \text{SSIM}(I_a, I_b)) + (1 - \alpha)\|I_a - I_b\|_1$

扰动大小默认 $\epsilon_m = 135.0$，$\kappa$ 默认 0.7。

## 实验关键数据

### 主实验 — KITTI-C 跨域泛化

| 方法 | mCE(%)↓ | mRR(%)↑ | Abs Rel↓ | δ<1.25↑ |
|------|---------|---------|----------|---------|
| MonoDepth2 | 101.04 | 84.08 | 0.248 | 0.698 |
| + SCAT | **86.32** | **90.13** | **0.165** | **0.762** |
| MonoVit | 80.54 | 88.98 | 0.191 | 0.771 |
| + SCAT | **62.74** | **95.38** | **0.127** | **0.846** |
| Robust-Depth | 55.72 | 96.46 | 0.121 | 0.854 |
| + SCAT | **53.37** | **98.19** | **0.117** | **0.861** |

### 主实验 — KITTI 原域精度保持

| 方法 | Abs Rel↓ | RMSE↓ | δ<1.25↑ |
|------|----------|-------|---------|
| MonoDepth2 | 0.115 | 4.863 | 0.877 |
| + ADA (直接) | 0.121 | 4.992 | 0.862 |
| + SCAT | **0.116** | **4.877** | **0.877** |
| MonoVit | 0.099 | 4.372 | 0.900 |
| + ADA (直接) | 0.106 | 4.591 | 0.897 |
| + SCAT | **0.100** | **4.389** | **0.899** |

### 消融实验

| CGS | SDN | KITTI Abs Rel↓ | KITTI δ<1.25↑ | KITTI-C Abs Rel↓ | KITTI-C δ<1.25↑ |
|-----|-----|----------------|---------------|------------------|-----------------|
| ✗ | ✗ | 0.121 | 0.862 | 0.193 | 0.734 |
| ✓ | ✗ | 0.117 | 0.869 | 0.174 | 0.752 |
| ✗ | ✓ | 0.118 | 0.865 | 0.179 | 0.748 |
| ✓ | ✓ | **0.116** | **0.877** | **0.165** | **0.762** |

### 关键发现

- 直接 ADA 在所有基线上均导致 KITTI 原域性能退化，而 SCAT 几乎完全保持原域精度
- CGS 和 SDN 各自贡献独立且互补，合用效果最佳
- $\kappa = 0.7$ 是最优平衡点；$\kappa = 1.0$（标准 UNet）敏感性最高
- 扰动大小 $\epsilon_m = 135$ 为最佳，过大（180）反而泛化略降
- SCAT 对 5 种不同基线模型均有效，证明了模型无关性

## 亮点与洞察

- 首次系统分析了对抗训练在自监督 MDE 中失败的两大原因（LSC 敏感性 + 梯度冲突），理论分析清晰
- 提出的 SDN 仅需修改一个超参数 $\kappa$，极其轻量
- 历史缓冲区 + 渐进式融合的梯度手术策略优于简单的梯度裁剪/投影
- 在 NuScenes 夜间、Foggy CityScapes、DrivingStereo 等真实跨域场景验证，高度实用

## 局限与展望

- 仅探索了固定 $\kappa$ 值，未尝试可学习的自适应缩放
- 对抗生成器采用全像素级扰动，未考虑语义感知的局部扰动
- 训练成本因对抗训练而增加（双分支前向 + 生成器优化）
- 未在 Transformer-based 深度网络上系统消融 LSC 的影响

## 相关工作与启发

- **Robust-Depth**：离线增强的代表方法，SCAT 在其基础上仍能持续提升
- **PCGrad / GradNorm**：梯度冲突处理的经典方法，CGS 借鉴了类似思想但面向对抗训练场景
- 可借鉴思路：将 SDN 的缩放策略应用于其他依赖跳跃连接的 encoder-decoder 架构

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将对抗训练成功引入自监督 MDE，理论分析深入
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5 个基线 × 5 个数据集，消融完整
- **写作质量**: ⭐⭐⭐⭐ — 问题剖析清晰，方法动机充分
- **价值**: ⭐⭐⭐⭐ — 通用框架，对领域有持续推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [\[ECCV 2024\] ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)
- [\[ECCV 2024\] Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation](camera_height_doesnapost_change_unsupervised_training_for_metric_monocular_road-.md)
- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
