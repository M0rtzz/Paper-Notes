---
title: >-
  [论文解读] Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)
description: >-
  [ICCV 2025][模型压缩][color matching] 提出cmKAN，利用超网络驱动的Kolmogorov-Arnold Network进行颜色匹配，通过生成器预测空间变化的KAN样条参数，支持有监督/无监督/配对优化三种场景和raw-to-raw/raw-to-sRGB/sRGB-to-sRGB三种任务，在所有任务上平均超越现有方法37.3%且极轻量（76.4K参数）。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "color matching"
  - "KAN"
  - "Kolmogorov-Arnold Network"
  - "hypernetwork"
  - "camera ISP"
  - "raw-to-sRGB"
  - "color transfer"
  - "lightweight"
---

# Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)

**会议**: ICCV 2025  
**arXiv**: [2503.11781](https://arxiv.org/abs/2503.11781)  
**代码**: [https://github.com/gosha20777/cmKAN](https://github.com/gosha20777/cmKAN)  
**机构**: Samara National Research University, University of Wurzburg, York University
**领域**: 模型压缩 / 颜色匹配 / 轻量化网络  
**关键词**: color matching, KAN, Kolmogorov-Arnold Network, hypernetwork, camera ISP, raw-to-sRGB, color transfer, lightweight

## 一句话总结
提出cmKAN，利用超网络驱动的Kolmogorov-Arnold Network进行颜色匹配，通过生成器预测空间变化的KAN样条参数，支持有监督/无监督/配对优化三种场景和raw-to-raw/raw-to-sRGB/sRGB-to-sRGB三种任务，在所有任务上平均超越现有方法37.3%且极轻量（76.4K参数）。

## 背景与动机
不同相机ISP产生不同颜色渲染，颜色匹配旨在将源图像颜色映射到与目标一致。现有方法的问题：
1. 多项式方法无法准确捕捉复杂非线性变换
2. 深度CNN/MLP方法计算量大、参数多
3. 大多数方法假设源和目标图像都可用
4. 缺乏统一多场景的框架

## 核心问题
如何设计一个轻量且精确的通用颜色匹配框架？

## 方法详解

### 核心思想：KAN天然适合颜色匹配
颜色匹配公式：y_hat = F(x)·L。KAN通过可训练B样条基函数直接建模非线性映射：
y_hat_j = sum_i (u_ij·silu(x_i) + v_ij · sum_m c_ijm·B_ijm(x_i))
- 仅90个参数表达3→3颜色变换
- B样条比多项式更平滑更精确

### 空间变化的超网络KAN
标准KAN是全局操作，无法处理空间非均匀色差。解决：
- Generator G(X, θ)生成2D空间参数图W = (W_u, W_v, W_c)
- 每个空间位置有独立的90个KAN参数
- Y_hat = KAN(G(X, θ), X)

### Generator网络架构

1. **Illumination Estimator (IE)**：
    - 小型CNN处理输入和通道均值
    - 1x1 conv → 3x3扩张深度卷积 → 1x1 conv输出光照特征和光照图
    - 防止过曝/欠曝区域色彩失真

2. **Color Transformer (CT)**：
    - ViT-inspired架构，处理DWT下采样输入
    - Multi-Scale Color Attention (MCA)：在通道维度操作，引入anchor作中间桥梁，对Q和K做空间压缩降低计算量；V被光照特征调制

3. **Color Feature Modulator (CFM)**：
    - 处理IE和CT的级联特征
    - 通过可训练偏置的线性投影调制：X_m = B_i · ReLU(X'' · B_j)
    - 输出经FFN生成最终KAN参数图

### 三种训练场景
1. **有监督**：L = L1 + 0.15·(1-SSIM)
2. **无监督**：预训练（随机色彩扰动重建）+ CycleGAN-like无配对训练
3. **配对优化**：cmKAN-Light简化版，先预训练→特定对仅10步L1微调

## 实验关键数据

### Raw-to-Raw无监督映射

| 方法 | PSNR | SSIM | ΔE |
|------|------|------|-----|
| UVCGANv2 | 36.32 | 0.94 | 4.21 |
| RawFormer | 40.98 | 0.97 | 2.09 |
| **cmKAN** | **41.01** | **0.97** | **1.23** |

### sRGB-to-sRGB有监督（自有数据集）

| 方法 | PSNR | SSIM | #Params | Time |
|------|------|------|---------|------|
| MW-ISP | 23.31 | 0.76 | 29.2M | 8.8s |
| SIRLUT | 24.12 | 0.78 | 113.3K | 2.1s |
| **cmKAN** | **25.94** | **0.89** | **76.4K** | **1.1s** |

### MIT-Adobe FiveK

| 方法 | PSNR | ΔE |
|------|------|-----|
| SIRLUT | 27.25 | 6.19 |
| **cmKAN** | **31.74** | **2.83** |

- PSNR +4.49dB，ΔE降低54%

### 消融
- MLP→KAN: +2.29 dB
- 1D全局→2D空间: 进一步提升
- 逐步加IE/MCA/CFM：持续改进，最终+3.33 dB vs baseline

## 亮点
- **KAN天然适配颜色匹配**：理论基础扎实
- **超网络设计实现空间自适应**：优雅处理空间非均匀色差
- **极致轻量**：76.4K参数+1.1s推理，比同精度方法小100-400倍
- **三种场景统一框架**：有监督/无监督/配对优化都用同一架构
- **新数据集**：2.5K对齐双摄图像对
- **用户研究**：MOS分数是其他方法的2倍

## 局限与展望
- 单层3→3映射对极端色差可能不够
- 无监督两阶段流程复杂
- 推理时间1.1s对实时应用仍较长
- 跨品牌/跨光照验证不足

## 与相关工作的对比
- **vs. RawFormer**：26.1M参数 vs 76.4K（小343倍），性能相当甚至更优
- **vs. SIRLUT/SepLUT**：LUT精度不足；cmKAN的连续样条更精确
- **vs. NeuralPreset**：5.15M+20.4s vs cmKAN-Light 7.8K+1.5s

## 启发与关联
- KAN在低维信号处理中比MLP更优，对其他低维回归任务有启发
- 正确归纳偏置（样条）比堆参数更有效
- 对双摄系统色彩一致性和ISP开发有直接工程价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ KAN颜色匹配的理论自然性+超网络空间自适应
- 实验充分度: ⭐⭐⭐⭐⭐ 三种任务×三种场景+消融+用户研究+新数据集
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰但架构描述稍冗长
- 价值: ⭐⭐⭐⭐⭐ 极轻量+高精度+多场景通用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement](colors_see_colors_ignore_clothes_changing_reid_with_color_disentanglement.md)
- [\[ICCV 2025\] ARGMatch: Adaptive Refinement Gathering for Efficient Dense Matching](argmatch_adaptive_refinement_gathering_for_efficient_dense_matching.md)
- [\[ICCV 2025\] Variance-Based Pruning for Accelerating and Compressing Trained Networks](variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [\[NeurIPS 2025\] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](../../NeurIPS2025/model_compression/universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](../../CVPR2025/model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)

</div>

<!-- RELATED:END -->
