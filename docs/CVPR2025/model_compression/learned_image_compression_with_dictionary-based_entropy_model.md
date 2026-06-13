---
title: >-
  [论文解读] Learned Image Compression with Dictionary-based Entropy Model
description: >-
  [CVPR 2025][模型压缩][learned image compression] 提出基于字典的交叉注意力熵模型 (DCAE)，引入可学习字典从训练数据集中提取自然图像的典型纹理结构先验，通过多尺度特征聚合 + 交叉注意力实现精确的概率分布估计，在编解码速度仅 193ms 的条件下实现 -17.0%/-21.1%/-19.7% 的 BD-rate（Kodak/Tecnick/CLIC），全面超越 SOTA。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "learned image compression"
  - "entropy model"
  - "dictionary learning"
  - "注意力机制"
  - "rate-distortion"
---

# Learned Image Compression with Dictionary-based Entropy Model

**会议**: CVPR 2025  
**arXiv**: [2504.00496](https://arxiv.org/abs/2504.00496)  
**代码**: [GitHub](https://github.com/LabShuHangGU/DCAE)  
**领域**: 模型压缩  
**关键词**: learned image compression, entropy model, dictionary learning, cross attention, rate-distortion

## 一句话总结

提出基于字典的交叉注意力熵模型 (DCAE)，引入可学习字典从训练数据集中提取自然图像的典型纹理结构先验，通过多尺度特征聚合 + 交叉注意力实现精确的概率分布估计，在编解码速度仅 193ms 的条件下实现 -17.0%/-21.1%/-19.7% 的 BD-rate（Kodak/Tecnick/CLIC），全面超越 SOTA。

## 研究背景与动机

**领域现状**: 学习图像压缩 (LIC) 由非线性自编码器和熵模型两大组件构成。熵模型负责估计潜在表示的概率分布以实现高效熵编码，是决定压缩比的关键。现有方法主要通过 hyper-prior 和 auto-regressive 架构挖掘潜在表示的**内部依赖**。

**现有方案的局限**:
1. 现有熵模型只关注潜在表示内部的空间/通道依赖，**忽略了从训练数据中提取外部先验**
2. 训练数据中包含丰富的自然图像先验（典型纹理、结构），在图像恢复领域已被证明有效，但在 LIC 中未被充分利用
3. 串行自回归上下文模型虽然有效但编解码延迟高（如 MLIC++ 达 772ms）

**核心动机**: 自然图像存在大量重复的局部纹理模式（格子、条纹、边缘等），如果用一个共享字典存储这些模式，在自回归预测过程中就可以用已解码的部分信息查询字典，补全缺失信息，从而显著提高概率分布估计精度。

## 方法详解

### 整体框架

标准 LIC 流程：编码器 $g_a$ → 潜在表示 $\bm{y}$ → 熵模型估计分布参数 $(\bm{\mu}, \bm{\sigma})$ → 量化 + 熵编码 → 解码器 $g_s$ 重建。本文核心贡献在熵模型中引入字典模块。

### 关键设计

#### 1. 可学习字典 (Learnable Dictionary)

- 字典 $\bm{D} \in \mathbb{R}^{N \times C_d}$（$N=128$ 个词条，$C_d=640$ 维）作为可学习网络参数
- 训练过程中自动学习拟合自然图像中的典型结构（类似传统字典学习从图像 patch 学习）
- **编码器和解码器共享同一字典**，无需额外比特传输

#### 2. 多尺度特征聚合模块 (MSFA)

为了让字典查询更精准，需要从不同尺度提取纹理信息：
- 使用多层高效卷积（两个线性层 + 3×3 深度可分离卷积）堆叠 $m=3$ 层
- 浅层捕捉细粒度纹理，深层捕捉大尺度纹理
- 拼接后通过空间注意力加权：$\text{MSFA}(\bm{X}_i) = \text{SA}(\bm{X}_i^{merge}) \odot \bm{X}_i^{merge}$

#### 3. 基于字典的交叉注意力模块 (DCA)

自回归预测第 $i$ 个 slice 时：
- 已解码的 slice $\bar{\bm{y}}_{<i}$ 和 hyper-prior 特征 $\mathcal{F}_z$ 经 MSFA 得到查询特征 $\bm{X}_{ms_i}$
- **Query** = $\bm{X}_{ms_i} \bm{W}^Q$，**Key** = $\bm{D}\bm{W}^K$，**Value** = $\bm{D}$
- 交叉注意力：$\bm{A}_i = \text{SoftMax}(\bm{Q}_i \bm{K}^T / \tau)$，$\mathcal{F}_{dict_i} = \bm{A}_i \bm{V}$
- $\tau$ 为可学习温度参数，字典特征 $\mathcal{F}_{dict_i}$ 与内部特征一起送入分布估计模块

### 损失函数

标准拉格朗日 R-D 损失：$\mathcal{L} = \mathcal{R}(\hat{\bm{y}}) + \mathcal{R}(\hat{\bm{z}}) + \lambda \cdot \mathcal{D}(\bm{x}, \hat{\bm{x}})$

失真度量使用 MSE，$\lambda \in \{0.0018, ..., 0.0500\}$ 对应不同压缩率。

## 实验关键数据

### 主实验表

**BD-rate（以 VVC 为锚点，越低越好）**:

| 方法 | Kodak | Tecnick | CLIC | 延迟(ms) |
|------|-------|---------|------|---------|
| ELIC (CVPR'22) | -7.1% | - | - | 210 |
| TCM (CVPR'23) | -11.8% | -12.0% | -12.0% | 293 |
| MLIC++ | -15.1% | -18.6% | -16.9% | **772** |
| FTIC (ICLR'24) | -14.6% | -15.1% | -13.6% | - |
| CCA (NeurIPS'24) | -13.7% | -15.3% | -14.5% | 223 |
| **Ours (DCAE)** | **-17.0%** | **-21.1%** | **-19.7%** | 193 |

延迟仅为 MLIC++ 的 **1/4**，BD-rate 更优。

### 消融实验表

| 模块 | BD-rate | 延迟(ms) |
|------|---------|---------|
| Baseline | -4.20% | 143 |
| + DCA | -7.28% | 153 |
| + DCA + MSFA | -8.50% | 160 |

**字典大小消融**:

| 词条数 N | 无 | 64 | 128 | 192 | 256 |
|---------|-----|-----|------|------|------|
| BD-rate | -4.20% | -6.84% | **-7.28%** | -7.26% | -6.92% |

**MSFA 卷积层数消融**:

| 层数 m | 0 | 1 | 2 | 3 | 4 |
|--------|-----|-----|-----|-----|-----|
| BD-rate | -7.28% | -7.62% | -8.04% | **-8.50%** | -8.36% |

### 关键发现

- DCA 单独贡献 3.08% BD-rate 提升，仅增加 10ms 延迟
- 128 个字典词条已足够，继续增加反而饱和/退化
- 3 层 MSFA 为最优，多尺度特征对精确字典查询非常重要
- 可视化显示：相同字典词条在不同图像的**相似纹理区域**激活，验证了字典确实学到了典型结构先验

## 亮点与洞察

1. **外部先验的引入**: 首次在 LIC 熵模型中系统性地引入训练数据的外部先验（字典），补充了仅依赖内部依赖的不足
2. **无额外比特开销**: 字典作为共享网络参数存在于编解码两端，无需像 global token 那样每张图额外传输
3. **速度-性能的极佳平衡**: 193ms 延迟 + SOTA BD-rate，远优于 MLIC++ (772ms)
4. **与 global token 的对比**（Tab.5）: 128 词条的字典比 8 个 global token 性能更好（-7.28% vs -6.59%），因为字典可以存储跨图像的通用模式

## 局限与展望

1. 字典对**非自然图像**（如医学影像、遥感图像）的泛化性未验证
2. 当前字典词条固定不变，**在线自适应更新**可能进一步提升特定域的压缩效率
3. 仅使用 MSE 作为失真度量，**感知质量**（MS-SSIM, LPIPS）优化尚未探索
4. 编码器/解码器的非对称设计（不同阶段不同通道数）虽有效但缺乏系统搜索（如 NAS）

## 相关工作与启发

- **MLIC++** (Jiang et al.): 当前最强 BD-rate 但延迟过高，本文在保持速度优势的同时超越其性能
- **Kim et al. (global token)**: 用 8 个可学习 token 捕获全局信息，需传输，本文字典设计更优
- **CompressAI**: 统一的 LIC 评测框架
- **启发**: 字典学习 + 交叉注意力的范式可以推广到视频压缩的时间熵模型、点云压缩等领域

## 评分

⭐⭐⭐⭐ — 方法清晰、实验充分，在性能和效率上均达到 SOTA；创新点在于将字典先验引入熵模型，方向新颖且实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](../../ICML2026/model_compression/efficient_learned_image_compression_without_entropy_coding.md)
- [\[ECCV 2024\] Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model](../../ECCV2024/model_compression/bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)
- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)

</div>

<!-- RELATED:END -->
