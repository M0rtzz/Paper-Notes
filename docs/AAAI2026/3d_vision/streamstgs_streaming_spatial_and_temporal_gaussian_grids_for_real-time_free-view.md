---
title: >-
  [论文解读] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video
description: >-
  [AAAI 2026][3D视觉][自由视角视频] 提出 StreamSTGS，一种可流式传输的时空高斯网格表示，将规范 3D 高斯属性编码为 2D 图像、时序特征编码为视频，实现实时自由视角视频流（帧大小仅 170KB），同时通过 Transformer 辅助训练和滑动窗口机制保证重建质量（PSNR 32.30dB）。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "自由视角视频"
  - "3D高斯溅射"
  - "实时流媒体"
  - "动态场景重建"
  - "自适应码率"
---

# StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video

**会议**: AAAI 2026  
**arXiv**: [2511.06046](https://arxiv.org/abs/2511.06046)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 自由视角视频, 3D高斯溅射, 实时流媒体, 动态场景重建, 自适应码率

## 一句话总结

提出 StreamSTGS，一种可流式传输的时空高斯网格表示，将规范 3D 高斯属性编码为 2D 图像、时序特征编码为视频，实现实时自由视角视频流（帧大小仅 170KB），同时通过 Transformer 辅助训练和滑动窗口机制保证重建质量（PSNR 32.30dB）。

## 研究背景与动机

### 问题定义

自由视角视频（FVV）是 VR 的核心应用，但 4D 表示需要巨大存储，实时传输面临三大挑战：

**训练效率**：逐帧训练策略导致累积误差

**渲染速度**：NeRF 基方法无法实时渲染

**传输效率**：现有 3DGS 基 FVV 方法每帧存储高达 10MB，无法实时流传输

### 已有方法的不足

**NeRF 基方法**（StreamRF, ReRF, HPC）：可以流传输残差，但体积渲染阻碍实时渲染。

**3DGS 基方法**存在以下问题：
- **3DGStream**：直接流传输 InstantNGP 预测属性偏移，模型体量大
- **HiCoM**：存储和流传输高斯属性偏移，但存在高时空冗余
- **QUEEN/4DGC**：使用熵编码压缩属性偏移，但解码延迟高
- **逐帧训练策略**带来显著累积误差，且需要为不同网络条件训练多个 LoD 模型
- 如果用户想看第 $i$ 帧，必须等待前 $i-1$ 帧全部推理完成

### 核心动机

作者的关键洞察是：如果将动态 3DGS 解耦为**规范高斯 + 时序特征 + 形变场**，然后将规范高斯属性存为图像、时序特征存为视频，就可以利用传统视频编解码器实现高效压缩和传输，同时天然支持自适应码率控制（无需额外训练）。

## 方法详解

### 整体框架

StreamSTGS 采用 GOP（Group of Pictures）结构，将长视频分割为多个组，每组独立重建。每个 GOP 内的 StreamSTGS 包含：
- **规范 3D 高斯集合 $\mathcal{G}$**：表示静态几何
- **时序特征 $\mathcal{E}$**：学习动态特征
- **形变场**：由多个 MLP 解码器组成，预测各属性的形变

最终，规范高斯属性被组织为 2D 图像（通过 PLAS 排序算法），时序特征被编码为视频，利用 H.264/HEVC 等标准编解码器压缩传输。

### 关键设计

#### 1. **滑动窗口时序特征聚合**：捕捉局部运动关系

**功能**：使用大小为 $W$ 的滑动窗口聚合相邻时序特征，作为形变场的输入。

**核心思路**：在时间戳 $t_i$ 时，拼接 $e_{i-1}, e_i, e_{i+1}$（$W=3$），经过时序 MLP $D_t$ 处理得到时序特征 $f_i$：

$$fe_i = \text{concat}(e_{i-1}, e_i, e_{i+1})$$
$$f_i = D_t(fe_i, \gamma(t_i))$$

其中 $\gamma(t_i)$ 是时间戳的位置编码。时序特征总数为 $E = G + W - 1$。

**设计动机**：真实世界中运动具有时间相关性，邻近帧的运动存在连续性。滑动窗口能让模型利用上下文信息更好地预测当前帧的运动，同时减少时序特征数量。

#### 2. **Transformer引导的辅助训练策略**：学习全局运动

**功能**：引入 Transformer 模块学习全局运动，然后通过自蒸馏将知识迁移到 StreamSTGS，推理时丢弃 Transformer 以保持高 FPS。

**核心思路**：采用两通道设计——高斯通道和辅助通道共享形变场。辅助通道中，Transformer 接收所有时间戳的特征作为序列，利用 Batch Attention 学习全局运动：

$$f'_i = \mathcal{F}(f_i, \gamma(t_i), \gamma(X))$$

自蒸馏损失将 Transformer 学到的全局运动知识迁移到主分支：

$$\mathcal{L}_{sd} = \|f_i - f'_i\|_1$$

**设计动机**：滑动窗口只能捕捉局部运动，对于大范围运动（如人的行走）无能为力。Transformer 可以关注所有时间戳，学习全局运动模式。但 Transformer 推理慢，因此训练时使用、推理时丢弃。

#### 3. **时空一致性正则化**：确保高压缩效率

**功能**：对规范高斯属性施加空间平滑正则化，对时序特征施加时间一致性正则化。

**核心思路**：
- **空间正则化** $\mathcal{L}_{spatial}$：对排序后的 2D 属性图像施加高斯滤波平滑
- **时间正则化** $\mathcal{L}_{temp}$：使用 Huber 损失衡量相邻时序特征的差异

$$L_i = \text{huber}(e_{i-1} - e_i), \quad \mathcal{L}_{temp} = \text{mean}(L_i, L_{i+1})$$

**设计动机**：Huber 损失的巧妙之处在于——对静态高斯（差异小）使用 MSE 保持一致性以减小数据量；对动态高斯（差异大）使用 MAE 容忍异常值避免过度平滑。这直接提升了视频编解码器的压缩效率。

#### 4. **动态感知密度控制与高斯重定位**

**Dynamic-aware density**：限制 SSIM 损失仅作用于动态区域像素，迫使更多高斯分配到动态区域：

$$\mathcal{L}_c = (1-\beta) \cdot \|\mathcal{I}_i - \mathcal{I}_i^{gt}\|_1 + \beta \cdot \text{SSIM}(\mathcal{I}_i \cdot I^{mask} - \mathcal{I}_i^{gt} \cdot I^{mask})$$

**Gaussian relocate**：在高斯数量达到上限（150k）后，将不必要的高斯移动到更优位置，避免陷入局部最优。

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L} = \mathcal{L}_c + 2\mathcal{L}_t + \mathcal{L}_{spatial} + \alpha_{temp} \cdot \mathcal{L}_{temp} + \alpha_o \cdot \mathcal{L}_o + \alpha_{sd} \cdot \mathcal{L}_{sd}$$

训练流程：先训练粗糙 3DGS 模型（3000 迭代），再对每个 GOP 进行精细训练（12000/7000 迭代）。训练时对滑动窗口特征添加小幅噪声（$\lambda=0.001$）模拟压缩损失。

## 实验关键数据

### 主实验

**N3DV 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 存储(KB/帧)↓ | K.F.大小(MB)↓ | 解码(ms)↓ | FPS↑ | 训练(s)↓ | 自适应码率 |
|------|-------|-------|--------|------------|-------------|----------|------|---------|----------|
| TeTriRF | 30.07 | 0.900 | 0.299 | 65.89 | 2.03 | 149 | 1.53 | 32 | ✓ |
| 3DGStream | 30.73 | 0.935 | 0.147 | 8204 | 42.22 | 7×n | 72 | 17 | ✗ |
| HiCoM | 31.32 | 0.939 | 0.147 | 10704 | 83.35 | 0 | 163 | 10 | ✗ |
| 4DGC | 31.52 | 0.941 | 0.143 | 784 | 21.94 | 2.5×n | 78.6 | 62 | ✗ |
| **Ours** | **32.30** | **0.943** | 0.147 | **173.6** | 3.86 | 8 | 100 | 67 | **✓** |

**MeetRoom 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 存储(KB/帧)↓ | FPS↑ |
|------|-------|-------|--------|------------|------|
| 3DGStream | 26.41 | 0.90 | 0.24 | 4108 | 121 |
| HiCoM | 26.69 | 0.90 | 0.23 | 5535 | 275 |
| 4DGC | 27.11 | 0.91 | 0.23 | 1196 | 110 |
| **Ours** | **27.41** | **0.92** | **0.21** | **142** | 126 |

### 消融实验

| 配置 | PSNR | 存储(KB) | 训练(s) | 说明 |
|------|------|---------|---------|------|
| Full model | **32.30** | 173.59 | 67 | 完整模型 |
| w/o Auxiliary training | 31.99 | 174.51 | 29 | 去掉Transformer辅助，质量下降 |
| w/o Dynamic density | 32.07 | 114.15 | 69 | 动态区域模糊，但压缩效率提高 |
| w/o Temporal reg. | 32.23 | **319.48** | 64 | 存储量激增近2倍 |
| w/o Gaussian relocate | 32.11 | 169.70 | 68 | 陷入局部最优 |

**滑动窗口大小消融**：

| 窗口大小 W | PSNR | SSIM | 存储(KB) |
|-----------|------|------|---------|
| W=1 (无滑窗) | 32.01 | 0.941 | 298.06 |
| W=3 (默认) | **32.30** | **0.944** | 173.59 |
| W=5 | 32.26 | 0.944 | 176.05 |

**压缩 QP 消融**：

| QP | N3DV PSNR | N3DV 存储(KB) | MeetRoom PSNR |
|----|-----------|-------------|---------------|
| 16 | 32.36 | 247.52 | 27.46 |
| 20 (默认) | 32.30 | 173.59 | 27.41 |
| 28 | 31.68 | 87.95 | 27.02 |
| 32 | 30.76 | 68.35 | 26.46 |

### 关键发现

1. **存储效率惊人**：每帧仅 170KB，比 3DGStream 小 47 倍，比 HiCoM 小 62 倍
2. **PSNR 最优**：比所有基线方法均高，N3DV 上比次优方法高 0.78dB
3. **天然支持自适应码率**：无需额外训练，直接通过调整 QP 参数实现不同码率
4. **时间一致性正则化的重要性**：移除后存储量增加 84%，说明它是压缩效率的关键
5. **GOP 长度 60 为最佳平衡**：质量、存储和训练时间的最优折衷

## 亮点与洞察

1. **设计哲学优雅**：将 3DGS 属性和时序特征分别映射到图像和视频这两种最成熟的媒体格式，充分利用现有编解码基础设施
2. **自适应码率免训练**：通过视频编解码器的 QP 参数直接控制码率，无需像其他方法那样训练多个 LoD 模型
3. **Huber 损失的精妙应用**：针对静态/动态高斯使用不同的惩罚机制，巧妙平衡一致性和表达力
4. **训练时添加噪声容忍压缩损失**：预先模拟编解码器引入的量化噪声，提升鲁棒性

## 局限与展望

1. **所有高斯都需要时序特征**：静态高斯不需要时序特征，分类处理可进一步减小存储和提升 FPS（论文已提及）
2. **GOP 边界可能引入不连续性**：不同 GOP 独立训练，结合处可能存在视觉跳变
3. **依赖预计算的动态 mask**：需要知道哪些像素是动态的，增加了预处理步骤
4. **Transformer辅助训练增加训练成本**：虽然推理时不用，但训练时间从 29s 增加到 67s

## 相关工作与启发

- 与 VideoRF/TeTriRF 的区别：后者使用 NeRF + 2D 网格存储体素特征并压缩为视频，但渲染速度慢且量化严重；StreamSTGS 使用 3DGS 实现实时渲染
- Compact-3DGS 的 PLAS 排序算法为本文的空间网格化提供了重要启发
- TimeFormer 的 Batch Attention 概念被借用到 Transformer 辅助训练模块

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 3DGS 属性映射为图像/视频是跨领域的有趣思路
- **实验充分度**: ⭐⭐⭐⭐⭐ — 两个数据集、全面的定量对比和详细的消融实验
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，但符号较多需要仔细阅读
- **价值**: ⭐⭐⭐⭐⭐ — 解决了 FVV 实时流传输的实际瓶颈，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](../../NeurIPS2025/3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[AAAI 2026\] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)

</div>

<!-- RELATED:END -->
