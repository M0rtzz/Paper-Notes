---
title: >-
  [论文解读] Unsupervised Exposure Correction
description: >-
  [ECCV 2024][无监督曝光校正] 提出首个无监督曝光校正（UEC）方法，利用ISP管线自由生成的多曝光序列让图像互为ground truth进行训练，设计仅含19K参数的像素级变换函数保留图像细节，在曝光校正和下游边缘检测上超越有监督SOTA。
tags:
  - ECCV 2024
  - 无监督曝光校正
  - 辐射测量建模
  - 像素级色彩变换
  - 多曝光序列
  - 边缘检测
---

# Unsupervised Exposure Correction

**会议**: ECCV 2024  
**arXiv**: [2507.17252](https://arxiv.org/abs/2507.17252)  
**代码**: https://github.com/BeyondHeaven/uec_code  
**领域**: 信号通信 / 图像曝光校正  
**关键词**: 无监督曝光校正, 辐射测量建模, 像素级色彩变换, 多曝光序列, 边缘检测

## 一句话总结

提出首个无监督曝光校正（UEC）方法，利用ISP管线自由生成的多曝光序列让图像互为ground truth进行训练，设计仅含19K参数的像素级变换函数保留图像细节，在曝光校正和下游边缘检测上超越有监督SOTA。

## 研究背景与动机

**领域现状**：曝光是影响图像质量的关键因素。尽管ISP可以自动调整曝光值（EV），但在非理想光照条件下，sRGB图像的后处理仍然至关重要。深度学习方法已在此领域取得显著成果，但现有方法面临三个核心挑战。

**挑战1：标注成本高昂**。有监督方法依赖专业摄影师手动调整生成配对数据作为ground truth，这一过程复杂且耗时——不同于分类任务的简单标注，每张图像需要精细的编辑和修正。MSEC数据集由5位专家分别标注，但如图所示，5位专家的标注结果在色彩和风格上存在明显差异。

**挑战2：泛化性有限**。(a) 手动标注效率低导致学术界可用数据集规模小；(b) 手动调整不可避免地引入个人风格偏差——不同修图师对"正确曝光"有不同理解，这意味着ground truth本身就是有噪声的。

**挑战3：低级特征退化**。现有方法主要追求生成美观的图像，但输出图像常在边缘等低级特征上出现显著退化。这使得增强后的图像在边缘检测、语义分割等依赖低级特征的下游任务中表现不佳。

**关键洞察**：获取配对数据不一定需要人工干预。通过模拟ISP管线在RAW数据上生成不同EV的多曝光序列，可以让同一序列中的图像互为ground truth——这些图像只在曝光（辐射度量）上不同，消除了风格偏差。

## 方法详解

### 整体框架

UEC框架包含三个核心网络：(1) **曝光特征编码器** $e(\cdot)$ 提取图像的曝光相关特征；(2) **参数预测器** $d(\cdot, \cdot)$ 计算两图之间的曝光差异并预测变换参数 $\lambda$；(3) **曝光校正器** $f(\cdot)$ 根据曝光差异对图像进行像素级校正。整个框架端到端训练，推理时只需一张参考图像即可确定目标曝光。

### 关键设计

#### 1. **无监督曝光校正建模**

**功能**：设计一种无需ground truth的自监督训练范式。

**核心思路**：利用ISP管线从RAW数据免费生成的多曝光序列，让序列中的图像互为ground truth。提出两条训练原则：

**原则1 — 恢复监督（Restoration Supervision）**：从同一多曝光序列采样输入 $I_1$ 和参考 $I_2$，训练变换函数使 $I_1' = I_2$。这作为pretext task建立基础映射：

$$I_1' = f(\Delta E, I_1), \quad \Delta E = d(e(I_1), e(I_2))$$

**原则2 — 单调性原则（Monopoly Principle）**：处理跨场景泛化问题。从不同序列 $J$ 中选择两张参考 $J_1, J_2$（$\text{EV}(J_1) > \text{EV}(J_2)$），对同一输入 $I_1$ 分别变换，要求输出满足逐像素亮度单调性：

$$\forall(x,y), \quad I_{J1}'(x,y) \geq I_{J2}'(x,y)$$

**设计动机**：恢复监督解决同场景曝光映射，单调性原则确保跨场景泛化。由于变换在潜空间而非像素空间计算曝光差异，只修改曝光不改变内容，因此可以适应不同场景的参考图像。

#### 2. **像素级曝光变换函数**

**功能**：设计保留图像细节的曝光校正操作。

**核心思路**：将曝光校正建模为直接缩放和非线性调整的插值，通过 $1 \times 1$ 卷积实现：

$$I_{out}(x,y) = \lambda \times I_{in}(x,y) + (1-\lambda) \times h(I_{in}(x,y))$$

其中 $\lambda$ 是参数预测器输出的混合权重，$h(\cdot)$ 是由 $1 \times 1$ 卷积层实现的非线性变换。该过程迭代3次以增强效果。

**设计动机**：与图像到图像翻译方法不同，像素级变换 (a) 保留原始像素关系，不引入伪影；(b) 可灵活处理任意分辨率输入，支持4K实时处理；(c) 参数量极少（仅19K），远低于ECM等方法（182M）。

#### 3. **曝光特征编码器**

**功能**：提取代表图像全局曝光属性的紧凑特征。

**核心思路**：编码器由两层卷积后接全局池化组成，通过最大值、均值和标准差三种统计量的组合生成96维特征表示。

**设计动机**：这些统计量关联图像的全局属性（如对比度、直方图分布），能有效刻画曝光特征。在潜空间而非像素空间计算曝光差，可省略低级特征信息，使 $d(\cdot,\cdot)$ 在不同场景间具有适应性。

### 损失函数 / 训练策略

总损失：$L = \alpha_1 \cdot L_{\text{restoration}} + \alpha_2 \cdot L_{\text{monopoly}} + \alpha_3 \cdot L_{\text{semantic}}$

1. **恢复损失**（同序列配对）：$L_{\text{restoration}} = \frac{1}{CHW}\|I^{\text{out}} - I^{\text{ref}}\|_2$

2. **单调性损失**（跨序列配对）：$L_{\text{monopoly}} = \frac{1}{CHW}\text{ReLU}(I^{\text{out2}} - I^{\text{out1}})$，当 $\text{EV}(I^{\text{ref1}}) > \text{EV}(I^{\text{ref2}})$ 时约束输出亮度单调

3. **语义保持损失**（全变分正则化）：$L_{\text{semantic}} = \frac{1}{CHW}\|\nabla I^{\text{out}}\|_2$，保持空间连贯性

权重设定：$\alpha_1 = \alpha_2 = 1, \alpha_3 = 0.1$。

**测试策略**：训练后固定一张曝光良好的参考图像，直接用其曝光特征作为所有测试样本的目标曝光，无需逐样本选择。

## 实验关键数据

### 主实验：MSEC数据集上的曝光校正

| 方法 | 监督类型 | PSNR↑ | SSIM↑ |
|------|----------|-------|-------|
| HDRCNN w/PS | 有监督 | 17.032 | 0.687 |
| DPED (iPhone) | 有监督 | 16.274 | 0.629 |
| DPE (S-FiveK) | 有监督 | 17.510 | 0.677 |
| Zero-DCE | 有监督 | 12.597 | 0.549 |
| Afifi et al. | 有监督 | 19.483 | 0.739 |
| ECM | 有监督 | **20.874** | **0.877** |
| **UEC (Ours)** | **无监督** | 18.756 | 0.812 |

UEC在无监督条件下接近SOTA有监督方法ECM的性能。

### 辐射校正数据集上的结果（消融：纯辐射性能）

| 方法 | 平均PSNR↑ | 平均SSIM↑ |
|------|-----------|-----------|
| ECM (有监督) | 20.445 | 0.744 |
| **UEC (无监督)** | **20.548** | **0.868** |

在纯辐射校正任务上，UEC在SSIM上显著优于ECM（0.868 vs 0.744），PSNR也略优。

### 消融实验：边缘检测下游任务

| 方法 | 平均PSNR↑ | 平均F1-Score↑ | 说明 |
|------|-----------|---------------|------|
| ECM (有监督) | 16.312 | 0.922 | 图像到图像翻译损失细节 |
| **UEC (无监督)** | **22.665** | **0.969** | 像素级变换保留低级特征 |

UEC在边缘检测PSNR上提升6.3dB（+39%），F1从0.922提升至0.969。

### 泛化性实验（MSEC训练→LOL测试）

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Afifi et al. (MSEC预训) | 14.268 | 0.638 |
| ECM (MSEC预训) | 15.439 | 0.650 |
| ECM (Radiometry预训) | 17.537 | 0.725 |
| **UEC (MSEC预训)** | **18.571** | **0.728** |

UEC泛化性最强，跨数据集PSNR提升3.1dB（vs ECM-MSEC）。

### 关键发现

1. **效率极高**：仅19K参数（ECM的0.01%），模型大小0.079MB vs 695MB，GPU推理速度快4.85倍（1.46ms vs 7.08ms），CPU快23.4倍
2. **辐射分离的优势**：仅学习辐射变化使模型泛化性更强，避免了人类标注的风格偏差
3. **单调性原则有效**：跨场景亮度单调性约束确保了合理的曝光迁移
4. **低级特征保留**：像素级变换相比图像翻译方法在边缘检测上提升显著

## 亮点与洞察

1. **巧妙的无监督设计**：利用多曝光序列图像互为ground truth的思路简洁而有效，完全消除了昂贵的人工标注
2. **极致的参数效率**：仅用19K参数（不到ECM的万分之一）实现竞争性性能，支持4K实时处理
3. **辐射-色度解耦**：仅修正辐射（亮度）而冻结其他ISP后处理，消除了风格偏差问题
4. **关注下游任务**：不仅追求视觉美观，还关注增强图像在边缘检测等下游任务中的实用性，这在曝光校正领域是有意义的新视角

## 局限与展望

1. **仅处理辐射校正**：不涉及色彩校正（色度调整），在需要色彩增强的场景中受限
2. **极端曝光下退化**：在-2EV到+3EV的大范围内，严重过/欠曝光图像因纹理丢失难以完全恢复
3. **单参考图像依赖**：测试时需选择一张合适的参考图像，虽然作者声称对参考选择鲁棒，但仍是一个额外假设
4. **评估指标局限**：PSNR/SSIM可能不完全反映人类对曝光质量的感知

## 相关工作与启发

- **Afifi et al. (MSEC)**：提出多曝光数据集和有监督曝光校正方法，本文正是在其ISP管线基础上转变为无监督范式
- **ECM**：当前SOTA有监督方法，采用图像到图像翻译，模型庞大且损失细节
- **Zero-DCE**：无参考方法但依赖手工设计的损失函数，非无监督
- **启发**：辐射-色度解耦的思路可推广到其他图像增强任务，"数据互为标签"的无监督策略值得在更多低级视觉任务中探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个无监督曝光校正方法，互为ground truth的设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集评估+泛化性分析+边缘检测下游任务+效率对比
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，恢复监督与单调性原则表述精炼
- **价值**: ⭐⭐⭐⭐ — 以极小参数量实现竞争性能，具有实际部署价值，辐射校正视角具有启发性

<!-- RELATED:START -->

## 相关论文

- [Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)
- [PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)
- [QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)
- [Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)

<!-- RELATED:END -->
