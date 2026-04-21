---
title: >-
  [论文解读] Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)
description: >-
  [ICCV 2025][图像恢复][All-in-One天气去除] SSGformer 提出一种基于光谱分解和分组注意力的 All-in-One 恶劣天气图像复原方法：利用 Sobel 算子提取高频边缘信息和 SVD 分析低频退化纹理，将二者融合后生成空间分组掩码（grouping-mask），在组内执行通道和空间注意力以实现对多种天气退化（雨、雪、雾、雨滴）的鲁棒去除。
tags:
  - ICCV 2025
  - 图像恢复
  - All-in-One天气去除
  - 光谱分解
  - 空间分组
  - Transformer
  - Sobel算子
  - SVD
  - 注意力机制
---

# Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)

**会议**: ICCV 2025  
**arXiv**: [2507.22498](https://arxiv.org/abs/2507.22498)  
**代码**: https://github.com/jeongyh98/SSGformer  
**领域**: 图像复原 / 恶劣天气去除  
**关键词**: All-in-One天气去除, 光谱分解, 空间分组, Transformer, Sobel算子, SVD, 注意力机制

## 一句话总结

SSGformer 提出一种基于光谱分解和分组注意力的 All-in-One 恶劣天气图像复原方法：利用 Sobel 算子提取高频边缘信息和 SVD 分析低频退化纹理，将二者融合后生成空间分组掩码（grouping-mask），在组内执行通道和空间注意力以实现对多种天气退化（雨、雪、雾、雨滴）的鲁棒去除。

## 研究背景与动机

**领域现状**：恶劣天气条件（雨、雪、雾、雨滴）会严重退化图像质量，影响下游视觉任务。早期研究针对单一天气条件建立专用模型，近年来 All-in-One（AiO）方法兴起，旨在用统一模型处理多种天气退化。代表方法有 AIRFormer、Fourmer、AdaIR、WeatherDiff 等。

**现有痛点**：
   - **频域方法的全局操作问题**：现有频域修复方法（Fourmer 通过傅里叶变换、AIRFormer 通过小波变换）对整个图像的频率做全局滤波。这对重复性退化（如模糊、均匀噪声）有效，但恶劣天气的退化是**高度不均匀和局部化**的（如雨滴只出现在局部位置），全局滤波难以精准处理。
   - **小波变换的分辨率损失**：小波变换会将图像分辨率减半，使用频域信息后需要上采样回原尺寸，可能引入失真。
   - **缺乏空间上下文的精细分组**：不同天气退化在空间上分布不均，需要识别并分组具有相似退化特性的区域来进行针对性修复。

**核心矛盾**：AiO 模型面临"多样退化模式"与"统一处理框架"之间的矛盾。天气退化的高度随机性和局部性要求模型既能提取有效的频谱先验，又能以空间感知的方式组织和处理特征。

**本文切入角度**：在保留空间细节的前提下提取频谱信息——用 Sobel 算子（不降分辨率）获取高频边缘特征，用 SVD（不降分辨率）获取低频退化纹理特征，然后基于这些特征生成空间分组掩码，在组内执行注意力。

## 方法详解

### 整体框架

SSGformer 是一个 4 阶段的 Transformer Encoder-Decoder 网络，包含三个核心模块：

1. **Spectral-based Decomposition Prompt (SDP)**：光谱分析与特征融合
2. **Mask Generator (MG)**：根据光谱特征生成空间分组掩码
3. **Spatial Grouping Transformer Block (SGTB)**：基于掩码的分组注意力

数据流：退化图像 $I_D$ → SDP 提取光谱特征 $F_S$ → MG 生成分组掩码 $M_p$ → SGTB 在组内做注意力 → 输出清晰图像 $I_C$

### 关键设计

1. **Spectral-based Decomposition Prompt (SDP)**：

    - **Sobel 算子**：检测高频信息，通过突出灰度强度变化来提取边缘特征 $F_{Sobel} \in \mathbb{R}^{H \times W \times 1}$。保持空间分辨率不变。
    - **SVD 滤波器**：对退化图像做奇异值分解，截断高频（保留前 $k$ 个奇异值），获取低频退化纹理特征 $F_{SVD} \in \mathbb{R}^{H \times W \times 1}$。保持空间分辨率不变。
    - **Spectral Feature Fusion Module**：对 $F_{Sobel}$ 和 $F_{SVD}$ 分别做精炼后（Sobel refinement block 用卷积 + 特征重组；SVD refinement block 加入可变形卷积与 $I_D$ 结合），通过多头线性注意力建模高频/低频特征间的互信息关系，输出融合特征 $F_S$。

2. **Mask Generator (MG)**：

    - 功能：从光谱融合特征 $F_S$ 生成空间分组掩码 $M_p$
    - 核心思路：掩码将图像区域按退化特性的空间相似度和纹理特征进行聚类分组。同一组内的区域共享相似的退化模式，便于在组内交互信息进行修复。
    - 设计动机：天气退化在空间上分布不均——雨滴覆盖局部、雪粒散布各处、雾霾在远处更浓。基于光谱信息的分组能自适应地识别这些退化区域。

3. **Spatial Grouping Transformer Block (SGTB)**：

    - 分为两种类型：**SGTB-C**（分组通道注意力）和 **SGTB-S**（分组空间注意力）
    - 功能：利用分组掩码将特征分组，在组内执行注意力操作
    - 核心思路：通道注意力关注同一组内不同通道之间的特征关系，空间注意力关注同一组内不同空间位置之间的关系。双注意力协同平衡了特征级和空间级的依赖关系。
    - 设计动机：在相似退化特性的区域内做注意力，比全局注意力更高效也更精准——避免了"干净区域"和"退化区域"之间的不当信息混合。

### 损失函数/训练策略

- 标准的图像复原损失（L1 loss + 感知损失）
- 多阶段训练：Encoder 4 阶段逐层下采样，Decoder 对称上采样 + 精炼块
- 每阶段中 SGTB-C 和 SGTB-S 交替使用，数量随阶段 $p$ 变化（由 $L_p$ 控制）

## 实验关键数据

### 主实验

在标准天气去除 benchmark 上评估（涵盖去雨、去雪、去雾、雨滴去除）：

- SSGformer 在多个 All-in-One 基准上取得 **SOTA 性能**
- 在复杂多天气混合退化场景中表现尤为突出
- 验证了跨多种退化类型的鲁棒一致性

### 与现有 AiO 方法对比

- 相比 Fourmer（傅里叶变换全局滤波）：SSGformer 在局部退化场景中优势更明显
- 相比 AIRFormer（小波先验）：SSGformer 保持了空间分辨率不损失，避免了上采样失真
- 相比 AdaIR（频域退化识别）：SSGformer 的空间分组机制更有效地处理了非均匀退化
- 不依赖外部知识（LLM/VLM），纯粹基于图像内部信息

### 消融实验关键结论

- **Sobel + SVD 组合的有效性**：仅用 Sobel 或仅用 SVD 效果均不如两者组合。Sobel 捕捉高频边缘（退化结构），SVD 捕捉低频成分（退化纹理模式），二者互补。
- **分组掩码的必要性**：去掉 MG 模块（直接全局注意力）性能显著下降，证实分组机制是性能提升的关键。
- **分组注意力 vs. 全局注意力**：在效率和效果上均优于全局注意力，因为分组减少了无关区域的信息干扰。
- **SDP 中多头线性注意力的作用**：比简单拼接更好地建模了高频/低频特征的交互关系。

### 关键发现

- 光谱先验无需傅里叶/小波变换：用传统的 Sobel 边缘检测和 SVD 低秩近似就能有效提取退化相关的频谱信息，且不损失空间分辨率
- 空间分组是 AiO 天气去除的关键：将空间上相似的退化区域聚类后进行组内注意力，比全局注意力更精准
- SGTB 的通道+空间双注意力设计平衡了多尺度依赖关系

## 亮点与洞察

- **"保留空间分辨率的光谱分析"理念**：传统频域方法（FFT、DWT）都会改变空间维度，而 SSGformer 选择 Sobel 算子和 SVD 在不改变空间分辨率的情况下提取频谱信息，这是一个简单但有效的设计选择。
- **分组注意力的退化感知性**：掩码不是固定的网格或规则分块，而是根据退化特性自适应生成的，这意味着模型能够自动识别"哪些区域受到类似退化并应该一起处理"。
- **不依赖外部知识**：在一些最新方法引入 LLM/VLM 外部知识的趋势下，SSGformer 坚持基于图像内部信息（光谱分解 + 空间聚类）来实现鲁棒修复，说明精心设计的内部先验利用足以达到 SOTA。
- **SVD 的退化纹理分析用途新颖**：SVD 通常用于低秩近似或降维，此处将其用于分析退化图像的纹理模式（截断后的低频信息 = 退化全局模式），是一个巧妙的应用扩展。

## 局限与展望

- **计算效率**：虽然分组注意力比全局注意力高效，但 SVD 分解本身的计算开销较大，可能限制在高分辨率图像或实时场景中的应用
- **分组数量和粒度**：MG 生成的分组掩码的粒度选择可能影响性能，过细或过粗的分组都可能带来问题，但论文中对此的自适应调节能力讨论有限
- **SVD 截断参数 $k$ 的选择**：保留多少奇异值是一个超参数，可能需要针对不同天气类型调优
- **混合天气退化**：虽然宣称处理多种天气，但对"同一图像中同时存在多种天气退化"（如雨+雾混合）的情况未做专门评估
- **Sobel 算子的固定性**：使用固定的 Sobel 核检测边缘，可能不如可学习的边缘检测器灵活

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Restoring Images in Adverse Weather Conditions via Histogram Transformer](../../ECCV2024/image_restoration/restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](../../ECCV2024/image_restoration/learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)
- [MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

<!-- RELATED:END -->
