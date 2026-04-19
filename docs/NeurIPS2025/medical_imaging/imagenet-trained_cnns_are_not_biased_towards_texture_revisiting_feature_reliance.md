---
title: >-
  [论文解读] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
description: >-
  [NeurIPS 2025][医学图像][texture bias] 通过系统化的特征抑制框架（而非冲突选择实验）重新评估 CNN 的特征依赖性，发现 CNN 并非天然偏向纹理，而是主要依赖局部形状特征；且不同领域（CV/MI/RS）的特征依赖模式显著不同。
tags:
  - NeurIPS 2025
  - 医学图像
  - texture bias
  - feature reliance
  - CNN
  - feature suppression
  - domain-specific analysis
---

# ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression

**会议**: NeurIPS 2025  
**arXiv**: [2509.20234](https://arxiv.org/abs/2509.20234)  
**代码**: [GitHub](https://github.com/tomburgert/feature-reliance)  
**领域**: Medical Imaging / Computer Vision  
**关键词**: texture bias, feature reliance, CNN, feature suppression, domain-specific analysis

## 一句话总结

通过系统化的特征抑制框架（而非冲突选择实验）重新评估 CNN 的特征依赖性，发现 CNN 并非天然偏向纹理，而是主要依赖局部形状特征；且不同领域（CV/MI/RS）的特征依赖模式显著不同。

## 研究背景与动机

**领域现状**：Geirhos et al. 2019 年的 cue-conflict 实验提出了极具影响力的观点——ImageNet 训练的 CNN 天然偏向纹理（texture bias hypothesis），而人类更依赖形状。这一结论深刻影响了后续大量关于模型可解释性和鲁棒性的研究。

**现有痛点**：cue-conflict 实验存在三个关键方法论问题：（i）**特征未隔离**——纹理 cue 中保留了颜色和局部形状信息；（ii）**纹理信号过载**——纹理不仅覆盖目标区域还覆盖背景，空间信号不均衡；（iii）**人类界面偏向形状**——人类通过轮廓图标选择类别，可能引导形状选择。

**核心矛盾**：领域广泛接受的"CNN 偏向纹理"结论可能是实验方法的 artifact，而非 CNN 的内在属性。

**本文目标**：提出一种无偏的、跨领域适用的特征依赖性评估框架。

**切入角度**：从"冲突选择"（preference）转向"抑制测量"（reliance）——通过系统性地抑制单一特征类型并测量性能下降来量化依赖程度。

**核心 idea**：特征依赖应通过目标特征被移除后的性能退化来衡量，而非在冲突刺激下的偏好选择。

## 方法详解

### 整体框架

提出一个领域无关的特征依赖评估框架：针对形状（shape）、纹理（texture）、颜色（color）三种特征类型，分别使用两种互补的抑制变换，测量模型在抑制条件下的准确率相对于原始准确率的比值（relative accuracy），以此量化对各特征的依赖程度。

### 关键设计

1. **特征定义与抑制变换**：

    - **功能**：为三种特征各设计两种互补的抑制方法
    - **为什么**：使用两种不同机制的变换可避免单一变换引入的 artifact 影响结论
    - **怎么做**：
        - 形状抑制：Patch Shuffle（打乱 patch 位置）+ Patch Rotation（旋转 patch 但保持局部性）
        - 纹理抑制：Bilateral Filter（保边平滑）+ Gaussian Blur（高斯模糊）
        - 颜色抑制：Grayscale（灰度化）+ Channel Shuffle（通道打乱）
    - **区别**：与 cue-conflict 不同，不生成对抗性刺激，而是直接对原始图像操作

2. **定量验证指标**：

    - **功能**：用四个量化指标验证每种变换确实抑制了目标特征
    - **为什么**：需要实证证据表明抑制变换有效且不影响其他特征
    - **怎么做**：
        - 纹理指标：Local Variance (LV) 和 High-Frequency Energy (HFE)，期望降低
        - 形状指标：Edge-SSIM (ESSIM) 和 Gradient Correlation (GC)，期望保持
    - **关键结果**：Bilateral Filter 在抑制纹理 (LV:0.54, HFE:0.49) 同时最好地保持形状 (ESSIM:0.74, GC:0.85)

3. **人机对比实验设计**：

    - **功能**：在相同抑制条件下对比人类和模型的特征依赖
    - **为什么**：消除 Geirhos 实验中人类界面偏向形状的问题
    - **怎么做**：20名人类参与者通过文字标签（而非轮廓图标）选择类别；300ms 注视 + 200ms 图像呈现 + 200ms 粉噪声掩蔽；每人只看每张图的一种抑制版本
    - **区别**：使用字母排序的文字类名 grid，避免形状引导

### 跨领域分析

在三个领域（CV/MI/RS）中使用相同框架，固定 ResNet50 架构，通过连续变化抑制强度绘制抑制曲线，并进行归一化以跨数据集/领域比较。

## 实验关键数据

### 主实验：人类 vs CNN 特征依赖（相对准确率）

| 模型 | 全局形状 | 局部形状 | 纹理 | 颜色 | 原始准确率 |
|------|---------|---------|------|------|-----------|
| **Humans** | 0.965 | **0.763** | 0.979 | 0.999 | 0.969 |
| ResNet50-standard | 0.832 | **0.276** | 0.795 | 0.924 | 0.954 |
| ResNet50-sota | 0.943 | **0.618** | 0.867 | 0.948 | 0.931 |
| ConvNeXtV2 | 0.949 | **0.647** | 0.925 | 0.969 | 0.940 |
| ViT | 0.930 | **0.636** | 0.921 | 0.977 | 0.929 |
| CLIP ViT | 0.959 | **0.758** | 0.949 | 0.984 | 0.936 |

**关键发现**：所有 CNN 在**局部形状抑制**下性能降幅最大，而非纹理抑制。这直接挑战了 texture bias 假说。

### 消融实验：跨领域特征依赖差异

| 领域 | 最依赖特征 | 具体表现 |
|------|-----------|---------|
| **Computer Vision** | 局部形状 | 形状抑制引起最大性能下降；纹理/颜色抑制影响较小 |
| **Medical Imaging** | 颜色 | 灰度化导致显著准确率下降；反映诊断中色彩信息的关键作用 |
| **Remote Sensing** | 纹理 | 纹理抑制影响最大；RS 类别多由纹理模式定义（如田地、居民区） |

### 关键发现

- **核心结论**：CNN 并非天然偏向纹理。ResNet50-standard 在纹理抑制下保留 80% 性能（接近全局形状抑制的 83%），但在局部形状抑制下仅保留 28%
- **现代训练的影响**：ResNet50-sota 将局部形状抑制下的保留率从 28% 提升到 62%，ConvNeXtV2 进一步提升到 65%
- **视觉-语言模型最接近人类**：CLIP ViT 在所有抑制条件下最接近人类表现（局部形状保留 0.758 vs 人类 0.763）
- **ViT 不比 CNN 更偏向形状**：ViT 与 ResNet50-sota 的依赖特征 profile 非常相似

## 亮点与洞察

- **范式转变**：从"特征偏好"（preference in conflict）转向"特征依赖"（reliance through suppression），概念上更清晰
- **反直觉结论**：被广泛引用的 texture bias 实际上是局部形状依赖的误判——cue-conflict 的纹理 cue 中包含了局部形状信息
- **领域差异的发现极有价值**：CV/MI/RS 的特征依赖 profile 完全不同，表明模型的感知策略由数据和领域特性塑形，而非固定的架构偏差
- **实验设计严谨**：每种特征使用两种互补变换并进行定量验证，显著提高了可信度

## 局限与展望

- 特征抑制无法做到完美隔离：纹理抑制后可能残留低频纹理信息，形状抑制不能完全消除所有形状线索
- 抑制变换本身可能引入影响模型行为的 artifact（如 Patch Shuffle 的块状结构）
- 预训练模型的结果可能受数据增强与抑制变换相似性的影响（如 Cutout 与 Patch Shuffle）
- 人类实验在受控条件下进行（简短曝光、有限类别），可能未完全反映真实视觉感知

## 相关工作与启发

- **Geirhos et al. (2019)**：texture bias hypothesis 的原始工作，本文直接挑战其方法论
- **Hermann et al. (2020)**：指出 texture bias 主要来自训练目标和增强策略，而非架构本身
- **启发**：在评估模型行为时，实验设计中的隐性偏差可能导致根本性的误判。这一教训具有普遍意义——需要更仔细地分析"标准"评估协议中的假设

## 评分

- 新颖性: ⭐⭐⭐⭐ 框架思路清晰但各抑制变换并非全新；贡献在于系统化组合和挑战主流观点
- 实验充分度: ⭐⭐⭐⭐⭐ 人机对比、多架构、多领域、多数据集、定量验证完备
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑严密，对已有工作的批判有理有据
- 价值: ⭐⭐⭐⭐⭐ 挑战领域内最有影响力的假说之一，跨领域分析极具实用价值

<!-- RELATED:START -->

## 相关论文

- [The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)
- [Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)
- [Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM](unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)
- [eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis](../../ICML2025/medical_imaging/eccdnamamba_a_pre-trained_model_for_ultra-long_eccdna_sequence_analysis.md)
- [Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)

<!-- RELATED:END -->
