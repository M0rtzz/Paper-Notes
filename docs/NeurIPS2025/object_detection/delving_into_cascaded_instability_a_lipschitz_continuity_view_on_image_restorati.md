---
title: >-
  [论文解读] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes
description: >-
  [NeurIPS 2025][目标检测][flicker removal] 提出首个面向动态场景的多帧去闪烁（MFFR）基准数据集 BurstDeflicker，通过 Retinex 合成、真实静态采集与绿幕合成三种互补策略构建大规模训练/测试数据，显著提升闪烁去除模型在真实动态场景中的泛化能力。
tags:
  - NeurIPS 2025
  - 目标检测
  - flicker removal
  - rolling shutter
  - Retinex theory
  - 图像恢复
  - benchmark dataset
---

# BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes

**会议**: NeurIPS 2025  
**arXiv**: [2510.09996](https://arxiv.org/abs/2510.09996)  
**代码**: [qulishen/BurstDeflicker](https://github.com/qulishen/BurstDeflicker)  
**领域**: object_detection  
**关键词**: flicker removal, rolling shutter, Retinex theory, burst image restoration, benchmark dataset  

## 一句话总结

提出首个面向动态场景的多帧去闪烁（MFFR）基准数据集 BurstDeflicker，通过 Retinex 合成、真实静态采集与绿幕合成三种互补策略构建大规模训练/测试数据，显著提升闪烁去除模型在真实动态场景中的泛化能力。

---

## 研究背景与动机

- **闪烁伪影普遍存在**：短曝光图像在交流电（AC）供电光源下极易产生闪烁伪影，表现为非均匀亮度分布和明显暗带，在高速摄影、HDR 成像和慢动作视频中尤为突出。
- **影响下游视觉任务**：闪烁不仅降低图像视觉质量，还会影响目标检测、跟踪和识别等高层任务的性能，在需要可靠光照的场景中尤为关键。
- **硬件方案存在副作用**：传统硬件方案通过延长曝光时间减轻闪烁，但会引入运动模糊（motion blur），无法兼顾闪烁消除与清晰度保持。
- **单帧方法天然受限**：单帧去闪烁方法（SIFR）难以区分闪烁区域与阴影等类似暗区，且严重退化区域缺乏像素上下文，恢复结果不可靠。
- **现有合成数据不够真实**：先前的闪烁合成方法原本为地理标记设计，将闪烁建模为完全去除闪烁光照，与真实物理过程不符，导致训练的模型泛化能力差。
- **缺乏动态场景数据集**：动态场景具有不可重复性，无法获取对齐的闪烁/无闪烁图像对；没有大规模多帧去闪烁数据集严重阻碍了该领域研究进展。

---

## 方法详解

### 整体框架

BurstDeflicker 数据集从三个互补角度构建：（1）基于 Retinex 理论的大规模合成数据；（2）真实世界静态场景采集的 4,000 张配对图像；（3）绿幕合成的 3,690 张动态图像对。训练采用"合成预训练 → 真实数据微调"的两阶段流水线。

### 关键设计一：Retinex-based 闪烁合成

- **功能**：基于 Retinex 理论合成具有多样化闪烁模式的训练图像，支持无限量生成。
- **核心思路**：将场景光照分解为环境光 $L_a$ 与闪烁光 $L_f$，闪烁图像表示为 $I_{flicker} = R \odot (L_a + L_f)$。与先前工作将去闪烁目标定义为 $I_{clean} = R \odot L_a$（完全去除闪烁光）不同，本文提出正确目标应为 $I_{clean} = R \odot (L_a + \overline{L_f})$，即将瞬时闪烁光调整到有效值而非完全去除。
- **设计动机**：先前方法将闪烁光照视为有害并完全移除，这与物理现实不符——闪烁灯在一个完整周期内确实提供了照明，正确做法是保留其有效值。通过调整环境光与闪烁光比例 $k$、光源整流模式（全波整流、半波整流、PWM）和初始相位 $\varphi$，可合成丰富多样的闪烁模式。

### 关键设计二：真实世界静态数据采集

- **功能**：从 369 个真实场景中采集高质量闪烁/无闪烁配对序列，构建 BurstDeflicker-S 子集。
- **核心思路**：使用三脚架固定专业相机，以 1/1000–1/2000s 快门速度连拍 10 帧捕获闪烁伪影，再以 1/50 或 1/60s（匹配电网频率）长曝光获取无闪烁参考图像，并通过 ISO 调整保证曝光量一致。
- **设计动机**：合成数据存在领域差距（domain gap），真实数据有助于模型理解真实闪烁伪影的时空特征，提升对野外场景的泛化能力。采用遥控快门和电子快门消除机械抖动确保空间对齐。

### 关键设计三：绿幕合成动态数据

- **功能**：将绿幕前景（含运动的人物/物体）叠加到真实闪烁背景上，生成包含运动的闪烁图像对，构建 BurstDeflicker-G 子集。
- **核心思路**：从 VideoMatte240K 数据集选取前景片段，利用 alpha 遮罩与真实闪烁背景合成，同时对 clean GT 复制 10 帧并叠加相同前景保持一致性。
- **设计动机**：动态场景的不可重复性使得无法直接获取运动场景的闪烁配对数据。缺乏动态训练数据会导致模型将运动引起的像素变化误判为闪烁，产生鬼影（ghosting）伪影。绿幕合成巧妙解决了这一难题。

### 关键设计四：多帧训练流水线

- **功能**：设计面向多帧去闪烁的训练策略，支持不同网络架构。
- **核心思路**：每次迭代随机采样 3 帧（间隔 1–3 帧模拟不同帧率），增强后沿通道维度拼接输入网络，输出对应某一帧的去闪烁结果。训练阶段使用 resize 而非 crop 以保留闪烁的行方向周期性。
- **设计动机**：裁剪会破坏闪烁沿行扫描方向的周期模式，resize 保留了这一关键结构特征。

---

## 损失与训练策略

- 采用两阶段训练：先在 Retinex 合成数据上预训练（提供强初始化），再在 BurstDeflicker-S + BurstDeflicker-G 上微调
- 引入随机相机抖动增强（旋转 ±3°、平移 ±5 像素）模拟手持拍摄
- 评估了 Restormer、Burstormer、HDRTransformer 等多种 backbone

---

## 实验关键数据

### 表1：不同方法在静态与动态测试集上的定量比较

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | PIQE ↓ | BRISQUE ↓ |
|------|--------|--------|---------|---------|--------|-----------|
| Retinexformer* (未训练) | 15.70 | 0.707 | 0.213 | 53.60 | 50.27 | 30.24 |
| Lin et al.* (原始) | 20.36 | 0.838 | 0.134 | 55.23 | 43.88 | 25.12 |
| Lin et al. (重训练) | 26.41 | 0.875 | 0.102 | 58.13 | 35.71 | 22.10 |
| Retinexformer (重训练) | 27.21 | 0.885 | 0.081 | 58.25 | 35.94 | 21.65 |
| Burstormer | 29.44 | 0.910 | 0.056 | 58.53 | 37.01 | 20.45 |
| HDRTransformer | 30.03 | 0.914 | 0.054 | 59.07 | 37.29 | 21.59 |
| **Restormer** | **30.63** | **0.918** | **0.045** | **59.10** | **34.90** | **19.32** |

在本数据集上重训练后，各方法性能均大幅提升，验证了数据集的有效性。Restormer 在所有指标上取得最优。

### 表2：数据集组成消融实验

| 合成数据 | BurstDeflicker-S | BurstDeflicker-G | PSNR ↑ | SSIM ↑ | MUSIQ ↑ | BRISQUE ↓ |
|:---:|:---:|:---:|--------|--------|---------|-----------|
| ✓ | | | 24.48 | 0.862 | 57.10 | 23.76 |
| ✓ | ✓ | | 30.48 | 0.915 | 58.01 | 21.52 |
| ✓ | | ✓ | 30.65 | 0.916 | 58.43 | 20.34 |
| ✓ | ✓ | ✓ | **30.63** | **0.918** | **59.10** | **19.32** |

绿幕动态数据（BurstDeflicker-G）对提升动态场景鲁棒性贡献显著，BRISQUE 从 21.52 降至 19.32。

---

## 亮点

- **首个多帧去闪烁数据集**：填补了 MFFR 领域数据空白，三种互补数据源覆盖合成、真实静态和动态场景
- **修正闪烁去除的物理建模**：指出先前方法将闪烁光照完全去除的不合理性，提出基于有效值的正确公式，更符合物理现实
- **绿幕合成方法**：巧妙解决动态场景无法获取配对数据的难题，有效减少模型对运动像素的误判
- **全面的实验验证**：覆盖多种 baseline、多种指标、多种场景，包含 50 个真实动态测试序列

---

## 局限性

- 绿幕合成需要人工选择语义兼容的前景片段，劳动密集且难以大规模扩展
- 数据采集设备局限于特定相机型号，不同设备的 rolling shutter 特性可能不同
- 当前仅针对 rolling shutter 相机的行扫描闪烁，未涉及 global shutter 等其他曝光机制
- 绿幕合成的前景运动模式受限于 VideoMatte240K，可能无法覆盖所有真实动态场景

---

## 相关工作

- **硬件方法**：基于光电二极管的闪烁检测与曝光调整、事件相机的线性梳状滤波器，但成本高、部署难
- **单帧方法**：DeflickerCycleGAN（Lin et al.）使用 CycleGAN 框架配合 Flicker Loss 和 Gradient Loss，但单帧缺乏时间上下文
- **合成数据**：Wong et al. 的正弦强度叠加方法为地理标记设计，用于去闪烁训练效果不佳
- **多帧图像恢复**：Burstormer（burst 恢复）、HDRTransformer（HDR 成像）、Restormer（通用恢复）均可适配多帧输入

---

## 评分

- 新颖性: ⭐⭐⭐⭐ — Retinex 合成修正了先前公式，绿幕动态数据构建方法新颖实用
- 实验充分度: ⭐⭐⭐⭐ — 多角度消融（数据组成、帧数、方法对比），含真实动态测试
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，物理建模推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 填补 MFFR 数据空白，数据集和基线实验为后续研究奠定基础

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)
- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)

<!-- RELATED:END -->
