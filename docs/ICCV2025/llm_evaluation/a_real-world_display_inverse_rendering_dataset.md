---
title: >-
  [论文解读] A Real-world Display Inverse Rendering Dataset
description: >-
  [ICCV 2025][逆渲染] 本文构建了首个基于LCD显示器-相机系统的真实世界逆渲染数据集，包含16个不同材质物体在OLAT照明模式下的立体偏振图像及高精度几何真值，并提出了一个简单有效的显示器逆渲染基线方法，超越了现有逆渲染方法。
tags:
  - ICCV 2025
  - 逆渲染
  - 显示器-相机系统
  - OLAT照明
  - 偏振成像
  - 光度立体
---

# A Real-world Display Inverse Rendering Dataset

**会议**: ICCV 2025  
**arXiv**: [2508.14411](https://arxiv.org/abs/2508.14411)  
**代码**: [https://michaelcsj.github.io/DIR/](https://michaelcsj.github.io/DIR/)  
**领域**: 计算机视觉 / 逆渲染  
**关键词**: 逆渲染, 显示器-相机系统, OLAT照明, 偏振成像, 光度立体

## 一句话总结
本文构建了首个基于LCD显示器-相机系统的真实世界逆渲染数据集，包含16个不同材质物体在OLAT照明模式下的立体偏振图像及高精度几何真值，并提出了一个简单有效的显示器逆渲染基线方法，超越了现有逆渲染方法。

## 研究背景与动机
- **领域现状**：逆渲染旨在从图像中恢复几何和反射率，现有方法依赖不同的成像系统——光照台（light stage）提供高质量多光源样本但造价高昂、体积庞大；闪光摄影需要多次移动相机；而显示器作为光源具有可编程、高分辨率、紧凑的优势
- **显示器-相机系统的独特优势**：每个像素可作为可编程点光源；LCD发出偏振光，天然支持漫反射/镜面反射分离
- **核心矛盾**：尽管显示器系统优势明显，却没有公开的实际数据集供研究。现有逆渲染数据集全部使用光照台、抓取机器人或自然光照等其他系统采集，无法评估显示器系统特有的挑战（近场照明、低信噪比、限制的光-视角采样等）
- **本文贡献**：填补这一空白——构建系统、采集数据、提供基准、验证方法

## 方法详解

### 整体框架
工作分为四个部分：（1）构建并标定LCD显示器+立体偏振相机系统；（2）采集16个物体在144个OLAT照明下的偏振图像；（3）提供结构光扫描的几何真值；（4）提出基线逆渲染方法并评估现有方法。

### 关键设计

1. **显示器-相机成像系统**:

    - 功能：构建由Samsung Odyssey Ark LCD显示器和两台FLIR偏振RGB相机组成的成像系统
    - 核心参数：显示器最大亮度600 cd/m²，每像素输出仅0.06 mcd；将显示像素分组为144个超像素（$16 \times 9$），每个超像素由 $240 \times 240$ 个显示像素组成
    - 标定内容：（a）显示器背光 $B_i$ 的空间变化建模；（b）非线性响应 $\gamma$ 的标定；（c）相机内外参标定；（d）超像素相对位置估计
    - 光强模型：$L_i = s(P_i + B_i)^\gamma$，其中 $s$ 是全局缩放因子

2. **数据采集与处理**:

    - 功能：采集16个不同材质物体在OLAT照明下的偏振图像，并获取几何真值
    - 材质覆盖：树脂、陶瓷、金属漆、木材、黏土、塑料、青铜、石膏等
    - 偏振处理：将四个角度的偏振图像转换为Stokes向量 $s_0, s_1, s_2$，分离镜面反射 $I_{\text{specular}} = \sqrt{s_1^2 + s_2^2}$ 和漫反射 $I_{\text{diffuse}} = s_0 - I_{\text{specular}}$
    - 几何真值：使用EinScan SP V2高精度3D扫描仪（精度0.05mm），通过互信息法将扫描网格与图像对齐

3. **图像形成模型与任意照明合成**:

    - 功能：利用非相干光传输的线性叠加性，支持任意显示模式下的图像合成
    - 核心公式：$I(\mathcal{P}) = \text{clip}(\sum_{i=1}^{N} I_i \cdot s(P_i + B_i)^\gamma + \epsilon)$
    - 设计动机：研究者可以合成任意照明模式的图像，并调整噪声水平，无需重新采集

4. **基线逆渲染方法**:

    - 功能：提出一个简单有效的显示器逆渲染基线
    - 流程：（a）用解析RGB光度立体法估计法线图；（b）用RAFT stereo估计深度图；（c）基于Cook-Torrance BRDF的基底BRDF表示，用可微渲染迭代优化法线和反射率
    - 关键技巧：用基底BRDF加权求和来建模空间变化BRDF，以应对受限的光-视角采样
    - 运行时间：仅需150秒完成优化

### 损失函数 / 训练策略
- 基线方法：最小化渲染图像与输入图像间的RMSE误差
- 显示器标定：优化全局标量 $s$、空间背光 $B_i$ 和非线性指数 $\gamma$，使渲染OLAT图像与采集图像匹配

## 实验关键数据

### 主实验（逆渲染评估）

| 方法 | 照明模式 | PSNR ↑ | SSIM ↑ | MAE (法线) ↓ |
|------|---------|--------|--------|-------------|
| SRSH | OLAT | 41.28 | 0.9895 | 25.25° |
| DPIR | OLAT | 34.30 | 0.9790 | 41.09° |
| IIR | OLAT | 38.20 | 0.9850 | 38.38° |
| **本文基线** | **OLAT** | **39.33** | **0.9821** | **20.94°** |
| **本文基线** | **Multiplexed** | **37.27** | **0.9766** | **23.97°** |

### 消融实验（光度立体评估 - 法线重建MAE）

| 方法 | 类型 | Elephant | Owl | Cat | Pig | 平均 |
|------|------|----------|-----|-----|-----|------|
| Woodham | 标定 | 27.02 | 26.60 | 21.05 | 17.02 | ~23° |
| PS-FCN | 标定 | 20.26 | 15.17 | 10.61 | 15.80 | ~15° |
| SDM-UniPS | 非标定 | 18.83 | 14.37 | 9.70 | 15.33 | ~15° |
| UniPS | 非标定 | 25.14 | 17.34 | 19.69 | 25.77 | ~22° |

### 关键发现
- SDM-UniPS在OLAT照明下表现最好，144张OLAT图像提供了充足的法线重建信息
- 本文基线在显示器逆渲染中全面超越现有方法，能有效处理近场照明和背光挑战
- 仅用2张复用照明模式即可实现合理的法线重建，但逆渲染精度仍不及144张OLAT
- 使用偏振分离的漫反射图像可提升法线重建精度，但效果因方法而异
- 光衰减建模至关重要——不建模时PSNR从39.78降至37.43

## 亮点与洞察
- **首创性**：第一个面向显示器-相机系统的真实世界逆渲染数据集，填补了重要的研究空白
- **完善的系统标定**：对背光、非线性、几何进行全面标定，使数据集具有高度可用性
- **偏振分离**：利用LCD偏振特性实现漫反射/镜面反射分离，为后续研究提供独特优势
- **光-视角分析**：通过Rusinkiewicz坐标系分析了采样覆盖范围，揭示了显示器系统在θ_h方向采样充分但θ_d方向受限的特点
- **合成能力**：基于光传输线性性，支持任意照明模式和噪声级别的图像合成

## 局限与展望
- 显示器单像素亮度极低（0.06 mcd），需要超像素分组才可用，限制了照明分辨率
- 当超像素过小（低于240×240）时拍摄图像太暗无法使用
- 光-视角采样范围有限，特别在 $\theta_d$ 方向覆盖不足
- 仅支持单视角+OLAT的采集方式，缺乏多视角同时采集的能力
- 物体数量（16个）和材质多样性仍可扩展

## 相关工作与启发
- **vs Light Stage数据集**：光照台提供远场均匀照明和更密集的角度采样，但成本高、体积大；显示器系统紧凑低成本但面临近场效应
- **vs DiLiGenT等光度立体数据集**：现有数据集用点光源，不涉及显示器特有的背光和近场问题
- **vs DDPS (可微显示光度立体)**：DDPS用3D打印物体，材质多样性有限；本文使用真实物体

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个显示器逆渲染数据集，填补空白
- 实验充分度: ⭐⭐⭐⭐⭐ 评估了大量光度立体和逆渲染方法，多维度消融
- 写作质量: ⭐⭐⭐⭐ 系统构建和数据集描述详实
- 价值: ⭐⭐⭐⭐ 为显示器逆渲染研究提供了标准化基准

<!-- RELATED:START -->

## 相关论文

- [OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition.md)
- [A Conditional Probability Framework for Compositional Zero-shot Learning](a_conditional_probability_framework_for_compositional_zerosh.md)
- [InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)
- [HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)

<!-- RELATED:END -->
