---
title: >-
  [论文解读] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass
description: >-
  [CVPR 2026][3D视觉][人体-场景联合重建] 提出CHROMM统一框架，从多人多视图视频中一次性联合估计相机参数、场景点云和人体网格（SMPL-X），无需外部模块或预处理数据，在全局人体运动估计和多视图位姿估计任务上取得竞争力性能，且比优化方法快8倍以上。
tags:
  - CVPR 2026
  - 3D视觉
  - 人体-场景联合重建
  - 多人多视图
  - SMPL-X
  - 3D基础模型
  - 前馈推理
---

# Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass

**会议**: CVPR 2026  
**arXiv**: [2603.12789](https://arxiv.org/abs/2603.12789)  
**代码**: https://nstar1125.github.io/chromm  
**领域**: 3D视觉  
**关键词**: 人体-场景联合重建, 多人多视图, SMPL-X, 3D基础模型, 前馈推理

## 一句话总结
提出CHROMM统一框架，从多人多视图视频中一次性联合估计相机参数、场景点云和人体网格（SMPL-X），无需外部模块或预处理数据，在全局人体运动估计和多视图位姿估计任务上取得竞争力性能，且比优化方法快8倍以上。

## 研究背景与动机
3D人体-场景联合重建是CV核心问题，应用于机器人、自动驾驶和AR/VR。**领域现状与痛点**：（1）UniSH、Human3R等**单目方法**取得进展但不支持多视图；（2）HSfM、HAMSt3R等**多视图方法**依赖外部模块（2D关键点检测器、跨视图ReID）或需要迭代优化，系统复杂度高、计算成本大；（3）基于外观的跨视图人物重识别（ReID）在穿制服等视觉相似场景中不可靠。

**核心矛盾**：如何在不依赖外部模块、预处理数据和迭代优化的条件下，一体化完成多人多视图的相机+场景+人体联合重建？**本文切入角度**：集成Pi3X（近度量尺度场景重建）和Multi-HMR（多人全身网格恢复）的先验到统一网络，加上尺度对齐模块解决两者间的尺度gap，设计测试时多视图融合策略和基于几何而非外观的多人关联方法。

## 方法详解

### 整体框架
输入多视图多时间步的RGB图像$\{I^v_t\}$，展平为序列$\{I_n\}$（利用Pi3的置换等变性）。双编码器提取场景特征$F^{scene}$（Pi3X）和人体特征$F^{human}$（Multi-HMR），场景特征通过Pi3X解码器重建点图和相机参数，人体特征中检测到的head token与场景token融合后回归SMPL-X参数。测试时进行多人关联+多视图融合+尺度调整。

### 关键设计
1. **双特征编码（Dual-Feature Encoding）**:

    - 功能：分别为场景和人体提取专门的特征表示
    - 核心思路：Pi3X编码器提取全局3D几何特征$F^{scene}$，Multi-HMR编码器提取人体专用特征$F^{human}$。两路特征**不做早期融合**——场景特征进Pi3X解码器，人体特征直接传给人体重建头
    - 设计动机：实验发现改变解码器输入分布（即使冻结权重）会损害几何重建性能，保持Pi3X解码器的输入分布不变才能充分利用其预训练先验

2. **Head-Pelvis尺度调整模块**:

    - 功能：解决Pi3X预测的近度量尺度场景与度量尺度SMPL之间的尺度不匹配
    - 核心思路：比较图像中的头-骨盆2D距离$\ell^{\text{img}}$与投影SMPL头-骨盆2D距离$\ell^{\text{smpl}}$，计算全局调整比例$r = \frac{1}{|\mathcal{S}|}\sum \frac{\ell^{\text{smpl}}}{\ell^{\text{img}}}$，最终$s^* = r \cdot s$
    - 骨盆定位采用**粗到精策略**：先用head token估计粗略骨盆位置，再在该位置采样patch做精细偏移
    - 设计动机：Pi3X的场景尺度可能偏小（SMPL穿透地面）或偏大（SMPL悬浮），head-pelvis距离是稳定的身体比例参考

3. **测试时多视图融合**:

    - 功能：无需优化地将各视角估计聚合为统一全局表示
    - 核心思路：将SMPL参数分为视角不变和视角依赖两类——
        - **视角不变**（shape $\beta$、canonical pose $\theta$）：直接取各视角预测均值
        - **视角依赖**（root rotation $R$、head translation $\tau$）：先用估计的相机外参变换到世界坐标系，旋转转为四元数取均值，平移用**多视图射线三角化**
    - 设计动机：Token级max-pooling会混合视角依赖特征损害视角不变参数估计；显式分开处理更合理

4. **基于几何的多人关联**:

    - 功能：建立跨视角的人物身份对应
    - 核心思路：单视角内用human token L2距离+Sinkhorn最优传输跟踪，帧间3D关节位移阈值过滤异常；跨视角用匹配代价$\mathcal{C}(a,b) = \lambda_p\|\mathcal{J}^a - \mathcal{J}^b\|_2 + \lambda_\theta\|\mathcal{J}^{a,\text{canon}} - \mathcal{J}^{b,\text{canon}}\|_2$（$\lambda_p=0.8, \lambda_\theta=0.2$），匈牙利匹配+阈值过滤
    - 设计动机：外观ReID在视觉相似场景中失败，3D位置+姿态的几何特征更鲁棒

### 损失函数 / 训练策略
- **两阶段训练**：Stage1冻结编码器，在BEDLAM上训练SMPL解码器+融合/mask/骨盆检测MLP（20 epochs）；Stage2仅训练骨盆检测MLP，在混合集上训练（10 epochs）
- Stage1：几何损失（3D顶点+关节+2D投影）+ 参数损失（pose/shape/translation）+ 检测损失（BCE for mask/head/pelvis）
- Stage2：骨盆检测 + 2D重投影 + Chamfer距离损失（可见SMPL顶点 vs 点图深度）
- 训练 4× A100 约2天

## 实验关键数据

### 主实验

| 数据集 | 指标 | CHROMM-multi | CHROMM-mono | Human3R | UniSH |
|--------|------|-------------|-------------|---------|-------|
| EMDB-2 | WA-MPJPE↓ | - | **102.6** | 112.2 | 118.5 |
| EMDB-2 | W-MPJPE↓ | - | **255.0** | 267.9 | 270.1 |
| RICH | WA-MPJPE↓ | **53.1** | 87.5 | 110.0 | 118.1 |
| RICH | W-MPJPE↓ | **79.0** | 138.3 | 184.9 | 183.2 |

多视图位姿（EgoHumans）：

| 方法 | W-MPJPE↓ | GA-MPJPE↓ | PA-MPJPE↓ | 时间↓ |
|------|----------|-----------|-----------|-------|
| **CHROMM** | **0.51** | **0.15** | **0.05** | ~4s |
| HSfM | 1.04 | 0.21 | 0.05 | ~118s |
| HAMSt3R | 3.80 | 0.42 | 0.14 | ~32s |

### 消融实验

| 配置 | WA-MPJPE↓ | W-MPJPE↓ | 说明 |
|------|-----------|----------|------|
| w/o scale adj. | 169.7 | 447.9 | 尺度不匹配严重影响全局精度 |
| w/ scale adj. | **102.6** | **255.0** | 尺度调整至关重要 |

多视图融合策略（RICH）：Only Avg. 69.3 → Max-Pool+Tri. 63.2 → **Avg.+Tri. 53.1**

多人关联：Position+Pose结合准确率**91.3%** vs 纯Pose 70.6% vs 纯Position 91.1%

### 关键发现
- 尺度调整模块将WA-MPJPE从169.7降至102.6（-39.5%）
- 显式分离视角不变/依赖属性+三角化比implicit max-pooling好16mm
- 基于几何的多人关联准确率91.3%，纯Pose精确率仅48.5%（over-matching严重）
- ~4s运行时间 vs HAMSt3R ~32s vs HSfM ~118s（8×加速）

## 亮点与洞察
- **"统一前馈"的系统级创新**：首个在多人多视图设置下不依赖外部模块、不需预处理、不需优化的联合重建框架
- **Head-Pelvis尺度比设计巧妙**：利用稳定身体比例关系解决Pi3X和SMPL的尺度gap
- **不做早期融合的决策**：反直觉但有效——保护预训练模型的输入分布比强行融合更重要
- **几何vs外观的ReID洞察**：3D位置+姿态比外观特征更鲁棒

## 局限与展望
- 重度依赖head token，头部严重遮挡时性能下降
- 编码器完全冻结，限制了模型对新场景的适应能力
- 基于几何关联需至少部分帧重叠可见
- 未来方向：集成双编码器为统一编码器、支持头部遮挡的鲁棒重建

## 相关工作与启发
- Pi3X + Multi-HMR集成范式可推广到其他"基础模型A + 专用模型B"的融合场景
- 测试时多视图融合策略（视角不变/依赖分离）值得在其他多视图估计任务中借鉴
- 几何ReID思路对多人多视角tracking有启发意义

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统级集成创新，各模块设计简洁实用
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集+两个任务+多组消融+运行时间对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表规范，贡献点表述明确
- 价值: ⭐⭐⭐⭐⭐ 首个统一多人多视图前馈重建框架，实际应用价值高

<!-- RELATED:START -->

## 相关论文

- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)
- [MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction](mv-roma_from_pairwise_matching_into_multi-view_track_reconstruction.md)
- [BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
