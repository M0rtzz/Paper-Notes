---
title: >-
  [论文解读] Sky2Ground: A Benchmark for Site Modeling under Varying Altitude
description: >-
  [CVPR 2026][3D视觉][跨视角定位] 本文提出Sky2Ground数据集（51个场景，80k图像，统一覆盖卫星/航拍/地面三种视角的合成+真实图像）和SkyNet模型（双流编码器+掩码卫星注意力+渐进式视角采样），首次系统研究了跨地面/航拍/卫星三视角联合相机定位问题，在RRA@5上提升9.6%，在RTA@5上提升18.1%。
tags:
  - CVPR 2026
  - 3D视觉
  - 跨视角定位
  - 卫星-航拍-地面
  - 多海拔3D重建
  - 高斯溅射
  - 课程学习
---

# Sky2Ground: A Benchmark for Site Modeling under Varying Altitude

**会议**: CVPR 2026  
**arXiv**: [2603.13740](https://arxiv.org/abs/2603.13740)  
**代码**: 即将公开  
**领域**: 3D视觉 / 跨视角定位  
**关键词**: 跨视角定位, 卫星-航拍-地面, 多海拔3D重建, 高斯溅射, 课程学习

## 一句话总结
本文提出Sky2Ground数据集（51个场景，80k图像，统一覆盖卫星/航拍/地面三种视角的合成+真实图像）和SkyNet模型（双流编码器+掩码卫星注意力+渐进式视角采样），首次系统研究了跨地面/航拍/卫星三视角联合相机定位问题，在RRA@5上提升9.6%，在RTA@5上提升18.1%。

## 研究背景与动机
1. **领域现状**：多视角3D重建和相机定位是计算机视觉的基础任务。近年来DUSt3R、MASt3R、VGGT等基于神经网络的方法取得了显著进展，但主要在地面-航拍视角上训练和评估。
2. **现有痛点**：(1) 缺乏同时包含地面、航拍、卫星三种视角的数据集——nuScenes/KITTI只有地面视角，AerialMegaDepth缺少卫星，MatrixCity/BungeeNeRF仅有合成数据；(2) 没有研究过三视角联合相机定位问题；(3) 卫星图像与地面/航拍图像之间存在巨大的分布偏移。
3. **核心矛盾**：卫星图像提供全局一致的地理覆盖和稳定参考，但与地面/航拍视角的视觉差异极大（近正交视角、千米级高度差）。直觉上加入卫星应该提供更多信息，但实验发现反而损害了定位性能。
4. **本文目标** (1) 构建首个覆盖三种视角+真实/合成图像的数据集；(2) 分析为什么卫星图像会损害现有模型性能；(3) 提出能有效利用卫星信息的新架构。
5. **切入角度**：作者发现简单微调VGGT加入卫星数据会导致性能暴跌18.2%，但DUSt3R/MASt3R这类逐对处理的网络反而能受益。这说明问题不在于分布偏移本身，而在于全局注意力架构让地面/航拍token与卫星token交互时受到了干扰。
6. **核心 idea**：通过掩码卫星注意力阻止地面/航拍token直接关注卫星token，并用渐进式采样策略逐步引入更远视角，实现跨海拔联合定位。

## 方法详解

### 整体框架
SkyNet基于VGGT构建，采用双流编码器架构：GAS编码器处理所有视角的联合表示（但限制地面/航拍与卫星的注意力交互），Sat编码器专门处理卫星图像。两个编码器通过卫星特征的加法融合连接。最终通过共享的Camera Head和DPT Head分别预测相机参数和深度图。

### 关键设计

1. **Sky2Ground数据集**:

    - 功能：提供首个覆盖卫星/航拍/地面三视角的多模态数据集
    - 核心思路：51个地理位置（覆盖全球），每个场景包含120张卫星图（1-2km高度，正射校正）、1080张合成航拍图（使用三相机虚拟设备沿螺旋轨迹下降，250-800m高度）、50-250张合成地面图、各120张真实航拍/地面图（从Google Maps和YouTube旅游视频手动搜集）。使用Google Earth Studio渲染合成数据，COLMAP生成稠密深度图标注
    - 设计动机：真实图像引入光照变化、天气噪声等，合成图像提供精确的相机pose和深度标注，两者互补

2. **掩码卫星注意力(MSA)**:

    - 功能：防止地面/航拍token被卫星token的异质分布干扰
    - 核心思路：在GAS编码器的每个block中，先执行标准自注意力（帧内），然后执行MSA：卫星token可以关注地面/航拍token，但地面/航拍token被禁止关注卫星token。注意力掩码矩阵在卫星→地面/航拍方向设为$-\infty$。GAS编码器的自注意力和MSA层用预训练的VGGT权重初始化并冻结
    - 设计动机：实验表明VGGT微调后性能暴跌是因为全局注意力让地面/航拍特征被卫星特征"污染"。MSA保留了VGGT在地面/航拍上的零样本能力（因为这些token永不与卫星交互），同时让卫星token从地面/航拍token中获取信息

3. **渐进式视角采样(P-VS)**:

    - 功能：通过课程学习策略逐步增加训练难度
    - 核心思路：训练初期采样更多航拍图像（$N_a \approx N$），作为地面和卫星之间的"桥梁"。随着训练推进，逐步减少航拍图像比例（$N_a \approx 0$），最终只保留地面和卫星图像。这使模型从简单问题（三视角联合定位）逐步过渡到困难问题（仅地面+卫星定位）
    - 设计动机：地面和卫星是极端视角对，直接联合训练困难太大。航拍视角可以作为中间"桥梁"，先建立地面-航拍-卫星的渐进关联

### 损失函数 / 训练策略
多任务损失 $\mathcal{L} = \mathcal{L}_{\text{cam, sat}} + 0.4 \cdot \mathcal{L}_{\text{cam, gnd/aerial}} + \mathcal{L}_{\text{depth}}$。另外有Curriculum Aware Camera-Sampling(CA-CS)策略：训练初期采样距离近的相机对，逐步扩展到远距离相机对，距离度量为旋转距离+0.5×平移距离。

## 实验关键数据

### 主实验（GAS设置，RRA@5 / RTA@5 %）

| 方法 | 训练数据 | Ground RRA/RTA | Sat RRA/RTA | Aerial RRA/RTA | 平均RRA/RTA |
|------|---------|---------------|-------------|---------------|------------|
| VGGT | 零样本 | 75.1/60.9 | 66.6/0.0 | 79.2/72.6 | 73.6/44.5 |
| VGGT | Sky2Ground | 50.0/46.1 | 86.6/53.3 | 29.7/31.5 | 55.4/43.6 |
| SkyNet | Sky2Ground | **76.7/64.2** | **88.9/57.3** | **84.0/78.1** | **83.2/66.5** |

### 消融实验（G+S设置）

| 配置 | MSA | CA-CS | P-VS | 平均性能 |
|------|-----|-------|------|---------|
| VGGT微调 | ✗ | ✗ | ✗ | 47.8 |
| VGGT零样本 | ✗ | ✗ | ✗ | 52.9 |
| +MSA | ✓ | ✗ | ✗ | 62.7 (+8.2) |
| +P-VS | ✗ | ✗ | ✓ | 61.1 (+7.3) |
| +MSA+CA-CS+P-VS | ✓ | ✓ | ✓ | **65.1 (+12.2)** |

### 关键发现
- **微调VGGT加卫星反而严重退化**：RRA从73.6%跌至55.4%（-18.2%），这是核心发现
- **MSA是贡献最大的单一组件**：+8.2%，因为它保护了地面/航拍特征不被卫星干扰
- **P-VS比CA-CS更有效**：+7.3% vs +1.4%，说明"用航拍做桥梁"比"由近及远采样"更关键
- **逐对处理的网络能受益于卫星**：DUSt3R/MASt3R加入卫星后性能提升，因为配对处理中卫星-卫星对的高共视率有利于全局对齐
- **真实图像损害渲染质量**：加入真实图像后PSNR一致下降，域差距导致GS难以混合两种来源
- **2DGS始终优于3DGS**：在所有视角和密度下，2D高斯溅射的感知质量更好

## 亮点与洞察
- **"加数据反而变差"的反直觉发现**极具启发性：加入卫星——这一在信息论上更丰富的数据源——反而损害性能，说明当分布偏移足够大时，更多数据不等于更好结果。这挑战了"scale everything"的思维
- **MSA的设计思路可广泛迁移**：任何涉及异质模态（如文本+图像、RGB+热成像）的Transformer架构中，如果某种模态的分布差异太大，可以用非对称注意力掩码来规避干扰
- **航拍作为"桥梁模态"的课程学习**：这种从中间模态逐步过渡到极端模态的训练策略，可以推广到任何多模态对齐任务

## 局限与展望
- 方法是两阶段的（先预测pose，再高斯溅射），未来可探索统一模型
- 51个场景对于大规模训练可能不足
- 卫星图像的正射校正依赖额外处理
- 真实图像的pose通过COLMAP估计，精度有限
- 未探索更先进的域适应技术来弥合合成-真实差距

## 相关工作与启发
- **vs AerialMegaDepth**: 最相关的数据集，但缺少卫星视角；Sky2Ground是其超集
- **vs VGGT**: SkyNet建立在VGGT之上但解决了其在卫星视角上的崩溃问题
- **vs DUSt3R/MASt3R**: 逐对处理虽然能利用卫星信息但复杂度为$O(N^2)$，不适合实时应用
- **vs Dragon**: Dragon也用渐进策略整合不同高度图像，但仅用于重建，不涉及定位

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究三视角联合定位，MSA和P-VS设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖定位和渲染两大任务，多种baseline对比，详细的消融
- 写作质量: ⭐⭐⭐⭐ 分析深入，反直觉发现表述清晰
- 价值: ⭐⭐⭐⭐ 数据集和benchmark对跨视角定位领域有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario](freeartgs_articulated_gaussian_splatting_under_free-moving_scenario.md)
- [\[CVPR 2026\] NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather](nimbusgs_unified_3d_scene_reconstruction_under_hybrid_weather.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)

</div>

<!-- RELATED:END -->
