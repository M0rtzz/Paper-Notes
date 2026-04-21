---
title: >-
  [论文解读] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field
description: >-
  [ECCV 2024][3D视觉][稠密视觉SLAM] 提出CG-SLAM,基于一致性和几何稳定性优化的不确定性感知3D高斯场,实现高效稠密RGB-D SLAM,在定位精度和建图质量上均达到SOTA,跟踪速度最高15Hz。
tags:
  - ECCV 2024
  - 3D视觉
  - 稠密视觉SLAM
  - 3D高斯溅射
  - 不确定性建模
  - 实时定位与建图
  - 各向同性正则化
---

# CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field

**会议**: ECCV 2024  
**arXiv**: [2403.16095](https://arxiv.org/abs/2403.16095)  
**代码**: [有](https://zju3dv.github.io/cg-slam)  
**领域**: 3D视觉  
**关键词**: 稠密视觉SLAM, 3D高斯溅射, 不确定性建模, 实时定位与建图, 各向同性正则化

## 一句话总结

提出CG-SLAM,基于一致性和几何稳定性优化的不确定性感知3D高斯场,实现高效稠密RGB-D SLAM,在定位精度和建图质量上均达到SOTA,跟踪速度最高15Hz。

## 研究背景与动机

基于NeRF的SLAM方法(NICE-SLAM、Co-SLAM、Point-SLAM)效果好但**体渲染管线计算密集且耗时**,只能采样有限的摄像线。3DGS的光栅化渲染天然更高效,但直接用于SLAM面临过拟合、几何不准确、效率瓶颈三个问题。并发工作GS-SLAM和SplaTAM缺乏针对性设计。

## 方法详解

### 整体框架

四个核心模块: GPU加速光栅化器(基于完整位姿导数分析)、不确定性建模、一致性建图、高效跟踪(序贯跟踪+滑动BA)。

### 关键设计

#### 1. 多模态渲染

光栅化器同时渲染颜色图、Alpha-blending深度、中位深度(累积透射率首次低于0.5处)和累积不透明度图(检测未观测区域)。

#### 2. 不确定性建模(核心创新)

**不确定性图**: 用深度渲染的方差衡量,几何方差损失约束高斯原语贴近真实深度。

**对齐损失**: 使alpha-blending深度和中位深度一致,迫使每个像素上权重最大的高斯原语("主导高斯")出现在中位深度位置。

**高斯原语不确定性**: 定义为关键帧窗口中所有主导像素上的深度偏差加权平均。超过阈值(0.025)的被降低不透明度后再优化,真正不可恢复的被移除——自适应渐进式清理策略。

#### 3. 各向同性正则化

引入软约束鼓励高斯椭球趋向球形: 最大/最小缩放因子比值不超过阈值(1.0)。防止箭头状伪影,在跟踪精度和渲染真实感间取平衡。

#### 4. 跟踪模块

首次对EWA溅射中位姿导数进行完整数学推导。**Lie代数表示**更适合高斯场中的相机跟踪。序贯跟踪: 恒速假设初始化+重渲染损失优化。滑动BA: NetVLAD确定共可见性(比视锥重叠更高效),联合优化位姿和场景。

### 损失函数 / 训练策略

建图总损失 = 颜色L1 + SSIM + 深度L1 + 对齐损失 + 各向同性正则 + 几何方差。跟踪损失 = 颜色L1 + 深度L1。高斯管理: 首帧密集初始化,后续对不透明度低的像素补充新高斯。i9-14900K + RTX 4090。

## 实验关键数据

### 主实验: Replica跟踪精度 (ATE RMSE [cm]↓)

| 方法 | rm-0 | rm-1 | rm-2 | off-0 | off-1 | Avg. |
|------|------|------|------|-------|-------|------|
| NICE-SLAM | 0.97 | 1.31 | 1.07 | 0.88 | 1.00 | 1.06 |
| Co-SLAM | 0.77 | 1.04 | 1.09 | 0.58 | 0.53 | 0.99 |
| Point-SLAM | 0.56 | 0.47 | 0.30 | 0.35 | 0.62 | 0.54 |
| SplaTAM | 0.31 | 0.40 | 0.29 | 0.47 | 0.27 | 0.36 |
| **CG-SLAM** | **0.29** | **0.27** | **0.25** | **0.33** | **0.14** | **0.27** |

平均ATE 0.27cm,超越SplaTAM约25%。

### TUM-RGBD跟踪精度 (ATE RMSE [cm]↓)

| 方法 | fr1/desk | fr2/xyz | fr3/office | Avg. |
|------|----------|---------|-----------|------|
| Co-SLAM | 2.7 | 1.9 | 2.6 | 8.38 |
| SplaTAM | 3.35 | 1.24 | 5.16 | 5.48 |
| **CG-SLAM** | **2.43** | **1.20** | **2.45** | **4.0** |

### Replica建图精度

| 方法 | Acc.[cm]↓ | Comp.[cm]↓ | Comp.Ratio↑ |
|------|-----------|------------|-------------|
| Point-SLAM | 1.26 | 3.00 | 88.73% |
| Co-SLAM | 2.10 | **2.08** | **93.44%** |
| **CG-SLAM** | **1.01** | 2.84 | 88.51% |

建图精度最优(1.01cm), 完整度略逊于Co-SLAM(无全局MLP的空洞填充能力)。

### 运行效率

| 方法 | 跟踪[ms x it] | 系统FPS↑ |
|------|--------------|----------|
| NICE-SLAM | 6.19x10 | 0.98 |
| Co-SLAM | 4.45x10 | 14.2 |
| SplaTAM | 41.7x40 | 0.21 |
| **CG-SLAM** | **7.89x15** | **8.5** |
| **CG-SLAM-light** | **3.80x15** | **15.4** |

轻量版15.4 FPS,比SplaTAM快70倍以上。

### 消融实验

**各向同性损失**: 无此项部分场景直接失败(off-3), rm-2误差翻倍。

**对齐与方差损失**:

| 设置 | ATE↓ | Chamfer距离↓ |
|------|------|-------------|
| w/o 对齐+方差 | 0.33 | 4.79 |
| **Full** | **0.26** | **3.85** |

两个损失协同将Chamfer距离降低20%。

### 关键发现

1. 不确定性模型有效消除极端跟踪误差,降低跟踪方差
2. 各向同性正则化对跟踪稳定性至关重要——无此项部分场景失败
3. 对齐损失是建立一致高斯场的核心
4. Lie代数表示优于其他位姿参数化方式

## 亮点与洞察

1. **效率质量兼得**: 首次在3DGS-SLAM中同时实现SOTA精度和接近实时速度
2. **不确定性双层建模**: 像素级(方差图)和原语级(主导像素偏差)互补
3. **深度对齐策略**: 解决高斯原语位置不约束的根本问题
4. **NetVLAD共可见性**: 替代传统视锥重叠,更高效
5. **完整位姿导数推导**: 首次给出EWA溅射中位姿导数的完整理论

## 局限与展望

1. **内存消耗**: 231.66MB vs Co-SLAM的6.37MB
2. **空洞填充弱**: 无全局MLP,未观测区域完整度略差
3. **无闭环检测**: 大场景累积误差仍是问题
4. **仅限RGB-D**: 无法直接用于单目设置
5. 未来: 自适应高斯管理减少内存、扩展到纯RGB

## 相关工作与启发

- 继承NICE-SLAM/Co-SLAM/Point-SLAM的NeRF-SLAM范式,用3DGS突破效率瓶颈
- SplaTAM和GS-SLAM是同期工作但缺乏一致性设计
- 不确定性建模首次适配到高斯场SLAM

## 评分

- 新颖性: ⭐⭐⭐⭐ — 不确定性建模和一致性高斯场设计有深度
- 实用性: ⭐⭐⭐⭐⭐ — 15Hz实时、精度SOTA
- 实验充分度: ⭐⭐⭐⭐⭐ — 3个数据集,6+方法对比,完整消融
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰,数学严谨,实验全面

<!-- RELATED:START -->

## 相关论文

- [SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)
- [VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](../../CVPR2025/3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [Deep Patch Visual SLAM](deep_patch_visual_slam.md)
- [Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM](hyperion_-_a_fast_versatile_symbolic_gaussian_belief_propagation_framework_for_c.md)

<!-- RELATED:END -->
