---
title: >-
  [论文解读] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][物理仿真] 提出PhysSplat，首次利用多模态大语言模型(MLLM)零样本估计3D场景中物体的物理属性，结合物理-几何自适应采样策略在单GPU上2分钟内实现逼真的物理仿真。
tags:
  - ICCV 2025
  - 3D视觉
  - 物理仿真
  - 3D高斯
  - MLLM
  - 物理属性估计
  - MPM
---

# PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2411.12789](https://arxiv.org/abs/2411.12789)  
**代码**: [项目页](https://github.com/PhysSplat)  
**领域**: 3D视觉  
**关键词**: 物理仿真, 3D高斯, MLLM, 物理属性估计, MPM

## 一句话总结

提出PhysSplat，首次利用多模态大语言模型(MLLM)零样本估计3D场景中物体的物理属性，结合物理-几何自适应采样策略在单GPU上2分钟内实现逼真的物理仿真。

## 研究背景与动机

赋予静态3D物体交互动力学仍然困难：

**手动设置参数**：PhysGaussian等需要手动指定物理属性

**视频扩散模型开销大**：PhysDreamer等通过视频生成模型估计Young's模量，计算昂贵(1.5小时+)，且可控性差

**仅限非刚性物体**：视频扩散方法无法处理刚性物体（桌椅等）

核心思路：人类擅长从视觉信息推断物体物理属性。因此利用MLLM模拟人类的视觉物理推理。

## 方法详解

### MLLM-based Physical Property Perception (MLLM-P3)

零样本估计物理属性的流程：
1. 渲染3D场景中的物体图像
2. 用VQA模型（BLIP）生成文本描述
3. 将图像+描述输入MLLM（GPT-4V），返回K个候选材料
4. 用CLIP相似度选择最匹配材料
5. MLLM返回物理属性 $M = \{\rho, E, \nu\}$（密度、杨氏模量、泊松比）

### Material Property Distribution Prediction (MPDP)

即使单一材料物体，不同区域物理属性也有内在变化。将问题从回归转为概率分布估计：

$$\mathcal{P} = \mathcal{D}_\theta(\mathcal{X})$$

网络以物体点云和MLLM预测的均值作为输入，预测几何感知的物理属性分布，然后乘以全局均值得最终逐点属性。

### Physical-Geometric Adaptive Sampling (PGAS)

自适应调整采样半径：

$$K = \frac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}$$

$$\hat{r} = \min(r, k\sqrt{\frac{E}{\hat{K}}} r)$$

更软的物体（小$E$）和高曲率区域使用更小半径、更多驱动粒子。

### MPM仿真

基于MLS-MPM模拟器，高斯核的时间依赖状态：
$$x_i(t) = \Delta(x_i, t), \quad \Sigma_i(t) = F_i(t)\Sigma_i F_i(t)^T$$

仅模拟PGAS采样的驱动粒子，其余高斯通过局部刚体变换拟合推导。

## 实验

### 定量对比

| 方法 | RS↑ | AS↑ | 时间 |
|------|-----|-----|------|
| PhysGaussian | 4.50 | 7.56 | - |
| PhysDreamer | 4.54 | 7.71 | - |
| Physics3D | 4.62 | 7.83 | 1.5h |
| DreamGaussian4D | 4.57 | 7.28 | 0.1h |
| **PhysSplat** | **4.66** | **7.89** | **2min** |

PhysSplat在真实感评分上最高，推理时间仅2分钟，比Physics3D快45倍。

### 合成数据集

| 方法 | RS↑ | AS↑ | 时间 |
|------|-----|-----|------|
| Physics3D | 5.10 | 8.01 | 1.5h |
| DreamPhysics | 5.05 | 7.92 | 1.5h |
| **PhysSplat** | **5.10** | **8.20** | **2min** |

在合成数据集上同样取得最佳或可比性能。

## 亮点与洞察

1. **首次用MLLM零样本估计物理属性**：完全绕过了昂贵的视频扩散优化
2. **问题重构巧妙**：从回归转为分布估计，MPDP仅需Physics3D约2%的计算量
3. **全场景仿真**：唯一支持整个场景仿真的方法（Table 1）
4. **PGAS设计合理**：自适应采样兼顾软物体细节和计算效率

## 局限性

- MLLM的物理属性估计可能存在幻觉
- MPDP依赖Physics3D的伪标签进行训练
- 对复杂材料组合物体的属性估计精度有限
- 硬性碰撞和断裂等极端物理现象未建模

## 相关工作

- PhysGaussian: 物理属性注入3DGS
- PhysDreamer, Physics3D: 视频扩散学习物理参数
- MPM: 物质点方法模拟器

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (MLLM做物理属性估计的新范式)
- 技术深度: ⭐⭐⭐⭐ (MLLM-P3+MPDP+PGAS完整流水线)
- 实验充分度: ⭐⭐⭐⭐ (多数据集+用户研究)
- 实用价值: ⭐⭐⭐⭐⭐ (2分钟推理极具实用性)

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update](2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)
- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [\[CVPR 2026\] Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation](../../CVPR2026/3d_vision/let_it_snow_animating_3d_gaussian_scenes_with_dynamic_weather_effects_via_physic.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)

<!-- RELATED:END -->
