---
title: >-
  [论文解读] Neural Inverse Rendering from Propagating Light
description: >-
  [CVPR 2025][自动驾驶][时间分辨逆渲染] 首个从多视角时间分辨 LiDAR 测量（飞行时间光子探测）中进行物理逆渲染的方法，通过时间分辨辐射缓存替代递归路径追踪来建模直接和间接光传输，在合成场景上法线 MAE 从 FWP++ 的 22.80° 降至 8.45°，同时支持新视角合成与重光照。
tags:
  - CVPR 2025
  - 自动驾驶
  - 时间分辨逆渲染
  - LiDAR
  - 间接光照
  - 辐射缓存
  - 物理渲染
---

# Neural Inverse Rendering from Propagating Light

**会议**: CVPR 2025  
**arXiv**: [2506.05347](https://arxiv.org/abs/2506.05347)  
**代码**: https://anaghmalik.com/InvProp  
**领域**: 自动驾驶  
**关键词**: 时间分辨逆渲染、LiDAR、间接光照、辐射缓存、物理渲染

## 一句话总结

首个从多视角时间分辨 LiDAR 测量（飞行时间光子探测）中进行物理逆渲染的方法，通过时间分辨辐射缓存替代递归路径追踪来建模直接和间接光传输，在合成场景上法线 MAE 从 FWP++ 的 22.80° 降至 8.45°，同时支持新视角合成与重光照。

## 研究背景与动机

1. **领域现状**：传统逆渲染方法从 RGB 图像恢复几何和材质，但无法有效处理强间接光照（如室内多次反射）。时间分辨 LiDAR（SPAD 探测器）可以捕捉光子飞行时间，提供额外的时间维度约束。
2. **现有痛点**：(1) T-NeRF 仅建模直射光，在间接光照强的场景中严重失真；(2) FWP++ 处理间接光但非物理模型，几何重建精度受限；(3) 递归路径追踪计算量太大，不适合嵌入神经网络优化循环。
3. **核心矛盾**：精确建模间接光需要求解完整渲染方程（递归求解），但递归不可微或计算量爆炸。
4. **本文目标**：用神经辐射缓存替代递归路径追踪，实现可微的物理逆渲染。
5. **切入角度**：时间分辨数据的光子飞行时间提供了光路长度约束——不仅知道有多少光到达探测器，还知道光走了多长路径（区分直射和间射）。
6. **核心 idea**：直接/间接光分解 + 神经辐射缓存（hash 编码）+ split-sum 近似处理间接光 BRDF 积分。

## 方法详解

### 整体框架

多视角时间分辨 LiDAR 测量 → 神经几何网络（密度 $\sigma$ + 法线 $n$）→ 外观特征 hash 编码 → 直接光缓存（解析 Fresnel + 光源可见性）+ 间接光缓存（split-sum 近似）→ Disney-GGX BRDF → 体渲染与 transient 信号对比优化。

### 关键设计

1. **时间分辨辐射缓存**

    - 功能：替代递归路径追踪，高效建模间接光
    - 核心思路：将入射辐射分为直接 $L_o^{cache,dir}$ 和间接 $L_o^{cache,indir}$ 两项。直接光用解析 BRDF + 光源位置计算。间接光用 split-sum 近似：$L_o^{indir} = f_\Omega^{indir}(f^{app}, n, \omega_o) \cdot L_{i,\Omega}^{indir}(f^{app}, x_\ell, n, \omega_o)$，两项分别由 MLP 近似
    - 设计动机：辐射缓存避免了递归求解——将入射辐射存储为空间-方向-时间的连续函数，查表即可

2. **直接/间接光分解**

    - 功能：利用 BRDF 物理模型精确建模直射光
    - 核心思路：直接光用完整 Disney-GGX BRDF 计算：$L_o^{dir} = f^{dir}(f^{app}, n, \omega_\ell, \omega_o) L_i^{dir}(x', \omega_\ell, \tau)(n \cdot \omega_\ell)$，间接光因积分不可解而用 split-sum 近似
    - 设计动机：直射光有解析形式（光源位置已知），精确建模比近似更好；间接光积分复杂只能近似

3. **多分辨率 Hash 编码**

    - 功能：高效表示空间变化的外观特征
    - 核心思路：$f^{app} = \mathcal{H}^{app}(x)$，多层级 hash 编码捕捉不同尺度的材质变化
    - 设计动机：hash 编码在 Instant-NGP 中已证明对空间特征的高效性

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{data} + \lambda_{cache} \mathcal{L}_{cache} + \lambda_{dir} \mathcal{L}_{dir} + \lambda_{indir} \mathcal{L}_{indir} + \text{regularizers}$。法线平滑、深度畸变、mask 正则化。

## 实验关键数据

### 主实验

| 方法 | 合成 PSNR↑ | 合成法线 MAE↓ | 合成深度 L1↓ |
|------|-----------|-------------|------------|
| T-NeRF | 22.44 | 28.00° | 0.59 |
| FWP++ | 29.00 | 22.80° | 0.47 |
| **Ours** | **30.99** | **8.45°** | **0.21** |

### 消融实验

| 设定 | 关键观察 |
|------|---------|
| 仅直接光 | 间接光强场景严重失真 |
| w/o split-sum | 间接光建模退化 |
| w/o 时间分辨 | 丧失光路长度约束 |

### 关键发现

- 法线精度比 FWP++ 提升 3.2 倍（22.80→8.45 MAE）——物理模型的关键优势
- 在真实捕捉数据上 PSNR 略低于 FWP++（27.39 vs 28.45），可能因标定误差
- 支持重光照（将光源位置改变后重新渲染），这是 FWP++ 无法做到的

## 亮点与洞察

- **首次将时间分辨 LiDAR 与物理逆渲染结合**：光子飞行时间提供了常规 RGB 不可能获得的约束
- **辐射缓存替代路径追踪**：将不可微的递归过程转化为可微的神经网络查表，工程上非常优雅
- **支持重光照**：获得了物理材质参数（反照率、粗糙度、金属度），可以在任意光照下重新渲染

## 局限与展望

- 需要专用硬件（皮秒激光 + SPAD 探测器），不适用于消费级设备
- 真实数据标定误差直接影响结果质量
- Disney-GGX BRDF 不能建模所有材质（如透明、次表面散射）
- 重光照需要微调 direct/indirect loss，非完全自动化

## 相关工作与启发

- **vs T-NeRF**: 仅建模直射光，在间接光强的 Cornell Box 场景中完全失效
- **vs FWP++**: 非物理模型，导致几何精度差（法线 MAE 22.8°），且不支持重光照

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将时间分辨LiDAR与物理逆渲染结合
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据+多指标，但场景数有限
- 写作质量: ⭐⭐⭐⭐ 物理推导严谨
- 价值: ⭐⭐⭐⭐ 为间接光照场景的逆渲染开辟新方向

<!-- RELATED:START -->

## 相关论文

- [RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [LightLoc: Learning Outdoor LiDAR Localization at Light Speed](lightloc_learning_outdoor_lidar_localization_at_light_speed.md)
- [GaussRender: Learning 3D Occupancy with Gaussian Rendering](../../ICCV2025/autonomous_driving/gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction](../../ICML2025/autonomous_driving/goirl_graph-oriented_inverse_reinforcement_learning_for_multimodal_trajectory_pr.md)
- [Single Pixel Image Classification using an Ultrafast Digital Light Projector](single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)

<!-- RELATED:END -->
