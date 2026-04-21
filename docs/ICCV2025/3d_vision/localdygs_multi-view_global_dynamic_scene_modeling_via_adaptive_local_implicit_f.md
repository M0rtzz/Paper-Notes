---
title: >-
  [论文解读] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 提出 LocalDyGS——将全局复杂动态场景分解为种子点定义的局部空间、并通过静态-动态特征解耦生成时序高斯来建模各局部运动的框架，首次实现了大尺度复杂动态场景的高质量重建。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯溅射
  - 动态场景重建
  - 多视角
  - 局部空间建模
  - 静态-动态解耦
  - 时序高斯
---

# LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling

**会议**: ICCV 2025  
**arXiv**: [2507.02363](https://arxiv.org/abs/2507.02363)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 动态场景重建, 多视角, 局部空间建模, 静态-动态解耦, 时序高斯

## 一句话总结

提出 LocalDyGS——将全局复杂动态场景分解为种子点定义的局部空间、并通过静态-动态特征解耦生成时序高斯来建模各局部运动的框架，首次实现了大尺度复杂动态场景的高质量重建。

## 研究背景与动机

多视角动态场景重建是 3D 视觉的关键问题，应用于自由视点体育直播、AR/VR、游戏等。现有方法面临：

- **变形场方法**（4DGaussian）：从 canonical 映射到变形场，难处理大尺度运动
- **轨迹跟踪方法**（SpaceTimeGS）：多项式/傅里叶级数表示轨迹，大规模运动中模糊闪烁
- **在线流式方法**（3DGStream）：逐帧建模，大范围运动能力有限
- **存储问题**：3DGStream 300帧需 1230MB，RealTimeGS 超 1000MB

核心问题：显式跟踪每个高斯点的长期运动在大尺度复杂运动下不可行。

核心思路：**将全局动态场景分解为局部空间，每个种子点独立建模短程运动**。移动物体被多个种子表示——种子在物体经过时激活，离开后失活。

## 方法详解

### 整体框架

两大组件：
1. 全局→局部空间分解：融合多帧 SfM 点云初始化种子
2. 局部空间时序高斯生成：静态-动态解耦 + MLP 解码

### 关键设计

**1. 全局种子初始化**

采样 N 帧提取并融合 SfM 点云初始化种子位置 μ。每个种子含：
- 位置 μ（全局参数）
- 静态特征 f_s ∈ R^64（跨时间步共享，初始化为0）
- 局部空间尺度 v ∈ R^3（初始化为3个最近邻种子平均距离）

**2. 特征解耦的时空场**

- **静态特征** f_s：每个局部空间独立优化，承载大部分场景信息
- **动态残差场** F_d：多分辨率 4D 哈希编码(位置+时间) + 浅层 MLP
    - 哈希表 2^17，L个分辨率级拼接后MLP融合
    - 输入 (μ, t) ∈ R^4
- **权重场** F_w：预测 w_s, w_d 平衡静态和动态
- 加权特征：f_w = w_s · f_s + w_d · f_d
- 关键发现：动态残差值趋近零，大部分信息由静态特征承载

**3. 时序高斯生成**

每种子生成 k=10 个时序高斯：
- 位置：μ_t = μ + v · F_μ(f_w)
- 不透明度：Sigmoid(F_o(f_w, d))，d为相机方向
- 旋转、缩放、颜色通过独立 MLP 解码
- **失活机制**：σ < τ_α=0.01 时失活，减少计算量

**4. 自适应种子增长（ASG）**

- 记录时序高斯在 n 次迭代中的最大 2D 投影梯度及 3D 位置
- ∇_max > τ_g=0.001 时添加新种子
- 3000-15000迭代，每100次执行

### 损失函数

L = (1 - 0.2)·L1 + 0.2·L_SSIM + 0.001·L_v
- L_v = Σ Prod(s_t^i)：体积正则化，鼓励小尺寸保持局部性
- Adam 优化器，30000 次迭代

## 实验关键数据

### N3DV 数据集（21相机，细粒度运动）

| 方法 | PSNR↑ | LPIPS↓ | FPS↑ | 时间↓ | 存储↓ |
|------|-------|--------|------|-------|-------|
| 4DGaussian | 31.02 | 0.150 | 30 | 0.67h | 90MB |
| SpaceTimeGS | 32.05 | 0.044 | 140 | >5h | 200MB |
| 3DGStream | 31.67 | - | 215 | 1.0h | 1230MB |
| RealTimeGS | 32.01 | 0.055 | 114 | 9.0h | >1000MB |
| **LocalDyGS** | **32.28** | **0.043** | 105 | **0.58h** | **100MB** |

### MeetRoom 数据集（13相机，稀疏视角）

| 方法 | PSNR↑ | 时间↓ | 存储↓ |
|------|-------|-------|-------|
| 3DGS(逐帧) | 31.31 | 13h | 6330MB |
| 3DGStream | 30.79 | 0.6h | 1230MB |
| **LocalDyGS** | **32.45** | **0.36h** | **90MB** |

### VRU 篮球场（34相机，大尺度复杂运动）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 4DGS | 28.32 | 0.930 | 0.186 |
| SpaceTimeGS | 27.42 | 0.926 | 0.193 |
| **LocalDyGS** | **30.58** | **0.944** | **0.173** |

超越动态方法 2dB 以上，接近静态 3DGS 上界。

### 消融实验

| 组件 | 效果 |
|------|------|
| ASG 种子增长 | PSNR 31.81→33.02（+1.21dB）|
| 去除静态特征 | 29.46 vs 31.40（-1.94dB）|
| 去除失活 | FPS 89→105，PSNR 几乎不变 |
| N=6 vs N=30 | 性能几乎相同（32.28 vs 32.30）|
| N=1 | 31.84（-0.44dB），覆盖不足 |
| k=5/10/20 | k=10 最优平衡 |

### 关键发现

- 首次成功处理 VRU 篮球场等大尺度动态数据集
- 静态-动态解耦贡献最大（-1.94dB if removed）
- 仅 6 帧 SfM 即可达最优，对初始点云不敏感
- 35 分钟训练（单卡 RTX 3090）
- 100MB/300帧 vs 3DGStream 1230MB

## 亮点与洞察

1. **局部分解核心思想**：将全局长程跟踪简化为局部短程建模，大尺度动态场景的关键突破
2. **静态-动态解耦有效性**：动态残差趋近零，解耦大幅降低建模难度
3. **极致效率**：最优质量+35分钟训练+100MB存储
4. **首次大尺度动态场景**：VRU篮球场验证实用性
5. **ScaffoldGS 巧妙动态扩展**：继承锚点结构，加入时间维度
6. **失活机制类比 MoE**：稀疏激活减少计算量

## 局限性

- 依赖多帧 SfM 初始化，快速运动/无纹理区域可能覆盖不足
- 渲染 105 FPS 低于 3DGStream 215 FPS
- 假设相机参数已知且标定准确
- 体积正则可能限制大尺度物体表示
- 4D 哈希编码内存随分辨率增长

## 相关工作

- **静态合成**：3DGS 实时渲染；ScaffoldGS 锚点表示
- **变形场方法**：4DGaussian canonical→deformation映射
- **轨迹跟踪**：SpaceTimeGS 多项式控制
- **流式方法**：3DGStream 逐帧在线重建
- **4DGS 扩展**：4D空间拟合高斯，高存储高计算

## 评分

- **新颖性**: ★★★★☆ — 局部空间分解+特征解耦是动态场景建模新思路
- **技术深度**: ★★★★★ — 三个不同运动尺度数据集，消融极详尽
- **实验充分度**: ★★★★★ — 定量定性全面，首次大尺度验证
- **实用性**: ★★★★★ — 训练快/存储小/质量高/首次大尺度动态
- **总分**: 9.0/10

<!-- RELATED:START -->

## 相关论文

- [Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling](atlas_decoupling_skeletal_and_shape_parameters_for_expressiv.md)
- [CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](../../CVPR2025/3d_vision/global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)

<!-- RELATED:END -->
