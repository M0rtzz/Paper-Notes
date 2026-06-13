---
title: >-
  [论文解读] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction
description: >-
  [3D视觉] S3R-GS 通过识别传统街景重建管线中的三大计算冗余（不必要的局部-全局坐标变换、过多的3D-2D投影、低效的远距离内容渲染），提出实例特定投影、时序可见性过滤和自适应LOD策略，将重建时间降至竞争方法的20%-50%，同时保持SOTA渲染质量。 大规模街景重建在自动驾驶等领域至关重要…
tags:
  - "3D视觉"
---

# S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.08217](https://arxiv.org/abs/2503.08217)
- **领域**: 3D视觉
- **关键词**: 3D Gaussian Splatting, 大规模街景重建, 自动驾驶, 动态场景, 计算效率

## 一句话总结
S3R-GS 通过识别传统街景重建管线中的三大计算冗余（不必要的局部-全局坐标变换、过多的3D-2D投影、低效的远距离内容渲染），提出实例特定投影、时序可见性过滤和自适应LOD策略，将重建时间降至竞争方法的20%-50%，同时保持SOTA渲染质量。

## 研究背景与动机

大规模街景重建在自动驾驶等领域至关重要，但现有 3DGS 方法应用于大规模场景时面临关键问题：**场景规模增大时，每视角重建成本急速攀升**。

分析传统管线后，作者识别出三大计算冗余：

**不必要的局部-全局变换**：每帧渲染时需要将每个动态目体的 Gaussians 从局部坐标系变换到全局坐标系，包含大量冗余矩阵乘法

**过量的3D-2D投影**：所有 Gaussians 都被投影到图像平面，但绝大多数在当前视锥外，投影计算完全浪费

**低效的远距离内容渲染**：对所有可见 Gaussians 一视同仁地执行 α-blending，远距离的细小 Gaussians 对画质贡献微小却占用大量计算

此外，现有方法依赖 **3D 真值边界框** 分离动态/静态目标，但 3D 标注获取困难，限制了实际应用。

## 方法详解

### 整体框架

S3R-GS 分为场景建模和场景重建两阶段，重点优化重建管线。

### 关键设计一：实例特定投影（Instance-Specific Projection）

传统方法先将动态目标 Gaussians 从局部变换到全局，再统一投影到相机平面。S3R-GS 跳过全局变换，直接为每个目标预计算实例特定相机参数：

$$W_{t,i} = W_t \cdot W_{t,i2g}$$

渲染时根据 Gaussian 的实例 ID 选择对应相机即可，一次矩阵乘法替代两次变换。

### 关键设计二：时序可见性分离（Temporal Separation）

为每个 Gaussian 分配时序可见性 $v = (v_s, v_e)$ 和生命周期 $l = (l_s, l_e)$。渲染时间 $t$ 时仅选择满足 $t_s \leq t \leq t_e$ 的 Gaussians 进行投影，大幅减少无效的 3D-2D 投影。

可见性通过渲染后的实际可见掩码 $M_t$ 动态更新：

$$l_{s,i} = \min(l_{s,i}, t),\quad l_{e,i} = \max(l_{e,i}, t)$$

$$t_s = l_s - 0.1,\quad t_e = l_e + 0.1$$

### 关键设计三：自适应LOD（Adaptive Level-of-Detail）

对投影后2D 尺度低于阈值 $r$ 的远距离 Gaussians：

1. **概率性剔除**：根据深度概率性丢弃，距离越远丢弃概率越高

$$p = p_{max} + (p_{max} - 10^{-2}) \cdot \min(0, \frac{d-D}{D})$$

2. **噪声偏移**：对保留的远距离 Gaussians 添加与深度相关的位置噪声，获得周围平均颜色
3. **距离感知神经场**：查询颜色时引入深度作为输入，使网络自动学习不同 LOD 的颜色变化

### 2D框场景分解

替代 3D 框的方案：利用 2D 框 + SAM 获取目标掩码，投影 LiDAR 点云获得粗略 3D 轨迹 $TXYZ \in \mathbb{R}^{T \times 3}$。引入 NeuralODE 学习连续运动轨迹：

$$\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, c)$$

其中 $c$ 为实例嵌入，$\mathbf{z}(t) = [\Delta XYZ_t + XYZ_t, R_t]$，实现平滑的姿态估计。

### BEV 语义初始化增强

针对 LiDAR 无法覆盖的高层建筑，在 BEV 网格中沿 z 轴补充初始化点，增强场景完整性。

## 实验

### 主实验：Argoverse 2 大规模街景

| 方法 | 平均 PSNR↑ | 平均 SSIM↑ | 平均 LPIPS↓ | 重建时间↓ |
|------|----------|----------|----------|----------|
| SUDS | 20.84 | 0.662 | 0.601 | - |
| ML-NSG | 21.15 | 0.680 | 0.555 | 49.10h |
| 4DGF | 24.97 | 0.772 | 0.447 | 54.39h |
| **S3R-GS** | **25.68** | **0.780** | **0.435** | **26.71h** |

重建质量超越所有方法的同时，时间降至 4DGF 的不到一半。

### 消融实验：各组件贡献（KITTI）

| 组件 | PSNR | 训练时间 |
|------|------|---------|
| Baseline (4DGF) | 基准 | 基准 |
| + 实例特定投影 | ≈ | 显著↓ |
| + 时序可见性 | ≈ | 进一步↓ |
| + 自适应 LOD | 略↑ | 进一步↓ |
| + 2D 分解 | 略↓ | ≈ |

实例特定投影和时序可见性是加速的主要贡献者，自适应 LOD 在保质的同时进一步加速。

### 关键发现
- 在长序列场景（KITTI 全长）中，加速效果更为显著，S3R-GS 仅需竞争方法约 20% 的时间
- 2D 框分解虽然精度略低于 3D 框，但大幅提升了方法的实际可用性
- BEV 语义增强有效改善了高层建筑的重建质量

## 亮点与洞察

1. **系统性分析**：不是提新模块，而是深入审视管线中的每一步计算冗余
2. **线性可扩展**：优化后每视角重建成本不随场景规模剧烈增长
3. **实用导向**：2D 框替代 3D 框的设计使方法能应用于 in-the-wild 场景
4. **即插即用**：各优化策略相互独立，可单独应用于其他街景 3DGS 方法

## 局限性
- 2D 分解 + NeuralODE 在极高速运动或频繁遮挡时可能不够鲁棒
- 自适应 LOD 的概率性剔除可能导致远距离区域的渲染不稳定
- 需要 LiDAR 点云进行初始化，无法完全无传感器运行

## 相关工作
- DrivingGaussian / StreetGaussian / HUGS / 4DGF：3DGS 街景重建方法
- EmerNeRF / NSG：NeRF 街景重建方法
- NeuralODE：连续时间建模框架

## 评分
- **创新性**: ⭐⭐⭐⭐ — 系统性优化管线而非堆叠模块
- **实用性**: ⭐⭐⭐⭐⭐ — 显著加速且提升质量，2D框降低标注门槛
- **实验完整度**: ⭐⭐⭐⭐ — 三个数据集全面验证，有消融
- **写作质量**: ⭐⭐⭐⭐ — 问题分析到位，方案简洁有效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](../../ICLR2026/3d_vision/urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

</div>

<!-- RELATED:END -->
