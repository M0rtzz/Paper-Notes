---
title: >-
  [论文解读] TT-Occ: Test-Time 3D Occupancy Prediction
description: >-
  [CVPR2026][自动驾驶][3D occupancy prediction] 提出 TT-Occ，一种无需预训练的测试时3D占用预测框架，通过在推理时集成视觉基础模型（VFMs）来增量构建、优化和体素化时间感知的3D高斯，在 Occ3D-nuScenes 和 nuCraft 上超越了所有需要大量训练的自监督方法。
tags:
  - CVPR2026
  - 自动驾驶
  - 3D occupancy prediction
  - test-time
  - 3D Gaussian Splatting
  - vision foundation models
  - 自监督学习
  - open-vocabulary
---

# TT-Occ: Test-Time 3D Occupancy Prediction

**会议**: CVPR2026  
**arXiv**: [2503.08485](https://arxiv.org/abs/2503.08485)  
**代码**: [Xian-Bei/TT-Occ](https://github.com/Xian-Bei/TT-Occ)  
**领域**: 自动驾驶 / 3D占用预测  
**关键词**: 3D occupancy prediction, test-time, 3D Gaussian Splatting, vision foundation models, self-supervised, open-vocabulary

## 一句话总结

提出 TT-Occ，一种无需预训练的测试时3D占用预测框架，通过在推理时集成视觉基础模型（VFMs）来增量构建、优化和体素化时间感知的3D高斯，在 Occ3D-nuScenes 和 nuCraft 上超越了所有需要大量训练的自监督方法。

## 研究背景与动机

**3D占用预测的重要性**：3D占用预测需要准确识别环境中被特定类别物体占据的区域和空闲区域，这对自动驾驶的无碰撞轨迹规划和可靠导航至关重要。

**监督方法标注代价高**：现有全监督方案严重依赖逐帧密集3D标注，在动态驾驶场景中标注成本极高（每帧覆盖80m范围）。

**自监督方法训练开销大**：虽然自监督方法减少了标注成本，但训练开销仍然巨大——例如 SelfOcc 在 Occ3D-nuScenes 0.4m分辨率下需要8卡训练2天（约384 GPU小时）。

**泛化性差**：一旦训练完成，要适配更细分辨率（如0.2m的nuCraft）或新物体类别，都需要大量重新训练，灵活性不足。

**VFMs的崛起改变格局**：VGGT、MapAnything 等3D视觉基础模型提供可靠的多视角几何，REX-Omni 支持开放词汇语义推理，这些能力可直接在测试时获得，无需任务特定训练。

**核心问题**：既然几何和语义信息不再需要网络学习获得，那么训练占用预测模型是否仍然必要？本文通过 TT-Occ 给出了否定回答。

## 方法详解

### 整体框架：Lift-Track-Voxelize

TT-Occ 遵循"提升-跟踪-体素化"三步流水线，提供 LiDAR 版本（TT-OccLiDAR）和纯视觉版本（TT-OccCamera）两个变体。

#### Step 1: Lift — 将几何和语义提升为3D高斯

- **模态特定初始化**：
    - **TT-OccLiDAR**：直接将稀疏LiDAR点初始化为3D高斯，继承真实世界测量的精确空间位置
    - **TT-OccCamera**：使用3D视觉基础模型（VGGT/MapAnything）从多视角RGB输入估计稠密深度图，通过多视角三角化解决尺度模糊问题
- **VFM语义**：使用开放词汇分割模型（OpenSeeD/GroundingSAM2/REX-Omni）提取M个环视图的语义图，通过可见性加权投影融合到3D：$\mathbf{m}_i = \frac{1}{M}\sum_{m=1}^{M}\mathbb{I}_m(\boldsymbol{\mu}_i)\mathcal{M}_m(\text{Proj}(\boldsymbol{\mu}_i;\mathbf{K}_m,\mathbf{E}_m))$
- **体素感知简化**：用sigmoid（而非指数）约束尺度参数防止过度增长，剪枝同一体素内的冗余高斯并合并语义概率

#### Step 2: Track — 跟踪动态高斯

核心问题：快速运动物体（车辆、行人）通常只被部分观测，在线优化3D高斯会产生严重的拖尾伪影。

- **TT-OccLiDAR**：无学习场景流估计——将LiDAR点投射到分割掩码关联实例 → DBSCAN去噪 → 基于空间/形状相似性跨帧匹配聚类 → ICP估计3D运动流
- **TT-OccCamera**：用RAFT估计光流，计算自运动流后相减得到残差动态流 → 阈值化得到动态掩码标识运动区域 → 对应3D高斯被排除出静态累积（妥协策略，避免3D反投影放大噪声）

#### Step 3: Voxelize — 高斯体素化

- **可微分优化**：通过颜色一致性损失在测试时精炼高斯参数
- **三边径向基函数（TRBF）平滑**：可选降噪模块，联合空间、颜色、语义亲和度对高斯参数进行周期性平滑：$\mathcal{K}(i,j) = \mathcal{K}_{\boldsymbol{\mu}}(i,j)\cdot\mathcal{K}_{\mathbf{c}}(i,j)\cdot\mathcal{K}_{\mathbf{m}}(i,j)$
- **灵活分辨率体素化**：语义概率按空间邻近度加权聚合到离散占用网格，支持任意用户指定分辨率

### 损失函数

- 颜色一致性损失（通过可微渲染将3D高斯投射回图像平面）
- 天空区域掩码排除

## 实验

### 主实验结果

**Occ3D-nuScenes（0.4m分辨率）**：

| 方法 | 输入 | 预训练 | mIoU |
|---|---|---|---|
| SelfOcc (CVPR'24) | C | ~384 GPU hrs | 10.54 |
| GaussianTR (CVPR'25) | C | 有 | 11.70 |
| VEON-LiDAR (ECCV'24) | C&L | 有 | 15.14 |
| **TT-OccCamera** | **C** | **无** | **16.70** |
| RenderOcc (ICRA'24) | C | 有(稀疏3D GT) | 23.93 |
| **TT-OccLiDAR** | **C&L** | **无** | **27.41** |
| BEVFormer (ECCV'22) | C | 有(密集3D GT) | 26.88 |

**nuCraft 高分辨率（0.2m分辨率）**：

| 方法 | 预训练时间 | mIoU |
|---|---|---|
| SelfOcc† | 384 hrs | 2.22 |
| TT-OccCamera | 0 | 5.95 |
| TT-OccLiDAR | 0 | 10.92 |

### 消融实验

| 配置 | TT-OccLiDAR mIoU | TT-OccCamera mIoU |
|---|---|---|
| A: 基线（单帧直接散射） | 7.3 | 4.2 |
| B: + 协方差感知体素化 | 18.3 (+11.0) | 8.5 (+4.3) |
| C: + 继承历史高斯（无跟踪） | 23.5 (+5.2) | 14.1 (+5.6) |
| D: + 动态高斯跟踪 | 25.6 (+2.1) | 14.1 (+0.0) |

### 关键发现

1. **零训练超越训练方法**：TT-OccLiDAR (27.41 mIoU) 甚至超过使用稀疏3D GT训练的 RenderOcc (23.93)，TT-OccCamera (16.70) 超过使用LiDAR监督训练的 VEON-LiDAR (15.14)
2. **分辨率适应性强**：在nuCraft高分辨率设置下，SelfOcc从10.54骤降至2.22，而TT-Occ无需重训即可适配
3. **RayIoU验证几何质量**：TT-OccCamera在RayIoU@4上比SelfOcc提升30.8%，TT-OccLiDAR提升115%
4. **模块化VFM设计**：REX-Omni语义最强，MapAnything深度优于VGGT（因提供度量尺度深度）；框架可即插即用最新VFM
5. **动态跟踪消除拖尾**：不跟踪时动态类（bus, ped）IoU下降严重，跟踪后显著恢复
6. **内存效率**：峰值GPU内存 LiDAR版5.6GB、Camera版9.9GB，均低于10GB

## 亮点

- **范式创新**：首次证明测试时集成VFMs可以完全替代训练密集占用解码器，将3D占用预测从"训练范式"转向"推理范式"
- **极强灵活性**：支持任意体素分辨率、开放词汇语义查询、即插即用替换VFM组件
- **零训练成本**：完全免去数百GPU小时的预训练，直接在验证集推理
- **时间感知高斯**：通过 Lift-Track-Voxelize 流水线实现在线递增式场景重建，动静分离消除拖尾伪影
- **TRBF平滑**：借鉴双边滤波思想扩展为三边核，联合空间-颜色-语义进行自适应降噪

## 局限性

- **依赖VFM质量**：性能上界受限于所使用VFM的能力，GroundingSAM2语义较弱时mIoU降至21.3
- **Camera版动态处理受限**：纯视觉版无法像LiDAR版那样累积动态物体，只能排除动态区域
- **推理速度**：语义分割（OpenSeeD）占整体运行时间的28.5%~77.9%，Camera版还需额外深度估计、三角化等步骤
- **远距离区域Camera版不佳**：因遮挡和深度分辨率限制，纯视觉版在远距离场景的几何精度不如LiDAR版
- **开放词汇依赖提示质量**：VFM的语义分割依赖文本提示（prompt），在标准benchmark上使用预定义label set但实际开放场景中prompt设计影响效果

## 相关工作

- **全监督占用预测**：BEVFormer、CTF-Occ、RenderOcc — 依赖密集/稀疏3D标注
- **自监督占用预测**：SelfOcc（SDF+多视角立体）、OccNeRF（光度一致性）、GaussianOcc/GaussianTR（3DGS表示）、LangOcc/VEON（开放词汇）
- **驾驶场景3D重建**：OmniRe、Street Gaussians、DrivingGaussian、HUGS — 依赖外部先验（HD地图、GT边界框）做离线逐场景重建，而TT-Occ仅用原始传感器流在线推理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个无需任何训练的测试时3D占用预测框架，范式性创新
- 实验充分度: ⭐⭐⭐⭐ — 两个数据集、两种模态变体、多VFM组合消融、RayIoU评估，较为全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，Lift-Track-Voxelize框架直观易懂
- 价值: ⭐⭐⭐⭐⭐ — 指出VFM时代训练占用模型的必要性可能不再，对自动驾驶感知范式有深远影响

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2occ_resilient_3d_semantic_occupancy_prediction_f.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_me.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](../../ICCV2025/autonomous_driving/sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth-_and_region-guided_3d_occupancy_from_surround-view_cameras_for_auton.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)

<!-- RELATED:END -->
