---
title: >-
  [论文解读] DROID-W: DROID-SLAM in the Wild
description: >-
  [CVPR2026][3D视觉][SLAM] 提出 DROID-W，通过将不确定性估计引入可微分 Bundle Adjustment（Uncertainty-aware BA），结合 DINOv2 特征驱动的动态不确定性更新机制和单目深度正则化…
tags:
  - "CVPR2026"
  - "3D视觉"
  - "SLAM"
  - "dynamic scenes"
  - "uncertainty estimation"
  - "bundle adjustment"
  - "DINOv2"
---

# DROID-W: DROID-SLAM in the Wild

**会议**: CVPR2026  
**arXiv**: [2603.19076](https://arxiv.org/abs/2603.19076)  
**代码**: [MoyangLi00/DROID-W](https://github.com/MoyangLi00/DROID-W.git)  
**领域**: 3D视觉  
**关键词**: SLAM, dynamic scenes, uncertainty estimation, bundle adjustment, DINOv2

## 一句话总结

提出 DROID-W，通过将不确定性估计引入可微分 Bundle Adjustment（Uncertainty-aware BA），结合 DINOv2 特征驱动的动态不确定性更新机制和单目深度正则化，使 DROID-SLAM 在高度动态的野外（in-the-wild）场景中实现鲁棒的相机位姿估计和场景重建，约 10 FPS 实时运行。

## 背景与动机

视觉 SLAM（Simultaneous Localization and Mapping）是机器人、AR/VR 和自动驾驶的核心技术，目标是从连续视频帧中同时估计相机位姿和构建场景三维结构。

DROID-SLAM 作为当前最优的深度学习 SLAM 系统之一，其核心优势在于可微分的 Dense Bundle Adjustment (DBA) 层，通过端到端训练实现了出色的精度。然而，DROID-SLAM 和几乎所有经典 SLAM 方法都建立在一个关键假设之上：

**静态世界假设**：场景中所有可观测点在不同时间帧之间是静态的，其三维位置不随时间变化。

这一假设在真实的"野外"（in-the-wild）场景中严重不成立：

1. **行人和车辆**：城市场景中大量运动物体破坏几何一致性
2. **风吹树叶、流水**：自然环境中的非刚性运动无处不在
3. **YouTube 视频**：互联网视频充满各种动态元素，传统 SLAM 无法处理

现有应对动态场景的方法主要分为两类：

- **基于语义分割的方法**：预先检测和屏蔽"可能运动"的物体类别（如行人、车辆），但依赖预定义的动态类别先验，无法处理意外运动的物体
- **基于 neural implicit map 的方法**（如 RoDynRF、DynaMoN）：用 NeRF 联合建模静态和动态区域，精度高但计算代价极大，无法实时运行

核心动机：**能否在不依赖预定义动态先验的情况下，让 BA 自适应地降低动态区域的影响？** 作者观察到，如果能为每个像素分配一个不确定性权重——动态区域高不确定性、静态区域低不确定性——那么 BA 的优化过程自然会"忽略"动态像素的贡献。

## 核心问题

- 动态物体违反静态世界假设，导致 BA 的重投影残差在动态区域产生大量 outlier，严重干扰位姿估计
- 预定义语义先验无法覆盖所有动态类别，且"可能运动的类别"并不总是在运动
- Neural implicit 方法虽能处理动态场景，但计算成本过高，无法实时部署

## 方法详解

### 整体架构

DROID-W 在 DROID-SLAM 的基础上引入三个关键改进：

1. **Uncertainty-aware BA (UBA)**：将 per-pixel 不确定性权重融入 Bundle Adjustment 优化
2. **Dynamic Uncertainty Update**：利用 DINOv2 视觉基础模型的语义特征进行动态区域检测和不确定性分配
3. **Monocular Depth Regularization**：在 BA 中加入单目深度先验约束，增强极端动态场景下的稳定性

### Uncertainty-aware Bundle Adjustment (UBA)

标准 DROID-SLAM 的 DBA 层通过最小化加权重投影误差来联合优化相机位姿 $\{G_i\}$ 和逆深度图 $\{d_i\}$：

$$E = \sum_{(i,j) \in \mathcal{E}} \| p_{ij}^* - \Pi_c(G_{ij} \cdot \Pi_c^{-1}(p_i, d_i)) \|_{\Sigma_{ij}}^2$$

其中 $p_{ij}^*$ 是通过相关性查找得到的对应点坐标，$\Sigma_{ij}$ 是预测的置信度权重。

DROID-W 的 UBA 在此基础上引入 per-pixel 不确定性 $u_{ij}$：

$$E_{UBA} = \sum_{(i,j) \in \mathcal{E}} \frac{1}{u_{ij}} \| p_{ij}^* - \Pi_c(G_{ij} \cdot \Pi_c^{-1}(p_i, d_i)) \|_{\Sigma_{ij}}^2 + \log u_{ij}$$

关键设计：

- 不确定性 $u_{ij}$ 越大，该像素对重投影误差的贡献越小，相当于自动降权
- $\log u_{ij}$ 正则项防止不确定性无限增大（避免平凡解）
- 不确定性 $u_{ij}$ 参与 BA 的迭代优化，与位姿和深度联合更新

### Dynamic Uncertainty Update

不确定性 $u_{ij}$ 的初始化和更新依赖于 DINOv2 提取的视觉语义特征：

**Step 1 — 特征提取**：对每一帧 $I_i$ 用预训练的 DINOv2 提取 dense 特征图 $F_i \in \mathbb{R}^{H \times W \times C}$

**Step 2 — 刚性运动对应**：利用当前 BA 估计的位姿和深度，计算帧 $i$ 到帧 $j$ 的刚性重投影坐标 $p_{ij}$

**Step 3 — 特征余弦相似度**：比较帧 $i$ 中像素 $p$ 的特征 $F_i(p)$ 与帧 $j$ 中对应位置 $p_{ij}$ 的特征 $F_j(p_{ij})$：

$$s_{ij}(p) = \frac{F_i(p) \cdot F_j(p_{ij})}{\|F_i(p)\| \cdot \|F_j(p_{ij})\|}$$

**Step 4 — 不确定性分配**：

$$u_{ij}(p) = 1 - s_{ij}(p)$$

核心直觉如下：

- **静态区域**：刚性重投影后特征高度一致，$s_{ij}$ 接近 1，$u_{ij}$ 接近 0（低不确定性）
- **动态区域**：物体运动后，刚性重投影指向了错误的位置，$F_j(p_{ij})$ 与 $F_i(p)$ 不匹配，$s_{ij}$ 低，$u_{ij}$ 高（高不确定性）

DINOv2 的语义特征比原始像素值更鲁棒——即使光照变化、轻微视角变化，静态区域的特征相似度仍然很高；而动态区域由于几何不一致，特征差异会被放大。

### Monocular Depth Regularization

在极端动态场景（如场景中 80%+ 区域都在运动）中，大部分像素被标记为高不确定性，BA 可用的约束过少导致优化不稳定。解决方案是加入单目深度先验作为正则化：

$$E_{depth} = \lambda \sum_i \| d_i - d_i^{mono} \|^2$$

其中 $d_i^{mono}$ 是由预训练单目深度估计模型（如 DPT/ZoeDepth）预测的深度。注意这里使用的是 scale-和 shift-invariant 的损失形式，因为单目深度缺乏绝对尺度。

### 迭代优化流程

1. 初始化：DROID-SLAM 标准流程，用 ConvGRU 迭代更新光流和置信度
2. 每 $K$ 次 BA 迭代后，调用 Dynamic Uncertainty Update 重新计算 $u_{ij}$
3. 将更新后的 $u_{ij}$ 注入下一轮 UBA 优化
4. 重复直到收敛

## 实验关键数据

### TUM RGB-D Dynamic 序列

| 方法 | ATE RMSE (cm)↓ | 动态占比 |
|------|----------------|---------|
| ORB-SLAM3 | 36.5 | 高 |
| DROID-SLAM | 28.3 | 高 |
| DynaSLAM | 3.8 | 高 |
| **DROID-W** | **2.1** | **高** |

在高动态 TUM 序列（如 walking 系列）中 ATE 降低到 2.1cm，相比原始 DROID-SLAM 提升 13× 以上。

### DROID-W Dataset（野外数据）

作者构建了专门的评估数据集，包含多样户外动态场景（街道行人、骑行者、跑步者等）以及 YouTube 视频片段。定性评估显示：

- DROID-SLAM 在动态场景中轨迹严重漂移
- DROID-W 保持稳定的轨迹估计，相机路径与真值高度吻合

### KITTI Dynamic 场景

| 方法 | 翻译误差↓ | 旋转误差↓ |
|------|----------|----------|
| DROID-SLAM | 失败/漂移 | 失败/漂移 |
| **DROID-W** | **显著改善** | **显著改善** |

在 KITTI 中车辆密集的序列上，DROID-SLAM 频繁失败，DROID-W 持续稳定跟踪。

### 消融实验

- **去掉 UBA（仅保留标准 BA）**：ATE 大幅上升，退化为 DROID-SLAM 的表现
- **去掉 DINOv2 特征（用原始像素相似度）**：不确定性估计不够鲁棒，ATE 上升约 40%
- **去掉 Monocular Depth Regularization**：在极高动态占比场景中 BA 偶尔发散
- **不同基础模型**：DINOv2 > DINO > CLIP > ResNet50 特征，验证 DINOv2 的语义鲁棒性

### 运行效率

- 约 10 FPS 实时运行
- 相比 neural implicit 方法（如 RoDynRF 约 0.1 FPS），速度快 100×
- DINOv2 特征提取可通过缓存和降采样优化，额外开销约 15%

## 亮点

- **优雅的不确定性建模**：将动态检测问题转化为不确定性权重，无缝融入已有 BA 框架，无需修改底层优化器架构
- **不依赖预定义动态先验**：通过特征相似度自适应检测动态区域，可处理任何类型的运动物体（包括训练时未见过的类别）
- **利用视觉基础模型**：DINOv2 的强语义特征使动态检测在光照变化、纹理不足等困难条件下仍然鲁棒
- **实时性保持**：~10 FPS 的运行速度使其具备实际部署价值，远超 neural implicit 方案
- **最小化修改**：在 DROID-SLAM 基础上仅增加了不确定性模块，改动量小、通用性强

## 局限与展望

- DINOv2 模型本身的计算开销不可忽略，在嵌入式设备上部署有挑战
- 对于静态背景几乎完全被遮挡的极端场景（如车内拍摄且窗外全是运动物体），单目深度正则化的约束力有限
- 未与最新的 3D Gaussian Splatting 动态场景方法（如 DynGaussian）进行对比
- 不确定性更新频率（每 $K$ 步）是超参数，不同场景的最优值可能不同
- 仅在单目场景验证，未探索双目或 RGB-D 输入的扩展

## 与相关工作的对比

| 维度 | DROID-SLAM | DynaSLAM | RoDynRF | DROID-W |
|------|-----------|----------|---------|---------|
| 动态处理 | 无 | 语义分割屏蔽 | Neural implicit | 不确定性加权 |
| 动态先验 | 无 | 需要（预定义类别） | 无 | 无 |
| 实时性 | ~15 FPS | ~5 FPS | ~0.1 FPS | ~10 FPS |
| 场景重建 | 稀疏/半稠密 | 稀疏 | 稠密 | 稀疏/半稠密 |
| 鲁棒性 | 动态失败 | 已知类别 OK | 通用 | 通用 |

DROID-W 在"通用动态鲁棒性"和"实时性"之间取得了最好的平衡，是工程实用性与学术新颖性兼具的方案。

## 启发与关联

- 不确定性加权的思路是通用的"鲁棒估计"技巧，可迁移到光流估计、立体匹配、SfM 等所有基于 BA/最小二乘的视觉几何任务
- DINOv2 特征作为"万能语义描述子"的使用方式令人印象深刻，后续可探索将其用于 loop closure detection、place recognition 等 SLAM 子模块
- 与 DMAligner 形成有趣对比：两者都面对"动态场景"挑战，但 DMAligner 用生成式方法"绕过"问题，DROID-W 用不确定性加权"容忍"问题——思路完全不同但各有优势
- 后续可探索将不确定性估计与 3DGS 动态重建结合，实现动态场景的实时高质量重建

## 评分

- 新颖性: 7/10 — 不确定性加权 BA 本身不新，但与 DINOv2 特征的结合以及在 DROID-SLAM 上的无缝集成是实用的创新
- 实验充分度: 8/10 — 多数据集评估 + 详细消融 + 自建野外数据集，但缺少与部分最新方法的定量对比
- 写作质量: 8/10 — 问题动机清晰，方法描述简洁，实验组织合理
- 价值: 8/10 — 直接提升经典 SLAM 系统的动态鲁棒性，工程部署价值高，10 FPS 实时性是一大卖点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](vggt-slam.md)
- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] Scene Grounding In the Wild](scene_grounding_in_the_wild.md)
- [\[CVPR 2026\] SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM](sgad-slam_splatting_gaussians_at_adjusted_depth_for_better_radiance_fields_in_rg.md)

</div>

<!-- RELATED:END -->
