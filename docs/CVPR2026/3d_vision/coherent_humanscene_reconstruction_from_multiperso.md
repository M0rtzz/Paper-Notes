---
title: >-
  [论文解读] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass
description: >-
  [CVPR 2026][3D视觉][多视图人体重建] 提出CHROMM统一框架，整合Pi3X几何先验和Multi-HMR人体先验到单一前馈网络，从多人多视图视频中一次性联合重建相机、场景点云和SMPL-X人体网格，无需外部模块、预处理或迭代优化，RICH上多视图WA-MPJPE达53.1mm且比HAMSt3R快8倍以上。
tags:
  - CVPR 2026
  - 3D视觉
  - 多视图人体重建
  - 多人场景
  - SMPL-X
  - 3D基础模型
  - 尺度对齐
---

# Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass

**会议**: CVPR 2026  
**arXiv**: [2603.12789](https://arxiv.org/abs/2603.12789)  
**代码**: [项目页面](https://nstar1125.github.io/chromm)  
**领域**: 3D视觉 / 人体-场景联合重建  
**关键词**: 多视图人体重建, 多人场景, SMPL-X, 3D基础模型, 尺度对齐

## 一句话总结

提出CHROMM统一框架，整合Pi3X几何先验和Multi-HMR人体先验到单一前馈网络，从多人多视图视频中一次性联合重建相机、场景点云和SMPL-X人体网格，无需外部模块、预处理或迭代优化，RICH上多视图WA-MPJPE达53.1mm且比HAMSt3R快8倍以上。

## 研究背景与动机

**领域现状**：3D人体-场景联合重建是计算机视觉核心问题，应用于机器人、自动驾驶和AR/VR。近年3D基础模型（DUSt3R、VGGT、Pi3X）推动了场景重建，Multi-HMR实现了多人人体网格恢复。

**现有痛点**：

1. UniSH、Human3R等单目方法无法利用多视图信息，精度受限
2. HSfM、HAMSt3R等多视图方法依赖额外模块（2D关键点检测器、跨视图ReID模块）或需迭代优化，系统复杂度高
3. 基于外观的Re-ID方法在视觉相似场景（穿制服等）失败
4. Pi3X输出的近度量尺度与SMPL真实度量尺度存在差异——人体穿透地面或漂浮

**核心矛盾**：需同时重建场景和多人人体，但两者尺度不一致、多人跨视图关联困难、且不想依赖外部预处理。

**本文目标** 构建不依赖外部模块和预处理数据的统一前馈框架，一次性完成多人多视图人体-场景联合重建。

**切入角度**：融合Pi3X(场景)和Multi-HMR(人体)两大先验，设计尺度调整模块桥接二者，用几何线索替代外观匹配做跨视图关联。

**核心 idea**：双编码器后期融合 + head-pelvis比率尺度调整 + 视图不变/依赖分解融合 + 几何驱动多人关联。

## 方法详解

### 整体框架

输入多视图多人视频{I_t^v} → 双编码器（Pi3X提取场景特征，Multi-HMR提取人体特征）→ Pi3X解码器重建点图和相机 → 头部检测提取人体token与场景token融合 → SMPL解码器回归姿态/体型/平移 → 测试时：per-view跟踪 → 几何驱动跨视图多人关联 → 视图不变/依赖分解融合 → 尺度调整模块对齐场景和人体。

### 关键设计

1. **双编码器后期融合架构**

    - Pi3X编码器捕获全局3D几何，Multi-HMR编码器针对人体表示优化
    - 关键决策：**避免早期融合**——将人体token送入Pi3X解码器会破坏输入分布，损害场景重建
    - 人体token仅在解码后与场景token通过MLP融合：$H_n = \text{MLP}_{\text{fuse}}([Z_n^{\text{scene}} | Z_n^{\text{human}}])$

2. **深度残差平移估计**

    - 不直接回归3D头部平移，而是利用Pi3X点图提供的深度先验
    - 预测相对场景深度图的残差 $d_n^m = d_{n,m}^{\text{coarse}} + \Delta d_n^m$，结合2D头部关键点和相机内参反投影为3D位置
    - 消融：深度残差(107.5) vs 直接深度(133.8) vs 直接平移回归(196.4)，差距显著

3. **Head-Pelvis比率尺度调整**

    - 问题：Pi3X输出近度量尺度s可能偏小(人穿地)或偏大(人浮空)
    - 解法：计算图像2D头-骨盆距离ℓ^img与投影SMPL头-骨盆距离ℓ^smpl的比值，跨所有帧和人平均得全局调整因子 $r = \frac{1}{|\mathcal{S}|}\sum \frac{\ell^{\text{smpl}}}{\ell^{\text{img}}}$，s*=r·s
    - 骨盆定位粗到精：头部token估粗位置 → 对应patch回归偏移 → 骨盆出界则用粗位置
    - 消融：尺度调整使WA-MPJPE从169.7降至102.6（-39.5%）

4. **多视图融合（测试时，无需优化）**

    - 视图不变量（体型β、姿态θ）：直接参数平均，优于隐式token max-pooling
    - 视图依赖量（旋转R、平移τ）：转到世界坐标系后分别用四元数平均和多视图射线三角化
    - 消融：Avg+Tri(53.1) > MaxPool+Tri(63.2) > Only Avg(69.3)

5. **基于几何的多人关联**

    - per-view跟踪：头部token L2距离帧间匹配，Sinkhorn最优传输处理未匹配检测
    - 跨视图关联代价：$\mathcal{C}(a,b) = 0.8 \cdot \|3D位置差\| + 0.2 \cdot \|规范姿态差\|$，匈牙利算法一对一匹配
    - 消融：Position alone 91.1% precision vs Pose alone 70.6%，Combined 91.3%

### 损失函数 / 训练策略

- **两阶段训练**：Stage 1冻结Pi3X+Multi-HMR编码器，训练SMPL解码器等新模块(20 epoch, BEDLAM, lr=5e-5, 前10 epoch不启用尺度调整)
- Stage 2仅解冻骨盆检测MLP(10 epoch, 混合3DPW+MPII+COCO+BEDLAM, lr=1e-4)
- Stage 1损失：3D顶点/关节L1(λ=5.0) + 2D重投影L1 + SMPL参数L1 + 检测BCE + 骨盆BCE
- Stage 2新增：Chamfer距离（可见SMPL顶点 vs 预测深度图）
- 训练设备：4×A100约2天

## 实验关键数据

### 主实验（全局人体运动估计）

| 方法 | 多视图 | 无外部模块 | EMDB-2 WA-MPJPE↓mm | RICH WA-MPJPE↓mm | RICH W-MPJPE↓mm |
|------|--------|-----------|---------------------|-------------------|-----------------|
| JOSH3R | ✗ | ✗ | 220.0 | - | - |
| UniSH | ✗ | ✗ | 118.5 | 118.1 | 183.2 |
| Human3R | ✗ | ✓ | 112.2 | 110.0 | 184.9 |
| CHROMM-mono | ✗ | ✓ | **102.6** | 87.5 | 138.3 |
| CHROMM-multi | ✓ | ✓ | - | **53.1** | **79.0** |

### 多视图姿态估计

| 方法 | 无ReID | 无优化 | EgoHumans W-MPJPE↓(m) | EgoHumans GA-MPJPE↓(m) | EgoExo4D W-MPJPE↓(m) |
|------|--------|--------|----------------------|----------------------|---------------------|
| HSfM | ✗ | ✗ | 1.04 | 0.21 | 0.56 |
| HAMSt3R | ✓ | △ | 3.80 | 0.42 | 0.51 |
| **CHROMM** | ✓ | ✓ | **0.51** | **0.15** | **0.26** |

### 运行时间

| 方法 | 单帧推理时间(3人4视图) |
|------|---------------------|
| HSfM | ~118s |
| HAMSt3R | ~32s |
| **CHROMM** | **~4s** (8×+加速) |

### 关键发现

- 多视图融合大幅提升：RICH WA-MPJPE从87.5(单目)降至53.1(多视图)，提升39.3%
- 尺度调整是最关键模块：去除后WA-MPJPE从102.6升至169.7(+65.5%)
- 深度残差策略比直接平移回归好89mm(107.5 vs 196.4)
- 几何关联(91.3% accuracy)远优于仅用姿态(70.6%)
- CHROMM比HSfM快29倍、比HAMSt3R快8倍，同时无需ReID

## 亮点与洞察

- **首个端到端多人多视图人体-场景联合重建框架**：不依赖任何外部模块、预处理或优化
- **Head-Pelvis比率尺度调整**：用人体解剖学比例桥接场景与人体的尺度差异，设计简洁有效
- **视图不变/依赖分解融合**：显式参数平均+三角化优于隐式token聚合
- **几何驱动跨视图关联**：避免外观匹配在制服场景的失败，3D位置+规范姿态组合设计精巧

## 局限与展望

- 严重依赖头部token进行人体检测——头部遮挡或不可见时性能下降
- 双编码器未整合为统一编码器——场景和人体的交互建模仍有提升空间
- 极端近景（头部占满图像）或近距离人际交互是典型失败案例
- 尺度调整依赖骨盆可见性——全身遮挡时退化

## 相关工作与启发

- **vs Human3R**：CHROMM扩展到多视图且无需外部模块，EMDB-2好9.6mm，RICH好57mm
- **vs HSfM**：CHROMM快29倍，EgoHumans W-MPJPE 0.51m vs 1.04m(好50%)
- **vs HAMSt3R**：CHROMM快8倍，支持多人关联无需外部ReID
- 启发：3D基础模型与人体先验融合是趋势，尺度对齐是核心工程问题；视图不变/依赖分解可推广到其他多视图估计任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个无外部依赖的多人多视图统一框架，尺度调整和几何关联有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集、单目/多视图、详尽消融、运行时分析齐全
- 写作质量: ⭐⭐⭐⭐ 贡献清晰，每个设计决策都有实验验证
- 价值: ⭐⭐⭐⭐ 快速推理+无需预处理对实际部署有意义

<!-- RELATED:START -->

## 相关论文

- [PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)
- [BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction](mv-roma_from_pairwise_matching_into_multi-view_track_reconstruction.md)
- [Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
