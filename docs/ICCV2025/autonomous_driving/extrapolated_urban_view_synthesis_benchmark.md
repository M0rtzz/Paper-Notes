---
title: >-
  [论文解读] Extrapolated Urban View Synthesis Benchmark
description: >-
  [ICCV2025][自动驾驶][novel view synthesis] 提出首个外推式城市视图合成（EUVS）基准，利用多遍历/多车辆/多相机公开数据集系统评估外推场景下 3DGS 及 NeRF 方法的泛化能力，揭示当前方法严重过拟合训练视角。 - 核心问题：自动驾驶的视觉仿真依赖新视角合成（NVS）…
tags:
  - "ICCV2025"
  - "自动驾驶"
  - "novel view synthesis"
  - "3D Gaussian Splatting"
  - "benchmark"
  - "extrapolation"
---

# Extrapolated Urban View Synthesis Benchmark

**会议**: ICCV2025  
**arXiv**: [2412.05256](https://arxiv.org/abs/2412.05256)  
**代码**: [项目主页](https://ai4ce.github.io/EUVS-Benchmark)  
**领域**: 自动驾驶  
**关键词**: novel view synthesis, 3D Gaussian Splatting, benchmark, extrapolation, autonomous driving

## 一句话总结

提出首个外推式城市视图合成（EUVS）基准，利用多遍历/多车辆/多相机公开数据集系统评估外推场景下 3DGS 及 NeRF 方法的泛化能力，揭示当前方法严重过拟合训练视角。

## 研究背景与动机

- **核心问题**：自动驾驶的视觉仿真依赖新视角合成（NVS），3D Gaussian Splatting 等方法在插值（interpolation）评估下表现优异，但实际驾驶中车辆视角变化远超训练分布（外推场景），现有方法在外推设置下的表现缺少系统评估
- **现状不足**：
    - 现有评估几乎都只在高度相关的训练/测试视角间做插值测试，训练/测试分布极为接近
    - 少数研究尝试了外推视角合成（如 GGS、VEGS、FreeSim），但受限于缺乏真实数据做定量评估，只能做定性分析
    - 现有方法仅聚焦单一难度，没有系统性的多层级评估框架
- **动机**：构建一个标准化的、包含多种难度等级的外推 NVS 基准，用真实数据实现定量+定性全面评估，推动更鲁棒的 NVS 方法发展

## 方法详解

### 数据来源与构建

EUVS benchmark 整合三个公开自动驾驶数据集：

| 数据集 | 特性 | 用途 |
|--------|------|------|
| **nuPlan** | 1200 小时驾驶数据，4 个城市，多遍历 + 多相机 | Level 1（多车道变道）和 Level 2（多相机旋转） |
| **Argoverse 2** | 1000 个标注 3D 场景，6 个美国城市 | Level 1 和 Level 3（跨路径遍历） |
| **MARS** | 多车辆协同 + 多遍历，同一区域异步重访 | Level 3（十字路口、T 型路口等） |

总量：**90,810 帧，345 个视频，104 个场景**。

### 评估框架：三级难度

论文将外推视角变化分为三个难度等级，对应不同驾驶场景：

- **Level 1（仅平移）**：车辆位置偏移但朝向不变，对应变道场景。使用多遍历的不同车道数据训练，在邻近车道做测试
- **Level 2（仅旋转）**：训练和测试在同一位置但朝向不同。使用 nuPlan 的前三后三共 6 个相机训练，两个侧向相机测试
- **Level 3（平移 + 旋转）**：位置和朝向同时变化，最具挑战性。使用同一路口不同路径遍历的数据（如十字路口的不同方向进入路线）

### 场景预处理

- 使用 **COLMAP** 进行多帧位姿估计和稀疏重建，为 3DGS 提供初始化
- 使用 **SegFormer** 分割并遮蔽可移动物体（训练和评估阶段均排除动态物体）

### 评估的基线方法（9类）

1. **Vanilla 3DGS**：标准 3D Gaussian Splatting
2. **3DGM**：利用多遍历一致性区分瞬态/永久元素
3. **GaussianPro (GSPro)**：多帧几何优化引导高斯致密化，平面化结构
4. **2DGS**：将 3D 体积投射为 2D 平面高斯盘
5. **PGSR**：无偏深度渲染 + 多视角正则化
6. **VEGS**：扩散先验引导增强视角，减少浮动伪影
7. **Feature 3DGS**：嵌入语义特征的并行 N 维高斯光栅化
8. **Zip-NeRF**：抗锯齿网格 NeRF，多采样 + 距离归一化
9. **Instant-NGP**：多分辨率哈希编码的快速 NeRF

### 评估指标

- **图像质量**：PSNR、SSIM、LPIPS
- **语义相似度**：DINOv2 特征余弦相似度（Feature Cosine Similarity）
- **几何精度**：RMSE、δ₁.₂₅（深度估计指标）

## 实验关键数据

### 三级难度下的性能衰退（所有方法平均值）

| 难度 | PSNR 下降 | SSIM 下降 | LPIPS 恶化 | 特征相似度下降 |
|------|----------|----------|-----------|-------------|
| Level 1（仅平移） | 24.6% | 12.8% | 25.3% | 11.2% |
| Level 2（仅旋转） | 25.6% | 14.7% | 62.5% | 14.8% |
| Level 3（平移+旋转） | 30.6% | 19.5% | 70.0% | 35.9% |

→ 难度越高性能衰退越严重，Level 3 下 LPIPS 恶化高达 70%，说明感知质量严重退化。

### 各方法 Level 1 测试集表现（代表性数据）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| GSPro | **16.39** | 0.7189 | **0.2450** |
| 3DGM | 16.35 | **0.7248** | 0.2542 |
| 3DGS | 16.37 | 0.7203 | 0.2599 |
| VEGS | 15.88 | 0.7047 | 0.3062 |

### Level 2 关键发现

- **VEGS 凭借扩散先验在旋转场景大幅领先**：PSNR 23.33（比 3DGS 的 19.53 高 19.4%），因为扩散模型能补全未见区域
- Zip-NeRF 在训练集上 PSNR 高达 29.06，但测试集暴跌至 17.36（下降 40.3%），过拟合最严重

### Level 3 关键发现

- 所有方法 PSNR 均低于 15，无一方法有显著优势
- Feature 3DGS 在 LPIPS 上最优（0.3816），但绝对值仍然很差
- 2DGS 表现最差（PSNR 仅 11.36），平面表示无法应对复杂城市场景的大视角变化

### 深度估计对比（Level 1）

3DGM 在大多数深度指标上最优（RMSE=0.301, δ₁=0.651），而 VEGS 虽然 SqRel 最低但整体深度精度欠佳。

### 多遍历数据量影响

使用 GSPro 在 Level 1 上逐步增加训练遍历次数，发现 PSNR 和 SSIM 随遍历数增加持续改善后趋于饱和，说明更多视觉数据确实有助于外推合成。

## 亮点与洞察

1. **首个外推 NVS 定量基准**：填补了长期以来只能做定性评估的空白，三级难度设计覆盖了真实驾驶中的典型场景（变道→换相机→跨路口）
2. **扩散先验并非万能**：VEGS 在旋转场景（Level 2）表现突出，但在 Level 3 下同样崩溃，说明扩散先验能补偿未见区域但不能从根本上解决大视角偏移
3. **几何改进收益有限**：平面化方法（2DGS、PGSR）在路面重建上更好，但处理复杂纹理对象时反而更差；椭球方法（3DGS、3DGM）在高纹理区域更优但路面过拟合
4. **GS vs NeRF 各有所长**：隐式表示（NeRF）擅长低纹理区域（道路、天空）的平滑过渡，显式表示（GS）更擅长保留高频细节（车道标线、栏杆），暗示混合表示是潜在方向
5. **光照不一致是额外挑战**：GS-W 通过分离内在/动态外观特征处理光照变化后，训练/测试指标均大幅提升，但外推性能下降仍然显著

## 局限与展望

- **静态场景为主**：基准测试剔除了动态物体，OmniRe 动态基线的初步结果（PSNR 仅 15.32）表明动态外推更难，但未深入探索
- **数据量有限**：虽然 9 万帧规模可观，但每个场景的独立训练集仍较小，更大规模训练数据可能显著改善外推泛化
- **缺少生成式基线**：未评估 DriveDreamer4D、Vista 等基于世界模型/视频生成的方法，这些方法可能天然更擅长外推
- **无深度真值**：深度评估使用伪真值（Depth Anything），引入额外噪声
- **混合表示方向未实践**：论文指出椭球+平面混合可能更优，但未实现和验证
- **COLMAP 初始化瓶颈**：多遍历数据的位姿配准依赖 COLMAP，大场景下的配准精度和效率仍是限制因素

## 相关工作与启发

- **RapNeRF / NerfVS**：早期外推视角合成尝试，通过随机光线采样或几何scaffold改善未见视角，但仅限室内
- **GGS**：虚拟车道生成模块处理变道数据不足，专注 Level 1
- **AutoSplat**：车道变换动态场景中的几何+反射一致性约束
- **FreeSim / VEGS / SGD**：用扩散模型先验增强 3DGS 泛化能力
- **GS-W**：分离内在属性和动态外观特征处理光照不一致，是多遍历数据的有效策略
- **启发方向**：
    - 更大规模预训练 + 场景级微调（类似 foundation model 路线）
    - 混合显式/隐式表示
    - 扩散先验 + 深度正则化的联合使用
    - 利用更多遍历数据做持续学习

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](../../CVPR2025/autonomous_driving/evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[NeurIPS 2025\] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles](../../NeurIPS2025/autonomous_driving/urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[CVPR 2025\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2025/autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2025\] Towards Autonomous Micromobility through Scalable Urban Simulation](../../CVPR2025/autonomous_driving/towards_autonomous_micromobility_through_scalable_urban_simulation.md)

</div>

<!-- RELATED:END -->
