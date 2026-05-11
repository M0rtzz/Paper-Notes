---
title: >-
  [论文解读] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
description: >-
  [CVPR 2026][3D视觉][real-to-sim] 提出 Wanderland real-to-sim 框架：利用手持多传感器扫描仪（LiDAR+IMU+RGB）采集开放世界室内外场景，通过 LIV-SLAM 获取度量级精确几何与相机位姿，结合 3DGS 实现光学真实感渲染 + 几何接地碰撞仿真，构建 530 场景/42 万帧/380 万 m² 的大规模数据集，系统证明纯视觉重建在度量精度、Mesh 质量和导航策略训练/评估可靠性上远不及 LiDAR 增强方案。
tags:
  - CVPR 2026
  - 3D视觉
  - real-to-sim
  - 3D Gaussian Splatting
  - LiDAR-Inertial-Visual SLAM
  - 导航仿真
  - 几何接地
  - 新视角合成
---

# Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI

**会议**: CVPR 2026  
**arXiv**: [2511.20620](https://arxiv.org/abs/2511.20620)  
**代码**: [https://ai4ce.github.io/wanderland/](https://ai4ce.github.io/wanderland/)  
**领域**: 3D视觉 / 具身AI / 仿真环境  
**关键词**: real-to-sim, 3D Gaussian Splatting, LiDAR-Inertial-Visual SLAM, 导航仿真, 几何接地, 新视角合成  

## 一句话总结

提出 Wanderland real-to-sim 框架：利用手持多传感器扫描仪（LiDAR+IMU+RGB）采集开放世界室内外场景，通过 LIV-SLAM 获取度量级精确几何与相机位姿，结合 3DGS 实现光学真实感渲染 + 几何接地碰撞仿真，构建 530 场景/42 万帧/380 万 m² 的大规模数据集，系统证明纯视觉重建在度量精度、Mesh 质量和导航策略训练/评估可靠性上远不及 LiDAR 增强方案。

## 研究背景与动机

**开放世界具身导航需要高保真仿真**：具身 AI 从室内扩展到城市街道、校园、商业区等开放场景，需要大空间尺度、室内外混合覆盖、高保真传感器模拟和可靠物理交互的仿真环境

**经典 RGB-D 数据集局限于室内**：Matterport3D、ScanNet、HM3D 等采用三脚架式 RGB-D 采集，室外阳光干扰和有限测距范围导致无法适用；大规模低纹理环境下位姿漂移严重

**视频-3DGS 方案几何不可靠**：Vid2Sim、GaussGym 等从在线视频构建仿真环境，存在三个根本缺陷：(a) 纯 RGB SfM/深度估计得到非度量位姿，(b) 从 3DGS 不透明度提取的碰撞 Mesh 碎片化且缺乏度量接地，(c) 单一视频轨迹导致外推视角渲染质量急剧下降

**训练 vs 评估的仿真需求不同**：视频-3DGS 环境或许可用于训练，但因几何不可靠导致无法作为标准化基准进行可复现的闭环评估

**3D 重建和 NVS 缺少大规模室外度量基准**：现有室外数据集缺乏高精度几何 ground truth，限制了重建和 NVS 方法的评估

## 方法详解

### 整体框架

多传感器采集 → LIV-SLAM 重建（全局一致的度量点云 + 精确相机位姿）→ 从点云初始化 3DGS + 深度正则化训练 → 从点云提取碰撞 Mesh → 3DGS 模型 + Mesh 集成为 USD 场景 → 加载到 Isaac Sim 进行导航训练/评估。核心设计原则：几何来源于 LiDAR（保证度量精度），外观来源于 3DGS（保证光学真实感），两者共享同一坐标系实现无缝集成。

### 1. 数据采集（Section 3.1）

- **设备**: MetaCam Air 手持 3D 扫描仪，集成 Livox Mid-360 非重复式 LiDAR（含内置 IMU）、RTK-GNSS 天线、两个同步 4K 鱼眼相机（>180° FOV），所有传感器出厂标定
- **LiDAR 安装**: 倾斜角度经调优以最大化地面细节捕获和相机 FOV 重叠
- **实时监控**: 配套移动端 App 实时显示彩色点云预览，操作者可主动填补覆盖空白
- **采集协议**:
    - 场景规模：5,000–10,000 m²，平衡复杂度与覆盖范围
    - 触发策略：非固定帧率，而是按位移/角度阈值触发 RGB 采集，确保视点均匀分布
    - 轨迹策略：**训练轨迹**采用闭环路径、密集多视角覆盖所有可导航区域；**外推轨迹**模拟自然导航路径、与训练轨迹最小重叠。这与 Vid2Sim/GaussGym 使用的单方向城市漫步视频形成鲜明对比
    - 质量控制：最小化动态障碍物和反射面、保持单次采集光照一致、实时点云监控验证覆盖完整性
- **多样性**: 在纽约市和泽西市采集，覆盖住宅楼、商业区、街道、广场、大学校园等场景类型，涵盖不同时段（早/中/傍晚）和天气条件（晴/多云/小雨）

### 2. 数据处理流程（Section 3.2）

**a) LIV-SLAM 映射与重建**

- 使用 MetaCam Studio 处理原始传感器数据，实现 LiDAR-惯性-视觉-GNSS 多传感器融合
- 基于成熟的多传感器融合方法（VINS-Mono、GVINS、FAST-LIO2 等）进行扩展
- 输出：密集、度量级点云（毫米级间距）+ 全局一致的相机轨迹
- 这一几何基础直接解决了纯视觉方案的度量精度和完整性问题

**b) 图像预处理**

- **隐私掩码**: 使用 EgoBlur 遮蔽人脸和车牌，确保数据发布的匿名性
- **动态物体掩码**: 使用 YOLOv11 检测并遮蔽行人、动物、车辆等动态物体，3DGS 训练时过滤无效像素
- **去畸变**: 从原始鱼眼图裁切 120° 视角并做透视去畸变，原因包括 3DGS 的低阶畸变近似限制以及检测模型多在针孔图像上训练

### 3. 3DGS 训练策略（Section 3.3）

**a) 度量点云初始化**

- 直接从 LIV-SLAM 输出的致密彩色点云初始化 3D 高斯：原始间距 5–10 mm，下采样至约 500 万点/场景
- 每个点创建一个高斯，通过 KNN 启发式设置初始尺度
- 不透明度参数化为体积密度的反比，抑制大高斯和浮动噪点在初始训练步的主导效应
- 训练分辨率 800×800，使用 gsplat 框架，共 15,000 步

**b) 深度正则化**

- 与使用单目深度估计作为伪 GT 不同，直接将初始化高斯投影到各相机位姿得到深度图作为 GT 深度
- 结合光度损失 + 深度损失进行联合优化
- 为何不冻结高斯中心？实验发现冻结均值虽保持多视角几何一致性，但产生次优视觉质量和退化行为；纯图像监督提高单视角保真度但牺牲未见视角一致性；深度正则化在两者间取得平衡

**c) 训练视角增强**

- 使用 Difix3D+ 预训练模型生成干净且几何准确的新视角进行增强
- 沿训练步逐步将增强视角向远离训练视角的方向扩展
- 这对稳定大规模 3DGS 训练和改善外推视角的传感器仿真质量至关重要

### 4. 几何接地仿真构建（Section 3.4）

**a) Mesh 提取**

- 从全局点云体素化为占据栅格，使用 Marching Cubes 提取几何 Mesh
- 后处理：移除远离采集轨迹的部分，过滤面片数过少的碎片
- 与从 3DGS 不透明度提取 Mesh（Vid2Sim/GaussGym）相比，直接从 LiDAR 点云提取的 Mesh 干净、完整、度量精确

**b) 场景集成**

- Mesh 和 3DGS 模型基于同一全局点云、共享坐标系，直接集成为 USD 格式
- Mesh 提供轻量级物理/碰撞层，3DGS 作为主渲染器
- 场景可直接加载到 Isaac Sim 中进行导航系统的训练和评估

### 5. 导航任务定义（Section 3.5）

- **专家轨迹**: 将 Mesh 导入 Unity，通过 NavMesh 烘焙 API 提取可导航三角面，使用寻路模块生成无碰撞最短路径。起点和终点在采集相机附近采样，支持点目标和图像目标导航
- **语言指令**: 回放轨迹生成第一人称视频 → Gemini VLM 自动生成自然语言导航指令 → 人工验证确保可靠性。相比全人工标注（R2R、RxR）更可扩展且跨场景指令质量更稳定

## 数据集规模

| 指标 | 数值 |
|:---|:---|
| 场景数 | 530 |
| 总帧数 | 420K+ |
| 总采集时长 | 100+ 小时 |
| 覆盖面积 | 3.8M+ m² |
| 场景类型 | 室内外混合（住宅/商业/街道/广场/校园） |
| 每场景数据 | 鱼眼 RGB + 标定参数 + 全局位姿 + 彩色点云 + 3DGS 模型 + 碰撞 Mesh + USD 场景 |
| 扩展计划 | 持续维护至 1,000+ 场景 |

## 实验与结果

### 1. 3D 重建精度（Q1: 纯视觉 vs LIV-SLAM）

以 LIV-SLAM 位姿为 GT，评估 8 种纯视觉方法的相机位姿精度：

| 方法 | T-ATE_raw (m) ↓ | T-ATE_scaled (m) ↓ | R-ATE (°) ↓ | AUC@30 ↑ | 成功率 ↑ |
|:---|:---|:---|:---|:---|:---|
| DUSt3R | 15/14 | 20/18 | 73/60 | 0.12/0.07 | 0.39 |
| MUSt3R | 7.8/5.7 | 10/3.7 | 26/13 | 0.53/0.61 | 0.81 |
| VGGT | 15/14 | 9.9/4.5 | 33/15 | 0.44/0.52 | 0.80 |
| π³ | 15/14 | 4.7/1.4 | 21/6.9 | 0.64/0.76 | 0.89 |
| COLMAP | 16/10 | 8.1/2.3 | 42/10 | 0.50/0.64 | 0.64 |
| COLMAP_calib | 10/9.7 | 4.8/0.30 | 15/5.0 | 0.73/0.83 | 0.87 |
| **所有方法最优** | **2.8** | **0.30** | **5.0** | **0.83** | **0.89** |

**关键发现**: 即使取所有方法的"最优"组合，在不到 100 米的场景中，原始度量误差仍达米级；尺度对齐后平均误差 30 cm / 5°。纯视觉方法在度量精度上的固有模态限制尚未被最新基础模型弥合。

### 2. 光学真实感仿真（Q2: NVS 质量）

在 Wanderland 数据集上统一训练/验证分割评估各方法的新视角合成质量：

| 方法 | 插值 PSNR ↑ | 插值 SSIM ↑ | 插值 LPIPS ↓ | 外推 PSNR ↑ | 外推 SSIM ↑ | 外推 LPIPS ↓ |
|:---|:---|:---|:---|:---|:---|:---|
| 3DGS (COLMAP) | 18.27 | 0.658 | 0.510 | 16.90 | 0.624 | 0.559 |
| 2DGS (COLMAP) | 17.98 | 0.593 | 0.550 | 16.81 | 0.631 | 0.508 |
| Vid2Sim | 17.20 | 0.549 | 0.399 | 16.49 | 0.573 | 0.371 |
| GaussGym | 12.17 | 0.440 | 0.738 | 12.63 | 0.436 | 0.725 |
| **Wanderland (Ours)** | **20.37** | **0.688** | **0.327** | **17.92** | 0.591 | 0.445 |

**关键发现**: Wanderland 在插值视角上全面最优（PSNR 领先 2+ dB），外推视角中 PSNR 同样最高。GaussGym 因 VGGT 重建不精确导致指标最差。Vid2Sim 依赖不可靠的单目深度估计引入额外噪声。语义一致性实验表明，GaussGym 渲染碎片化导致 Grounded SAM 2 无法检测关键环境元素，Vid2Sim 的 DINOv3 特征与 GT 严重偏离。

### 3. 导航策略训练与评估（Q3: 仿真可靠性）

**RL 训练对比**: 在 Vid2Sim 环境中 RL 训练后模型普遍恶化（CityWalker SR 下降 21%），因不准确几何鼓励局部短但全局不可靠行为；在 Wanderland 环境中所有模型显著提升（CityWalker SR 提升 14%，干预率降低 23%）。

**评估可靠性**: 同一模型在 Vid2Sim 环境中评估显示更低成功率和更高干预率，说明几何不可靠的环境无法支撑忠实评估。

**导航基准**: 在 Wanderland 全量数据集上基准测试 5 个预训练模型，没有任何模型室外成功率超过 31%，凸显开放世界导航的巨大研究空间。NaVILA（VLN 模型）表现最优（室内 SR=0.47，室外 SR=0.31），得益于 LLM 的语义理解能力。

## 局限性

1. **采集帧率限制**: 当前硬件限制为 1 FPS，视点采样密度不足，影响复杂场景渲染质量
2. **静态场景假设**: 未建模城市动态元素（行人、车辆、交通模式），需要未来集成行为预测和交互仿真

## 个人思考

1. **LiDAR 依赖的可扩展性权衡**: 框架依赖专业硬件（MetaCam ≈ 商用 3D 扫描仪），数据集规模受采集成本制约。与从海量网络视频零成本构建环境的 GaussGym 路线形成对比——两条路线可能最终走向混合方案：LiDAR 提供少量高质量锚点场景用于标定和评估，视觉方法大规模扩展覆盖
2. **几何接地的必要性被量化**: 论文对"仿真几何质量如何影响下游导航策略"给出了罕见的定量分析——RL 训练在不可靠环境中不仅无效还会恶化模型，这个结论对整个 sim-to-real 社区有重要参考价值
3. **与 NeRF/3DGS 场景理解的连接**: Wanderland 的多传感器数据天然适合评估最新的 feed-forward 3D 重建模型（DUSt3R、VGGT 等），作为 GT 基准的价值可能超越导航仿真本身
4. **动态场景扩展**: 当前静态假设是重要限制。结合 4D Gaussian Splatting 或视频生成模型注入动态代理可能是下一步方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)
- [\[CVPR 2026\] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [\[CVPR 2026\] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction](fluidgaussian_propagating_simulation-based_uncertainty_toward_functionally-intel.md)
- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](../../NeurIPS2025/3d_vision/towards_3d_objectness_learning_in_an_open_world.md)

</div>

<!-- RELATED:END -->
