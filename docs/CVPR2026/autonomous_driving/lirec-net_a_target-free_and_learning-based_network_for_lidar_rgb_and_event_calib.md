---
title: >-
  [论文解读] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration
description: >-
  [CVPR 2026][自动驾驶][多传感器标定] 提出LiREC-Net，首个统一框架同时完成LiDAR-RGB和LiDAR-Event相机的无靶标外参标定，通过共享LiDAR表示（融合3D点特征和投影深度特征）和成对代价体积实现跨模态对齐，在KITTI上达到1.80cm/0.11°、DSEC上达到2.51cm/0.14°（LiDAR-RGB）和1.18cm/0.07°（LiDAR-Event）的标定精度。
tags:
  - CVPR 2026
  - 自动驾驶
  - 多传感器标定
  - 无靶标标定
  - 三模态融合
  - 事件相机
  - 外参估计
---

# LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration

**会议**: CVPR 2026  
**arXiv**: [2602.21754](https://arxiv.org/abs/2602.21754)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 多传感器标定, 无靶标标定, 三模态融合, 事件相机, 外参估计

## 一句话总结

提出LiREC-Net，首个统一框架同时完成LiDAR-RGB和LiDAR-Event相机的无靶标外参标定，通过共享LiDAR表示（融合3D点特征和投影深度特征）和成对代价体积实现跨模态对齐，在KITTI上达到1.80cm/0.11°、DSEC上达到2.51cm/0.14°（LiDAR-RGB）和1.18cm/0.07°（LiDAR-Event）的标定精度。

## 研究背景与动机

自动驾驶系统依赖多传感器融合来构建一致的环境感知。传感器融合的前提是**精确的外参标定**——知道每个传感器在公共坐标系中的相对位姿。然而在实际部署中，车辆振动、温度变化、轻微碰撞和日常维护会导致传感器位姿逐渐漂移，初始标定不再准确。

传统的**靶标标定**（棋盘格、ArUco标记等）虽然精度高，但需要专门的受控场景、仔细摆放、重复采集和人工监督，**无法在运行中频繁执行**。**无靶标方法**直接从自然驾驶场景中标定，无需特殊setup，可随时重复。

基于深度学习的无靶标标定方法（LCCNet、RegNet、CalibNet等）已取得进展，但存在一个关键局限：**它们只处理单一模态对**。例如LCCNet只做LiDAR-RGB，MULiEv只做LiDAR-Event。当一个系统同时有3种传感器时，需要**独立训练两个网络**，这不仅增加了计算冗余，还可能导致**标定不一致**——两个独立网络分别估计的LiDAR-RGB和LiDAR-Event变换可能在3D空间中不自洽。

LiREC-Net的核心idea是：**设计一个共享LiDAR表示，让同一个LiDAR特征同时服务于LiDAR-RGB和LiDAR-Event两条标定路径**，既减少冗余又保证一致性。

## 方法详解

### 整体框架

LiREC-Net是一个双路径架构：

1. **共享LiDAR分支**：从点云提取统一表示
2. **RGB编码器** + **Event编码器**：分别提取视觉/事件特征
3. **两个成对代价体积**：LiDAR-RGB和LiDAR-Event各一个
4. **两个上下文模块 + 预测头**：分别输出外参

### 关键设计

1. **共享LiDAR表示（Point+Depth融合）**: LiDAR特征通过**两个互补编码器**并行提取：

    - **点编码器**：使用Point-Transformer-V3 (PTV3)直接处理无序3D点，通过空间填充曲线序列化实现高效局部注意力，捕获**细粒度几何结构**
    - **深度编码器**：将点云投影到图像平面生成单通道深度图，用MViTV2提取**密集空间上下文**

   两者特征通过**缩放特征投影(SFP)**对齐到相同分辨率后拼接融合：

    $\mathbf{F}^{\text{Li}} = \text{Concat}(\text{SFP}(\mathbf{F}^{\text{point}}), \mathbf{F}^{\text{depth}})$

   消融实验证明这两种特征缺一不可——仅用点特征时LiDAR-RGB平移误差从2.51cm暴增至14.43cm。

2. **缩放深度投影 (SDP) 和缩放特征投影 (SFP)**: 关键的工程细节。投影点云到图像平面时，传统做法是先用原始内参投影，再resize。但resize会引入**模糊伪影**，破坏精细的特征对齐。SDP通过先缩放内参矩阵再投影来避免此问题：

    $R' = \text{diag}\left(\frac{W'}{W}, \frac{H'}{H}, 1\right), \quad \mathbf{K'}_{\text{Cam}} = R' \mathbf{K}_{\text{Cam}}$

   消融显示移除SDP和SFP后LiDAR-Event误差从1.18cm/0.07°恶化到3.35cm/0.30°。

3. **成对代价体积**: 借鉴PWC-Net和LCCNet，计算LiDAR特征与相机特征之间逐像素的局部相关性。对像素 $\mathbf{p}=(x,y)$ 和位移 $(\Delta x, \Delta y)$：

    $\mathcal{C}(y,x,\Delta x,\Delta y) = \frac{1}{C}\sum_{c=1}^{C} \mathbf{F}^{\text{Li}}_{c,y,x} \cdot \mathbf{F}^{\text{Cam}}_{c,y+\Delta y,x+\Delta x}$

   代价体积维度为 $H'' \times W'' \times (2d+1)^2$，通过滑动窗口内的逐通道内积衡量跨模态局部相似性。

4. **迭代细化**: 训练多个独立模型，每个针对不同的误差范围（从大到小：±20°/150cm → ±1°/10cm），评估时级联应用：

    $\hat{\mathbf{T}}^{v,(k)} = \Delta\hat{\mathbf{T}}^{v,(k)} \hat{\mathbf{T}}^{v,(k-1)}$

   这种粗到精的策略让每个stage专注于特定精度范围的校正。

### 损失函数 / 训练策略

总损失由两个模态对的损失相加：$\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{Li-RGB}} + \mathcal{L}^{\text{Li-Ev}}$

每对的损失包含3项：

$$\mathcal{L}^v = (1-w)(\lambda_t \mathcal{L}^v_{\text{trans}} + \lambda_r \mathcal{L}^v_{\text{rot}}) + w \mathcal{L}^v_{\text{pcd}}$$

- **平移损失**: Smooth L1 loss on $\hat{\mathbf{t}}^v$
- **旋转损失**: 预测和真值四元数之间的角距离 $\theta(\hat{\mathbf{q}}^v, \mathbf{q}^v)$
- **点云距离损失**: 变换后点云与真值变换后点云的L2距离，确保几何一致

训练细节：Adam优化器，lr=3e-4，milestone衰减(×0.5)；DSEC第一阶段150 epochs，后续70 epochs；batch size 64，4×A6000/L40S GPU。

## 实验关键数据

### 主实验 — KITTI数据集

| 方法 | LiDAR-RGB误差 | LiDAR-Event误差 |
|------|-------------|----------------|
| RegNet | 6.00cm / 0.28° | — |
| CalibNet | 4.34cm / 0.41° | — |
| LCCNet | 1.59cm / 0.16° | — |
| PseudoCal | **1.18cm / 0.05°** | — |
| **LiREC-Net** | 1.80cm / 0.11° | **1.82cm / 0.12°** |

### 主实验 — DSEC数据集（5 stages）

| 方法 | LiDAR-RGB误差 | LiDAR-Event误差 |
|------|-------------|----------------|
| MULiEv (2 stages) | — | 0.81cm / 0.10° |
| LiREC-Net (2 stages) | 2.62cm / 0.30° | 2.05cm / 0.25° |
| **LiREC-Net (5 stages)** | **2.51cm / 0.14°** | **1.18cm / 0.07°** |

### 消融实验 — 特征融合（DSEC）

| Point特征 | Depth特征 | LiDAR-RGB | LiDAR-Event |
|-----------|-----------|-----------|-------------|
| ✓ | ✗ | 14.43cm/0.70° | 14.05cm/0.64° |
| ✗ | ✓ | 2.97cm/0.70° | 2.16cm/0.60° |
| ✓ | ✓ | **2.51cm/0.14°** | **1.18cm/0.07°** |

### 三模态 vs 双模态效率对比

| 配置 | 推理时间(s) | 参数量(B) | 显存(GiB) |
|------|------------|----------|-----------|
| 双模态（KITTI, 两个独立网络） | 0.51 | 1.9 | 14.6 |
| **三模态（KITTI, 统一网络）** | **0.33** | **1.7** | **11.1** |
| 双模态（DSEC） | 0.44 | 1.9 | 14.6 |
| **三模态（DSEC）** | **0.31** | **1.7** | **11.1** |

### 关键发现

- **点特征+深度特征缺一不可**：仅用点特征时平移误差暴增5.8倍（2.51→14.43cm），因为点特征缺少密集空间上下文；仅用深度特征时旋转误差高5倍（0.14→0.70°），因为深度图丢失了3D细粒度结构
- **三模态优于双模态**：不仅精度相当甚至更好（共享LiDAR特征提供正则化），效率也更高（推理快35%，显存省24%）
- **SDP和SFP的投影精度至关重要**：两种缩放投影各自贡献了互补的精度提升
- **MViTV2 > ResNet**：Transformer的全局特征建模能力对跨模态对齐更有利

## 亮点与洞察

- **"三模态统一"的思路有实用价值**：随着事件相机在自动驾驶中越来越普及（如Prophesee），统一的多传感器标定方案是实用需求
- **共享LiDAR表示既是效率优势也是精度优势**：避免了两个独立LiDAR编码器的不一致性
- **SDP/SFP的工程洞察**很有价值：resize引入的模糊伪影在精细对齐任务中是关键瓶颈
- 在DSEC上首次建立了LiDAR-RGB标定基线（此前没有方法在DSEC上报告LiDAR-RGB结果）

## 局限与展望

- **假设RGB和Event相机已预标定**：即 $\mathbf{T}^{\text{Ev} \to \text{RGB}}$ 已知。作者指出可以尝试在框架内联合估计此变换
- 仅处理LiDAR/RGB/Event三种传感器，未扩展到热成像、毫米波雷达等
- 需要为每个误差范围独立训练模型（5 stages = 5个模型），训练成本较高
- 点云固定采样 $N$ 个点（KITTI 20000, DSEC 5000），不够灵活
- 代价体积的滑动窗口大小 $d$ 未充分分析其对不同误差范围的影响

## 相关工作与启发

- 直接受LCCNet启发，将其核心思想（代价体积+迭代细化）从双模态扩展到三模态
- Point-Transformer-V3作为点云编码器的选择值得注意——空间填充曲线序列化使其比PointNet++更高效
- MULiEv是唯一的LiDAR-Event标定先驱工作，但仅处理单模态对
- 与CalibNet的自监督标定思路不同，LiREC-Net采用有监督回归，依赖人工扰动的伪不对齐

## 评分

- **新颖性**: ⭐⭐⭐⭐ 三模态统一标定是新颖且有意义的问题定义，共享LiDAR表示的设计有创意
- **实验充分度**: ⭐⭐⭐⭐ 两个数据集、全面消融、效率分析，但缺少真实场景（非人工扰动）的验证
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，formulation严谨，图表直观
- **价值**: ⭐⭐⭐⭐ 对多传感器自动驾驶系统有直接应用价值，三模态基线具有参考意义

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network](../../CVPR2025/autonomous_driving/rc-autocalib_an_end-to-end_radar-camera_automatic_calibration_network.md)
- [FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)

<!-- RELATED:END -->
