---
title: >-
  [论文解读] MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond
description: >-
  [CVPR 2025][3D视觉][human motion capture] 构建大规模压力-RGB-光学动捕数据集 MotionPRO（70人/400类动作/12.4M帧），并提出 FRAPPE 基线将压力信号与单目 RGB 融合，显著提升全身姿态估计的物理合理性和全局轨迹精度，进一步将压力先验扩展至人形机器人驱动。
tags:
  - CVPR 2025
  - 3D视觉
  - human motion capture
  - pressure sensing
  - 多模态
  - global trajectory
  - humanoid robot
---

# MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond

**会议**: CVPR 2025  
**arXiv**: [2504.05046](https://arxiv.org/abs/2504.05046)  
**代码**: [项目页](https://nju-cite-mocaphumanoid.github.io/MotionPRO/)  
**领域**: 3d_vision  
**关键词**: human motion capture, pressure sensing, multi-modal fusion, global trajectory, humanoid robot

## 一句话总结

构建大规模压力-RGB-光学动捕数据集 MotionPRO（70人/400类动作/12.4M帧），并提出 FRAPPE 基线将压力信号与单目 RGB 融合，显著提升全身姿态估计的物理合理性和全局轨迹精度，进一步将压力先验扩展至人形机器人驱动。

## 研究背景与动机

**领域现状**: 现有人体动捕方法主要关注视觉几何相似性，依赖 RGB 图像进行姿态估计，在局部姿态精度上已取得较好成绩。

**现有痛点**: 当用于驱动 3D 场景中的虚拟人或真实世界人形机器人时，现有方法存在严重的动态不准确性：时间维度的漂移和抖动、空间维度的滑步、浮空和穿透。这些问题根源在于忽略了人体与场景的**物理交互**。

**核心矛盾**: RGB 缺乏深度信息，无法捕捉人体与地面的接触关系和动态物理交互；已有压力数据集要么限于卧姿、要么动作类型单一（瑜伽/太极）、规模过小。

**本文目标**: (1) 构建大规模全身压力动捕数据集；(2) 验证压力信号对姿态估计的必要性和有效性；(3) 探索压力在人形机器人驱动中的作用。

**切入角度**: 从人体与物理世界的交互视角重新审视动捕问题，压力信号编码了地面对人体的支撑力，包含丰富的动态机制和物理信息。

**核心 idea**: 压力信号提供了 RGB 无法获取的物理交互先验，融合两者可实现精确且物理合理的全身动捕。

## 方法详解

### 整体框架

系统包含两大部分：(1) MotionPRO 数据集构建——同步采集 4 视角 RGB（30Hz）、120×160 全身压力垫（100Hz→30Hz）、50 标记点光学动捕（120Hz→30Hz），通过 Mosh++ 获取 SMPL 标注；(2) 两个基线任务——仅压力姿态估计和压力-RGB 融合姿态估计（FRAPPE）。

### 关键设计

#### 1. 仅压力的姿态与轨迹估计

压力信号高度稀疏（日常动作中仅脚部接触地面），单帧压力对应成千上万种可能姿态。网络由三部分组成：
- **小核压力编码器**: 缩小 ResNet 卷积核尺寸，提取更精细的稀疏压力特征
- **长短期注意力模块（LSAM）**: GRU 提取短期上下文动作，多头自注意力提取长期依赖
- **人体姿态回归器**: 预测 SMPL 姿态参数和全局平移

#### 2. FRAPPE: 压力-RGB 融合

在仅压力基线上增加 RGB 分支和融合交叉注意力模块（FCAM）：
- RGB 分支使用冻结预训练 HRNet 作为图像编码器
- FCAM 以压力特征为 Query、图像特征为 Key/Value（压力包含更多真实物理世界信息）
- 采用**正交投影**而非弱透视投影，保持深度方向尺度不变，避免 2D 图像对 3D 轨迹的误导

#### 3. 全身接触标注策略

对 SMPL 的每个关节，垂直投影到地面计算邻域压力和 $P_j$ 和距地面距离 $D_j$，当 $P_j \geq \tau_1$ 且 $D_j \leq \tau_2$ 时标注为接触状态。

### 损失函数

仅压力网络: $\mathcal{L} = \lambda_{pose}\mathcal{L}_{pose} + \lambda_{3d}\mathcal{L}_{3d} + \lambda_{trans}\mathcal{L}_{trans} + \lambda_{contact}\mathcal{L}_{contact}$

FRAPPE 在此基础上增加正交投影 2D 对齐损失 $\mathcal{L}_{2d}$，约束相机轴方向的正交投影一致性。

## 实验关键数据

### 主实验：全局姿态估计

| 方法 | MPJPE↓ | PMPJPE↓ | PVE↓ | Accel↓ |
|------|--------|---------|------|--------|
| VIBE | 59.7 | 40.9 | 82.9 | 19.6 |
| CLIFF | 54.7 | 39.7 | 68.6 | 24.3 |
| WHAM | 160.4 | 28.3 | 227.5 | 2.9 |
| **FRAPPE** | **41.8** | **30.2** | **58.6** | **3.0** |

### 全局轨迹估计

| 方法 | WMPJPE↓ | RTE↓ | Jitter↓ | WBCE↓ |
|------|---------|------|---------|-------|
| TRACE | 141.2 | 1193 | 68.6 | 10272 |
| WHAM | 75.6 | 1023 | 9.2 | 1218 |
| **FRAPPE** | **60.8** | **41.6** | **6.0** | **110.2** |

FRAPPE 在 RTE 上较 WHAM 降低 **96%**，WBCE 降低 **91%**。

### 消融实验

| 变体 | GTraj↓ | GMPJPE↓ | MPJPE↓ |
|------|--------|---------|--------|
| w/o LSAM | 93.1 | 92.9 | 44.4 |
| w/o FCAM | 66.0 | 70.7 | 39.5 |
| w/o contact loss | 68.7 | 75.9 | 42.6 |
| w/o 2d loss | 82.1 | 95.5 | 52.2 |
| **Full** | **62.2** | **68.6** | **40.5** |

### 关键发现

1. 仅压力即可提供准确的全局轨迹和合理的下肢姿态（下肢 PMPJPE 32.4mm）
2. 去掉 2D loss 后 Jitter 和 FS 反而更好，说明 2D/3D loss 之间需要权衡
3. 在人形机器人驱动中，FRAPPE 的 CoM/CoP 偏差（15.32/22.35mm）远低于纯 RGB 方法

## 亮点与洞察

- **数据集规模领先**: 70人/400动作类/12.4M帧，远超 MoYo（1人/82动作）等前作
- **物理交互视角**: 把动捕问题从纯视觉问题重新定义为人-场景物理交互问题
- **正交投影的洞察**: 发现弱透视投影在关注全局轨迹时会导致姿态-轨迹耦合，而正交投影在深度方向保持尺度
- **从虚拟人到真实机器人**: 完整验证了压力先验从 SMPL 驱动到 NAO 机器人驱动的可行性
- **接触标注方法**: 结合压力和与地面距离的双阈值策略，得到全身接触标注

## 局限性

1. 压力垫限制了动作范围，所有运动必须在垫上完成
2. 数据集缺乏户外场景和非平面地面情况
3. FRAPPE 对相机内参已知的假设在野外场景中受限
4. 压力垫分辨率（120×160）可能对精细足部动作不足

## 相关工作与启发

- **MoYo** 首次引入压力质心（CoP）约束，但仅对准静态动作有效；MotionPRO 将全身接触扩展到动态动作
- **WHAM** 使用仅基于脚部速度计算的接触标签，精度低且忽略其他身体部位接触
- **PIMesh** 利用多帧压力信息实现高精度卧姿估计，但无法泛化到日常运动
- **启发**: 压力信号作为"物理传感器"的范式可扩展到其他交互传感（力矩、IMU），为具身智能提供更丰富的环境交互信息

## 评分

⭐⭐⭐⭐ — 数据集工程量大、实验全面（虚拟人+真实机器人），洞察有价值；方法侧偏基线设计较简单，可在融合机制上进一步深入。

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[CVPR 2025\] Gaussian Eigen Models for Human Heads](gaussian_eigen_models_for_human_heads.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)
- [\[NeurIPS 2025\] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](../../NeurIPS2025/3d_vision/gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)
- [\[NeurIPS 2025\] Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](../../NeurIPS2025/3d_vision/cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)

<!-- RELATED:END -->
