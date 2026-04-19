---
title: "Dual Exposure Stereo for Extended Dynamic Range 3D Imaging"
conference: "CVPR 2025"
arxiv: "2412.02351"
code: ""
project: ""
领域: "三维视觉"
关键词: ["立体匹配", "高动态范围", "双曝光", "自动曝光控制", "深度估计"]
评分: "⭐⭐⭐⭐"
---

# Dual Exposure Stereo for Extended Dynamic Range 3D Imaging

## 一句话总结

提出双曝光立体成像(Dual-Exposure Stereo)方法，通过自动双曝光控制(ADEC)在交替帧中使用不同曝光，结合运动感知的双曝光特征融合网络进行视差估计，将立体相机的有效动态范围扩展至 160%，实现极端光照条件下的鲁棒 3D 成像。

## 研究背景与动机

在自动驾驶、机器人导航等场景中，鲁棒的立体 3D 成像至关重要。然而，现实世界的光照动态范围(DR)远超常规相机的捕获能力(8-bit 相机仅 42dB)，导致过曝和欠曝区域的深度估计严重退化。

现有方法的局限：
- **单曝光 AEC**：仅调整单一曝光值来适应场景，无法扩展相机本身的动态范围
- **曝光包围(Exposure Bracketing)**：使用预定义的多曝光设置，不能自适应场景动态范围变化，且增加捕获时间
- **立体相机独立曝光**：左右相机使用不同曝光会破坏立体一致性，影响匹配质量

本文的核心洞察是：通过在**时间轴**上交替使用不同曝光（而非空间上左右不同），既保持了立体图像的亮度一致性，又通过双帧互补实现了动态范围的有效扩展。

## 方法详解

### 整体框架

系统包含两大核心组件：(1) 自动双曝光控制(ADEC)模块，基于直方图统计自适应调整交替帧的曝光值；(2) 运动感知双曝光视差估计网络，融合双曝光帧的特征并补偿帧间运动进行深度估计。

### 关键设计

#### 1. 自动双曝光控制 (ADEC)

- **功能**：根据场景动态范围自适应调整双曝光值 $(e_1, e_2)$，在场景 DR 超过相机 DR 时发散曝光，在场景 DR 可控时收敛曝光
- **核心思路**：利用直方图偏度 $S_i$ 和极端像素比例 $L_i, H_i$ 作为统计指标。当 $L_i > \tau_h$ 且 $H_i > \tau_h$（同时存在大量过曝和欠曝像素）时判定场景 DR 超过相机 DR，按比例发散双曝光；否则调整曝光使偏度趋近零
- **设计动机**：联合 AEC 和 exposure bracketing 的优势——自适应调整+扩展 DR。通过限制曝光差 $\Delta e < \tau_{\Delta e}$ 防止过度发散导致不稳定。运行速度 >120 FPS，支持实时应用

#### 2. 运动感知双曝光特征融合

- **功能**：将交替帧中不同曝光的特征在空间上对齐并融合，生成包含宽动态范围信息的统一特征图
- **核心思路**：(1) 用预训练光流网络估计帧间运动；(2) 将第二帧特征 warp 到第一帧视角；(3) 使用基于亮度的梯形权重函数进行加权融合，良好曝光区域权重高，过/欠曝区域权重低
- **设计动机**：双曝光帧之间存在时间差导致的运动位移，必须补偿才能有效融合。梯形权重函数 $W_i^c$ 根据像素亮度自动分配融合权重（阈值 $\alpha=0.02, \beta=0.98$），充分利用每帧的有效信息

#### 3. 基于融合特征的立体视差估计

- **功能**：从融合后的左右特征图构建相关体积(correlation volume)并估计视差图
- **核心思路**：融合后的特征 $\hat{F}^{\text{left}}, \hat{F}^{\text{right}}$ 编码了宽动态范围信息，标准相关体积 $C(x,y,d) = \hat{F}^{\text{left}}(x,y) \cdot \hat{F}^{\text{right}}(x+d, y)$ 即可捕获明暗区域的匹配信息。采用多尺度特征融合增强鲁棒性
- **设计动机**：核心思想是在特征层面而非图像层面进行 HDR 融合，避免 tone mapping 等预处理引入的信息损失

### 损失函数

使用 RAFT-Stereo 的标准视差估计损失对融合后的视差图进行监督训练，在合成数据集上微调。

## 实验关键数据

### 主实验表

| 方法 | 合成数据 Disp MAE (px) ↓ | 真实数据 Depth MAE (m) ↓ | FPS ↑ |
|------|-------------------------|-------------------------|-------|
| AverageAE | 2.823 | 2.7679 | 616.27 |
| GradientAE | 2.948 | 2.5847 | 42.10 |
| NeuralAE | 2.778 | 1.9232 | **0.25** |
| **ADEC (ours)** | **1.355** | **1.9142** | 124.58 |

### 消融表

| ADEC | 加权融合 | 运动补偿 | Disp MAE (px) ↓ |
|------|---------|---------|-----------------|
| ✗ | ✓ | ✓ | 6.2775 |
| ✓ | ✗ | ✓ | 3.3968 |
| ✓ | ✓ | ✗ | 8.3657 |
| ✓ | ✓ | ✓ | **2.9010** |

### 关键发现

- **DR 扩展率**：在 160% DR 扩展率下仍保持高深度精度，其他 AEC 方法无法扩展 DR
- **运动补偿最关键**：消融实验显示移除运动补偿导致误差最大(8.37 vs 2.90)，帧间运动对齐是双曝光方法的瓶颈
- **ADEC 模块次关键**：固定双曝光(移除 ADEC)误差 6.28，自适应调整带来显著提升
- **实时性**：ADEC 运行 124 FPS，远优于 NeuralAE 的 0.25 FPS，且深度精度相当

## 亮点与洞察

1. **问题定义精准**：将 DR 扩展问题拆解为曝光控制+特征融合+运动补偿三个子问题，每个都有简洁有效的解决方案
2. **AEC + Bracketing 的优雅结合**：时间维度交替曝光保持了左右一致性同时扩展 DR，是两种经典方法的自然统一
3. **完整系统**：从硬件(机器人视觉系统)到算法(ADEC + 网络)到数据集(真实 + 合成)的全链路贡献
4. **实用性强**：120+ FPS 的 ADEC 适合实时系统，方法可应用于任意位深的相机

## 局限与展望

1. **双帧时间延迟**：交替曝光引入帧间运动，快速运动场景(如高速驾驶)可能导致对齐失败
2. **曝光差限制**：$\tau_{\Delta e}=2.5$ 限制了极端场景下的 DR 扩展能力
3. **依赖光流估计**：光流网络的精度直接影响融合质量，在极端曝光差下光流估计本身可能失效
4. **仅验证立体视差**：未探索对单目深度估计、多视图重建等其他 3D 任务的推广
5. **合成数据与真实的域差距**：网络在 CARLA 合成数据上训练，真实场景泛化有待进一步验证

## 相关工作与启发

- **RAFT-Stereo**：视差估计的骨干网络，本文在其上添加双曝光融合模块
- **HDR 3D 成像**：使用事件相机、SPAD 等非常规传感器的方案硬件成本高，本文用普通相机实现 DR 扩展
- **启发**：在传感器物理限制和算法补偿之间寻找最优平衡——不需要昂贵的 HDR 传感器，通过简单的曝光策略+网络融合即可扩展有效 DR

## 评分

⭐⭐⭐⭐

问题实际重要且定义清晰，系统设计完整（硬件+算法+数据集），方法简洁有效。消融实验充分验证了每个组件的必要性。

<!-- RELATED:START -->

## 相关论文

- [High Dynamic Range Novel View Synthesis with Single Exposure](../../ICML2025/3d_vision/high_dynamic_range_novel_view_synthesis_with_single_exposure.md)
- [InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](../../ICLR2026/3d_vision/mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)
- [Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)

<!-- RELATED:END -->
