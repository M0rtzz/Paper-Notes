---
title: >-
  [论文解读] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image
description: >-
  [ICCV 2025][3D视觉][运动模糊] 本文将运动模糊从"不需要的伪影"转变为"有价值的运动线索"，通过从单张模糊图像预测稠密光流场和单目深度图，再用可微分最小二乘求解器恢复相机6DoF瞬时速度，实现媲美甚至超越IMU的运动估计精度和30FPS实时性能。
tags:
  - ICCV 2025
  - 3D视觉
  - 运动模糊
  - 相机运动估计
  - 6DoF速度估计
  - 单图像运动估计
  - IMU替代
---

# Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image

**会议**: ICCV 2025  
**arXiv**: [2503.17358](https://arxiv.org/abs/2503.17358)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 运动模糊, 相机运动估计, 6DoF速度估计, 单图像运动估计, IMU替代

## 一句话总结

本文将运动模糊从"不需要的伪影"转变为"有价值的运动线索"，通过从单张模糊图像预测稠密光流场和单目深度图，再用可微分最小二乘求解器恢复相机6DoF瞬时速度，实现媲美甚至超越IMU的运动估计精度和30FPS实时性能。

## 研究背景与动机

相机运动估计是3D重建、SLAM、VR/AR的基础。传统的VO/SfM方法假设相机在曝光期间近似静止，把每帧图像当作场景快照，通过帧间匹配计算相对位姿。但在快速运动中，这个假设完全失效——运动模糊严重破坏特征匹配，导致COLMAP、ORB-SLAM等方法彻底崩溃。

**现有方案的局限**：
- **丢弃模糊帧**：损失信息，在持续快速运动时可用帧很少
- **引入IMU**：增加硬件成本，传感器同步困难，且IMU存在积分漂移问题——仅20秒后速度估计就会明显偏移
- **学习型方法**（DUSt3R、MASt3R等）：虽然在稀疏视角上更鲁棒，但仍需多帧输入，且面对严重运动模糊时特征匹配退化

**革命性的思路转变**：运动模糊包含了相机在曝光期间运动的直接编码——模糊条纹的方向和长度反映了相机的运动方向和速度。如果能"解读"这些模糊轨迹，就能从单张图片恢复相机的瞬时运动。本质上，这使得相机可以像IMU一样提供运动测量——但不需要额外传感器，不会漂移，且直接给出速度而非需要积分的加速度。

**关键insight**：一张运动模糊的图像可以看作是曝光期间多张"虚拟图像"的叠加。模糊轨迹提供了首帧虚拟图像和末帧虚拟图像之间的"虚拟对应关系"，本质上等价于光流场。结合深度估计，就能从像素运动恢复3D空间中的相机运动。

## 方法详解

### 整体框架

方法分为两个阶段：(1) 网络阶段——从单张模糊图像预测运动光流场 $\mathcal{F}$ 和单目深度图 $\mathcal{D}$；(2) 求解阶段——将光流和深度代入运动场方程，通过可微分最小二乘求解器计算相机的6DoF瞬时速度 $(v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)$。

输入仅需单张运动模糊图像（+已知焦距和曝光时间），输出相机在曝光期间的瞬时线速度和角速度。

### 关键设计

1. **光流与深度预测网络**:

    - 功能：从模糊图像同时预测像素级运动光流场和度量深度图
    - 核心思路：使用SegNeXt作为共享backbone编码器，分别接两个解码器输出光流 $\mathcal{F} \in \mathbb{R}^{2 \times H \times W}$ 和深度 $\mathcal{D} \in \mathbb{R}^{1 \times H \times W}$。光流定义为第一帧虚拟图像到最后一帧的像素位移
    - 关键设计——**重定向函数** $h_f$：由于从模糊图像中无法确定运动的起始方向（左移和右移产生相同的水平模糊），训练时使用重定向函数选择与预测方向一致的GT标签：
    $h_f(\hat{\mathcal{F}}_{fw}, \hat{\mathcal{F}}_{bw}; \mathcal{F}) = \begin{cases} \hat{\mathcal{F}}_{fw} & \text{if } \langle\hat{\mathcal{F}}_{fw}, \mathcal{F}\rangle > \langle\hat{\mathcal{F}}_{bw}, \mathcal{F}\rangle \\ \hat{\mathcal{F}}_{bw} & \text{otherwise} \end{cases}$
    - 训练损失：$\mathcal{L}_1 = \lambda_F\|\mathcal{F} - h_f(\hat{\mathcal{F}}_{fw}, \hat{\mathcal{F}}_{bw})\| + \lambda_D\|\mathcal{D} - \hat{\mathcal{D}}\|$

2. **可微分速度求解器**:

    - 功能：从光流场和深度图恢复相机的平移和旋转参数
    - 核心思路：利用经典的运动场方程将每个像素的光流分解为平移和旋转分量：
    $F_x = \frac{t_z p_x - t_x f}{d} - \theta_y f + \theta_z p_y + \frac{\theta_x p_x p_y}{f} - \frac{\theta_y(p_x)^2}{f}$
    $F_y = \frac{t_z p_y - t_y f}{d} + \theta_x f - \theta_z p_x - \frac{\theta_y p_x p_y}{f} + \frac{\theta_x(p_y)^2}{f}$
   将所有像素组成超定线性方程组 $\bm{A}\bm{x}=\bm{b}$，用最小二乘求解：$\bm{x} = (\bm{A}^\top\bm{A})^{-1}\bm{A}^\top\bm{b}$
    - 设计动机：最小二乘求解器完全可微分，因此可以端到端地用位姿监督来训练整个网络。端到端损失：
    $\mathcal{L}_2 = \lambda_R\|R - h_p(\hat{R})\|_2 + \lambda_t\|\bm{t} - h_p(\hat{\bm{t}})\|_2 + \mathcal{L}_1$

3. **方向消歧**:

    - 功能：解决模糊导致的180°方向歧义（左移和右移产生相同的模糊）
    - 核心思路：利用视频中的相邻帧，将当前帧按正向和反向光流分别warp，比较与前后帧的光度误差来判断正确方向：
    $e_{fw} = \mathcal{P}(I_{i+1}, I'_{i,fw}) + \mathcal{P}(I_{i-1}, I'_{i,bw})$
   选择光度误差更小的方向作为最终运动方向
    - 设计动机：这是一个推理时的后处理启发式方法，利用视频时序信息消除固有的180°歧义

### 损失函数 / 训练策略

三阶段训练：
1. 仅在合成数据上训练光流和深度解码器（无位姿监督），batch size 32
2. 加入位姿监督端到端训练，batch size 8，300K步
3. 在真实世界运动模糊数据上微调10K步（利用可微分pipeline）

合成数据集基于ScanNet++v2的150个序列生成约120K训练样本。用RIFE帧插值网络在真实帧之间插值生成虚拟帧，然后在线性空间中平均得到合成模糊图像。深度GT使用PromptDA从ARKit低分辨率深度上采样获得。

## 实验关键数据

### 主实验（真实世界4个场景角速度RMSE，rad/s）

| 方法 | 输入 | billiards | commonroom | dining | office | 平均 |
|------|------|-----------|------------|--------|--------|------|
| COLMAP | 多帧 | ×失败 | ×失败 | ×失败 | ×失败 | - |
| MASt3R | 双帧 | 5.30/2.85/4.45 | 3.70/3.75/3.26 | 2.36/0.84/1.67 | 4.78/3.03/6.21 | 4.04/2.62/3.90 |
| DROID-SLAM | 多帧 | 5.39/3.33/5.31 | 3.01/5.89/3.57 | 2.92/1.20/1.98 | 6.33/4.90/5.56 | 4.41/3.83/4.10 |
| **Ours** | **单帧** | **1.31/0.87/1.60** | **0.93/0.88/1.04** | **0.87/0.50/1.33** | **1.76/1.38/3.08** | **1.22/0.91/1.76** |

### 消融实验（平移速度RMSE对比，m/s）

| 方法 | 输入 | 平均 $v_x / v_y / v_z$ | 说明 |
|------|------|----------------------|------|
| COLMAP(SIFT) | 多帧 | 完全失败 | 运动模糊导致特征匹配崩溃 |
| COLMAP(D+LG) | 多帧 | -（部分失败） | 学习型特征稍鲁棒但仍不稳定 |
| MASt3R | 双帧 | 1.60/1.54/2.17 | 学习型方法，稀疏视角下较鲁棒 |
| DROID-SLAM | 多帧 | 2.23/1.63/1.51 | 基于光流的SLAM |
| **Ours** | **单帧** | **1.11/1.03/0.92** | 平移速度RMSE降低24% |
| 零速度基线 | - | 2.01/1.61/1.24 | 假设相机静止的下界 |

### 关键发现

- COLMAP在有严重运动模糊的序列中完全失败，连位姿都重建不出来，凸显了问题的重要性
- 本方法仅用单帧输入就超过所有需要多帧的基线方法：角速度RMSE降低31%，平移速度降低24%
- 30FPS实时运行速度（RTX 3090），比MASt3R（2.83 FPS）快10倍以上
- 与真实IMU对比：IMU积分20秒后出现明显漂移，而本方法全程稳定无漂移
- 在机器人手臂实验中成功估计末端执行器速度，展示了实际应用潜力

## 亮点与洞察

- **思维模式的根本转变**：把运动模糊从"敌人"变为"朋友"，将通常被丢弃的模糊帧变成运动传感器。这种"化弊为利"的思路非常启发人
- **全可微分pipeline**：从光流/深度预测到最小二乘求解全程可微，允许用真实数据端到端微调，闭合合成到真实的gap
- **无漂移**：与IMU的积分漂移不同，本方法每帧独立估计速度，天然无漂移
- **数据合成pipeline**：展示了如何从标准视觉数据集（ScanNet++v2）生成带GT的模糊训练数据，方法论值得借鉴
- **优雅的方向消歧**：利用视频时序和光度一致性的简单启发式方法解决180°歧义

## 局限与展望

- 假设场景是刚体的，不处理rolling shutter和曝光内非均匀运动（虽然实验显示对此有一定鲁棒性）
- 训练数据限于室内场景，室外场景的推广能力未验证
- 需要已知曝光时间和焦距
- 存在180°方向歧义，需要相邻帧辅助消歧，纯单帧场景无法消歧
- 在translation估计中可能受深度估计精度的制约

## 相关工作与启发

- **Klein & Murray (2005)**：从单张模糊图像估计旋转的开创性工作，但仅处理纯旋转
- **MBA-VO (Liu et al., 2021)**：考虑模糊的VO方法，但需要双帧+去模糊步骤
- **MASt3R/DUSt3R**：当前学习型3D视觉的SOTA，但不专门处理模糊场景
- 启发：传感器融合中可以将运动模糊估计作为IMU的替代或互补来源，特别适合不方便安装IMU的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (从模糊中提取运动的思路具有范式转变意义)
- 实验充分度: ⭐⭐⭐⭐ (真实世界评估+IMU对比+机器人应用，但数据集规模较小)
- 写作质量: ⭐⭐⭐⭐⭐ (motivation清晰，pipeline紧凑，图表直观)
- 价值: ⭐⭐⭐⭐⭐ (实时+无漂移+单帧，对SLAM/VR有重大实用价值)

<!-- RELATED:START -->

## 相关论文

- [CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)
- [Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)
- [Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)

<!-- RELATED:END -->
