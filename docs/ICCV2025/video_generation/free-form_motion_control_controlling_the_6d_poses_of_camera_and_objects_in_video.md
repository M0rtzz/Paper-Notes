---
title: >-
  [论文解读] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation
description: >-
  [ICCV 2025][视频生成][6D位姿控制] 提出 SynFMC 合成数据集（首个包含相机和物体完整 6D 位姿标注的视频数据集）和 FMC 方法，实现了在文本到视频生成中独立或同时控制相机和物体的 6D 位姿，在多种场景下生成高保真视频，且兼容多种个性化 T2I 模型。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "6D位姿控制"
  - "相机运动"
  - "物体运动"
  - "合成数据集"
  - "文本到视频"
---

# Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation

**会议**: ICCV 2025  
**arXiv**: [2501.01425](https://arxiv.org/abs/2501.01425)  
**代码**: [https://henghuiding.com/SynFMC/](https://henghuiding.com/SynFMC/)  
**领域**: 视频生成  
**关键词**: 视频生成, 6D位姿控制, 相机运动, 物体运动, 合成数据集, 文本到视频

## 一句话总结

提出 SynFMC 合成数据集（首个包含相机和物体完整 6D 位姿标注的视频数据集）和 FMC 方法，实现了在文本到视频生成中独立或同时控制相机和物体的 6D 位姿，在多种场景下生成高保真视频，且兼容多种个性化 T2I 模型。

## 研究背景与动机

**运动控制的核心困难是什么？** 在视频生成中精确控制动态物体和相机运动是一个有意义但极具挑战性的任务（如电影中导演精心编排演员动作和相机轨迹），但现有方法面临两大根本限制：

**缺乏完整 6D 位姿标注的数据集**：
   - 现有物体运动数据集（VideoHD、DragNUWA）仅提供图像空间 2D 轨迹，无法区分"物体右移"还是"相机左移"
   - 用于相机运动的数据集（RealEstate10K、MVImgNet）主要是静态场景，缺乏动态物体
   - 少量合成数据集（360°-Motion、HumanVid-Syn）要么仅限静态相机，要么仅限人体动作，运动多样性不足

**缺乏能同时控制物体和相机 6D 位姿的方法**：
   - CameraCtrl 只能控制相机运动；Motion-Zero 仅控制物体但不具备 3D 感知（如朝向）
   - MotionCtrl 虽然分别训练了相机和物体模块，但因缺乏完整 6D 标注，无法实现同步逼真控制
   - 图像空间轨迹方法本质上将物体和相机运动耦合在一起

## 方法详解

### 整体框架

系统分为三个核心部分：SynFMC 数据集构建、Camera Motion Controller (CMC)、Object Motion Controller (OMC)，采用三阶段训练策略。

### SynFMC 数据集

基于 Unreal Engine 构建的合成数据集，包含 62K 视频，分为四组：

- **15K 静态单物** / **15K 静态多物**（物体位置固定，相机可动）
- **16K 动态单物** / **16K 动态多物**（物体和相机均可动）

**物体运动设计**：基于 Bézier 曲线设计轨迹，旋转从切线和法向量导出；控制点根据物体速度属性约束。运动类型包括静止、水平移动、非水平移动、原地运动等。

**相机运动设计**：分解为三个独立维度 —— 视角（前/后/左/右/顶）、距离（拉近/拉远/静止）、高度（上/下/静止），支持细粒度组合控制。

**资产标注**：利用 InternVL + 人工标注为物体标注类别、栖息环境、速度、大小等属性，确保生成运动的合理性。

### FMC 方法

基于 AnimateDiff V3，三阶段训练：

**Stage 1: Domain LoRA**  
向空间模块注入 LoRA 适配器，在合成数据的随机帧上训练。目的是弥合合成数据与真实数据的域差异。推理时丢弃 LoRA 以保持基础模型质量。

**Stage 2: Camera Motion Controller (CMC)**  
由 Camera Encoder（处理 Plücker 嵌入）和 Camera Adapter（调制时序模块特征）组成。使用相机损失 $L_{cam}$ 强调背景区域：

$$L_{cam} = E[\mathcal{M}_{bg}\|\varepsilon_{\theta,\theta_c}(\mathbf{z}_t^{1:N}, t, \mathbf{C}_p, \mathcal{C}_{RT}) - \epsilon\|^2 + \lambda_c\|\varepsilon_{\theta,\theta_c}(\cdot) - \epsilon\|^2]$$

其中 $\mathcal{M}_{bg}$ 为背景掩码，$\lambda_c = 0.6$。由于背景动态仅受相机运动影响，聚焦背景可更精确地学习相机控制。

**Stage 3: Object Motion Controller (OMC)**  
接收物体 6D 位姿信息，在对应物体区域内复制相对相机的位姿特征。使用高斯模糊核（以物体质心为中心）替代精确掩码，避免用户需要提供精确分割。物体损失 $L_{obj}$ 强调前景区域：

$$L_{obj} = E[\mathcal{M}_{fg}\|\varepsilon_{\theta,\theta_c,\theta_o}(\cdot) - \epsilon\|^2 + \lambda_o\|\varepsilon_{\theta,\theta_c,\theta_o}(\cdot) - \epsilon\|^2]$$

其中 $\lambda_o = 0.3$。OMC 输出乘以粗糙掩码后加到空间特征上，防止影响背景。

### 用户接口

提供两种交互方式：用户直接绘制 3D 曲线指定轨迹；或指定运动类型后由规则算法自动生成轨迹。

## 实验关键数据

### 主实验：定量对比

| 方法 | FID↓ | FVD↓ | CLIPSIM↑ | CamTransErr↓ | CamRotErr↓ | ObjTransErr↓ | ObjRotErr↓ |
|------|------|------|----------|--------------|------------|--------------|------------|
| AnimateDiff | 149.61 | 868.97 | 29.33 | - | - | - | - |
| CameraCtrl | 137.96 | 805.25 | 29.21 | 18.16 | **0.94** | - | - |
| MotionCtrl | 125.52 | 952.31 | 26.83 | **17.84** | 1.11 | 80.66 | 1.77 |
| **FMC (ours)** | 133.42 | 846.51 | **31.01** | 18.12 | 1.03 | **42.25** | **0.96** |

FMC 在物体运动控制上大幅领先 MotionCtrl（ObjTransErr: 42.25 vs 80.66，ObjRotErr: 0.96 vs 1.77），相机控制与专门方法持平。

### 消融实验

| 设置 | CamTransErr | CamRotErr | ObjTransErr | ObjRotErr |
|------|-------------|-----------|-------------|-----------|
| MotionCtrl (w/o $\mathcal{C}_{RT}$) | 18.24 | 1.08 | 78.82 | 1.65 |
| MotionCtrl (w/ $\mathcal{C}_{RT}$) | 18.24 | 1.08 | 55.33 | 1.26 |
| FMC (w/o $L_{cam}$) | 20.35 | 1.19 | - | - |
| FMC (w/o $L_{obj}$) | 18.12 | 1.03 | 46.62 | 1.15 |
| **FMC (完整)** | **18.12** | **1.03** | **42.25** | **0.96** |

**关键发现**：
- 在 SynFMC 上训练 MotionCtrl 并提供相机位姿，ObjTransErr 从 78.82 降至 55.33，证明完整位姿标注的价值
- 去除 $L_{cam}$ 导致相机误差显著增大，模型倾向于移动前景物体来模拟相对运动
- 去除 $L_{obj}$ 导致物体外观质量下降

### 用户研究

| 方法 | 质量 | 文本相似度 | 相机运动 | 物体运动 |
|------|------|-----------|---------|---------|
| CameraCtrl | 0.88 | 0.84 | 0.95 | - |
| MotionCtrl | 0.89 | 0.81 | 0.93 | 0.53 |
| **FMC** | **0.91** | **0.95** | **0.95** | **0.98** |

物体运动评分差距显著（0.98 vs 0.53），验证了 3D 感知控制的优势。

## 亮点与洞察

1. **解耦控制的关键在于数据**：完整 6D 位姿标注使模型能够学会区分全局（相机）和局部（物体）动态，这是图像空间轨迹无法实现的
2. **规则生成算法的巧妙设计**：将相机运动分解为视角/距离/高度三个独立维度，支持复杂电影镜头的可控生成
3. **Domain LoRA 策略实用**：训练时使用、推理时丢弃，既利用合成数据学习运动控制，又保持真实感
4. **粗糙掩码替代精确分割**：高斯模糊核的设计降低了用户输入门槛，是非常工程友好的做法

## 局限性

- 多物体复杂运动的控制能力仍有限
- 物体运动评估指标需要改进（当前依赖深度估计推算全局位置）
- 基于 U-Net 架构（AnimateDiff V3），未扩展到更新的 DiT 架构
- 缺乏参考图像输入，无法为特定主体定制运动视频

## 相关工作与启发

- **CameraCtrl**：仅控制相机运动的代表方法
- **MotionCtrl**：最接近的工作，分别训练相机和物体模块但无法有效同步
- **3DTrajMaster**：提供物体 6D 位姿但仅限静态相机
- **启发**：合成数据 + 域适应的范式可推广到其他需要精确物理标注的视频生成任务（如手部操作、机器人规划等）

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 首个同时提供相机和物体完整 6D 位姿的数据集 + 控制方法
- 实验充分度：⭐⭐⭐⭐ — 独立/联合控制对比、消融、用户研究、多风格适配
- 实用价值：⭐⭐⭐⭐ — 电影制作/游戏/AR 等应用场景明确
- 写作质量：⭐⭐⭐⭐ — 图表丰富，方法叙述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](../../CVPR2025/video_generation/motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[CVPR 2025\] Dynamic Camera Poses and Where to Find Them](../../CVPR2025/video_generation/dynamic_camera_poses_and_where_to_find_them.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)

</div>

<!-- RELATED:END -->
