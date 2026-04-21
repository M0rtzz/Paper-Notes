---
title: >-
  [论文解读] Physical Simulator In-the-Loop Video Generation
description: >-
  [CVPR 2026][物理模拟器在环] 提出PSIVG——首个将物理模拟器嵌入视频扩散生成循环的训练-free推理时框架：从模板视频中重建4D场景和物体网格，在MPM模拟器中生成物理一致轨迹，用光流引导视频生成，并通过TTCO测试时优化保证运动物体纹理一致性，用户偏好率达82.3%。
tags:
  - CVPR 2026
  - 物理模拟器在环
  - 视频扩散模型
  - MPM模拟
  - 测试时优化
  - 物理一致生成
---

# Physical Simulator In-the-Loop Video Generation

**会议**: CVPR 2026  
**arXiv**: [2603.06408](https://arxiv.org/abs/2603.06408)  
**代码**: https://vcai.mpi-inf.mpg.de/projects/PSIVG  
**领域**: 视频生成 / 物理一致性  
**关键词**: 物理模拟器在环, 视频扩散模型, MPM模拟, 测试时优化, 物理一致生成

## 一句话总结
提出PSIVG——首个将物理模拟器嵌入视频扩散生成循环的训练-free推理时框架：从模板视频中重建4D场景和物体网格，在MPM模拟器中生成物理一致轨迹，用光流引导视频生成，并通过TTCO测试时优化保证运动物体纹理一致性，用户偏好率达82.3%。

## 研究背景与动机

**领域现状**：扩散视频生成模型（CogVideoX、HunyuanVideo等）已达到出色的视觉真实感，但生成的视频经常违反重力、惯性、碰撞等基本物理定律——物体凭空消失、运动轨迹不合理、物理交互不真实。

**现有痛点**：(1) 现代视频生成模型基于去噪/重建目标训练，本质上优化逐像素/逐patch的重建，缺乏显式物理约束机制；(2) 早期物理感知方法耦合2D刚体模拟器与图像生成器，但受限于简化的2D假设；(3) PhysAnimator等方法专注2D网格模拟卡通动画，PhysGen3D需要输入图像做3D重建；(4) LLM-based提示方法是正交的探索但不直接在生成器中施加物理约束。

**核心矛盾**：视频扩散模型的训练目标（去噪/重建）不包含任何物理约束，没有机制强制学习物理定律。要在保持视觉质量的同时实现物理一致性，需要在生成过程中引入物理引导。

**本文目标** 如何有效地将物理模拟器的信息整合到视频扩散过程中，实现物理一致的视频生成？

**切入角度**：提出"simulation-in-the-loop"范式——物理模拟器作为物理感知约束，在扩散生成循环中引导模型维持时空一致性。

**核心 idea**：先用预训练视频模型生成模板视频，从中重建4D场景和物体网格放入物理模拟器，用模拟器输出的物理一致轨迹引导视频重生成，并通过测试时优化提升纹理一致性。

## 方法详解

### 整体框架
PSIVG是一个多阶段pipeline：(1) 用预训练视频生成器从文本prompt生成**模板视频**（提供场景构成、相机运动、物体外观等，但物理不一致）；(2) **感知管线**从模板视频中提取3D前景物体网格、4D背景场景重建和相机轨迹；(3) 将场景初始化到**MPM物理模拟器**中进行前向模拟，获得物理一致的轨迹；(4) 将模拟器的渲染输出（RGB、分割mask、像素对应关系）作为引导信号，通过光流条件化视频生成；(5) 可选地，通过**TTCO测试时优化**进一步提升运动物体的纹理一致性。

### 关键设计

1. **感知管线（从模板视频到模拟器资产）**:

    - 功能：将2D生成视频转化为模拟器可用的3D资产
    - 核心思路：分三路提取信息——(a) **前景物体几何**：用SAM/GroundedSAM检测分割动态物体，从首帧裁剪物体局部图像，送入InstantMesh做单图像3D网格重建（比多帧重建更可靠，因为视频帧间几何不一致）；(b) **背景场景几何**：用ViPE（mask掉前景后）进行4D重建，用bundle adjustment获取相机轨迹，将逐帧度量深度点图聚合成全局3D背景点云；(c) **前景物体动力学**：选两个关键帧，线性速度=3D位移/Δt，旋转速度通过SuperGlue特征匹配计算相对于质心的2D流场估计
    - 设计动机：模板视频虽然物理不一致，但提供了场景的整体构成信息。通过感知管线桥接生成视频和物理模拟器，是实现"simulation-in-the-loop"的关键环节

2. **物理模拟器场景初始化**:

    - 功能：在MPM模拟器中复现模板视频的场景
    - 核心思路：(a) **模拟域确定**：包围前景动力学范围+背景几何，用空间偏移系数确定合适的模拟立方体[0,2]³，确定metric-to-simulation缩放比例；(b) **物理属性估计**：使用GPT-5从模板视频首帧推断物体的密度、杨氏模量等。采用分层提示——先查询物体材质组成、弹性特征、表面粗糙度，再映射到数值物理参数，比直接估算数值更可靠；(c) **模拟与渲染**：MPM前向模拟获得高分辨率粒子轨迹，用Mitsuba渲染为RGB帧、分割mask和像素对应关系
    - 设计动机：模拟器渲染虽不够逼真（风格人工、缺少光影、可能有网格缺陷），但蕴含忠实的运动物理信息，作为引导信号足矣

3. **物理一致视频生成（光流条件化）**:

    - 功能：用模拟器输出引导视频扩散模型生成物理一致的视频
    - 核心思路：使用Go-with-the-Flow (GwtF)光流条件化视频生成模型。计算混合光流：前景光流来自模拟器渲染的RGB（确保物理一致运动），背景光流来自模板视频（保留场景运动和相机动态），用分割mask融合。光流用于warp噪声latent输入模型
    - 设计动机：直接用模拟器输出做条件输入不够（视觉质量差），光流条件化同时编码轨迹和旋转信息，且易于建模相机运动

4. **TTCO测试时纹理一致性优化**:

    - 功能：在测试时优化可学习参数，使生成视频中运动物体的纹理在帧间一致
    - 核心思路：将模板视频首帧 $\hat{I}_1$ 通过模拟器的像素-像素对应关系warp到每帧，作为纹理一致的目标。优化可学习的零初始化嵌入（加到前景物体对应的文本token上+DiT层的特征调制），使生成视频的像素跟随物理模拟器的前景运动：$\mathcal{L}_{\text{TTCO}} = \sum_t \sum_j \|[De(h_0(\hat{L}_\tau))]_{q_{t,j}} - [W_t(\hat{I}_1)]_{q_{t,j}}\|_2^2$。聚焦于较早（更noisy）的扩散步骤(700-1000)引导纹理生成。50次迭代即可完成
    - 设计动机：光流条件化只能引导运动方向，不保证纹理一致性（旋转/遮挡时可能出现闪烁）。通过前景文本token优化实现局部化适配——只影响前景物体而不破坏背景

### 损失函数 / 训练策略
PSIVG不需要额外训练数据。TTCO在测试时使用AdamW优化器，LR=2e-4，50次迭代。模板视频由SD3生成图像+CogVideoX-I2V-5B或HunyuanVideo-I2V生成。

## 实验关键数据

### 主实验

| 方法类型 | 方法 | SAM mIoU↑ | Corr.Pixel MSE↓ | CLIP Text↑ | Subj. Consis.↑ |
|----------|------|-----------|----------------|------------|----------------|
| Text-based | CogVideoX | 0.47 | 0.032 | 0.34 | 0.93 |
| Text-based | HunyuanVideo | 0.46 | 0.017 | 0.35 | 0.95 |
| Physics | PISA-Seg | 0.50 | 0.012 | 0.35 | 0.95 |
| Controllable | SG-I2V | 0.75 | 0.021 | 0.34 | 0.95 |
| Controllable | MotionClone | 0.68 | 0.019 | 0.35 | 0.87 |
| **Ours** | **PSIVG** | **0.84** | **0.007** | **0.35** | **0.95** |

### 用户研究

| 方法 | 偏好率(%) |
|------|-----------|
| CogVideoX | 7.2 |
| HunyuanVideo | 4.5 |
| PISA-Seg | 2.6 |
| SG-I2V | 2.5 |
| MotionClone | 0.9 |
| **PSIVG (Ours)** | **82.3** |

32名参与者一致认为PSIVG生成的视频物理上最合理。

### 消融实验

| 配置 | SAM mIoU↑ | Corr. Pixel MSE↓ | Subj. Consis.↑ |
|------|-----------|----------------|----------------|
| w/o TTCO | 0.82 | 0.009 | 0.93 |
| **w/ TTCO** | **0.84** | **0.007** | **0.95** |

### 关键发现
- **PSIVG在运动可控性指标上全面最优**——SAM mIoU 0.84（比第二SG-I2V高0.09），Corr. Pixel MSE仅0.007（最低）
- PISA-Seg等方法虽时间稳定性指标高，但实际生成几乎静态的视频（帧间变化极小），缺乏真实运动
- **TTCO的效果主要体现在纹理一致性**——Corr. Pixel MSE从0.009降到0.007，Subject Consistency从0.93提升到0.95
- 基于Prompt的优化比LoRA-based设计更好——LoRA经常降低背景质量产生伪影，prompt优化更轻量且局部性更好
- 直接优化spatio-temporal token（而非text token）会产生网格状伪影

## 亮点与洞察
- **"Simulation-in-the-loop"范式**是最大贡献——不修改生成模型、不需要额外训练，纯推理时引入物理约束。这种与生成模型解耦的设计使其可以即插即用到各种视频生成模型上
- **感知管线的巧妙设计**：用InstantMesh对单帧做3D重建（而非多帧），因为生成的视频帧间几何不一致——这是对生成视频特性的深刻理解
- **GPT-5分层物理属性估计**：先推断材质描述（组成、弹性、粗糙度），再映射到数值物理参数——比直接让LLM输出数值更可靠。这种coarse-to-fine的LLM使用范式可推广到其他需要从视觉估计物理量的场景
- **TTCO的"文本token=物体控制"发现**：修改前景物体对应的text embedding主要影响该物体的外观而不破坏背景，与其他扩散研究的发现一致，进一步确认了text token的空间对应性

## 局限与展望
- 依赖MPM模拟器，无法处理复杂代理（人、车辆）和铰接结构
- 感知管线中初始物体重建的质量直接影响下游——重建误差会传播到模拟和生成
- 继承GwtF视频模型的限制——难以生成非常小或非常细的物体
- 整个pipeline比端到端方法复杂得多（模板视频→感知→模拟→重生成→TTCO），延时较高
- 仅支持刚体/材料点模型描述的物体交互，不支持流体、布料等复杂材料

## 相关工作与启发
- **vs PhysAnimator**: 专注2D卡通动画的2D网格+2D模拟器，PSIVG是3D+训练-free的开放词表视频生成
- **vs PhysGen3D**: 从输入图像获取3D表示做MPM模拟然后直接渲染，PSIVG额外使用视频扩散模型弥补模拟器渲染的不足（低分辨率、缺少光影、风格不自然）
- **vs WonderPlay**: 先生成3DGS surfel场景再用视频监督更新，PSIVG直接用TTCO做视频精炼更简洁高效
- **vs PISA**: 通过微调扩散模型学习物理交互，需要大量训练数据。PSIVG完全training-free
- **vs Phantom**: Phantom将物理推理内化到模型中（需要训练），PSIVG在推理时外部注入物理约束（不需要训练），两者互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将3D物理模拟器嵌入文本到视频扩散管线的训练-free框架，TTCO的设计也有创意
- 实验充分度: ⭐⭐⭐⭐ 定量比较全面，用户研究(82.3%偏好率)说服力强，消融覆盖了关键组件
- 写作质量: ⭐⭐⭐⭐ 方法流程清晰易懂，图示直观
- 价值: ⭐⭐⭐⭐ 提出了一种通用范式可即插即用到任何视频生成模型，但pipeline复杂度和MPM限制制约了实用性

<!-- RELATED:START -->

## 相关论文

- [Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [How Far is Video Generation from World Model: A Physical Law Perspective](../../ICML2025/video_generation/how_far_is_video_generation_from_world_model_a_physical_law_perspective.md)
- [Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)
- [LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_linear_attention_video_generation.md)
- [Anti-I2V: Safeguarding your photos from malicious image-to-video generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)

<!-- RELATED:END -->
