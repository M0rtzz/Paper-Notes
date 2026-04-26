---
title: >-
  [论文解读] Lighting-grounded Video Generation with Renderer-based Agent Reasoning
description: >-
  [CVPR 2026][光照可控视频生成] LiVER 提出了一种光照驱动的视频生成框架，通过渲染器Agent将文本描述转化为显式3D场景代理（包含布局、光照、相机轨迹），再利用物理渲染生成diffuse/glossy/rough GGX的场景proxy，注入视频扩散模型实现物理准确的光照效果与精确场景控制。
tags:
  - CVPR 2026
  - 光照可控视频生成
  - 3D场景代理
  - 物理渲染
  - 扩散模型
  - 场景Agent
---

# Lighting-grounded Video Generation with Renderer-based Agent Reasoning

**会议**: CVPR 2026  
**arXiv**: [2604.07966](https://arxiv.org/abs/2604.07966)  
**代码**: 无  
**领域**: 3D视觉 / 视频生成  
**关键词**: 光照可控视频生成, 3D场景代理, 物理渲染, 扩散模型, 场景Agent

## 一句话总结

LiVER 提出了一种光照驱动的视频生成框架，通过渲染器Agent将文本描述转化为显式3D场景代理（包含布局、光照、相机轨迹），再利用物理渲染生成diffuse/glossy/rough GGX的场景proxy，注入视频扩散模型实现物理准确的光照效果与精确场景控制。

## 研究背景与动机

扩散模型在视频生成领域取得了显著进步，但可控性仍是核心瓶颈。现有方法主要通过数据驱动的方式提升视觉质量，但在显式建模场景因素（如布局、光照、相机轨迹）方面存在明显不足。虽然已有工作尝试引入3D感知条件（如相机控制 CameraCtrl、物体轨迹 MotionCtrl），但这些方法几乎完全忽略了物理准确的光照建模——阴影、反射、环境光遮蔽等效果在真实材质（皮肤、金属、玻璃）上仍然不真实。

核心矛盾在于：现有方法要么只关注几何/运动控制而忽略光照，要么将光照与其他属性纠缠在一起无法解耦控制。本文的切入角度是将场景的光照属性通过物理渲染器显式解耦为可控的2D渲染通道（diffuse、glossy GGX、rough GGX），保留3D场景的光照物理信息同时以图像序列形式注入视频扩散模型。

核心idea：**用渲染器Agent自动将文本转化为3D场景 → 通过PBR渲染获得光照感知的场景代理 → 轻量级编码器+三阶段训练将物理光照信号注入视频扩散模型**。

## 方法详解

### 整体框架

LiVER的pipeline分三步：(1) 渲染器Agent解析用户文本，提取物体类别、空间关系、光照线索和相机运动，构建结构化场景图并从资产库检索3D模型；(2) 物理渲染器（Blender）根据3D场景、HDR环境贴图和相机轨迹渲染场景代理 $y \in \mathbb{R}^{F \times 9 \times H \times W}$，包含三组RGB渲染通道；(3) 轻量级proxy编码器将场景代理映射到视频latent空间，通过残差注入引导Wan2.2-5B视频扩散模型生成具有物理光照效果的视频。

### 关键设计

1. **渲染器Agent推理系统**:
    - 功能：将高层文本描述转化为结构化3D控制信号
    - 核心思路：分三步执行——Scene Agent 解析物体和空间关系构建场景图 $\mathcal{G}=(V,E)$，从Objaverse-XL检索资产；Lighting Agent 根据文本光照线索（如"warm mood"）从Poly Haven库选择HDR环境贴图；Camera Agent 解析运动语义（如"orbit"、"dolly zoom"）并通过spline插值生成平滑相机轨迹
    - 设计动机：自动化生成3D控制信号，降低用户使用门槛，同时支持手动编辑以满足专业需求

2. **物理光照场景代理**:
    - 功能：将3D场景的光照信息编码为可注入扩散模型的2D信号
    - 核心思路：通过PBR渲染器生成三组渲染通道——diffuse（低频环境光）、rough GGX（中频宽反射）、glossy GGX（高频镜面高光），堆叠为9通道图像序列 $y = [x^{\text{DIFF}}, x^{\text{GGX1}}, x^{\text{GGX2}}] = R(s^i, l^i, c^i)$
    - 设计动机：直接使用完整3D表示对视频扩散模型来说过于复杂，而2D渲染通道既保留物理光照信息又兼容图像处理流程

3. **轻量级代理编码器与适配器**:
    - 功能：将场景代理特征与视频latent空间对齐
    - 核心思路：2D卷积编码器将9通道输入下采样至与VAE latent相同分辨率 $z^y \in \mathbb{R}^{F \times C \times H' \times W'}$，通过零初始化的可学习标量 $\alpha$ 进行残差注入 $z' = z + \alpha \cdot z^y$
    - 设计动机：零初始化确保训练初期不影响原有生成能力，proxy特征逐渐引导latent空间实现光照控制

### 损失函数 / 训练策略

采用三阶段训练方案：
- **Stage 1 - 条件通路训练**：冻结视频扩散主干，仅训练proxy编码器和适配器（10 epoch），学习将场景代理转化为粗略控制信号
- **Stage 2 - 联合LoRA微调**：解冻主干中的LoRA层，与编码器/适配器联合训练（10 epoch），精细化语义对齐
- **Stage 3 - 光照多样性扩展**：继续联合训练，以1:1混合真实与合成数据，增强模型对多样光照现象的泛化能力

训练损失采用标准flow matching目标：$\mathcal{L} = \mathbb{E}_{z,\epsilon,t}[|u_\theta(z_t, y, c^{\text{txt}}, t) - v_t|^2]$

## 实验关键数据

### 主实验

| 方法 | FVD ↓ | FID ↓ | CLIP ↑ | ATE ↓ | LE ↓ | mIoU ↑ |
|------|-------|-------|--------|-------|------|--------|
| CameraCtrl | 48.03 | 98.29 | 28.75 | 2.15 | 0.06 | 0.68 |
| MotionCtrl | 63.13 | 97.21 | 26.67 | 3.42 | 0.07 | 0.66 |
| VideoFrom3D | 36.94 | 157.89 | 24.51 | 17.55 | 0.05 | 0.74 |
| **LiVER** | **32.56** | **129.56** | **30.97** | **2.48** | **0.04** | **0.87** |

16帧对比（vs CameraCtrl/MotionCtrl）中，LiVER FVD=32.45, FID=42.32, CLIP=29.62，全面领先。

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 无合成数据 | 光照均匀、错误 | 缺乏动态光照多样性，过拟合真实数据的有限光照模式 |
| 无分阶段训练 | 输出几乎静止 | 同时学习控制信号和适配预训练模型优化困难 |
| 完整模型 | 最佳效果 | 三阶段方案确保稳定收敛和高质量生成 |

### 关键发现

- 合成数据对光照多样性至关重要：仅用真实数据训练导致光照均匀平淡
- 分阶段训练是模型稳定收敛的关键：端到端训练导致生成结果几乎静态
- 用户研究（25人×20组）显示LiVER在视频质量(83.4%)、场景控制(83.3%)、相机控制(72.1%)、光照控制(59.3%)四个维度全面优于竞争方法

## 亮点与洞察

- 将物理渲染的光照分解（diffuse/glossy/rough）作为视频生成的条件控制信号，是一个非常优雅的设计——既保留了物理意义又兼容2D处理流程
- 渲染器Agent的设计使系统既可自动化使用又支持手动编辑，满足从普通用户到专业影视制作的不同需求
- 零初始化残差注入策略是一个成熟的工程选择，确保预训练模型能力不被破坏

## 局限与展望

- 初始3D重建较粗糙，几何细节和材质效果依赖文本描述的精确度，对prompt敏感
- 3D资产检索的质量和覆盖度受限于现有资产库
- 仅支持HDR环境贴图作为全局光照，不支持局部光源的精细控制
- 场景代理的渲染需要Blender等引擎，增加了推理pipeline的复杂度

## 相关工作与启发

- **vs CameraCtrl**: CameraCtrl仅控制相机运动，LiVER额外实现了光照和布局的解耦控制
- **vs VideoFrom3D**: VideoFrom3D需要为每个测试样本训练style LoRA（~40分钟），且忽略光照，LiVER端到端且光照感知
- **vs Light-A-Video/LumiSculpt**: 这些方法聚焦于视频重光照，但将光照与其他物理属性纠缠；LiVER从3D场景代理出发实现解耦
- **启发**: 将PBR渲染的中间结果作为生成模型的条件信号是一个可推广的范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 将PBR渲染通道作为视频生成条件的思路新颖，Agent+渲染器+扩散模型的组合设计完整
- 实验充分度: ⭐⭐⭐⭐ 定量/定性/用户研究/消融实验完备，但数据集规模偏小(11K视频)
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法描述详细，图示质量高
- 价值: ⭐⭐⭐⭐ 对影视制作和虚拟内容生产有实际应用价值，推动了可控视频生成的发展

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](../../NeurIPS2025/video_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](../../CVPR2025/video_generation/phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)
- [\[CVPR 2026\] Gloria: Consistent Character Video Generation via Content Anchors](gloria_consistent_character_video_generation_via_content_anchors.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)

<!-- RELATED:END -->
