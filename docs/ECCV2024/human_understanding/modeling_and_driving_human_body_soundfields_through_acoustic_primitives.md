---
title: >-
  [论文解读] Modeling and Driving Human Body Soundfields through Acoustic Primitives
description: >-
  [ECCV 2024][人体理解][声场建模] 提出基于声学基元(Acoustic Primitives)的人体3D声场建模与渲染框架，将多个低阶球谐声场挂载到人体骨骼关节上，在保持与SOTA可比的音质的同时，实现了15倍加速和近场声音渲染能力。
tags:
  - ECCV 2024
  - 人体理解
  - 声场建模
  - 声学基元
  - 空间音频
  - 人体姿态
  - AR/VR
---

# Modeling and Driving Human Body Soundfields through Acoustic Primitives

**会议**: ECCV 2024  
**arXiv**: [2407.13083](https://arxiv.org/abs/2407.13083)  
**代码**: 有 (https://wikichao.github.io/Acoustic-Primitives/)  
**领域**: 人体理解  
**关键词**: 声场建模, 声学基元, 空间音频, 人体姿态, AR/VR

## 一句话总结

提出基于声学基元(Acoustic Primitives)的人体3D声场建模与渲染框架，将多个低阶球谐声场挂载到人体骨骼关节上，在保持与SOTA可比的音质的同时，实现了15倍加速和近场声音渲染能力。

## 研究背景与动机

3D人体视觉建模（如MetaHumans、Codec Avatars）已达到高度逼真的水平，但**声学建模**严重滞后。对于沉浸式AR/VR体验，精确的音视觉同步至关重要，然而几乎没有研究能为虚拟人渲染空间音频。

先驱工作SoundingBodies [Xu et al.]开发了基于体姿和头戴麦克风的全身声场渲染系统，但存在重大局限：

**无法近场渲染**：使用单一高阶球谐（17阶）表示，声音只能在包围人体的约2m直径球面外渲染，球面内产生混乱结果

**计算效率低**：需预测345个麦克风的原始音频信号，再通过传统DSP处理转换为球谐系数，无法实时

**参数量大且不稳定**：17阶球谐需要大量系数，估计过程不稳定

本文的核心思路来源于视觉神经渲染中的体积基元（如3D Gaussians），将其类比地迁移到声学领域，用多个附着于人体关节的小型低阶声场基元替代单一高阶声场。

## 方法详解

### 整体框架

系统由两个主要阶段组成：

1. **声学基元学习**：给定头戴麦克风音频和3D体姿序列，通过神经网络预测一组声学基元的球谐系数、权重和位置偏移
2. **声场渲染**：利用球面波函数进行可微渲染，将所有基元在目标3D位置的贡献叠加得到最终音频

| 组件 | 功能 | 输出 |
|------|------|------|
| 姿态编码器 | 时序卷积提取体姿时空特征 | 姿态特征 $f_p$ |
| 音频编码器 | ResNet+LSTM提取音频特征（含时延补偿） | 音频特征 $f_a$ |
| 音视融合模块 | ResNet+Attention融合两种特征 | 融合特征 $f_{ap}$ |
| 声场解码器 | U-Net式解码，输出所有基元的球谐系数 | $K$ 组球谐系数 $\{\mathcal{S}_i\}$ |
| 偏移MLP | 预测基元相对关节的位置偏移 | 偏移量 $\Delta(x,y,z)$ |
| 权重MLP | 预测基元的时变权重 | 权重 $W$ |

### 关键设计

**1. 声学基元定义**

每个声学基元是一个挂载于人体关节的小型球面声场，用低阶（2阶）球谐系数描述。其音压可通过球面波函数精确计算：

$$\mathbf{w}(r,\theta,\varphi) = \sum_{n=0}^{N}\sum_{m=-n}^{n} \tilde{c}_{nm} \cdot \frac{h_n(kr)}{h_n(kr_{ref})} \cdot Y_{nm}(\theta,\varphi)$$

整个人体声场是所有 $K$ 个基元在监听位置产生声音的线性叠加。

**2. 时延补偿**

身体不同部位（如脚、手）产生的声音需经过不同时间到达头戴麦克风。方法利用姿态特征通过MLP估计每个基元的时延，并对音频信号进行时间扭曲（time-warp）以补偿。

**3. 基元位置偏移**

关节位置并不精确对应声源位置（如手腕关节 vs 手指打响声的位置），因此学习一个有界偏移量（tanh约束在20cm范围内），使基元位置更准确地反映真实声源。

**4. 基元权重**

不同时刻各基元的重要性不同（如打响指时手部基元活跃、其他基元静默），通过softmax激活的权重显式建模这种时变性。

### 损失函数 / 训练策略

总损失函数由四项组成：

$$\mathcal{L}_{total} = \lambda_{amp}\mathcal{L}_{amp} + \lambda_{ri}\mathcal{L}_{ri} + \lambda_{s\ell1}\mathcal{L}_{s\ell1} + \lambda_{cts}\mathcal{L}_{cts}$$

| 损失项 | 功能 | 权重 |
|--------|------|------|
| $\mathcal{L}_{amp}$ | 多尺度STFT幅度谱损失（窗口2048/1024/512/256） | 7 |
| $\mathcal{L}_{ri}$ | STFT实部和虚部损失 | 3 |
| $\mathcal{L}_{s\ell1}$ | Shift-L1损失，减小空间对齐误差 | 0.5 |
| $\mathcal{L}_{cts}$ | 交叉熵损失，用片段级标签监督基元权重 | 1 |

训练细节：每GPU batch size=1，每次前向传播随机选20个目标麦克风，AdamW优化器(lr=0.0002)，训练100 epochs，4×A100 GPU约55小时。

## 实验关键数据

### 主实验（表格）

在SoundingBodies公开数据集上评估（消声室录制，345麦克风球形阵列，6.5个人体）。

| 方法 | 推理速度 | 非语音SDR↑ | 非语音幅度误差↓ | 非语音相位误差↓ | 语音SDR↑ | 语音幅度误差↓ | 语音相位误差↓ |
|------|---------|-----------|---------------|---------------|---------|-------------|-------------|
| SoundingBodies | 3.56s | 3.052 | 0.832 | 0.314 | **9.635** | **0.701** | 0.464 |
| **Ours (12基元, 2阶)** | **0.24s** | **3.597** | **0.883** | 0.323 | 8.448 | 0.943 | **0.417** |

核心结论：**15×加速**，非语音SDR更好，语音质量接近但有小幅下降。

### 消融实验（表格）

不同基元数量和球谐阶数的影响：

| 基元数K | 阶数N | 非语音SDR↑ | 语音SDR↑ |
|---------|-------|-----------|---------|
| 5 | 0阶 | 2.009 | 4.775 |
| 5 | 2阶 | 3.552 | 7.981 |
| 9 | 2阶 | 3.569 | 8.200 |
| 12 | 2阶 | **3.597** | **8.448** |
| 12 (无偏移) | 2阶 | 3.528 | 7.730 |

### 关键发现

1. **基元数量和阶数均正相关于性能**：12个2阶基元效果最优。语音尤其受益于更多基元和更高阶数
2. **位置偏移机制有效**：去掉偏移约等于将12基元退化为9基元（重复基元坍缩到同一位置）
3. **0阶到1阶提升最大**：0阶只能建模无方向性声场，1阶及以上才能捕捉声音方向性
4. **声源分配正确**：可视化显示语音能量主要分配给头部基元，手拍声分配给手部基元，方向性模式与头部朝向一致

## 亮点与洞察

- **视觉-声学的优雅类比**：将3D Gaussian Splatting思想迁移到声学领域，用声学基元类比体积基元，是跨模态方法论迁移的精彩案例
- **可微声学渲染器**：球面波函数的可微实现使端到端训练成为可能，渲染方程提供了强物理先验
- **倒逼声源分离**：系统隐式学会了将混合音频信号分解到不同身体部位的声源，无需显式声源分离标注
- **实时渲染能力**：0.24s处理1秒48kHz音频，已接近实时，为AR/VR应用奠定基础

## 局限与展望

1. **球谐阶数受限**：PyTorch仅支持2阶球面波函数实现，更高阶可能进一步提升精度
2. **需要大规模麦克风阵列训练数据**：345个麦克风的消声室录制设备昂贵，难以规模化
3. **未建模室内混响**：在消声室中训练和测试，无法处理真实室内环境的混响效应
4. **仅支持单人场景**：多人交互场景的声场建模未被探索
5. **身体拍击声幅度匹配较差**：高变化性的辐射模式对低阶基元构成挑战

## 相关工作与启发

- **与NeRF/3DGS的关系**：方法论上高度类似——用基元分布代替全局表示，用可微渲染代替显式求解
- **与SoundingBodies的互补性**：SoundingBodies追求高保真但牺牲效率，本文追求效率同时保持质量
- **潜在扩展**：可结合文本/音频驱动的体动生成模型，实现完整的声-视-动一体化虚拟人

## 评分

| 维度 | 分数 (1-5) | 评价 |
|------|-----------|------|
| 新颖性 | 4.5 | 声学基元概念新颖，跨模态类比精彩 |
| 技术深度 | 4 | 物理声学与深度学习的有机结合，可微渲染设计精巧 |
| 实验充分性 | 3.5 | 消融充分，但仅一个数据集，缺少真实环境测试 |
| 写作质量 | 4 | 动机清晰，与视觉渲染的类比有助于理解 |
| 实用价值 | 4 | 对AR/VR中的空间音频渲染有直接应用价值 |

<!-- RELATED:START -->

## 相关论文

- [Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving](upper-body_hierarchical_graph_for_skeleton_based_emotion_recognition_in_assistiv.md)
- [HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding](egoexo-fitness_towards_egocentric_and_exocentric_full-body_action_understanding.md)
- [PoseSOR: Human Pose Can Guide Our Attention](posesor_human_pose_can_guide_our_attention.md)
- [LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)

<!-- RELATED:END -->
