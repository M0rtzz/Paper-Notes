---
title: >-
  [论文解读] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis
description: >-
  [ICCV 2025][扩散模型蒸馏] 本文提出对抗分布匹配（ADM）框架，通过基于扩散模型的判别器以对抗方式对齐真假分数估计器的潜在预测，替代DMD中预定义的KL散度，结合对抗蒸馏预训练（ADP），在SDXL上实现一步生成超越DMD2，并在SD3和CogVideoX上刷新多步蒸馏基准。
tags:
  - ICCV 2025
  - 扩散模型蒸馏
  - 对抗分布匹配
  - 少步生成
  - 分数蒸馏
  - 视频合成加速
---

# Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis

**会议**: ICCV 2025  
**arXiv**: [2507.18569](https://arxiv.org/abs/2507.18569)  
**代码**: 无  
**领域**: image_generation  
**关键词**: 扩散模型蒸馏, 对抗分布匹配, 少步生成, 分数蒸馏, 视频合成加速

## 一句话总结
本文提出对抗分布匹配（ADM）框架，通过基于扩散模型的判别器以对抗方式对齐真假分数估计器的潜在预测，替代DMD中预定义的KL散度，结合对抗蒸馏预训练（ADP），在SDXL上实现一步生成超越DMD2，并在SD3和CogVideoX上刷新多步蒸馏基准。

## 研究背景与动机

1. **领域现状**: Distribution Matching Distillation（DMD）是主流的分数蒸馏方法，通过最小化逆KL散度将教师扩散模型压缩为高效的一步/多步学生生成器。
2. **现有痛点**: DMD依赖逆KL散度最小化，具有zero-forcing特性——驱使低概率区域趋零，导致模型只关注少数主要模式，易产生模式崩塌（mode collapse）。DMD/DMD2中额外引入的ODE/GAN正则化只是权衡抵消，未根本解决模式寻求行为。
3. **核心矛盾**: 预定义的显式散度度量（逆KL、Fisher散度等）难以充分捕获复杂高维文本条件图像/视频分布中的多方面对齐需求。一步蒸馏时学生和教师分布支撑集重叠不足，导致梯度爆炸/消失。
4. **本文目标**: (1) 如何绕过预定义散度限制，实现更灵活的分布匹配？(2) 如何为极具挑战的一步蒸馏提供更好的初始化？
5. **切入角度**: 利用GAN的隐式数据驱动度量替代显式散度。Hinge GAN理论上最小化Total-Variation Distance（TVD），具有对称性和有界性，比逆KL更适合支撑集重叠少的情况。
6. **核心 idea**: 用基于扩散模型的对抗判别器对齐不同噪声水平下真假分数估计器的ODE预测，实现隐式、自适应的分布匹配蒸馏。

## 方法详解

### 整体框架
DMDX为统一流水线：先进行对抗蒸馏预训练（ADP）为学生模型提供更好的初始化，再用ADM进行分数蒸馏微调。输入为噪声$\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，生成器$G_\theta$输出$\hat{x}_0$，对齐目标为教师模型的输出分布。

### 关键设计

1. **对抗分布匹配（ADM）**:
    - 功能: 替代DMD损失，以对抗方式对齐真假分数估计器在不同噪声水平下的预测
    - 核心思路: 判别器$D_\tau$由冻结的教师扩散模型加多个可训练头组成。给定学生生成的$\hat{x}_0$再扩散得到$x_t$，分别通过真假分数估计器沿PF-ODE走一步到$t-\Delta t$得到$x_{t-\Delta t}^{\text{fake}}$和$x_{t-\Delta t}^{\text{real}}$，用Hinge损失训练判别器区分两者: $\mathcal{L}_{\text{GAN}}(\theta) = -\mathbb{E}[D_\tau(x_{t-\Delta t}^{\text{fake}}, t-\Delta t)]$。时间步间隔默认$\Delta t = T/64$。
    - 设计动机: 相比DMD的逆KL散度，Hinge GAN理论最小化TVD，具有对称性（不存在mode-seeking行为）和有界性$[0,1]$（避免梯度爆炸）。判别器可学习任意非线性函数隐式度量分布差异，具备数据驱动自适应特性。

2. **对抗蒸馏预训练（ADP）**:
    - 功能: 为一步蒸馏的ADM微调提供更好初始化，增大学生和教师分布的支撑集重叠
    - 核心思路: 离线收集教师模型ODE对$(x_T, x_0)$，线性插值构造噪声样本并使用velocity预测。采用**双空间判别器**：潜空间判别器$D_{\tau_1}$（由教师模型初始化）和像素空间判别器$D_{\tau_2}$（由SAM视觉编码器初始化），权重$\lambda_1=0.85, \lambda_2=0.15$。引入立方时间步调度（cubic timestep schedule）$[1-(t/T)^3]*T$，偏向高噪声水平以鼓励探索新模式。
    - 设计动机: 一步蒸馏时学生输出质量差导致$p_{\text{fake}}$和$p_{\text{real}}$支撑集重叠极少，逆KL散度在$p_{\text{fake}} \to 0$处梯度消失、在$p_{\text{real}} \to 0$处梯度爆炸。分布级别的对抗蒸馏可让学生捕获教师更多潜在模式。

3. **ADM与ADP的差异**:
    - 功能: ADM属于分数蒸馏（监督不同噪声水平的完整去噪过程），ADP属于对抗蒸馏（只关心$t=0$时的干净数据分布）
    - 核心思路: ADM通过解PF-ODE保留分数估计器的输入时间步信息，让判别器在噪声空间中工作；ADP通过随机扩散生成器输出来人为创造重叠区域，使判别更困难、梯度更平滑
    - 设计动机: ADM在分布支撑集重叠少时面临判别器容易区分导致的极端梯度信号，所以需要ADP先拉近分布

### 损失函数 / 训练策略
- ADM阶段: Hinge GAN损失，生成器和判别器交替训练，同时动态学习fake score estimator
- ADP阶段: 基于ODE对的分布级Hinge GAN损失 + velocity MSE预训练损失
- ADM不需要额外正则化项（与DMD/DMD2不同），GAN训练隐式包含了逆KL散度的优化方向

## 实验关键数据

### 主实验

| 模型/数据集 | 指标 | 本文（DMDX/ADM） | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| SDXL 1-step | CLIP Score | 35.26 | 35.22 (DMD2) | +0.04 |
| SDXL 1-step | HPSv2 | 27.70 | 27.45 (DMD2) | +0.25 |
| SDXL 1-step | MPS | 11.20 | 10.69 (DMD2) | +0.51 |
| SD3-Medium 4-step | CLIP Score | 34.91 | 34.40 (Flash) | +0.51 |
| SD3-Medium 4-step | Pick Score | 22.55 | 22.09 (Flash) | +0.46 |
| SD3.5-Large 4-step | Pick Score | 22.88 | 22.40 (LADD) | +0.48 |
| CogVideoX 4-step | VBench Quality | 85.25 | 84.73 (Teacher 50-step) | 超越教师 |

### 消融实验

| 配置 | FID↓ | 说明 |
|------|---------|------|
| DMD loss (逆KL) | 较高 | 训练中DMD损失显示稳定下降趋势，表明ADM隐式包含逆KL |
| ADM w/o pretraining | 不稳定 | 梯度爆炸/消失问题 |
| ADP + ADM (DMDX) | 最优 | 完整pipeline |
| 均匀vs立方时间步调度 | — | 立方调度偏向高噪声，促进模式多样性 |

### 关键发现
- ADM在不直接优化DMD损失的情况下，其DMD损失值呈稳定下降趋势，验证了Hinge GAN隐式包含逆KL散度优化
- 当提供更好的初始化时，TTUR（两时间尺度更新规则）对最终性能影响很小
- 4步ADM蒸馏在SD3和SD3.5上均超越教师模型50步采样的质量
- CogVideoX 4步蒸馏在VBench质量分数上超越50步教师模型

## 亮点与洞察
- 从TVD vs 逆KL的理论视角论证了对抗方法在支撑集重叠少时的优势：TVD的对称性避免mode-seeking，有界性避免数值不稳定
- 将分数蒸馏中的GAN判别器设计为"沿PF-ODE走$\Delta t$步"，优雅保留了时间步信息
- 首次在CogVideoX这样的大规模视频模型上成功应用分数蒸馏，实现4步超越50步教师
- 双空间判别器（潜空间+像素空间）的组合增强了判别能力

## 局限与展望
- 一步生成在极高分辨率下质量仍有提升空间
- 对抗训练的稳定性依赖于预训练阶段的质量
- 判别器的dominant FC识别是否可以自动化/自适应调整
- 未讨论在更大规模视频模型（如HunyuanVideo 13B）上的扩展性

## 相关工作与启发
- **vs DMD/DMD2**: ADM用GAN隐式度量替代显式逆KL散度，不需要额外正则化；ADP用分布级对抗蒸馏替代MSE预训练
- **vs SDXL-Lightning**: 共享对抗蒸馏思路，但DMDX进一步引入ADM做分数蒸馏微调
- **vs LADD**: ADP借鉴其合成数据对抗蒸馏思路，但改用ODE对构造噪声、立方调度、双空间判别器

## 评分
- 新颖性: ⭐⭐⭐⭐ 从理论（TVD vs KL）和实践（隐式vs显式度量）两个角度创新分数蒸馏
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖SDXL/SD3/SD3.5/CogVideoX，一步和多步，图像和视频
- 写作质量: ⭐⭐⭐⭐ 理论讨论深入，公式推导清晰
- 价值: ⭐⭐⭐⭐ 为大规模扩散模型提供了统一高效的蒸馏框架

<!-- RELATED:START -->

## 相关论文

- [V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](../../CVPR2025/video_generation/levitor_3d_trajectory_oriented_image-to-video_synthesis.md)

<!-- RELATED:END -->
