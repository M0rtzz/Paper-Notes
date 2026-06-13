---
title: >-
  [论文解读] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation
description: >-
  [CVPR 2025][视频生成][视频扩散模型] TokenMotion 提出首个基于 DiT 的视频扩散框架，通过将相机轨迹和人体姿态表示为时空 token，并利用"解耦-融合"策略与人体感知动态掩码，实现对相机运动与人体运动的精细联合控制，在 text-to-video 和 image-to-video 范式上均超越现有 SOTA。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型"
  - "运动控制"
  - "token解耦"
  - "相机运动"
  - "人体姿态"
---

# TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation

**会议**: CVPR 2025  
**arXiv**: [2504.08181](https://arxiv.org/abs/2504.08181)  
**代码**: 无  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频扩散模型, 运动控制, token解耦, 相机运动, 人体姿态

## 一句话总结
TokenMotion 提出首个基于 DiT 的视频扩散框架，通过将相机轨迹和人体姿态表示为时空 token，并利用"解耦-融合"策略与人体感知动态掩码，实现对相机运动与人体运动的精细联合控制，在 text-to-video 和 image-to-video 范式上均超越现有 SOTA。

## 研究背景与动机

**领域现状**：以人为中心的视频生成是当前视频扩散模型的核心应用场景之一，尤其是在创意制作中（如格莱美 Glambot 慢动作拍摄效果），需要同时精确控制相机运动（推拉摇移）和人体动作（姿态序列）。

**现有痛点**：现有视频扩散方法在运动控制方面存在两大问题：(1) 运动表示能力有限——大多数方法使用全局条件（如文本描述或单一运动向量）来指导运动，无法实现逐帧、逐区域的精细控制；(2) 相机运动和人体运动的集成不充分——现有方法通常只能控制其中一种运动，难以处理两者的时空耦合关系。

**核心矛盾**：相机运动和人体运动在视频中是时空耦合的——相机移动会改变人体在画面中的位置和尺度，而人体运动本身又独立于相机。直接混合两种控制信号会导致冲突和不自然的结果。问题的本质在于：如何在统一框架中既保持两种运动的独立可控性，又能正确建模它们的交互关系。

**本文目标**：构建一个统一的视频扩散框架，能够分别控制相机运动、人体运动、以及两者的联合交互，同时支持 T2V 和 I2V 两种生成范式。

**切入角度**：作者观察到 DiT（Diffusion Transformer）架构天然以 token 为基本单元进行处理，因此可以将运动信号也表示为 token——不同于全局条件注入，token 化的运动表示能实现"在正确的时间、正确的位置"施加控制力。

**核心 idea**：用时空 token 分别表示相机轨迹和人体姿态，通过"先解耦、再融合"的策略，配合人体感知动态掩码来处理两种运动信号在时空上的重叠与分离。

## 方法详解

### 整体框架
TokenMotion 基于 DiT 架构构建。输入为文本提示（或参考图像），以及相机轨迹序列和/或人体姿态序列。整个 pipeline 分为三个阶段：(1) 运动 token 化——将相机轨迹和人体骨架分别编码为时空 token；(2) 解耦控制注入——通过独立的控制分支将两种运动 token 注入 DiT 的去噪过程；(3) 动态掩码融合——用人体感知动态掩码区分"人体区域"和"背景区域"，合理融合两种运动信号的影响。最终输出为受控的高质量人物视频。

### 关键设计

1. **运动信号的时空 Token 化**:

    - 功能：将连续的相机轨迹和人体姿态序列转化为与视频 latent token 对齐的时空 token 表示
    - 核心思路：对于相机运动，将每帧的相机外参（旋转矩阵+平移向量）通过 Plücker 坐标表示编码为逐像素的射线图（ray map），然后用 patchify 操作转为与 DiT latent 空间对齐的 token 序列。对于人体运动，将 DWPose 提取的关键点序列渲染为逐帧的骨架热力图，同样 patchify 后得到人体运动 token。两种 token 都保留了时空位置信息，使得控制可以精确作用于视频的特定区域和帧。
    - 设计动机：相比全局条件编码（如将相机参数拼接为一个向量），token 化表示能保留局部空间信息，实现"哪里需要控制就在哪里施加影响"的效果，这对于处理人体在画面中持续移动的场景尤为关键。

2. **解耦-融合（Decouple-and-Fuse）控制策略**:

    - 功能：统一框架中独立注入相机和人体运动控制信号，再动态融合
    - 核心思路：框架使用两条并行的 ControlNet-style 分支分别处理相机运动 token 和人体运动 token。每条分支内部有独立的 DiT block 来提取对应的运动特征。关键在于融合阶段——不是简单相加或拼接，而是通过人体感知动态掩码（Human-Aware Dynamic Mask）来决定每个时空位置应该更多受相机控制还是人体控制的影响。具体来说，掩码在人体所在区域权重偏向人体控制分支，在背景区域权重偏向相机控制分支，在边界区域则平滑过渡。
    - 设计动机：直接混合两种运动信号会导致冲突（如相机平移时人体姿态被扭曲），解耦策略保证各自控制的独立性，动态掩码则解决了"同一空间位置两种信号如何协调"的问题。

3. **人体感知动态掩码（Human-Aware Dynamic Mask）**:

    - 功能：生成时空变化的注意力掩码，指导两种运动信号的融合权重
    - 核心思路：利用人体骨架序列生成逐帧的人体区域掩码，通过高斯模糊扩展边界区域。对于每个去噪步骤 $t$，掩码值 $M_t(x,y)$ 在人体区域接近 1（偏向人体控制），在背景区域接近 0（偏向相机控制），边界区域在 0-1 之间平滑过渡。该掩码随帧变化，能自适应处理人体在画面中的移动，避免了固定区域划分的局限。
    - 设计动机：人体在视频中的位置和尺度是动态变化的（尤其当相机也在移动时），静态掩码无法正确处理这种时空变化。动态掩码保证了在任何时刻都能正确区分"人体运动主导区域"和"相机运动主导区域"。

### 损失函数 / 训练策略
TokenMotion 采用标准的扩散去噪损失（v-prediction 形式），在三种训练模式下联合优化：(1) 仅相机控制；(2) 仅人体控制；(3) 联合控制。训练时随机 drop 某一控制信号以增强模型的单控制能力。人体感知掩码在训练中通过 GT 骨架生成，推理时通过输入骨架序列自动获取。

## 实验关键数据

### 主实验

| 任务 | 指标 | TokenMotion | CameraCtrl | MotionCtrl | Direct-a-Video |
|------|------|-------------|------------|------------|----------------|
| 相机控制 (T2V) | RotErr ↓ | **0.87** | 1.34 | 2.01 | 1.56 |
| 相机控制 (T2V) | TransErr ↓ | **0.42** | 0.71 | 1.15 | 0.83 |
| 人体控制 (T2V) | PCK@0.2 ↑ | **78.3** | - | 61.5 | - |
| 联合控制 (T2V) | FVD ↓ | **198** | 287 | 312 | 265 |
| 联合控制 (I2V) | FVD ↓ | **172** | 241 | 278 | 233 |
| 视觉质量 | FID ↓ | **14.2** | 18.7 | 22.3 | 17.5 |

### 消融实验

| 配置 | RotErr ↓ | PCK@0.2 ↑ | FVD ↓ | 说明 |
|------|----------|-----------|-------|------|
| Full TokenMotion | **0.87** | **78.3** | **198** | 完整模型 |
| w/o 动态掩码 | 1.12 | 71.6 | 234 | 去掉掩码后联合控制质量显著下降 |
| w/o 解耦分支 | 1.25 | 68.2 | 251 | 单一分支混合处理两种运动，效果最差 |
| 全局条件替代 token | 1.08 | 73.1 | 225 | 全局注入比 token 化差，验证了局部控制的必要性 |
| w/o 随机 drop 训练 | 0.95 | 74.8 | 213 | 去掉 drop 训练后单独控制能力变弱 |

### 关键发现
- 解耦分支是最关键的设计，去掉后联合控制 FVD 从 198 恶化到 251（+27%）
- 动态掩码在联合控制场景中贡献显著（FVD 198 vs 234），但对单一控制影响较小
- Token 化表示相比全局条件在所有指标上均有提升，证明了局部控制粒度的价值
- 在 I2V 模式下，TokenMotion 的优势更加明显，因为参考图片提供了外观先验，运动控制的精细程度成为主要差异因素

## 亮点与洞察
- **时空 token 化运动表示**：将运动信号与视频 latent 在同一 token 空间对齐，是一个非常自然且高效的设计。这个思路可以迁移到其他条件视频生成任务（如物体轨迹控制、场景流控制）
- **人体感知动态掩码**巧妙地解决了多信号融合的空间冲突问题，本质上是一种基于语义区域的自适应条件权重分配机制，可推广到任意多种条件信号的融合场景
- "先解耦、再融合"的设计范式在多条件控制中具有通用价值，避免了多信号直接混合时的相互干扰

## 局限与展望
- 当前仅支持单人场景，多人场景中的运动控制和掩码生成仍未解决
- 对极端相机运动（如 360° 环绕）的处理能力有限，可能出现几何失真
- 人体骨架作为运动表示缺乏手部细节和面部表情，限制了精细表演控制
- 推理速度受限于双分支结构，实时生成仍有距离

## 相关工作与启发
- **vs CameraCtrl**: CameraCtrl 仅支持相机控制且使用全局条件注入，TokenMotion 扩展到联合控制并采用 token 化局部表示，在相机控制精度上也有提升
- **vs MotionCtrl**: MotionCtrl 支持相机+物体运动但采用简单的条件拼接，缺乏解耦机制，导致联合控制效果较差
- **vs AnimateAnyone**: 人体动画方法通常不考虑相机运动，TokenMotion 首次在统一框架下解决了两者的联合控制

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 DiT 框架下的相机+人体联合运动控制，解耦-融合策略有创意
- 实验充分度: ⭐⭐⭐⭐ T2V/I2V 双范式验证，消融全面，但缺少用户研究的定量细节
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，图示说明力强
- 价值: ⭐⭐⭐⭐ 对创意视频制作有直接应用价值，解耦策略可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] VidTwin: Video VAE with Decoupled Structure and Dynamics](vidtwin_video_vae_with_decoupled_structure_and_dynamics.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] Video-Bench: Human-Aligned Video Generation Benchmark](video-bench_human-aligned_video_generation_benchmark.md)

</div>

<!-- RELATED:END -->
