---
title: >-
  [论文解读] X-Dyna: Expressive Dynamic Human Image Animation
description: >-
  [CVPR 2025][人体理解][人体动画] X-Dyna提出了一种基于扩散模型的零样本人体图像动画管线，通过轻量级Dynamics-Adapter模块在保持外观一致性的同时生成逼真的人体和场景动态效果，并引入S-Face ControlNet实现身份解耦的面部表情迁移。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "人体动画"
  - "扩散模型"
  - "动态生成"
  - "表情控制"
  - "外观参考"
---

# X-Dyna: Expressive Dynamic Human Image Animation

**会议**: CVPR 2025  
**arXiv**: [2501.10021](https://arxiv.org/abs/2501.10021)  
**代码**: [GitHub](https://github.com/bytedance/X-Dyna)  
**领域**: 人体理解  
**关键词**: 人体动画, 扩散模型, 动态生成, 表情控制, 外观参考

## 一句话总结

X-Dyna提出了一种基于扩散模型的零样本人体图像动画管线，通过轻量级Dynamics-Adapter模块在保持外观一致性的同时生成逼真的人体和场景动态效果，并引入S-Face ControlNet实现身份解耦的面部表情迁移。

## 研究背景与动机

人体视频动画旨在利用驱动视频中的身体姿态和面部表情来驱动单张人体图像生成视频，在数字艺术、社交媒体和虚拟人领域有广泛应用。

现有方法主要存在以下问题：
- **ReferenceNet方案**：使用并行UNet复制体来提取外观特征，虽然能有效传递外观信息，但其强约束会导致空间注意力过度受限，生成的背景静止且人体运动僵硬
- **IP-Adapter方案**：基于CLIP图像嵌入注入跨注意力层，但CLIP嵌入难以捕捉详细外观信息，导致明显的身份丢失
- **动态细节缺失**：现有方法在训练数据和网络设计上均不利于动态纹理的生成，如飘动的头发、流动的衣服、瀑布等自然场景效果
- **面部表情控制不足**：简化的面部关键点图缺乏表达细节，且包含身份线索导致跨身份迁移时面部身份泄露

这些问题的根本原因在于外观参考模块对空间注意力施加了过强的约束，抑制了扩散模型原有的动态合成能力。

## 方法详解

### 整体框架

X-Dyna基于预训练的Stable Diffusion 1.5构建，包含三个核心模块：(1) Dynamics-Adapter用于外观参考注入；(2) Pose ControlNet $\mathcal{C}_P$用于身体姿态控制；(3) S-Face ControlNet $\mathcal{C}_F$用于面部表情控制。训练采用混合数据策略，同时使用人体运动视频和自然场景视频。

### 关键设计

**设计一：Dynamics-Adapter — 轻量级跨帧注意力外观注入**

- **功能**：在不损害扩散骨干模型动态合成能力的前提下，有效传递参考图像的外观信息
- **核心思路**：将去噪后的参考图像$I_R$与噪声序列并行输入共享权重的UNet分支，通过跨帧注意力机制计算外观引导。具体地，使用原UNet的$K$和$V$投影器生成参考图像的$K_R$和$V_R$，同时使用可训练的Query投影器副本生成$Q'_i$，计算跨帧注意力$A'_i = \text{softmax}(\frac{Q'_i K_R^\top}{\sqrt{d}}) V_R$，通过零初始化的输出投影层$W'_O$以残差方式加入原始自注意力输出
- **设计动机**：ReferenceNet对所有空间像素施加过强约束导致动态丢失；Dynamics-Adapter通过最小化可训练参数（仅Query投影器和输出投影层），保持扩散骨干的时空生成能力不变，实现外观控制与运动生成的有效解耦

**设计二：S-Face ControlNet — 身份解耦的隐式面部表情控制**

- **功能**：在跨身份驱动场景下实现精准的面部表情迁移，避免驱动信号中的身份信息泄露
- **核心思路**：训练时使用预训练的肖像重演网络$\mathcal{S}$（如FaceVid2Vid）将驱动帧的面部表情迁移到随机选择的不同面部属性的主体上，生成换脸的面部补丁作为额外ControlNet $\mathcal{C}_F$的条件输入。推理时直接使用驱动视频的裁剪面部作为输入，无需换脸网络
- **设计动机**：简化的面部关键点图包含身份线索（如脸型），跨身份迁移时会发生身份泄露。通过跨身份训练策略，ControlNet学习从合成的换脸图像中隐式提取身份无关的表情信息

**设计三：Harmonic Data Fusion Training — 混合数据融合训练**

- **功能**：使模型同时学习人体动态和背景场景动态效果
- **核心思路**：将自然场景视频（如瀑布、烟花、风）与真实人体运动视频混合训练。对于无人体的场景视频，将Pose ControlNet和S-Face ControlNet的条件输入留空
- **设计动机**：现有方法主要在静态背景的人体视频上训练，无法捕捉动态环境细节。混合训练策略使模型不仅能学习人体微妙动态，还能减少ControlNet对背景运动的意外影响

### 损失函数

采用标准的DDPM去噪损失进行端到端训练。训练分阶段进行：首先训练Dynamics-Adapter、Pose ControlNet和运动模块5个epoch（混合数据），然后冻结这些模块，单独训练S-Face ControlNet 2个epoch（仅人体视频）。

## 实验关键数据

### 主实验：动态纹理生成质量

| 方法 | FG-DTFVD ↓ | BG-DTFVD ↓ | DTFVD ↓ |
|------|-----------|-----------|---------|
| MagicAnimate | 1.753 | 2.142 | 2.601 |
| Animate-Anyone | 1.789 | 2.034 | 2.310 |
| MagicPose | 1.846 | 1.901 | 2.412 |
| MimicMotion | 2.639 | 3.274 | 3.590 |
| **X-Dyna** | **0.900** | **1.101** | **1.518** |

### 消融实验：各模块贡献

| 方法 | FG-DTFVD ↓ | BG-DTFVD ↓ | DTFVD ↓ | Face-Cos ↑ |
|------|-----------|-----------|---------|-----------|
| w/RefNet | 2.137 | 2.694 | 2.823 | 0.466 |
| w/IP-A | 3.738 | 4.702 | 4.851 | 0.292 |
| w/lmk | 0.914 | 1.125 | 1.589 | 0.406 |
| wo/face | 0.912 | 1.098 | 1.550 | 0.442 |
| wo/fusion | 1.301 | 1.467 | 1.652 | 0.495 |
| **X-Dyna** | **0.900** | **1.101** | **1.518** | **0.497** |

### 关键发现

- Dynamics-Adapter相比ReferenceNet在DTFVD上降低了约46%，验证了轻量级跨帧注意力在保持动态能力方面的优势
- 用户研究中X-Dyna在前景动态（3.87 vs 最高2.34）、背景动态（4.26 vs 最高2.78）和身份保持（4.14）上全面领先
- 混合数据训练策略使BG-DTFVD从1.467降至1.101，显著提升了背景动态质量

## 亮点与洞察

1. **精准的问题诊断**：准确识别到ReferenceNet对空间注意力的过强约束是动态细节丢失的根本原因
2. **优雅的解耦设计**：Dynamics-Adapter通过零初始化的残差机制实现了外观注入和动态生成的解耦，设计简洁有效
3. **跨身份训练的巧妙思路**：利用换脸网络合成训练数据，让ControlNet隐式学习身份无关的表情特征

## 局限与展望

- 当目标姿态与参考人体差异极大时（如极端缩放），外观和身份保持可能不完美
- 手部姿态生成质量仍有待提高
- 未来可将Dynamics-Adapter应用于更强大的基础模型（SVD、SDXL、SD3），并结合相机轨迹或拖拽控制

## 相关工作与启发

- **ReferenceNet系列**（MagicAnimate, Animate-Anyone, MagicPose）：并行UNet方案在外观保持上有效但牺牲动态性
- **AnimateDiff**：为扩散模型注入时序一致性的通用时序模块
- **IP-Adapter**：通过CLIP嵌入注入外观信息，但细节保持不足
- 启发：在扩散模型中注入条件信息时，核心挑战是在条件约束强度和模型生成自由度之间找到平衡

## 评分

⭐⭐⭐⭐ — 在人体动画领域提出了清晰的问题诊断和优雅的解决方案，Dynamics-Adapter的设计思路具有通用性，混合数据训练策略简单有效。实验全面且结果显著。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance](../../ICCV2025/human_understanding/dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[CVPR 2025\] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)

</div>

<!-- RELATED:END -->
