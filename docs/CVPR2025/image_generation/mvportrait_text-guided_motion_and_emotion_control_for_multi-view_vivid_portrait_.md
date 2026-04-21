---
title: >-
  [论文解读] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation
description: >-
  [CVPR 2025][图像生成][肖像动画] 本文提出MVPortrait，一个两阶段文本引导框架（Text2FLAME + FLAME2Video），通过将FLAME 3D参数化面部模型作为中间表示，分别用MotionDM和EmotionDM扩散模型生成运动和表情参数序列，再用多视角视频生成模型将FLAME渲染序列转化为逼真的多视角肖像动画，首次实现文本/语音/视频三种信号兼容的可控肖像动画。
tags:
  - CVPR 2025
  - 图像生成
  - 肖像动画
  - 文本驱动
  - FLAME
  - 多视角一致性
  - 扩散模型
  - 面部表情控制
---

# MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation

**会议**: CVPR 2025  
**arXiv**: [2503.19383](https://arxiv.org/abs/2503.19383)  
**代码**: 未开源  
**领域**: 图像生成/肖像动画  
**关键词**: 肖像动画, 文本驱动, FLAME, 多视角一致性, 扩散模型, 面部表情控制

## 一句话总结

本文提出MVPortrait，一个两阶段文本引导框架（Text2FLAME + FLAME2Video），通过将FLAME 3D参数化面部模型作为中间表示，分别用MotionDM和EmotionDM扩散模型生成运动和表情参数序列，再用多视角视频生成模型将FLAME渲染序列转化为逼真的多视角肖像动画，首次实现文本/语音/视频三种信号兼容的可控肖像动画。

## 研究背景与动机

**领域现状**：肖像动画领域主要有三类方法——唇形同步（如EMO、VASA-1）、动作控制（如FollowYourEmoji、AniPortrait）和3D感知面部重建（如Portrait4D-v2）。唇形同步方法仅关注嘴部运动，缺乏全面的表情学习能力；动作控制方法依赖关键点图但可能遗漏面部细节；3D方法（如基于Triplane）在侧视图中会出现"多面"伪影。

**现有痛点**：(1) 缺乏对头部运动和面部表情的显式独立控制能力；(2) 无法从多视角生成一致的视频；(3) 文本驱动的肖像动画尽管用户友好度最高，但研究极少——没有专门的面部运动文本-参数数据集和生成框架。

**核心矛盾**：文本描述通常同时包含运动（"摇头"）和表情（"惊讶"），这两者的振幅和强度变化规律截然不同，直接联合生成会导致控制粒度不足。同时，多视角一致性和文本对齐是两个相互竞争的目标。

**本文目标** 如何通过文本描述，同时精确控制面部运动和表情，并生成多视角时序一致的肖像动画。

**切入角度**：利用FLAME 3D参数化面部模型的紧凑参数空间（shape/pose/expression），将运动和表情解耦为独立的扩散模型生成任务，并通过调整头部朝向参数直接获取多视角渲染。

**核心 idea**：以FLAME作为桥梁，将肖像动画分解为Text→FLAME参数序列→多视角视频两个阶段，实现解耦的运动/表情控制和天然的多视角一致性。

## 方法详解

### 整体框架

MVPortrait包含两阶段：**Text2FLAME**阶段，从参考图像用DECA提取FLAME shape参数，文本提示分为motion和emotion描述分别送入MotionDM和EmotionDM生成pose和expression参数序列，组合形成完整FLAME序列；**FLAME2Video**阶段，将FLAME序列从多视角渲染为条件图像序列，与参考图像一起输入多视角动画生成网络（含VAE、FLAME Encoder、Reference UNet和Denoising UNet），生成时序一致且多视角一致的肖像动画。

### 关键设计

1. **解耦的MotionDM与EmotionDM**：
    - 功能：分别生成FLAME pose参数序列（$f_{\text{pose}} \in \mathbb{R}^{12}$）和expression参数序列（$f_{\text{exp}} \in \mathbb{R}^{50}$）
    - 核心思路：两个独立的Transformer-based扩散模型共享相同网络架构（单层encoder-transformer，latent维度64），分别以motion描述和emotion描述为文本条件。采用MDM架构，通过DDPM框架逐步去噪，同时使用滑动窗口平滑（窗口大小3）消除原始数据噪声导致的抖动
    - 设计动机：运动和表情的振幅特征差异巨大（頭部运动vs微表情），联合训练会导致两者都不精确。消融实验证实Joint Generation变体表情错误且运动静止，验证了解耦训练的必要性

2. **View Attention多视角一致性模块**：
    - 功能：在Denoising UNet中实现跨视角信息共享，确保多视角动画的一致性
    - 核心思路：在motion module（时序注意力）之后插入view module，将特征图从$\mathbb{R}^{(b \times t \times h \times w) \times m \times c}$维度进行view attention（沿视角维度$m$的自注意力）。前向传播时将多视角特征合并为$\mathbb{R}^{(b \times m) \times t \times h \times w \times c}$保持与单视角相同流程
    - 设计动机：仅靠FLAME渲染的多视角条件不足以保证生成视频的跨视角一致性。消融实验显示去掉view module后侧视角出现严重伪影和身份偏移

3. **FLAME Encoder条件注入**：
    - 功能：将FLAME渲染图中的姿态和表情信息注入到Denoising UNet
    - 核心思路：采用与AniPortrait的pose guider相同架构，将FLAME渲染图编码后的特征图与噪声latent相加后传入UNet。同时通过Reference UNet的空间注意力将参考图像的外观信息通过K/V拼接注入，CLIP高级特征通过交叉注意力适配
    - 设计动机：FLAME渲染图携带了shape、pose和expression的完整3D信息，直接编码后注入比仅用关键点更全面，且约束生成视频的面部形状与参考图像一致

### 损失函数

- **Text2FLAME阶段**：$\mathcal{L}_{DM} = \mathcal{L}_{\text{simple}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}$，其中$\mathcal{L}_{\text{simple}}$为标准DDPM重建损失，$\mathcal{L}_{\text{vel}}$为速度几何损失（相邻帧间速度差的L2范数），$\lambda_{\text{vel}}=0.5$
- **FLAME2Video阶段**：采用三阶段训练策略——Phase 1训练FLAME Encoder、Reference UNet和2D分量；Phase 2冻结其他组件单独训练motion module；Phase 3单独训练view module

## 实验关键数据

### 主实验表

**文本引导单视角肖像动画（CelebV-Text测试集）**：

| 方法 | LIQE↑ | FID↓ | CLIPSIM↑ | VideoClip↑ | Variability↑ | MC↑ | EC↑ |
|------|-------|------|----------|------------|-------------|-----|-----|
| AnimateAnything | 4.024 | 34.9 | 0.171 | 0.564 | 0.095 | 1.33 | 1.23 |
| MMVID-interp | 1.541 | 217.4 | 0.175 | 0.517 | 0.092 | 1.67 | 1.56 |
| **MVPortrait** | **4.760** | **28.6** | **0.183** | **0.595** | **0.110** | **2.57** | **2.29** |

**多视角合成对比**：

| 方法 | LPIPS↓ | SSIM↑ | ID↑ |
|------|--------|-------|-----|
| Triplanenet | 0.0936 | 0.5974 | 0.7710 |
| Portrait4D-v2 | 0.4468 | 0.5278 | 0.8006 |
| **MVPortrait** | 0.2101 | **0.6224** | **0.8409** |

### 消融实验表

**Text2FLAME阶段消融（CelebV-Text）**：

| 变体 | CLIPSIM↑ | VideoClip↑ | MC↑ | EC↑ |
|------|----------|------------|-----|-----|
| No smoothing | 0.169 | 0.586 | 1.88 | 1.22 |
| Larger network | 0.174 | 0.548 | 1.44 | 1.75 |
| Joint training | 0.175 | 0.559 | 1.50 | 1.44 |
| **MVPortrait** | **0.183** | **0.595** | **2.57** | **2.29** |

### 关键发现

- MVPortrait在运动一致性(MC=2.57)和表情一致性(EC=2.29)的主观评分上远超基线，说明解耦生成确实带来了更精确的控制
- 去掉滑动窗口平滑后Variability异常升高（0.243），但MC仅1.88——说明高运动多样性实际来自抖动而非有意义的运动
- SSIM=0.6224和ID=0.8409证明FLAME作为中间表示能有效保持身份一致性同时支持多视角
- 统一框架在视频驱动（FLAME-L1=0.196，优于FollowYourEmoji和LivePortrait）和音频驱动场景也具竞争力

## 亮点与洞察

1. **FLAME作为统一桥梁**的设计非常优雅：一个中间表示同时解决了多信号兼容（text/audio/video）、运动/表情解耦、多视角渲染三个难题
2. **解耦优于联合**的实验结论很有说服力：Joint Generation变体同时导致表情错误和运动缺失，说明这两个参数空间的分布确实差异很大
3. 文本→面部运动是一个**新开辟的任务方向**，之前没有专门的数据集——作者从CelebV-Text构建了CelebV-TF数据集（15k+文本-FLAME对）
4. 三阶段分步训练策略（先2D基础→再时序→再视角）是处理多维度一致性的有效工程实践

## 局限性

- 文本标注质量受限于视频数据集（CelebV-Text），面部运动和表情分布不均匀
- FLAME模型在捕捉微表情方面存在先天局限，细微情感变化难以编码
- FVD指标（570.0）不如AnimateAnything（283.0），原因是头部运动幅度更大导致与原始数据分布偏移
- 3D参数空间的分辨率有限——FLAME仅5023个顶点，精细面部褶皱无法表达
- 推理速度（0.45 s/frame for FLAME2Video）仍制约实时应用

## 相关工作与启发

- **MDM** [Tevet et al.]: 文本驱动身体运动生成的扩散模型，本文将其思路迁移到面部运动空间
- **AniPortrait** [Wei et al.]: 使用3D面部mesh预测参考和目标pose，本文继承了其FLAME Encoder架构
- **AnimateDiff** [Guo et al.]: 通过在SD中插入temporal层实现视频生成，本文的motion module直接借鉴
- **Portrait4D-v2** [Deng et al.]: 基于Triplane的多视角方法，但受"多面"问题困扰；FLAME避免了此问题
- **启发**：FLAME这类参数化模型可以在更多生成任务中充当"结构化中间表示"，将高维生成问题分解为低维参数空间的可控生成

## 评分

⭐⭐⭐⭐ — 提出了首个文本驱动多视角肖像动画框架，FLAME作为统一桥梁的设计思路优雅且实用，解耦运动/表情的消融实验充分。但FLAME的表达能力上限和推理速度限制了实际应用。

<!-- RELATED:START -->

## 相关论文

- [FG-Portrait: 3D Flow Guided Editable Portrait Animation](../../CVPR2026/image_generation/fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)
- [RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)
- [LivePhoto: Real Image Animation with Text-guided Motion Control](../../ECCV2024/image_generation/livephoto_real_image_animation_with_text-guided_motion_control.md)

<!-- RELATED:END -->
