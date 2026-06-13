---
title: >-
  [论文解读] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance
description: >-
  [ICCV 2025][人体理解][人体动画] DreamActor-M1提出基于DiT架构的人体图像动画框架，通过隐式面部表征+3D头部球体+3D身体骨架的混合控制信号实现精细面部和身体控制，结合互补外观引导和渐进式训练策略支持肖像到全身的多尺度生成。 人体图像动画是视频生成中的热门研究方向，广泛应用于影视制作、广告和游戏…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "人体动画"
  - "扩散变换器"
  - "混合控制信号"
  - "多尺度适应"
  - "长时一致性"
---

# DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance

**会议**: ICCV 2025  
**arXiv**: [2504.01724](https://arxiv.org/abs/2504.01724)  
**代码**: 无（项目页面: [https://grisoon.github.io/DreamActor-M1/](https://grisoon.github.io/DreamActor-M1/)）  
**领域**: 人体理解  
**关键词**: 人体动画, 扩散变换器, 混合控制信号, 多尺度适应, 长时一致性

## 一句话总结
DreamActor-M1提出基于DiT架构的人体图像动画框架，通过隐式面部表征+3D头部球体+3D身体骨架的混合控制信号实现精细面部和身体控制，结合互补外观引导和渐进式训练策略支持肖像到全身的多尺度生成。

## 研究背景与动机
人体图像动画是视频生成中的热门研究方向，广泛应用于影视制作、广告和游戏领域。基于单张图像驱动人体运动的方法已取得显著进展，但仍存在几个关键挑战：

**精细整体控制不足**：现有方法要么擅长面部（GAN/NeRF方法仅限肖像区域），要么擅长身体（扩散方法忽略精细面部表情），无法同时精确控制面部和身体

**多尺度适应能力差**：不同输入（肖像/上半身/全身）的信息密度和关注重点不同，现有方法难以在单一框架下处理

**长时一致性问题**：由于无法一次性生成长视频，多段拼接时参考图像中未见区域（如背面衣服纹理）的信息逐渐丢失，导致不一致

**单一控制信号困境**：用骨架控制面部不够精细，用3DMM参数又无法控制身体

**切入角度**：设计混合控制信号解耦面部表情和身体动作，引入互补外观引导弥补未见区域信息缺口，通过渐进式训练策略适应多尺度输入。

## 方法详解

### 整体框架
基于预训练的Image-to-Video DiT模型（Seaweed），采用MMDiT作为backbone。将参考图像latent和驱动视频latent拼接后送入DiT，通过3D自注意力和空间交叉注意力实现参考-视频信息交互。关键创新在于混合控制信号的设计和注入方式。

### 关键设计

1. **混合运动引导信号**:

    - **隐式面部表征（Implicit Facial Representations）**:
        - 功能：精细控制面部表情，同时解耦表情、身份和头部姿态
        - 核心思路：检测并裁剪驱动视频中的人脸，标准化为 $F \in \mathbb{R}^{t \times 3 \times 224 \times 224}$，用预训练的面部运动编码器 $\textbf{E}_f$ 编码为面部运动token $M \in \mathbb{R}^{t \times c}$，通过DiT block中的cross-attention注入
        - 设计动机：相比facial landmark，隐式表征能捕获更微妙的表情细节（如眨眼、嘴唇颤动）。面部运动编码器用大规模数据集预训练的身份无关表情特征初始化，确保鲁棒性。额外训练了音频驱动编码器支持语音驱动口型

    - **3D头部球体（3D Head Spheres）**:
        - 功能：独立控制头部姿态（旋转、位置、大小）
        - 核心思路：用3D人脸追踪提取相机参数和旋转角度，渲染为投影到2D平面的彩色球体。球体位置与驱动头部对齐，大小匹配参考头部，颜色由驱动头部朝向决定
        - 设计动机：将复杂的3D头部运动抽象为简单的2D球体表示，降低模型学习复杂度，特别有利于保持动漫/卡通角色的特殊头部结构

    - **3D身体骨架（3D Body Skeletons）**:
        - 功能：控制身体和手部动作
        - 核心思路：使用4DHumans和HaMeR估计SMPL-X参数，提取关节点投影到2D并连线。推理时通过骨骼长度归一化匹配参考与驱动人物的体型差异
        - 设计动机：选择骨架而非全身渲染（如Champ的方式），避免对体型的强引导，鼓励模型从参考图像学习角色形状和外观

   三者组合：身体骨架和头部球体通过pose encoder编码后与噪声视频特征拼接，面部运动token通过cross-attention注入。

2. **互补外观引导（Complementary Appearance Guidance）**:

    - 功能：提供未见区域的视觉信息，维持长时一致性
    - 核心思路：
        - 训练时：按yaw角排序选择三个关键帧（最大、最小、中位数旋转角），全身视频还额外裁剪半身肖像作为辅助参考
        - 推理时可选两阶段模式：先从单张参考图生成多视角序列，选取关键帧作为互补参考，再用多参考推理生成最终视频
    - 设计动机：单张参考图无法提供旋转/侧视等场景下的外观信息，多参考策略通过覆盖多个视角弥补信息缺口

3. **渐进式训练策略（Progressive Training）**:

    - 功能：分三阶段逐步引入控制信号
    - 阶段一（20k步）：仅用3D骨架和头部球体，建立基础人体动画能力
    - 阶段二（20k步）：引入隐式面部表征，冻结其他参数，仅训练face motion encoder和face attention层
    - 阶段三（30k步）：解冻所有参数联合优化
    - 设计动机：避免一次性引入过多复杂信号导致模型学习困难，确保稳定有效的训练过程

### 损失函数 / 训练策略
使用Flow Matching作为训练目标。训练视频长度随机选取25-121帧，空间分辨率resize到 $960 \times 640$ 面积保持原始纵横比。8×H20 GPU，AdamW优化器，学习率 $5 \times 10^{-6}$。推理时每段73帧，用当前段最后一帧latent作为下一段初始latent。CFG参数设为2.5。

## 实验关键数据

### 主实验：身体动画对比

| 方法 | FID↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ |
|------|------|-------|-------|--------|------|
| Animate Anyone | 36.72 | 0.791 | 21.74 | 0.266 | 158.3 |
| Champ | 40.21 | 0.732 | 20.18 | 0.281 | 171.2 |
| MimicMotion | 35.90 | 0.799 | 22.25 | 0.253 | 149.9 |
| DisPose | 33.01 | 0.804 | 21.99 | 0.248 | 144.7 |
| **DreamActor-M1** | **27.27** | **0.821** | **23.93** | **0.206** | **122.0** |

### 主实验：肖像动画对比

| 方法 | FID↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ |
|------|------|-------|-------|--------|------|
| LivePortrait | 31.72 | 0.809 | 24.25 | 0.270 | 147.1 |
| X-Portrait | 30.09 | 0.774 | 22.98 | 0.281 | 150.9 |
| SkyReels-A1 | 30.66 | 0.811 | 24.11 | 0.262 | 133.8 |
| Act-One | 29.84 | 0.817 | 25.07 | 0.259 | 135.2 |
| **DreamActor-M1** | **25.70** | **0.823** | **28.44** | **0.238** | **110.3** |

### 消融实验

| 配置 | FID↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | 说明 |
|------|------|-------|-------|--------|------|------|
| 单参考推理 | 28.22 | 0.798 | 25.86 | 0.223 | 120.5 | 基线 |
| 多参考（伪）推理 | 26.53 | 0.812 | 26.22 | 0.219 | 116.6 | 多参考提升长时一致性 |

控制信号消融（定性）：
- 用3D mesh替代骨架+球体 → 性能显著下降，骨骼长度调整和简化表示更有效
- 用3D facial landmarks替代隐式面部表征 → 精细表情细节丢失

### 关键发现
- DreamActor-M1在身体动画和肖像动画任务上均全面超越SOTA
- 身体动画FID从33.01降至27.27（提升17.4%），FVD从144.7降至122.0（提升15.7%）
- 肖像动画PSNR从25.07提升至28.44（+3.37 dB），说明面部重建精度极高
- 多参考推理在FVD上从120.5降至116.6，但单参考已足够strong
- 隐式面部表征比3D landmarks在细微表情捕捉上有质的飞跃
- 框架支持音频驱动的口型同步（speech-to-lip）

## 亮点与洞察
- 混合控制信号设计非常巧妙：面部用隐式表征保精度、头部用球体简化学习、身体用骨架留灵活度
- 不使用额外ReferenceNet作为外观注入（如Animate Anyone），而是将参考token直接与视频token拼接通过自注意力交互，架构更简洁
- 渐进式训练策略有效避免了复杂多信号的学习冲突
- 互补外观引导的两阶段推理模式巧妙解决了长时视频生成中未见区域的一致性问题
- 支持表情与身体的独立解耦控制，面部运动可由视频或语音驱动

## 局限与展望
- 无法控制动态相机运动——当前框架仅支持固定或简单相机设置
- 无法生成与环境物体的物理交互（如拿起杯子）
- 骨骼长度调整依赖预训练图像编辑模型，边缘情况不稳定需多次迭代手动选择
- 数据集私有（500小时自采集视频），可复现性受限
- 训练成本高：8×H20 GPU，总约70k步

## 相关工作与启发
- 在Animate Anyone、MimicMotion、Champ等基础上的全面升级：从coarse到fine-grained控制
- DiT架构替代UNet的趋势延续：SkyReels-A1、HumanDiT等
- 隐式面部表征（EMOPortraits等）与显式骨架控制的巧妙组合
- 启发：多模态控制信号的解耦设计是实现holistic animation的关键

## 评分
- 新颖性: ⭐⭐⭐⭐ 混合控制信号的解耦设计和互补外观引导是核心创新
- 实验充分度: ⭐⭐⭐⭐ 身体+肖像双任务对比全面，但消融深度可加强
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但部分细节（如音频驱动）未充分展开
- 价值: ⭐⭐⭐⭐⭐ 在实用性和效果上显著超越SOTA，是人体动画领域重要进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](../../CVPR2025/human_understanding/x-dyna_expressive_dynamic_human_image_animation.md)
- [\[ICCV 2025\] Controllable and Expressive One-Shot Video Head Swapping](controllable_and_expressive_one-shot_video_head_swapping.md)
- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](../../CVPR2025/human_understanding/keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[CVPR 2026\] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](../../CVPR2026/human_understanding/party_part-guidance_for_expressive_text-to-motion_synthesis.md)

</div>

<!-- RELATED:END -->
