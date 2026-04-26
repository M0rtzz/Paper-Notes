---
title: >-
  [论文解读] Pathways on the Image Manifold: Image Editing via Video Generation
description: >-
  [CVPR 2025][图像编辑] Frame2Frame (F2F) 将图像编辑重新定义为视频生成任务——利用 image-to-video 模型在图像流形上从源图像到目标编辑生成一条平滑的时间路径，通过 VLM 生成时间编辑描述并自动选帧，在编辑精度和图像保真度之间取得了 SOTA 平衡。
tags:
  - CVPR 2025
  - 图像编辑
  - 视频生成
  - 图像流形
  - CogVideoX
  - 时间一致性
---

# Pathways on the Image Manifold: Image Editing via Video Generation

**会议**: CVPR 2025  
**arXiv**: [2411.16819](https://arxiv.org/abs/2411.16819)  
**代码**: 无（项目主页可见）  
**领域**: 图像生成 / 图像编辑  
**关键词**: 图像编辑, 视频生成, 图像流形, CogVideoX, 时间一致性

## 一句话总结

Frame2Frame (F2F) 将图像编辑重新定义为视频生成任务——利用 image-to-video 模型在图像流形上从源图像到目标编辑生成一条平滑的时间路径，通过 VLM 生成时间编辑描述并自动选帧，在编辑精度和图像保真度之间取得了 SOTA 平衡。

## 研究背景与动机

**领域现状**：文本引导的图像编辑主要依赖扩散模型，通过对源图像进行隐空间反转或模型微调来条件化生成过程。代表性方法包括 SDEdit（加噪去噪）、Imagic（微调+插值）、InstructPix2Pix（监督训练）、LEDITS++（无训练反转编辑）等。同时，视频生成模型（如 Stable Video Diffusion、CogVideoX、Sora）已发展为出色的"世界模拟器"，具备强大的时间一致性和物理理解能力。

**现有痛点**：现有图像编辑方法面临两个核心挑战的权衡：(1) 编辑精度——复杂的编辑指令难以被准确执行；(2) 保真度——编辑过程经常意外修改原图中不该改变的关键元素（如文字、背景物体）。这两者之间存在根本性的矛盾：模型需要在单次生成中同时实现精准编辑和内容保持。

**核心矛盾**：传统编辑方法从初始噪声出发，投影到图像流形上的一个目标点，要求该点同时满足源图像保真和编辑请求。这种"跳跃式"转换容易偏离图像流形，落入同时满足编辑但破坏保真的区域。

**本文目标**：利用视频生成模型的时间一致性，将图像编辑从"单点跳跃"转变为"流形上的连续路径"，使编辑过程经历物理上合理的中间状态，从而在保真度和编辑精度之间取得更好的平衡。

**切入角度**：从几何视角观察——视频生成模型在训练中学到了图像流形的连续结构，源图像作为视频第一帧自然地锚定在流形上，后续帧沿流形平滑演进，每一帧都是真实可信的图像，直到达到编辑目标。

**核心 idea**：将图像编辑重新定义为"从源图像出发生成一段视频，视频最终帧即为编辑结果"，利用视频模型的时间一致性天然保证源图像的关键属性在编辑过程中得以保持。

## 方法详解

### 整体框架

Frame2Frame (F2F) 包含三个步骤：(1) **时间编辑描述生成**——用 VLM（GPT-4o）将源图像和编辑指令转换为描述时间演变过程的视频描述（temporal editing caption）；(2) **视频生成**——用 CogVideoX (I2V-5B) 以源图像为首帧、时间描述为条件生成视频序列；(3) **帧选择**——利用 VLM 从生成的视频帧中自动选出最佳实现编辑目标的帧。

### 关键设计

1. **时间编辑描述 (Temporal Editing Captions)**:

    - 功能：将静态的编辑指令转化为描述时间演变过程的视频场景描述
    - 核心思路：利用 GPT-4o 作为 VLM，输入源图像 $I_s$ 和目标编辑描述 $c$，通过 in-context learning（提供 9 个示例 prompt-caption 对）指导模型生成简洁的视频场景描述 $\tilde{c}$，强调元素如何随时间变化或移动，保持静态相机视角。例如将"一个人做心形手势"转化为"一个人非常缓慢地举起双手形成心形"
    - 设计动机：直接将编辑描述作为 I2V 模型的文本条件会导致生成结果不可控——视频模型需要的是"过程描述"而非"状态描述"。时间编辑描述弥合了图像编辑和视频生成之间的语义鸿沟

2. **基于视频生成的编辑 (Video-based Editing)**:

    - 功能：在图像流形上生成从源图像到目标编辑的连续路径
    - 核心思路：使用 CogVideoX (I2V-5B)，一个基于 Transformer 的视频潜在扩散模型，其 3D VAE 在空间和时间维度上压缩视频。源图像 $I_s$ 编码后与噪声在潜在空间拼接，去噪过程受时间描述 $\tilde{c}$ 引导。AdaptiveLayerNorm 实现视觉-文本模态的深度融合。生成 $T=49$ 帧，约 6 秒时长（8fps）。图像预处理时将 1:1 图像 resize 到 480×480 后左右各填充 120 像素黑边到 720×480
    - 设计动机：视频模型在大规模互联网数据上训练获得了对物理世界动态的理解，这种"世界认知"使得生成的中间帧是物理上合理的——如一个人从站立到举手的过程中不会突然换衣服或改变背景

3. **自动帧选择 (Frame Selection)**:

    - 功能：从视频序列中自动识别最佳编辑结果帧
    - 核心思路：每 4 帧采样一次，为每帧标注唯一标识符后组成图像拼贴，连同源图像 $I_s$ 和编辑指令 $c$ 提供给 GPT-4o，指示其选择最早完成编辑的帧 $f_{t^*}$。选择"最早"是因为后续帧倾向于偏离源图像更远
    - 设计动机：不同编辑所需的帧数不同——简单修改可能几帧就完成，复杂变换需要更多帧。固定取最后一帧会导致过度偏离源图像。自动帧选择也可作为用户交互的灵活接口

### 损失函数 / 训练策略

- **无需训练**：F2F 不涉及任何模型训练或微调，完全利用预训练的 CogVideoX 和 GPT-4o
- 视频生成使用默认超参数：guidance scale = 6，49 帧，50 去噪步
- 评估时每个方法对每张源图像用 15 个随机种子生成结果，手动选取最佳

## 实验关键数据

### 主实验 (TEdBench)

| 方法 | LPIPS↓ (保真) | CLIP-I↑ (保真) | CLIP↑ (编辑精度) |
|------|:---:|:---:|:---:|
| SDEdit | 0.30 | 0.85 | 0.60 |
| Pix2Pix-ZERO | 0.29 | 0.84 | 0.62 |
| Imagic | 0.52 | 0.86 | 0.63 |
| LEDITS++ | 0.23 | 0.87 | 0.63 |
| FlowEdit | 0.22 | 0.89 | 0.61 |
| **F2F** | **0.22** | **0.89** | **0.63** |

### PosEdit (人体姿态编辑)

| 方法 | Source LPIPS↓ | Source CLIP-I↑ | Target LPIPS↓ | Target CLIP-I↑ | CLIP↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| SDEdit | 0.39 | 0.61 | 0.39 | 0.64 | 0.57 |
| LEDITS++ | 0.26 | 0.65 | 0.28 | 0.69 | 0.64 |
| **F2F** | **0.14** | **0.82** | **0.15** | **0.84** | **0.64** |
| GT (参考) | 0.08 | 0.91 | 0 | 1.0 | 0.61 |

### 人类评估 (TEdBench, F2F vs LEDITS++)

| 指标 | F2F | LEDITS++ |
|------|:---:|:---:|
| 编辑精度 (Overall) | **54.1%** | 45.9% |
| 编辑质量 (Overall) | **65.6%** | 34.4% |

### 关键发现

- 在 TEdBench 上 F2F 在保真度和编辑精度两个维度上同时达到最佳或持平
- 在 PosEdit 上 F2F 的 Source LPIPS (0.14) 远低于 LEDITS++ (0.26)，表明视频路径方法在保持身份特征方面优势巨大
- 人类评估表明 F2F 在编辑质量（保真度）上的领先幅度远大于自动指标所反映的（65.6% vs 34.4%）
- F2F 还可直接应用于去噪、去模糊、outpainting 和 relighting 等传统视觉任务，因为这些操作天然对应常见的视频场景（如对焦、相机移动、延时摄影）
- 图像流形可视化实验直观地展示了视频路径方法的优势——沿流形平滑移动可以保持"AI"T恤文字等细节，而跳跃式编辑会丢失

## 亮点与洞察

- 将图像编辑重新定义为视频生成是一个真正的范式转换——利用视频模型的世界知识来保证编辑的物理合理性
- 图像流形可视化分析（PCA 投影 + 集群可视化）很好地解释了"为什么沿路径编辑优于跳跃编辑"
- 方法极其简洁——无需训练、无需反转、三步 pipeline，赋予了该方法极强的实用性和可扩展性
- PosEdit 数据集（58 个编辑任务，含 ground truth）是一个有价值的补充基准

## 局限与展望

- 视频生成中可能出现非预期的相机运动，导致编辑结果存在视角偏移
- 计算开销较大——生成 49 帧视频比直接图像编辑慢得多（但视频生成速度正在快速提升，如 LTX-Video 可在秒级完成）
- 强依赖 CogVideoX 和 GPT-4o 两个闭源/大型模型，端到端成本较高
- 编辑风格受限于视频模型的训练数据分布——以真实世界变换为主的训练数据可能难以处理"魔法"式编辑
- 未来方向：专门针对图像编辑微调视频生成器、减少帧数以加速、自定义帧选择策略

## 相关工作与启发

- 与传统编辑方法（SDEdit、Imagic、InstructPix2Pix）的本质区别在于将"单步投影"变为"多步路径"
- 与 AnyDoor、MagicFixup 的区别：后者从视频中采样帧对构建训练数据，F2F 直接用视频生成做编辑
- 类似的"借助视频模型理解 3D/物理世界"思路也出现在 Make-A-Video3D、PhysDreamer 等工作中

## 评分

- **新颖性**: 9/10 — 将图像编辑重新定义为视频生成是真正的范式创新，流形路径视角很有洞察
- **实验充分度**: 7/10 — TEdBench + PosEdit + 人类评估较全面，但缺少与更多最新方法的对比，传统视觉任务仅定性展示
- **写作质量**: 9/10 — 行文流畅，流形可视化分析极具说服力
- **价值**: 8/10 — 范式启发性强，方法简洁但实用，随视频生成技术进步价值会持续提升

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] MotiF: Making Text Count in Image Animation with Motion Focal Loss](motif_making_text_count_in_image_animation_with_motion_focal_loss.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](sketchvideo_sketch-based_video_generation_and_editing.md)
- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)

<!-- RELATED:END -->
